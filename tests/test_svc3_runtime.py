import sys
import threading
import unittest
from pathlib import Path

import numpy as np


SVC3_ROOT = Path(__file__).resolve().parents[1] / "ubuntuserver" / "hay_say" / "so_vits_svc_3"
if str(SVC3_ROOT) not in sys.path:
    sys.path.insert(0, str(SVC3_ROOT))

from inference.runtime import ModelCache, ModelSpec, SvcRuntime, normalize_device


class FakeHubert:
    def __init__(self, device):
        self.device = device
        self.closed = False

    def to(self, device):
        if device == "cpu":
            self.closed = True
        return self


class FakeModel:
    def __init__(self, spec, device, hubert):
        self.spec = spec
        self.device = device
        self.hubert = hubert
        self.target_sample = 32000
        self.closed = False
        self.prepared = []
        self.inferred = []

    def prepare_features(self, audio, sample_rate):
        feature = (len(audio), sample_rate)
        self.prepared.append(feature)
        return feature

    def infer_from_features(self, speaker, pitch, feature):
        self.inferred.append((speaker, pitch, feature))
        return np.array([feature[0], pitch], dtype=np.float32), 2

    def close(self):
        self.closed = True


class RuntimeFixture:
    def __init__(self, max_models=2):
        self.hubert_loads = []
        self.model_loads = []
        self.models = []

        def load_hubert(path, device):
            self.hubert_loads.append((path, device))
            return FakeHubert(device)

        def load_model(spec, device, hubert):
            self.model_loads.append((spec, device, hubert))
            model = FakeModel(spec, device, hubert)
            self.models.append(model)
            return model

        self.cache = ModelCache(
            "/weights/hubert.pt",
            max_models_per_device=max_models,
            model_loader=load_model,
            hubert_loader=load_hubert,
        )

    @staticmethod
    def spec(character):
        return ModelSpec(
            character,
            "/weights/{}/G_1.pth".format(character),
            "/weights/{}/config.json".format(character),
        )


class ModelCacheTests(unittest.TestCase):
    def test_models_on_one_device_share_exactly_one_hubert(self):
        fixture = RuntimeFixture(max_models=2)
        with fixture.cache.acquire(fixture.spec("alpha"), "cuda:0"):
            pass
        with fixture.cache.acquire(fixture.spec("beta"), "cuda:0"):
            pass

        self.assertEqual(1, len(fixture.hubert_loads))
        self.assertIs(fixture.models[0].hubert, fixture.models[1].hubert)
        self.assertEqual(2, len(fixture.cache.snapshot()["loaded_models"]))

    def test_hubert_pool_is_device_aware(self):
        fixture = RuntimeFixture(max_models=2)
        with fixture.cache.acquire(fixture.spec("alpha"), "cpu"):
            pass
        with fixture.cache.acquire(fixture.spec("alpha"), "cuda:0"):
            pass

        self.assertEqual(["cpu", "cuda:0"], [item[1] for item in fixture.hubert_loads])

    def test_lru_evicts_idle_model_but_keeps_shared_hubert_warm(self):
        fixture = RuntimeFixture(max_models=1)
        with fixture.cache.acquire(fixture.spec("alpha"), "cpu"):
            pass
        first_model = fixture.models[0]
        with fixture.cache.acquire(fixture.spec("beta"), "cpu"):
            pass

        self.assertTrue(first_model.closed)
        self.assertEqual(1, len(fixture.hubert_loads))
        loaded = fixture.cache.snapshot()["loaded_models"]
        self.assertEqual(["beta"], [item["character"] for item in loaded])

    def test_unload_does_not_close_a_leased_model(self):
        fixture = RuntimeFixture(max_models=1)
        with fixture.cache.acquire(fixture.spec("alpha"), "cpu") as model:
            result = fixture.cache.unload(character="alpha")
            self.assertEqual(1, len(result["busy_models"]))
            self.assertFalse(model.closed)

        result = fixture.cache.unload(character="alpha")
        self.assertEqual(1, len(result["unloaded_models"]))
        self.assertEqual(["cpu"], result["released_hubert_devices"])

    def test_concurrent_acquires_only_load_a_model_once(self):
        fixture = RuntimeFixture(max_models=2)
        spec = fixture.spec("alpha")
        barrier = threading.Barrier(4)
        failures = []

        def acquire_model():
            try:
                with fixture.cache.acquire(spec, "cpu"):
                    barrier.wait(timeout=2)
            except Exception as exc:  # pragma: no cover - assertion reports thread errors
                failures.append(exc)

        threads = [threading.Thread(target=acquire_model) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)

        self.assertEqual([], failures)
        self.assertEqual(1, len(fixture.model_loads))
        self.assertEqual(1, len(fixture.hubert_loads))


class SvcRuntimeTests(unittest.TestCase):
    def test_pitch_batch_reuses_prepared_source_features(self):
        fixture = RuntimeFixture(max_models=1)

        def segmenter(audio, sample_rate, slice_db):
            del audio, sample_rate, slice_db
            yield False, np.ones(5, dtype=np.float32), 32000
            yield True, np.zeros(3, dtype=np.float32), 32000
            yield False, np.ones(7, dtype=np.float32), 32000

        runtime = SvcRuntime(fixture.cache, segmenter=segmenter)
        outputs, sample_rate = runtime.generate(
            fixture.spec("alpha"),
            "speaker-a",
            [-2, 0, 3],
            np.ones(20, dtype=np.float32),
            44100,
            "cpu",
        )

        model = fixture.models[0]
        self.assertEqual(2, len(model.prepared))
        self.assertEqual(6, len(model.inferred))
        self.assertEqual([-2, 0, 3], [pitch for pitch, _ in outputs])
        self.assertEqual(32000, sample_rate)
        self.assertTrue(all(len(audio) == 7 for _, audio in outputs))

    def test_state_distinguishes_cold_and_warm(self):
        fixture = RuntimeFixture(max_models=1)
        runtime = SvcRuntime(fixture.cache, segmenter=lambda *_: [])
        runtime.resource_usage = lambda *_: {"process_rss_bytes": 1, "gpus": []}
        self.assertEqual("ready-cold", runtime.state()["status"])

        runtime.warm(fixture.spec("alpha"), "cpu")
        state = runtime.state()
        self.assertEqual("warm-idle", state["status"])
        self.assertTrue(state["warm"])
        self.assertFalse(state["busy"])
        self.assertEqual(0, state["active_jobs"])
        self.assertEqual(["alpha"], state["loaded_models"])
        self.assertEqual("cpu", state["device"])

    def test_legacy_device_ids_are_normalized(self):
        self.assertEqual("cpu", normalize_device(""))
        self.assertEqual("cpu", normalize_device(-1))
        self.assertEqual("cuda:2", normalize_device("2"))
        self.assertEqual("cuda:3", normalize_device("cuda:3"))
        with self.assertRaises(ValueError):
            normalize_device("gpu-zero")


if __name__ == "__main__":
    unittest.main()

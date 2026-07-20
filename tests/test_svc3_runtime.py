import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from unittest.mock import patch


SVC3_ROOT = Path(__file__).resolve().parents[1] / "ubuntuserver" / "hay_say" / "so_vits_svc_3"
if str(SVC3_ROOT) not in sys.path:
    sys.path.insert(0, str(SVC3_ROOT))

from inference.runtime import (
    GenerationCancelled,
    ModelCache,
    ModelSpec,
    SvcRuntime,
    configure_svc3_cpu_threads,
    file_revision,
    normalize_device,
    svc3_cpu_thread_settings,
)


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
    def test_svc3_divides_the_host_thread_budget_across_pitch_replicas(self):
        environment = {
            "HAY_SAY_SVC3_CPU_THREADS": "180",
            "HAY_SAY_SVC3_CPU_PITCH_WORKERS": "24",
        }
        with patch.dict("os.environ", environment, clear=True), patch("os.cpu_count", return_value=120):
            with patch("inference.runtime.configure_torch_threads") as configure:
                configure.return_value = object()
                self.assertIs(configure.return_value, configure_svc3_cpu_threads())
                configure.assert_called_once_with(force=True, intraop_threads=10)

    def test_svc3_explicit_thread_budget_caps_each_replica(self):
        environment = {
            "HAY_SAY_SVC3_CPU_THREADS": "96",
            "HAY_SAY_SVC3_CPU_PITCH_WORKERS": "12",
            "HAY_SAY_SVC3_CPU_THREAD_BUDGET": "240",
        }
        with patch.dict("os.environ", environment, clear=True):
            self.assertEqual(
                {
                    "requested_threads_per_worker": 96,
                    "threads_per_worker": 20,
                    "thread_budget": 240,
                    "pitch_workers": 12,
                },
                svc3_cpu_thread_settings(),
            )

    def test_svc3_thread_budget_is_a_hard_pool_ceiling(self):
        environment = {
            "HAY_SAY_SVC3_CPU_THREADS": "999",
            "HAY_SAY_SVC3_CPU_PITCH_WORKERS": "24",
            "HAY_SAY_SVC3_CPU_THREAD_BUDGET": "239",
        }
        with patch.dict("os.environ", environment, clear=True):
            settings = svc3_cpu_thread_settings()
        self.assertLessEqual(
            settings["threads_per_worker"] * settings["pitch_workers"],
            settings["thread_budget"],
        )

    def test_svc3_thread_budget_must_cover_every_pitch_worker(self):
        environment = {
            "HAY_SAY_SVC3_CPU_THREADS": "1",
            "HAY_SAY_SVC3_CPU_PITCH_WORKERS": "24",
            "HAY_SAY_SVC3_CPU_THREAD_BUDGET": "23",
        }
        with patch.dict("os.environ", environment, clear=True):
            with self.assertRaisesRegex(ValueError, "must be at least"):
                svc3_cpu_thread_settings()

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

    def test_lru_respects_minimum_idle_residency_before_eviction(self):
        fixture = RuntimeFixture(max_models=1)
        with fixture.cache.acquire(fixture.spec("alpha"), "cpu"):
            pass
        first_model = fixture.models[0]
        with fixture.cache.acquire(fixture.spec("beta"), "cpu"):
            pass

        self.assertFalse(first_model.closed)
        self.assertEqual(1, len(fixture.hubert_loads))
        loaded = fixture.cache.snapshot()["loaded_models"]
        self.assertEqual(["alpha", "beta"], [item["character"] for item in loaded])
        self.assertEqual(1800, fixture.cache.idle_ttl_seconds)
        self.assertGreater(loaded[0]["idle_ttl_remaining_seconds"], 1790)

        with fixture.cache._lock:
            next(iter(fixture.cache._entries.values())).last_used -= 1801
        with fixture.cache.acquire(fixture.spec("gamma"), "cpu"):
            pass

        self.assertTrue(first_model.closed)
        loaded = fixture.cache.snapshot()["loaded_models"]
        self.assertEqual(["beta", "gamma"], [item["character"] for item in loaded])

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

    def test_replacing_checkpoint_in_place_loads_the_new_revision(self):
        import tempfile

        fixture = RuntimeFixture(max_models=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "G_1.pth"
            config = root / "config.json"
            model.write_bytes(b"first")
            config.write_text("{}", encoding="utf-8")
            first = ModelSpec("alpha", str(model), str(config), file_revision(model), file_revision(config))
            with fixture.cache.acquire(first, "cpu"):
                pass

            model.write_bytes(b"second-revision")
            second = ModelSpec("alpha", str(model), str(config), file_revision(model), file_revision(config))
            with fixture.cache.acquire(second, "cpu"):
                pass

        self.assertEqual(2, len(fixture.model_loads))
        loaded = fixture.cache.snapshot()["loaded_models"]
        self.assertEqual(2, len(loaded))
        self.assertNotEqual(loaded[0]["model_revision"], loaded[1]["model_revision"])


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

    def test_cpu_pitch_batch_runs_isolated_replicas_concurrently_in_order(self):
        fixture = RuntimeFixture(max_models=1)
        barrier = threading.Barrier(3)

        class ReplicaModel(FakeModel):
            def __init__(self, spec, device, hubert, ordinal):
                super().__init__(spec, device, hubert)
                self.ordinal = ordinal

            def ensure_inference_replicas(self, count):
                return tuple(replicas[:count])

            def infer_from_features(self, speaker, pitch, feature):
                barrier.wait(timeout=2)
                time.sleep(0.01 * (2 - self.ordinal))
                return np.array([pitch, self.ordinal], dtype=np.float32), 2

        spec = fixture.spec("alpha")
        hubert = FakeHubert("cpu")
        replicas = [ReplicaModel(spec, "cpu", hubert, index) for index in range(3)]
        fixture.cache._model_loader = lambda *_: replicas[0]
        fixture.cache._hubert_loader = lambda *_: hubert
        runtime = SvcRuntime(
            fixture.cache,
            segmenter=lambda *_: [(False, np.ones(5, dtype=np.float32), 32000)],
            cpu_pitch_workers=3,
        )

        outputs, _ = runtime.generate(
            spec,
            "speaker-a",
            [-3, 0, 4],
            np.ones(5, dtype=np.float32),
            32000,
            "cpu",
        )

        self.assertEqual([-3, 0, 4], [pitch for pitch, _ in outputs])
        self.assertEqual([-3, 0, 4], [int(audio[0]) for _, audio in outputs])

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

    def test_cancel_drops_queued_work_and_keeps_active_model_warm(self):
        fixture = RuntimeFixture(max_models=1)
        started = threading.Event()
        release = threading.Event()

        class BlockingModel(FakeModel):
            def infer_from_features(self, speaker, pitch, feature):
                self.inferred.append((speaker, pitch, feature))
                started.set()
                if not release.wait(timeout=3):
                    raise RuntimeError("test inference was not released")
                return np.array([pitch], dtype=np.float32), 1

        fixture.cache._model_loader = lambda spec, device, hubert: BlockingModel(
            spec, device, hubert
        )
        runtime = SvcRuntime(
            fixture.cache,
            segmenter=lambda *_: [(False, np.ones(5, dtype=np.float32), 32000)],
            cpu_pitch_workers=1,
        )
        errors = {}

        def generate(request_id):
            try:
                with runtime.cancellation_scope(request_id) as cancellation:
                    runtime.generate(
                        fixture.spec("alpha"),
                        "speaker-a",
                        [0, 1, 2],
                        np.ones(5, dtype=np.float32),
                        32000,
                        "cpu",
                        cancellation=cancellation,
                    )
            except BaseException as exc:  # assertion below reports thread failures
                errors[request_id] = exc

        active = threading.Thread(target=generate, args=("request-active",))
        active.start()
        self.assertTrue(started.wait(timeout=2))

        queued = threading.Thread(target=generate, args=("request-queued",))
        queued.start()
        deadline = time.time() + 2
        while runtime.state()["queued_jobs"] != 1 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(1, runtime.state()["queued_jobs"])

        result = runtime.cancel(["request-queued"])
        queued.join(timeout=1)
        self.assertFalse(queued.is_alive())
        self.assertIsInstance(errors.get("request-queued"), GenerationCancelled)
        self.assertEqual(["request-queued"], result["active_request_ids"])
        self.assertTrue(active.is_alive())

        runtime.cancel(["request-active"])
        time.sleep(0.05)
        self.assertTrue(active.is_alive())
        self.assertGreater(
            fixture.cache.snapshot()["loaded_models"][0]["active_leases"], 0
        )
        release.set()
        active.join(timeout=2)

        self.assertFalse(active.is_alive())
        self.assertIsInstance(errors.get("request-active"), GenerationCancelled)
        model = next(iter(fixture.cache._entries.values())).model
        self.assertEqual(1, len(model.inferred))
        self.assertFalse(model.closed)
        self.assertEqual("warm-idle", runtime.state()["status"])

    def test_cancel_tombstone_rejects_a_request_that_has_not_started(self):
        fixture = RuntimeFixture(max_models=1)
        runtime = SvcRuntime(fixture.cache, segmenter=lambda *_: [])

        runtime.cancel(["request-future"])

        with self.assertRaises(GenerationCancelled):
            with runtime.cancellation_scope("request-future"):
                self.fail("a pre-cancelled request entered its runtime scope")
        self.assertEqual([], fixture.model_loads)

    def test_output_commit_is_linearized_before_later_cancellation(self):
        fixture = RuntimeFixture(max_models=1)
        runtime = SvcRuntime(fixture.cache, segmenter=lambda *_: [])
        commit_started = threading.Event()
        release_commit = threading.Event()
        committed = []
        errors = []

        def commit_request():
            try:
                with runtime.cancellation_scope("request-commit") as cancellation:
                    def commit():
                        commit_started.set()
                        if not release_commit.wait(timeout=2):
                            raise RuntimeError("test commit was not released")
                        committed.append(True)

                    runtime.commit_if_active(cancellation, commit)
            except BaseException as exc:  # assertion below reports thread failures
                errors.append(exc)

        commit_thread = threading.Thread(target=commit_request)
        commit_thread.start()
        self.assertTrue(commit_started.wait(timeout=1))
        cancel_thread = threading.Thread(
            target=lambda: runtime.cancel(["request-commit"])
        )
        cancel_thread.start()
        time.sleep(0.05)
        self.assertTrue(cancel_thread.is_alive())

        release_commit.set()
        commit_thread.join(timeout=2)
        cancel_thread.join(timeout=2)

        self.assertEqual([], errors)
        self.assertEqual([True], committed)
        self.assertFalse(commit_thread.is_alive())
        self.assertFalse(cancel_thread.is_alive())

    def test_legacy_device_ids_are_normalized(self):
        self.assertEqual("cpu", normalize_device(""))
        self.assertEqual("cpu", normalize_device(-1))
        self.assertEqual("cuda:2", normalize_device("2"))
        self.assertEqual("cuda:3", normalize_device("cuda:3"))
        with self.assertRaises(ValueError):
            normalize_device("gpu-zero")


if __name__ == "__main__":
    unittest.main()

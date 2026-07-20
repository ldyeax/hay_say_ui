import importlib.util
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
RUNTIME_PATH = (
    REPOSITORY / "ubuntuserver/hay_say/so_vits_svc_5_server/svc5_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("svc5_runtime", RUNTIME_PATH)
runtime_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime_module
SPEC.loader.exec_module(runtime_module)


def model_spec(character="Fluttershy", version=2):
    return runtime_module.ModelSpec(
        character=character,
        version=version,
        source_root="/tmp/svc5-v{}".format(version),
        checkpoint_path="/tmp/{}/sovits5.0.pth".format(character),
        config_path="/tmp/svc5-v{}/configs/base.yaml".format(version),
        speaker_path="/tmp/{}/singer/voice.spk.npy".format(character),
        checkpoint_revision=(1, 2, 3, 4),
        config_revision=(1, version, 3, 4),
        speaker_revision=(1, 5, 6, 7),
    )


class FakeReplica:
    def __init__(self, spec, device, name, activity):
        self.spec = spec
        self.device = device
        self.name = name
        self.activity = activity
        self.pid = activity["next_pid"]
        activity["next_pid"] += 1
        self.prepared_inputs = 0
        self.completed_inferences = 0
        self.closed = False
        self.cancel_event = threading.Event()
        activity["created"].append(self)

    @property
    def is_alive(self):
        return not self.closed

    def begin_request(self):
        self.cancel_event.clear()

    def cancel(self):
        self.cancel_event.set()

    def prepare(self, _job):
        with self.activity["lock"]:
            self.activity["prepares"] += 1
        self.prepared_inputs += 1
        time.sleep(0.02)
        return runtime_module.PreparedFeatures(
            np.ones((3, 2), dtype=np.float32),
            np.ones(3, dtype=np.float32) * 100,
            np.ones((3, 2), dtype=np.float32),
        )

    def infer(self, job):
        with self.activity["lock"]:
            self.activity["active"] += 1
            self.activity["maximum"] = max(
                self.activity["maximum"], self.activity["active"]
            )
        self.activity["infer_started"].set()
        try:
            deadline = time.monotonic() + 2
            while not self.activity["infer_release"].wait(timeout=0.01):
                if self.cancel_event.is_set():
                    raise runtime_module.GenerationCancelled("fake replica cancelled")
                if time.monotonic() >= deadline:
                    raise RuntimeError("fake inference timed out")
            if self.cancel_event.is_set():
                raise runtime_module.GenerationCancelled("fake replica cancelled")
            time.sleep(0.02)
            self.completed_inferences += 1
            return np.full(4, job.pitch_shift, dtype=np.float32)
        finally:
            with self.activity["lock"]:
                self.activity["active"] -= 1

    def close(self):
        self.closed = True


def make_runtime(cpu_workers=4, idle_ttl=1800, replica_type=FakeReplica):
    activity = {
        "lock": threading.Lock(),
        "prepares": 0,
        "active": 0,
        "maximum": 0,
        "created": [],
        "next_pid": 1000,
        "infer_started": threading.Event(),
        "infer_release": threading.Event(),
    }
    activity["infer_release"].set()

    def factory(spec, device, name):
        return replica_type(spec, device, name, activity)

    cache = runtime_module.ModelCache(
        cpu_workers,
        1,
        min(cpu_workers, 4),
        idle_ttl,
        replica_factory=factory,
    )
    return runtime_module.SVC5Runtime(cache), activity


def generate(current, spec, pitch, request_id, input_key="same-input"):
    return current.generate(
        spec,
        input_key,
        np.ones(160, dtype=np.float32),
        16000,
        "cpu",
        pitch,
        False,
        request_id,
    )


def test_parallel_pitch_requests_share_features_and_grow_replica_pool():
    current, activity = make_runtime(4)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(generate, current, model_spec(), pitch, "request-{}".format(pitch))
            for pitch in range(8)
        ]
        outputs = [future.result()[0] for future in futures]

    assert activity["prepares"] == 1
    assert len(activity["created"]) == 4
    assert activity["maximum"] >= 3
    assert [int(output[0]) for output in outputs] == list(range(8))
    state = current.state()
    assert state["status"] == "warm-idle"
    assert state["workers"] == 4
    assert state["feature_cache"]["entries"] == 1


def test_cancelled_inference_releases_and_reuses_same_warm_replica_pid():
    current, activity = make_runtime(1)
    activity["infer_release"].clear()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(generate, current, model_spec(), 1, "cancel-me")
        assert activity["infer_started"].wait(timeout=2)
        result = current.cancel(["cancel-me"])
        assert result == {"cancelled": ["cancel-me"], "active": ["cancel-me"]}
        with pytest.raises(runtime_module.GenerationCancelled):
            future.result(timeout=0.5)

    first_pid = activity["created"][0].pid
    activity["infer_release"].set()
    output, _ = generate(current, model_spec(), 2, "next-request")
    assert output[0] == 2
    assert len(activity["created"]) == 1
    assert activity["created"][0].pid == first_pid
    assert activity["created"][0].completed_inferences == 1
    state = current.state()
    assert state["status"] == "warm-idle"
    assert state["busy_workers"] == 0


def test_cancel_signals_all_parallel_replicas_for_one_browser_request():
    current, activity = make_runtime(2)
    activity["infer_release"].clear()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(generate, current, model_spec(), pitch, "shared-request")
            for pitch in (1, 2)
        ]
        deadline = time.monotonic() + 2
        while current.state()["busy_workers"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert current.state()["busy_workers"] == 2
        result = current.cancel(["shared-request"])
        assert result == {"cancelled": ["shared-request"], "active": ["shared-request"]}
        for future in futures:
            with pytest.raises(runtime_module.GenerationCancelled):
                future.result(timeout=0.5)

    state = current.state()
    assert state["workers"] == 2
    assert state["busy_workers"] == 0
    assert state["active_request_ids"] == []
    assert all(not replica.closed for replica in activity["created"])


def test_multi_chunk_inference_observes_cancel_between_model_calls():
    class FakeTensor:
        def __getitem__(self, _key):
            return self

        def unsqueeze(self, _axis):
            return self

        def to(self, _device):
            return self

    class FakeTorch:
        @staticmethod
        def LongTensor(_value):
            return FakeTensor()

    class FakeModel:
        def __init__(self, cancel_event):
            self.cancel_event = cancel_event
            self.calls = 0

        def inference(self, *_args):
            self.calls += 1
            self.cancel_event.set()
            return np.zeros((1, 1, 1), dtype=np.float32)

    cancel_event = threading.Event()
    pipeline = runtime_module._LoadedPipeline.__new__(runtime_module._LoadedPipeline)
    pipeline.torch = FakeTorch()
    pipeline.device = "cpu"
    pipeline.hp = type("HP", (), {"data": type("Data", (), {"hop_length": 320})()})()
    pipeline.model = FakeModel(cancel_event)

    with pytest.raises(runtime_module.GenerationCancelled, match="inference chunk"):
        pipeline._infer_v2(
            FakeTensor(),
            FakeTensor(),
            FakeTensor(),
            FakeTensor(),
            FakeTensor(),
            6000,
            cancel_event,
        )

    assert pipeline.model.calls == 1


def test_dead_replica_is_retired_and_replaced_on_next_request():
    class DiesOnFirstInference(FakeReplica):
        def infer(self, job):
            if not self.activity.get("crashed"):
                self.activity["crashed"] = True
                self.closed = True
                raise runtime_module.ReplicaUnavailableError("replica process exited")
            return super().infer(job)

    current, activity = make_runtime(1, replica_type=DiesOnFirstInference)
    with pytest.raises(runtime_module.ReplicaUnavailableError, match="exited"):
        generate(current, model_spec(), 1, "first-request")

    assert len(activity["created"]) == 1
    assert activity["created"][0].closed
    first_pid = activity["created"][0].pid

    output, _ = generate(current, model_spec(), 2, "second-request")

    assert output[0] == 2
    assert len(activity["created"]) == 2
    assert activity["created"][1].pid != first_pid
    assert current.state()["workers"] == 1


def test_shared_feature_waiter_retries_when_only_the_owner_was_cancelled():
    cache = runtime_module.FeatureCache()
    owner_started = threading.Event()
    release_owner = threading.Event()
    expected = runtime_module.PreparedFeatures(
        np.ones((2, 2), dtype=np.float32),
        np.ones(2, dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
    )

    def cancelled_owner_loader():
        owner_started.set()
        release_owner.wait(timeout=2)
        raise runtime_module.GenerationCancelled("owner cancelled")

    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(cache.get, "same-input", cancelled_owner_loader, lambda: False)
        assert owner_started.wait(timeout=2)
        waiter = executor.submit(cache.get, "same-input", lambda: expected, lambda: False)
        release_owner.set()

        with pytest.raises(runtime_module.GenerationCancelled):
            owner.result(timeout=2)
        assert waiter.result(timeout=2) is expected

    assert cache.state()["entries"] == 1


def test_recent_model_is_not_age_evicted_but_explicit_unload_is_immediate():
    current, activity = make_runtime(1, idle_ttl=1800)
    generate(current, model_spec("Fluttershy"), 0, "first", "input-a")
    generate(current, model_spec("Rarity"), 0, "second", "input-b")

    state = current.state()
    assert sorted(state["loaded_models"]) == ["Fluttershy", "Rarity"]
    fluttershy = next(
        detail
        for detail in state["loaded_model_details"]
        if detail["character"] == "Fluttershy"
    )
    assert fluttershy["minimum_residency_remaining_seconds"] > 1700

    unloaded = current.unload("Fluttershy", "cpu")
    assert [item["character"] for item in unloaded["unloaded_models"]] == ["Fluttershy"]
    assert activity["created"][0].closed


def test_cpu_worker_applies_replica_thread_setting_before_model_import():
    with patch.dict(
        os.environ,
        {
            "HAY_SAY_MODEL_CPU_THREADS": "12",
            "HAY_SAY_SVC5_CPU_THREADS_PER_WORKER": "8",
        },
    ):
        with patch("hay_say_torch_bootstrap.configure_torch_threads") as configure:
            expected = object()
            configure.return_value = expected
            assert runtime_module.configure_worker_threads("cpu") is expected
            configure.assert_called_once_with(force=True, intraop_threads=8)


def test_cancel_tombstone_blocks_the_output_cache_commit():
    current, _activity = make_runtime(1)
    committed = []
    current.cancel(["too-late"])

    with pytest.raises(runtime_module.GenerationCancelled):
        current.commit_if_active("too-late", lambda: committed.append(True))

    assert committed == []


def test_idle_ttl_is_clamped_to_thirty_minutes_and_invalid_values_fail():
    cache = runtime_module.ModelCache(1, 1, 1, 0, replica_factory=lambda *_args: None)
    assert cache.idle_ttl_seconds == runtime_module.MIN_MODEL_IDLE_TTL_SECONDS

    for value in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite non-negative"):
            runtime_module.ModelCache(
                1, 1, 1, value, replica_factory=lambda *_args: None
            )


def test_explicit_zero_worker_count_is_not_replaced_by_environment_default():
    with pytest.raises(ValueError, match="cpu_workers must be a positive integer"):
        runtime_module.build_runtime(cpu_workers=0)

import io
import os
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import hay_say_torch_bootstrap


from ubuntuserver.hay_say.so_vits_svc_4.runtime import (
    ClipJob,
    GenerationCancelled,
    ModelCache,
    ModelSpec,
    ProcessModelWorker,
    SVC4Runtime,
    _call_model_infer,
    configure_worker_threads,
)


def lightweight_worker_target(connection, spec, _device, _enhance):
    connection.send(("ready", {"sample_rate": spec.target_sample}))
    while True:
        command, _payload = connection.recv()
        if command == "close":
            connection.send(("closed", None))
            break
        connection.send(("error", "unexpected command"))
    connection.close()


def model_spec(version="4.0", target_sample=20):
    return ModelSpec(
        character="Fluttershy",
        version=version,
        source_root="/tmp/svc4-source",
        model_path="/tmp/Fluttershy/G_1.pth",
        config_path="/tmp/Fluttershy/config.json",
        cluster_path="",
        target_sample=target_sample,
        model_revision=(1, 2, 3, 4),
        config_revision=(1, 2, 5, 6),
    )


class FakeWorker:
    def __init__(self, name, activity):
        self.name = name
        self.activity = activity
        self.pid = None
        self.closed = False

    def segment(self, job):
        return ((False, job.audio),), job.sample_rate

    def infer(self, job):
        with self.activity["lock"]:
            self.activity["active"] += 1
            self.activity["maximum"] = max(
                self.activity["maximum"], self.activity["active"]
            )
        try:
            time.sleep(0.02)
            length = round(len(job.audio) * 20 / job.sample_rate)
            return np.full(length, job.index, dtype=np.float32)
        finally:
            with self.activity["lock"]:
                self.activity["active"] -= 1

    def close(self):
        self.closed = True


class RuntimeTests(unittest.TestCase):
    def make_runtime(self, workers=4):
        activity = {"lock": threading.Lock(), "active": 0, "maximum": 0}
        created = []

        def factory(_spec, _device, _enhance, name):
            worker = FakeWorker(name, activity)
            created.append(worker)
            return worker

        runtime = SVC4Runtime(ModelCache(workers, worker_factory=factory))
        return runtime, activity, created

    def test_forced_slices_run_in_parallel_and_keep_rate_correct_length(self):
        runtime, activity, created = self.make_runtime(4)
        output, sample_rate, workers = runtime.generate(
            model_spec(),
            "Fluttershy",
            np.arange(30, dtype=np.float32),
            10,
            "cpu",
            pitch=0,
            slice_seconds=1.0,
            crossfade_seconds=0.2,
            slice_workers=3,
            cluster_ratio=0,
            predict_pitch=False,
            reduce_hoarseness=False,
            enhance=False,
            noise_scale=0.4,
        )

        self.assertEqual(20, sample_rate)
        self.assertEqual(60, len(output))
        self.assertEqual(3, workers)
        self.assertEqual(3, len(created))
        self.assertGreaterEqual(activity["maximum"], 2)
        state = runtime.state()
        self.assertEqual("warm-idle", state["status"])
        self.assertEqual(0, state["active_jobs"])
        self.assertEqual(["Fluttershy"], state["loaded_models"])

    def test_gpu_never_uses_more_than_one_slice_worker(self):
        runtime, activity, created = self.make_runtime(4)
        output, _, workers = runtime.generate(
            model_spec(),
            "Fluttershy",
            np.arange(30, dtype=np.float32),
            10,
            "cuda:0",
            0,
            1.0,
            0.2,
            4,
            0,
            False,
            False,
            False,
            0.4,
        )
        self.assertEqual(60, len(output))
        self.assertEqual(1, workers)
        self.assertEqual(1, len(created))
        self.assertEqual(1, activity["maximum"])

    def test_requested_cpu_workers_are_capped_by_server_limit(self):
        runtime, _, _ = self.make_runtime(4)
        self.assertEqual(4, runtime._workers(64, "cpu", 20))

    def test_model_groups_stay_warm_for_thirty_minutes_under_cache_pressure(self):
        activity = {"lock": threading.Lock(), "active": 0, "maximum": 0}
        created = []

        def factory(_spec, _device, _enhance, name):
            worker = FakeWorker(name, activity)
            created.append(worker)
            return worker

        cache = ModelCache(2, max_models_per_device=1, worker_factory=factory)
        first = model_spec()
        second = replace(
            first,
            character="Rainbow Dash",
            model_path="/tmp/Rainbow Dash/G_1.pth",
        )
        third = replace(
            first,
            character="Applejack",
            model_path="/tmp/Applejack/G_1.pth",
        )
        with cache.acquire(first, "cpu", False) as group:
            group.warm(1)
        first_worker = created[0]
        with cache.acquire(second, "cpu", False) as group:
            group.warm(1)

        state = cache.snapshot()
        self.assertEqual(1800, state["idle_ttl_seconds"])
        self.assertEqual(["Fluttershy", "Rainbow Dash"], [
            item["character"] for item in state["loaded_models"]
        ])
        self.assertFalse(first_worker.closed)

        with cache._lock:
            next(iter(cache._entries.values())).last_used -= 1801
        with cache.acquire(third, "cpu", False) as group:
            group.warm(1)

        self.assertTrue(first_worker.closed)
        self.assertEqual(["Rainbow Dash", "Applejack"], [
            item["character"] for item in cache.snapshot()["loaded_models"]
        ])
        cache.close()

    def test_server_rejects_a_worker_limit_above_ui_maximum(self):
        with self.assertRaisesRegex(ValueError, "between one and 64"):
            ModelCache(65)

    def test_crossfade_validation_applies_to_all_silence(self):
        runtime, _, _ = self.make_runtime(2)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            runtime.generate(
                model_spec(), "Fluttershy", np.zeros(30), 10, "cpu", 0,
                0.1, 0.2, 0, 0, False, False, False, 0.4,
            )

    def test_cancel_drops_queued_clips_and_requests_without_closing_worker(self):
        started = threading.Event()
        release = threading.Event()
        created = []

        class BlockingWorker(FakeWorker):
            def __init__(self, name, activity):
                super().__init__(name, activity)
                self.infer_calls = 0

            def infer(self, job):
                self.infer_calls += 1
                started.set()
                if not release.wait(timeout=3):
                    raise RuntimeError("test inference was not released")
                length = round(len(job.audio) * 20 / job.sample_rate)
                return np.full(length, job.index, dtype=np.float32)

        activity = {"lock": threading.Lock(), "active": 0, "maximum": 0}

        def factory(_spec, _device, _enhance, name):
            worker = BlockingWorker(name, activity)
            created.append(worker)
            return worker

        runtime = SVC4Runtime(ModelCache(1, worker_factory=factory))
        errors = {}

        def generate(request_id):
            try:
                with runtime.cancellation_scope(request_id) as cancellation:
                    runtime.generate(
                        model_spec(),
                        "Fluttershy",
                        np.arange(30, dtype=np.float32),
                        10,
                        "cpu",
                        pitch=0,
                        slice_seconds=1.0,
                        crossfade_seconds=0.2,
                        slice_workers=1,
                        cluster_ratio=0,
                        predict_pitch=False,
                        reduce_hoarseness=False,
                        enhance=False,
                        noise_scale=0.4,
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

        runtime.cancel(["request-active"])
        time.sleep(0.05)
        self.assertTrue(active.is_alive())
        self.assertFalse(created[0].closed)
        release.set()
        active.join(timeout=2)

        self.assertFalse(active.is_alive())
        self.assertIsInstance(errors.get("request-active"), GenerationCancelled)
        self.assertEqual(1, created[0].infer_calls)
        self.assertFalse(created[0].closed)
        self.assertEqual("warm-idle", runtime.state()["status"])

    def test_cancel_tombstone_rejects_a_request_that_has_not_started(self):
        runtime, _, created = self.make_runtime(1)

        runtime.cancel(["request-future"])

        with self.assertRaises(GenerationCancelled):
            with runtime.cancellation_scope("request-future"):
                self.fail("a pre-cancelled request entered its runtime scope")
        self.assertEqual([], created)

    def test_output_commit_is_linearized_before_later_cancellation(self):
        runtime, _, created = self.make_runtime(1)
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
        self.assertEqual([], created)
        self.assertFalse(commit_thread.is_alive())
        self.assertFalse(cancel_thread.is_alive())


class VersionSignatureTests(unittest.TestCase):
    class Model:
        def __init__(self):
            self.calls = []

        def infer(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return np.zeros(2), 2

    def job(self, reduce=True):
        return ClipJob(
            0, np.ones(2), 10, "Fluttershy", 3, 0.25, False, 0.4, reduce, False
        )

    def test_version_40_uses_mean_pooling_argument(self):
        model = self.Model()
        _call_model_infer(model, model_spec("4.0"), "cpu", self.job(), io.BytesIO())
        kwargs = model.calls[0][1]
        self.assertTrue(kwargs["F0_mean_pooling"])
        self.assertNotIn("f0_predictor", kwargs)

    def test_version_41_maps_reduce_hoarseness_to_crepe_predictor(self):
        model = self.Model()
        _call_model_infer(model, model_spec("4.1"), "cpu", self.job(), io.BytesIO())
        kwargs = model.calls[0][1]
        self.assertEqual("crepe", kwargs["f0_predictor"])
        self.assertNotIn("F0_mean_pooling", kwargs)


class ProcessWorkerTests(unittest.TestCase):
    def test_cpu_worker_uses_svc4_thread_setting_before_model_import(self):
        with patch.dict(
            os.environ,
            {
                "HAY_SAY_MODEL_CPU_THREADS": "12",
                "HAY_SAY_SVC4_CPU_THREADS_PER_WORKER": "1",
            },
        ):
            with patch("hay_say_torch_bootstrap.configure_torch_threads") as configure:
                configure.return_value = object()
                self.assertIs(configure.return_value, configure_worker_threads("cpu"))
                configure.assert_called_once_with(force=True, intraop_threads=1)
                self.assertEqual("12", os.environ["HAY_SAY_MODEL_CPU_THREADS"])

    def test_gpu_worker_keeps_general_model_thread_setting(self):
        with patch.dict(
            os.environ,
            {
                "HAY_SAY_MODEL_CPU_THREADS": "12",
                "HAY_SAY_SVC4_CPU_THREADS_PER_WORKER": "1",
            },
        ):
            with patch("hay_say_torch_bootstrap.configure_torch_threads") as configure:
                configure_worker_threads("cuda:0")
                configure.assert_called_once_with(force=True)
                self.assertEqual("12", os.environ["HAY_SAY_MODEL_CPU_THREADS"])

    def test_cpu_worker_reconfigures_an_already_initialized_torch_pool(self):
        calls = []
        fake_torch = SimpleNamespace(
            set_num_threads=lambda value: calls.append(("intra", value)),
            set_num_interop_threads=lambda value: calls.append(("interop", value)),
            get_num_interop_threads=lambda: 1,
        )
        with patch.dict(
            os.environ,
            {"HAY_SAY_SVC4_CPU_THREADS_PER_WORKER": "2"},
        ):
            with patch.object(hay_say_torch_bootstrap, "_CONFIGURED_TORCH", fake_torch):
                self.assertIs(fake_torch, configure_worker_threads("cpu"))

        self.assertEqual([("intra", 2)], calls)

    def test_spawned_worker_starts_and_closes_cleanly(self):
        worker = ProcessModelWorker(
            model_spec(),
            "cpu",
            False,
            "lifecycle-test",
            process_target=lightweight_worker_target,
        )
        process = worker._process
        self.assertTrue(process.is_alive())
        self.assertIsNotNone(worker.pid)
        worker.close()
        self.assertFalse(process.is_alive())
        self.assertEqual(0, process.exitcode)
        worker.close()


if __name__ == "__main__":
    unittest.main()

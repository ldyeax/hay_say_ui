import threading
import time
import unittest
from dataclasses import dataclass

import numpy as np


from ubuntuserver.hay_say.so_vits_svc_4.slice_runtime import (
    IsolatedWorkerPool,
    WorkerSpec,
    assemble_crossfaded,
    build_worker_specs,
    plan_forced_clips,
    silence_for_input_length,
)


class ClipPlanningTests(unittest.TestCase):
    def test_crossfade_uses_output_rate_samples(self):
        for input_rate in (22050, 44100, 48000):
            for output_rate in (22050, 44100, 48000):
                with self.subTest(input_rate=input_rate, output_rate=output_rate):
                    clips = plan_forced_clips(
                        total_input_samples=input_rate * 3,
                        input_sample_rate=input_rate,
                        output_sample_rate=output_rate,
                        slice_seconds=1.0,
                        crossfade_seconds=0.1,
                    )
                    self.assertEqual(round(output_rate * 0.1), clips[1].crossfade_output_samples)
                    self.assertEqual(round(input_rate * 0.1), clips[1].crossfade_input_samples)

    def test_boundaries_are_correct_across_arbitrary_rates(self):
        clips = plan_forced_clips(
            total_input_samples=22050 * 2 + 17,
            input_sample_rate=22050,
            output_sample_rate=48000,
            slice_seconds=0.7,
            crossfade_seconds=0.125,
        )
        outputs = {clip.index: np.ones(clip.expected_output_samples) for clip in clips}
        assembled = assemble_crossfaded(clips, outputs)
        self.assertEqual(round((22050 * 2 + 17) * 48000 / 22050), len(assembled))
        self.assertTrue(np.allclose(1.0, assembled))

    def test_crossfade_mixes_ordered_clips(self):
        clips = plan_forced_clips(30, 10, 10, slice_seconds=1.0, crossfade_seconds=0.2)
        outputs = {
            2: np.full(clips[2].expected_output_samples, 20.0),
            0: np.zeros(clips[0].expected_output_samples),
            1: np.full(clips[1].expected_output_samples, 10.0),
        }
        assembled = assemble_crossfaded(clips, outputs)
        self.assertEqual(30, len(assembled))
        np.testing.assert_allclose([0.0, 10.0], assembled[8:10])
        np.testing.assert_allclose([10.0, 20.0], assembled[18:20])

    def test_crossfade_retain_discards_outer_overlap_edges(self):
        clips = plan_forced_clips(20, 10, 10, slice_seconds=1.0, crossfade_seconds=0.4)
        outputs = {
            0: np.zeros(clips[0].expected_output_samples, dtype=np.float32),
            1: np.full(clips[1].expected_output_samples, 10.0, dtype=np.float32),
        }
        assembled = assemble_crossfaded(clips, outputs, retain_ratio=0.5)
        self.assertEqual(20, len(assembled))
        np.testing.assert_allclose([0.0, 10.0], assembled[7:9])
        self.assertEqual(np.float32, assembled.dtype)

    def test_crossfade_must_not_exceed_slice(self):
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            plan_forced_clips(100, 100, 100, slice_seconds=0.2, crossfade_seconds=0.21)
        with self.assertRaisesRegex(ValueError, "requires forced slicing"):
            plan_forced_clips(100, 100, 100, slice_seconds=0, crossfade_seconds=0.1)

    def test_silence_duration_is_converted_to_target_rate(self):
        silence = silence_for_input_length(2205, 22050, 48000)
        self.assertEqual(4800, len(silence))
        self.assertEqual(np.float32, silence.dtype)
        self.assertFalse(np.any(silence))


@dataclass(frozen=True)
class Job:
    index: int
    delay: float


class RecordingWorker:
    def __init__(self, name, activity):
        self.name = name
        self.activity = activity
        self.lock = threading.Lock()
        self.closed = False

    def infer(self, job):
        if not self.lock.acquire(blocking=False):
            raise AssertionError("the same model object ran concurrently")
        try:
            with self.activity["lock"]:
                self.activity["active"] += 1
                self.activity["maximum"] = max(self.activity["maximum"], self.activity["active"])
            time.sleep(job.delay)
            return f"{job.index}:{self.name}"
        finally:
            with self.activity["lock"]:
                self.activity["active"] -= 1
            self.lock.release()

    def close(self):
        self.closed = True


class IsolatedWorkerPoolTests(unittest.TestCase):
    def test_results_are_ordered_while_distinct_workers_run_concurrently(self):
        activity = {"lock": threading.Lock(), "active": 0, "maximum": 0}
        workers = []

        def factory(ordinal):
            worker = RecordingWorker(f"cpu-{ordinal}", activity)
            workers.append(worker)
            return worker

        jobs = [Job(0, 0.08), Job(1, 0.01), Job(2, 0.04), Job(3, 0.01), Job(4, 0.01)]
        with IsolatedWorkerPool(build_worker_specs(3, factory)) as pool:
            results = pool.map_indexed(jobs, lambda worker, job: worker.infer(job))

        self.assertEqual([0, 1, 2, 3, 4], [result.index for result in results])
        self.assertGreaterEqual(activity["maximum"], 2)
        self.assertEqual(3, len(workers))
        self.assertTrue(all(worker.closed for worker in workers))

    def test_cpu_count_is_configurable_and_gpu_count_is_one(self):
        specs = build_worker_specs(4, lambda ordinal: ordinal, gpu_factory=lambda: "gpu")
        self.assertEqual(["cpu", "cpu", "cpu", "cpu", "cuda"], [spec.device for spec in specs])
        with self.assertRaisesRegex(ValueError, "at most one GPU"):
            IsolatedWorkerPool(
                (
                    WorkerSpec("gpu-0", "cuda:0", lambda: object()),
                    WorkerSpec("gpu-1", "cuda:1", lambda: object()),
                )
            )

    def test_one_worker_is_never_reentered(self):
        activity = {"lock": threading.Lock(), "active": 0, "maximum": 0}
        worker = RecordingWorker("only", activity)
        specs = (WorkerSpec("cpu-0", "cpu", lambda: worker),)
        with IsolatedWorkerPool(specs) as pool:
            pool.map_indexed([Job(index, 0.005) for index in range(5)], lambda model, job: model.infer(job))
        self.assertEqual(1, activity["maximum"])

    def test_pool_grows_lazily_and_honors_per_map_concurrency(self):
        activity = {"lock": threading.Lock(), "active": 0, "maximum": 0}
        workers = []

        def factory(ordinal):
            worker = RecordingWorker(f"cpu-{ordinal}", activity)
            workers.append(worker)
            return worker

        pool = IsolatedWorkerPool(build_worker_specs(4, factory))
        first = [Job(index, 0.005) for index in range(3)]
        pool.map_indexed(first, lambda worker, job: worker.infer(job), max_workers=1)
        self.assertEqual(1, pool.started_workers)
        self.assertEqual(1, activity["maximum"])

        activity["maximum"] = 0
        pool.map_indexed(first, lambda worker, job: worker.infer(job), max_workers=3)
        self.assertEqual(3, pool.started_workers)
        self.assertGreaterEqual(activity["maximum"], 2)
        self.assertEqual(3, len(pool.state()))
        pool.close()
        self.assertTrue(all(worker.closed for worker in workers))

    def test_new_workers_start_with_bounded_parallelism_and_keep_spec_order(self):
        activity = {"lock": threading.Lock(), "active": 0, "maximum": 0}
        workers = []

        def factory(ordinal):
            with activity["lock"]:
                activity["active"] += 1
                activity["maximum"] = max(activity["maximum"], activity["active"])
            try:
                time.sleep(0.02)
                worker = RecordingWorker(f"cpu-{ordinal}", activity)
                workers.append(worker)
                return worker
            finally:
                with activity["lock"]:
                    activity["active"] -= 1

        pool = IsolatedWorkerPool(build_worker_specs(8, factory), startup_concurrency=3)
        pool.start(8)

        self.assertGreaterEqual(activity["maximum"], 2)
        self.assertLessEqual(activity["maximum"], 3)
        self.assertEqual(
            [f"cpu-{ordinal}" for ordinal in range(8)],
            [worker["name"] for worker in pool.state()],
        )
        pool.close()
        self.assertTrue(all(worker.closed for worker in workers))

    def test_parallel_start_failure_closes_only_new_workers(self):
        activity = {"lock": threading.Lock(), "active": 0, "maximum": 0}
        workers = {}

        def factory(ordinal):
            if ordinal == 2:
                time.sleep(0.01)
                raise RuntimeError("worker startup failed")
            worker = RecordingWorker(f"cpu-{ordinal}", activity)
            workers[ordinal] = worker
            return worker

        pool = IsolatedWorkerPool(build_worker_specs(4, factory), startup_concurrency=3)
        pool.start(1)

        with self.assertRaisesRegex(RuntimeError, "worker startup failed"):
            pool.start(4)

        self.assertEqual(1, pool.started_workers)
        self.assertFalse(workers[0].closed)
        self.assertTrue(workers[1].closed)
        self.assertTrue(workers[3].closed)
        pool.close()
        self.assertTrue(workers[0].closed)

    def test_rejects_invalid_startup_concurrency(self):
        specs = build_worker_specs(1, lambda ordinal: ordinal)
        with self.assertRaisesRegex(ValueError, "startup_concurrency"):
            IsolatedWorkerPool(specs, startup_concurrency=0)


if __name__ == "__main__":
    unittest.main()

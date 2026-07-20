import importlib.util
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import hay_say_common as hsc


RUNTIME_PATH = (
    Path(__file__).resolve().parents[1]
    / "ubuntuserver/hay_say/rvc_server/rvc_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("test_rvc_runtime_module", RUNTIME_PATH)
rvc_runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rvc_runtime
SPEC.loader.exec_module(rvc_runtime)


class FakeWorker:
    instances = []
    next_pid = 1000

    def __init__(self, spec, **_options):
        self.spec = spec
        self.created_at = time.monotonic()
        self.last_used = self.created_at
        self.request_id = None
        self.reported_device = spec.device
        self.cancelled = threading.Event()
        self.started = threading.Event()
        self.release = threading.Event()
        self.alive = True
        self.runs = []
        self.process = SimpleNamespace(pid=FakeWorker.next_pid)
        FakeWorker.next_pid += 1
        FakeWorker.instances.append(self)

    @property
    def is_alive(self):
        return self.alive

    def run(self, job):
        self.runs.append(job["request_id"])
        self.started.set()
        if job.get("block"):
            while not self.cancelled.wait(0.01) and not self.release.is_set():
                pass
        if self.cancelled.is_set():
            self.cancelled.clear()
            raise hsc.InferenceCancelled("cancelled")

    def cancel(self, request_id):
        if self.request_id != request_id:
            return False
        self.cancelled.set()
        return True

    def stop(self):
        self.alive = False
        self.release.set()


def runtime(monkeypatch, workers=2):
    FakeWorker.instances = []
    monkeypatch.setenv("HAY_SAY_RVC_CPU_WORKERS", str(workers))
    monkeypatch.setenv("HAY_SAY_RVC_GPU_WORKERS", "1")
    monkeypatch.setenv("HAY_SAY_MODEL_IDLE_TTL_SECONDS", "1800")
    return rvc_runtime.RvcRuntime(
        worker_factory=FakeWorker,
        idle_ttl_seconds=1800,
        reaper_interval=3600,
    )


def run_job(runtime, request_id, character="Fluttershy", gpu_id="", **job):
    return runtime.run(
        {"request_id": request_id, **job},
        character=character,
        gpu_id=gpu_id,
        environment={},
        python_executable="python",
        worker_script="worker.py",
        cwd="/tmp",
    )


def test_parallel_cpu_jobs_spawn_duplicate_warm_replicas_and_reuse_them(monkeypatch):
    pool = runtime(monkeypatch, workers=2)
    errors = []

    def invoke(request_id):
        try:
            run_job(pool, request_id, block=True)
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=invoke, args=(f"job-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 5
    while len(FakeWorker.instances) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(FakeWorker.instances) == 2
    assert all(worker.started.wait(1) for worker in FakeWorker.instances)
    for worker in FakeWorker.instances:
        worker.release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    state = pool.state()
    assert state["status"] == "warm-idle"
    assert state["workers"] == 2
    assert state["active_jobs"] == 0
    assert state["queued_jobs"] == 0
    assert state["device"] == "cpu"
    assert state["loaded_models"] == ["Fluttershy", "Fluttershy"]
    assert state["idle_ttl_seconds"] == 1800
    assert all(worker["idle_ttl_remaining_seconds"] > 1790 for worker in state["loaded_model_details"])

    run_job(pool, "next-job")
    assert len(FakeWorker.instances) == 2
    pool.close()


def test_cancel_keeps_persistent_worker_warm_for_the_next_request(monkeypatch):
    pool = runtime(monkeypatch, workers=1)
    errors = []

    def invoke():
        try:
            run_job(pool, "cancel-this", block=True)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=invoke)
    thread.start()
    deadline = time.monotonic() + 5
    while not FakeWorker.instances and time.monotonic() < deadline:
        time.sleep(0.01)
    worker = FakeWorker.instances[0]
    assert worker.started.wait(1)

    result = pool.cancel(["cancel-this"])
    thread.join(timeout=5)

    assert result["active_workers_signalled"] == 1
    assert len(errors) == 1
    assert isinstance(errors[0], hsc.InferenceCancelled)
    assert pool.state()["status"] == "warm-idle"
    assert worker.is_alive

    run_job(pool, "next-request")
    assert FakeWorker.instances == [worker]
    assert worker.runs == ["cancel-this", "next-request"]
    pool.close()


def test_cancel_uses_request_id_snapshotted_with_worker(monkeypatch):
    class ReassignedWorker(FakeWorker):
        def __init__(self, spec, **options):
            super().__init__(spec, **options)
            self.reassign_on_cancel_lookup = False
            self.cancel_arguments = []

        def __getattribute__(self, name):
            if name == "cancel" and object.__getattribute__(
                self, "reassign_on_cancel_lookup"
            ):
                object.__setattr__(self, "reassign_on_cancel_lookup", False)
                object.__setattr__(self, "request_id", "request-b")
            return super().__getattribute__(name)

        def cancel(self, request_id):
            self.cancel_arguments.append(request_id)
            return True

    FakeWorker.instances = []
    monkeypatch.setenv("HAY_SAY_RVC_CPU_WORKERS", "1")
    monkeypatch.setenv("HAY_SAY_RVC_GPU_WORKERS", "1")
    pool = rvc_runtime.RvcRuntime(
        worker_factory=ReassignedWorker,
        idle_ttl_seconds=1800,
        reaper_interval=3600,
    )
    worker = ReassignedWorker(hsc.WorkerSpec("Fluttershy", "cpu"))
    worker.request_id = "request-a"
    worker.reassign_on_cancel_lookup = True
    with pool._condition:
        pool._workers.append(worker)

    result = pool.cancel(["request-a"])

    assert result["active_workers_signalled"] == 1
    assert worker.request_id == "request-b"
    assert worker.cancel_arguments == ["request-a"]
    pool.close()


def test_cancel_continues_signalling_healthy_replicas_after_transport_failure(monkeypatch):
    pool = runtime(monkeypatch, workers=2)
    first = FakeWorker(hsc.WorkerSpec("Fluttershy", "cpu"))
    second = FakeWorker(hsc.WorkerSpec("Fluttershy", "cpu"))
    first.request_id = second.request_id = "request-a"

    def fail_cancel(_request_id):
        raise BrokenPipeError("control socket closed")

    first.cancel = fail_cancel
    with pool._condition:
        pool._workers.extend([first, second])

    result = pool.cancel(["request-a"])

    assert result["active_workers_signalled"] == 1
    assert result["active_workers_signal_failed"] == 1
    assert second.cancelled.is_set()
    assert first.is_alive
    assert second.is_alive
    pool.close()


def test_cancelled_request_cannot_commit_after_worker_returns(monkeypatch):
    pool = runtime(monkeypatch, workers=1)
    pool.cancel(["already-cancelled"])
    with pytest.raises(hsc.InferenceCancelled):
        run_job(pool, "already-cancelled")
    committed = []
    with pytest.raises(hsc.InferenceCancelled):
        pool.commit_if_active("already-cancelled", lambda: committed.append(True))
    assert not committed
    assert not FakeWorker.instances
    pool.close()


def test_cancel_during_worker_warmup_keeps_the_new_replica(monkeypatch):
    FakeWorker.instances = []
    monkeypatch.setenv("HAY_SAY_RVC_CPU_WORKERS", "1")
    monkeypatch.setenv("HAY_SAY_RVC_GPU_WORKERS", "1")
    monkeypatch.setenv("HAY_SAY_MODEL_IDLE_TTL_SECONDS", "1800")
    constructing = threading.Event()
    finish_warmup = threading.Event()

    def slow_factory(spec, **options):
        constructing.set()
        assert finish_warmup.wait(5)
        return FakeWorker(spec, **options)

    pool = rvc_runtime.RvcRuntime(
        worker_factory=slow_factory,
        idle_ttl_seconds=1800,
        reaper_interval=3600,
    )
    errors = []

    def invoke():
        try:
            run_job(pool, "cancel-during-warmup")
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=invoke)
    thread.start()
    assert constructing.wait(2)
    pool.cancel(["cancel-during-warmup"])
    finish_warmup.set()
    thread.join(timeout=5)

    assert len(errors) == 1
    assert isinstance(errors[0], hsc.InferenceCancelled)
    assert len(FakeWorker.instances) == 1
    worker = FakeWorker.instances[0]
    assert worker.is_alive
    assert pool.state()["status"] == "warm-idle"

    run_job(pool, "next-request")
    assert FakeWorker.instances == [worker]
    pool.close()


def test_idle_gpu_character_does_not_block_a_different_warm_replica(monkeypatch):
    pool = runtime(monkeypatch, workers=1)

    run_job(pool, "first", character="Fluttershy", gpu_id=0)
    first = FakeWorker.instances[0]
    run_job(pool, "second", character="Rarity", gpu_id=0)

    assert len(FakeWorker.instances) == 2
    assert first.is_alive
    state = pool.state()
    assert state["device"] == "cuda:0"
    assert state["loaded_models"] == ["Fluttershy", "Rarity"]
    assert state["workers"] == 2
    pool.close()


def test_cpu_worker_limit_is_global_across_characters(monkeypatch):
    pool = runtime(monkeypatch, workers=1)
    errors = []

    def invoke(request_id, character):
        try:
            run_job(pool, request_id, character=character, block=True)
        except Exception as error:
            errors.append(error)

    first_thread = threading.Thread(target=invoke, args=("first", "Fluttershy"))
    second_thread = threading.Thread(target=invoke, args=("second", "Rarity"))
    first_thread.start()
    deadline = time.monotonic() + 5
    while not FakeWorker.instances and time.monotonic() < deadline:
        time.sleep(0.01)
    assert FakeWorker.instances[0].started.wait(1)

    second_thread.start()
    deadline = time.monotonic() + 5
    while pool.state()["queued_jobs"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pool.state()["queued_jobs"] == 1
    assert pool.state()["active_jobs"] == 1
    assert len(FakeWorker.instances) == 1

    FakeWorker.instances[0].release.set()
    deadline = time.monotonic() + 5
    while len(FakeWorker.instances) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(FakeWorker.instances) == 2
    assert FakeWorker.instances[1].started.wait(1)
    FakeWorker.instances[1].release.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not errors
    assert pool.state()["loaded_models"] == ["Fluttershy", "Rarity"]
    pool.close()


@pytest.mark.parametrize("gpu_id", ["", 0])
def test_idle_matching_replica_still_obeys_global_device_admission(monkeypatch, gpu_id):
    pool = runtime(monkeypatch, workers=1)
    run_job(pool, "warm-a", character="Fluttershy", gpu_id=gpu_id)
    warm_a = FakeWorker.instances[0]
    warm_a.started.clear()
    errors = []

    def invoke(request_id, character):
        try:
            run_job(
                pool, request_id, character=character, gpu_id=gpu_id, block=True
            )
        except Exception as error:
            errors.append(error)

    busy_b = threading.Thread(target=invoke, args=("busy-b", "Rarity"))
    waiting_a = threading.Thread(target=invoke, args=("waiting-a", "Fluttershy"))
    busy_b.start()
    deadline = time.monotonic() + 5
    while len(FakeWorker.instances) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(FakeWorker.instances) == 2
    busy_worker = FakeWorker.instances[1]
    assert busy_worker.started.wait(1)

    waiting_a.start()
    deadline = time.monotonic() + 5
    while pool.state()["queued_jobs"] != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pool.state()["queued_jobs"] == 1
    assert not warm_a.started.is_set()

    busy_worker.release.set()
    assert warm_a.started.wait(5)
    warm_a.release.set()
    busy_b.join(timeout=5)
    waiting_a.join(timeout=5)

    assert not errors
    pool.close()


def test_dead_idle_worker_is_replaced_without_waiting_for_ttl(monkeypatch):
    pool = runtime(monkeypatch, workers=1)
    run_job(pool, "first")
    first = FakeWorker.instances[0]
    first.alive = False

    run_job(pool, "replacement")

    assert len(FakeWorker.instances) == 2
    assert FakeWorker.instances[1].is_alive
    assert pool.state()["workers"] == 1
    pool.close()


def test_persistent_worker_stays_in_server_process_group(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(
        "import argparse, json, socket\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--control-fd', type=int, required=True)\n"
        "parser.add_argument('--character', required=True)\n"
        "args = parser.parse_args()\n"
        "control = socket.socket(fileno=args.control_fd)\n"
        "reader = control.makefile('r', encoding='utf-8')\n"
        "writer = control.makefile('w', encoding='utf-8')\n"
        "writer.write(json.dumps({'status': 'ready', 'device': 'cpu'}) + '\\n')\n"
        "writer.flush()\n"
        "for line in reader:\n"
        "    if json.loads(line).get('action') == 'stop':\n"
        "        break\n",
        encoding="utf-8",
    )
    worker = hsc.PersistentModelWorker(
        hsc.WorkerSpec("Fluttershy", "cpu"),
        environment=os.environ.copy(),
        python_executable=sys.executable,
        worker_script=str(script),
        cwd=str(tmp_path),
        startup_timeout=10,
    )
    try:
        assert os.getpgid(worker.process.pid) == os.getpgrp()
    finally:
        worker.stop()

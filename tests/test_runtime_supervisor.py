import os
import signal
import sys
import tempfile
import threading
import unittest
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import psutil

from ubuntuserver.runtime.config import ManagerConfig, RuntimeConfig, RuntimeSpec
from ubuntuserver.runtime.supervisor import (
    NvidiaSmiCollector,
    RuntimeOperationError,
    RuntimeSupervisor,
    UnsafeProcessError,
)


MemoryInfo = namedtuple("MemoryInfo", ["rss"])
Uids = namedtuple("Uids", ["real", "effective", "saved"])


class FakeProcess:
    def __init__(self, pid, command, create_time, *, pgid=None, cpu=0.0, rss=0):
        self.pid = pid
        self.command = list(command)
        self.created = create_time
        self.pgid = pgid if pgid is not None else pid
        self.cpu = cpu
        self.rss = rss
        self.alive = True
        self.child_processes = []

    def is_running(self):
        return self.alive

    def status(self):
        return psutil.STATUS_RUNNING

    def create_time(self):
        return self.created

    def cmdline(self):
        return list(self.command)

    def uids(self):
        return Uids(os.getuid(), os.getuid(), os.getuid())

    def cpu_percent(self, interval=None):
        return self.cpu

    def memory_info(self):
        return MemoryInfo(self.rss)

    def children(self, recursive=True):
        return list(self.child_processes)


class FakePopen:
    def __init__(self, process):
        self.process = process
        self.pid = process.pid

    def poll(self):
        return None if self.process.alive else 0


class FakeProcesses:
    def __init__(self, *, with_child=False):
        self.processes = {}
        self.calls = []
        self.signals = []
        self.next_pid = 41000
        self.with_child = with_child
        self._lock = threading.Lock()

    def popen(self, command, **kwargs):
        with self._lock:
            pid = self.next_pid
            self.next_pid += 10
            process = FakeProcess(pid, command, 1_700_000_000.0, cpu=12.5, rss=1000)
            self.processes[pid] = process
            if self.with_child:
                child = FakeProcess(pid + 1, ["worker"], 1_700_000_001.0, pgid=pid, cpu=3.5, rss=500)
                process.child_processes.append(child)
                self.processes[child.pid] = child
            self.calls.append((list(command), kwargs))
            return FakePopen(process)

    def process(self, pid):
        try:
            return self.processes[pid]
        except KeyError as exc:
            raise psutil.NoSuchProcess(pid) from exc

    def getpgid(self, pid):
        process = self.process(pid)
        if not process.alive:
            raise ProcessLookupError(pid)
        return process.pgid

    def killpg(self, pgid, sig):
        self.signals.append((pgid, sig))
        found = False
        for process in self.processes.values():
            if process.pgid == pgid and process.alive:
                process.alive = False
                found = True
        if not found:
            raise ProcessLookupError(pgid)


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeGpuCollector:
    def __init__(self, usage=None):
        self.usage = usage or {}
        self.requested_pids = []

    def collect(self, pids):
        self.requested_pids.append(set(pids))
        return {pid: value for pid, value in self.usage.items() if pid in pids}


def make_config(root, *, collect_gpu_memory=True, start_grace_seconds=30.0):
    manager = ManagerConfig(
        state_dir=root / "state",
        log_dir=root / "logs",
        request_timeout_seconds=0.05,
        start_grace_seconds=start_grace_seconds,
        stop_timeout_seconds=0.1,
        collect_gpu_memory=collect_gpu_memory,
    )
    spec = RuntimeSpec(
        id="rvc",
        label="RVC",
        port=6578,
        command=(sys.executable, "/tmp/rvc-server.py", "--cache_implementation", "file"),
        cwd=Path("/tmp"),
        env={"PYTHONUNBUFFERED": "1"},
    )
    return RuntimeConfig(manager=manager, runtimes={spec.id: spec})


def make_supervisor(root, fakes, *, http_get=None, gpu_collector=None, config=None, wall_time=None):
    def default_http_get(url, timeout):
        return FakeResponse(200, {}) if url.endswith("/gpu-info") else FakeResponse(404)

    return RuntimeSupervisor(
        config or make_config(root),
        popen_factory=fakes.popen,
        process_factory=fakes.process,
        http_get=http_get or default_http_get,
        killpg=fakes.killpg,
        getpgid=fakes.getpgid,
        wall_time=wall_time or (lambda: 1_700_000_010.0),
        gpu_collector=gpu_collector or FakeGpuCollector(),
    )


class RuntimeSupervisorTests(unittest.TestCase):
    def test_cpu_sampling_reuses_process_baseline_across_status_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supervisor = make_supervisor(root, FakeProcesses())

            class BaselineProcess(FakeProcess):
                def __init__(self):
                    super().__init__(9000, ["runtime"], 1234.0)
                    self.samples = 0

                def cpu_percent(self, interval=None):
                    self.samples += 1
                    return 0.0 if self.samples == 1 else 27.5

            first = BaselineProcess()
            second = BaselineProcess()
            self.assertEqual(supervisor._safe_cpu_percent(first), 0.0)
            self.assertEqual(supervisor._safe_cpu_percent(second), 27.5)

    def test_start_and_stop_are_idempotent_and_use_safe_popen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fakes = FakeProcesses()
            supervisor = make_supervisor(root, fakes)

            started = supervisor.start("rvc")
            started_again = supervisor.start("rvc")

            self.assertEqual(started["status"], "starting")
            self.assertEqual(started_again["status"], "ready-cold")
            self.assertEqual(len(fakes.calls), 1)
            command, kwargs = fakes.calls[0]
            self.assertEqual(command, list(make_config(root).runtimes["rvc"].command))
            self.assertIs(kwargs["shell"], False)
            self.assertIs(kwargs["start_new_session"], True)
            self.assertIs(kwargs["close_fds"], True)
            self.assertEqual(kwargs["env"]["HAY_SAY_RUNTIME_PORT"], "6578")
            self.assertTrue((root / "state/rvc.pid").exists())
            self.assertTrue((root / "state/rvc.state.json").exists())
            self.assertTrue((root / "logs/rvc.log").exists())

            stopped = supervisor.stop("rvc")
            stopped_again = supervisor.stop("rvc")

            self.assertEqual(stopped["status"], "stopped")
            self.assertEqual(stopped_again["status"], "stopped")
            self.assertEqual(fakes.signals, [(started["pid"], signal.SIGTERM)])
            self.assertFalse((root / "state/rvc.pid").exists())

    def test_start_does_not_inherit_parent_server_activation_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fakes = FakeProcesses()
            supervisor = make_supervisor(root, fakes)
            transient = {
                "FLASK_RUN_FROM_CLI": "true",
                "LISTEN_FDNAMES": "http",
                "LISTEN_FDS": "1",
                "LISTEN_PID": "1234",
                "WERKZEUG_RUN_MAIN": "true",
                "WERKZEUG_SERVER_FD": "9",
            }

            with mock.patch.dict(os.environ, transient):
                supervisor.start("rvc")

            environment = fakes.calls[0][1]["env"]
            for name in transient:
                self.assertNotIn(name, environment)
            self.assertEqual(environment["PYTHONUNBUFFERED"], "1")
            supervisor.stop("rvc")

    def test_runtime_endpoint_drives_warm_busy_and_resource_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fakes = FakeProcesses(with_child=True)
            payload = {
                "status": "warm-idle",
                "device": "cuda:0",
                "loaded_models": ["twilight-v2"],
                "active_jobs": 0,
                "queued_jobs": 0,
                "last_error": None,
            }
            http_get = lambda url, timeout: FakeResponse(200, payload)
            gpu = FakeGpuCollector({41000: 2 * 1024 * 1024, 41001: 3 * 1024 * 1024})
            supervisor = make_supervisor(root, fakes, http_get=http_get, gpu_collector=gpu)
            supervisor.start("rvc")

            status = supervisor.status("rvc")

            self.assertEqual(status["status"], "warm-idle")
            self.assertEqual(status["device"], "cuda:0")
            self.assertEqual(status["loaded_models"], ["twilight-v2"])
            self.assertEqual(status["cpu_percent"], 12.5)
            self.assertEqual(status["rss_bytes"], 1000)
            self.assertEqual(status["children"], {"count": 1, "cpu_percent": 3.5, "rss_bytes": 500})
            self.assertEqual(status["total_cpu_percent"], 16.0)
            self.assertEqual(status["total_rss_bytes"], 1500)
            self.assertEqual(status["gpu_memory_bytes"], 5 * 1024 * 1024)
            self.assertEqual(status["gpu_memory_by_pid"], {"41000": 2 * 1024 * 1024, "41001": 3 * 1024 * 1024})

            payload.update({"status": "busy", "active_jobs": 1, "queued_jobs": 2})
            busy = supervisor.status("rvc")
            self.assertEqual(busy["status"], "busy")
            self.assertEqual((busy["active_jobs"], busy["queued_jobs"]), (1, 2))
            supervisor.stop("rvc")

    def test_new_supervisor_adopts_only_matching_manager_owned_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fakes = FakeProcesses()
            first = make_supervisor(root, fakes)
            started = first.start("rvc")

            second = make_supervisor(root, fakes)
            adopted = second.start("rvc")

            self.assertEqual(adopted["pid"], started["pid"])
            self.assertEqual(len(fakes.calls), 1)
            second.stop("rvc")
            # Close the original manager's inherited log handle as well.
            first.stop("rvc")

    def test_pid_reuse_mismatch_is_reported_and_never_signaled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fakes = FakeProcesses()
            supervisor = make_supervisor(root, fakes)
            started = supervisor.start("rvc")
            fakes.processes[started["pid"]].created += 100

            status = supervisor.status("rvc")

            self.assertEqual(status["status"], "error")
            self.assertIn("creation time", status["last_error"])
            with self.assertRaises(UnsafeProcessError):
                supervisor.stop("rvc")
            self.assertEqual(fakes.signals, [])
            fakes.processes[started["pid"]].alive = False
            supervisor._close_log_handle("rvc")

    def test_concurrent_start_spawns_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fakes = FakeProcesses()
            supervisor = make_supervisor(root, fakes)

            with ThreadPoolExecutor(max_workers=8) as executor:
                statuses = list(executor.map(lambda _: supervisor.start("rvc"), range(8)))

            self.assertEqual(len(fakes.calls), 1)
            self.assertEqual({status["pid"] for status in statuses}, {41000})
            supervisor.stop("rvc")

    def test_failed_spawn_is_persisted_as_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fakes = FakeProcesses()

            def fail_to_spawn(command, **kwargs):
                raise FileNotFoundError(command[0])

            fakes.popen = fail_to_spawn
            supervisor = make_supervisor(root, fakes)

            with self.assertRaises(RuntimeOperationError):
                supervisor.start("rvc")

            status = supervisor.status("rvc")
            self.assertEqual(status["status"], "error")
            self.assertIn("Unable to start rvc", status["last_error"])

    def test_unreachable_runtime_transitions_from_starting_to_error_after_grace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fakes = FakeProcesses()
            clock = {"now": 100.0}

            def unavailable(url, timeout):
                raise TimeoutError("connection timed out")

            config = make_config(root, start_grace_seconds=5.0)
            supervisor = make_supervisor(
                root,
                fakes,
                config=config,
                http_get=unavailable,
                wall_time=lambda: clock["now"],
            )
            supervisor.start("rvc")

            self.assertEqual(supervisor.status("rvc")["status"], "starting")
            clock["now"] += 6.0
            status = supervisor.status("rvc")

            self.assertEqual(status["status"], "error")
            self.assertIn("connection timed out", status["last_error"])
            supervisor.stop("rvc")

    def test_missing_runtime_telemetry_still_requires_a_passing_health_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fakes = FakeProcesses()
            clock = {"now": 100.0}
            requested_urls = []

            def unhealthy(url, timeout):
                requested_urls.append(url)
                if url.endswith("/runtime"):
                    return FakeResponse(404)
                return FakeResponse(500)

            supervisor = make_supervisor(
                root,
                fakes,
                http_get=unhealthy,
                wall_time=lambda: clock["now"],
            )
            supervisor.start("rvc")

            self.assertEqual(supervisor.status("rvc")["status"], "starting")
            clock["now"] += 31.0
            status = supervisor.status("rvc")

            self.assertEqual(status["status"], "error")
            self.assertIn("Health endpoint returned HTTP 500", status["last_error"])
            self.assertTrue(any(url.endswith("/runtime") for url in requested_urls))
            self.assertTrue(any(url.endswith("/gpu-info") for url in requested_urls))
            supervisor.stop("rvc")

    def test_legacy_health_probe_runs_once_after_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fakes = FakeProcesses()
            requested_urls = []

            def legacy_server(url, timeout):
                requested_urls.append(url)
                if url.endswith("/runtime"):
                    return FakeResponse(404)
                return FakeResponse(200, [{"Index": 0}])

            supervisor = make_supervisor(root, fakes, http_get=legacy_server)
            supervisor.start("rvc")

            self.assertEqual(supervisor.status("rvc")["status"], "ready-cold")
            self.assertEqual(supervisor.status("rvc")["status"], "ready-cold")

            health_requests = [url for url in requested_urls if url.endswith("/gpu-info")]
            self.assertEqual(len(health_requests), 1)
            supervisor.stop("rvc")


class NvidiaSmiCollectorTests(unittest.TestCase):
    def test_parses_memory_by_requested_pid(self):
        completed = mock.Mock(returncode=0, stdout="101, 12\n202, 1.5\n303, invalid\n")
        run = mock.Mock(return_value=completed)
        collector = NvidiaSmiCollector(timeout_seconds=0.2, run=run)

        usage = collector.collect({101, 202})

        self.assertEqual(usage, {101: 12 * 1024 * 1024, 202: int(1.5 * 1024 * 1024)})
        command = run.call_args.args[0]
        self.assertEqual(command[0], "nvidia-smi")
        self.assertNotIn("shell", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()

"""Process lifecycle and telemetry for native model runtimes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import psutil
import requests

from .config import RuntimeConfig, RuntimeSpec


STATE_VERSION = 1
UNSUPPORTED_RUNTIME_ENDPOINT_CODES = {404, 405, 501}
TRANSIENT_SERVER_ENVIRONMENT = (
    "FLASK_RUN_FROM_CLI",
    "LISTEN_FDNAMES",
    "LISTEN_FDS",
    "LISTEN_PID",
    "WERKZEUG_RUN_MAIN",
    "WERKZEUG_SERVER_FD",
)


class RuntimeSupervisorError(RuntimeError):
    """Base class for supervisor failures that can be returned by the API."""


class RuntimeNotFoundError(RuntimeSupervisorError):
    pass


class RuntimeDisabledError(RuntimeSupervisorError):
    pass


class UnsafeProcessError(RuntimeSupervisorError):
    """The PID files point at a process the manager cannot prove it owns."""


class RuntimeOperationError(RuntimeSupervisorError):
    pass


class NvidiaSmiCollector:
    """Collect per-process GPU memory without importing a CUDA framework."""

    def __init__(
        self,
        timeout_seconds: float = 1.0,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._run = run

    def collect(self, pids: set[int]) -> dict[int, int] | None:
        if not pids:
            return {}
        try:
            result = self._run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        usage: dict[int, int] = {}
        for line in result.stdout.splitlines():
            columns = [column.strip() for column in line.split(",")]
            if len(columns) < 2:
                continue
            try:
                pid = int(columns[0])
                mebibytes = float(columns[1])
            except ValueError:
                continue
            if pid in pids:
                usage[pid] = usage.get(pid, 0) + int(mebibytes * 1024 * 1024)
        return usage


def _command_fingerprint(spec: RuntimeSpec) -> str:
    value = json.dumps(
        {"command": list(spec.command), "cwd": str(spec.cwd)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, content: bytes, mode: int = 0o600) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    content = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _atomic_write(path, content)


class RuntimeSupervisor:
    """Own, inspect, and stop the configured model service process groups."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        process_factory: Callable[[int], Any] = psutil.Process,
        http_get: Callable[..., Any] = requests.get,
        killpg: Callable[[int, int], None] = os.killpg,
        getpgid: Callable[[int], int] = os.getpgid,
        wall_time: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        gpu_collector: Any | None = None,
    ) -> None:
        self.config = config
        self._popen = popen_factory
        self._process = process_factory
        self._http_get = http_get
        self._killpg = killpg
        self._getpgid = getpgid
        self._wall_time = wall_time
        self._monotonic = monotonic
        self._sleep = sleep
        self._thread_locks = {runtime_id: threading.RLock() for runtime_id in config.runtimes}
        self._handles: dict[str, Any] = {}
        self._log_handles: dict[str, Any] = {}
        self._cpu_processes: dict[tuple[int, float], Any] = {}
        self._cpu_sample_lock = threading.Lock()
        self._gpu_collector = gpu_collector or NvidiaSmiCollector(
            min(config.manager.request_timeout_seconds, 2.0)
        )

        for directory in (config.manager.state_dir, config.manager.log_dir):
            existed = directory.exists()
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not existed:
                directory.chmod(0o700)
        self.owner_id = self._load_or_create_owner_id()

    def _load_or_create_owner_id(self) -> str:
        owner_path = self.config.manager.state_dir / "manager.json"
        lock_path = self.config.manager.state_dir / ".manager.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            os.chmod(lock_path, 0o600)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                if owner_path.exists():
                    try:
                        owner = json.loads(owner_path.read_text(encoding="utf-8"))
                        owner_id = owner["owner_id"]
                    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                        raise RuntimeOperationError(f"Invalid manager ownership file: {owner_path}") from exc
                    if (
                        owner.get("state_version") != STATE_VERSION
                        or not isinstance(owner_id, str)
                        or len(owner_id) != 32
                        or any(character not in "0123456789abcdef" for character in owner_id)
                    ):
                        raise RuntimeOperationError(f"Invalid manager ownership file: {owner_path}")
                    return owner_id
                owner_id = uuid.uuid4().hex
                _atomic_write_json(
                    owner_path,
                    {"state_version": STATE_VERSION, "owner_id": owner_id},
                )
                return owner_id
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _spec(self, runtime_id: str) -> RuntimeSpec:
        try:
            return self.config.runtimes[runtime_id]
        except KeyError as exc:
            raise RuntimeNotFoundError(f"Unknown runtime: {runtime_id}") from exc

    def _state_path(self, runtime_id: str) -> Path:
        return self.config.manager.state_dir / f"{runtime_id}.state.json"

    def _pid_path(self, runtime_id: str) -> Path:
        return self.config.manager.state_dir / f"{runtime_id}.pid"

    def _lock_path(self, runtime_id: str) -> Path:
        return self.config.manager.state_dir / f"{runtime_id}.lock"

    def _log_path(self, runtime_id: str) -> Path:
        return self.config.manager.log_dir / f"{runtime_id}.log"

    @contextmanager
    def _runtime_lock(self, runtime_id: str) -> Iterator[None]:
        with self._thread_locks[runtime_id]:
            lock_path = self._lock_path(runtime_id)
            with lock_path.open("a+", encoding="utf-8") as lock_file:
                os.chmod(lock_path, 0o600)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_state(self, runtime_id: str) -> dict[str, Any] | None:
        path = self._state_path(runtime_id)
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UnsafeProcessError(f"Cannot safely read runtime state file {path}") from exc
        if not isinstance(state, dict):
            raise UnsafeProcessError(f"Runtime state file is not a JSON object: {path}")
        return state

    def _read_pid(self, runtime_id: str) -> int | None:
        path = self._pid_path(runtime_id)
        if not path.exists():
            return None
        try:
            pid = int(path.read_text(encoding="ascii").strip())
        except (OSError, UnicodeError, ValueError) as exc:
            raise UnsafeProcessError(f"Cannot safely read runtime PID file {path}") from exc
        if pid <= 1:
            raise UnsafeProcessError(f"Unsafe PID {pid} in {path}")
        return pid

    def _write_state(self, runtime_id: str, state: Mapping[str, Any]) -> None:
        _atomic_write_json(self._state_path(runtime_id), state)

    def _write_pid(self, runtime_id: str, pid: int) -> None:
        _atomic_write(self._pid_path(runtime_id), f"{pid}\n".encode("ascii"))

    def _remove_pid_file(self, runtime_id: str) -> None:
        try:
            self._pid_path(runtime_id).unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _process_is_alive(process: Any) -> bool:
        try:
            return bool(process.is_running()) and process.status() != psutil.STATUS_ZOMBIE
        except psutil.AccessDenied:
            # Never interpret an uninspectable process as dead; later ownership
            # checks will reject it if required attributes are inaccessible.
            return True
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return False

    def _find_process(self, pid: int) -> Any | None:
        try:
            process = self._process(pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return None
        except psutil.AccessDenied as exc:
            raise UnsafeProcessError(f"Cannot inspect live PID {pid}; refusing to manage it") from exc
        return process if self._process_is_alive(process) else None

    def _resolve_owned_process(
        self,
        spec: RuntimeSpec,
        state: Mapping[str, Any] | None,
    ) -> Any | None:
        pid_file = self._read_pid(spec.id)
        state_pid_raw = state.get("pid") if state else None
        try:
            state_pid = int(state_pid_raw) if state_pid_raw is not None else None
        except (TypeError, ValueError) as exc:
            raise UnsafeProcessError(f"Invalid PID in state for {spec.id}") from exc

        if pid_file is None and state_pid is None:
            return None
        if pid_file != state_pid:
            candidate_pids = {pid for pid in (pid_file, state_pid) if pid is not None}
            live_pids = [pid for pid in candidate_pids if self._find_process(pid) is not None]
            if live_pids:
                raise UnsafeProcessError(
                    f"PID/state mismatch for {spec.id}; refusing to adopt or signal live PID(s) {live_pids}"
                )
            self._remove_pid_file(spec.id)
            return None

        assert pid_file is not None
        process = self._find_process(pid_file)
        if process is None:
            self._remove_pid_file(spec.id)
            return None
        if state is None:
            raise UnsafeProcessError(
                f"Live PID {pid_file} has no manager state; refusing to adopt it for {spec.id}"
            )

        violations: list[str] = []
        if state.get("state_version") != STATE_VERSION:
            violations.append("state version")
        if state.get("owner_id") != self.owner_id:
            violations.append("manager owner")
        if state.get("runtime_id") != spec.id:
            violations.append("runtime id")
        if state.get("command_fingerprint") != _command_fingerprint(spec):
            violations.append("configured command")
        try:
            expected_create_time = float(state["create_time"])
            if abs(float(process.create_time()) - expected_create_time) > 0.01:
                violations.append("process creation time")
        except (KeyError, TypeError, ValueError, psutil.Error):
            violations.append("process creation time")
        try:
            if list(process.cmdline()) != list(spec.command):
                violations.append("process command line")
        except psutil.Error:
            violations.append("process command line")
        try:
            uids = process.uids()
            if int(uids.real) != os.getuid():
                violations.append("process user")
        except AttributeError:
            # psutil does not expose uids on every platform. This service is
            # targeted at Ubuntu, but command/create-time/PGID checks remain.
            pass
        except psutil.Error:
            violations.append("process user")
        try:
            if self._getpgid(pid_file) != pid_file:
                violations.append("process group")
        except (OSError, ProcessLookupError):
            violations.append("process group")

        if violations:
            joined = ", ".join(violations)
            raise UnsafeProcessError(
                f"Live PID {pid_file} failed ownership validation for {spec.id}: {joined}"
            )
        return process

    def _empty_status(
        self,
        spec: RuntimeSpec,
        status: str,
        *,
        state: Mapping[str, Any] | None = None,
        pid: int | None = None,
        last_error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": spec.id,
            "label": spec.label,
            "port": spec.port,
            "enabled": spec.enabled,
            "status": status,
            "pid": pid,
            "started_at": state.get("started_at") if state else None,
            "uptime_seconds": None,
            "cpu_percent": None,
            "rss_bytes": None,
            "children": {"count": 0, "cpu_percent": 0.0, "rss_bytes": 0},
            "total_cpu_percent": None,
            "total_rss_bytes": None,
            "gpu_memory_bytes": None,
            "gpu_memory_by_pid": {},
            "device": spec.device,
            "loaded_models": [],
            "active_jobs": 0,
            "queued_jobs": 0,
            "last_error": last_error,
            "log_path": str(self._log_path(spec.id)),
        }

    def _safe_cpu_percent(self, process: Any) -> float:
        try:
            key = (int(process.pid), float(process.create_time()))
            with self._cpu_sample_lock:
                sampler = self._cpu_processes.setdefault(key, process)
                return float(sampler.cpu_percent(interval=None))
        except (psutil.Error, AttributeError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_rss(process: Any) -> int:
        try:
            return int(process.memory_info().rss)
        except (psutil.Error, AttributeError, TypeError, ValueError):
            return 0

    def _resource_usage(self, process: Any) -> dict[str, Any]:
        root_cpu = self._safe_cpu_percent(process)
        root_rss = self._safe_rss(process)
        try:
            children = [child for child in process.children(recursive=True) if self._process_is_alive(child)]
        except psutil.Error:
            children = []
        children_cpu = sum(self._safe_cpu_percent(child) for child in children)
        children_rss = sum(self._safe_rss(child) for child in children)
        pids = {int(process.pid), *(int(child.pid) for child in children)}
        gpu_by_pid: dict[int, int] | None = None
        if self.config.manager.collect_gpu_memory:
            try:
                gpu_by_pid = self._gpu_collector.collect(pids)
            except Exception:
                gpu_by_pid = None
        return {
            "cpu_percent": root_cpu,
            "rss_bytes": root_rss,
            "children": {
                "count": len(children),
                "cpu_percent": children_cpu,
                "rss_bytes": children_rss,
            },
            "total_cpu_percent": root_cpu + children_cpu,
            "total_rss_bytes": root_rss + children_rss,
            "gpu_memory_bytes": sum(gpu_by_pid.values()) if gpu_by_pid is not None else None,
            "gpu_memory_by_pid": (
                {str(pid): value for pid, value in sorted(gpu_by_pid.items())}
                if gpu_by_pid is not None
                else {}
            ),
        }

    @staticmethod
    def _loaded_models(payload: Mapping[str, Any]) -> list[str]:
        value = payload.get("loaded_models", payload.get("models", []))
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Mapping):
            return [str(name) for name in value]
        if isinstance(value, (list, tuple, set)):
            return [str(name) for name in value]
        return []

    @staticmethod
    def _job_count(payload: Mapping[str, Any], name: str) -> int:
        try:
            return max(0, int(payload.get(name, 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _telemetry_from_payload(self, spec: RuntimeSpec, payload: Mapping[str, Any]) -> dict[str, Any]:
        remote_state = str(payload.get("status", payload.get("state", ""))).strip().lower()
        loaded_models = self._loaded_models(payload)
        active_jobs = self._job_count(payload, "active_jobs")
        queued_jobs = self._job_count(payload, "queued_jobs")
        last_error_raw = payload.get("last_error")
        last_error = str(last_error_raw) if last_error_raw not in (None, "") else None

        if remote_state in {"error", "failed", "unhealthy"}:
            status = "error"
        elif remote_state in {"starting", "loading", "warming"}:
            status = "starting"
        elif remote_state in {"busy", "running", "processing"} or active_jobs + queued_jobs > 0:
            status = "busy"
        elif remote_state in {"warm-idle", "warm_idle", "warm", "idle"}:
            status = "warm-idle"
        elif payload.get("warm") is True or loaded_models:
            status = "warm-idle"
        else:
            status = "ready-cold"
        return {
            "status": status,
            "device": str(payload.get("device", spec.device)),
            "loaded_models": loaded_models,
            "active_jobs": active_jobs,
            "queued_jobs": queued_jobs,
            "last_error": last_error,
        }

    def _not_ready_telemetry(
        self,
        state: Mapping[str, Any],
        error: str,
    ) -> dict[str, Any]:
        started_at = float(state.get("started_at", self._wall_time()))
        has_been_ready = state.get("ready_at") is not None
        within_grace = self._wall_time() - started_at < self.config.manager.start_grace_seconds
        status = "starting" if not has_been_ready and within_grace else "error"
        return {
            "status": status,
            "loaded_models": [],
            "active_jobs": 0,
            "queued_jobs": 0,
            "last_error": error if status == "error" else None,
        }

    def _probe_health(self, spec: RuntimeSpec, state: Mapping[str, Any]) -> dict[str, Any]:
        if spec.health_endpoint is None:
            return self._not_ready_telemetry(state, "No health endpoint is configured")
        url = f"http://127.0.0.1:{spec.port}{spec.health_endpoint}"
        try:
            response = self._http_get(url, timeout=self.config.manager.request_timeout_seconds)
        except Exception as exc:
            return self._not_ready_telemetry(state, f"Health request failed: {exc}")
        if 200 <= int(response.status_code) < 400:
            return {
                "status": "ready-cold",
                "loaded_models": [],
                "active_jobs": 0,
                "queued_jobs": 0,
                "last_error": None,
            }
        return self._not_ready_telemetry(
            state,
            f"Health endpoint returned HTTP {response.status_code}",
        )

    def _probe_runtime(self, spec: RuntimeSpec, state: Mapping[str, Any]) -> dict[str, Any]:
        if spec.runtime_endpoint is None:
            return self._probe_health(spec, state)
        url = f"http://127.0.0.1:{spec.port}{spec.runtime_endpoint}"
        try:
            response = self._http_get(url, timeout=self.config.manager.request_timeout_seconds)
        except Exception as exc:
            return self._not_ready_telemetry(state, f"Runtime request failed: {exc}")
        status_code = int(response.status_code)
        if status_code in UNSUPPORTED_RUNTIME_ENDPOINT_CODES:
            # Older model servers do not expose warm-cache telemetry. Their
            # configured health endpoint still has to prove the model stack
            # imports successfully once before the process is considered
            # ready. Avoid repeatedly launching their expensive Torch probes.
            if state.get("ready_at") is not None:
                return {
                    "status": "ready-cold",
                    "loaded_models": [],
                    "active_jobs": 0,
                    "queued_jobs": 0,
                    "last_error": None,
                }
            return self._probe_health(spec, state)
        if not 200 <= status_code < 300:
            return self._not_ready_telemetry(
                state,
                f"Runtime endpoint returned HTTP {status_code}",
            )
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            return self._not_ready_telemetry(state, f"Runtime endpoint returned invalid JSON: {exc}")
        if not isinstance(payload, Mapping):
            return self._not_ready_telemetry(state, "Runtime endpoint must return a JSON object")
        return self._telemetry_from_payload(spec, payload)

    def _live_status(
        self,
        spec: RuntimeSpec,
        process: Any,
        state: dict[str, Any],
        *,
        probe: bool,
    ) -> dict[str, Any]:
        telemetry = (
            self._probe_runtime(spec, state)
            if probe
            else {
                "status": "starting",
                "loaded_models": [],
                "active_jobs": 0,
                "queued_jobs": 0,
                "last_error": None,
            }
        )
        ready = telemetry["status"] in {"ready-cold", "warm-idle", "busy"}
        state_changed = False
        if ready and state.get("ready_at") is None:
            state["ready_at"] = self._wall_time()
            state_changed = True
        next_error = telemetry.get("last_error") if telemetry["status"] == "error" else None
        if state.get("last_error") != next_error:
            state["last_error"] = next_error
            state_changed = True
        if state_changed:
            self._write_state(spec.id, state)

        result = self._empty_status(
            spec,
            telemetry["status"],
            state=state,
            pid=int(process.pid),
            last_error=telemetry.get("last_error"),
        )
        try:
            create_time = float(process.create_time())
        except (psutil.Error, TypeError, ValueError):
            create_time = float(state.get("create_time", self._wall_time()))
        result["uptime_seconds"] = max(0.0, self._wall_time() - create_time)
        result.update(self._resource_usage(process))
        result.update({key: value for key, value in telemetry.items() if key != "status"})
        result["status"] = telemetry["status"]
        return result

    def _status_locked(self, spec: RuntimeSpec, *, probe: bool = True) -> dict[str, Any]:
        try:
            state = self._read_state(spec.id)
            process = self._resolve_owned_process(spec, state)
        except UnsafeProcessError as exc:
            state = None
            try:
                state = self._read_state(spec.id)
            except UnsafeProcessError:
                pass
            pid = None
            try:
                pid = self._read_pid(spec.id)
            except UnsafeProcessError:
                pass
            return self._empty_status(spec, "error", state=state, pid=pid, last_error=str(exc))

        if process is not None:
            assert state is not None
            return self._live_status(spec, process, state, probe=probe)

        desired_state = state.get("desired_state") if state else "stopped"
        last_error = state.get("last_error") if state else None
        if desired_state == "running":
            handle = self._handles.get(spec.id)
            exit_code = handle.poll() if handle is not None else None
            if last_error is None:
                last_error = (
                    f"Runtime process exited with code {exit_code}"
                    if exit_code is not None
                    else "Runtime process is no longer running"
                )
            failed_state = dict(state or {})
            failed_state.update({"pid": None, "last_error": last_error})
            self._write_state(spec.id, failed_state)
            self._remove_pid_file(spec.id)
            self._close_log_handle(spec.id)
            return self._empty_status(spec, "error", state=failed_state, last_error=last_error)
        return self._empty_status(spec, "stopped", state=state, last_error=last_error)

    def status(self, runtime_id: str) -> dict[str, Any]:
        spec = self._spec(runtime_id)
        with self._runtime_lock(runtime_id):
            return self._status_locked(spec)

    def list_statuses(self) -> list[dict[str, Any]]:
        runtime_ids = list(self.config.runtimes)
        if not runtime_ids:
            return []
        with ThreadPoolExecutor(max_workers=len(runtime_ids), thread_name_prefix="runtime-status") as executor:
            statuses = dict(zip(runtime_ids, executor.map(self.status, runtime_ids)))
        return [statuses[runtime_id] for runtime_id in runtime_ids]

    def _new_running_state(self, spec: RuntimeSpec, process: Any) -> dict[str, Any]:
        return {
            "state_version": STATE_VERSION,
            "owner_id": self.owner_id,
            "runtime_id": spec.id,
            "pid": int(process.pid),
            "create_time": float(process.create_time()),
            "command_fingerprint": _command_fingerprint(spec),
            "desired_state": "running",
            "started_at": self._wall_time(),
            "ready_at": None,
            "last_error": None,
        }

    def _open_log(self, runtime_id: str) -> Any:
        descriptor = os.open(self._log_path(runtime_id), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
        return os.fdopen(descriptor, "ab", buffering=0)

    def _close_log_handle(self, runtime_id: str) -> None:
        handle = self._log_handles.pop(runtime_id, None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        self._handles.pop(runtime_id, None)

    def _record_start_failure(self, spec: RuntimeSpec, message: str) -> None:
        state = {
            "state_version": STATE_VERSION,
            "owner_id": self.owner_id,
            "runtime_id": spec.id,
            "pid": None,
            "command_fingerprint": _command_fingerprint(spec),
            "desired_state": "running",
            "started_at": self._wall_time(),
            "ready_at": None,
            "last_error": message,
        }
        self._write_state(spec.id, state)
        self._remove_pid_file(spec.id)

    def _start_locked(self, spec: RuntimeSpec) -> dict[str, Any]:
        state = self._read_state(spec.id)
        process = self._resolve_owned_process(spec, state)
        if process is not None:
            assert state is not None
            return self._live_status(spec, process, state, probe=True)
        if not spec.enabled:
            raise RuntimeDisabledError(f"Runtime {spec.id} is disabled")

        log_handle = self._open_log(spec.id)
        spawned: Any | None = None
        try:
            environment = os.environ.copy()
            for name in TRANSIENT_SERVER_ENVIRONMENT:
                environment.pop(name, None)
            environment.update(spec.env)
            environment["HAY_SAY_RUNTIME_ID"] = spec.id
            environment["HAY_SAY_RUNTIME_PORT"] = str(spec.port)
            spawned = self._popen(
                list(spec.command),
                cwd=str(spec.cwd),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
                close_fds=True,
                start_new_session=True,
            )
            process = self._process(int(spawned.pid))
            running_state = self._new_running_state(spec, process)
            self._write_state(spec.id, running_state)
            self._write_pid(spec.id, int(spawned.pid))
        except Exception as exc:
            if spawned is not None:
                try:
                    self._killpg(int(spawned.pid), signal.SIGKILL)
                except OSError:
                    pass
            log_handle.close()
            message = f"Unable to start {spec.id}: {exc}"
            self._record_start_failure(spec, message)
            raise RuntimeOperationError(message) from exc

        self._handles[spec.id] = spawned
        self._log_handles[spec.id] = log_handle
        return self._live_status(spec, process, running_state, probe=False)

    def start(self, runtime_id: str) -> dict[str, Any]:
        spec = self._spec(runtime_id)
        with self._runtime_lock(runtime_id):
            return self._start_locked(spec)

    def _group_members(self, process: Any) -> list[Any]:
        try:
            children = process.children(recursive=True)
        except psutil.Error:
            children = []
        return [process, *children]

    def _member_is_in_group(self, member: Any, pgid: int) -> bool:
        if not self._process_is_alive(member):
            return False
        try:
            return self._getpgid(int(member.pid)) == pgid
        except (OSError, ProcessLookupError):
            return False

    def _group_is_alive(self, pgid: int, members: list[Any]) -> bool:
        return any(self._member_is_in_group(member, pgid) for member in members)

    def _wait_for_group(self, pgid: int, members: list[Any], timeout: float) -> bool:
        deadline = self._monotonic() + timeout
        while self._group_is_alive(pgid, members):
            if self._monotonic() >= deadline:
                return False
            self._sleep(min(0.05, max(0.0, deadline - self._monotonic())))
        return True

    def _signal_validated_group(self, process: Any, sig: int) -> None:
        pid = int(process.pid)
        try:
            pgid = self._getpgid(pid)
        except ProcessLookupError:
            return
        except OSError as exc:
            raise RuntimeOperationError(f"Cannot inspect process group for PID {pid}: {exc}") from exc
        if pgid != pid:
            raise UnsafeProcessError(f"Refusing to signal PID {pid}: it is not its own process-group leader")
        try:
            self._killpg(pgid, sig)
        except ProcessLookupError:
            return
        except OSError as exc:
            raise RuntimeOperationError(f"Unable to signal process group {pgid}: {exc}") from exc

    def _write_stopped_state(self, spec: RuntimeSpec) -> dict[str, Any]:
        stopped_state = {
            "state_version": STATE_VERSION,
            "owner_id": self.owner_id,
            "runtime_id": spec.id,
            "pid": None,
            "command_fingerprint": _command_fingerprint(spec),
            "desired_state": "stopped",
            "started_at": None,
            "ready_at": None,
            "last_error": None,
        }
        self._write_state(spec.id, stopped_state)
        self._remove_pid_file(spec.id)
        self._close_log_handle(spec.id)
        return stopped_state

    def _stop_locked(self, spec: RuntimeSpec) -> dict[str, Any]:
        state = self._read_state(spec.id)
        process = self._resolve_owned_process(spec, state)
        if process is None:
            stopped_state = self._write_stopped_state(spec)
            return self._empty_status(spec, "stopped", state=stopped_state)

        members = self._group_members(process)
        self._signal_validated_group(process, signal.SIGTERM)
        if not self._wait_for_group(int(process.pid), members, self.config.manager.stop_timeout_seconds):
            # The group was validated immediately before TERM. Validate a live
            # member still belongs to it before escalating to KILL.
            leader = next(
                (member for member in members if self._member_is_in_group(member, int(process.pid))),
                None,
            )
            if leader is not None:
                try:
                    self._killpg(int(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    raise RuntimeOperationError(
                        f"Unable to kill process group {process.pid}: {exc}"
                    ) from exc
            if not self._wait_for_group(int(process.pid), members, 2.0):
                raise RuntimeOperationError(f"Runtime {spec.id} did not stop after SIGKILL")
        stopped_state = self._write_stopped_state(spec)
        return self._empty_status(spec, "stopped", state=stopped_state)

    def stop(self, runtime_id: str) -> dict[str, Any]:
        spec = self._spec(runtime_id)
        with self._runtime_lock(runtime_id):
            return self._stop_locked(spec)

    def stop_all(self) -> list[dict[str, Any]]:
        """Stop every managed runtime concurrently during host-service shutdown."""
        runtime_ids = list(self.config.runtimes)
        if not runtime_ids:
            return []
        with ThreadPoolExecutor(max_workers=len(runtime_ids), thread_name_prefix="runtime-stop") as executor:
            statuses = dict(zip(runtime_ids, executor.map(self.stop, runtime_ids)))
        return [statuses[runtime_id] for runtime_id in runtime_ids]

    def restart(self, runtime_id: str) -> dict[str, Any]:
        spec = self._spec(runtime_id)
        with self._runtime_lock(runtime_id):
            self._stop_locked(spec)
            return self._start_locked(spec)

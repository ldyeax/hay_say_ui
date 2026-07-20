"""Persistent model replica subprocesses with cooperative request cancellation."""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass

import hay_say_common as hsc


def positive_environment_int(name, default):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def device_key(gpu_id):
    return "cpu" if gpu_id in (None, "", -1, "-1", "CPU", "cpu") else f"cuda:{int(gpu_id)}"


@dataclass(frozen=True)
class WorkerSpec:
    character: str
    device: str
    cpu_bf16: bool = False


class PersistentModelWorker:
    def __init__(self, spec, *, environment, python_executable, worker_script, cwd, startup_timeout=900):
        self.spec = spec
        self.created_at = time.monotonic()
        self.last_used = self.created_at
        self.request_id = None
        self._write_lock = threading.Lock()
        parent_socket, child_socket = socket.socketpair()
        self._socket = parent_socket
        self._reader = None
        self._writer = None
        command = [
            python_executable,
            worker_script,
            "--control-fd",
            str(child_socket.fileno()),
            "--character",
            spec.character,
        ]
        try:
            self.process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                pass_fds=(child_socket.fileno(),),
            )
        except Exception:
            parent_socket.close()
            raise
        finally:
            child_socket.close()
        self._reader = parent_socket.makefile("r", encoding="utf-8")
        self._writer = parent_socket.makefile("w", encoding="utf-8")
        parent_socket.settimeout(float(startup_timeout))
        try:
            response = self._receive()
        except Exception:
            self.stop()
            raise
        finally:
            parent_socket.settimeout(None)
        if response.get("status") != "ready":
            self.stop()
            raise RuntimeError(f"Model worker failed to start: {response}")
        self.reported_device = response.get("device", spec.device)

    def run(self, job):
        self._send({"action": "generate", **job})
        response = self._receive()
        if response.get("request_id") != job["request_id"]:
            raise RuntimeError(f"Model worker returned a mismatched response: {response}")
        status = response.get("status")
        if status == "cancelled":
            raise hsc.InferenceCancelled(f"Inference request {job['request_id']} was cancelled")
        if status != "completed":
            raise RuntimeError(response.get("error") or f"Model worker failed: {response}")

    def cancel(self, request_id):
        if self.request_id == request_id and self.is_alive:
            self._send({"action": "cancel", "request_id": request_id})
            return True
        return False

    @property
    def is_alive(self):
        return self.process.poll() is None

    def stop(self):
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            try:
                self._send({"action": "stop"})
                process.wait(timeout=5)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        self._close_transport()

    def _send(self, payload):
        with self._write_lock:
            self._writer.write(json.dumps(payload, sort_keys=True) + "\n")
            self._writer.flush()

    def _receive(self):
        line = self._reader.readline()
        if not line:
            return_code = self.process.poll()
            raise RuntimeError(f"Model worker exited before responding (status {return_code})")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("Model worker response must be an object")
        return response

    def _close_transport(self):
        for stream_name in ("_reader", "_writer"):
            stream = getattr(self, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
                setattr(self, stream_name, None)
        sock = getattr(self, "_socket", None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
            self._socket = None


class PersistentModelRuntime:
    def __init__(self, *, worker_factory=PersistentModelWorker, idle_ttl_seconds=None, reaper_interval=30,
                 cpu_workers_environment="HAY_SAY_RVC_CPU_WORKERS", cpu_workers_default=12,
                 gpu_workers_environment="HAY_SAY_RVC_GPU_WORKERS", gpu_workers_default=1,
                 thread_name_prefix="model"):
        self.worker_factory = worker_factory
        self.idle_ttl_seconds = max(
            1800,
            positive_environment_int(
                "HAY_SAY_MODEL_IDLE_TTL_SECONDS", idle_ttl_seconds or 1800
            ),
        )
        self.cpu_workers = positive_environment_int(cpu_workers_environment, cpu_workers_default)
        self.gpu_workers = positive_environment_int(gpu_workers_environment, gpu_workers_default)
        self._condition = threading.Condition(threading.RLock())
        self._workers = []
        self._starting = Counter()
        self._queued = Counter()
        self._cancelled_at = {}
        self._closed = False
        self._reaper_interval = max(1, int(reaper_interval))
        self._reaper = threading.Thread(
            target=self._reap_loop,
            name=f"{thread_name_prefix}-worker-reaper",
            daemon=True,
        )
        self._reaper.start()
        atexit.register(self.close)

    def run(self, job, *, character, gpu_id, environment, python_executable, worker_script, cwd):
        request_id = self._normalize_request_id(job.get("request_id"))
        job = {**job, "request_id": request_id}
        self.raise_if_cancelled(request_id)
        device = device_key(gpu_id)
        spec = WorkerSpec(
            character=character,
            device=device,
            cpu_bf16=device == "cpu" and environment.get("HAY_SAY_CPU_BF16_AUTOCAST") == "1",
        )
        with self._condition:
            self._queued[spec] += 1
        try:
            worker = self._acquire(
                spec,
                request_id,
                environment=environment,
                python_executable=python_executable,
                worker_script=worker_script,
                cwd=cwd,
            )
        finally:
            with self._condition:
                self._queued[spec] -= 1
                if not self._queued[spec]:
                    del self._queued[spec]
                self._condition.notify_all()
        failed = False
        try:
            worker.run(job)
            self.raise_if_cancelled(request_id)
        except Exception:
            failed = not worker.is_alive
            raise
        finally:
            self._release(worker, failed=failed)

    def cancel(self, request_ids):
        normalized = {self._normalize_request_id(value) for value in request_ids if value}
        now = time.monotonic()
        with self._condition:
            self._prune_cancelled_locked(now)
            for request_id in normalized:
                self._cancelled_at[request_id] = now
            workers = [
                (worker, worker.request_id)
                for worker in self._workers
                if worker.request_id in normalized
            ]
            self._condition.notify_all()
        signalled = 0
        failures = 0
        for worker, matched_request_id in workers:
            try:
                signalled += bool(worker.cancel(matched_request_id))
            except Exception as error:
                failures += 1
                pid = getattr(getattr(worker, "process", None), "pid", "unknown")
                print(
                    f"Could not signal cancellation to model worker {pid}: {error}",
                    flush=True,
                )
        return {
            "request_ids": sorted(normalized),
            "active_workers_signalled": signalled,
            "active_workers_signal_failed": failures,
            "runtime_preserved": True,
        }

    def raise_if_cancelled(self, request_id):
        request_id = self._normalize_request_id(request_id)
        with self._condition:
            self._prune_cancelled_locked()
            if request_id in self._cancelled_at:
                raise hsc.InferenceCancelled(f"Inference request {request_id} was cancelled")

    def commit_if_active(self, request_id, callback):
        """Serialize the final cache commit against cancellation tombstones."""
        request_id = self._normalize_request_id(request_id)
        with self._condition:
            self._prune_cancelled_locked()
            if request_id in self._cancelled_at:
                raise hsc.InferenceCancelled(f"Inference request {request_id} was cancelled")
            return callback()

    def state(self):
        with self._condition:
            workers = list(self._workers)
            starting = sum(self._starting.values())
            queued = sum(self._queued.values())
            starting_by_device = Counter()
            queued_by_device = Counter()
            for spec, count in self._starting.items():
                starting_by_device[spec.device] += count
            for spec, count in self._queued.items():
                queued_by_device[spec.device] += count
        now = time.monotonic()
        details = [
            {
                "character": worker.spec.character,
                "device": worker.spec.device,
                "cpu_bf16": worker.spec.cpu_bf16,
                "pid": worker.process.pid,
                "busy": worker.request_id is not None,
                "request_id": worker.request_id,
                "last_used": worker.last_used,
                "warm_until": worker.last_used + self.idle_ttl_seconds,
                "idle_ttl_remaining_seconds": max(
                    0.0, worker.last_used + self.idle_ttl_seconds - now
                ),
            }
            for worker in workers
            if worker.is_alive
        ]
        warm = bool(details)
        active_jobs = sum(worker["busy"] for worker in details)
        active_by_device = Counter(
            worker["device"] for worker in details if worker["busy"]
        )
        loaded_devices = sorted({worker["device"] for worker in details})
        known_devices = sorted(
            set(loaded_devices) | set(starting_by_device) | set(queued_by_device)
        )
        busy = active_jobs > 0 or queued > 0
        if len(known_devices) == 1:
            current_device = known_devices[0]
        elif known_devices:
            current_device = "multiple"
        else:
            current_device = None
        return {
            "status": "busy" if busy else ("warm-idle" if warm else "ready-cold"),
            "device": current_device,
            "multiple": len(known_devices) > 1,
            "busy": busy,
            "warm": warm,
            "workers": len(details),
            "starting_workers": starting,
            "active_jobs": active_jobs,
            "queued_jobs": queued,
            "active_requests": active_jobs,
            "active_jobs_by_device": dict(sorted(active_by_device.items())),
            "queued_jobs_by_device": dict(sorted(queued_by_device.items())),
            "devices": [
                {
                    "device": device,
                    "active_jobs": active_by_device.get(device, 0),
                    "queued_jobs": queued_by_device.get(device, 0),
                    "starting_workers": starting_by_device.get(device, 0),
                    "loaded_models": sum(
                        worker["device"] == device for worker in details
                    ),
                }
                for device in known_devices
            ],
            "loaded_models": [worker["character"] for worker in details],
            "idle_ttl_seconds": self.idle_ttl_seconds,
            "loaded_model_details": details,
        }

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            workers, self._workers = self._workers, []
            self._condition.notify_all()
        for worker in workers:
            worker.stop()

    def _acquire(self, spec, request_id, **worker_options):
        while True:
            expired = []
            spawn = False
            selected = None
            with self._condition:
                if self._closed:
                    raise RuntimeError("Model runtime is closed")
                self.raise_if_cancelled(request_id)
                expired = self._remove_expired_locked()
                if self._has_capacity_locked(spec):
                    for worker in self._workers:
                        if worker.spec == spec and worker.request_id is None and worker.is_alive:
                            worker.request_id = request_id
                            selected = worker
                            break
                    if selected is None:
                        self._starting[spec] += 1
                        spawn = True
                else:
                    self._condition.wait(timeout=1)
            for worker in expired:
                worker.stop()
            if selected is not None:
                return selected
            if not spawn:
                continue
            try:
                worker = self.worker_factory(spec, **worker_options)
            except Exception:
                with self._condition:
                    self._starting[spec] -= 1
                    self._condition.notify_all()
                raise
            with self._condition:
                self._starting[spec] -= 1
                if self._closed:
                    error = RuntimeError("Model runtime closed while a worker was starting")
                    preserve_worker = False
                elif request_id in self._cancelled_at:
                    error = hsc.InferenceCancelled(f"Inference request {request_id} was cancelled")
                    preserve_worker = True
                    worker.request_id = None
                    worker.last_used = time.monotonic()
                    self._workers.append(worker)
                    self._condition.notify_all()
                else:
                    error = None
                    preserve_worker = True
                    worker.request_id = request_id
                    self._workers.append(worker)
            if error is not None:
                if not preserve_worker:
                    worker.stop()
                raise error
            return worker

    def _release(self, worker, *, failed):
        with self._condition:
            worker.request_id = None
            worker.last_used = time.monotonic()
            if failed and worker in self._workers:
                self._workers.remove(worker)
            self._condition.notify_all()
        if failed:
            worker.stop()

    def _has_capacity_locked(self, spec):
        if spec.device == "cpu":
            active = sum(
                worker.request_id is not None
                for worker in self._workers
                if worker.spec.device == "cpu" and worker.is_alive
            )
            active += sum(
                count for key, count in self._starting.items() if key.device == "cpu"
            )
            return active < self.cpu_workers
        active = sum(
            worker.request_id is not None
            for worker in self._workers
            if worker.spec.device == spec.device and worker.is_alive
        )
        active += sum(count for key, count in self._starting.items() if key.device == spec.device)
        return active < self.gpu_workers

    def _remove_expired_locked(self):
        now = time.monotonic()
        expired = [
            worker for worker in self._workers
            if worker.request_id is None and (
                not worker.is_alive or now - worker.last_used >= self.idle_ttl_seconds
            )
        ]
        if expired:
            self._workers = [worker for worker in self._workers if worker not in expired]
            self._condition.notify_all()
        return expired

    def _reap_loop(self):
        while True:
            with self._condition:
                if self._closed:
                    return
                expired = self._remove_expired_locked()
            for worker in expired:
                worker.stop()
            time.sleep(self._reaper_interval)

    def _prune_cancelled_locked(self, now=None):
        now = time.monotonic() if now is None else now
        cutoff = now - max(3600, self.idle_ttl_seconds)
        self._cancelled_at = {
            request_id: cancelled_at
            for request_id, cancelled_at in self._cancelled_at.items()
            if cancelled_at >= cutoff
        }

    @staticmethod
    def _normalize_request_id(request_id):
        value = str(request_id or "").strip()
        return value or f"untracked-{uuid.uuid4().hex}"

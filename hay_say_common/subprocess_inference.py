"""Request-isolated subprocess execution for legacy native model wrappers."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager


class InferenceCancelled(RuntimeError):
    """Raised when a browser generation request cancels a child inference."""


class InferenceProcessRegistry:
    """Track every child belonging to a browser request without stopping its server."""

    def __init__(self, cancellation_ttl_seconds=3600):
        self._lock = threading.RLock()
        self._processes = {}
        self._cancelled_at = {}
        self._cancellation_ttl_seconds = max(60, int(cancellation_ttl_seconds))

    def run(self, command, *, request_id=None, **popen_kwargs):
        request_id = self.normalize_request_id(request_id)
        with self._lock:
            self._prune_cancelled_locked()
            if request_id in self._cancelled_at:
                raise InferenceCancelled(f"Inference request {request_id} was cancelled")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            **popen_kwargs,
        )
        with self._lock:
            self._processes.setdefault(request_id, set()).add(process)
            cancelled = request_id in self._cancelled_at
        if cancelled:
            self._terminate(process)

        try:
            stdout, stderr = process.communicate()
        finally:
            with self._lock:
                processes = self._processes.get(request_id)
                if processes is not None:
                    processes.discard(process)
                    if not processes:
                        self._processes.pop(request_id, None)

        with self._lock:
            cancelled = request_id in self._cancelled_at
        if cancelled:
            raise InferenceCancelled(f"Inference request {request_id} was cancelled")
        if process.returncode:
            details = "\n".join(
                output.strip()
                for output in (stdout, stderr)
                if output and output.strip()
            )
            raise RuntimeError(
                f"Inference exited with status {process.returncode}"
                + (f":\n{details}" if details else " without diagnostic output.")
            )
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    def cancel(self, request_ids):
        normalized = {self.normalize_request_id(value) for value in request_ids if value}
        now = time.monotonic()
        with self._lock:
            self._prune_cancelled_locked(now)
            for request_id in normalized:
                self._cancelled_at[request_id] = now
            processes = {
                process
                for request_id in normalized
                for process in self._processes.get(request_id, ())
            }
        for process in processes:
            self._terminate(process)
        return {
            "request_ids": sorted(normalized),
            "active_processes_signalled": len(processes),
            "runtime_preserved": True,
        }

    def raise_if_cancelled(self, request_id):
        request_id = self.normalize_request_id(request_id)
        with self._lock:
            self._prune_cancelled_locked()
            if request_id in self._cancelled_at:
                raise InferenceCancelled(f"Inference request {request_id} was cancelled")

    def commit_if_active(self, request_id, callback):
        """Serialize the final cache commit against cancellation tombstones."""
        request_id = self.normalize_request_id(request_id)
        with self._lock:
            self._prune_cancelled_locked()
            if request_id in self._cancelled_at:
                raise InferenceCancelled(f"Inference request {request_id} was cancelled")
            return callback()

    def state(self):
        with self._lock:
            active_requests = sum(bool(processes) for processes in self._processes.values())
            active_processes = sum(len(processes) for processes in self._processes.values())
        return {
            "status": "busy" if active_processes else "ready-cold",
            "busy": bool(active_processes),
            "warm": False,
            "active_requests": active_requests,
            "active_processes": active_processes,
        }

    @staticmethod
    def normalize_request_id(request_id):
        value = str(request_id or "").strip()
        return value or f"untracked-{uuid.uuid4().hex}"

    def _prune_cancelled_locked(self, now=None):
        now = time.monotonic() if now is None else now
        cutoff = now - self._cancellation_ttl_seconds
        self._cancelled_at = {
            request_id: cancelled_at
            for request_id, cancelled_at in self._cancelled_at.items()
            if cancelled_at >= cutoff or request_id in self._processes
        }

    @staticmethod
    def _terminate(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


@contextmanager
def request_workspace(root, prefix="request-"):
    """Create and remove one request's files without touching sibling requests."""

    os.makedirs(root, exist_ok=True)
    path = tempfile.mkdtemp(prefix=prefix, dir=root)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

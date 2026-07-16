"""Client helpers for native model services and their loopback supervisor."""

from __future__ import annotations

import os
import fcntl
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests


DEFAULT_MANAGER_URL = "http://127.0.0.1:6588"
RUNNING_STATES = {"ready-cold", "warm-idle", "busy"}
RUNTIME_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class RuntimeManagerError(RuntimeError):
    pass


class RuntimeManagerUnavailable(RuntimeManagerError):
    pass


def native_mode_enabled() -> bool:
    return os.environ.get("HAY_SAY_NATIVE", "").lower() in {"1", "true", "yes", "on"}


def manager_enabled() -> bool:
    return native_mode_enabled() or bool(os.environ.get("HAY_SAY_RUNTIME_MANAGER_URL"))


def admin_enabled() -> bool:
    configured = os.environ.get("HAY_SAY_ENABLE_RUNTIME_ADMIN")
    if configured is None:
        return manager_enabled()
    return configured.lower() in {"1", "true", "yes", "on"}


def manager_url() -> str:
    return os.environ.get("HAY_SAY_RUNTIME_MANAGER_URL", DEFAULT_MANAGER_URL).rstrip("/")


def service_endpoint(runtime_id: str, port: int) -> tuple[str, int]:
    prefix = f"HAY_SAY_{runtime_id.upper()}".replace("-", "_")
    default_host = "127.0.0.1" if native_mode_enabled() else runtime_id + "_server"
    return os.environ.get(prefix + "_HOST", default_host), int(os.environ.get(prefix + "_PORT", port))


@contextmanager
def generation_lock(runtime_id: str):
    """Serialize legacy service requests across native Celery processes."""
    if not RUNTIME_ID.fullmatch(runtime_id):
        raise ValueError(f"Invalid runtime id: {runtime_id}")
    default_root = Path(os.environ.get("HAY_SAY_HOME", Path.home() / "hay_say")) / ".locks"
    lock_root = Path(os.environ.get("HAY_SAY_REQUEST_LOCK_DIR", default_root)).expanduser()
    lock_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = lock_root / f"{runtime_id}.generation.lock"
    with path.open("a+") as lock_file:
        os.chmod(path, 0o600)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _request(method: str, path: str, *, timeout: float = 3.0) -> Any:
    try:
        response = requests.request(method, manager_url() + path, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeManagerUnavailable(f"Runtime manager is unavailable: {exc}") from exc
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not response.ok:
        message = body.get("error") if isinstance(body, dict) else None
        raise RuntimeManagerError(message or f"Runtime manager returned HTTP {response.status_code}")
    return body


def list_runtimes() -> list[dict[str, Any]]:
    body = _request("GET", "/runtimes")
    runtimes = body.get("runtimes") if isinstance(body, dict) else None
    if not isinstance(runtimes, list):
        raise RuntimeManagerError("Runtime manager returned an invalid runtime list")
    return runtimes


def runtime_status(runtime_id: str) -> dict[str, Any]:
    body = _request("GET", f"/runtimes/{runtime_id}")
    if not isinstance(body, dict):
        raise RuntimeManagerError("Runtime manager returned an invalid status")
    return body


def runtime_action(runtime_id: str, action: str) -> dict[str, Any]:
    if action not in {"start", "stop", "restart"}:
        raise ValueError(f"Unsupported runtime action: {action}")
    body = _request("POST", f"/runtimes/{runtime_id}/{action}", timeout=35.0)
    if not isinstance(body, dict):
        raise RuntimeManagerError("Runtime manager returned an invalid action response")
    return body


def service_is_healthy(runtime_id: str, port: int, timeout: float = 1.0) -> bool:
    host, resolved_port = service_endpoint(runtime_id, port)
    try:
        response = requests.get(f"http://{host}:{resolved_port}/gpu-info", timeout=timeout)
        return response.ok
    except requests.RequestException:
        return False


def ensure_runtime_started(runtime_id: str, port: int, timeout: float | None = None) -> None:
    """Start a native runtime on demand, retaining support for manually-run services."""
    if not manager_enabled():
        return
    timeout = timeout or float(os.environ.get("HAY_SAY_RUNTIME_START_TIMEOUT", "45"))
    try:
        status = runtime_status(runtime_id)
        if status.get("status") not in RUNNING_STATES:
            status = runtime_action(runtime_id, "start")
    except RuntimeManagerUnavailable:
        if service_is_healthy(runtime_id, port):
            return
        raise

    deadline = time.monotonic() + timeout
    while status.get("status") not in RUNNING_STATES:
        state = status.get("status")
        if state == "error":
            raise RuntimeManagerError(status.get("last_error") or f"{runtime_id} failed to start")
        if time.monotonic() >= deadline:
            raise RuntimeManagerError(f"Timed out waiting for {runtime_id} to start")
        time.sleep(0.25)
        status = runtime_status(runtime_id)

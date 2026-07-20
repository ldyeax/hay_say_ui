"""Durable generation-job state shared by web and Celery processes."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


CLIENT_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")
RUNTIME_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
QUEUE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")
OPERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")
ACTIVE_STATUSES = frozenset({"queued", "running", "cancelling"})
CANCEL_STATUSES = frozenset({"cancelling", "cancelled"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
OPERATION_STATUSES = frozenset({"pending", "running", "cancelling", "completed", "failed", "cancelled"})
TERMINAL_OPERATION_STATUSES = frozenset({"completed", "failed", "cancelled"})
MAX_REQUEST_DATA_BYTES = 1024 * 1024
MAX_PROGRESS_OPERATIONS = 256


class GenerationJobError(RuntimeError):
    """Base error for invalid or unavailable generation-job state."""


class GenerationJobConflict(GenerationJobError):
    """Raised when a client already owns a different active request."""


class CorruptGenerationJobError(GenerationJobError):
    """Raised when a persisted job cannot be decoded safely."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _validate_client_id(client_id: object) -> str:
    if not isinstance(client_id, str) or CLIENT_ID_PATTERN.fullmatch(client_id) is None:
        raise ValueError("client_id must be exactly 32 hexadecimal characters")
    return client_id.lower()


def _validate_runtime_id(runtime_id: object) -> str:
    if not isinstance(runtime_id, str) or RUNTIME_ID_PATTERN.fullmatch(runtime_id) is None:
        raise ValueError("runtime_id must contain lowercase letters, numbers, and underscores")
    return runtime_id


def _validate_text(name: str, value: object, *, maximum: int = 4096, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    if len(value) > maximum:
        raise ValueError(f"{name} must be at most {maximum} characters")
    if any(ord(character) < 32 and character not in "\t\n" for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _validate_request_id(request_id: object) -> str:
    return _validate_text("request_id", request_id, maximum=256)


def _validate_queue(queue: object) -> str:
    value = _validate_text("queue", queue, maximum=128)
    if QUEUE_PATTERN.fullmatch(value) is None:
        raise ValueError("queue contains unsupported characters")
    return value


def _validate_operation_id(operation_id: object) -> str:
    value = _validate_text("operation_id", operation_id, maximum=128)
    if OPERATION_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("operation_id contains unsupported characters")
    return value


def _validate_progress_values(
    current: object,
    total: object,
) -> tuple[int | float | None, int | float | None]:
    if (current is None) != (total is None):
        raise ValueError("current and total must be supplied together")
    if current is None:
        return None, None
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError("current must be numeric")
    if isinstance(total, bool) or not isinstance(total, (int, float)) or total <= 0:
        raise ValueError("total must be a positive number")
    if current < 0 or current > total:
        raise ValueError("current must be between zero and total")
    return current, total


def _validate_request_data(request_data: object) -> dict[str, Any] | None:
    if request_data is None:
        return None
    if not isinstance(request_data, Mapping):
        raise ValueError("request_data must be an object")
    try:
        encoded = json.dumps(request_data, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("request_data must be JSON serializable") from exc
    if len(encoded.encode("utf-8")) > MAX_REQUEST_DATA_BYTES:
        raise ValueError(f"request_data must be at most {MAX_REQUEST_DATA_BYTES} bytes")
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("request_data must be an object")
    return decoded


def _root() -> Path:
    state_root = Path(
        os.environ.get("HAY_SAY_STATE_DIR", Path.home() / ".local" / "state" / "hay-say")
    ).expanduser()
    root = state_root / "generation-jobs"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    return root


@contextmanager
def _registry_lock(*, exclusive: bool):
    root = _root()
    lock_path = root / ".registry.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield root
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _client_commit_lock(client_id: str):
    root = _root()
    lock_path = root / f".{client_id}.commit.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _state_path(root: Path, client_id: str) -> Path:
    return root / f"{client_id}.json"


def _read_unlocked(root: Path, client_id: str) -> dict[str, Any] | None:
    path = _state_path(root, client_id)
    try:
        with path.open(encoding="utf-8") as state_file:
            state = json.load(state_file)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorruptGenerationJobError(f"Could not read generation job for {client_id}: {exc}") from exc
    if not isinstance(state, dict) or state.get("client_id") != client_id:
        raise CorruptGenerationJobError(f"Invalid generation job for {client_id}")
    return state


def _write_unlocked(root: Path, state: Mapping[str, Any]) -> None:
    client_id = _validate_client_id(state.get("client_id"))
    target = _state_path(root, client_id)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{client_id}-", suffix=".tmp", dir=root)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        state_file = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with state_file:
            json.dump(state, state_file, sort_keys=True, separators=(",", ":"))
            state_file.write("\n")
            state_file.flush()
            os.fsync(state_file.fileno())
        os.replace(temporary, target)
        directory_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _touch(state: dict[str, Any], status: str | None = None, message: str | None = None) -> str:
    timestamp = _now()
    state["updated_at"] = timestamp
    if status is not None:
        state["status"] = status
    if message is not None:
        state["message"] = message
    return timestamp


def _mutate(
    client_id: object,
    request_id: object,
    updater: Callable[[dict[str, Any]], None],
) -> dict[str, Any] | None:
    normalized_client_id = _validate_client_id(client_id)
    normalized_request_id = _validate_request_id(request_id)
    with _registry_lock(exclusive=True) as root:
        state = _read_unlocked(root, normalized_client_id)
        if state is None or state.get("request_id") != normalized_request_id:
            return None
        updater(state)
        _write_unlocked(root, state)
        return state


def active(state: Mapping[str, Any] | str | None) -> bool:
    """Return whether a state can still consume queued or runtime resources."""
    status = state if isinstance(state, str) else state.get("status") if isinstance(state, Mapping) else None
    return status in ACTIVE_STATUSES


def create_queued(
    client_id: object,
    request_id: object,
    runtime_id: object,
    queue: object,
    message: object,
    *,
    request_data: object = None,
) -> dict[str, Any]:
    normalized_client_id = _validate_client_id(client_id)
    normalized_request_id = _validate_request_id(request_id)
    normalized_runtime_id = _validate_runtime_id(runtime_id)
    normalized_queue = _validate_queue(queue)
    normalized_message = _validate_text("message", message, allow_empty=True)
    normalized_request_data = _validate_request_data(request_data)
    with _registry_lock(exclusive=True) as root:
        previous = _read_unlocked(root, normalized_client_id)
        if previous is not None and previous.get("request_id") == normalized_request_id:
            return previous
        if active(previous):
            raise GenerationJobConflict(f"Client {normalized_client_id} already has an active generation")
        timestamp = _now()
        state = {
            "client_id": normalized_client_id,
            "request_id": normalized_request_id,
            "runtime_id": normalized_runtime_id,
            "queue": normalized_queue,
            "request_data": normalized_request_data,
            "status": "queued",
            "message": normalized_message,
            "task_id": None,
            "progress": None,
            "operations": {},
            "cancel_requested_at": None,
            "created_at": timestamp,
            "updated_at": timestamp,
            "started_at": None,
            "finished_at": None,
        }
        _write_unlocked(root, state)
        return state


def claim_running(
    client_id: object,
    request_id: object,
    task_id: object,
    message: object | None = None,
) -> dict[str, Any] | None:
    """Atomically claim a queued job for exactly one Celery task.

    A persisted browser request can be submitted again after a page refresh. Only
    the first task may transition it from queued to running; duplicate or stale
    tasks receive ``None`` and must not perform inference.
    """
    normalized_task_id = _validate_text("task_id", task_id, maximum=256)
    normalized_message = None if message is None else _validate_text("message", message, allow_empty=True)
    normalized_client_id = _validate_client_id(client_id)
    normalized_request_id = _validate_request_id(request_id)
    with _registry_lock(exclusive=True) as root:
        state = _read_unlocked(root, normalized_client_id)
        if (
            state is None
            or state.get("request_id") != normalized_request_id
            or state.get("status") != "queued"
        ):
            return None
        state["task_id"] = normalized_task_id
        state["request_data"] = None
        timestamp = _touch(state, "running", normalized_message)
        state["started_at"] = timestamp
        _write_unlocked(root, state)
        return state


def mark_running(
    client_id: object,
    request_id: object,
    task_id: object,
    message: object | None = None,
) -> dict[str, Any] | None:
    """Compatibility name for :func:`claim_running`."""
    return claim_running(client_id, request_id, task_id, message)


def update_progress(
    client_id: object,
    request_id: object,
    message: object,
    current: int | float | None = None,
    total: int | float | None = None,
) -> dict[str, Any] | None:
    normalized_message = _validate_text("message", message, allow_empty=True)
    current, total = _validate_progress_values(current, total)
    progress = None
    if current is not None and total is not None:
        progress = {"current": current, "total": total}

    def update(state: dict[str, Any]) -> None:
        if state.get("status") in TERMINAL_STATUSES:
            return
        if state.get("status") != "cancelling":
            state["message"] = normalized_message
            state["progress"] = progress
        _touch(state)

    return _mutate(client_id, request_id, update)


def update_operation_progress(
    client_id: object,
    request_id: object,
    operation_id: object,
    label: object,
    current: int | float | None = None,
    total: int | float | None = None,
    *,
    status: object = "running",
    message: object | None = None,
    device: object | None = None,
) -> dict[str, Any] | None:
    """Create or update one bounded progress row for a parallel operation."""
    normalized_operation_id = _validate_operation_id(operation_id)
    normalized_label = _validate_text("label", label, maximum=256)
    normalized_status = _validate_text("status", status, maximum=32)
    if normalized_status not in OPERATION_STATUSES:
        raise ValueError(f"status must be one of {', '.join(sorted(OPERATION_STATUSES))}")
    normalized_message = (
        None if message is None else _validate_text("message", message, maximum=512, allow_empty=True)
    )
    normalized_device = (
        None if device is None else _validate_text("device", device, maximum=64, allow_empty=True)
    )
    current, total = _validate_progress_values(current, total)

    def update(state: dict[str, Any]) -> None:
        if state.get("status") in TERMINAL_STATUSES or state.get("status") == "cancelling":
            return
        operations = state.setdefault("operations", {})
        if not isinstance(operations, dict):
            operations = {}
            state["operations"] = operations
        previous = operations.get(normalized_operation_id)
        if previous is None and len(operations) >= MAX_PROGRESS_OPERATIONS:
            raise ValueError(f"a job may track at most {MAX_PROGRESS_OPERATIONS} operations")
        timestamp = _now()
        started_at = previous.get("started_at") if isinstance(previous, dict) else None
        operation = {
            "id": normalized_operation_id,
            "label": normalized_label,
            "status": normalized_status,
            "message": normalized_message,
            "device": normalized_device,
            "current": current,
            "total": total,
            "started_at": started_at or timestamp,
            "updated_at": timestamp,
            "finished_at": (
                timestamp if normalized_status in TERMINAL_OPERATION_STATUSES else None
            ),
        }
        operations[normalized_operation_id] = operation
        state["updated_at"] = timestamp

    return _mutate(client_id, request_id, update)


def commit_if_active(
    client_id: object,
    request_id: object,
    callback: Callable[[], Any],
) -> tuple[bool, Any]:
    """Run one cache commit atomically before cancellation, or skip it."""
    if not callable(callback):
        raise TypeError("callback must be callable")
    normalized_client_id = _validate_client_id(client_id)
    normalized_request_id = _validate_request_id(request_id)
    with _client_commit_lock(normalized_client_id):
        with _registry_lock(exclusive=False) as root:
            state = _read_unlocked(root, normalized_client_id)
            if (
                state is None
                or state.get("request_id") != normalized_request_id
                or state.get("status") in CANCEL_STATUSES
                or state.get("status") in TERMINAL_STATUSES
            ):
                return False, None
        return True, callback()


def _finish_operations(state: dict[str, Any], status: str, timestamp: str) -> None:
    operations = state.get("operations")
    if not isinstance(operations, dict):
        return
    for operation in operations.values():
        if not isinstance(operation, dict) or operation.get("status") in TERMINAL_OPERATION_STATUSES:
            continue
        operation["status"] = status
        if status == "completed" and operation.get("total") is not None:
            operation["current"] = operation["total"]
        operation["updated_at"] = timestamp
        operation["finished_at"] = timestamp


def request_cancel(client_id: object, request_id: object) -> dict[str, Any] | None:
    """Request cancellation for one browser-owned job without affecting peers."""
    normalized_client_id = _validate_client_id(client_id)
    normalized_request_id = _validate_request_id(request_id)
    with _client_commit_lock(normalized_client_id):
        timestamp = _now()
        with _registry_lock(exclusive=True) as root:
            state = _read_unlocked(root, normalized_client_id)
            if (
                state is None
                or state.get("request_id") != normalized_request_id
                or state.get("status") in TERMINAL_STATUSES
            ):
                return None
            state["status"] = "cancelling"
            state["message"] = "Cancelling generation..."
            state["cancel_requested_at"] = state.get("cancel_requested_at") or timestamp
            state["updated_at"] = timestamp
            operations = state.get("operations")
            if isinstance(operations, dict):
                for operation in operations.values():
                    if (
                        isinstance(operation, dict)
                        and operation.get("status") not in TERMINAL_OPERATION_STATUSES
                    ):
                        operation["status"] = "cancelling"
                        operation["updated_at"] = timestamp
            _write_unlocked(root, state)
            return state


def mark_completed(
    client_id: object,
    request_id: object,
    message: object = "Generation complete.",
) -> dict[str, Any] | None:
    normalized_message = _validate_text("message", message, allow_empty=True)

    def update(state: dict[str, Any]) -> None:
        if state.get("status") in CANCEL_STATUSES or state.get("status") in TERMINAL_STATUSES:
            return
        timestamp = _touch(state, "completed", normalized_message)
        state["progress"] = None
        state["finished_at"] = timestamp
        _finish_operations(state, "completed", timestamp)

    return _mutate(client_id, request_id, update)


def mark_failed(client_id: object, request_id: object, message: object) -> dict[str, Any] | None:
    normalized_message = _validate_text("message", message, allow_empty=True)

    def update(state: dict[str, Any]) -> None:
        if state.get("status") in CANCEL_STATUSES or state.get("status") in TERMINAL_STATUSES:
            return
        timestamp = _touch(state, "failed", normalized_message)
        state["progress"] = None
        state["finished_at"] = timestamp
        _finish_operations(state, "failed", timestamp)

    return _mutate(client_id, request_id, update)


def mark_cancelled(
    client_id: object,
    request_id: object,
    message: object = "Generation cancelled.",
) -> dict[str, Any] | None:
    """Finish one cooperatively cancelled job while leaving its runtime warm."""
    normalized_message = _validate_text("message", message, allow_empty=True)

    def update(state: dict[str, Any]) -> None:
        if state.get("status") == "completed" or state.get("status") == "failed":
            return
        timestamp = _touch(state, "cancelled", normalized_message)
        state["cancel_requested_at"] = state.get("cancel_requested_at") or timestamp
        state["progress"] = None
        state["finished_at"] = timestamp
        _finish_operations(state, "cancelled", timestamp)

    return _mutate(client_id, request_id, update)


def get(client_id: object) -> dict[str, Any] | None:
    normalized_client_id = _validate_client_id(client_id)
    with _registry_lock(exclusive=False) as root:
        return _read_unlocked(root, normalized_client_id)


def is_cancel_requested(client_id: object, request_id: object) -> bool:
    normalized_request_id = _validate_request_id(request_id)
    state = get(client_id)
    return bool(
        state is not None
        and state.get("request_id") == normalized_request_id
        and (state.get("status") in CANCEL_STATUSES or state.get("cancel_requested_at") is not None)
    )

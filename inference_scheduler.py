"""Process-shared admission control for CPU and CUDA inference."""

from __future__ import annotations

import fcntl
import hashlib
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from gpu_selection import configured_gpu_ids, detected_gpu_status


AUTO_DEVICE = "auto"
CPU_DEVICE = ""


class DeviceUnavailableError(RuntimeError):
    pass


class DeviceSlotTimeout(RuntimeError):
    pass


@dataclass
class _SlotLease:
    device: int | str
    lock_file: object

    def release(self):
        if self.lock_file is None:
            return
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
        self.lock_file.close()
        self.lock_file = None


@dataclass
class DeviceReservation:
    """An idempotent device reservation that can be released by its worker."""

    device: int | str
    leases: tuple
    _released: bool = field(default=False, init=False, repr=False)
    _release_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def release(self):
        with self._release_lock:
            if self._released:
                return
            self._released = True
            leases = self.leases
        first_error = None
        for lease in reversed(leases):
            try:
                lease.release()
            except Exception as exc:  # Release every capacity/lane lock before surfacing an error.
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self):
        return self.device

    def __exit__(self, _exception_type, _exception, _traceback):
        self.release()


def _positive_int(name, default):
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name, default):
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _lock_root():
    default_root = Path(os.environ.get("HAY_SAY_HOME", Path.home() / "hay_say")) / ".locks" / "devices"
    root = Path(os.environ.get("HAY_SAY_DEVICE_LOCK_DIR", default_root)).expanduser()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root


def _slot_paths(kind, identifier, count):
    prefix = kind if identifier is None else f"{kind}-{identifier}"
    return [_lock_root() / f"{prefix}-{slot}.lock" for slot in range(count)]


def _lane_paths(serial_device_key, device):
    digest = hashlib.sha256(str(serial_device_key).encode("utf-8", errors="surrogateescape")).hexdigest()[:24]
    device_name = "cpu" if device == CPU_DEVICE else f"gpu-{device}"
    return _slot_paths(f"lane-{digest}-{device_name}", None, 1)


def _try_paths(paths, device):
    if not paths:
        return None
    offset = (os.getpid() + threading.get_ident()) % len(paths)
    for index in range(len(paths)):
        path = paths[(offset + index) % len(paths)]
        lock_file = path.open("a+")
        os.chmod(path, 0o600)
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            continue
        return _SlotLease(device=device, lock_file=lock_file)
    return None


def _wait_for_paths(paths, device, cancel_check=None):
    timeout = float(os.environ.get("HAY_SAY_DEVICE_SLOT_TIMEOUT", "900"))
    if timeout <= 0:
        raise ValueError("HAY_SAY_DEVICE_SLOT_TIMEOUT must be greater than zero")
    deadline = time.monotonic() + timeout
    while True:
        if cancel_check is not None:
            cancel_check()
        lease = _try_paths(paths, device)
        if lease is not None:
            return lease
        if time.monotonic() >= deadline:
            raise DeviceSlotTimeout(f"Timed out waiting for an inference slot on {device_label(device)}")
        time.sleep(0.05)


def _capacity_lease(paths, device, blocking, serial_device_key=None, cancel_check=None):
    lane = None
    if serial_device_key is not None:
        lane_paths = _lane_paths(serial_device_key, device)
        lane = _wait_for_paths(lane_paths, device, cancel_check) if blocking else _try_paths(lane_paths, device)
        if lane is None:
            return None
    try:
        capacity = _wait_for_paths(paths, device, cancel_check) if blocking else _try_paths(paths, device)
    except Exception:
        if lane is not None:
            lane.release()
        raise
    if capacity is None:
        if lane is not None:
            lane.release()
        return None
    leases = (capacity,) if lane is None else (lane, capacity)
    return DeviceReservation(device=device, leases=leases)


def _cpu_lease(blocking=True, serial_device_key=None, cancel_check=None):
    paths = _slot_paths("cpu", None, _positive_int("HAY_SAY_CPU_INFERENCE_SLOTS", 1))
    return _capacity_lease(paths, CPU_DEVICE, blocking, serial_device_key, cancel_check)


def _gpu_paths(identifier):
    count = _positive_int("HAY_SAY_GPU_INFERENCE_SLOTS", 1)
    return _slot_paths("gpu", int(identifier), count)


def _auto_gpu_candidates():
    statuses = detected_gpu_status()
    configured = set(configured_gpu_ids())
    minimum_free = _nonnegative_int("HAY_SAY_AUTO_GPU_MIN_FREE_MIB", 4096)
    maximum_utilization = _nonnegative_int("HAY_SAY_AUTO_GPU_MAX_UTILIZATION", 95)
    if maximum_utilization > 100:
        raise ValueError("HAY_SAY_AUTO_GPU_MAX_UTILIZATION must be between 0 and 100")
    eligible = [
        (identifier, status)
        for identifier, status in statuses.items()
        if identifier in configured
        and status["free_memory_mib"] >= minimum_free
        and status["utilization_percent"] <= maximum_utilization
    ]
    eligible.sort(key=lambda item: (item[1]["utilization_percent"], -item[1]["free_memory_mib"], item[0]))
    return [identifier for identifier, _ in eligible]


def _gpu_lease(identifier, blocking, serial_device_key=None):
    return _capacity_lease(_gpu_paths(identifier), int(identifier), blocking, serial_device_key)


def _try_auto_gpu(serial_device_key=None):
    for identifier in _auto_gpu_candidates():
        lease = _gpu_lease(identifier, False, serial_device_key)
        if lease is not None:
            return lease
    return None


def _explicit_gpu_lease(identifier, serial_device_key=None, cancel_check=None):
    identifier = int(identifier)
    if identifier not in detected_gpu_status():
        raise DeviceUnavailableError(f"GPU #{identifier} is not currently available")
    return _capacity_lease(
        _gpu_paths(identifier), int(identifier), True, serial_device_key, cancel_check
    )


def is_auto_device(device):
    return isinstance(device, str) and device.strip().lower() == AUTO_DEVICE


def device_label(device):
    return "CPU" if device == CPU_DEVICE else f"GPU #{device}"


@contextmanager
def inference_device(requested_device, *, allow_gpu=True, serial_device_key=None, cancel_check=None):
    """Resolve a strict/Auto request and hold its host-wide capacity slot."""
    if is_auto_device(requested_device):
        lease = _try_auto_gpu(serial_device_key) if allow_gpu else None
        if lease is None:
            lease = _cpu_lease(serial_device_key=serial_device_key, cancel_check=cancel_check)
    elif requested_device in (None, CPU_DEVICE, -1, "-1", "cpu", "CPU"):
        lease = _cpu_lease(serial_device_key=serial_device_key, cancel_check=cancel_check)
    else:
        try:
            identifier = int(requested_device)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported inference device: {requested_device!r}") from exc
        if identifier < 0:
            lease = _cpu_lease(serial_device_key=serial_device_key, cancel_check=cancel_check)
        else:
            lease = _explicit_gpu_lease(identifier, serial_device_key, cancel_check)
    try:
        yield lease.device
    finally:
        lease.release()


@contextmanager
def mixed_inference_reservations(*, allow_gpu=True, serial_device_key=None, cancel_check=None):
    """Reserve available CPU/GPU capacity and let each worker release its own slot."""
    gpu_lease = _try_auto_gpu(serial_device_key) if allow_gpu else None
    try:
        cpu_lease = _cpu_lease(blocking=False, serial_device_key=serial_device_key)
        if gpu_lease is None and cpu_lease is None:
            cpu_lease = _cpu_lease(
                serial_device_key=serial_device_key, cancel_check=cancel_check
            )
    except Exception:
        if gpu_lease is not None:
            gpu_lease.release()
        raise
    try:
        yield cpu_lease, gpu_lease
    finally:
        if gpu_lease is not None:
            gpu_lease.release()
        if cpu_lease is not None:
            cpu_lease.release()

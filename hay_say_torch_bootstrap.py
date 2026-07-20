"""Configure PyTorch thread pools before model inference starts."""

import contextlib
import os
import threading


_TRUE_VALUES = {"1", "true", "yes", "on"}
_CONFIGURE_LOCK = threading.Lock()
_AUTOCAST_LOCK = threading.Lock()
_CONFIGURED_TORCH = None
_PROCESS_CPU_AUTOCAST = None


def cpu_supports_amx_bf16(cpuinfo_path="/proc/cpuinfo"):
    """Return whether every reported CPU advertises AMX tile and BF16."""
    try:
        with open(cpuinfo_path, "r", encoding="utf-8", errors="replace") as cpuinfo:
            contents = cpuinfo.read()
    except OSError:
        return False
    saw_flags = False
    for block in contents.split("\n\n"):
        fields = {}
        for line in block.splitlines():
            name, separator, value = line.partition(":")
            if separator:
                fields[name.strip().lower()] = value.strip()
        flags = set(fields.get("flags", "").split())
        if not flags:
            continue
        saw_flags = True
        if not {"amx_bf16", "amx_tile"}.issubset(flags):
            return False
    return saw_flags


def _positive_environment_int(name, default):
    raw = os.environ.get(name, str(default))
    return _positive_int(name, raw)


def _positive_int(name, raw):
    if isinstance(raw, bool):
        raise RuntimeError(f"{name} must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def configure_torch_threads(force=False, *, intraop_threads=None, interop_threads=None):
    """Apply independently configurable intra-op and inter-op thread limits."""
    global _CONFIGURED_TORCH
    explicit_override = intraop_threads is not None or interop_threads is not None
    if _CONFIGURED_TORCH is not None and not explicit_override:
        return _CONFIGURED_TORCH
    enabled = os.environ.get("HAY_SAY_CONFIGURE_TORCH_THREADS", "").strip().lower() in _TRUE_VALUES
    if not force and not enabled:
        return None

    with _CONFIGURE_LOCK:
        if _CONFIGURED_TORCH is not None and not explicit_override:
            return _CONFIGURED_TORCH
        already_configured = _CONFIGURED_TORCH is not None
        if intraop_threads is not None:
            intraop_threads = _positive_int("intraop_threads", intraop_threads)
        elif not already_configured:
            intraop_threads = _positive_environment_int("HAY_SAY_MODEL_CPU_THREADS", 4)
        if interop_threads is not None:
            interop_threads = _positive_int("interop_threads", interop_threads)
        elif not already_configured:
            interop_threads = _positive_environment_int("HAY_SAY_MODEL_CPU_INTEROP_THREADS", 1)

        if already_configured:
            torch = _CONFIGURED_TORCH
        else:
            import torch
        if intraop_threads is not None:
            torch.set_num_threads(intraop_threads)
        if interop_threads is not None:
            try:
                torch.set_num_interop_threads(interop_threads)
            except RuntimeError:
                if torch.get_num_interop_threads() != interop_threads:
                    raise
        _CONFIGURED_TORCH = torch
        return torch


def cpu_bf16_autocast(enabled=True):
    """Return a CPU BF16 autocast context compatible with installed Torch versions."""
    if not enabled or not cpu_supports_amx_bf16():
        return contextlib.nullcontext()
    torch = configure_torch_threads(force=True)
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16)


def configure_process_cpu_bf16_autocast():
    """Keep autocast active for a short-lived inference subprocess."""
    global _PROCESS_CPU_AUTOCAST
    if _PROCESS_CPU_AUTOCAST is not None:
        return _PROCESS_CPU_AUTOCAST
    enabled = (
        os.environ.get("HAY_SAY_CPU_BF16_AUTOCAST", "").strip().lower() in _TRUE_VALUES
        and cpu_supports_amx_bf16()
    )
    if not enabled:
        return None
    with _AUTOCAST_LOCK:
        if _PROCESS_CPU_AUTOCAST is None:
            context = cpu_bf16_autocast(True)
            context.__enter__()
            _PROCESS_CPU_AUTOCAST = context
    return _PROCESS_CPU_AUTOCAST


# Model subprocesses opt in through hay_say_common.select_hardware().
configure_torch_threads()
configure_process_cpu_bf16_autocast()

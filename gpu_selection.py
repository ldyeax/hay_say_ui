"""CUDA discovery and deterministic assignment for generation workers."""

import os
import shutil
import subprocess


def configured_gpu_ids(value=None):
    value = os.environ.get("HAY_SAY_GPU_IDS", "0") if value is None else value
    try:
        identifiers = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("HAY_SAY_GPU_IDS must be a comma-separated list of non-negative integers") from exc
    if not identifiers or any(identifier < 0 for identifier in identifiers):
        raise ValueError("HAY_SAY_GPU_IDS must contain at least one non-negative GPU id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("HAY_SAY_GPU_IDS cannot contain duplicates")
    return identifiers


def gpu_id_for_worker(process_index, value=None):
    identifiers = configured_gpu_ids(value)
    # Billiard prefork indices are one-based. Treat a missing/zero index as the
    # first slot too, which also makes direct callback tests deterministic.
    slot = max(0, int(process_index or 1) - 1)
    return identifiers[slot % len(identifiers)]


def detected_gpu_ids():
    """Return live host GPU indices without importing a model runtime's Torch build."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ()
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return tuple(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return ()


def detected_gpu_status():
    """Return live free-memory and utilization data keyed by physical GPU id."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        statuses = {}
        for line in result.stdout.splitlines():
            identifier, free_memory, utilization = (item.strip() for item in line.split(","))
            statuses[int(identifier)] = {
                "free_memory_mib": int(free_memory),
                "utilization_percent": int(utilization),
            }
        return statuses
    except (OSError, subprocess.SubprocessError, TypeError, ValueError):
        return {}


def configured_gpu_available(value=None):
    return bool(set(configured_gpu_ids(value)).intersection(detected_gpu_ids()))

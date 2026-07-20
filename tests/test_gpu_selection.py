from types import SimpleNamespace

import pytest

import gpu_selection
from gpu_selection import configured_gpu_available, configured_gpu_ids, gpu_id_for_worker


def test_single_worker_uses_gpu_zero():
    assert gpu_id_for_worker(1, "0") == 0


def test_worker_indices_cycle_through_configured_gpus():
    assert [gpu_id_for_worker(index, "2,4") for index in (1, 2, 3)] == [2, 4, 2]


@pytest.mark.parametrize("value", ["", "-1", "0,0", "gpu0"])
def test_invalid_gpu_configuration_fails_early(value):
    with pytest.raises(ValueError):
        configured_gpu_ids(value)


def test_cold_runtime_can_use_a_configured_host_gpu(monkeypatch):
    monkeypatch.setattr(gpu_selection.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gpu_selection.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(stdout="0\n2\n"),
    )

    assert configured_gpu_available("2")
    assert not configured_gpu_available("1")


def test_live_gpu_status_includes_free_memory_and_utilization(monkeypatch):
    monkeypatch.setattr(gpu_selection.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gpu_selection.subprocess,
        "run",
        lambda *_, **__: SimpleNamespace(stdout="0, 22000, 7\n2, 1000, 98\n"),
    )

    assert gpu_selection.detected_gpu_status() == {
        0: {"free_memory_mib": 22000, "utilization_percent": 7},
        2: {"free_memory_mib": 1000, "utilization_percent": 98},
    }

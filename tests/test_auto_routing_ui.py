from types import SimpleNamespace

from celery_generate_cpu import requested_device, select_hardware


def test_hardware_selection_defaults_to_auto():
    assert select_hardware(None) == (["Auto", "CPU"], "Auto")


def test_hardware_selection_preserves_explicit_gpu_when_supported():
    tab = SimpleNamespace(hardware_options=["Auto", "GPU", "CPU"])
    assert select_hardware("GPU", tab) == (["Auto", "GPU", "CPU"], "GPU")


def test_hardware_selection_falls_back_to_auto_when_gpu_disappears():
    tab = SimpleNamespace(hardware_options=["Auto", "CPU"])
    assert select_hardware("GPU", tab) == (["Auto", "CPU"], "Auto")


def test_cpu_dispatcher_maps_auto_and_cpu_to_internal_device_values():
    assert requested_device(None) == "auto"
    assert requested_device("Auto") == "auto"
    assert requested_device("CPU") == ""

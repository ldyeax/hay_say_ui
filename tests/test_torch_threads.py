from types import SimpleNamespace
import builtins
import os
from pathlib import Path
import subprocess
import sys

import pytest

import hay_say_torch_bootstrap
from hay_say_common import server_utility
from hay_say_common.server_utility import (
    model_cpu_bf16_enabled,
    model_cpu_bf16_policy,
    runtime_state_with_cpu_bf16,
    select_hardware,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def test_select_hardware_enables_torch_bootstrap_only_in_inference_child():
    assert select_hardware(0)["HAY_SAY_CONFIGURE_TORCH_THREADS"] == "1"
    assert select_hardware(0)["CUDA_VISIBLE_DEVICES"] == "0"
    assert select_hardware("")["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("model_id", ("so_vits_svc_4", "styletts_2"))
def test_cpu_bf16_defaults_on_only_for_the_beneficial_models(model_id):
    assert model_cpu_bf16_enabled(
        model_id,
        environ={},
        cpu_flags={"amx_tile", "amx_bf16"},
    )


@pytest.mark.parametrize(
    "model_id",
    ("controllable_talknet", "so_vits_svc_3", "so_vits_svc_5", "rvc", "gpt_so_vits"),
)
def test_cpu_bf16_defaults_off_for_other_models(model_id):
    assert not model_cpu_bf16_enabled(
        model_id,
        environ={},
        cpu_flags={"amx_tile", "amx_bf16"},
    )


def test_cpu_bf16_policy_honors_per_model_override_and_requires_amx():
    variable = "HAY_SAY_SVC3_CPU_BF16_AUTOCAST"
    assert model_cpu_bf16_enabled(
        "so_vits_svc_3",
        environ={variable: "yes"},
        cpu_flags={"amx_tile", "amx_bf16"},
    )
    assert not model_cpu_bf16_enabled(
        "so_vits_svc_3",
        environ={variable: "1"},
        cpu_flags={"avx512_bf16"},
    )
    assert not model_cpu_bf16_enabled(
        "so_vits_svc_4",
        environ={"HAY_SAY_SVC4_CPU_BF16_AUTOCAST": "off"},
        cpu_flags={"amx_tile", "amx_bf16"},
    )


def test_cpu_bf16_policy_exposes_configuration_capability_and_effective_state():
    policy = model_cpu_bf16_policy(
        "styletts_2",
        environ={"HAY_SAY_STYLETTS_CPU_BF16_AUTOCAST": "on"},
        cpu_flags={"amx_tile"},
    )

    assert policy == {
        "environment_variable": "HAY_SAY_STYLETTS_CPU_BF16_AUTOCAST",
        "default": True,
        "requested": True,
        "source": "environment",
        "amx_available": False,
        "effective": False,
    }


def test_runtime_state_cpu_bf16_telemetry_does_not_mutate_runtime_state():
    state = {"status": "ready-cold"}

    snapshot = runtime_state_with_cpu_bf16(state, "so_vits_svc_3")

    assert snapshot["status"] == "ready-cold"
    assert snapshot["cpu_bf16"]["environment_variable"] == "HAY_SAY_SVC3_CPU_BF16_AUTOCAST"
    assert "cpu_bf16" not in state


def test_cpu_bf16_policy_rejects_invalid_backend_value():
    with pytest.raises(ValueError, match="HAY_SAY_SVC4_CPU_BF16_AUTOCAST"):
        model_cpu_bf16_enabled(
            "so_vits_svc_4",
            environ={"HAY_SAY_SVC4_CPU_BF16_AUTOCAST": "sometimes"},
            cpu_flags={"amx_tile", "amx_bf16"},
        )


def test_select_hardware_applies_policy_only_to_cpu(monkeypatch):
    monkeypatch.setattr(server_utility, "model_cpu_bf16_enabled", lambda model_id: True)

    assert select_hardware("", "styletts_2")["HAY_SAY_CPU_BF16_AUTOCAST"] == "1"
    assert select_hardware(0, "styletts_2")["HAY_SAY_CPU_BF16_AUTOCAST"] == "0"
    assert select_hardware("")["HAY_SAY_CPU_BF16_AUTOCAST"] == "0"


def test_amx_detection_requires_tile_and_bf16_flags(tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor: 0\nflags: avx512f amx_tile amx_bf16\n", encoding="utf-8")
    assert hay_say_torch_bootstrap.cpu_supports_amx_bf16(cpuinfo)

    cpuinfo.write_text("processor: 0\nflags: avx512f amx_bf16\n", encoding="utf-8")
    assert not hay_say_torch_bootstrap.cpu_supports_amx_bf16(cpuinfo)


def test_amx_detection_requires_support_on_every_reported_processor(tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor: 0\nflags: avx512f amx_tile amx_bf16\n\n"
        "processor: 1\nflags: avx512f amx_tile\n",
        encoding="utf-8",
    )

    assert not hay_say_torch_bootstrap.cpu_supports_amx_bf16(cpuinfo)


def test_autocast_is_a_noop_when_amx_is_unavailable(monkeypatch):
    monkeypatch.setattr(hay_say_torch_bootstrap, "cpu_supports_amx_bf16", lambda: False)
    monkeypatch.setattr(
        hay_say_torch_bootstrap,
        "configure_torch_threads",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("torch should stay cold")),
    )

    with hay_say_torch_bootstrap.cpu_bf16_autocast(True):
        pass


def test_torch_bootstrap_does_not_import_torch_without_inference_flag(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("Torch should stay cold outside an inference child")
        return original_import(name, *args, **kwargs)

    monkeypatch.delenv("HAY_SAY_CONFIGURE_TORCH_THREADS", raising=False)
    monkeypatch.setattr(hay_say_torch_bootstrap, "_CONFIGURED_TORCH", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert hay_say_torch_bootstrap.configure_torch_threads() is None


def test_torch_bootstrap_sets_independent_thread_limits(monkeypatch):
    calls = []
    fake_torch = SimpleNamespace(
        set_num_threads=lambda value: calls.append(("intra", value)),
        set_num_interop_threads=lambda value: calls.append(("interop", value)),
        get_num_interop_threads=lambda: 2,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("HAY_SAY_CONFIGURE_TORCH_THREADS", "1")
    monkeypatch.setenv("HAY_SAY_MODEL_CPU_THREADS", "12")
    monkeypatch.setenv("HAY_SAY_MODEL_CPU_INTEROP_THREADS", "2")
    monkeypatch.setattr(hay_say_torch_bootstrap, "_CONFIGURED_TORCH", None)

    configured = hay_say_torch_bootstrap.configure_torch_threads()
    repeated = hay_say_torch_bootstrap.configure_torch_threads()

    assert configured is fake_torch
    assert repeated is fake_torch
    assert calls == [("intra", 12), ("interop", 2)]


def test_torch_bootstrap_reconfigures_only_intraop_after_early_initialization(monkeypatch):
    calls = []
    fake_torch = SimpleNamespace(
        set_num_threads=lambda value: calls.append(("intra", value)),
        set_num_interop_threads=lambda value: calls.append(("interop", value)),
        get_num_interop_threads=lambda: 1,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("HAY_SAY_CONFIGURE_TORCH_THREADS", "1")
    monkeypatch.setenv("HAY_SAY_MODEL_CPU_THREADS", "12")
    monkeypatch.setenv("HAY_SAY_MODEL_CPU_INTEROP_THREADS", "1")
    monkeypatch.setattr(hay_say_torch_bootstrap, "_CONFIGURED_TORCH", None)

    hay_say_torch_bootstrap.configure_torch_threads()
    configured = hay_say_torch_bootstrap.configure_torch_threads(
        force=True,
        intraop_threads=1,
    )

    assert configured is fake_torch
    assert calls == [("intra", 12), ("interop", 1), ("intra", 1)]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("HAY_SAY_MODEL_CPU_THREADS", "0"),
        ("HAY_SAY_MODEL_CPU_INTEROP_THREADS", "many"),
    ],
)
def test_torch_bootstrap_rejects_invalid_thread_limits(monkeypatch, name, value):
    monkeypatch.setenv("HAY_SAY_CONFIGURE_TORCH_THREADS", "1")
    monkeypatch.setenv("HAY_SAY_MODEL_CPU_THREADS", "4")
    monkeypatch.setenv("HAY_SAY_MODEL_CPU_INTEROP_THREADS", "1")
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(hay_say_torch_bootstrap, "_CONFIGURED_TORCH", None)

    with pytest.raises(RuntimeError, match=name):
        hay_say_torch_bootstrap.configure_torch_threads()


@pytest.mark.parametrize("value", (0, "many", True))
def test_torch_bootstrap_rejects_invalid_explicit_intraop_limit(monkeypatch, value):
    monkeypatch.setenv("HAY_SAY_CONFIGURE_TORCH_THREADS", "1")
    monkeypatch.setattr(hay_say_torch_bootstrap, "_CONFIGURED_TORCH", SimpleNamespace())

    with pytest.raises(RuntimeError, match="intraop_threads"):
        hay_say_torch_bootstrap.configure_torch_threads(
            force=True,
            intraop_threads=value,
        )


def test_pth_bootstrap_configures_a_fresh_interpreter_and_stays_cold_without_flag(tmp_path):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    (site_packages / "hay_say_torch_bootstrap.py").write_text(
        (REPOSITORY / "hay_say_torch_bootstrap.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (site_packages / "hay_say_torch_bootstrap.pth").write_text(
        "import hay_say_torch_bootstrap\n",
        encoding="utf-8",
    )
    (site_packages / "torch.py").write_text(
        "intra = 60\n"
        "interop = 60\n"
        "def set_num_threads(value):\n    global intra; intra = value\n"
        "def set_num_interop_threads(value):\n    global interop; interop = value\n"
        "def get_num_threads(): return intra\n"
        "def get_num_interop_threads(): return interop\n",
        encoding="utf-8",
    )
    code = (
        "import site,sys; "
        f"site.addsitedir({str(site_packages)!r}); "
        "print('cold' if 'torch' not in sys.modules else "
        "f'{sys.modules[\"torch\"].get_num_threads()}/' "
        "f'{sys.modules[\"torch\"].get_num_interop_threads()}')"
    )

    cold = subprocess.run([sys.executable, "-S", "-c", code], check=True, capture_output=True, text=True)
    environment = os.environ.copy()
    environment.update(
        {
            "HAY_SAY_CONFIGURE_TORCH_THREADS": "1",
            "HAY_SAY_MODEL_CPU_THREADS": "7",
            "HAY_SAY_MODEL_CPU_INTEROP_THREADS": "2",
        }
    )
    configured = subprocess.run(
        [sys.executable, "-S", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert cold.stdout.strip() == "cold"
    assert configured.stdout.strip() == "7/2"

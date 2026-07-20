"""Helpers shared by native HTTP wrappers around model runtimes."""

import json
import os
import subprocess
import traceback

from flask import has_request_context, request
from hay_say_torch_bootstrap import cpu_supports_amx_bf16


MODEL_CPU_BF16_ENVIRONMENTS = {
    "controllable_talknet": ("HAY_SAY_TALKNET_CPU_BF16_AUTOCAST", False),
    "so_vits_svc_3": ("HAY_SAY_SVC3_CPU_BF16_AUTOCAST", False),
    "so_vits_svc_4": ("HAY_SAY_SVC4_CPU_BF16_AUTOCAST", True),
    "so_vits_svc_5": ("HAY_SAY_SVC5_CPU_BF16_AUTOCAST", False),
    "rvc": ("HAY_SAY_RVC_CPU_BF16_AUTOCAST", False),
    "styletts_2": ("HAY_SAY_STYLETTS_CPU_BF16_AUTOCAST", True),
    "gpt_so_vits": ("HAY_SAY_GPT_SOVITS_CPU_BF16_AUTOCAST", False),
}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def clean_up(files_to_delete):
    for path in files_to_delete or []:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def construct_full_error_message(architecture_root_dir, files_to_delete):
    message = construct_error_message(architecture_root_dir)
    try:
        clean_up(files_to_delete)
    except Exception:
        message += "\n\nCleanup also failed:\n" + traceback.format_exc(chain=False)
    return message


def construct_error_message(architecture_root_dir):
    payload = request.get_json(silent=True) if has_request_context() else None
    return (
        "An error occurred while generating output:\n"
        + traceback.format_exc()
        + "\nPayload:\n"
        + json.dumps(payload, default=str)
        + "\nArchitecture directory:\n"
        + get_file_list(architecture_root_dir)
    )


def get_file_list(folder):
    if not os.path.exists(folder):
        return f"{folder} does not exist"
    return ", ".join(sorted(os.listdir(folder)))


def model_cpu_bf16_policy(model_id, *, environ=None, cpu_flags=None):
    """Describe a model's configured and effective CPU BF16 policy."""
    try:
        variable, default = MODEL_CPU_BF16_ENVIRONMENTS[model_id]
    except KeyError as error:
        raise ValueError("Unknown model BF16 policy: {}".format(model_id)) from error
    environment = os.environ if environ is None else environ
    configured_value = environment.get(variable)
    raw_value = configured_value if configured_value is not None else ("1" if default else "0")
    configured = str(raw_value).strip().lower()
    if configured in _TRUE_VALUES:
        requested = True
    elif configured in _FALSE_VALUES:
        requested = False
    else:
        raise ValueError("{} must be a boolean, got: {}".format(variable, configured))
    if cpu_flags is not None:
        amx_available = {"amx_bf16", "amx_tile"}.issubset(set(cpu_flags))
    else:
        amx_available = cpu_supports_amx_bf16()
    return {
        "environment_variable": variable,
        "default": bool(default),
        "requested": requested,
        "source": "environment" if configured_value is not None else "default",
        "amx_available": amx_available,
        "effective": requested and amx_available,
    }


def model_cpu_bf16_enabled(model_id, *, environ=None, cpu_flags=None):
    """Resolve a model's effective server-side BF16 policy."""
    return model_cpu_bf16_policy(
        model_id,
        environ=environ,
        cpu_flags=cpu_flags,
    )["effective"]


def runtime_state_with_cpu_bf16(state, model_id):
    """Return a runtime state snapshot with the process's BF16 policy."""
    snapshot = dict(state)
    snapshot["cpu_bf16"] = model_cpu_bf16_policy(model_id)
    return snapshot


def select_hardware(gpu_id, model_id=None):
    env = os.environ.copy()
    use_cpu = gpu_id in (None, "", -1, "-1", "CPU", "cpu")
    env["CUDA_VISIBLE_DEVICES"] = "" if use_cpu else str(gpu_id)
    env["HAY_SAY_CONFIGURE_TORCH_THREADS"] = "1"
    cpu_bf16 = bool(model_id) and model_cpu_bf16_enabled(model_id)
    env["HAY_SAY_CPU_BF16_AUTOCAST"] = "1" if use_cpu and cpu_bf16 else "0"
    return env


def get_gpu_info_from_another_venv(path_to_python_executable, timeout=15):
    code = """
import json
import torch
print(json.dumps([
    {
        'Index': index,
        'Name': torch.cuda.get_device_properties(index).name,
        'Free Memory': torch.cuda.mem_get_info(index)[0],
        'Total Memory': torch.cuda.mem_get_info(index)[1],
    }
    for index in range(torch.cuda.device_count())
]))
"""
    completed = subprocess.run(
        [path_to_python_executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return json.loads(completed.stdout)

"""Helpers shared by native HTTP wrappers around model runtimes."""

import json
import os
import subprocess
import traceback

from flask import has_request_context, request


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


def select_hardware(gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu_id in (None, "", -1, "-1", "CPU") else str(gpu_id)
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

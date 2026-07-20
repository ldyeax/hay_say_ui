import argparse
import base64
import json
import os
import threading
import traceback

import hay_say_common as hsc
import jsonschema
import soundfile
from flask import Flask, request
from hay_say_common.cache import Stage
from jsonschema.exceptions import ValidationError


ARCHITECTURE_NAME = "controllable_talknet"
ARCHITECTURE_ROOT = os.path.join(hsc.ROOT_DIR, ARCHITECTURE_NAME)
PYTHON_EXECUTABLE = os.path.join(hsc.ROOT_DIR, ".venvs", ARCHITECTURE_NAME, "bin", "python")
INFERENCE_CODE_PATH = os.path.join(ARCHITECTURE_ROOT, "controllable_talknet_cli.py")
REQUEST_ROOT = os.environ.get(
    "HAY_SAY_TALKNET_REQUEST_ROOT", os.path.join(ARCHITECTURE_ROOT, "temp", "hay-say-server")
)

app = Flask(__name__)
runtime_manager = hsc.PersistentModelRuntime(
    cpu_workers_environment="HAY_SAY_TALKNET_CPU_WORKERS",
    cpu_workers_default=8,
    gpu_workers_environment="HAY_SAY_TALKNET_GPU_WORKERS",
    gpu_workers_default=1,
    thread_name_prefix="talknet",
)
model_link_lock = threading.Lock()


class BadInputException(Exception):
    pass


def encoded_response(message="", **values):
    values["message"] = base64.b64encode(message.encode("utf-8")).decode("ascii")
    return json.dumps(values, sort_keys=True, indent=4)


def register_methods(cache):
    @app.route("/generate", methods=["POST"])
    def generate():
        try:
            values = parse_inputs()
            request_id = values.pop("request_id")
            output_name = values.pop("output_name")
            session_id = values.pop("session_id")
            input_name = values.pop("input_name")
            disable_reference = values.pop("disable_reference")
            with hsc.request_workspace(REQUEST_ROOT, "talknet-") as workspace:
                reference_audio = prepare_reference_audio(
                    cache, input_name, disable_reference, session_id, workspace
                )
                output_path = os.path.join(workspace, "output.flac")
                link_model_path(values["character"])
                execute_program(
                    reference_audio=reference_audio,
                    output_path=output_path,
                    request_id=request_id,
                    **values,
                )
                runtime_manager.commit_if_active(
                    request_id,
                    lambda: copy_output_audio(cache, output_path, output_name, session_id),
                )
            return encoded_response(), 200
        except BadInputException:
            return encoded_response(traceback.format_exc()), 400
        except hsc.InferenceCancelled as error:
            return encoded_response(str(error), cancelled=True), 499
        except Exception:
            return encoded_response(hsc.construct_error_message(ARCHITECTURE_ROOT)), 500

    @app.route("/cancel", methods=["POST"])
    def cancel():
        payload = request.get_json(silent=True) or {}
        request_ids = payload.get("Request IDs", [])
        if not isinstance(request_ids, list):
            return encoded_response("Request IDs must be an array"), 400
        return encoded_response(**runtime_manager.cancel(request_ids)), 200

    @app.route("/runtime", methods=["GET"])
    def runtime():
        state = hsc.runtime_state_with_cpu_bf16(runtime_manager.state(), ARCHITECTURE_NAME)
        return json.dumps(state), 200

    @app.route("/gpu-info", methods=["GET"])
    def gpu_info():
        return json.dumps(hsc.get_gpu_info_from_another_venv(PYTHON_EXECUTABLE))


def parse_inputs():
    payload = request.get_json(silent=True)
    schema = {
        "type": "object",
        "properties": {
            "Inputs": {
                "type": "object",
                "properties": {
                    "User Text": {"type": "string"},
                    "User Audio": {"type": ["string", "null"]},
                },
                "required": ["User Text"],
            },
            "Options": {
                "type": "object",
                "properties": {
                    "Disable Reference Audio": {"type": "boolean"},
                    "Character": {"type": "string"},
                    "Pitch Factor": {"type": "integer"},
                    "Auto Tune": {"type": "boolean"},
                    "Reduce Metallic Sound": {"type": "boolean"},
                },
                "required": [
                    "Disable Reference Audio", "Character", "Pitch Factor", "Auto Tune",
                    "Reduce Metallic Sound",
                ],
            },
            "Output File": {"type": "string"},
            "GPU ID": {"type": ["string", "integer"]},
            "Session ID": {"type": ["string", "null"]},
            "Request ID": {"type": "string"},
        },
        "required": ["Inputs", "Options", "Output File", "GPU ID", "Session ID"],
    }
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except ValidationError as error:
        raise BadInputException(error.message) from error
    inputs = payload["Inputs"]
    options = payload["Options"]
    return {
        "user_text": inputs["User Text"],
        "input_name": inputs.get("User Audio"),
        "disable_reference": options["Disable Reference Audio"],
        "character": options["Character"],
        "pitch_factor": options["Pitch Factor"],
        "auto_tune": options["Auto Tune"],
        "reduce_metallic": options["Reduce Metallic Sound"],
        "output_name": payload["Output File"],
        "gpu_id": payload["GPU ID"],
        "session_id": payload["Session ID"],
        "request_id": payload.get("Request ID"),
    }


def link_model_path(character):
    character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
    with model_link_lock:
        hsc.create_link(character_dir, os.path.join(ARCHITECTURE_ROOT, "models", character))


def prepare_reference_audio(cache, input_name, disabled, session_id, workspace):
    if disabled or input_name is None:
        return None
    target = os.path.join(workspace, "reference.wav")
    audio, sample_rate = cache.read_audio_from_cache(Stage.PREPROCESSED, session_id, input_name)
    soundfile.write(target, audio, sample_rate, format="WAV")
    return target


def execute_program(user_text, reference_audio, character, pitch_factor, auto_tune, reduce_metallic, gpu_id,
                    request_id, output_path):
    pitch_options = [
        *(("pc",) if auto_tune else ()),
        *(("srec",) if reduce_metallic else ()),
    ]
    runtime_manager.run(
        {
            "request_id": request_id,
            "user_text": user_text,
            "reference_audio": reference_audio,
            "pitch_factor": pitch_factor,
            "pitch_options": pitch_options,
            "output_path": output_path,
        },
        character=character,
        gpu_id=gpu_id,
        environment=hsc.select_hardware(gpu_id, ARCHITECTURE_NAME),
        python_executable=PYTHON_EXECUTABLE,
        worker_script=os.path.join(ARCHITECTURE_ROOT, "hay_say_worker.py"),
        cwd=ARCHITECTURE_ROOT,
    )


def copy_output_audio(cache, path, output_name, session_id):
    audio, sample_rate = hsc.read_audio(path)
    cache.save_audio_to_cache(Stage.OUTPUT, session_id, output_name, audio, sample_rate)


def parse_arguments():
    parser = argparse.ArgumentParser(description="A webservice interface for Controllable TalkNet")
    parser.add_argument("--cache_implementation", default="file", choices=hsc.cache_implementation_map.keys())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    register_methods(hsc.select_cache_implementation(args.cache_implementation))
    app.run(
        debug=False,
        host=os.environ.get("HAY_SAY_CONTROLLABLE_TALKNET_HOST", "127.0.0.1"),
        port=int(os.environ.get("HAY_SAY_CONTROLLABLE_TALKNET_PORT", "6574")),
        threaded=True,
    )

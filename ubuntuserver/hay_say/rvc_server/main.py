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
from jsonschema import ValidationError

from hay_say_common.persistent_inference import PersistentModelRuntime


ARCHITECTURE_NAME = "rvc"
ARCHITECTURE_ROOT = os.path.join(hsc.ROOT_DIR, ARCHITECTURE_NAME)
WEIGHTS_FOLDER = os.path.join(ARCHITECTURE_ROOT, "assets", "weights")
REQUEST_ROOT = os.environ.get(
    "HAY_SAY_RVC_REQUEST_ROOT", os.path.join(ARCHITECTURE_ROOT, "temp", "hay-say-server")
)
PYTHON_EXECUTABLE = os.path.join(hsc.ROOT_DIR, ".venvs", ARCHITECTURE_NAME, "bin", "python")
WORKER_SCRIPT_PATH = os.path.join(ARCHITECTURE_ROOT, "hay_say_worker.py")

INDEX_FILE_EXTENSION = ".index"
WEIGHTS_FILE_EXTENSION = ".pth"
F0_OPTIONS = {"crepe": "crepe", "harvest": "harvest", "parselmouth": "pm", "rmvpe": "rmvpe"}

app = Flask(__name__)
runtime_manager = PersistentModelRuntime(thread_name_prefix="rvc")
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
            with hsc.request_workspace(REQUEST_ROOT, "rvc-") as workspace:
                input_path = os.path.join(workspace, "input.flac")
                output_path = os.path.join(workspace, "output.flac")
                link_model_path(values["character"])
                copy_input_audio(cache, values.pop("input_name"), session_id, input_path)
                execute_program(request_id=request_id, input_path=input_path, output_path=output_path, **values)
                runtime_manager.commit_if_active(
                    request_id,
                    lambda: copy_output(cache, output_path, output_name, session_id),
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
                "properties": {"User Audio": {"type": "string"}},
                "required": ["User Audio"],
            },
            "Options": {
                "type": "object",
                "properties": {
                    "Character": {"type": "string"},
                    "Pitch Shift": {"type": "integer"},
                    "f0 Extraction Method": {"enum": sorted(F0_OPTIONS)},
                    "Index Ratio": {"type": "number", "minimum": 0, "maximum": 1},
                    "Filter Radius": {"type": "integer", "minimum": 0},
                    "Voice Envelope Mix Ratio": {"type": "number", "minimum": 0, "maximum": 1},
                    "Voiceless Consonants Protection Ratio": {
                        "type": "number", "minimum": 0, "maximum": 0.5
                    },
                },
                "required": [
                    "Character", "Pitch Shift", "f0 Extraction Method", "Index Ratio",
                    "Voice Envelope Mix Ratio", "Voiceless Consonants Protection Ratio",
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
    options = payload["Options"]
    return {
        "character": options["Character"],
        "input_name": payload["Inputs"]["User Audio"],
        "pitch_shift": options["Pitch Shift"],
        "f0_method": options["f0 Extraction Method"],
        "index_ratio": options["Index Ratio"],
        "filter_radius": options.get("Filter Radius"),
        "rms_mix_ratio": options["Voice Envelope Mix Ratio"],
        "protect": options["Voiceless Consonants Protection Ratio"],
        "output_name": payload["Output File"],
        "gpu_id": payload["GPU ID"],
        "session_id": payload["Session ID"],
        "request_id": payload.get("Request ID"),
    }


def link_model_path(character):
    character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
    weight_file = hsc.get_single_file_with_extension(character_dir, WEIGHTS_FILE_EXTENSION)
    with model_link_lock:
        hsc.create_link(weight_file, os.path.join(WEIGHTS_FOLDER, character + WEIGHTS_FILE_EXTENSION))


def copy_input_audio(cache, input_name, session_id, target):
    data, sample_rate = cache.read_audio_from_cache(Stage.PREPROCESSED, session_id, input_name)
    soundfile.write(target, data, sample_rate)


def execute_program(character, pitch_shift, f0_method, index_ratio, filter_radius, rms_mix_ratio, protect, gpu_id,
                    request_id, input_path, output_path):
    index_path = get_index_path(character)
    runtime_manager.run(
        {
            "input_path": input_path,
            "output_path": output_path,
            "pitch_shift": pitch_shift,
            "f0_method": F0_OPTIONS[f0_method],
            "index_path": index_path or "",
            "index_ratio": index_ratio,
            "filter_radius": filter_radius,
            "rms_mix_ratio": rms_mix_ratio,
            "protect": protect,
            "request_id": request_id,
        },
        character=character,
        gpu_id=gpu_id,
        environment=hsc.select_hardware(gpu_id, ARCHITECTURE_NAME),
        python_executable=PYTHON_EXECUTABLE,
        worker_script=WORKER_SCRIPT_PATH,
        cwd=ARCHITECTURE_ROOT,
    )


def get_index_path(character):
    try:
        return hsc.get_single_file_with_extension(
            hsc.character_dir(ARCHITECTURE_NAME, character), INDEX_FILE_EXTENSION
        )
    except Exception:
        return None


def copy_output(cache, path, output_name, session_id):
    audio, sample_rate = hsc.read_audio(path)
    cache.save_audio_to_cache(Stage.OUTPUT, session_id, output_name, audio, sample_rate)


def parse_arguments():
    parser = argparse.ArgumentParser(description="A webservice interface for voice conversion with RVC")
    parser.add_argument("--cache_implementation", default="file", choices=hsc.cache_implementation_map.keys())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    register_methods(hsc.select_cache_implementation(args.cache_implementation))
    app.run(
        host=os.environ.get("HAY_SAY_RVC_HOST", "127.0.0.1"),
        port=int(os.environ.get("HAY_SAY_RVC_PORT", "6578")),
        threaded=True,
    )

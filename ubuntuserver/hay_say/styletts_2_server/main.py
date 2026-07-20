import argparse
import base64
import json
import os
import traceback

import hay_say_common as hsc
import jsonschema
import soundfile
from flask import Flask, request
from hay_say_common.cache import Stage
from jsonschema import ValidationError


ARCHITECTURE_NAME = "styletts_2"
ARCHITECTURE_ROOT = os.path.join(hsc.ROOT_DIR, ARCHITECTURE_NAME)
PYTHON_EXECUTABLE = os.path.join(hsc.ROOT_DIR, ".venvs", ARCHITECTURE_NAME, "bin", "python")
WORKER_SCRIPT_PATH = os.path.join(ARCHITECTURE_ROOT, "hay_say_worker.py")
REQUEST_ROOT = os.environ.get(
    "HAY_SAY_STYLETTS_REQUEST_ROOT", os.path.join(ARCHITECTURE_ROOT, "temp", "hay-say-server")
)

PRECOMPUTED_STYLE = "Use Precomputed Style"
USE_REFERENCE_AUDIO = "Use Reference Audio"
DISABLE = "Disable"

app = Flask(__name__)
runtime_manager = hsc.PersistentModelRuntime(
    cpu_workers_environment="HAY_SAY_STYLETTS_CPU_WORKERS",
    cpu_workers_default=8,
    gpu_workers_environment="HAY_SAY_STYLETTS_GPU_WORKERS",
    gpu_workers_default=1,
    thread_name_prefix="styletts",
)


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
            input_hash = values.pop("input_hash")
            with hsc.request_workspace(REQUEST_ROOT, "styletts-") as workspace:
                reference_audio = prepare_reference_audio(
                    cache, input_hash, values["reference_source"], session_id, workspace
                )
                output_path = os.path.join(workspace, "output.flac")
                execute_program(
                    reference_audio=reference_audio, output_path=output_path,
                    request_id=request_id, **values,
                )
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
    option_schema = {
        "Character": {"type": "string"}, "Noise": {"type": "number"},
        "Style Blend": {"type": "number", "minimum": 0, "maximum": 1},
        "Diffusion Steps": {"type": "integer", "minimum": 1}, "Embedding Scale": {"type": "number"},
        "Use Long Form": {"type": "boolean"},
        "Reference Style Source": {"enum": [PRECOMPUTED_STYLE, USE_REFERENCE_AUDIO, DISABLE]},
        "Timbre Reference Blend": {"type": "number", "minimum": 0, "maximum": 1},
        "Prosody Reference Blend": {"type": "number", "minimum": 0, "maximum": 1},
        "Precomputed Style Character": {"type": ["string", "null"]},
        "Precomputed Style Trait": {"type": ["string", "null"]},
        "Speed": {"type": "number", "minimum": 0.1, "maximum": 5},
    }
    schema = {
        "type": "object", "properties": {
            "Inputs": {"type": "object", "properties": {
                "User Text": {"type": "string"}, "User Audio": {"type": ["string", "null"]},
            }, "required": ["User Text"]},
            "Options": {"type": "object", "properties": option_schema, "required": list(option_schema)},
            "Output File": {"type": "string"}, "GPU ID": {"type": ["string", "integer"]},
            "Session ID": {"type": ["string", "null"]}, "Request ID": {"type": "string"},
        }, "required": ["Inputs", "Options", "Output File", "GPU ID", "Session ID"],
    }
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except ValidationError as error:
        raise BadInputException(error.message) from error
    inputs, options = payload["Inputs"], payload["Options"]
    return {
        "user_text": inputs["User Text"], "character": options["Character"], "noise": options["Noise"],
        "style_blend": options["Style Blend"], "diffusion_steps": options["Diffusion Steps"],
        "embedding_scale": options["Embedding Scale"], "use_long_form": options["Use Long Form"],
        "input_hash": inputs.get("User Audio"), "reference_source": options["Reference Style Source"],
        "timbre_blend": options["Timbre Reference Blend"], "prosody_blend": options["Prosody Reference Blend"],
        "style_character": options["Precomputed Style Character"],
        "style_trait": options["Precomputed Style Trait"], "speed": options["Speed"],
        "output_name": payload["Output File"], "gpu_id": payload["GPU ID"],
        "session_id": payload["Session ID"], "request_id": payload.get("Request ID"),
    }


def prepare_reference_audio(cache, input_hash, reference_source, session_id, workspace):
    if reference_source != USE_REFERENCE_AUDIO or input_hash is None:
        return None
    target = os.path.join(workspace, "reference.wav")
    audio, sample_rate = cache.read_audio_from_cache(Stage.PREPROCESSED, session_id, input_hash)
    soundfile.write(target, audio, sample_rate)
    return target


def execute_program(user_text, character, noise, style_blend, diffusion_steps, embedding_scale, use_long_form,
                    reference_audio, reference_source, timbre_blend, prosody_blend, style_character, style_trait,
                    speed, gpu_id, request_id, output_path):
    style_file = os.path.join(
        hsc.guarantee_directory(os.path.join(hsc.MODELS_DIR, ARCHITECTURE_NAME)), "precomputed_styles.json"
    )
    runtime_manager.run(
        {
            "request_id": request_id,
            "options": {
                "text": user_text,
                "output_path": output_path,
                "noise": noise,
                "style_blend": style_blend,
                "diffusion_steps": diffusion_steps,
                "embedding_scale": embedding_scale,
                "use_long_form": use_long_form,
                "reference_audio": reference_audio,
                "reference_style_json": (
                    style_file if reference_source == PRECOMPUTED_STYLE else None
                ),
                "precomputed_style_model": character,
                "precomputed_style_character": style_character,
                "precomputed_style_trait": style_trait,
                "timbre_blend": timbre_blend,
                "prosody_blend": prosody_blend,
                "speed": speed,
            },
        },
        character=character,
        gpu_id=gpu_id,
        environment=hsc.select_hardware(gpu_id, ARCHITECTURE_NAME),
        python_executable=PYTHON_EXECUTABLE,
        worker_script=WORKER_SCRIPT_PATH,
        cwd=ARCHITECTURE_ROOT,
    )


def copy_output(cache, path, output_name, session_id):
    audio, sample_rate = hsc.read_audio(path)
    cache.save_audio_to_cache(Stage.OUTPUT, session_id, output_name, audio, sample_rate)


def parse_arguments():
    parser = argparse.ArgumentParser(description="A webservice interface for StyleTTS2")
    parser.add_argument("--cache_implementation", default="file", choices=hsc.cache_implementation_map.keys())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    register_methods(hsc.select_cache_implementation(args.cache_implementation))
    app.run(
        host=os.environ.get("HAY_SAY_STYLETTS_2_HOST", "127.0.0.1"),
        port=int(os.environ.get("HAY_SAY_STYLETTS_2_PORT", "6580")), threaded=True,
    )

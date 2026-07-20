import argparse
import base64
import json
import os
import subprocess
import traceback

import hay_say_common as hsc
import jsonschema
import soundfile
from flask import Flask, request
from hay_say_common.cache import Stage
from jsonschema import ValidationError


ARCHITECTURE_NAME = "gpt_so_vits"
ARCHITECTURE_ROOT = os.path.join(hsc.ROOT_DIR, ARCHITECTURE_NAME)
PYTHON_EXECUTABLE = os.path.join(hsc.ROOT_DIR, ".venvs", ARCHITECTURE_NAME, "bin", "python")
REQUEST_ROOT = os.environ.get(
    "HAY_SAY_GPT_SOVITS_REQUEST_ROOT", os.path.join(ARCHITECTURE_ROOT, "temp", "hay-say-server")
)

GPT_WEIGHTS_FILE_EXTENSION = ".ckpt"
SO_VITS_WEIGHTS_FILE_EXTENSION = ".pth"
PRECOMPUTATIONS_FILE_EXTENSION = ".safetensors"
USE_PRECOMPUTED_EMBEDDING = "Use Precomputed Embeddings"
USE_REFERENCE_AUDIO = "Use Reference Audio"

SUPPORTED_LANGUAGES_MAP = {
    "Chinese (Mandarin)": "中文", "English": "英文", "Japanese": "日文",
    "Chinese (Cantonese)": "粤语", "Korean": "韩文", "Mandarin-English Mix": "中英混合",
    "Japanese-English Mix": "日英混合", "Cantonese-English Mix": "粤英混合",
    "Korean-English Mix": "韩英混合", "Auto Multilingual": "多语种混合",
    "Auto Multilingual (Cantonese)": "多语种混合(粤语)",
}
CUTTING_STRATEGIES_MAP = {
    "No slicing": "不切", "One slice every 4 sentences": "凑四句一切",
    "One slice every 50 characters": "凑50字一切",
    "Slice by Mandarin Chinese punctuation": "按中文句号。切",
    "Slice by English punctuation": "按英文句号.切",
    "Slice by punctuation (any language)": "按标点符号切",
}

app = Flask(__name__)
runtime_manager = hsc.PersistentModelRuntime(
    cpu_workers_environment="HAY_SAY_GPT_SOVITS_CPU_WORKERS",
    cpu_workers_default=8,
    gpu_workers_environment="HAY_SAY_GPT_SOVITS_GPU_WORKERS",
    gpu_workers_default=1,
    thread_name_prefix="gpt-sovits",
)


class BadInputException(Exception):
    pass


def encoded_response(message="", **values):
    values["message"] = base64.b64encode(message.encode("utf-8")).decode("ascii")
    return json.dumps(values, sort_keys=True, indent=4)


def get_traits(path_to_python_executable, path_to_safetensors):
    code = (
        "import json,sys\n"
        "from safetensors import safe_open\n"
        "with safe_open(sys.argv[1], framework='pt') as source:\n"
        " print(json.dumps(sorted({key.split('.')[0] for key in source.keys()})))\n"
    )
    output = subprocess.check_output([path_to_python_executable, "-c", code, path_to_safetensors])
    return json.loads(output.decode("utf-8"))


def register_methods(cache):
    @app.route("/generate", methods=["POST"])
    def generate():
        try:
            values = parse_inputs()
            request_id = values.pop("request_id")
            output_name = values.pop("output_name")
            session_id = values.pop("session_id")
            reference_option = values.pop("reference_option")
            reference_hash = values.pop("reference_hash")
            reference_text = values.pop("reference_text")
            additional_refs = values.pop("additional_refs")
            with hsc.request_workspace(REQUEST_ROOT, "gpt-sovits-") as workspace:
                if reference_option == USE_REFERENCE_AUDIO:
                    reference_audio = prepare_reference_audio(
                        cache, reference_hash, session_id, workspace, "reference.wav"
                    )
                    prepared_refs = [
                        prepare_reference_audio(
                            cache, input_hash, session_id, workspace, f"additional-{index}.wav"
                        )
                        for index, input_hash in enumerate(dict.fromkeys(additional_refs))
                    ]
                    reference_text_file = write_text(workspace, "reference.txt", reference_text or "")
                    values["trait"] = None
                else:
                    reference_audio = None
                    prepared_refs = []
                    reference_text_file = None
                    values["ref_free"] = None
                target_text_file = write_text(workspace, "target.txt", values.pop("user_text"))
                execute_program(
                    request_id=request_id,
                    workspace=workspace,
                    target_text_file=target_text_file,
                    reference_audio=reference_audio,
                    reference_text_file=reference_text_file,
                    additional_refs=prepared_refs,
                    **values,
                )
                runtime_manager.commit_if_active(
                    request_id,
                    lambda: copy_output(
                        cache, os.path.join(workspace, "output.wav"), output_name, session_id
                    ),
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

    @app.route("/available-traits/<character>", methods=["GET"])
    def available_traits(character):
        paths = hsc.get_files_with_extension(
            hsc.character_dir(ARCHITECTURE_NAME, character), PRECOMPUTATIONS_FILE_EXTENSION
        )
        return get_traits(PYTHON_EXECUTABLE, paths[0]) if paths else []


def parse_inputs():
    payload = request.get_json(silent=True)
    schema = {
        "type": "object",
        "properties": {
            "Inputs": {"type": "object", "properties": {
                "User Text": {"type": "string"}, "User Audio": {"type": ["string", "null"]},
            }, "required": ["User Text"]},
            "Options": {"type": "object", "properties": {
                "Character": {"type": "string"}, "Reference Audio": {"type": ["string", "null"]},
                "Reference Text": {"type": ["string", "null"]},
                "Reference Language": {"enum": list(SUPPORTED_LANGUAGES_MAP)},
                "Target Language": {"enum": list(SUPPORTED_LANGUAGES_MAP)},
                "Cutting Strategy": {"enum": list(CUTTING_STRATEGIES_MAP)},
                "Top-K": {"type": "integer", "minimum": 1},
                "Top-P": {"type": "number", "minimum": 0, "maximum": 1},
                "Temperature": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                "Speed": {"type": "number", "minimum": 0.1, "maximum": 5},
                "Additional Reference Audios": {"type": "array", "items": {"type": "string"}},
                "Trait": {"type": ["string", "null"]},
                "Reference Option": {"enum": [USE_PRECOMPUTED_EMBEDDING, USE_REFERENCE_AUDIO]},
            }, "required": [
                "Character", "Reference Language", "Target Language", "Cutting Strategy", "Top-K",
                "Top-P", "Temperature", "Speed", "Reference Option",
            ]},
            "Output File": {"type": "string"}, "GPU ID": {"type": ["string", "integer"]},
            "Session ID": {"type": ["string", "null"]}, "Request ID": {"type": "string"},
        },
        "required": ["Inputs", "Options", "Output File", "GPU ID", "Session ID"],
    }
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except ValidationError as error:
        raise BadInputException(error.message) from error
    inputs, options = payload["Inputs"], payload["Options"]
    return {
        "user_text": inputs["User Text"], "character": options["Character"],
        "reference_hash": options.get("Reference Audio"), "reference_text": options.get("Reference Text"),
        "reference_language": SUPPORTED_LANGUAGES_MAP[options["Reference Language"]],
        "target_language": SUPPORTED_LANGUAGES_MAP[options["Target Language"]],
        "cutting_strategy": CUTTING_STRATEGIES_MAP[options["Cutting Strategy"]],
        "top_k": options["Top-K"], "top_p": options["Top-P"], "temperature": options["Temperature"],
        "ref_free": not options.get("Reference Text"), "speed": options["Speed"],
        "additional_refs": options.get("Additional Reference Audios", []), "trait": options.get("Trait"),
        "reference_option": options["Reference Option"], "output_name": payload["Output File"],
        "gpu_id": payload["GPU ID"], "session_id": payload["Session ID"],
        "request_id": payload.get("Request ID"),
    }


def prepare_reference_audio(cache, input_hash, session_id, workspace, filename):
    if input_hash is None:
        return None
    if input_hash not in cache.read_metadata(Stage.RAW, session_id):
        raise BadInputException("Reference audio is not present in this session's raw-audio cache.")
    target = os.path.join(workspace, filename)
    audio, sample_rate = cache.read_audio_from_cache(Stage.RAW, session_id, input_hash)
    soundfile.write(target, audio, sample_rate)
    return target


def write_text(workspace, filename, value):
    target = os.path.join(workspace, filename)
    with open(target, "w", encoding="utf-8") as output:
        output.write(value)
    return target


def execute_program(character, reference_audio, reference_text_file, reference_language, target_text_file,
                    target_language, cutting_strategy, top_k, top_p, temperature, ref_free, speed,
                    additional_refs, trait, gpu_id, request_id, workspace):
    character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
    runtime_manager.run(
        {
            "request_id": request_id,
            "precomputed_traits_file": hsc.get_single_file_with_extension(
                character_dir, PRECOMPUTATIONS_FILE_EXTENSION
            ) if trait is not None else None,
            "reference_audio": reference_audio,
            "reference_text_file": reference_text_file,
            "reference_language": reference_language,
            "target_text_file": target_text_file,
            "target_language": target_language,
            "workspace": workspace,
            "cutting_strategy": cutting_strategy,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "ref_free": ref_free,
            "speed": speed,
            "additional_refs": additional_refs,
            "trait": trait,
        },
        character=character,
        gpu_id=gpu_id,
        environment=hsc.select_hardware(gpu_id, ARCHITECTURE_NAME),
        python_executable=PYTHON_EXECUTABLE,
        worker_script=os.path.join(ARCHITECTURE_ROOT, "hay_say_worker.py"),
        cwd=ARCHITECTURE_ROOT,
    )


def copy_output(cache, path, output_name, session_id):
    audio, sample_rate = hsc.read_audio(path)
    cache.save_audio_to_cache(Stage.OUTPUT, session_id, output_name, audio, sample_rate)


def parse_arguments():
    parser = argparse.ArgumentParser(description="A webservice interface for GPT-SoVITS")
    parser.add_argument("--cache_implementation", default="file", choices=hsc.cache_implementation_map.keys())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    register_methods(hsc.select_cache_implementation(args.cache_implementation))
    app.run(
        host=os.environ.get("HAY_SAY_GPT_SO_VITS_HOST", "127.0.0.1"),
        port=int(os.environ.get("HAY_SAY_GPT_SO_VITS_PORT", "6581")), threaded=True,
    )

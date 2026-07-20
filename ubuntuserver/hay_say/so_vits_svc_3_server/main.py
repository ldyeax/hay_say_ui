"""Native REST service for long-lived so-vits-svc 3 inference."""

import argparse
import base64
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import hay_say_common as hsc
from flask import Flask, jsonify, request
from hay_say_common.cache import Stage
from werkzeug.exceptions import HTTPException


ARCHITECTURE_NAME = "so_vits_svc_3"
ARCHITECTURE_ROOT = os.path.join(hsc.ROOT_DIR, ARCHITECTURE_NAME)
DEFAULT_HUBERT_PATH = os.path.join(ARCHITECTURE_ROOT, "hubert", "hubert-soft-0d54a1f4.pt")
DEFAULT_HOST = os.environ.get("HAY_SAY_SVC3_HOST", os.environ.get("SVC3_HOST", "127.0.0.1"))
DEFAULT_PORT = int(os.environ.get("HAY_SAY_SVC3_PORT", os.environ.get("SVC3_PORT", "6575")))

if ARCHITECTURE_ROOT not in sys.path:
    sys.path.insert(0, ARCHITECTURE_ROOT)

from inference.runtime import (  # noqa: E402
    GenerationCancelled,
    ModelCache,
    ModelSpec,
    SvcRuntime,
    file_revision,
    normalize_device,
)


SAFE_CACHE_KEY = re.compile(r"^[A-Za-z0-9_.-]{1,255}$")
MAX_PITCH_BATCH = 64
MAX_CANCEL_BATCH = 256


class RequestError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = int(status_code)


@dataclass(frozen=True)
class GenerateRequest:
    input_name: str
    character: str
    speaker: str
    pitches: tuple
    output_names: tuple
    device: str
    cpu_bf16: bool
    session_id: object
    request_id: object
    slice_db: float
    spec: ModelSpec


class ModelResolver:
    """Resolve a character folder without allowing request-controlled paths."""

    def __init__(self, character_dir_func=None):
        self._character_dir = character_dir_func or hsc.character_dir

    def resolve(self, character, requested_speaker=None):
        character = validate_path_component(character, "Character")
        character_dir = Path(self._character_dir(ARCHITECTURE_NAME, character))
        root_dir = Path(self._character_dir(ARCHITECTURE_NAME, "__root_probe__")).parent
        if Path(os.path.abspath(str(character_dir))).parent != Path(os.path.abspath(str(root_dir))):
            raise RequestError("Character resolves outside the model directory")
        if not character_dir.is_dir():
            raise RequestError("Character model was not found: {}".format(character), 404)

        config_path = character_dir / "config.json"
        if not config_path.is_file():
            raise RequestError("config.json was not found for character {}".format(character), 404)
        model_paths = sorted(
            path for path in character_dir.iterdir()
            if path.is_file() and path.name.startswith("G_") and path.suffix == ".pth"
        )
        if not model_paths:
            raise RequestError("No G_*.pth model was found for character {}".format(character), 404)
        if len(model_paths) > 1:
            raise RequestError(
                "Character {} has multiple G_*.pth files; keep exactly one checkpoint".format(character)
            )

        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
        except (OSError, ValueError) as exc:
            raise RequestError("Unable to read {}: {}".format(config_path.name, exc)) from exc
        speakers = config.get("spk")
        if not isinstance(speakers, dict) or not speakers:
            raise RequestError("config.json must contain a non-empty 'spk' object")
        speaker = self._resolve_speaker(character_dir, speakers, requested_speaker)
        spec = ModelSpec(
            character,
            str(model_paths[0]),
            str(config_path),
            file_revision(model_paths[0]),
            file_revision(config_path),
        )
        return spec, speaker, sorted(speakers.keys())

    @staticmethod
    def _resolve_speaker(character_dir, speakers, requested_speaker):
        if requested_speaker is not None:
            if not isinstance(requested_speaker, str) or not requested_speaker:
                raise RequestError("Speaker must be a non-empty string")
            selected = requested_speaker
        elif len(speakers) == 1:
            selected = next(iter(speakers))
        else:
            speaker_path = character_dir / "speaker.json"
            if not speaker_path.is_file():
                raise RequestError(
                    "speaker.json is required when config.json contains multiple speakers"
                )
            try:
                with speaker_path.open("r", encoding="utf-8") as speaker_file:
                    selector = json.load(speaker_file)
                selected = selector["speaker"]
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise RequestError(
                    "speaker.json must contain a string in the form {'speaker': '<name>'}"
                ) from exc
        if selected not in speakers:
            raise RequestError(
                "Speaker {!r} is not present in config.json; expected one of {}".format(
                    selected, sorted(speakers.keys())
                )
            )
        return selected


def encode_message(message):
    return base64.b64encode(str(message).encode("utf-8")).decode("ascii")


def json_response(message="", status=200, error_type=None, **values):
    body = {"message": encode_message(message)}
    body.update(values)
    if error_type is not None:
        body["error"] = {"type": error_type, "detail": str(message)}
    return jsonify(body), int(status)


def validate_cache_key(value, label, allow_none=False):
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or not SAFE_CACHE_KEY.fullmatch(value):
        raise RequestError(
            "{} must be a 1-255 character cache identifier containing only letters, numbers, '.', '_' or '-'".format(
                label
            )
        )
    if value in (".", ".."):
        raise RequestError("{} cannot be '.' or '..'".format(label))
    return value


def validate_path_component(value, label):
    if not isinstance(value, str) or not value or len(value) > 255:
        raise RequestError("{} must be a non-empty string no longer than 255 characters".format(label))
    if value in (".", "..") or "\x00" in value or Path(value).name != value or "/" in value or "\\" in value:
        raise RequestError("{} must be a model directory name, not a path".format(label))
    return value


def validate_pitch(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError("Pitch shifts must be integers")
    if value < -96 or value > 96:
        raise RequestError("Pitch shifts must be between -96 and 96 semitones")
    return value


def require_object(value, label):
    if not isinstance(value, dict):
        raise RequestError("{} must be a JSON object".format(label))
    return value


def parse_cancel_request_ids(payload):
    payload = require_object(payload, "Request body")
    values = payload.get("Request IDs")
    if not isinstance(values, list) or not values:
        raise RequestError("Request IDs must be a non-empty array")
    if len(values) > MAX_CANCEL_BATCH:
        raise RequestError("Request IDs may contain at most {} entries".format(MAX_CANCEL_BATCH))
    return tuple(dict.fromkeys(
        validate_cache_key(value, "Request IDs") for value in values
    ))


def parse_generate_request(payload, resolver):
    payload = require_object(payload, "Request body")
    inputs = require_object(payload.get("Inputs"), "Inputs")
    options = require_object(payload.get("Options"), "Options")

    input_name = validate_cache_key(inputs.get("User Audio"), "Inputs.User Audio")
    character = validate_path_component(options.get("Character"), "Options.Character")
    requested_speaker = options.get("Speaker")

    if "Pitch Shifts" in options:
        pitch_values = options["Pitch Shifts"]
        output_values = payload.get("Output Files")
        if not isinstance(pitch_values, list) or not pitch_values:
            raise RequestError("Options.Pitch Shifts must be a non-empty array")
        if not isinstance(output_values, list):
            raise RequestError("Output Files must be an array when Pitch Shifts is used")
    else:
        if "Pitch Shift" not in options:
            raise RequestError("Options must include Pitch Shift or Pitch Shifts")
        pitch_values = [options["Pitch Shift"]]
        output_values = [payload.get("Output File")]

    if len(pitch_values) > MAX_PITCH_BATCH:
        raise RequestError("A pitch batch may contain at most {} outputs".format(MAX_PITCH_BATCH))
    if len(pitch_values) != len(output_values):
        raise RequestError("Pitch Shifts and Output Files must have the same length")
    pitches = tuple(validate_pitch(value) for value in pitch_values)
    output_names = tuple(validate_cache_key(value, "Output File") for value in output_values)
    if len(set(output_names)) != len(output_names):
        raise RequestError("Output Files must be unique")

    if "GPU ID" not in payload:
        raise RequestError("GPU ID is required")
    try:
        device = normalize_device(payload["GPU ID"])
    except ValueError as exc:
        raise RequestError(str(exc)) from exc
    session_id = validate_cache_key(payload.get("Session ID"), "Session ID", allow_none=True)
    request_id = validate_cache_key(payload.get("Request ID"), "Request ID", allow_none=True)
    cpu_bf16 = device == "cpu" and hsc.model_cpu_bf16_enabled(ARCHITECTURE_NAME)

    slice_db = options.get("Slice Threshold", -40.0)
    if isinstance(slice_db, bool) or not isinstance(slice_db, (int, float)):
        raise RequestError("Slice Threshold must be a number")
    slice_db = float(slice_db)
    if slice_db < -100.0 or slice_db > 0.0:
        raise RequestError("Slice Threshold must be between -100 and 0 dB")

    spec, speaker, _ = resolver.resolve(character, requested_speaker=requested_speaker)
    return GenerateRequest(
        input_name=input_name,
        character=character,
        speaker=speaker,
        pitches=pitches,
        output_names=output_names,
        device=device,
        cpu_bf16=cpu_bf16,
        session_id=session_id,
        request_id=request_id,
        slice_db=slice_db,
        spec=spec,
    )


def build_runtime(hubert_path=DEFAULT_HUBERT_PATH, max_models_per_device=2):
    return SvcRuntime(ModelCache(hubert_path, max_models_per_device=max_models_per_device))


def create_app(cache, runtime=None, character_dir_func=None):
    application = Flask(__name__)
    register_methods(
        cache,
        runtime=runtime,
        application=application,
        character_dir_func=character_dir_func,
    )
    return application


def register_methods(cache, runtime=None, application=None, character_dir_func=None):
    application = application or app
    runtime = runtime or build_runtime()
    resolver = ModelResolver(character_dir_func)
    cache_write_lock = threading.RLock()

    @application.errorhandler(RequestError)
    def handle_request_error(error):
        return json_response(str(error), error.status_code, error_type="bad_request")

    @application.errorhandler(GenerationCancelled)
    def handle_generation_cancelled(error):
        return json_response(str(error), 409, error_type="cancelled", cancelled=True)

    @application.errorhandler(Exception)
    def handle_unexpected_error(error):
        if isinstance(error, HTTPException):
            return json_response(error.description, error.code, error_type="http_error")
        application.logger.exception("SVC3 request failed")
        return json_response(
            "SVC3 inference failed: {}".format(error),
            500,
            error_type="inference_error",
        )

    @application.route("/generate", methods=["POST"])
    def generate():
        parsed = parse_generate_request(request.get_json(silent=True), resolver)
        with runtime.cancellation_scope(parsed.request_id) as cancellation:
            if hasattr(cache, "file_is_already_cached") and not cache.file_is_already_cached(
                Stage.PREPROCESSED, parsed.session_id, parsed.input_name
            ):
                raise RequestError(
                    "Input audio {!r} was not found in the preprocess cache".format(parsed.input_name),
                    404,
                )
            try:
                source_audio, sample_rate = cache.read_audio_from_cache(
                    Stage.PREPROCESSED, parsed.session_id, parsed.input_name
                )
            except Exception as exc:
                raise RequestError(
                    "Unable to read input audio {!r} from the preprocess cache".format(parsed.input_name),
                    404,
                ) from exc

            outputs, output_sample_rate = runtime.generate(
                parsed.spec,
                parsed.speaker,
                parsed.pitches,
                source_audio,
                sample_rate,
                parsed.device,
                slice_db=parsed.slice_db,
                cpu_bf16=parsed.cpu_bf16,
                cancellation=cancellation,
            )
            runtime.raise_if_cancelled(cancellation)
            def commit_outputs():
                response_outputs = []
                with cache_write_lock:
                    for output_name, (pitch, output_audio) in zip(parsed.output_names, outputs):
                        cache.save_audio_to_cache(
                            Stage.OUTPUT,
                            parsed.session_id,
                            output_name,
                            output_audio,
                            output_sample_rate,
                        )
                        response_outputs.append({"pitch_shift": pitch, "output_file": output_name})
                return response_outputs

            response_outputs = runtime.commit_if_active(cancellation, commit_outputs)
            return json_response(outputs=response_outputs, device=parsed.device)

    @application.route("/cancel", methods=["POST"])
    def cancel():
        request_ids = parse_cancel_request_ids(request.get_json(silent=True))
        result = runtime.cancel(request_ids)
        return json_response(**result)

    @application.route("/health", methods=["GET"])
    def health():
        state = runtime.state()
        return jsonify({
            "status": "ok",
            "architecture": ARCHITECTURE_NAME,
            "warm": state["warm"],
            "busy": state["busy"],
        })

    @application.route("/gpu-info", methods=["GET"])
    def gpu_info():
        return jsonify(runtime.gpu_info())

    @application.route("/runtime", methods=["GET"])
    def runtime_state():
        state = hsc.runtime_state_with_cpu_bf16(runtime.state(), ARCHITECTURE_NAME)
        return jsonify(state)

    @application.route("/warm", methods=["POST"])
    @application.route("/runtime/warm", methods=["POST"])
    def warm():
        payload = require_object(request.get_json(silent=True), "Request body")
        options = payload.get("Options") if isinstance(payload.get("Options"), dict) else payload
        character = validate_path_component(options.get("Character"), "Character")
        spec, speaker, speakers = resolver.resolve(character, options.get("Speaker"))
        try:
            device = normalize_device(payload.get("GPU ID", options.get("GPU ID", "")))
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
        warmed = runtime.warm(spec, device)
        return json_response(warmed=warmed, speaker=speaker, speakers=speakers, runtime=runtime.state())

    @application.route("/unload", methods=["POST"])
    @application.route("/runtime/unload", methods=["POST"])
    def unload():
        payload = request.get_json(silent=True)
        if payload is None:
            payload = {}
        payload = require_object(payload, "Request body")
        character = payload.get("Character")
        if character is not None:
            character = validate_path_component(character, "Character")
        device = None
        if "GPU ID" in payload:
            try:
                device = normalize_device(payload["GPU ID"])
            except ValueError as exc:
                raise RequestError(str(exc)) from exc
        result = runtime.unload(character=character, device=device)
        status = 409 if result["busy_models"] else 200
        return json_response(status=status, result=result, runtime=runtime.state())

    return application


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Native web service for so-vits-svc 3 voice conversion",
    )
    parser.add_argument(
        "--cache_implementation",
        "--cache-implementation",
        default="file",
        choices=hsc.cache_implementation_map.keys(),
        help="Hay Say audio cache implementation",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host (defaults to loopback)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--hubert-path", default=DEFAULT_HUBERT_PATH)
    parser.add_argument("--max-models-per-device", type=int, default=2)
    return parser.parse_args(argv)


app = Flask(__name__)


def main(argv=None):
    args = parse_arguments(argv)
    cache = hsc.select_cache_implementation(args.cache_implementation)
    runtime = build_runtime(args.hubert_path, args.max_models_per_device)
    register_methods(cache, runtime=runtime, application=app)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

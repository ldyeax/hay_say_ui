"""Native HTTP service for persistent So-VITS-SVC 4 inference."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import hay_say_common as hsc
from flask import Flask, jsonify, request
from hay_say_common.cache import Stage
from werkzeug.exceptions import HTTPException


ARCHITECTURE_NAME = "so_vits_svc_4"
DEFAULT_HOST = os.environ.get("HAY_SAY_SO_VITS_SVC_4_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("HAY_SAY_SO_VITS_SVC_4_PORT", "6576"))
INSTALL_ROOT = os.path.realpath(os.environ.get("HAY_SAY_HOME", hsc.ROOT_DIR))
SVC4_ROOT = os.path.join(INSTALL_ROOT, ARCHITECTURE_NAME)
SVC41_ROOT = os.path.join(INSTALL_ROOT, "so_vits_svc_4_dot_1_stable")
RUNTIME_ROOT = os.path.realpath(os.environ.get("HAY_SAY_SVC4_RUNTIME_ROOT", SVC4_ROOT))
if RUNTIME_ROOT not in sys.path:
    sys.path.insert(0, RUNTIME_ROOT)

from runtime import (  # noqa: E402
    GenerationCancelled,
    ModelSpec,
    build_runtime,
    config_details,
    file_revision,
    normalize_device,
)


SAFE_CACHE_KEY = re.compile(r"^[A-Za-z0-9_.-]{1,255}$")
MAX_CANCEL_BATCH = 256


class RequestError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def json_response(message: Optional[str] = None, http_status: int = 200, **values: Any):
    body = dict(values)
    if message is not None:
        body["error"] = message
    return jsonify(body), http_status


def require_object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise RequestError("{} must be a JSON object".format(name))
    return value


def cache_key(value: Any, name: str, allow_none: bool = False) -> Optional[str]:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not SAFE_CACHE_KEY.fullmatch(value):
        raise RequestError(
            "{} must be a 1-255 character cache identifier containing only letters, numbers, '.', '_' or '-'".format(
                name
            )
        )
    if value in (".", ".."):
        raise RequestError("{} cannot be '.' or '..'".format(name))
    return value


def component(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 255:
        raise RequestError("{} must be a non-empty string no longer than 255 characters".format(name))
    if value in (".", "..") or "\x00" in value or Path(value).name != value or "/" in value or "\\" in value:
        raise RequestError("{} must be a model directory name, not a path".format(name))
    return value


def number(value: Any, name: str, minimum: float, maximum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError("{} must be a number".format(name))
    result = float(value)
    if result < minimum or (maximum is not None and result > maximum):
        bounds = "at least {}".format(minimum)
        if maximum is not None:
            bounds = "between {} and {}".format(minimum, maximum)
        raise RequestError("{} must be {}".format(name, bounds))
    return result


def integer(value: Any, name: str, minimum: int, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError("{} must be an integer".format(name))
    if value < minimum or (maximum is not None and value > maximum):
        bounds = "at least {}".format(minimum)
        if maximum is not None:
            bounds = "between {} and {}".format(minimum, maximum)
        raise RequestError("{} must be {}".format(name, bounds))
    return value


def boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RequestError("{} must be a boolean".format(name))
    return value


def parse_cancel_request_ids(payload: Any) -> tuple[str, ...]:
    payload = require_object(payload, "Request body")
    values = payload.get("Request IDs")
    if not isinstance(values, list) or not values:
        raise RequestError("Request IDs must be a non-empty array")
    if len(values) > MAX_CANCEL_BATCH:
        raise RequestError("Request IDs may contain at most {} entries".format(MAX_CANCEL_BATCH))
    return tuple(dict.fromkeys(cache_key(value, "Request IDs") for value in values))


class ModelResolver:
    def __init__(self, character_dir_func=None):
        self.character_dir_func = character_dir_func or hsc.character_dir

    def resolve(self, character: str) -> tuple[ModelSpec, str]:
        candidate = self.character_dir_func(ARCHITECTURE_NAME, character)
        root_probe = self.character_dir_func(ARCHITECTURE_NAME, "__root_probe__")
        root_dir = os.path.realpath(os.path.dirname(root_probe))
        character_dir = os.path.realpath(candidate)
        if os.path.dirname(character_dir) != root_dir:
            raise RequestError("Character resolves outside the model directory")
        if not os.path.isdir(character_dir):
            raise RequestError("Character model was not found: {}".format(character), 404)
        model_names = sorted(
            name
            for name in os.listdir(character_dir)
            if name.startswith("G_")
            and name.endswith(".pth")
            and os.path.isfile(os.path.join(character_dir, name))
        )
        if len(model_names) != 1:
            raise RequestError("Character must contain exactly one G_*.pth checkpoint")
        model_path = os.path.join(character_dir, model_names[0])
        config_path = os.path.join(character_dir, "config.json")
        if not os.path.isfile(config_path):
            raise RequestError("Character config.json was not found")
        cluster_names = sorted(
            name
            for name in os.listdir(character_dir)
            if name.startswith("kmeans") and os.path.isfile(os.path.join(character_dir, name))
        )
        if len(cluster_names) > 1:
            raise RequestError("Character contains more than one kmeans model")
        cluster_path = os.path.join(character_dir, cluster_names[0]) if cluster_names else ""

        version, target_sample = config_details(config_path)
        with open(config_path, encoding="utf-8") as source:
            speakers = json.load(source).get("spk")
        if not isinstance(speakers, dict) or not speakers:
            raise RequestError("Character config has no speakers")
        speaker_path = os.path.join(character_dir, "speaker.json")
        if len(speakers) == 1:
            speaker = next(iter(speakers))
        elif not os.path.isfile(speaker_path):
            raise RequestError("speaker.json is required when config.json contains multiple speakers")
        else:
            try:
                with open(speaker_path, encoding="utf-8") as source:
                    configured = json.load(source).get("speaker")
            except (OSError, AttributeError, json.JSONDecodeError) as exc:
                raise RequestError("speaker.json is malformed") from exc
            if configured not in speakers:
                raise RequestError("speaker.json selects a speaker absent from config.json")
            speaker = configured

        source_root = SVC41_ROOT if version == "4.1" else SVC4_ROOT
        if not os.path.isfile(os.path.join(source_root, "inference", "infer_tool.py")):
            raise RequestError("SVC{} runtime source is not installed".format(version), 503)
        return ModelSpec(
            character=character,
            version=version,
            source_root=source_root,
            model_path=model_path,
            config_path=config_path,
            cluster_path=cluster_path,
            target_sample=target_sample,
            model_revision=file_revision(model_path),
            config_revision=file_revision(config_path),
            cluster_revision=file_revision(cluster_path) if cluster_path else None,
        ), speaker


@dataclass(frozen=True)
class GenerateRequest:
    input_name: str
    output_name: str
    session_id: Optional[str]
    request_id: Optional[str]
    spec: ModelSpec
    speaker: str
    device: str
    pitch: int
    predict_pitch: bool
    slice_length: float
    crossfade_length: float
    slice_workers: int
    character_likeness: float
    reduce_hoarseness: bool
    enhance: bool
    noise_scale: float
    cpu_bf16: bool


def parse_generate_request(payload: Any, resolver: ModelResolver) -> GenerateRequest:
    payload = require_object(payload, "Request body")
    inputs = require_object(payload.get("Inputs"), "Inputs")
    options = require_object(payload.get("Options"), "Options")
    input_name = cache_key(inputs.get("User Audio"), "Inputs.User Audio")
    output_name = cache_key(payload.get("Output File"), "Output File")
    session_id = cache_key(payload.get("Session ID"), "Session ID", allow_none=True)
    request_id = cache_key(payload.get("Request ID"), "Request ID", allow_none=True)
    character = component(options.get("Character"), "Options.Character")
    spec, speaker = resolver.resolve(character)
    if "GPU ID" not in payload:
        raise RequestError("GPU ID is required")
    try:
        device = normalize_device(payload["GPU ID"])
    except ValueError as exc:
        raise RequestError(str(exc)) from exc

    slice_length = number(options.get("Slice Length"), "Slice Length", 0, 20)
    crossfade = number(options.get("Cross-Fade Length"), "Cross-Fade Length", 0, 20)
    if crossfade and not slice_length:
        raise RequestError("Cross-Fade Length requires a positive Slice Length")
    if crossfade > slice_length:
        raise RequestError("Cross-Fade Length cannot exceed Slice Length")
    return GenerateRequest(
        input_name=input_name,
        output_name=output_name,
        session_id=session_id,
        request_id=request_id,
        spec=spec,
        speaker=speaker,
        device=device,
        pitch=integer(options.get("Pitch Shift"), "Pitch Shift", -36, 36),
        predict_pitch=boolean(options.get("Predict Pitch"), "Predict Pitch"),
        slice_length=slice_length,
        crossfade_length=crossfade,
        slice_workers=integer(options.get("Slice Workers", 0), "Slice Workers", 0, 64),
        character_likeness=number(
            options.get("Character Likeness"), "Character Likeness", 0, 1
        ),
        reduce_hoarseness=boolean(options.get("Reduce Hoarseness"), "Reduce Hoarseness"),
        enhance=boolean(options.get("Apply nsf_hifigan"), "Apply nsf_hifigan"),
        noise_scale=number(options.get("Noise Scale"), "Noise Scale", 0, 5),
        cpu_bf16=(
            device == "cpu" and hsc.model_cpu_bf16_enabled(ARCHITECTURE_NAME)
        ),
    )


def create_app(cache, runtime=None, character_dir_func=None) -> Flask:
    application = Flask(__name__)
    register_methods(application, cache, runtime, character_dir_func)
    return application


def register_methods(application, cache, runtime=None, character_dir_func=None):
    runtime = runtime or build_runtime()
    resolver = ModelResolver(character_dir_func)
    cache_write_lock = threading.Lock()

    @application.errorhandler(RequestError)
    def handle_request_error(error):
        return json_response(str(error), error.status_code, error_type="bad_request")

    @application.errorhandler(GenerationCancelled)
    def handle_generation_cancelled(error):
        return json_response(str(error), 409, error_type="cancelled", status="cancelled")

    @application.errorhandler(Exception)
    def handle_unexpected_error(error):
        if isinstance(error, HTTPException):
            return json_response(error.description, error.code, error_type="http_error")
        application.logger.exception("SVC4 request failed")
        return json_response(
            "SVC4 inference failed: {}".format(error),
            500,
            error_type="inference_error",
        )

    @application.route("/generate", methods=["POST"])
    def generate():
        parsed = parse_generate_request(request.get_json(silent=True), resolver)
        with runtime.cancellation_scope(parsed.request_id) as cancellation:
            try:
                source_audio, sample_rate = cache.read_audio_from_cache(
                    Stage.PREPROCESSED, parsed.session_id, parsed.input_name
                )
            except Exception as exc:
                raise RequestError(
                    "Input audio {!r} was not found in the preprocess cache".format(parsed.input_name),
                    404,
                ) from exc
            output, output_rate, workers = runtime.generate(
                parsed.spec,
                parsed.speaker,
                source_audio,
                sample_rate,
                parsed.device,
                parsed.pitch,
                parsed.slice_length,
                parsed.crossfade_length,
                parsed.slice_workers,
                parsed.character_likeness,
                parsed.predict_pitch,
                parsed.reduce_hoarseness,
                parsed.enhance,
                parsed.noise_scale,
                parsed.cpu_bf16,
                cancellation=cancellation,
            )
            runtime.raise_if_cancelled(cancellation)
            def commit_output():
                with cache_write_lock:
                    cache.save_audio_to_cache(
                        Stage.OUTPUT, parsed.session_id, parsed.output_name, output, output_rate
                    )

            runtime.commit_if_active(cancellation, commit_output)
            return json_response(
                output_file=parsed.output_name,
                device=parsed.device,
                slice_workers=workers,
            )

    @application.route("/cancel", methods=["POST"])
    def cancel():
        request_ids = parse_cancel_request_ids(request.get_json(silent=True))
        return json_response(**runtime.cancel(request_ids))

    @application.route("/health", methods=["GET"])
    def health():
        state = runtime.state()
        return jsonify(
            status="ok",
            architecture=ARCHITECTURE_NAME,
            warm=state["warm"],
            busy=state["busy"],
        )

    @application.route("/gpu-info", methods=["GET"])
    def gpu_info():
        return jsonify(hsc.get_gpu_info_from_another_venv(sys.executable))

    @application.route("/runtime", methods=["GET"])
    def runtime_state():
        state = hsc.runtime_state_with_cpu_bf16(runtime.state(), ARCHITECTURE_NAME)
        return jsonify(state)

    @application.route("/warm", methods=["POST"])
    @application.route("/runtime/warm", methods=["POST"])
    def warm():
        payload = require_object(request.get_json(silent=True), "Request body")
        options = payload.get("Options") if isinstance(payload.get("Options"), dict) else payload
        character = component(options.get("Character"), "Character")
        spec, _speaker = resolver.resolve(character)
        try:
            device = normalize_device(payload.get("GPU ID", options.get("GPU ID", "")))
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
        enhance = options.get("Apply nsf_hifigan", False)
        workers = options.get("Slice Workers", 1)
        warmed = runtime.warm(
            spec,
            device,
            boolean(enhance, "Apply nsf_hifigan"),
            integer(workers, "Slice Workers", 0, 64),
        )
        return json_response(warmed=warmed, runtime=runtime.state())

    @application.route("/unload", methods=["POST"])
    @application.route("/runtime/unload", methods=["POST"])
    def unload():
        payload = request.get_json(silent=True) or {}
        payload = require_object(payload, "Request body")
        character = payload.get("Character")
        if character is not None:
            character = component(character, "Character")
        device = None
        if "GPU ID" in payload:
            try:
                device = normalize_device(payload["GPU ID"])
            except ValueError as exc:
                raise RequestError(str(exc)) from exc
        result = runtime.unload(character, device)
        operation_status = "busy" if result["busy_models"] else "unloaded"
        return json_response(
            None,
            409 if result["busy_models"] else 200,
            status=operation_status,
            result=result,
            runtime=runtime.state(),
        )

    return application


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Native So-VITS-SVC 4 service")
    parser.add_argument(
        "--cache_implementation",
        "--cache-implementation",
        default="file",
        choices=hsc.cache_implementation_map.keys(),
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--cpu-slice-workers",
        type=int,
        default=int(os.environ.get("HAY_SAY_SVC4_CPU_SLICE_WORKERS", "4")),
        choices=range(1, 65),
    )
    parser.add_argument("--max-models-per-device", type=int, default=1)
    return parser.parse_args(argv)


app = Flask(__name__)


def main(argv=None):
    args = parse_arguments(argv)
    cache = hsc.select_cache_implementation(args.cache_implementation)
    runtime = build_runtime(args.cpu_slice_workers, args.max_models_per_device)
    atexit.register(runtime.close)
    register_methods(app, cache, runtime)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

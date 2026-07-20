"""Native HTTP service for persistent So-VITS-SVC 5 inference."""

from __future__ import annotations

import argparse
import atexit
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import hay_say_common as hsc
import yaml
from flask import Flask, jsonify, request
from hay_say_common.cache import Stage
from werkzeug.exceptions import HTTPException


ARCHITECTURE_NAME = "so_vits_svc_5"
DEFAULT_HOST = os.environ.get("HAY_SAY_SO_VITS_SVC_5_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("HAY_SAY_SO_VITS_SVC_5_PORT", "6577"))
INSTALL_ROOT = os.path.realpath(os.environ.get("HAY_SAY_HOME", hsc.ROOT_DIR))
SVC5_V1_ROOT = os.path.join(INSTALL_ROOT, "so_vits_svc_5_v1")
SVC5_V2_ROOT = os.path.join(INSTALL_ROOT, "so_vits_svc_5_v2")
SERVER_ROOT = os.path.dirname(os.path.realpath(__file__))
VERSION_DETERMINATOR = os.path.join(SERVER_ROOT, "version_determinator.py")
if SERVER_ROOT not in sys.path:
    sys.path.insert(0, SERVER_ROOT)

from svc5_runtime import (  # noqa: E402
    GenerationCancelled,
    ModelSpec,
    build_runtime,
    file_revision,
    normalize_device,
)


SAFE_CACHE_KEY = re.compile(r"^[A-Za-z0-9_.-]{1,255}$")
SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


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


def request_id(value: Any, name: str = "Request ID", allow_none: bool = True) -> Optional[str]:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not SAFE_REQUEST_ID.fullmatch(value):
        raise RequestError(
            "{} must be a 1-128 character identifier containing only letters, numbers, '.', '_', ':' or '-'".format(
                name
            )
        )
    return value


def component(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 255:
        raise RequestError("{} must be a non-empty string no longer than 255 characters".format(name))
    if value in (".", "..") or "\x00" in value or Path(value).name != value or "/" in value or "\\" in value:
        raise RequestError("{} must be a model directory name, not a path".format(name))
    return value


def integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError("{} must be an integer".format(name))
    if value < minimum or value > maximum:
        raise RequestError("{} must be between {} and {}".format(name, minimum, maximum))
    return value


def boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RequestError("{} must be a boolean".format(name))
    return value


def _single_file(directory: str, predicate: Callable[[str], bool], description: str) -> str:
    try:
        names = sorted(
            name
            for name in os.listdir(directory)
            if predicate(name) and os.path.isfile(os.path.join(directory, name))
        )
    except OSError as exc:
        raise RequestError("Unable to inspect {}".format(description)) from exc
    if len(names) != 1:
        raise RequestError(
            "Expected exactly one {} in {}, found {}".format(description, directory, len(names))
        )
    return os.path.join(directory, names[0])


def configuration_sample_rate(config_path: str) -> int:
    try:
        with open(config_path, encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RequestError("Unable to read the SVC5 YAML configuration", 503) from exc
    data = config.get("data") if isinstance(config, dict) else None
    sample_rate = data.get("sampling_rate") if isinstance(data, dict) else None
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise RequestError(
            "SVC5 configuration data.sampling_rate must be a positive integer", 503
        )
    return sample_rate


class VersionResolver:
    """Run the legacy checkpoint probe once per checkpoint revision."""

    def __init__(self, determine: Optional[Callable[[str], int]] = None):
        self.determine = determine or self._determine
        self._cache: Dict[Tuple[str, tuple], int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _determine(checkpoint_path: str) -> int:
        if not os.path.isfile(VERSION_DETERMINATOR):
            raise RequestError("SVC5 version determinator is not installed", 503)
        completed = subprocess.run(
            [sys.executable, VERSION_DETERMINATOR, checkpoint_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        try:
            version = int(completed.stdout.strip())
        except ValueError as exc:
            raise RequestError("SVC5 checkpoint version probe returned invalid output") from exc
        if version not in (1, 2):
            raise RequestError("Unsupported SVC5 checkpoint version: {}".format(version))
        return version

    def resolve(self, checkpoint_path: str) -> int:
        key = (os.path.realpath(checkpoint_path), file_revision(checkpoint_path))
        with self._lock:
            version = self._cache.get(key)
            if version is None:
                version = self.determine(checkpoint_path)
                self._cache[key] = version
            return version


class ModelResolver:
    def __init__(self, character_dir_func=None, version_resolver=None):
        self.character_dir_func = character_dir_func or hsc.character_dir
        self.version_resolver = version_resolver or VersionResolver()

    def resolve(self, character: str) -> ModelSpec:
        candidate = self.character_dir_func(ARCHITECTURE_NAME, character)
        root_probe = self.character_dir_func(ARCHITECTURE_NAME, "__root_probe__")
        model_root = os.path.realpath(os.path.dirname(root_probe))
        character_dir = os.path.realpath(candidate)
        if os.path.dirname(character_dir) != model_root:
            raise RequestError("Character resolves outside the model directory")
        if not os.path.isdir(character_dir):
            raise RequestError("Character model was not found: {}".format(character), 404)

        original_checkpoint = _single_file(
            character_dir,
            lambda name: name.endswith(".pt"),
            "SVC5 .pt checkpoint",
        )
        version = self.version_resolver.resolve(original_checkpoint)
        source_root = SVC5_V1_ROOT if version == 1 else SVC5_V2_ROOT
        if not os.path.isfile(os.path.join(source_root, "svc_inference.py")):
            raise RequestError("SVC5 v{} runtime source is not installed".format(version), 503)

        exported = os.path.join(character_dir, "sovits5.0.pth")
        checkpoint_path = exported if os.path.isfile(exported) else original_checkpoint
        configs = sorted(
            os.path.join(character_dir, name)
            for name in os.listdir(character_dir)
            if name.endswith((".yaml", ".yml"))
            and os.path.isfile(os.path.join(character_dir, name))
        )
        if len(configs) > 1:
            raise RequestError("Character contains more than one YAML configuration")
        config_path = configs[0] if configs else os.path.join(source_root, "configs", "base.yaml")
        if not os.path.isfile(config_path):
            raise RequestError("SVC5 base configuration is not installed", 503)
        sample_rate = configuration_sample_rate(config_path)
        singer_dir = os.path.join(character_dir, "singer")
        speaker_path = _single_file(
            singer_dir,
            lambda name: name.endswith(".spk.npy"),
            "SVC5 speaker embedding",
        )
        return ModelSpec(
            character=character,
            version=version,
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            speaker_path=speaker_path,
            checkpoint_revision=file_revision(checkpoint_path),
            config_revision=file_revision(config_path),
            speaker_revision=file_revision(speaker_path),
            sample_rate=sample_rate,
        )


@dataclass(frozen=True)
class GenerateRequest:
    input_name: str
    output_name: str
    session_id: Optional[str]
    request_id: Optional[str]
    spec: ModelSpec
    device: str
    pitch_shift: int
    cpu_bf16: bool


def parse_generate_request(payload: Any, resolver: ModelResolver) -> GenerateRequest:
    payload = require_object(payload, "Request body")
    inputs = require_object(payload.get("Inputs"), "Inputs")
    options = require_object(payload.get("Options"), "Options")
    if "GPU ID" not in payload:
        raise RequestError("GPU ID is required")
    try:
        device = normalize_device(payload["GPU ID"])
    except ValueError as exc:
        raise RequestError(str(exc)) from exc
    character = component(options.get("Character"), "Options.Character")
    return GenerateRequest(
        input_name=cache_key(inputs.get("User Audio"), "Inputs.User Audio"),
        output_name=cache_key(payload.get("Output File"), "Output File"),
        session_id=cache_key(payload.get("Session ID"), "Session ID", allow_none=True),
        request_id=request_id(payload.get("Request ID")),
        spec=resolver.resolve(character),
        device=device,
        pitch_shift=integer(options.get("Pitch Shift"), "Pitch Shift", -36, 36),
        cpu_bf16=(
            device == "cpu" and hsc.model_cpu_bf16_enabled(ARCHITECTURE_NAME)
        ),
    )


def create_app(cache, runtime=None, character_dir_func=None, version_resolver=None) -> Flask:
    application = Flask(__name__)
    register_methods(application, cache, runtime, character_dir_func, version_resolver)
    return application


def register_methods(
    application,
    cache,
    runtime=None,
    character_dir_func=None,
    version_resolver=None,
):
    runtime = runtime or build_runtime()
    resolver = ModelResolver(character_dir_func, version_resolver)

    @application.errorhandler(RequestError)
    def handle_request_error(error):
        return json_response(str(error), error.status_code, error_type="bad_request")

    @application.errorhandler(GenerationCancelled)
    def handle_cancelled(error):
        return json_response(str(error), 409, error_type="cancelled", cancelled=True)

    @application.errorhandler(Exception)
    def handle_unexpected_error(error):
        if isinstance(error, HTTPException):
            return json_response(error.description, error.code, error_type="http_error")
        application.logger.exception("SVC5 request failed")
        return json_response(
            "SVC5 inference failed: {}".format(error),
            500,
            error_type="inference_error",
        )

    @application.route("/generate", methods=["POST"])
    def generate():
        parsed = parse_generate_request(request.get_json(silent=True), resolver)
        try:
            source_audio, sample_rate = cache.read_audio_from_cache(
                Stage.PREPROCESSED, parsed.session_id, parsed.input_name
            )
        except Exception as exc:
            raise RequestError(
                "Input audio {!r} was not found in the preprocess cache".format(
                    parsed.input_name
                ),
                404,
            ) from exc
        output, output_rate = runtime.generate(
            parsed.spec,
            parsed.input_name,
            source_audio,
            sample_rate,
            parsed.device,
            parsed.pitch_shift,
            parsed.cpu_bf16,
            parsed.request_id,
        )
        runtime.commit_if_active(
            parsed.request_id,
            lambda: cache.save_audio_to_cache(
                Stage.OUTPUT,
                parsed.session_id,
                parsed.output_name,
                output,
                output_rate,
            ),
        )
        return json_response(
            output_file=parsed.output_name,
            device=parsed.device,
            request_id=parsed.request_id,
        )

    @application.route("/cancel", methods=["POST"])
    def cancel():
        payload = require_object(request.get_json(silent=True), "Request body")
        values = payload.get("Request IDs")
        if not isinstance(values, list) or not values:
            raise RequestError("Request IDs must be a non-empty array")
        ids = [request_id(value, "Request IDs[]", allow_none=False) for value in values]
        result = runtime.cancel(ids)
        return json_response(**result)

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
        spec = resolver.resolve(character)
        try:
            device = normalize_device(payload.get("GPU ID", options.get("GPU ID", "")))
        except ValueError as exc:
            raise RequestError(str(exc)) from exc
        workers = options.get("Workers", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise RequestError("Workers must be a positive integer")
        warmed = runtime.warm(spec, device, workers)
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
        status = "busy" if result["busy_models"] else "unloaded"
        return json_response(
            None,
            409 if result["busy_models"] else 200,
            status=status,
            result=result,
            runtime=runtime.state(),
        )

    return application


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Native So-VITS-SVC 5 service")
    parser.add_argument(
        "--cache_implementation",
        "--cache-implementation",
        default="file",
        choices=hsc.cache_implementation_map.keys(),
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=int(os.environ.get("HAY_SAY_SVC5_CPU_WORKERS", "4")),
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=int(os.environ.get("HAY_SAY_SVC5_GPU_WORKERS", "1")),
    )
    parser.add_argument(
        "--startup-concurrency",
        type=int,
        default=int(os.environ.get("HAY_SAY_SVC5_STARTUP_CONCURRENCY", "4")),
    )
    parser.add_argument(
        "--idle-ttl-seconds",
        type=float,
        default=float(os.environ.get("HAY_SAY_MODEL_IDLE_TTL_SECONDS", "1800")),
    )
    return parser.parse_args(argv)


app = Flask(__name__)


def main(argv=None):
    args = parse_arguments(argv)
    cache = hsc.select_cache_implementation(args.cache_implementation)
    runtime = build_runtime(
        args.cpu_workers,
        args.gpu_workers,
        args.startup_concurrency,
        args.idle_ttl_seconds,
    )
    atexit.register(runtime.close)
    register_methods(app, cache, runtime)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

"""Typed loading and validation for the native runtime registry."""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = Path(__file__).with_name("runtimes.json")


class ConfigError(ValueError):
    """Raised when the runtime registry is invalid."""


@dataclass(frozen=True)
class ManagerConfig:
    state_dir: Path
    log_dir: Path
    api_host: str = "127.0.0.1"
    api_port: int = 6588
    request_timeout_seconds: float = 1.0
    start_grace_seconds: float = 30.0
    stop_timeout_seconds: float = 15.0
    collect_gpu_memory: bool = True


@dataclass(frozen=True)
class RuntimeSpec:
    id: str
    label: str
    port: int
    command: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    device: str = "auto"
    enabled: bool = True
    runtime_endpoint: str | None = "/runtime"
    health_endpoint: str | None = "/gpu-info"


@dataclass(frozen=True)
class RuntimeConfig:
    manager: ManagerConfig
    runtimes: Mapping[str, RuntimeSpec]

    def runtime(self, runtime_id: str) -> RuntimeSpec:
        try:
            return self.runtimes[runtime_id]
        except KeyError as exc:
            raise KeyError(f"Unknown runtime: {runtime_id}") from exc


_BRACED_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
_SIMPLE_ENV = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_RUNTIME_ID = re.compile(r"^[a-z][a-z0-9_]*$")


def _expand_string(value: str, environ: Mapping[str, str]) -> str:
    sentinel = "\0HAY_SAY_DOLLAR\0"
    expanded = value.replace("$$", sentinel)

    def replace_braced(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        current = environ.get(name)
        if current not in (None, ""):
            return current
        if default is not None:
            return default
        raise ConfigError(f"Environment variable {name} is required by the runtime config")

    def replace_simple(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in environ:
            raise ConfigError(f"Environment variable {name} is required by the runtime config")
        return environ[name]

    # A small fixed-point loop permits defaults and environment values to refer
    # to another variable without turning configuration loading into evaluation.
    for _ in range(10):
        previous = expanded
        expanded = _BRACED_ENV.sub(replace_braced, expanded)
        expanded = _SIMPLE_ENV.sub(replace_simple, expanded)
        if expanded == previous:
            break
    else:
        raise ConfigError(f"Environment expansion did not converge for {value!r}")
    return expanded.replace(sentinel, "$")


def _expand(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _expand_string(value, environ)
    if isinstance(value, list):
        return [_expand(item, environ) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, environ) for key, item in value.items()}
    return value


def _parse_json_env(name: str, environ: Mapping[str, str], expected_type: type) -> Any | None:
    raw = environ.get(name)
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{name} must contain valid JSON: {exc}") from exc
    if not isinstance(parsed, expected_type):
        raise ConfigError(f"{name} must contain a JSON {expected_type.__name__}")
    return parsed


def _apply_environment_overrides(raw: dict[str, Any], environ: Mapping[str, str]) -> None:
    manager = raw.setdefault("manager", {})
    manager_overrides = {
        "HAY_SAY_RUNTIME_STATE_DIR": "state_dir",
        "HAY_SAY_RUNTIME_LOG_DIR": "log_dir",
        "HAY_SAY_RUNTIME_API_HOST": "api_host",
        "HAY_SAY_RUNTIME_API_PORT": "api_port",
        "HAY_SAY_RUNTIME_REQUEST_TIMEOUT": "request_timeout_seconds",
        "HAY_SAY_RUNTIME_START_GRACE": "start_grace_seconds",
        "HAY_SAY_RUNTIME_STOP_TIMEOUT": "stop_timeout_seconds",
        "HAY_SAY_RUNTIME_COLLECT_GPU_MEMORY": "collect_gpu_memory",
    }
    for env_name, field in manager_overrides.items():
        if env_name in environ:
            manager[field] = environ[env_name]

    for runtime in raw.get("runtimes", []):
        runtime_id = runtime.get("id")
        if not isinstance(runtime_id, str):
            continue
        normalized_id = re.sub(r"[^A-Z0-9]+", "_", runtime_id.upper())
        prefix = f"HAY_SAY_RUNTIME_{normalized_id}_"
        for suffix, field in {
            "PORT": "port",
            "CWD": "cwd",
            "DEVICE": "device",
            "ENABLED": "enabled",
            "RUNTIME_ENDPOINT": "runtime_endpoint",
            "HEALTH_ENDPOINT": "health_endpoint",
        }.items():
            if prefix + suffix in environ:
                runtime[field] = environ[prefix + suffix]

        command = _parse_json_env(prefix + "COMMAND_JSON", environ, list)
        if command is not None:
            runtime["command"] = command
        env_override = _parse_json_env(prefix + "ENV_JSON", environ, dict)
        if env_override is not None:
            runtime.setdefault("env", {}).update(env_override)


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"{field} must be a boolean")


def _as_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be an integer") from exc
    return result


def _as_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be a number") from exc
    if result <= 0:
        raise ConfigError(f"{field} must be greater than zero")
    return result


def _optional_endpoint(value: Any, field: str) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or not value.startswith("/") or value.startswith("//"):
        raise ConfigError(f"{field} must be null or an absolute HTTP path")
    return value


def _build_config(raw: Mapping[str, Any]) -> RuntimeConfig:
    if raw.get("version") != 1:
        raise ConfigError("Runtime config version must be 1")
    manager_raw = raw.get("manager")
    if not isinstance(manager_raw, dict):
        raise ConfigError("manager must be a JSON object")

    api_host = manager_raw.get("api_host", "127.0.0.1")
    if api_host != "127.0.0.1":
        raise ConfigError("manager.api_host must be 127.0.0.1")
    api_port = _as_int(manager_raw.get("api_port", 6588), "manager.api_port")
    if not 1 <= api_port <= 65535:
        raise ConfigError("manager.api_port must be between 1 and 65535")

    try:
        state_dir = Path(manager_raw["state_dir"]).expanduser()
        log_dir = Path(manager_raw["log_dir"]).expanduser()
    except (KeyError, TypeError) as exc:
        raise ConfigError("manager.state_dir and manager.log_dir are required strings") from exc
    if not state_dir.is_absolute() or not log_dir.is_absolute():
        raise ConfigError("manager.state_dir and manager.log_dir must be absolute paths")

    manager = ManagerConfig(
        state_dir=state_dir,
        log_dir=log_dir,
        api_host=api_host,
        api_port=api_port,
        request_timeout_seconds=_as_float(
            manager_raw.get("request_timeout_seconds", 1.0), "manager.request_timeout_seconds"
        ),
        start_grace_seconds=_as_float(
            manager_raw.get("start_grace_seconds", 30.0), "manager.start_grace_seconds"
        ),
        stop_timeout_seconds=_as_float(
            manager_raw.get("stop_timeout_seconds", 15.0), "manager.stop_timeout_seconds"
        ),
        collect_gpu_memory=_as_bool(
            manager_raw.get("collect_gpu_memory", True), "manager.collect_gpu_memory"
        ),
    )

    runtime_rows = raw.get("runtimes")
    if not isinstance(runtime_rows, list) or not runtime_rows:
        raise ConfigError("runtimes must be a non-empty JSON array")
    runtimes: dict[str, RuntimeSpec] = {}
    ports: set[int] = set()
    for index, row in enumerate(runtime_rows):
        field_prefix = f"runtimes[{index}]"
        if not isinstance(row, dict):
            raise ConfigError(f"{field_prefix} must be a JSON object")
        runtime_id = row.get("id")
        if not isinstance(runtime_id, str) or not _RUNTIME_ID.fullmatch(runtime_id):
            raise ConfigError(f"{field_prefix}.id must be a lowercase snake_case identifier")
        if runtime_id in runtimes:
            raise ConfigError(f"Duplicate runtime id: {runtime_id}")
        label = row.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ConfigError(f"{field_prefix}.label must be a non-empty string")
        port = _as_int(row.get("port"), f"{field_prefix}.port")
        if not 1 <= port <= 65535:
            raise ConfigError(f"{field_prefix}.port must be between 1 and 65535")
        if port in ports:
            raise ConfigError(f"Duplicate runtime port: {port}")
        ports.add(port)

        command = row.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(argument, str) or not argument or "\0" in argument for argument in command)
        ):
            raise ConfigError(f"{field_prefix}.command must be a non-empty array of non-empty strings")
        if not Path(command[0]).is_absolute():
            raise ConfigError(f"{field_prefix}.command[0] must be an absolute executable path")
        cwd_raw = row.get("cwd")
        if not isinstance(cwd_raw, str) or not Path(cwd_raw).is_absolute():
            raise ConfigError(f"{field_prefix}.cwd must be an absolute path")
        env = row.get("env", {})
        if not isinstance(env, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()
        ):
            raise ConfigError(f"{field_prefix}.env must map strings to strings")

        spec = RuntimeSpec(
            id=runtime_id,
            label=label,
            port=port,
            command=tuple(command),
            cwd=Path(cwd_raw),
            env=dict(env),
            device=str(row.get("device", "auto")),
            enabled=_as_bool(row.get("enabled", True), f"{field_prefix}.enabled"),
            runtime_endpoint=_optional_endpoint(
                row.get("runtime_endpoint", "/runtime"), f"{field_prefix}.runtime_endpoint"
            ),
            health_endpoint=_optional_endpoint(
                row.get("health_endpoint", "/gpu-info"), f"{field_prefix}.health_endpoint"
            ),
        )
        runtimes[runtime_id] = spec
    return RuntimeConfig(manager=manager, runtimes=runtimes)


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    """Load the JSON registry, apply environment overrides, expand variables, and validate it."""

    env = dict(os.environ if environ is None else environ)
    configured_path = path or env.get("HAY_SAY_RUNTIME_CONFIG") or DEFAULT_CONFIG_PATH
    try:
        with Path(configured_path).expanduser().open("r", encoding="utf-8") as config_file:
            raw = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Unable to load runtime config {configured_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("The runtime config root must be a JSON object")
    mutable_raw = copy.deepcopy(raw)
    _apply_environment_overrides(mutable_raw, env)
    return _build_config(_expand(mutable_raw, env))

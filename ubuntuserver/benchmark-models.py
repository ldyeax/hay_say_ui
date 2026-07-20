#!/usr/bin/env python3
"""Benchmark every installed native model with one reproducible input."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import socket
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import generator
import hay_say_common as hsc
import runtime_client


def _load_smoke_models():
    """Load the sibling script despite the hyphen in its filename."""
    module_name = "hay_say_native_smoke_models"
    module_path = Path(__file__).with_name("smoke-models.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load smoke helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


smoke_models = _load_smoke_models()
SMOKE_SPECS = smoke_models.SMOKE_SPECS
DEFAULT_TEXT = smoke_models.DEFAULT_TEXT


@dataclass(frozen=True)
class BenchmarkMode:
    name: str
    cpu_bf16_autocast: bool
    uses_gpu: bool

    def device(self, gpu_id: int) -> int | str:
        return gpu_id if self.uses_gpu else ""


BENCHMARK_MODES = {
    "cpu-fp32": BenchmarkMode("cpu-fp32", cpu_bf16_autocast=False, uses_gpu=False),
    "cpu-bf16": BenchmarkMode("cpu-bf16", cpu_bf16_autocast=True, uses_gpu=False),
    "gpu-native": BenchmarkMode("gpu-native", cpu_bf16_autocast=False, uses_gpu=True),
}

SUMMARY_KEYS = ("median", "min", "max", "mean", "stdev")


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _linux_cpu_metadata(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> dict[str, Any]:
    try:
        contents = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"cpu_model": None, "cpu_flags": [], "amx": {}}
    fields = {}
    for line in contents.split("\n\n", 1)[0].splitlines():
        name, separator, value = line.partition(":")
        if separator:
            fields[name.strip().lower()] = value.strip()
    flags = sorted(set(fields.get("flags", "").split()))
    return {
        "cpu_model": fields.get("model name"),
        "cpu_flags": flags,
        "amx": {
            "bf16": "amx_bf16" in flags,
            "int8": "amx_int8" in flags,
            "tile": "amx_tile" in flags,
        },
    }


def _host_metadata() -> dict[str, Any]:
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity_count = None
    return {
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "available_cpu_count": affinity_count,
        **_linux_cpu_metadata(),
    }


def _sha256_bytes(*values: bytes) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(len(value).to_bytes(8, byteorder="little", signed=False))
        digest.update(value)
    return digest.hexdigest()


def _input_metadata(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    reference_audio: str | None,
) -> dict[str, Any]:
    canonical_audio = np.ascontiguousarray(audio, dtype=np.dtype("<f4"))
    audio_bytes = canonical_audio.tobytes()
    sample_rate_bytes = int(sample_rate).to_bytes(8, byteorder="little", signed=False)
    text_bytes = text.encode("utf-8")
    return {
        "sha256": _sha256_bytes(audio_bytes, sample_rate_bytes, text_bytes),
        "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
        "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
        "reference": str(Path(reference_audio).resolve()) if reference_audio else "synthesized-espeak",
        "sample_rate": int(sample_rate),
        "sample_count": int(canonical_audio.shape[0]),
        "duration_seconds": float(canonical_audio.shape[0] / sample_rate),
        "text": text,
    }


def _summary(values: Iterable[float]) -> dict[str, float] | None:
    samples = [float(value) for value in values]
    if not samples:
        return None
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _output_hash(runtime_id: str, mode: str, phase: str, index: int) -> str:
    return f"benchmark-{runtime_id}-{mode}-{phase}-{index}-{uuid.uuid4().hex}"


def _payload(
    runtime_id: str,
    mode: BenchmarkMode,
    host: str,
    session_id: str,
    input_hash: str,
    output_hash: str,
    text: str,
    gpu_id: int,
) -> dict[str, Any]:
    spec = SMOKE_SPECS[runtime_id]
    return smoke_models._payload(
        spec,
        host,
        session_id,
        input_hash,
        output_hash,
        text,
        mode.device(gpu_id),
    )


def _backend_cpu_bf16_policy(
    runtime_id: str,
    runtime_state: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(runtime_state, dict):
        raise RuntimeError(
            f"{runtime_id} did not return runtime telemetry; update and restart the runtime"
        )
    policy = runtime_state.get("cpu_bf16")
    if not isinstance(policy, dict):
        raise RuntimeError(
            f"{runtime_id} /runtime does not report cpu_bf16 policy; "
            "update and restart the runtime"
        )
    expected_variable = hsc.server_utility.MODEL_CPU_BF16_ENVIRONMENTS[runtime_id][0]
    if policy.get("environment_variable") != expected_variable:
        raise RuntimeError(
            f"{runtime_id} /runtime reported an unexpected CPU BF16 environment variable"
        )
    for field in ("default", "requested", "amx_available", "effective"):
        if not isinstance(policy.get(field), bool):
            raise RuntimeError(f"{runtime_id} /runtime cpu_bf16.{field} must be boolean")
    if policy["effective"] != (policy["requested"] and policy["amx_available"]):
        raise RuntimeError(f"{runtime_id} /runtime reported inconsistent CPU BF16 policy")
    return policy


def _validate_backend_precision(
    runtime_id: str,
    mode: BenchmarkMode,
    runtime_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if mode.uses_gpu:
        return None
    policy = _backend_cpu_bf16_policy(runtime_id, runtime_state)
    if policy["effective"] == mode.cpu_bf16_autocast:
        return policy
    variable = policy["environment_variable"]
    expected = "1" if mode.cpu_bf16_autocast else "0"
    raise RuntimeError(
        f"{mode.name} requires effective {variable}={expected}; the running runtime reports "
        f"requested={int(policy['requested'])}, amx_available={int(policy['amx_available'])}, "
        f"effective={int(policy['effective'])}. Update the backend environment and restart "
        "this runtime before benchmarking"
    )


def _request_sample(
    runtime_id: str,
    mode: BenchmarkMode,
    cache: type,
    host: str,
    port: int,
    session_id: str,
    input_hash: str,
    text: str,
    gpu_id: int,
    timeout: float,
    phase: str,
    index: int,
) -> dict[str, Any]:
    output_hash = _output_hash(runtime_id, mode.name, phase, index)
    payload = _payload(
        runtime_id,
        mode,
        host,
        session_id,
        input_hash,
        output_hash,
        text,
        gpu_id,
    )
    with runtime_client.generation_lock(runtime_id):
        request_started_at = time.monotonic()
        generator.send_payload(payload, host, port, timeout=timeout)
        request_wall_seconds = time.monotonic() - request_started_at
        duration, sample_rate, peak = smoke_models._validate_output(cache, session_id, output_hash)
    return {
        "phase": phase,
        "index": index,
        "output_hash": output_hash,
        "request_wall_seconds": request_wall_seconds,
        "output_duration_seconds": duration,
        "real_time_factor": request_wall_seconds / duration,
        "sample_rate": sample_rate,
        "peak": peak,
    }


def _empty_result(
    runtime_id: str,
    mode: str,
    initial_status: str | None,
    started_by_benchmark: bool,
    runtime_start_seconds: float | None,
    warmups: int,
    runs: int,
) -> dict[str, Any]:
    return {
        "runtime_id": runtime_id,
        "character": SMOKE_SPECS[runtime_id].character,
        "mode": mode,
        "status": "error",
        "runtime_initial_status": initial_status,
        "runtime_started_by_benchmark": started_by_benchmark,
        "runtime_start_seconds": runtime_start_seconds,
        "warmups_requested": warmups,
        "warmups_completed": 0,
        "runs_requested": runs,
        "runs_completed": 0,
        "samples": [],
        "request_wall_seconds": None,
        "output_duration_seconds": None,
        "real_time_factor": None,
        "rtf": None,
        "backend_cpu_bf16": None,
        "errors": [],
    }


def benchmark_mode(
    runtime_id: str,
    mode: BenchmarkMode,
    cache: type,
    host: str,
    port: int,
    session_id: str,
    input_hash: str,
    text: str,
    gpu_id: int,
    timeout: float,
    warmups: int,
    runs: int,
    initial_status: str | None,
    started_by_benchmark: bool,
    runtime_start_seconds: float,
) -> dict[str, Any]:
    result = _empty_result(
        runtime_id,
        mode.name,
        initial_status,
        started_by_benchmark,
        runtime_start_seconds,
        warmups,
        runs,
    )

    for phase, count in (("warmup", warmups), ("measured", runs)):
        for index in range(count):
            try:
                sample = _request_sample(
                    runtime_id,
                    mode,
                    cache,
                    host,
                    port,
                    session_id,
                    input_hash,
                    text,
                    gpu_id,
                    timeout,
                    phase,
                    index,
                )
            except Exception as error:
                result["errors"].append(f"{phase} {index + 1}: {error}")
                continue
            if phase == "warmup":
                result["warmups_completed"] += 1
            else:
                result["runs_completed"] += 1
                result["samples"].append(sample)

    samples = result["samples"]
    result["request_wall_seconds"] = _summary(item["request_wall_seconds"] for item in samples)
    result["output_duration_seconds"] = _summary(item["output_duration_seconds"] for item in samples)
    result["real_time_factor"] = _summary(item["real_time_factor"] for item in samples)
    if result["real_time_factor"] is not None:
        result["rtf"] = result["real_time_factor"]["mean"]
    result["status"] = "ok" if result["runs_completed"] == runs and not result["errors"] else "error"
    return result


def _runtime_error_results(
    runtime_id: str,
    modes: list[str],
    message: str,
    initial_status: str | None,
    started_by_benchmark: bool,
    runtime_start_seconds: float | None,
    warmups: int,
    runs: int,
) -> list[dict[str, Any]]:
    results = []
    for mode in modes:
        result = _empty_result(
            runtime_id,
            mode,
            initial_status,
            started_by_benchmark,
            runtime_start_seconds,
            warmups,
            runs,
        )
        result["errors"].append(message)
        results.append(result)
    return results


def benchmark_runtime(
    runtime_id: str,
    modes: list[str],
    cache: type,
    session_id: str,
    input_hash: str,
    text: str,
    gpu_id: int,
    timeout: float,
    warmups: int,
    runs: int,
) -> list[dict[str, Any]]:
    spec = SMOKE_SPECS[runtime_id]
    model_directory = smoke_models._model_directory(runtime_id, spec.character)
    if not model_directory.is_dir() or not any(model_directory.rglob("*")):
        return _runtime_error_results(
            runtime_id,
            modes,
            f"model is not installed: {model_directory}",
            None,
            False,
            None,
            warmups,
            runs,
        )

    initial_status: str | None = None
    started_by_benchmark = False
    runtime_start_seconds: float | None = None
    results: list[dict[str, Any]] = []
    try:
        initial_status = runtime_client.runtime_status(runtime_id).get("status")
        started_by_benchmark = initial_status not in runtime_client.RUNNING_STATES | {"starting"}
        runtime_started_at = time.monotonic()
        try:
            runtime_client.ensure_runtime_started(runtime_id, spec.port, timeout=120)
        finally:
            runtime_start_seconds = time.monotonic() - runtime_started_at
        host, port = runtime_client.service_endpoint(runtime_id, spec.port)
        runtime_state = runtime_client.service_runtime_state(
            host,
            port,
            timeout=min(5.0, max(0.1, timeout)),
        )

        for mode_name in modes:
            print(f"[benchmark] {runtime_id} / {mode_name}", flush=True)
            policy = None
            try:
                policy = _validate_backend_precision(
                    runtime_id,
                    BENCHMARK_MODES[mode_name],
                    runtime_state,
                )
                result = benchmark_mode(
                    runtime_id,
                    BENCHMARK_MODES[mode_name],
                    cache,
                    host,
                    port,
                    session_id,
                    input_hash,
                    text,
                    gpu_id,
                    timeout,
                    warmups,
                    runs,
                    initial_status,
                    started_by_benchmark,
                    runtime_start_seconds,
                )
            except Exception as error:
                result = _runtime_error_results(
                    runtime_id,
                    [mode_name],
                    f"benchmark setup failed: {error}",
                    initial_status,
                    started_by_benchmark,
                    runtime_start_seconds,
                    warmups,
                    runs,
                )[0]
            if policy is None and isinstance(runtime_state, dict):
                try:
                    policy = _backend_cpu_bf16_policy(runtime_id, runtime_state)
                except RuntimeError:
                    pass
            result["backend_cpu_bf16"] = policy
            results.append(result)
    except Exception as error:
        results = _runtime_error_results(
            runtime_id,
            modes,
            f"runtime start failed: {error}",
            initial_status,
            started_by_benchmark,
            runtime_start_seconds,
            warmups,
            runs,
        )
    finally:
        if started_by_benchmark:
            try:
                runtime_client.runtime_action(runtime_id, "stop")
            except Exception as error:
                if not results:
                    results = _runtime_error_results(
                        runtime_id,
                        modes,
                        f"runtime stop failed: {error}",
                        initial_status,
                        started_by_benchmark,
                        runtime_start_seconds,
                        warmups,
                        runs,
                    )
                else:
                    for result in results:
                        result["status"] = "error"
                        result["errors"].append(f"runtime stop failed: {error}")
    return results


def run_benchmarks(
    runtimes: list[str],
    modes: list[str],
    cache: type,
    session_id: str,
    input_hash: str,
    text: str,
    gpu_id: int,
    timeout: float,
    warmups: int,
    runs: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for runtime_id in runtimes:
        try:
            results.extend(
                benchmark_runtime(
                    runtime_id,
                    modes,
                    cache,
                    session_id,
                    input_hash,
                    text,
                    gpu_id,
                    timeout,
                    warmups,
                    runs,
                )
            )
        except Exception as error:
            results.extend(
                _runtime_error_results(
                    runtime_id,
                    modes,
                    f"unexpected benchmark failure: {error}",
                    None,
                    False,
                    None,
                    warmups,
                    runs,
                )
            )
    return results


def _format_number(value: Any) -> str:
    return "" if value is None else f"{float(value):.9f}"


def _csv_rows(report: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for result in report["results"]:
        row = {
            "created_at": report["created_at"],
            "hostname": report["host"]["hostname"],
            "input_sha256": report["input"]["sha256"],
            "runtime_id": result["runtime_id"],
            "character": result["character"],
            "mode": result["mode"],
            "status": result["status"],
            "runtime_initial_status": result["runtime_initial_status"] or "",
            "runtime_started_by_benchmark": result["runtime_started_by_benchmark"],
            "runtime_start_seconds": _format_number(result["runtime_start_seconds"]),
            "warmups_requested": result["warmups_requested"],
            "warmups_completed": result["warmups_completed"],
            "runs_requested": result["runs_requested"],
            "runs_completed": result["runs_completed"],
            "rtf": _format_number(result["rtf"]),
            "cpu_bf16_requested": (
                result["backend_cpu_bf16"]["requested"]
                if result["backend_cpu_bf16"] is not None else ""
            ),
            "cpu_bf16_amx_available": (
                result["backend_cpu_bf16"]["amx_available"]
                if result["backend_cpu_bf16"] is not None else ""
            ),
            "cpu_bf16_effective": (
                result["backend_cpu_bf16"]["effective"]
                if result["backend_cpu_bf16"] is not None else ""
            ),
            "errors": " | ".join(result["errors"]),
        }
        for source, prefix in (
            (result["request_wall_seconds"], "request_wall_seconds"),
            (result["output_duration_seconds"], "output_duration_seconds"),
            (result["real_time_factor"], "real_time_factor"),
        ):
            for key in SUMMARY_KEYS:
                row[f"{prefix}_{key}"] = _format_number(source[key] if source else None)
        yield row


def write_report(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")

    rows = list(_csv_rows(report))
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        fieldnames = list(rows[0]) if rows else ["created_at", "hostname", "input_sha256"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be one or greater")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="append",
        choices=SMOKE_SPECS,
        help="Runtime to benchmark; repeat for multiple runtimes (default: all)",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=BENCHMARK_MODES,
        help="Mode to benchmark; repeat for multiple modes (default: all)",
    )
    parser.add_argument("--warmups", type=_non_negative_int, default=1)
    parser.add_argument("--runs", type=_positive_int, default=3)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--reference-audio", help="Use this audio instead of synthesized speech")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(f"hay-say-benchmark-{timestamp}.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path(f"hay-say-benchmark-{timestamp}.csv"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_runtimes = args.runtime or list(SMOKE_SPECS)
    selected_modes = args.mode or list(BENCHMARK_MODES)
    session_id = f"native-benchmark-{uuid.uuid4().hex}"
    input_hash = "consistent-reference"
    cache = hsc.select_cache_implementation("file")
    audio, sample_rate = smoke_models._reference_audio(args.reference_audio)
    smoke_models._save_preprocessed_input(cache, session_id, input_hash, audio, sample_rate)
    created_at = _utc_now()

    try:
        results = run_benchmarks(
            selected_runtimes,
            selected_modes,
            cache,
            session_id,
            input_hash,
            args.text,
            args.gpu_id,
            args.timeout,
            args.warmups,
            args.runs,
        )
    finally:
        if not args.keep_cache:
            cache.delete_session_data(session_id)

    report = {
        "schema_version": 2,
        "created_at": created_at,
        "host": _host_metadata(),
        "input": _input_metadata(audio, sample_rate, args.text, args.reference_audio),
        "configuration": {
            "runtimes": selected_runtimes,
            "modes": selected_modes,
            "warmups": args.warmups,
            "runs": args.runs,
            "gpu_id": args.gpu_id,
            "timeout_seconds": args.timeout,
        },
        "results": results,
    }
    write_report(report, args.json_output, args.csv_output)

    failures = sum(result["status"] != "ok" for result in results)
    print(f"[benchmark] JSON: {args.json_output}", flush=True)
    print(f"[benchmark] CSV:  {args.csv_output}", flush=True)
    print(f"[benchmark] completed with {failures} failed mode(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

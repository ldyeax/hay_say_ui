import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def load_benchmark_models():
    module_name = "hay_say_benchmark_models"
    module_path = REPOSITORY / "ubuntuserver" / "benchmark-models.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def backend_state(*, requested, amx_available):
    return {
        "cpu_bf16": {
            "environment_variable": "HAY_SAY_SVC3_CPU_BF16_AUTOCAST",
            "default": False,
            "requested": requested,
            "source": "environment",
            "amx_available": amx_available,
            "effective": requested and amx_available,
        }
    }


def test_cli_defaults_to_all_modes_with_one_warmup_and_three_runs():
    benchmark = load_benchmark_models()

    args = benchmark.parse_args([])

    assert args.runtime is None
    assert args.mode is None
    assert args.warmups == 1
    assert args.runs == 3
    assert list(benchmark.BENCHMARK_MODES) == ["cpu-fp32", "cpu-bf16", "gpu-native"]


def test_linux_cpu_metadata_records_amx_capabilities(tmp_path):
    benchmark = load_benchmark_models()
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor : 0\n"
        "model name : Example Xeon\n"
        "flags : avx512f amx_tile amx_bf16 amx_int8\n\n"
        "processor : 1\n",
        encoding="utf-8",
    )

    metadata = benchmark._linux_cpu_metadata(cpuinfo)

    assert metadata["cpu_model"] == "Example Xeon"
    assert metadata["amx"] == {"bf16": True, "int8": True, "tile": True}
    assert metadata["cpu_flags"] == ["amx_bf16", "amx_int8", "amx_tile", "avx512f"]


def test_payload_sets_only_the_mode_device_and_has_no_precision_override(monkeypatch):
    benchmark = load_benchmark_models()

    def fake_payload(_spec, _host, _session, _input, output, _text, device):
        return {"Options": {"Character": "Fluttershy"}, "Output File": output, "GPU ID": device}

    monkeypatch.setattr(benchmark.smoke_models, "_payload", fake_payload)

    fp32 = benchmark._payload(
        "so_vits_svc_3",
        benchmark.BENCHMARK_MODES["cpu-fp32"],
        "127.0.0.1",
        "session",
        "input",
        "fp32-output",
        "text",
        7,
    )
    bf16 = benchmark._payload(
        "so_vits_svc_3",
        benchmark.BENCHMARK_MODES["cpu-bf16"],
        "127.0.0.1",
        "session",
        "input",
        "bf16-output",
        "text",
        7,
    )
    gpu = benchmark._payload(
        "so_vits_svc_3",
        benchmark.BENCHMARK_MODES["gpu-native"],
        "127.0.0.1",
        "session",
        "input",
        "gpu-output",
        "text",
        7,
    )

    assert fp32["GPU ID"] == ""
    assert bf16["GPU ID"] == ""
    assert gpu["GPU ID"] == 7
    assert all("CPU BF16 Autocast" not in payload["Options"] for payload in (fp32, bf16, gpu))


def test_benchmark_rejects_a_cpu_mode_that_does_not_match_running_backend_policy(monkeypatch):
    benchmark = load_benchmark_models()
    monkeypatch.setattr(
        benchmark.hsc,
        "model_cpu_bf16_enabled",
        lambda _runtime_id: (_ for _ in ()).throw(AssertionError("caller env must not be used")),
    )
    state = backend_state(requested=True, amx_available=True)

    benchmark._validate_backend_precision(
        "so_vits_svc_3", benchmark.BENCHMARK_MODES["cpu-bf16"], state
    )
    with pytest.raises(RuntimeError, match="HAY_SAY_SVC3_CPU_BF16_AUTOCAST=0"):
        benchmark._validate_backend_precision(
            "so_vits_svc_3", benchmark.BENCHMARK_MODES["cpu-fp32"], state
        )


def test_benchmark_rejects_requested_bf16_when_running_backend_lacks_amx():
    benchmark = load_benchmark_models()
    state = backend_state(requested=True, amx_available=False)

    with pytest.raises(RuntimeError, match="amx_available=0"):
        benchmark._validate_backend_precision(
            "so_vits_svc_3", benchmark.BENCHMARK_MODES["cpu-bf16"], state
        )


def test_output_ids_are_unique_for_every_request():
    benchmark = load_benchmark_models()

    first = benchmark._output_hash("so_vits_svc_3", "cpu-fp32", "measured", 0)
    second = benchmark._output_hash("so_vits_svc_3", "cpu-fp32", "measured", 0)

    assert first != second
    assert first.startswith("benchmark-so_vits_svc_3-cpu-fp32-measured-0-")


def test_summary_reports_request_distribution():
    benchmark = load_benchmark_models()

    summary = benchmark._summary([1.0, 2.0, 4.0])

    assert summary["median"] == 2.0
    assert summary["min"] == 1.0
    assert summary["max"] == 4.0
    assert summary["mean"] == pytest.approx(7 / 3)
    assert summary["stdev"] == pytest.approx(math.sqrt(7 / 3))
    assert benchmark._summary([]) is None


def test_benchmark_mode_continues_after_an_individual_request_error(monkeypatch):
    benchmark = load_benchmark_models()

    def fake_request(*args):
        phase = args[-2]
        index = args[-1]
        if phase == "measured" and index == 1:
            raise RuntimeError("unsupported precision")
        wall = float(index + 1)
        return {
            "phase": phase,
            "index": index,
            "output_hash": f"output-{phase}-{index}",
            "request_wall_seconds": wall,
            "output_duration_seconds": 2.0,
            "real_time_factor": wall / 2.0,
            "sample_rate": 32000,
            "peak": 0.5,
        }

    monkeypatch.setattr(benchmark, "_request_sample", fake_request)
    result = benchmark.benchmark_mode(
        "so_vits_svc_3",
        benchmark.BENCHMARK_MODES["cpu-bf16"],
        object,
        "127.0.0.1",
        6575,
        "session",
        "input",
        "text",
        0,
        30.0,
        1,
        3,
        "ready-cold",
        False,
        0.25,
    )

    assert result["status"] == "error"
    assert result["warmups_completed"] == 1
    assert result["runs_completed"] == 2
    assert len(result["samples"]) == 2
    assert result["request_wall_seconds"]["mean"] == 2.0
    assert result["real_time_factor"]["median"] == 1.0
    assert result["rtf"] == 1.0
    assert result["errors"] == ["measured 2: unsupported precision"]


@pytest.mark.parametrize(
    ("initial_status", "expected_stop"),
    [("ready-cold", False), ("warm-idle", False), ("stopped", True)],
)
def test_benchmark_stops_only_a_runtime_it_started(
    monkeypatch,
    tmp_path,
    initial_status,
    expected_stop,
):
    benchmark = load_benchmark_models()
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    (model_directory / "weights.pth").write_bytes(b"weights")
    actions = []

    monkeypatch.setattr(benchmark.smoke_models, "_model_directory", lambda *_args: model_directory)
    monkeypatch.setattr(
        benchmark.runtime_client,
        "runtime_status",
        lambda _runtime_id: {"status": initial_status},
    )
    monkeypatch.setattr(benchmark.runtime_client, "ensure_runtime_started", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        benchmark.runtime_client,
        "service_endpoint",
        lambda _runtime_id, port: ("127.0.0.1", port),
    )
    monkeypatch.setattr(
        benchmark.runtime_client,
        "service_runtime_state",
        lambda *_args, **_kwargs: backend_state(requested=False, amx_available=True),
    )
    monkeypatch.setattr(
        benchmark.runtime_client,
        "runtime_action",
        lambda runtime_id, action: actions.append((runtime_id, action)),
    )

    def fake_benchmark_mode(*args):
        result = benchmark._empty_result(
            "so_vits_svc_3",
            "cpu-fp32",
            initial_status,
            initial_status == "stopped",
            0.1,
            0,
            1,
        )
        result["status"] = "ok"
        result["runs_completed"] = 1
        return result

    monkeypatch.setattr(benchmark, "benchmark_mode", fake_benchmark_mode)

    results = benchmark.benchmark_runtime(
        "so_vits_svc_3",
        ["cpu-fp32"],
        object,
        "session",
        "input",
        "text",
        0,
        30.0,
        0,
        1,
    )

    assert results[0]["status"] == "ok"
    assert actions == ([("so_vits_svc_3", "stop")] if expected_stop else [])


def test_json_and_csv_include_host_input_sha_and_timing_summaries(tmp_path):
    benchmark = load_benchmark_models()
    result = benchmark._empty_result(
        "so_vits_svc_3",
        "cpu-fp32",
        "stopped",
        True,
        1.25,
        1,
        3,
    )
    result.update(
        {
            "status": "ok",
            "warmups_completed": 1,
            "runs_completed": 3,
            "request_wall_seconds": benchmark._summary([1.0, 2.0, 3.0]),
            "output_duration_seconds": benchmark._summary([2.0, 2.0, 2.0]),
            "real_time_factor": benchmark._summary([0.5, 1.0, 1.5]),
            "rtf": 1.0,
        }
    )
    report = {
        "schema_version": 1,
        "created_at": "2026-07-15T00:00:00+00:00",
        "host": {"hostname": "benchmark-host"},
        "input": {"sha256": "abc123"},
        "results": [result],
    }
    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"

    benchmark.write_report(report, json_path, csv_path)

    decoded = json.loads(json_path.read_text(encoding="utf-8"))
    assert decoded["host"]["hostname"] == "benchmark-host"
    assert decoded["input"]["sha256"] == "abc123"
    assert decoded["results"][0]["request_wall_seconds"]["median"] == 2.0
    with csv_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["hostname"] == "benchmark-host"
    assert rows[0]["input_sha256"] == "abc123"
    assert rows[0]["runtime_start_seconds"] == "1.250000000"
    assert rows[0]["request_wall_seconds_median"] == "2.000000000"
    assert rows[0]["real_time_factor_mean"] == "1.000000000"
    assert rows[0]["rtf"] == "1.000000000"

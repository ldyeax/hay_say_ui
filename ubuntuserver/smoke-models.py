#!/usr/bin/env python3
"""Run end-to-end generation checks against installed native model runtimes."""

from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import requests

import generator
import hay_say_common as hsc
import runtime_client
from hay_say_common.cache import Stage


DEFAULT_TEXT = "This is a short test of the native Fluttershy voice model."
STYLE_MODEL = (
    "Multi-speaker (40 epochs) (Mane 6, CMC, Princesses, Discord, Gilda, "
    "Zecora, Trixie, Starlight, Chrysalis, Tirek, Cozy Glow, Flim, Flam, "
    "and Shining Armor)"
)


@dataclass(frozen=True)
class SmokeSpec:
    port: int
    character: str
    options: Callable[[str, int], dict[str, Any]]
    needs_audio: bool = False


def _static_options(values: dict[str, Any]) -> Callable[[str, int], dict[str, Any]]:
    return lambda _host, _port: dict(values)


def _gpt_options(host: str, port: int) -> dict[str, Any]:
    response = requests.get(
        f"http://{host}:{port}/available-traits/Fluttershy",
        timeout=60,
    )
    response.raise_for_status()
    traits = response.json()
    if not isinstance(traits, list) or not traits:
        raise RuntimeError("GPT-SoVITS did not expose any Fluttershy traits")
    trait = next((item for item in traits if "neutral" in item.lower()), traits[0])
    return {
        "Character": "Fluttershy",
        "Reference Language": "English",
        "Target Language": "English",
        "Cutting Strategy": "Slice by English punctuation",
        "Top-K": 15,
        "Top-P": 1.0,
        "Temperature": 1.0,
        "Speed": 1.0,
        "Additional Reference Audios": [],
        "Trait": trait,
        "Reference Option": "Use Precomputed Embeddings",
    }


SMOKE_SPECS = {
    "controllable_talknet": SmokeSpec(
        6574,
        "Fluttershy",
        _static_options(
            {
                "Character": "Fluttershy",
                "Disable Reference Audio": True,
                "Pitch Factor": 0,
                "Auto Tune": False,
                "Reduce Metallic Sound": False,
            }
        ),
    ),
    "so_vits_svc_3": SmokeSpec(
        6575,
        "Fluttershy",
        _static_options(
            {
                "Character": "Fluttershy",
                "Pitch Shift": 0,
                "Speaker": "Fluttershy (speaking)",
                "Slice Threshold": -40.0,
            }
        ),
        needs_audio=True,
    ),
    "so_vits_svc_4": SmokeSpec(
        6576,
        "Fluttershy",
        _static_options(
            {
                "Character": "Fluttershy",
                "Pitch Shift": 0,
                "Predict Pitch": False,
                "Slice Length": 0.0,
                "Cross-Fade Length": 0.0,
                "Character Likeness": 0.0,
                "Reduce Hoarseness": False,
                "Apply nsf_hifigan": False,
                "Noise Scale": 0.4,
            }
        ),
        needs_audio=True,
    ),
    "so_vits_svc_5": SmokeSpec(
        6577,
        "Fluttershy (singing)",
        _static_options(
            {
                "Character": "Fluttershy (singing)",
                "Pitch Shift": 0,
            }
        ),
        needs_audio=True,
    ),
    "rvc": SmokeSpec(
        6578,
        "Fluttershy",
        _static_options(
            {
                "Character": "Fluttershy",
                "Pitch Shift": 0,
                "f0 Extraction Method": "rmvpe",
                "Index Ratio": 0.88,
                "Voice Envelope Mix Ratio": 1.0,
                "Voiceless Consonants Protection Ratio": 0.33,
            }
        ),
        needs_audio=True,
    ),
    "styletts_2": SmokeSpec(
        6580,
        STYLE_MODEL,
        _static_options(
            {
                "Character": STYLE_MODEL,
                "Noise": 0.3,
                "Diffusion Steps": 5,
                "Embedding Scale": 1.5,
                "Use Long Form": False,
                "Style Blend": 0.5,
                "Reference Style Source": "Use Precomputed Style",
                "Timbre Reference Blend": 0.1,
                "Prosody Reference Blend": 0.1,
                "Precomputed Style Character": "Fluttershy",
                "Precomputed Style Trait": "Neutral 1",
                "Speed": 1.0,
            }
        ),
    ),
    "gpt_so_vits": SmokeSpec(6581, "Fluttershy", _gpt_options),
}


def _reference_audio(path: str | None) -> tuple[np.ndarray, int]:
    if path:
        return hsc.read_audio(path)
    executable = shutil.which("espeak-ng") or shutil.which("espeak")
    if executable is None:
        raise RuntimeError("espeak-ng is required when --reference-audio is omitted")
    with tempfile.TemporaryDirectory(prefix="hay-say-smoke-") as temporary:
        output = Path(temporary) / "reference.wav"
        subprocess.run(
            [executable, "-w", str(output), DEFAULT_TEXT],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        return hsc.read_audio(str(output))


def _save_preprocessed_input(
    cache: type,
    session_id: str,
    input_hash: str,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    cache.save_audio_to_cache(Stage.PREPROCESSED, session_id, input_hash, audio, sample_rate)
    entry = {
        "Raw File": None,
        "Options": {
            "Semitone Pitch": 0,
            "Debug Pitch": False,
            "Reduce Noise": False,
            "Crop Silence": False,
        },
        "Time of Creation": datetime.datetime.now().strftime(hsc.cache.TIMESTAMP_FORMAT),
    }
    cache.update_metadata(
        Stage.PREPROCESSED,
        session_id,
        lambda metadata: metadata.update({input_hash: entry}),
    )


def _payload(
    spec: SmokeSpec,
    host: str,
    session_id: str,
    input_hash: str,
    output_hash: str,
    text: str,
    gpu_id: int | str,
) -> dict[str, Any]:
    inputs: dict[str, Any]
    if spec.needs_audio:
        inputs = {"User Audio": input_hash}
    else:
        inputs = {"User Text": text, "User Audio": None}
    return {
        "Inputs": inputs,
        "Options": spec.options(host, spec.port),
        "Output File": output_hash,
        "GPU ID": gpu_id,
        "Session ID": session_id,
    }


def _validate_output(cache: type, session_id: str, output_hash: str) -> tuple[float, int, float]:
    audio, sample_rate = cache.read_audio_from_cache(Stage.OUTPUT, session_id, output_hash)
    values = np.asarray(audio)
    if values.size == 0 or sample_rate <= 0:
        raise RuntimeError("the generated audio is empty")
    if not np.isfinite(values).all():
        raise RuntimeError("the generated audio contains non-finite samples")
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-5:
        raise RuntimeError("the generated audio is silent")
    duration = values.shape[0] / sample_rate
    if duration < 0.05:
        raise RuntimeError(f"the generated audio is too short ({duration:.3f}s)")
    return duration, sample_rate, peak


def _model_directory(runtime_id: str, character: str) -> Path:
    return Path(hsc.character_dir(runtime_id, character))


def run_smoke(
    runtime_id: str,
    spec: SmokeSpec,
    cache: type,
    session_id: str,
    input_hash: str,
    text: str,
    gpu_id: int | str,
    timeout: float,
) -> tuple[float, int, float, float]:
    model_directory = _model_directory(runtime_id, spec.character)
    if not model_directory.is_dir() or not any(model_directory.rglob("*")):
        raise RuntimeError(f"model is not installed: {model_directory}")

    initial_status = runtime_client.runtime_status(runtime_id).get("status")
    stop_afterward = initial_status not in runtime_client.RUNNING_STATES | {"starting"}
    started_at = time.monotonic()
    try:
        runtime_client.ensure_runtime_started(runtime_id, spec.port, timeout=120)
        host, port = runtime_client.service_endpoint(runtime_id, spec.port)
        output_hash = f"smoke-{runtime_id}"
        payload = _payload(
            spec,
            host,
            session_id,
            input_hash,
            output_hash,
            text,
            gpu_id,
        )
        with runtime_client.generation_lock(runtime_id):
            generator.send_payload(payload, host, port, timeout=timeout)
            duration, sample_rate, peak = _validate_output(cache, session_id, output_hash)
        return duration, sample_rate, peak, time.monotonic() - started_at
    finally:
        if stop_afterward:
            runtime_client.runtime_action(runtime_id, "stop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="append",
        choices=SMOKE_SPECS,
        help="Runtime to test; repeat for multiple runtimes (default: all)",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--reference-audio", help="Use this audio instead of synthesized speech")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--keep-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.runtime or list(SMOKE_SPECS)
    gpu_id: int | str = "" if args.cpu else args.gpu_id
    session_id = f"native-smoke-{uuid.uuid4().hex}"
    input_hash = "espeak-reference"
    cache = hsc.select_cache_implementation("file")
    audio, sample_rate = _reference_audio(args.reference_audio)
    _save_preprocessed_input(cache, session_id, input_hash, audio, sample_rate)

    failures = 0
    try:
        for runtime_id in selected:
            print(f"[smoke] {runtime_id}: starting", flush=True)
            try:
                duration, output_rate, peak, elapsed = run_smoke(
                    runtime_id,
                    SMOKE_SPECS[runtime_id],
                    cache,
                    session_id,
                    input_hash,
                    args.text,
                    gpu_id,
                    args.timeout,
                )
            except Exception as error:
                failures += 1
                print(f"[fail]  {runtime_id}: {error}", flush=True)
            else:
                print(
                    f"[ok]    {runtime_id}: {duration:.2f}s at {output_rate} Hz, "
                    f"peak {peak:.4f}, elapsed {elapsed:.1f}s",
                    flush=True,
                )
    finally:
        if not args.keep_cache:
            cache.delete_session_data(session_id)

    print(f"[smoke] completed with {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

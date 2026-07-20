import contextlib
import importlib.util
import json
import os
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from ubuntuserver.hay_say.so_vits_svc_4 import runtime as svc4_runtime


REPOSITORY = Path(__file__).resolve().parents[1]
SVC4_ROOT = REPOSITORY / "ubuntuserver" / "hay_say" / "so_vits_svc_4"
SERVER_PATH = REPOSITORY / "ubuntuserver" / "hay_say" / "so_vits_svc_4_server" / "main.py"
SPEC = importlib.util.spec_from_file_location("hay_say_svc4_server", SERVER_PATH)
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
previous_runtime = sys.modules.get("runtime")
original_path = list(sys.path)
try:
    # The installed server imports ``runtime`` from its configured source root.
    # Bind that compatibility name only while loading the server so this suite
    # cannot leak or consume another model's generic module.
    sys.modules["runtime"] = svc4_runtime
    with patch.dict(os.environ, {"HAY_SAY_SVC4_RUNTIME_ROOT": str(SVC4_ROOT)}):
        SPEC.loader.exec_module(server)
finally:
    sys.path[:] = original_path
    if previous_runtime is None:
        sys.modules.pop("runtime", None)
    else:
        sys.modules["runtime"] = previous_runtime


class FakeCache:
    def __init__(self):
        self.saved = []

    def read_audio_from_cache(self, *_args):
        return np.ones(20, dtype=np.float32), 10

    def save_audio_to_cache(self, stage, session, name, audio, sample_rate):
        self.saved.append((stage, session, name, np.asarray(audio), sample_rate))


class FakeRuntime:
    def __init__(self):
        self.generate_call = None
        self.generate_kwargs = None
        self.cancel_calls = []
        self.cancellation_scope_calls = []
        self.cancel_during_generate = False
        self.commit_calls = 0

    def generate(self, *args, **kwargs):
        self.generate_call = args
        self.generate_kwargs = kwargs
        if self.cancel_during_generate:
            kwargs["cancellation"].set()
        return np.ones(40, dtype=np.float32), 20, 3

    @contextlib.contextmanager
    def cancellation_scope(self, request_id):
        self.cancellation_scope_calls.append(request_id)
        cancellation = threading.Event()
        yield cancellation

    def commit_if_active(self, cancellation, callback):
        self.raise_if_cancelled(cancellation)
        self.commit_calls += 1
        return callback()

    def cancel(self, request_ids):
        values = list(request_ids)
        self.cancel_calls.append(values)
        return {"cancelled_request_ids": values, "active_request_ids": []}

    @staticmethod
    def raise_if_cancelled(cancellation):
        if cancellation is not None and cancellation.is_set():
            raise server.GenerationCancelled("Generation cancelled")

    def warm(self, spec, device, enhance, workers):
        return {"character": spec.character, "device": device, "slice_workers": workers}

    def unload(self, *_args):
        return {"unloaded_models": [], "busy_models": []}

    def state(self):
        return {
            "status": "ready-cold",
            "warm": False,
            "busy": False,
            "active_jobs": 0,
            "queued_jobs": 0,
            "loaded_models": [],
        }


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        root = Path(self.temporary.name)
        self.character = root / "characters" / "Fluttershy"
        self.character.mkdir(parents=True)
        (self.character / "G_1.pth").write_bytes(b"model")
        (self.character / "config.json").write_text(
            json.dumps({"data": {"sampling_rate": 20}, "spk": {"Fluttershy": 0}}),
            encoding="utf-8",
        )
        self.source = root / "source"
        (self.source / "inference").mkdir(parents=True)
        (self.source / "inference" / "infer_tool.py").write_text("", encoding="utf-8")
        self.old_root = server.SVC4_ROOT
        server.SVC4_ROOT = str(self.source)

    def tearDown(self):
        server.SVC4_ROOT = self.old_root
        self.temporary.cleanup()

    def character_dir(self, *_args):
        return str(self.character)

    def payload(self):
        return {
            "Inputs": {"User Audio": "input-hash"},
            "Options": {
                "Character": "Fluttershy",
                "Pitch Shift": 2,
                "Predict Pitch": False,
                "Slice Length": 1.0,
                "Cross-Fade Length": 0.2,
                "Slice Workers": 3,
                "Character Likeness": 0.0,
                "Reduce Hoarseness": False,
                "Apply nsf_hifigan": False,
                "Noise Scale": 0.4,
                "CPU BF16 Autocast": "ignored request override",
            },
            "Output File": "output-hash",
            "GPU ID": "",
            "Session ID": "session-id",
            "Request ID": "request-1",
        }

    def test_generate_forwards_slice_workers_and_writes_cache_directly(self):
        cache = FakeCache()
        runtime = FakeRuntime()
        app = server.create_app(cache, runtime, self.character_dir)
        with patch.object(server.hsc, "model_cpu_bf16_enabled", return_value=True):
            response = app.test_client().post("/generate", json=self.payload())
        self.assertEqual(200, response.status_code, response.get_data(as_text=True))
        self.assertEqual(3, response.get_json()["slice_workers"])
        self.assertEqual(3, runtime.generate_call[8])
        self.assertTrue(runtime.generate_call[-1])
        self.assertEqual(["request-1"], runtime.cancellation_scope_calls)
        self.assertIn("cancellation", runtime.generate_kwargs)
        self.assertEqual(1, runtime.commit_calls)
        self.assertEqual("output-hash", cache.saved[0][2])

    def test_crossfade_cannot_exceed_slice(self):
        payload = self.payload()
        payload["Options"]["Cross-Fade Length"] = 2.0
        app = server.create_app(FakeCache(), FakeRuntime(), self.character_dir)
        response = app.test_client().post("/generate", json=payload)
        self.assertEqual(400, response.status_code)
        self.assertIn("cannot exceed", response.get_json()["error"])

    def test_path_components_and_cache_keys_are_bounded(self):
        for value in (".", "..", "nested/character", "x" * 256):
            with self.subTest(value=value):
                with self.assertRaises(server.RequestError):
                    server.component(value, "Character")
        for value in (".", "..", "space is not valid", "x" * 256):
            with self.subTest(value=value):
                with self.assertRaises(server.RequestError):
                    server.cache_key(value, "Output File")

    def test_resolver_rejects_character_directory_escape(self):
        model_root = self.character.parent

        def escaping_dir(_architecture, character):
            if character == "__root_probe__":
                return str(model_root / character)
            return str(model_root / character)

        with self.assertRaisesRegex(server.RequestError, "outside the model directory"):
            server.ModelResolver(escaping_dir).resolve("..")

    def test_multispeaker_model_requires_selector(self):
        (self.character / "config.json").write_text(
            json.dumps({"data": {"sampling_rate": 20}, "spk": {"soft": 0, "strong": 1}}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(server.RequestError, "speaker.json is required"):
            server.ModelResolver(self.character_dir).resolve("Fluttershy")

    def test_unload_returns_a_valid_operation_status(self):
        app = server.create_app(FakeCache(), FakeRuntime(), self.character_dir)
        response = app.test_client().post("/runtime/unload", json={})
        self.assertEqual(200, response.status_code)
        self.assertEqual("unloaded", response.get_json()["status"])

    def test_runtime_reports_effective_backend_bf16_policy(self):
        app = server.create_app(FakeCache(), FakeRuntime(), self.character_dir)
        response = app.test_client().get("/runtime")

        self.assertEqual(200, response.status_code)
        policy = response.get_json()["cpu_bf16"]
        self.assertEqual("HAY_SAY_SVC4_CPU_BF16_AUTOCAST", policy["environment_variable"])
        self.assertEqual(policy["effective"], policy["requested"] and policy["amx_available"])

    def test_cancel_endpoint_deduplicates_request_ids(self):
        runtime = FakeRuntime()
        app = server.create_app(FakeCache(), runtime, self.character_dir)
        response = app.test_client().post(
            "/cancel",
            json={"Request IDs": ["request-a", "request-b", "request-a"]},
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual([["request-a", "request-b"]], runtime.cancel_calls)
        self.assertEqual(
            ["request-a", "request-b"], response.get_json()["cancelled_request_ids"]
        )

    def test_cancelled_generate_never_writes_output(self):
        cache = FakeCache()
        runtime = FakeRuntime()
        runtime.cancel_during_generate = True
        app = server.create_app(cache, runtime, self.character_dir)

        response = app.test_client().post("/generate", json=self.payload())

        self.assertEqual(409, response.status_code)
        self.assertEqual("cancelled", response.get_json()["status"])
        self.assertEqual([], cache.saved)

    def test_cancel_requires_a_nonempty_request_id_array(self):
        app = server.create_app(FakeCache(), FakeRuntime(), self.character_dir)
        for payload in ({}, {"Request IDs": []}, {"Request IDs": ["not valid"]}):
            with self.subTest(payload=payload):
                response = app.test_client().post("/cancel", json=payload)
                self.assertEqual(400, response.status_code)


if __name__ == "__main__":
    unittest.main()

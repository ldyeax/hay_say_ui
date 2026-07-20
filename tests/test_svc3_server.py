import base64
import contextlib
import importlib.util
import json
import sys
import tempfile
import threading
import types
import unittest
from enum import Enum
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SVC3_ROOT = REPO_ROOT / "ubuntuserver" / "hay_say" / "so_vits_svc_3"
SERVER_PATH = REPO_ROOT / "ubuntuserver" / "hay_say" / "so_vits_svc_3_server" / "main.py"
if str(SVC3_ROOT) not in sys.path:
    sys.path.insert(0, str(SVC3_ROOT))


class Stage(Enum):
    PREPROCESSED = "preprocessed"
    OUTPUT = "output"


def load_server_module():
    fake_hsc = types.ModuleType("hay_say_common")
    fake_hsc.ROOT_DIR = str(SVC3_ROOT.parent)
    fake_hsc.cache_implementation_map = {"file": object()}
    fake_hsc.character_dir = lambda architecture, character: str(
        SVC3_ROOT.parent / "models" / architecture / "characters" / character
    )
    fake_hsc.select_cache_implementation = lambda name: None
    fake_hsc.model_cpu_bf16_enabled = lambda _model_id: False
    fake_hsc.runtime_state_with_cpu_bf16 = lambda state, _model_id: {
        **state,
        "cpu_bf16": {
            "environment_variable": "HAY_SAY_SVC3_CPU_BF16_AUTOCAST",
            "default": False,
            "requested": False,
            "source": "default",
            "amx_available": True,
            "effective": False,
        },
    }
    fake_cache_module = types.ModuleType("hay_say_common.cache")
    fake_cache_module.Stage = Stage

    old_hsc = sys.modules.get("hay_say_common")
    old_cache = sys.modules.get("hay_say_common.cache")
    sys.modules["hay_say_common"] = fake_hsc
    sys.modules["hay_say_common.cache"] = fake_cache_module
    try:
        spec = importlib.util.spec_from_file_location("svc3_server_under_test", SERVER_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if old_hsc is None:
            sys.modules.pop("hay_say_common", None)
        else:
            sys.modules["hay_say_common"] = old_hsc
        if old_cache is None:
            sys.modules.pop("hay_say_common.cache", None)
        else:
            sys.modules["hay_say_common.cache"] = old_cache


SERVER = load_server_module()


class FakeCache:
    def __init__(self):
        self.input_audio = np.arange(8, dtype=np.float32)
        self.saved = []
        self.exists = True

    def file_is_already_cached(self, stage, session_id, name):
        return self.exists

    def read_audio_from_cache(self, stage, session_id, name):
        return self.input_audio, 16000

    def save_audio_to_cache(self, stage, session_id, name, audio, sample_rate):
        self.saved.append((stage, session_id, name, np.asarray(audio), sample_rate))


class FakeRuntime:
    def __init__(self):
        self.generate_calls = []
        self.warm_calls = []
        self.unload_calls = []
        self.cancellation_scope_calls = []
        self.cancel_calls = []
        self.cancel_during_generate = False
        self.commit_calls = 0

    def generate(
        self,
        spec,
        speaker,
        pitches,
        source,
        sample_rate,
        device,
        slice_db=-40,
        cpu_bf16=False,
        cancellation=None,
    ):
        self.generate_calls.append(
            (spec, speaker, tuple(pitches), source, sample_rate, device, slice_db, cpu_bf16,
             cancellation)
        )
        if self.cancel_during_generate:
            cancellation.set()
        return [
            (pitch, np.full(4, pitch, dtype=np.float32)) for pitch in pitches
        ], 32000

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
            raise SERVER.GenerationCancelled("Generation cancelled")

    def warm(self, spec, device):
        self.warm_calls.append((spec, device))
        return {"character": spec.character, "device": device, "sample_rate": 32000}

    def unload(self, character=None, device=None):
        self.unload_calls.append((character, device))
        return {"unloaded_models": [], "busy_models": [], "released_hubert_devices": []}

    def state(self):
        return {
            "status": "ready-cold",
            "device": None,
            "warm": False,
            "busy": False,
            "active_jobs": 0,
            "queued_jobs": 0,
            "loaded_models": [],
            "hubert_devices": [],
            "devices": [],
            "max_models_per_device": 2,
            "resources": {"process_rss_bytes": 1, "gpus": []},
        }

    def gpu_info(self):
        return [{"Index": 0, "Name": "test-gpu", "Free Memory": 10, "Total Memory": 20}]


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.characters_root = Path(self.temp_dir.name) / "characters"
        self.cache = FakeCache()
        self.runtime = FakeRuntime()
        self.app = SERVER.create_app(
            self.cache,
            runtime=self.runtime,
            character_dir_func=lambda architecture, character: str(self.characters_root / character),
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def add_character(self, name="Test Character", speakers=None, selected=None):
        speakers = speakers or {"voice": 0}
        character_dir = self.characters_root / name
        character_dir.mkdir(parents=True)
        (character_dir / "G_100.pth").touch()
        (character_dir / "config.json").write_text(json.dumps({"spk": speakers}), encoding="utf-8")
        if selected is not None:
            (character_dir / "speaker.json").write_text(
                json.dumps({"speaker": selected}), encoding="utf-8"
            )
        return character_dir

    @staticmethod
    def legacy_payload(character="Test Character"):
        return {
            "Inputs": {"User Audio": "input-hash"},
            "Options": {"Architecture": "so_vits_svc_3", "Character": character, "Pitch Shift": 2},
            "Output File": "output-hash",
            "GPU ID": "",
            "Session ID": "session-1",
            "Request ID": "request-1",
        }

    def test_legacy_scalar_generate_contract(self):
        self.add_character()
        response = self.client.post("/generate", json=self.legacy_payload())

        self.assertEqual(200, response.status_code)
        body = response.get_json()
        self.assertEqual("", base64.b64decode(body["message"]).decode("utf-8"))
        self.assertEqual([{"output_file": "output-hash", "pitch_shift": 2}], body["outputs"])
        self.assertEqual("voice", self.runtime.generate_calls[0][1])
        self.assertEqual((2,), self.runtime.generate_calls[0][2])
        self.assertEqual("cpu", self.runtime.generate_calls[0][5])
        self.assertEqual(["request-1"], self.runtime.cancellation_scope_calls)
        self.assertIs(self.runtime.generate_calls[0][8].is_set(), False)
        self.assertEqual(1, self.runtime.commit_calls)
        self.assertEqual(["output-hash"], [item[2] for item in self.cache.saved])

    def test_batch_generate_uses_speaker_json_and_saves_every_output(self):
        self.add_character(speakers={"voice-a": 0, "voice-b": 1}, selected="voice-b")
        payload = self.legacy_payload()
        payload["Options"].pop("Pitch Shift")
        payload["Options"]["Pitch Shifts"] = [-2, 0, 4]
        payload.pop("Output File")
        payload["Output Files"] = ["out-low", "out-mid", "out-high"]

        response = self.client.post("/generate", json=payload)

        self.assertEqual(200, response.status_code)
        self.assertEqual("voice-b", self.runtime.generate_calls[0][1])
        self.assertEqual((-2, 0, 4), self.runtime.generate_calls[0][2])
        self.assertEqual(
            ["out-low", "out-mid", "out-high"],
            [item[2] for item in self.cache.saved],
        )

    def test_cpu_bf16_is_backend_controlled_and_request_value_is_ignored(self):
        self.add_character()
        payload = self.legacy_payload()
        payload["Options"]["CPU BF16 Autocast"] = "yes"
        original = SERVER.hsc.model_cpu_bf16_enabled
        try:
            SERVER.hsc.model_cpu_bf16_enabled = lambda _model_id: True
            response = self.client.post("/generate", json=payload)
        finally:
            SERVER.hsc.model_cpu_bf16_enabled = original

        self.assertEqual(200, response.status_code)
        self.assertIs(self.runtime.generate_calls[0][7], True)

    def test_explicit_speaker_can_select_a_multispeaker_voice(self):
        self.add_character(speakers={"voice-a": 0, "voice-b": 1}, selected="voice-b")
        payload = self.legacy_payload()
        payload["Options"]["Speaker"] = "voice-a"

        response = self.client.post("/generate", json=payload)

        self.assertEqual(200, response.status_code)
        self.assertEqual("voice-a", self.runtime.generate_calls[0][1])

    def test_validation_errors_are_json_with_legacy_base64_message(self):
        self.add_character()
        payload = self.legacy_payload()
        payload["Output File"] = "../escape"

        response = self.client.post("/generate", json=payload)

        self.assertEqual(400, response.status_code)
        body = response.get_json()
        decoded = base64.b64decode(body["message"]).decode("utf-8")
        self.assertIn("Output File", decoded)
        self.assertEqual("bad_request", body["error"]["type"])

    def test_pitch_batch_lengths_must_match(self):
        self.add_character()
        payload = self.legacy_payload()
        payload["Options"]["Pitch Shifts"] = [0, 1]
        payload["Output Files"] = ["only-one"]

        response = self.client.post("/generate", json=payload)

        self.assertEqual(400, response.status_code)
        self.assertEqual([], self.runtime.generate_calls)

    def test_lifecycle_and_observability_endpoints(self):
        self.add_character()
        self.assertEqual("ok", self.client.get("/health").get_json()["status"])
        runtime_state = self.client.get("/runtime").get_json()
        self.assertEqual("ready-cold", runtime_state["status"])
        self.assertEqual(
            "HAY_SAY_SVC3_CPU_BF16_AUTOCAST",
            runtime_state["cpu_bf16"]["environment_variable"],
        )
        self.assertEqual("test-gpu", self.client.get("/gpu-info").get_json()[0]["Name"])

        warm = self.client.post("/warm", json={"Character": "Test Character", "GPU ID": ""})
        self.assertEqual(200, warm.status_code)
        self.assertEqual("cpu", self.runtime.warm_calls[0][1])

        unload = self.client.post("/unload", json={"Character": "Test Character", "GPU ID": ""})
        self.assertEqual(200, unload.status_code)
        self.assertEqual(("Test Character", "cpu"), self.runtime.unload_calls[0])

    def test_cancel_endpoint_deduplicates_request_ids(self):
        response = self.client.post(
            "/cancel",
            json={"Request IDs": ["request-a", "request-b", "request-a"]},
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual([["request-a", "request-b"]], self.runtime.cancel_calls)
        self.assertEqual(
            ["request-a", "request-b"], response.get_json()["cancelled_request_ids"]
        )

    def test_cancelled_generate_never_writes_output(self):
        self.add_character()
        self.runtime.cancel_during_generate = True

        response = self.client.post("/generate", json=self.legacy_payload())

        self.assertEqual(409, response.status_code)
        self.assertEqual("cancelled", response.get_json()["error"]["type"])
        self.assertEqual([], self.cache.saved)

    def test_cancel_requires_a_nonempty_request_id_array(self):
        for payload in ({}, {"Request IDs": []}, {"Request IDs": ["not valid"]}):
            with self.subTest(payload=payload):
                response = self.client.post("/cancel", json=payload)
                self.assertEqual(400, response.status_code)

    def test_server_has_no_template_or_subprocess_execution_path(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("subprocess", source)
        self.assertNotIn("inference_main_template", source)
        self.assertNotIn("RAW_COPY_FOLDER", source)
        self.assertNotIn("OUTPUT_COPY_FOLDER", source)
        self.assertFalse((SVC3_ROOT / "inference_main.py").exists())
        self.assertFalse((SVC3_ROOT / "inference_main_template.py").exists())


if __name__ == "__main__":
    unittest.main()

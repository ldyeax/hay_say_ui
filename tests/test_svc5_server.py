import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SERVER_ROOT = REPOSITORY / "ubuntuserver/hay_say/so_vits_svc_5_server"
RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "svc5_runtime", SERVER_ROOT / "svc5_runtime.py"
)
runtime_module = importlib.util.module_from_spec(RUNTIME_SPEC)
sys.modules[RUNTIME_SPEC.name] = runtime_module
RUNTIME_SPEC.loader.exec_module(runtime_module)
SERVER_SPEC = importlib.util.spec_from_file_location(
    "hay_say_svc5_server", SERVER_ROOT / "main.py"
)
server = importlib.util.module_from_spec(SERVER_SPEC)
sys.modules[SERVER_SPEC.name] = server
SERVER_SPEC.loader.exec_module(server)


class FakeCache:
    def __init__(self):
        self.saved = []

    def read_audio_from_cache(self, *_args):
        return np.ones(160, dtype=np.float32), 16000

    def save_audio_to_cache(self, stage, session, name, audio, sample_rate):
        self.saved.append((stage, session, name, np.asarray(audio), sample_rate))


class FakeRuntime:
    def __init__(self):
        self.generate_call = None
        self.cancel_call = None

    def generate(self, *args):
        self.generate_call = args
        return np.ones(320, dtype=np.float32), 32000

    def cancel(self, request_ids):
        self.cancel_call = list(request_ids)
        return {"cancelled": list(request_ids), "active": list(request_ids[:1])}

    def commit_if_active(self, _request_id, callback):
        return callback()

    def warm(self, spec, device, workers):
        return {"character": spec.character, "device": device, "workers": workers}

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


class TestServer:
    def setup_method(self):
        self.temporary = TemporaryDirectory()
        root = Path(self.temporary.name)
        self.model_root = root / "characters"
        self.character = self.model_root / "Fluttershy"
        (self.character / "singer").mkdir(parents=True)
        (self.character / "Fluttershy.pt").write_bytes(b"original")
        (self.character / "sovits5.0.pth").write_bytes(b"exported")
        np.save(self.character / "singer/Fluttershy.spk.npy", np.ones(4))
        self.source = root / "svc5-v2"
        (self.source / "configs").mkdir(parents=True)
        (self.source / "configs/base.yaml").write_text("data:\n  sampling_rate: 32000\n")
        (self.source / "svc_inference.py").write_text("")
        self.old_v2 = server.SVC5_V2_ROOT
        server.SVC5_V2_ROOT = str(self.source)
        self.version_resolver = server.VersionResolver(lambda _path: 2)

    def teardown_method(self):
        server.SVC5_V2_ROOT = self.old_v2
        self.temporary.cleanup()

    def character_dir(self, _architecture, character):
        return str(self.model_root / character)

    def payload(self):
        return {
            "Inputs": {"User Audio": "input-hash"},
            "Options": {
                "Character": "Fluttershy",
                "Pitch Shift": 2,
                "CPU BF16 Autocast": "ignored request override",
            },
            "Output File": "output-hash",
            "GPU ID": "",
            "Session ID": "session-id",
            "Request ID": "browser:variant-2",
        }

    def test_generate_forwards_request_identity_and_writes_cache(self, monkeypatch):
        cache = FakeCache()
        current = FakeRuntime()
        app = server.create_app(
            cache, current, self.character_dir, self.version_resolver
        )
        monkeypatch.setattr(server.hsc, "model_cpu_bf16_enabled", lambda _model_id: True)
        response = app.test_client().post("/generate", json=self.payload())

        assert response.status_code == 200, response.get_data(as_text=True)
        assert current.generate_call[1] == "input-hash"
        assert current.generate_call[4] == "cpu"
        assert current.generate_call[5:] == (2, True, "browser:variant-2")
        assert cache.saved[0][2] == "output-hash"
        assert response.get_json()["request_id"] == "browser:variant-2"

    def test_cancel_uses_exact_request_ids_contract(self):
        current = FakeRuntime()
        app = server.create_app(
            FakeCache(), current, self.character_dir, self.version_resolver
        )
        response = app.test_client().post(
            "/cancel", json={"Request IDs": ["one", "two"]}
        )

        assert response.status_code == 200
        assert current.cancel_call == ["one", "two"]
        assert response.get_json() == {
            "cancelled": ["one", "two"],
            "active": ["one"],
        }

    def test_runtime_reports_effective_backend_bf16_policy(self):
        app = server.create_app(
            FakeCache(), FakeRuntime(), self.character_dir, self.version_resolver
        )

        response = app.test_client().get("/runtime")

        assert response.status_code == 200
        policy = response.get_json()["cpu_bf16"]
        assert policy["environment_variable"] == "HAY_SAY_SVC5_CPU_BF16_AUTOCAST"
        assert policy["effective"] == (policy["requested"] and policy["amx_available"])

    def test_resolver_prefers_small_exported_checkpoint_and_caches_version_probe(self):
        calls = []
        versions = server.VersionResolver(lambda path: calls.append(path) or 2)
        resolver = server.ModelResolver(self.character_dir, versions)

        first = resolver.resolve("Fluttershy")
        second = resolver.resolve("Fluttershy")

        assert first.checkpoint_path.endswith("sovits5.0.pth")
        assert first.version == 2
        assert second == first
        assert len(calls) == 1

    def test_resolver_propagates_non_default_configured_sample_rate(self):
        (self.source / "configs/base.yaml").write_text(
            "data:\n  sampling_rate: 44100\n", encoding="utf-8"
        )

        resolved = server.ModelResolver(
            self.character_dir, self.version_resolver
        ).resolve("Fluttershy")

        assert resolved.sample_rate == 44100

    def test_resolver_rejects_invalid_configured_sample_rate(self):
        (self.source / "configs/base.yaml").write_text(
            "data:\n  sampling_rate: fast\n", encoding="utf-8"
        )

        with pytest.raises(server.RequestError, match="positive integer") as error:
            server.ModelResolver(
                self.character_dir, self.version_resolver
            ).resolve("Fluttershy")

        assert error.value.status_code == 503

    def test_invalid_cancel_request_does_not_accept_paths(self):
        app = server.create_app(
            FakeCache(), FakeRuntime(), self.character_dir, self.version_resolver
        )
        response = app.test_client().post(
            "/cancel", json={"Request IDs": ["../request"]}
        )
        assert response.status_code == 400

import importlib.util
import sys
from pathlib import Path

import numpy as np

from hay_say_common.cache import Stage


REPOSITORY = Path(__file__).resolve().parents[1]


def load_smoke_models():
    module_name = "hay_say_smoke_models"
    module_path = REPOSITORY / "ubuntuserver" / "smoke-models.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_svc3_smoke_payload_uses_normal_fluttershy_voice():
    smoke_models = load_smoke_models()
    spec = smoke_models.SMOKE_SPECS["so_vits_svc_3"]

    assert spec.port == 6575
    assert spec.character == "Fluttershy"
    assert spec.needs_audio is True
    assert spec.options("127.0.0.1", spec.port) == {
        "Character": "Fluttershy",
        "Pitch Shift": 0,
        "Speaker": "Fluttershy (speaking)",
        "Slice Threshold": -40.0,
    }

    payload = smoke_models._payload(
        spec,
        "127.0.0.1",
        "smoke-session",
        "reference-audio",
        "svc3-output",
        "ignored text",
        0,
    )
    assert payload["Inputs"] == {"User Audio": "reference-audio"}
    assert payload["Output File"] == "svc3-output"
    assert payload["GPU ID"] == 0
    assert payload["Session ID"] == "smoke-session"


def test_smoke_input_is_registered_in_preprocessed_cache_metadata():
    smoke_models = load_smoke_models()

    class Cache:
        metadata = {}
        saved = None

        @classmethod
        def save_audio_to_cache(cls, stage, session_id, name, audio, sample_rate):
            cls.saved = (stage, session_id, name, audio, sample_rate)

        @classmethod
        def update_metadata(cls, stage, session_id, updater):
            assert stage is Stage.PREPROCESSED
            assert session_id == "smoke-session"
            updater(cls.metadata)

    audio = np.array([0.0, 0.25, -0.25], dtype=np.float32)
    smoke_models._save_preprocessed_input(Cache, "smoke-session", "reference-audio", audio, 32000)

    assert Cache.saved[:3] == (Stage.PREPROCESSED, "smoke-session", "reference-audio")
    assert Cache.saved[3] is audio
    assert Cache.saved[4] == 32000
    assert Cache.metadata["reference-audio"]["Time of Creation"]

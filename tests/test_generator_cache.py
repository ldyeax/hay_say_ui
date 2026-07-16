from contextlib import nullcontext
import numpy

import generator
from hay_say_common.cache import Stage


class FakeCache:
    files = set()
    metadata = {}

    @classmethod
    def reset(cls):
        cls.files = set()
        cls.metadata = {}

    @classmethod
    def file_is_already_cached(cls, stage, session_id, key):
        return (stage, session_id, key) in cls.files

    @classmethod
    def read_audio_from_cache(cls, stage, session_id, key):
        if (stage, session_id, key) not in cls.files:
            raise FileNotFoundError(key)
        return numpy.ones(32), 32000

    @classmethod
    def update_metadata(cls, stage, session_id, updater):
        values = cls.metadata.setdefault((stage, session_id), {})
        updater(values)
        return values


class FakeTab:
    id = "rvc"
    label = "RVC"
    port = 6578
    pitch_batch_key = "Pitch Shift"
    pitch_batch_bounds = (-36, 36)
    supports_native_pitch_batch = False

    def __init__(self, cache_generated_output):
        self.cache_generated_output = cache_generated_output

    def construct_input_dict(self, session_data, *args):
        return {"Character": "Fluttershy", "Pitch Shift": 0}


def _configure_generation(monkeypatch):
    sent = []
    monkeypatch.setattr(generator, "_model_identity", lambda *_: "weights-v1")
    monkeypatch.setattr(generator.runtime_client, "ensure_runtime_started", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generator.runtime_client, "service_endpoint", lambda *_: ("localhost", 6578))
    monkeypatch.setattr(generator.runtime_client, "generation_lock", lambda *_: nullcontext())

    def send(payload, *_args, **_kwargs):
        sent.append(payload)
        FakeCache.files.add((Stage.OUTPUT, payload["Session ID"], payload["Output File"]))

    monkeypatch.setattr(generator, "send_payload", send)
    return sent


def test_deterministic_generation_reuses_matching_model_cache(monkeypatch):
    FakeCache.reset()
    sent = _configure_generation(monkeypatch)
    tab = FakeTab(cache_generated_output=True)
    session = {"id": "session"}

    first = generator.process_batch(FakeCache, None, "input", tab, [], session, 0)
    second = generator.process_batch(FakeCache, None, "input", tab, [], session, 0)

    assert first == second
    assert len(sent) == 1


def test_stochastic_generation_gets_a_fresh_cache_key(monkeypatch):
    FakeCache.reset()
    sent = _configure_generation(monkeypatch)
    tab = FakeTab(cache_generated_output=False)
    session = {"id": "session"}

    first = generator.process_batch(FakeCache, None, "input", tab, [], session, 0)
    second = generator.process_batch(FakeCache, None, "input", tab, [], session, 0)

    assert first != second
    assert len(sent) == 2


def test_model_identity_changes_when_weight_file_changes(tmp_path, monkeypatch):
    character = tmp_path / "models" / "rvc" / "characters" / "Fluttershy"
    character.mkdir(parents=True)
    weight = character / "voice.pth"
    weight.write_bytes(b"first")
    monkeypatch.setattr(generator.hsc, "character_dir", lambda *_: str(character))

    first = generator._model_identity("rvc", "Fluttershy")
    weight.write_bytes(b"second-version")
    second = generator._model_identity("rvc", "Fluttershy")

    assert first != second

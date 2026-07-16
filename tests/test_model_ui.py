import json
from architectures.gpt_so_vits.GPTSoVITSTab import GPTSoVITSTab
from architectures.so_vits_svc_3.SoVitsSvc3Tab import SoVitsSvc3Tab


def test_svc3_reads_multispeaker_options_without_starting_runtime(tmp_path, monkeypatch):
    character = tmp_path / "Fluttershy"
    character.mkdir()
    (character / "config.json").write_text(json.dumps({"spk": {"soft": 0, "strong": 1}}))
    (character / "speaker.json").write_text(json.dumps({"speaker": "strong"}))
    monkeypatch.setattr(
        "architectures.so_vits_svc_3.SoVitsSvc3Tab.hsc.character_dir",
        lambda *_: str(character),
    )

    assert SoVitsSvc3Tab(None).available_speakers("Fluttershy") == (["soft", "strong"], "strong")


def test_gpt_traits_are_read_from_local_safetensors_header(tmp_path, monkeypatch):
    character = tmp_path / "Fluttershy"
    character.mkdir()
    header = json.dumps({"neutral.embedding": {}, "happy.embedding": {}, "__metadata__": {}}).encode()
    (character / "voice.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    monkeypatch.setattr(
        "architectures.gpt_so_vits.GPTSoVITSTab.hsc.character_dir",
        lambda *_: str(character),
    )

    assert GPTSoVITSTab(None).available_precomputed_traits("Fluttershy") == ["happy", "neutral"]

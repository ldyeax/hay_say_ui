import json
import os
from architectures.controllable_talknet.ControllableTalknetTab import ControllableTalknetTab
from architectures.gpt_so_vits.GPTSoVITSTab import GPTSoVITSTab
from architectures.rvc.RvcTab import RvcTab
from architectures.so_vits_svc_3.SoVitsSvc3Tab import SoVitsSvc3Tab
from architectures.so_vits_svc_4.SoVitsSvc4Tab import SoVitsSvc4Tab
from architectures.so_vits_svc_5.SoVitsSvc5Tab import SoVitsSvc5Tab
from architectures.styletts_2.StyleTTS2Tab import StyleTTS2Tab


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


def test_hardware_options_put_auto_first(monkeypatch):
    monkeypatch.setattr(SoVitsSvc3Tab, "is_gpu_available", property(lambda _self: True))
    assert SoVitsSvc3Tab(None).hardware_options == ["Auto", "GPU", "CPU"]


def test_model_scheduler_contracts_match_backend_isolation():
    expected = (
        (ControllableTalknetTab, True, True, True, False),
        (GPTSoVITSTab, False, True, False, False),
        (RvcTab, True, True, True, False),
        (SoVitsSvc3Tab, True, True, True, True),
        (SoVitsSvc4Tab, True, True, True, True),
        (SoVitsSvc5Tab, True, True, True, False),
        (StyleTTS2Tab, False, True, False, False),
    )
    for tab_type, has_pitch, parallel, mixed, device_serialized in expected:
        tab = tab_type(None)
        assert (tab.pitch_batch_key is not None) is has_pitch, tab.id
        assert tab.supports_parallel_requests is parallel, tab.id
        assert tab.supports_mixed_device_pitch_batch is mixed, tab.id
        assert tab.serializes_device_requests is device_serialized, tab.id


def test_svc5_pitch_workers_follow_the_isolated_backend_pools(monkeypatch):
    monkeypatch.delenv("HAY_SAY_SVC5_CPU_WORKERS", raising=False)
    monkeypatch.delenv("HAY_SAY_SVC5_GPU_WORKERS", raising=False)
    tab = SoVitsSvc5Tab(None)

    assert tab.pitch_batch_request_workers("") == 24
    assert tab.pitch_batch_request_workers(0) == 1

    monkeypatch.setenv("HAY_SAY_SVC5_CPU_WORKERS", "37")
    monkeypatch.setenv("HAY_SAY_SVC5_GPU_WORKERS", "2")
    assert tab.pitch_batch_request_workers("") == 37
    assert tab.pitch_batch_request_workers(0) == 2


def test_legacy_isolated_wrappers_expose_env_sized_pitch_lanes(monkeypatch):
    monkeypatch.delenv("HAY_SAY_RVC_CPU_WORKERS", raising=False)
    monkeypatch.delenv("HAY_SAY_TALKNET_CPU_WORKERS", raising=False)
    assert RvcTab(None).pitch_batch_request_workers("") == 12
    assert ControllableTalknetTab(None).pitch_batch_request_workers("") == 8
    assert RvcTab(None).pitch_batch_request_workers(0) == 1
    assert ControllableTalknetTab(None).pitch_batch_request_workers(0) == 1

    monkeypatch.setenv("HAY_SAY_RVC_CPU_WORKERS", "19")
    monkeypatch.setenv("HAY_SAY_TALKNET_CPU_WORKERS", "11")
    assert RvcTab(None).pitch_batch_request_workers("") == 19
    assert ControllableTalknetTab(None).pitch_batch_request_workers("") == 11


def test_svc4_exposes_slice_workers_and_parallel_device_traits():
    tab = SoVitsSvc4Tab(None)
    options = tab.construct_input_dict(
        {"id": "session"},
        "Fluttershy", 0, [], 1.0, 0.1, 0.0, [], [], 0.4, 6,
    )
    assert options["Slice Workers"] == 6
    assert tab.supports_parallel_requests
    assert tab.supports_mixed_device_pitch_batch
    assert tab.serializes_device_requests
    assert not tab.supports_native_pitch_batch
    slicing_off = tab.construct_input_dict(
        {"id": "session"},
        "Fluttershy", 0, [], 0.0, 0.1, 0.0, [], [], 0.4, 6,
    )
    assert slicing_off["Cross-Fade Length"] == 0.0
    assert slicing_off["Slice Workers"] == 0


def test_svc4_mixed_device_cache_requires_exact_warm_models(tmp_path, monkeypatch):
    character = tmp_path / "Fluttershy"
    character.mkdir()
    config = character / "config.json"
    model = character / "G_100.pth"
    config.write_text(json.dumps({"data": {"sampling_rate": 44100}, "spk": {"Fluttershy": 0}}))
    model.write_bytes(b"weights")
    monkeypatch.setattr(
        "architectures.so_vits_svc_4.SoVitsSvc4Tab.hsc.character_dir",
        lambda *_: str(character),
    )
    tab = SoVitsSvc4Tab(None)
    details = [
        {
            "character": "Fluttershy",
            "device": device,
            "model_path": str(model),
            "config_path": str(config),
            "model_revision": tab._file_revision(model),
            "config_revision": tab._file_revision(config),
            "enhance": False,
            "workers": 1,
        }
        for device in ("cpu", "cuda:0")
    ]
    options = {"Character": "Fluttershy", "Apply nsf_hifigan": False}
    assert tab.mixed_device_caches_are_warm(
        {"loaded_model_details": details}, options, "", 0
    )
    assert not tab.mixed_device_caches_are_warm(
        {"loaded_model_details": [{**entry, "workers": 0} for entry in details]},
        options,
        "",
        0,
    )
    assert not tab.mixed_device_caches_are_warm(
        {"loaded_model_details": [{**entry, "enhance": True} for entry in details]},
        options,
        "",
        0,
    )


def test_svc3_mixed_device_cache_check_matches_exact_character_checkpoint_and_gpu(tmp_path, monkeypatch):
    character = tmp_path / "Fluttershy"
    character.mkdir()
    config = character / "config.json"
    model = character / "G_100.pth"
    config.write_text(json.dumps({"spk": {"Fluttershy": 0}}))
    model.write_bytes(b"weights")
    monkeypatch.setattr(
        "architectures.so_vits_svc_3.SoVitsSvc3Tab.hsc.character_dir",
        lambda *_: str(character),
    )
    tab = SoVitsSvc3Tab(None)
    model_revision = tab._file_revision(model)
    config_revision = tab._file_revision(config)
    details = [
        {
            "character": "Fluttershy",
            "device": device,
            "model_path": os.path.realpath(model),
            "config_path": os.path.realpath(config),
            "model_revision": model_revision,
            "config_revision": config_revision,
        }
        for device in ("cpu", "cuda:0")
    ]

    assert tab.mixed_device_caches_are_warm(
        {"loaded_model_details": details},
        {"Character": "Fluttershy"},
        "",
        0,
    )

    for changed_details in (
        details[:1],
        [{**entry, "character": "Other"} for entry in details],
        [{**entry, "model_path": str(tmp_path / "other.pth")} for entry in details],
        [details[0], {**details[1], "device": "cuda:1"}],
    ):
        assert not tab.mixed_device_caches_are_warm(
            {"loaded_model_details": changed_details},
            {"Character": "Fluttershy"},
            "",
            0,
        )

    assert not tab.mixed_device_caches_are_warm(None, {"Character": "Fluttershy"}, "", 0)

    model.write_bytes(b"replacement-weights")
    assert not tab.mixed_device_caches_are_warm(
        {"loaded_model_details": details},
        {"Character": "Fluttershy"},
        "",
        0,
    )


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

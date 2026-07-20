from contextlib import nullcontext
import threading

import numpy
import pytest

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
    supports_parallel_requests = False
    supports_mixed_device_pitch_batch = False
    serializes_device_requests = False
    is_gpu_available = True

    def __init__(self, cache_generated_output):
        self.cache_generated_output = cache_generated_output

    def construct_input_dict(self, session_data, *args):
        return {"Character": "Fluttershy", "Pitch Shift": 0}

    def mixed_device_caches_are_warm(self, runtime_state, options, cpu_device, gpu_device):
        return True


def _configure_generation(monkeypatch):
    sent = []
    generator._GPU_CAPABILITY_CACHE.clear()
    monkeypatch.setattr(generator, "_model_identity", lambda *_: "weights-v1")
    monkeypatch.setattr(generator.runtime_client, "ensure_runtime_started", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generator.runtime_client, "service_endpoint", lambda *_: ("localhost", 6578))
    monkeypatch.setattr(generator.runtime_client, "service_runtime_state", lambda *_: {})
    monkeypatch.setattr(generator.runtime_client, "generation_lock", lambda *_, **__: nullcontext())
    monkeypatch.setattr(generator.runtime_client, "output_locks", lambda *_, **__: nullcontext())
    monkeypatch.setattr(
        generator.inference_scheduler,
        "inference_device",
        lambda requested, allow_gpu=True, serial_device_key=None, cancel_check=None: nullcontext(
            0 if requested == "auto" and allow_gpu else ("" if requested == "auto" else requested)
        ),
    )

    def send(payload, *_args, **_kwargs):
        sent.append(payload)
        for output_file in payload.get("Output Files", [payload["Output File"]]):
            FakeCache.files.add((Stage.OUTPUT, payload["Session ID"], output_file))

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


def test_generation_payload_does_not_expose_backend_cpu_precision(monkeypatch):
    FakeCache.reset()
    sent = _configure_generation(monkeypatch)
    tab = FakeTab(cache_generated_output=True)
    session = {"id": "session"}

    generator.process_batch(FakeCache, None, "input", tab, [], session, "")

    assert len(sent) == 1
    assert "CPU BF16 Autocast" not in sent[0]["Options"]


def test_backend_precision_policy_is_part_of_the_internal_cache_identity(monkeypatch):
    FakeCache.reset()
    sent = _configure_generation(monkeypatch)

    class PolicyTab(FakeTab):
        id = "so_vits_svc_4"

    session = {"id": "session"}
    monkeypatch.setattr(generator.hsc, "model_cpu_bf16_enabled", lambda _runtime_id: False)
    fp32 = generator.process_batch(FakeCache, None, "input", PolicyTab(True), [], session, "")
    monkeypatch.setattr(generator.hsc, "model_cpu_bf16_enabled", lambda _runtime_id: True)
    bf16 = generator.process_batch(FakeCache, None, "input", PolicyTab(True), [], session, "")

    assert fp32 != bf16
    assert len(sent) == 2
    assert all("CPU BF16 Autocast" not in payload["Options"] for payload in sent)


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


def test_cached_pitch_output_preserves_every_batch_membership():
    FakeCache.reset()
    session = {"id": "session"}
    first = {"Output IDs": ["shared", "first"], "Pitches": [0, 2]}
    second = {"Output IDs": ["shared", "second"], "Pitches": [0, 4]}

    generator.write_output_metadata(
        FakeCache,
        "input",
        None,
        "shared",
        {"Pitch Shift": 0},
        session,
        batch_id="batch-1",
        batch_manifest=first,
        pitch=0,
    )
    generator.write_output_metadata(
        FakeCache,
        "input",
        None,
        "shared",
        {"Pitch Shift": 0},
        session,
        batch_id="batch-2",
        batch_manifest=second,
        pitch=0,
    )

    metadata = FakeCache.metadata[(Stage.OUTPUT, "session")]["shared"]
    assert metadata["Batch ID"] == "batch-2"
    assert metadata["Pitch Batches"] == {"batch-1": first, "batch-2": second}


def test_auto_svc3_pitch_batch_uses_cpu_and_gpu_without_forwarding_auto(monkeypatch):
    FakeCache.reset()
    sent = _configure_generation(monkeypatch)

    class MixedTab(FakeTab):
        id = "so_vits_svc_3"
        label = "so-vits-svc 3.0"
        port = 6575
        supports_native_pitch_batch = True
        supports_parallel_requests = True
        supports_mixed_device_pitch_batch = True
        serializes_device_requests = True

    monkeypatch.setattr(
        generator.inference_scheduler,
        "mixed_inference_reservations",
        lambda allow_gpu=True, serial_device_key=None, cancel_check=None: nullcontext((
            generator.inference_scheduler.DeviceReservation("", ()),
            generator.inference_scheduler.DeviceReservation(0, ()) if allow_gpu else None,
        )),
    )
    tab = MixedTab(cache_generated_output=True)

    outputs = generator.process_batch(
        FakeCache,
        None,
        "input",
        tab,
        [],
        {"id": "session"},
        "auto",
        pitch_batch_enabled=True,
        pitch_batch_values="-2,0,2",
    )

    assert len(outputs) == 3
    assert len(sent) == 2
    by_device = {payload["GPU ID"]: payload for payload in sent}
    assert set(by_device) == {"", 0}
    assert by_device[0]["Options"]["Pitch Shift"] == -2
    assert by_device[""]["Options"]["Pitch Shifts"] == [0, 2]
    assert all(payload["GPU ID"] != "auto" for payload in sent)


def test_auto_svc3_pitch_batch_uses_one_device_when_only_cpu_is_available(monkeypatch):
    FakeCache.reset()
    sent = _configure_generation(monkeypatch)

    class MixedTab(FakeTab):
        id = "so_vits_svc_3"
        label = "so-vits-svc 3.0"
        port = 6575
        supports_native_pitch_batch = True
        supports_parallel_requests = True
        supports_mixed_device_pitch_batch = True
        serializes_device_requests = True

    monkeypatch.setattr(
        generator.inference_scheduler,
        "mixed_inference_reservations",
        lambda allow_gpu=True, serial_device_key=None, cancel_check=None: nullcontext((
            generator.inference_scheduler.DeviceReservation("", ()),
            None,
        )),
    )

    outputs = generator.process_batch(
        FakeCache,
        None,
        "input",
        MixedTab(cache_generated_output=True),
        [],
        {"id": "session"},
        "auto",
        pitch_batch_enabled=True,
        pitch_batch_values="-2,0,2",
    )

    assert len(outputs) == 3
    assert len(sent) == 1
    assert sent[0]["GPU ID"] == ""
    assert sent[0]["Options"]["Pitch Shifts"] == [-2, 0, 2]


def test_auto_svc3_pitch_batch_warms_cold_cpu_and_gpu_caches_in_parallel(monkeypatch):
    FakeCache.reset()
    sent = _configure_generation(monkeypatch)

    class ColdMixedTab(FakeTab):
        id = "so_vits_svc_3"
        label = "so-vits-svc 3.0"
        port = 6575
        supports_native_pitch_batch = True
        supports_parallel_requests = True
        supports_mixed_device_pitch_batch = True
        serializes_device_requests = True

        def mixed_device_caches_are_warm(self, runtime_state, options, cpu_device, gpu_device):
            return False

    cpu_reservation = generator.inference_scheduler.DeviceReservation("", ())
    gpu_reservation = generator.inference_scheduler.DeviceReservation(0, ())
    monkeypatch.setattr(
        generator.inference_scheduler,
        "mixed_inference_reservations",
        lambda allow_gpu=True, serial_device_key=None, cancel_check=None: nullcontext((
            cpu_reservation, gpu_reservation
        )),
    )

    generator.process_batch(
        FakeCache,
        None,
        "input",
        ColdMixedTab(cache_generated_output=True),
        [],
        {"id": "session"},
        "auto",
        pitch_batch_enabled=True,
        pitch_batch_values="-2,0,2",
    )

    assert cpu_reservation._released is True
    assert gpu_reservation._released is True
    assert len(sent) == 2
    by_device = {payload["GPU ID"]: payload for payload in sent}
    assert by_device[0]["Options"]["Pitch Shift"] == -2
    assert by_device[""]["Options"]["Pitch Shifts"] == [0, 2]


def test_auto_svc3_pitch_batch_does_not_bypass_cpu_slots_when_only_gpu_is_available(monkeypatch):
    FakeCache.reset()
    sent = _configure_generation(monkeypatch)

    class MixedTab(FakeTab):
        id = "so_vits_svc_3"
        label = "so-vits-svc 3.0"
        port = 6575
        supports_native_pitch_batch = True
        supports_parallel_requests = True
        supports_mixed_device_pitch_batch = True
        serializes_device_requests = True

    monkeypatch.setattr(
        generator.inference_scheduler,
        "mixed_inference_reservations",
        lambda allow_gpu=True, serial_device_key=None, cancel_check=None: nullcontext((
            None,
            generator.inference_scheduler.DeviceReservation(0, ()) if allow_gpu else None,
        )),
    )

    generator.process_batch(
        FakeCache,
        None,
        "input",
        MixedTab(cache_generated_output=True),
        [],
        {"id": "session"},
        "auto",
        pitch_batch_enabled=True,
        pitch_batch_values="-2,0,2",
    )

    assert len(sent) == 1
    assert sent[0]["GPU ID"] == 0
    assert sent[0]["Options"]["Pitch Shifts"] == [-2, 0, 2]


def test_auto_svc4_finished_cpu_lane_refills_without_starving_gpu(monkeypatch):
    FakeCache.reset()
    _configure_generation(monkeypatch)

    class MixedTab(FakeTab):
        id = "so_vits_svc_4"
        label = "so-vits-svc 4.0"
        port = 6576
        supports_parallel_requests = True
        supports_mixed_device_pitch_batch = True
        serializes_device_requests = True

    monkeypatch.setattr(
        generator.inference_scheduler,
        "mixed_inference_reservations",
        lambda allow_gpu=True, serial_device_key=None, cancel_check=None: nullcontext((
            generator.inference_scheduler.DeviceReservation("", ()),
            generator.inference_scheduler.DeviceReservation(0, ()) if allow_gpu else None,
        )),
    )
    gpu_started = threading.Event()
    cpu_drained_queue = threading.Event()
    sent = []
    progress = []
    progress_lock = threading.Lock()

    def send(payload, *_args, **_kwargs):
        device = payload["GPU ID"]
        if device == 0:
            gpu_started.set()
            assert cpu_drained_queue.wait(timeout=2)
        else:
            assert gpu_started.wait(timeout=2)
        with progress_lock:
            sent.append(payload)
            cpu_count = sum(item["GPU ID"] == "" for item in sent)
            if cpu_count == 3:
                cpu_drained_queue.set()
        FakeCache.files.add((Stage.OUTPUT, payload["Session ID"], payload["Output File"]))

    def report(message, **details):
        with progress_lock:
            progress.append((message, details))

    monkeypatch.setattr(generator, "send_payload", send)
    outputs = generator.process_batch(
        FakeCache,
        None,
        "input",
        MixedTab(cache_generated_output=True),
        [],
        {"id": "session"},
        "auto",
        pitch_batch_enabled=True,
        pitch_batch_values="-2,-1,0,1,2",
        set_progress=report,
    )

    assert len(outputs) == 5
    assert [payload["GPU ID"] for payload in sent].count(0) == 2
    assert [payload["GPU ID"] for payload in sent].count("") == 3
    operation_events = [details for _, details in progress if details.get("operation_id")]
    operation_ids = {details["operation_id"] for details in operation_events}
    assert len(operation_ids) == 5
    pending_ids = [
        details["operation_id"] for details in operation_events
        if details["status"] == "pending"
    ]
    assert [operation_id.split(":")[1] for operation_id in pending_ids] == [
        "000", "001", "002", "003", "004"
    ]
    for operation_id in operation_ids:
        statuses = [
            details["status"] for details in operation_events
            if details["operation_id"] == operation_id
        ]
        assert statuses == ["pending", "running", "completed"]
    completed_devices = {
        details["device"] for details in operation_events
        if details["status"] == "completed"
    }
    assert completed_devices == {"CPU", "GPU #0"}


def test_serial_pitch_fallback_marks_only_the_request_in_flight_running(monkeypatch):
    FakeCache.reset()
    _configure_generation(monkeypatch)
    latest_operations = {}
    snapshots = []

    def report(_message, **details):
        if details.get("operation_id"):
            latest_operations[details["operation_label"]] = details

    def send(payload, *_args, **_kwargs):
        snapshots.append({label: details["status"] for label, details in latest_operations.items()})
        FakeCache.files.add((Stage.OUTPUT, payload["Session ID"], payload["Output File"]))

    monkeypatch.setattr(generator, "send_payload", send)
    generator.process_batch(
        FakeCache,
        None,
        "input",
        FakeTab(cache_generated_output=True),
        [],
        {"id": "session"},
        "auto",
        pitch_batch_enabled=True,
        pitch_batch_values="-1,0,1",
        set_progress=report,
    )

    assert snapshots == [
        {"Pitch -1": "running", "Pitch +0": "pending", "Pitch +1": "pending"},
        {"Pitch -1": "completed", "Pitch +0": "running", "Pitch +1": "pending"},
        {"Pitch -1": "completed", "Pitch +0": "completed", "Pitch +1": "running"},
    ]
    assert {label: details["status"] for label, details in latest_operations.items()} == {
        "Pitch -1": "completed",
        "Pitch +0": "completed",
        "Pitch +1": "completed",
    }


def test_serial_pitch_cancellation_does_not_start_pending_variants(monkeypatch):
    FakeCache.reset()
    _configure_generation(monkeypatch)
    cancelled = False
    latest_operations = {}

    def report(_message, **details):
        if details.get("operation_id"):
            latest_operations[details["operation_label"]] = details["status"]

    def send(payload, *_args, **_kwargs):
        nonlocal cancelled
        FakeCache.files.add((Stage.OUTPUT, payload["Session ID"], payload["Output File"]))
        cancelled = True

    monkeypatch.setattr(generator, "send_payload", send)
    monkeypatch.setattr(generator.generation_jobs, "is_cancel_requested", lambda *_args: cancelled)
    with pytest.raises(generator.GenerationCancelled):
        generator.process_batch(
            FakeCache,
            None,
            "input",
            FakeTab(cache_generated_output=True),
            [],
            {"id": "session", "client_id": "0" * 32, "request_id": "request"},
            "auto",
            pitch_batch_enabled=True,
            pitch_batch_values="-1,0,1",
            set_progress=report,
        )

    assert latest_operations == {
        "Pitch -1": "cancelled",
        "Pitch +0": "pending",
        "Pitch +1": "pending",
    }


def test_replicated_pitch_backend_runs_cpu_lanes_and_gpu_from_one_dynamic_queue(monkeypatch):
    FakeCache.reset()
    _configure_generation(monkeypatch)

    class ReplicatedTab(FakeTab):
        id = "so_vits_svc_5"
        label = "so-vits-svc 5.0"
        port = 6577
        supports_parallel_requests = True
        supports_mixed_device_pitch_batch = True

        def pitch_batch_request_workers(self, selected_device):
            return 4 if selected_device == "" else 1

    monkeypatch.setattr(
        generator.inference_scheduler,
        "mixed_inference_reservations",
        lambda allow_gpu=True, serial_device_key=None, cancel_check=None: nullcontext((
            generator.inference_scheduler.DeviceReservation("", ()),
            generator.inference_scheduler.DeviceReservation(0, ()) if allow_gpu else None,
        )),
    )
    cpu_lanes_ready = threading.Event()
    release_lanes = threading.Event()
    state_lock = threading.Lock()
    active_cpu = 0
    maximum_active_cpu = 0
    sent_devices = []

    def send(payload, *_args, **_kwargs):
        nonlocal active_cpu, maximum_active_cpu
        device = payload["GPU ID"]
        with state_lock:
            sent_devices.append(device)
            if device == "":
                active_cpu += 1
                maximum_active_cpu = max(maximum_active_cpu, active_cpu)
                if active_cpu == 4:
                    cpu_lanes_ready.set()
        if device == "":
            assert release_lanes.wait(timeout=2)
        else:
            assert cpu_lanes_ready.wait(timeout=2)
            release_lanes.set()
        FakeCache.files.add((Stage.OUTPUT, payload["Session ID"], payload["Output File"]))
        if device == "":
            with state_lock:
                active_cpu -= 1

    monkeypatch.setattr(generator, "send_payload", send)
    outputs = generator.process_batch(
        FakeCache,
        None,
        "input",
        ReplicatedTab(cache_generated_output=True),
        [],
        {"id": "session"},
        "auto",
        pitch_batch_enabled=True,
        pitch_batch_values="-4,-3,-2,-1,0,1,2,3,4",
    )

    assert len(outputs) == 9
    assert maximum_active_cpu == 4
    assert "" in sent_devices
    assert 0 in sent_devices


def test_mixed_replicas_leave_a_refill_for_the_first_device_to_finish(monkeypatch):
    queue = generator._VariantWorkQueue(range(5))

    gpu = generator._seed_variant_lanes(queue, 1, 1)
    cpu = generator._seed_variant_lanes(queue, 4, 1, reserve=1)

    assert gpu == ((0,),)
    assert cpu == ((1,), (2,), (3,))
    assert queue.claim(1) == (4,)


def test_large_mixed_batch_leaves_half_unclaimed_for_realtime_refills():
    queue = generator._VariantWorkQueue(range(24))

    gpu = generator._seed_variant_lanes(queue, 1, 1)
    cpu = generator._seed_variant_lanes(queue, 1, 24, reserve=12)

    assert gpu == ((0,),)
    assert cpu == (tuple(range(1, 12)),)
    assert queue.claim(24) == tuple(range(12, 24))


def test_mixed_cpu_refills_preserve_half_the_pending_work_for_gpu():
    queue = generator._VariantWorkQueue(range(12), balance_cpu_refills=True)

    assert queue.claim_for_device(12, "") == tuple(range(6))
    assert queue.claim_for_device(1, 0) == (6,)
    assert queue.claim_for_device(12, "") == (7, 8)
    assert queue.claim_for_device(12, "") == (9,)
    assert queue.claim_for_device(1, 0) == (10,)
    assert queue.claim_for_device(1, 0) == (11,)


def test_explicit_cpu_pitch_batch_uses_backend_request_workers(monkeypatch):
    FakeCache.reset()
    _configure_generation(monkeypatch)

    class ReplicatedTab(FakeTab):
        supports_parallel_requests = True

        def pitch_batch_request_workers(self, selected_device):
            assert selected_device == ""
            return 3

    lanes_ready = threading.Event()
    release_lanes = threading.Event()
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def send(payload, *_args, **_kwargs):
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 3:
                lanes_ready.set()
        assert lanes_ready.wait(timeout=2)
        release_lanes.set()
        assert release_lanes.wait(timeout=2)
        FakeCache.files.add((Stage.OUTPUT, payload["Session ID"], payload["Output File"]))
        with state_lock:
            active -= 1

    monkeypatch.setattr(generator, "send_payload", send)
    generator.process_batch(
        FakeCache,
        None,
        "input",
        ReplicatedTab(cache_generated_output=True),
        [],
        {"id": "session"},
        "",
        pitch_batch_enabled=True,
        pitch_batch_values="-2,-1,0,1,2",
    )

    assert maximum_active == 3


def test_native_cpu_claims_are_bounded_by_active_pitch_workers(monkeypatch):
    class NativeTab(FakeTab):
        supports_native_pitch_batch = True

    monkeypatch.setenv("HAY_SAY_AUTO_CPU_PITCH_VARIANTS", "10")
    monkeypatch.setenv("HAY_SAY_SVC3_CPU_PITCH_WORKERS", "5")

    assert generator._device_pitch_claim_size(NativeTab(True), "") == 5
    assert generator._device_pitch_claim_size(NativeTab(True), 0) == 1
    assert generator._device_pitch_claim_size(FakeTab(True), "") == 1


def test_native_cpu_claim_default_limits_the_nonstealable_tail(monkeypatch):
    class NativeTab(FakeTab):
        supports_native_pitch_batch = True

    monkeypatch.delenv("HAY_SAY_AUTO_CPU_PITCH_VARIANTS", raising=False)
    monkeypatch.setenv("HAY_SAY_SVC3_CPU_PITCH_WORKERS", "24")

    assert generator._device_pitch_claim_size(NativeTab(True), "") == 4


def test_single_request_lane_still_chunks_a_native_cpu_pitch_batch(monkeypatch):
    FakeCache.reset()
    sent = _configure_generation(monkeypatch)

    class NativeTab(FakeTab):
        id = "so_vits_svc_3"
        supports_native_pitch_batch = True
        supports_parallel_requests = True

    monkeypatch.setenv("HAY_SAY_AUTO_CPU_PITCH_VARIANTS", "4")
    monkeypatch.setenv("HAY_SAY_SVC3_CPU_PITCH_WORKERS", "24")

    generator.process_batch(
        FakeCache,
        None,
        "input",
        NativeTab(cache_generated_output=True),
        [],
        {"id": "session"},
        "",
        pitch_batch_enabled=True,
        pitch_batch_values="0:9:1",
    )

    assert [payload["Options"]["Pitch Shifts"] for payload in sent] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]


def test_auto_routes_cpu_only_model_to_cpu_on_gpu_host(monkeypatch):
    FakeCache.reset()
    sent = _configure_generation(monkeypatch)
    tab = FakeTab(cache_generated_output=True)
    tab.is_gpu_available = False

    generator.process_batch(FakeCache, None, "input", tab, [], {"id": "session"}, "auto")

    assert len(sent) == 1
    assert sent[0]["GPU ID"] == ""

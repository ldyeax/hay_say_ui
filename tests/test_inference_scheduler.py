from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import threading

import pytest

import inference_scheduler


def _hold_scheduler_slot(device, ready_connection, release_event):
    """Hold a real scheduler lease in a spawned process until released or killed."""
    if device == "cpu":
        lease = inference_scheduler._cpu_lease()
    else:
        lease = inference_scheduler._gpu_lease(device, blocking=True)
    try:
        ready_connection.send(True)
        ready_connection.close()
        if not release_event.wait(timeout=10):
            raise TimeoutError("parent did not release the child scheduler lease")
    finally:
        lease.release()


@pytest.fixture(autouse=True)
def scheduler_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HAY_SAY_DEVICE_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv("HAY_SAY_GPU_IDS", "0")
    monkeypatch.setenv("HAY_SAY_CPU_INFERENCE_SLOTS", "1")
    monkeypatch.setenv("HAY_SAY_GPU_INFERENCE_SLOTS", "1")
    monkeypatch.setenv("HAY_SAY_DEVICE_SLOT_TIMEOUT", "1")
    monkeypatch.setattr(
        inference_scheduler,
        "detected_gpu_status",
        lambda: {0: {"free_memory_mib": 22000, "utilization_percent": 0}},
    )


def test_auto_prefers_an_available_gpu():
    with inference_scheduler.inference_device("auto") as selected:
        assert selected == 0


def test_auto_falls_back_to_cpu_while_gpu_slot_is_busy():
    with inference_scheduler.inference_device(0):
        with inference_scheduler.inference_device("auto") as selected:
            assert selected == ""


def test_auto_falls_back_to_cpu_when_gpu_is_not_live(monkeypatch):
    monkeypatch.setattr(inference_scheduler, "detected_gpu_status", lambda: {})

    with inference_scheduler.inference_device("auto") as selected:
        assert selected == ""


def test_auto_uses_cpu_when_selected_model_does_not_support_gpu():
    with inference_scheduler.inference_device("auto", allow_gpu=False) as selected:
        assert selected == ""


def test_explicit_gpu_request_stays_strict_when_device_is_unavailable(monkeypatch):
    monkeypatch.setattr(inference_scheduler, "detected_gpu_status", lambda: {})

    with pytest.raises(inference_scheduler.DeviceUnavailableError, match="not currently available"):
        with inference_scheduler.inference_device(0):
            pass


def test_mixed_request_reserves_cpu_and_gpu_together():
    with inference_scheduler.mixed_inference_reservations() as selected:
        assert tuple(reservation.device for reservation in selected) == ("", 0)


def test_mixed_request_keeps_cpu_when_gpu_is_busy():
    with inference_scheduler.inference_device(0):
        with inference_scheduler.mixed_inference_reservations() as selected:
            assert selected[0].device == ""
            assert selected[1] is None


def test_mixed_request_keeps_gpu_when_cpu_is_busy():
    with inference_scheduler.inference_device(""):
        with inference_scheduler.mixed_inference_reservations() as selected:
            assert selected[0] is None
            assert selected[1].device == 0


def test_completed_gpu_work_releases_its_slot_while_mixed_cpu_work_continues():
    cpu_started = threading.Event()
    release_cpu = threading.Event()

    def use_cpu(reservation):
        with reservation:
            cpu_started.set()
            assert release_cpu.wait(timeout=1)

    def use_gpu(reservation):
        with reservation:
            assert cpu_started.wait(timeout=1)

    with inference_scheduler.mixed_inference_reservations() as (cpu_reservation, gpu_reservation):
        with ThreadPoolExecutor(max_workers=2) as executor:
            cpu_future = executor.submit(use_cpu, cpu_reservation)
            gpu_future = executor.submit(use_gpu, gpu_reservation)
            gpu_future.result(timeout=1)

            next_gpu = inference_scheduler._gpu_lease(0, blocking=False)
            try:
                assert next_gpu is not None
            finally:
                if next_gpu is not None:
                    next_gpu.release()
                release_cpu.set()
            cpu_future.result(timeout=1)


def test_mixed_cpu_reservation_error_releases_gpu(monkeypatch):
    def fail_cpu_reservation(*_args, **_kwargs):
        raise RuntimeError("CPU reservation failed")

    monkeypatch.setattr(inference_scheduler, "_cpu_lease", fail_cpu_reservation)

    with pytest.raises(RuntimeError, match="CPU reservation failed"):
        with inference_scheduler.mixed_inference_reservations():
            pass

    recovered_gpu = inference_scheduler._gpu_lease(0, blocking=False)
    assert recovered_gpu is not None
    recovered_gpu.release()


def test_blocked_device_admission_observes_cancellation(monkeypatch):
    checks = []

    def cancel_after_one_poll():
        checks.append(True)
        if len(checks) > 1:
            raise RuntimeError("generation cancelled")

    monkeypatch.setattr(inference_scheduler, "_try_paths", lambda *_args: None)
    monkeypatch.setattr(inference_scheduler.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="generation cancelled"):
        inference_scheduler._wait_for_paths(
            [object()], "", cancel_check=cancel_after_one_poll
        )

    assert len(checks) == 2


def test_device_reservation_attempts_every_release_after_an_error():
    released = []

    class Lease:
        def __init__(self, name, fails=False):
            self.name = name
            self.fails = fails

        def release(self):
            released.append(self.name)
            if self.fails:
                raise OSError("unlock failed")

    reservation = inference_scheduler.DeviceReservation(
        "",
        (Lease("lane"), Lease("capacity", fails=True)),
    )

    with pytest.raises(OSError, match="unlock failed"):
        reservation.release()

    assert released == ["capacity", "lane"]


def test_serial_device_lane_does_not_consume_extra_global_cpu_slots(monkeypatch):
    monkeypatch.setenv("HAY_SAY_CPU_INFERENCE_SLOTS", "2")
    first = inference_scheduler._cpu_lease(serial_device_key="so_vits_svc_3")
    second_same_runtime = None
    other_runtime = None
    try:
        second_same_runtime = inference_scheduler._cpu_lease(
            blocking=False,
            serial_device_key="so_vits_svc_3",
        )
        other_runtime = inference_scheduler._cpu_lease(
            blocking=False,
            serial_device_key="gpt_so_vits",
        )
        assert second_same_runtime is None
        assert other_runtime is not None
    finally:
        if other_runtime is not None:
            other_runtime.release()
        first.release()


def test_spawned_gpu_holder_forces_auto_request_to_fall_back_to_cpu():
    context = multiprocessing.get_context("spawn")
    ready_reader, ready_writer = context.Pipe(duplex=False)
    release_child = context.Event()
    child = context.Process(
        target=_hold_scheduler_slot,
        args=(0, ready_writer, release_child),
    )
    child.start()
    ready_writer.close()
    try:
        assert ready_reader.poll(5), "child did not acquire the GPU scheduler slot"
        assert ready_reader.recv() is True

        with inference_scheduler.inference_device("auto") as selected:
            assert selected == ""
    finally:
        release_child.set()
        child.join(timeout=5)
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        ready_reader.close()

    assert child.exitcode == 0
    with inference_scheduler.inference_device("auto") as selected:
        assert selected == 0


def test_spawned_child_normal_completion_releases_cpu_slot():
    context = multiprocessing.get_context("spawn")
    ready_reader, ready_writer = context.Pipe(duplex=False)
    release_child = context.Event()
    child = context.Process(
        target=_hold_scheduler_slot,
        args=("cpu", ready_writer, release_child),
    )
    child.start()
    ready_writer.close()
    try:
        assert ready_reader.poll(5), "child did not acquire the CPU scheduler slot"
        assert ready_reader.recv() is True
        assert inference_scheduler._cpu_lease(blocking=False) is None

        release_child.set()
        child.join(timeout=5)
        assert not child.is_alive(), "child did not complete after releasing its slot"
        assert child.exitcode == 0

        recovered_lease = inference_scheduler._cpu_lease(blocking=False)
        assert recovered_lease is not None
        recovered_lease.release()
    finally:
        release_child.set()
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        ready_reader.close()


def test_kernel_releases_cpu_slot_when_spawned_child_is_terminated():
    context = multiprocessing.get_context("spawn")
    ready_reader, ready_writer = context.Pipe(duplex=False)
    release_child = context.Event()
    child = context.Process(
        target=_hold_scheduler_slot,
        args=("cpu", ready_writer, release_child),
    )
    child.start()
    ready_writer.close()
    try:
        assert ready_reader.poll(5), "child did not acquire the CPU scheduler slot"
        assert ready_reader.recv() is True
        assert inference_scheduler._cpu_lease(blocking=False) is None

        child.terminate()
        child.join(timeout=5)
        assert not child.is_alive(), "terminated child did not exit"
        assert child.exitcode is not None and child.exitcode != 0

        recovered_lease = inference_scheduler._cpu_lease(blocking=False)
        assert recovered_lease is not None
        recovered_lease.release()
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)
        ready_reader.close()

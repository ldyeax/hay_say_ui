from hay_say_common.worker_control import WorkerControl


def test_cancel_after_queue_before_begin_is_retained():
    control = WorkerControl()
    control.submit({"action": "generate", "request_id": "job"})
    control.submit({"action": "cancel", "request_id": "job"})

    command = control.commands.get_nowait()
    control.begin(command["request_id"])

    assert control.cancelled
    control.finish("job")
    assert not control.cancelled


def test_pending_cancel_only_applies_to_its_request():
    control = WorkerControl()
    control.submit({"action": "cancel", "request_id": "later"})
    control.begin("current")
    assert not control.cancelled
    control.finish("current")

    control.begin("later")
    assert control.cancelled
    control.finish("later")


def test_late_cancellation_ids_are_bounded_for_long_lived_workers():
    control = WorkerControl(max_pending_cancellations=3)

    for index in range(10):
        control.submit({"action": "cancel", "request_id": f"late-{index}"})

    assert control._pending_cancellations == {"late-7", "late-8", "late-9"}
    assert list(control._pending_order) == ["late-7", "late-8", "late-9"]

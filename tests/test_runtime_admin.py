from runtime_admin import _runtime_row


def test_error_runtime_with_live_pid_can_be_stopped_or_restarted():
    row = _runtime_row({"id": "rvc", "status": "error", "pid": 1234})
    buttons = row.children[-1].children.children
    states = {button.id["action"]: button.disabled for button in buttons}

    assert states == {"start": True, "stop": False, "restart": False}

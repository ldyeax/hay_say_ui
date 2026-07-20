from unittest.mock import Mock

import pytest

import runtime_client


def _response(body, status=200):
    response = Mock()
    response.ok = status < 400
    response.status_code = status
    response.json.return_value = body
    return response


def test_service_endpoint_switches_to_loopback_in_native_mode(monkeypatch):
    monkeypatch.setenv("HAY_SAY_NATIVE", "1")
    assert runtime_client.service_endpoint("rvc", 6578) == ("127.0.0.1", 6578)


def test_service_runtime_state_returns_direct_model_telemetry(monkeypatch):
    get = Mock(return_value=_response({"loaded_model_details": []}))
    monkeypatch.setattr(runtime_client.requests, "get", get)

    assert runtime_client.service_runtime_state("127.0.0.1", 6575) == {"loaded_model_details": []}
    get.assert_called_once_with("http://127.0.0.1:6575/runtime", timeout=0.5)


@pytest.mark.parametrize("response", [_response([], 200), _response({}, 503)])
def test_service_runtime_state_ignores_malformed_or_unavailable_telemetry(monkeypatch, response):
    monkeypatch.setattr(runtime_client.requests, "get", Mock(return_value=response))

    assert runtime_client.service_runtime_state("127.0.0.1", 6575) is None


def test_ensure_runtime_starts_and_waits_until_ready(monkeypatch):
    monkeypatch.setenv("HAY_SAY_NATIVE", "1")
    responses = iter([
        _response({"id": "rvc", "status": "stopped"}),
        _response({"id": "rvc", "status": "starting"}, 202),
        _response({"id": "rvc", "status": "ready-cold"}),
    ])
    request = Mock(side_effect=lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(runtime_client.requests, "request", request)
    monkeypatch.setattr(runtime_client.time, "sleep", lambda _: None)

    runtime_client.ensure_runtime_started("rvc", 6578, timeout=1)

    assert [call.args[0] for call in request.call_args_list] == ["GET", "POST", "GET"]


def test_manager_http_errors_are_actionable(monkeypatch):
    monkeypatch.setattr(runtime_client.requests, "request", Mock(return_value=_response({"error": "disabled"}, 409)))
    with pytest.raises(runtime_client.RuntimeManagerError, match="disabled"):
        runtime_client.runtime_action("rvc", "start")


def test_generation_cancel_targets_model_service_without_stopping_runtime(monkeypatch):
    monkeypatch.setenv("HAY_SAY_NATIVE", "1")
    manager_request = Mock(return_value=_response({"id": "so_vits_svc_5", "port": 6577}))
    service_post = Mock(return_value=_response({"cancelled": ["request-a"]}))
    monkeypatch.setattr(runtime_client.requests, "request", manager_request)
    monkeypatch.setattr(runtime_client.requests, "post", service_post)

    assert runtime_client.cancel_generation("so_vits_svc_5", ["request-a", "request-a"])
    manager_request.assert_called_once_with(
        "GET", runtime_client.manager_url() + "/runtimes/so_vits_svc_5", timeout=3.0
    )
    service_post.assert_called_once_with(
        "http://127.0.0.1:6577/cancel",
        json={"Request IDs": ["request-a"]},
        timeout=2.0,
    )


def test_legacy_runtime_without_cancel_endpoint_falls_back_to_boundary_checks(monkeypatch):
    monkeypatch.setattr(
        runtime_client,
        "runtime_status",
        lambda _runtime_id: {"port": 6578},
    )
    monkeypatch.setattr(runtime_client.requests, "post", Mock(return_value=_response({}, 404)))

    assert runtime_client.cancel_generation("rvc", ["request-a"]) is False


def test_blocked_generation_file_lock_observes_cancellation(monkeypatch, tmp_path):
    checks = []

    def fake_flock(_descriptor, operation):
        if operation & runtime_client.fcntl.LOCK_NB:
            raise BlockingIOError

    def cancel_after_one_poll():
        checks.append(True)
        if len(checks) > 1:
            raise RuntimeError("generation cancelled")

    monkeypatch.setenv("HAY_SAY_REQUEST_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(runtime_client.fcntl, "flock", fake_flock)
    monkeypatch.setattr(runtime_client.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="generation cancelled"):
        with runtime_client.generation_lock("rvc", cancel_check=cancel_after_one_poll):
            pass

    assert len(checks) == 2

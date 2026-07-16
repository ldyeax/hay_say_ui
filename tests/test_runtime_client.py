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

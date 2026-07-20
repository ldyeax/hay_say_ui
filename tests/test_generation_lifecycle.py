import inspect
from types import SimpleNamespace
from pathlib import Path

import pytest
from dash import _callback
from dash.exceptions import PreventUpdate

import generation_jobs
import generator
import main
import plotly_celery_common
import postprocessed_display
import runtime_client


CLIENT_A = "a" * 32
CLIENT_B = "b" * 32


@pytest.fixture(autouse=True)
def job_state_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HAY_SAY_STATE_DIR", str(tmp_path / "state"))


def _snapshot(**overrides):
    snapshot = {
        "hardware_selection": "CPU",
        "user_text": "Persisted text",
        "selected_file": "input.wav",
        "semitone_pitch": 0,
        "debug_pitch": [],
        "reduce_noise": [],
        "crop_silence": [],
        "reduce_metallic_noise": [],
        "auto_tune_output": [],
        "output_speed_adjustment": 1,
        "pitch_batch_enabled": True,
        "pitch_batch_values": "-2:2",
        "hidden_states": [False],
        "architecture_inputs": [],
    }
    snapshot.update(overrides)
    return snapshot


def _request(client_id=CLIENT_A, runtime_id="rvc", queue="cpu", snapshot=None):
    request_id = f"request-{client_id[0]}"
    generation_jobs.create_queued(client_id, request_id, runtime_id, queue, "Queued")
    return {
        "request_id": request_id,
        "client_id": client_id,
        "session_id": "cache-session",
        "snapshot": snapshot or _snapshot(),
    }


def _component_with_id(nodes, component_id):
    if not isinstance(nodes, (list, tuple)):
        nodes = [nodes]
    for node in nodes:
        if getattr(node, "id", None) == component_id:
            return node
        children = getattr(node, "children", None)
        if children is not None:
            found = _component_with_id(children, component_id)
            if found is not None:
                return found
    return None


def _component_with_class(nodes, class_name):
    if not isinstance(nodes, (list, tuple)):
        nodes = [nodes]
    for node in nodes:
        classes = (getattr(node, "className", None) or "").split()
        if class_name in classes:
            return node
        children = getattr(node, "children", None)
        if children is not None:
            found = _component_with_class(children, class_name)
            if found is not None:
                return found
    return None


def _components_by_type(nodes, component_type):
    if not isinstance(nodes, (list, tuple)):
        nodes = [nodes]
    matches = []
    for node in nodes:
        if not hasattr(node, "to_plotly_json"):
            continue
        if node.to_plotly_json().get("type") == component_type:
            matches.append(node)
        children = getattr(node, "children", None)
        if children is not None:
            matches.extend(_components_by_type(children, component_type))
    return matches


def _contains_component_id(nodes, component_id):
    return _component_with_id(nodes, component_id) is not None


def test_fresh_page_selects_first_architecture_and_marks_its_tab_cell():
    tabs = [SimpleNamespace(id="first"), SimpleNamespace(id="second")]

    assert main.tab_visibility_and_classes(None, tabs) == [
        False,
        True,
        "tab-cell-selected",
        "tab-cell",
    ]
    assert main.tab_visibility_and_classes("second-tab-button", tabs) == [
        True,
        False,
        "tab-cell",
        "tab-cell-selected",
    ]


def test_bf16_precision_is_not_exposed_in_the_frontend():
    layout = main.construct_main_interface([], [], enable_session_caches=True)

    assert _component_with_id(layout, "cpu-bf16-autocast") is None


def test_generation_poll_has_the_postprocessed_output_renderer_bound():
    assert (
        main.prepare_postprocessed_display
        is postprocessed_display.prepare_postprocessed_display
    )


def test_app_declares_a_device_width_mobile_viewport():
    app = main.construct_app_layout(False, "file", [], False)
    index = app.index()

    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in index


def test_layout_exposes_scoped_mobile_overflow_boundaries():
    tabs = [SimpleNamespace(id="first", label="A deliberately long architecture name")]
    layout = main.construct_main_interface(main.construct_tab_buttons(tabs), [], True)

    tab_strip = _component_with_class(layout, "architecture-tab-strip")
    output_controls = _component_with_class(layout, "output-controls")

    assert tab_strip is not None
    assert tab_strip.children.className == "tab-table-header"
    assert output_controls is not None
    assert _component_with_class(output_controls, "output-format-control") is not None


def test_generate_button_validation_is_not_driven_by_generation_poll():
    original_callback_map = dict(_callback.GLOBAL_CALLBACK_MAP)
    original_callback_list = list(_callback.GLOBAL_CALLBACK_LIST)
    original_architecture_map = plotly_celery_common._memoized_architecture_map
    try:
        _callback.GLOBAL_CALLBACK_MAP.clear()
        _callback.GLOBAL_CALLBACK_LIST.clear()
        plotly_celery_common._memoized_architecture_map = None
        plotly_celery_common.construct_architecture_tabs(["SoVitsSvc3"], "file")

        main.register_main_callbacks(False, "file", ["SoVitsSvc3"])

        matches = [
            callback
            for output, callback in _callback.GLOBAL_CALLBACK_MAP.items()
            if "generate-button-gpu.disabled" in output
        ]
        assert len(matches) == 1
        inputs = {
            (dependency["id"], dependency["property"])
            for dependency in matches[0]["inputs"]
        }
        assert ("generation-poll", "n_intervals") not in inputs

        poll_owners = [
            callback
            for output, callback in _callback.GLOBAL_CALLBACK_MAP.items()
            if "generation-poll.disabled" in output
        ]
        assert len(poll_owners) == 1
        assert poll_owners[0]["inputs"] == [
            {"id": "generation-job-state", "property": "data"}
        ]

        queue_callbacks = [
            output
            for output, callback in _callback.GLOBAL_CALLBACK_MAP.items()
            if (
                "cpu-generation-request.data" in output
                or "gpu-generation-request.data" in output
            )
            and "generation-job-state.data" in output
            and any(
                dependency["id"] in {"generate-button-cpu", "generate-button-gpu"}
                for dependency in callback["inputs"]
            )
        ]
        assert len(queue_callbacks) == 2
        assert all("generation-poll.disabled" not in output for output in queue_callbacks)
    finally:
        _callback.GLOBAL_CALLBACK_MAP.clear()
        _callback.GLOBAL_CALLBACK_MAP.update(original_callback_map)
        _callback.GLOBAL_CALLBACK_LIST[:] = original_callback_list
        plotly_celery_common._memoized_architecture_map = original_architecture_map


def test_generate_buttons_are_not_replaced_by_loading_components():
    layout = main.construct_main_interface([], [], enable_session_caches=True)

    for loading in _components_by_type(layout, "Loading"):
        assert not _contains_component_id(loading.children, "generate-button-cpu")
        assert not _contains_component_id(loading.children, "generate-button-gpu")


def test_terminal_jobs_clear_persisted_generation_triggers():
    cpu_request = {"request_id": "cpu-request"}
    gpu_request = {"request_id": "gpu-request"}

    assert main.generation_trigger_updates(
        {"status": "completed"}, cpu_request, gpu_request
    ) == (None, None)
    assert main.generation_trigger_updates(
        {"status": "failed"}, None, gpu_request
    ) == (main.no_update, None)


def test_browser_job_view_excludes_the_persisted_request_snapshot():
    state = {
        "request_id": "request",
        "runtime_id": "rvc",
        "status": "running",
        "message": "Generating",
        "progress": {"current": 1, "total": 2},
        "operations": {"cpu-0": {"label": "CPU worker"}},
        "updated_at": "now",
        "request_data": {"snapshot": {"large": "payload"}},
    }

    assert main.generation_job_view(state) == {
        "request_id": "request",
        "runtime_id": "rvc",
        "status": "running",
        "message": "Generating",
        "progress": {"current": 1, "total": 2},
        "operations": {"cpu-0": {"label": "CPU worker"}},
        "updated_at": "now",
    }


def test_active_job_immediately_shows_stop_and_individual_progress_rows():
    state = {
        "status": "running",
        "message": "Generating pitch variants...",
        "operations": {
            "gpu-0": {
                "label": "GPU worker",
                "device": "GPU #0",
                "status": "running",
                "current": 2,
                "total": 5,
            },
            "cpu-0": {
                "label": "CPU worker",
                "device": "CPU",
                "status": "completed",
                "current": 4,
                "total": 4,
            },
        },
    }

    message, message_hidden, stop_hidden, stop_disabled, rows, progress_hidden, poll_disabled = (
        main.generation_controls_view(state)
    )

    assert message == "Generating pitch variants..."
    assert message_hidden is False
    assert stop_hidden is False
    assert stop_disabled is False
    assert progress_hidden is False
    assert poll_disabled is False
    assert len(rows) == 2
    serialized = [row.to_plotly_json()["props"] for row in rows]
    assert [props["children"][0].children[0].children for props in serialized] == [
        "CPU worker - CPU",
        "GPU worker - GPU #0",
    ]
    assert [props["children"][1].value for props in serialized] == [4, 2]


def test_running_zero_progress_is_indeterminate_instead_of_appearing_stuck():
    rows = main.generation_progress_components({
        "operations": {
            "pitch-queued": {
                "label": "Pitch +1",
                "status": "pending",
                "current": 0,
                "total": 1,
            },
            "pitch-active": {
                "label": "Pitch +2",
                "device": "CPU",
                "status": "running",
                "current": 0,
                "total": 1,
            },
        },
    })

    active, queued = [row.to_plotly_json()["props"] for row in rows]
    assert active["children"][0].children[1].children == "Running"
    assert "value" not in active["children"][1].to_plotly_json()["props"]
    assert queued["children"][0].children[1].children == "Waiting"
    assert queued["children"][1].value == 0


def test_stop_button_remains_available_to_retry_cooperative_cancellation():
    view = main.generation_controls_view({
        "status": "cancelling",
        "message": "Cancelling generation...",
        "operations": {},
    })

    assert view[1] is False
    assert view[2] is False
    assert view[3] is False
    assert view[6] is False


def test_browser_merge_script_prefers_terminal_state_when_timestamps_tie():
    source = inspect.getsource(main.register_main_callbacks)

    assert 'const polledRank = rank(polled);' in source
    assert 'if (polledRank > currentRank)' in source
    assert '(polled.updated_at || "") >= (current.updated_at || "")' in source


def test_terminal_job_stops_polling_and_hides_progress_controls():
    assert main.generation_controls_view({
        "status": "failed",
        "message": "Runtime failed",
        "operations": {
            "cpu": {
                "label": "CPU",
                "status": "failed",
                "current": 1,
                "total": 2,
            }
        },
    }) == ("Runtime failed", False, True, True, [], True, True)


def test_mobile_css_contains_tabs_and_stacks_wide_table_cells():
    stylesheet = (Path(__file__).parents[1] / "assets" / "haySayStyle.css").read_text()

    assert ".architecture-tab-strip" in stylesheet
    assert "overflow-x: auto" in stylesheet
    assert ".tab-table tr[id]:not([hidden]) > td" in stylesheet
    assert ".output-controls-row > td" in stylesheet
    assert ".generation-progress-bar" in stylesheet
    assert "margin: 0 auto" in stylesheet


def test_refresh_preserves_identity_request_snapshot_and_trigger_storage(monkeypatch):
    session = {"id": "cache-session", "client_id": CLIENT_A}
    refreshed = main.normalize_session_data(session, enable_session_caches=True)
    assert refreshed == session

    request = main.create_generation_request(
        "cpu", refreshed, SimpleNamespace(id="rvc"), _snapshot(user_text="before refresh")
    )
    queued = main.browser_job(refreshed)
    assert queued["request_id"] == request["request_id"]
    assert request["snapshot"]["user_text"] == "before refresh"

    recovered_cpu, recovered_gpu = main.recover_generation_triggers(queued, None, None)
    assert recovered_cpu == request
    assert recovered_gpu is main.no_update
    assert main.recover_generation_triggers(queued, request, None) == (
        main.no_update,
        main.no_update,
    )

    layout = main.construct_main_interface([], [], enable_session_caches=True)
    assert _component_with_id(layout, "session").storage_type == "session"
    assert _component_with_id(layout, "cpu-generation-request").storage_type == "session"
    assert _component_with_id(layout, "gpu-generation-request").storage_type == "session"

    captured = {}

    def generate_from_snapshot(*args, **_kwargs):
        captured["session"] = args[2]
        captured["text"] = args[4]
        captured["file"] = args[5]
        return ["resumed-output"]

    monkeypatch.setattr(generator, "generate", generate_from_snapshot)
    generator.generate_and_prepare_postprocessed_display(
        request, "Generating", "file", "", [SimpleNamespace(id="rvc")]
    )
    assert captured == {
        "session": {"id": "cache-session", "client_id": CLIENT_A, "request_id": request["request_id"]},
        "text": "before refresh",
        "file": "input.wav",
    }


def test_each_completed_enqueue_gets_a_unique_request_id(monkeypatch):
    identifiers = iter(("1" * 32, "2" * 32))
    monkeypatch.setattr(main.uuid, "uuid4", lambda: SimpleNamespace(hex=next(identifiers)))
    session = {"id": None, "client_id": CLIENT_A}
    tab = SimpleNamespace(id="rvc")

    first = main.create_generation_request("cpu", session, tab, _snapshot())
    with pytest.raises(PreventUpdate):
        main.create_generation_request("gpu", session, tab, _snapshot())

    generation_jobs.mark_completed(CLIENT_A, first["request_id"])
    second = main.create_generation_request("gpu", session, tab, _snapshot())

    assert first["request_id"] == "1" * 32
    assert second["request_id"] == "2" * 32
    assert first["request_id"] != second["request_id"]
    assert second["session_id"] is None


class _Control:
    def __init__(self, name, events, failure=None):
        self.name = name
        self.events = events
        self.failure = failure

    def revoke(self, task_id, **options):
        self.events.append(("revoke", self.name, task_id, options))
        if self.failure is not None:
            raise self.failure


def _celery_app(name, events, failure=None):
    return SimpleNamespace(control=_Control(name, events, failure))


def test_stop_cancels_only_the_browser_job_and_keeps_runtime_alive():
    generation_jobs.create_queued(CLIENT_A, "request-a", "rvc", "cpu", "Queued")
    generation_jobs.claim_running(CLIENT_A, "request-a", "cpu-task")
    generation_jobs.create_queued(CLIENT_B, "request-b", "rvc", "gpu", "Queued")
    generation_jobs.claim_running(CLIENT_B, "request-b", "gpu-task")
    events = []

    def cancel_runtime(runtime_id, request_ids):
        assert generation_jobs.get(CLIENT_A)["status"] == "cancelling"
        assert generation_jobs.get(CLIENT_B)["status"] == "running"
        events.append(("cancel", runtime_id, request_ids))

    result = main.cancel_browser_generation(
        {"client_id": CLIENT_A},
        runtime_cancel=cancel_runtime,
        celery_apps={
            "cpu": _celery_app("cpu", events),
            "gpu": _celery_app("gpu", events),
        },
    )

    assert events[0] == ("cancel", "rvc", ["request-a"])
    assert events[1] == ("revoke", "cpu", "cpu-task", {"terminate": False})
    assert result == {"request_id": "request-a", "runtime_id": "rvc"}
    assert generation_jobs.get(CLIENT_A)["status"] == "cancelling"
    assert generation_jobs.get(CLIENT_B)["status"] == "running"


def test_revoke_failure_leaves_durable_cooperative_cancellation_requested():
    request = _request()
    generation_jobs.claim_running(CLIENT_A, request["request_id"], "task")

    result = main.cancel_browser_generation(
        {"client_id": CLIENT_A},
        runtime_cancel=lambda *_: None,
        celery_apps={"cpu": _celery_app("cpu", [], RuntimeError("broker unavailable"))},
    )

    assert result["runtime_id"] == "rvc"
    assert generation_jobs.get(CLIENT_A)["status"] == "cancelling"


def test_runtime_cancel_failure_remains_cancelling_for_an_explicit_retry():
    request = _request()
    generation_jobs.claim_running(CLIENT_A, request["request_id"], "task")
    events = []

    def fail_cancel(*_):
        events.append("runtime")
        raise RuntimeError("manager unavailable")

    with pytest.raises(RuntimeError, match="manager unavailable"):
        main.cancel_browser_generation(
            {"client_id": CLIENT_A},
            runtime_cancel=fail_cancel,
            celery_apps={"cpu": _celery_app("cpu", events)},
        )

    assert events == ["runtime"]
    assert generation_jobs.get(CLIENT_A)["status"] == "cancelling"


def test_runtime_manager_failure_schedules_bounded_cooperative_retries(monkeypatch):
    request = _request()
    generation_jobs.claim_running(CLIENT_A, request["request_id"], "task")
    scheduled = []

    monkeypatch.setattr(
        main,
        "_schedule_runtime_cancel_retry",
        lambda *arguments: scheduled.append(arguments),
    )

    def fail_cancel(*_):
        raise runtime_client.RuntimeManagerUnavailable("manager unavailable")

    result = main.cancel_browser_generation(
        {"client_id": CLIENT_A},
        runtime_cancel=fail_cancel,
        celery_apps={"cpu": _celery_app("cpu", [])},
    )

    assert result["request_id"] == request["request_id"]
    assert scheduled == [(CLIENT_A, "rvc", request["request_id"], fail_cancel)]


def test_cooperative_cancel_retry_stops_after_endpoint_acknowledges():
    request = _request()
    generation_jobs.claim_running(CLIENT_A, request["request_id"], "task")
    generation_jobs.request_cancel(CLIENT_A, request["request_id"])
    attempts = []

    def eventually_available(runtime_id, request_ids):
        attempts.append((runtime_id, request_ids))
        if len(attempts) == 1:
            raise runtime_client.RuntimeManagerUnavailable("not yet")

    acknowledged = main._retry_runtime_cancel(
        CLIENT_A,
        "rvc",
        request["request_id"],
        eventually_available,
        delays=(0, 0),
        sleeper=lambda _delay: None,
    )

    assert acknowledged is True
    assert len(attempts) == 2


def test_stop_marks_a_never_started_task_cancelled_without_killing_workers():
    request = _request()
    events = []

    result = main.cancel_browser_generation(
        {"client_id": CLIENT_A},
        runtime_cancel=lambda runtime_id, ids: events.append((runtime_id, ids)),
        celery_apps={"cpu": _celery_app("cpu", events)},
    )

    assert result == {"request_id": request["request_id"], "runtime_id": "rvc"}
    assert events == [("rvc", [request["request_id"]])]
    assert generation_jobs.get(CLIENT_A)["status"] == "cancelled"


def test_worker_success_and_failure_set_durable_terminal_states(monkeypatch):
    tab = SimpleNamespace(id="rvc")
    request = _request()
    monkeypatch.setattr(generator, "generate", lambda *_args, **_kwargs: ["output-hash"])

    result = generator.generate_and_prepare_postprocessed_display(
        request, "Generating", "file", "", [tab]
    )

    assert result[0]["outputs"] == ["output-hash"]
    assert generation_jobs.get(CLIENT_A)["status"] == "completed"

    failed_request = _request(client_id=CLIENT_B)

    def fail_generation(*_args, **_kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr(generator, "generate", fail_generation)
    with pytest.raises(RuntimeError, match="model failed"):
        generator.generate_and_prepare_postprocessed_display(
            failed_request, "Generating", "file", "", [tab]
        )

    failed = generation_jobs.get(CLIENT_B)
    assert failed["status"] == "failed"
    assert failed["message"] == "RuntimeError: model failed"
    assert failed["finished_at"] is not None


def test_invalid_persisted_snapshot_becomes_a_failed_job():
    request = _request(snapshot={"hidden_states": [False], "architecture_inputs": []})

    with pytest.raises(ValueError, match="snapshot is missing"):
        generator.generate_and_prepare_postprocessed_display(
            request, "Generating", "file", "", [SimpleNamespace(id="rvc")]
        )

    state = generation_jobs.get(CLIENT_A)
    assert state["status"] == "failed"
    assert state["message"].startswith("ValueError: Generation request snapshot is missing")


def test_duplicate_refresh_task_cannot_repeat_inference(monkeypatch):
    calls = []
    request = _request()
    tab = SimpleNamespace(id="rvc")

    def generate_once(*_args, **_kwargs):
        calls.append("generate")
        return ["output-hash"]

    monkeypatch.setattr(generator, "generate", generate_once)
    generator.generate_and_prepare_postprocessed_display(request, "Generating", "file", "", [tab])

    with pytest.raises(generator.GenerationRequestUnavailable):
        generator.generate_and_prepare_postprocessed_display(
            request, "Generating", "file", "", [tab]
        )

    assert calls == ["generate"]


def test_cancellation_wins_after_inference_returns(monkeypatch):
    request = _request()
    tab = SimpleNamespace(id="rvc")

    def cancel_during_generation(*_args, **_kwargs):
        generation_jobs.request_cancel(CLIENT_A, request["request_id"])
        return ["partial-output"]

    monkeypatch.setattr(generator, "generate", cancel_during_generation)
    with pytest.raises(generator.GenerationCancelled):
        generator.generate_and_prepare_postprocessed_display(
            request, "Generating", "file", "", [tab]
        )

    state = generation_jobs.get(CLIENT_A)
    assert state["status"] == "cancelled"
    assert state["finished_at"] is not None


def test_backend_cancellation_response_finishes_the_durable_job(monkeypatch):
    request = _request()
    tab = SimpleNamespace(id="rvc")

    def fail_at_cancel_boundary(*_args, **_kwargs):
        generation_jobs.request_cancel(CLIENT_A, request["request_id"])
        raise RuntimeError("model service returned HTTP 409")

    monkeypatch.setattr(generator, "generate", fail_at_cancel_boundary)
    with pytest.raises(generator.GenerationCancelled):
        generator.generate_and_prepare_postprocessed_display(
            request, "Generating", "file", "", [tab]
        )

    state = generation_jobs.get(CLIENT_A)
    assert state["status"] == "cancelled"
    assert state["finished_at"] is not None


def test_cancellation_winning_the_completion_lock_finishes_cancelled(monkeypatch):
    request = _request()
    tab = SimpleNamespace(id="rvc")
    original_mark_completed = generation_jobs.mark_completed

    def cancel_before_completion(client_id, request_id, *args, **kwargs):
        generation_jobs.request_cancel(client_id, request_id)
        return original_mark_completed(client_id, request_id, *args, **kwargs)

    monkeypatch.setattr(generator, "generate", lambda *_args, **_kwargs: ["partial-output"])
    monkeypatch.setattr(generation_jobs, "mark_completed", cancel_before_completion)

    with pytest.raises(generator.GenerationCancelled):
        generator.generate_and_prepare_postprocessed_display(
            request, "Generating", "file", "", [tab]
        )

    assert generation_jobs.get(CLIENT_A)["status"] == "cancelled"


def test_cancellation_winning_the_failure_lock_finishes_cancelled(monkeypatch):
    request = _request()
    tab = SimpleNamespace(id="rvc")
    original_mark_failed = generation_jobs.mark_failed

    def fail_generation(*_args, **_kwargs):
        raise RuntimeError("model failed at cancellation boundary")

    def cancel_before_failure(client_id, request_id, message):
        generation_jobs.request_cancel(client_id, request_id)
        return original_mark_failed(client_id, request_id, message)

    monkeypatch.setattr(generator, "generate", fail_generation)
    monkeypatch.setattr(generation_jobs, "mark_failed", cancel_before_failure)

    with pytest.raises(generator.GenerationCancelled):
        generator.generate_and_prepare_postprocessed_display(
            request, "Generating", "file", "", [tab]
        )

    assert generation_jobs.get(CLIENT_A)["status"] == "cancelled"

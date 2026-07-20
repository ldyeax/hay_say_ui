from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import threading

import pytest

import generation_jobs


CLIENT_A = "a" * 32
CLIENT_B = "b" * 32
CLIENT_C = "c" * 32


def _write_progress_in_process(client_id, prefix):
    for index in range(25):
        generation_jobs.update_progress(client_id, "request", f"{prefix}-{index}", index, 25)


@pytest.fixture(autouse=True)
def job_state_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HAY_SAY_STATE_DIR", str(tmp_path / "state"))


def test_job_lifecycle_is_durable_and_idempotent():
    queued = generation_jobs.create_queued(CLIENT_A, "request-1", "so_vits_svc_3", "cpu", "Queued")
    assert queued["status"] == "queued"
    assert generation_jobs.active(queued)
    assert generation_jobs.get(CLIENT_A) == queued

    running = generation_jobs.mark_running(CLIENT_A, "request-1", "task-1", "Starting")
    assert running["status"] == "running"
    assert running["task_id"] == "task-1"
    assert running["started_at"] is not None

    progressed = generation_jobs.update_progress(CLIENT_A, "request-1", "Variant 3 of 12", 3, 12)
    assert progressed["progress"] == {"current": 3, "total": 12}
    assert progressed["message"] == "Variant 3 of 12"

    progressed = generation_jobs.update_operation_progress(
        CLIENT_A,
        "request-1",
        "cpu-1",
        "CPU worker 1",
        3,
        12,
        device="CPU",
    )
    assert progressed["operations"]["cpu-1"] == {
        "id": "cpu-1",
        "label": "CPU worker 1",
        "status": "running",
        "message": None,
        "device": "CPU",
        "current": 3,
        "total": 12,
        "started_at": progressed["operations"]["cpu-1"]["started_at"],
        "updated_at": progressed["operations"]["cpu-1"]["updated_at"],
        "finished_at": None,
    }

    completed = generation_jobs.mark_completed(CLIENT_A, "request-1")
    assert completed["status"] == "completed"
    assert completed["finished_at"] is not None
    assert completed["operations"]["cpu-1"]["status"] == "completed"
    assert completed["operations"]["cpu-1"]["current"] == 12
    assert not generation_jobs.active(completed)

    assert generation_jobs.create_queued(
        CLIENT_A, "request-1", "so_vits_svc_3", "cpu", "Ignored"
    ) == completed
    replacement = generation_jobs.create_queued(
        CLIENT_A, "request-2", "rvc", "gpu", "Queued again"
    )
    assert replacement["request_id"] == "request-2"


def test_active_job_cannot_be_replaced_by_a_different_request():
    generation_jobs.create_queued(CLIENT_A, "request-1", "rvc", "cpu", "Queued")
    with pytest.raises(generation_jobs.GenerationJobConflict):
        generation_jobs.create_queued(CLIENT_A, "request-2", "rvc", "cpu", "Queued")


def test_only_one_task_can_claim_a_queued_job():
    generation_jobs.create_queued(CLIENT_A, "request-1", "rvc", "cpu", "Queued")

    with ThreadPoolExecutor(max_workers=4) as executor:
        claims = list(executor.map(
            lambda task_id: generation_jobs.claim_running(CLIENT_A, "request-1", task_id),
            ["task-1", "task-2", "task-3", "task-4"],
        ))

    claimed = [state for state in claims if state is not None]
    assert len(claimed) == 1
    assert generation_jobs.get(CLIENT_A)["task_id"] == claimed[0]["task_id"]


def test_cancellation_wins_a_worker_failure_race():
    generation_jobs.create_queued(CLIENT_A, "request-1", "so_vits_svc_4", "gpu", "Queued")
    generation_jobs.mark_running(CLIENT_A, "request-1", "task-1")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            lambda operation: operation(),
            [
                lambda: generation_jobs.request_cancel(CLIENT_A, "request-1"),
                lambda: generation_jobs.mark_failed(CLIENT_A, "request-1", "Runtime exited"),
            ],
        ))

    state = generation_jobs.get(CLIENT_A)
    if state["status"] == "failed":
        # A failure that commits before the stop request is a legitimate terminal result.
        assert results[0] is None
    else:
        assert state["status"] == "cancelling"
        assert state["message"] == "Cancelling generation..."
        generation_jobs.mark_failed(CLIENT_A, "request-1", "Runtime exited")
        assert generation_jobs.get(CLIENT_A)["status"] == "cancelling"
        assert generation_jobs.is_cancel_requested(CLIENT_A, "request-1")


def test_scoped_cancel_does_not_touch_peers_on_the_same_runtime():
    generation_jobs.create_queued(CLIENT_A, "request-a", "so_vits_svc_3", "cpu", "Queued")
    generation_jobs.create_queued(CLIENT_B, "request-b", "so_vits_svc_3", "gpu", "Queued")
    generation_jobs.create_queued(CLIENT_C, "request-c", "rvc", "cpu", "Queued")

    affected = generation_jobs.request_cancel(CLIENT_A, "request-a")
    assert affected["client_id"] == CLIENT_A
    assert generation_jobs.get(CLIENT_A)["status"] == "cancelling"
    assert generation_jobs.get(CLIENT_B)["status"] == "queued"
    assert generation_jobs.get(CLIENT_C)["status"] == "queued"

    cancelled = generation_jobs.mark_cancelled(CLIENT_A, "request-a")
    assert cancelled["client_id"] == CLIENT_A
    assert generation_jobs.get(CLIENT_A)["status"] == "cancelled"
    assert generation_jobs.get(CLIENT_B)["status"] == "queued"
    assert generation_jobs.get(CLIENT_C)["status"] == "queued"


def test_scoped_cancel_is_an_atomic_noop_after_completion():
    generation_jobs.create_queued(CLIENT_A, "request", "rvc", "cpu", "Queued")
    generation_jobs.mark_completed(CLIENT_A, "request")

    assert generation_jobs.request_cancel(CLIENT_A, "request") is None
    assert generation_jobs.get(CLIENT_A)["status"] == "completed"


def test_late_operation_update_cannot_reverse_cancelling_status():
    generation_jobs.create_queued(CLIENT_A, "request", "rvc", "cpu", "Queued")
    generation_jobs.update_operation_progress(
        CLIENT_A, "request", "pitch", "Pitch +1", 0, 1, status="running"
    )
    generation_jobs.request_cancel(CLIENT_A, "request")

    generation_jobs.update_operation_progress(
        CLIENT_A, "request", "pitch", "Pitch +1", 1, 1, status="completed"
    )

    state = generation_jobs.get(CLIENT_A)
    assert state["status"] == "cancelling"
    assert state["operations"]["pitch"]["status"] == "cancelling"


def test_cache_commit_is_skipped_after_cancellation():
    generation_jobs.create_queued(CLIENT_A, "request", "rvc", "cpu", "Queued")
    generation_jobs.request_cancel(CLIENT_A, "request")
    calls = []

    committed, result = generation_jobs.commit_if_active(
        CLIENT_A, "request", lambda: calls.append("commit")
    )

    assert committed is False
    assert result is None
    assert calls == []


def test_one_clients_commit_does_not_block_another_clients_progress():
    generation_jobs.create_queued(CLIENT_A, "request-a", "rvc", "cpu", "Queued")
    generation_jobs.create_queued(CLIENT_B, "request-b", "rvc", "cpu", "Queued")
    commit_started = threading.Event()
    release_commit = threading.Event()

    def commit():
        commit_started.set()
        assert release_commit.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            generation_jobs.commit_if_active, CLIENT_A, "request-a", commit
        )
        assert commit_started.wait(timeout=2)
        generation_jobs.update_progress(CLIENT_B, "request-b", "Still responsive")
        assert generation_jobs.get(CLIENT_B)["message"] == "Still responsive"
        release_commit.set()
        assert future.result(timeout=2)[0] is True


def test_stale_worker_updates_do_not_replace_a_new_request():
    generation_jobs.create_queued(CLIENT_A, "old", "rvc", "cpu", "Queued")
    generation_jobs.mark_completed(CLIENT_A, "old")
    current = generation_jobs.create_queued(CLIENT_A, "new", "rvc", "cpu", "Queued")

    assert generation_jobs.mark_failed(CLIENT_A, "old", "Late failure") is None
    assert generation_jobs.get(CLIENT_A) == current


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (generation_jobs.get, ("../not-a-client",)),
        (generation_jobs.create_queued, ("z" * 32, "request", "rvc", "cpu", "Queued")),
        (generation_jobs.create_queued, (CLIENT_A, "", "rvc", "cpu", "Queued")),
        (generation_jobs.create_queued, (CLIENT_A, "request", "RVC", "cpu", "Queued")),
        (generation_jobs.create_queued, (CLIENT_A, "request", "rvc", "../../cpu", "Queued")),
        (generation_jobs.update_progress, (CLIENT_A, "request", "Progress", 2, 1)),
        (
            generation_jobs.update_operation_progress,
            (CLIENT_A, "request", "../worker", "Worker", 0, 1),
        ),
        (
            generation_jobs.update_operation_progress,
            (CLIENT_A, "request", "worker", "Worker", 2, 1),
        ),
    ],
)
def test_untrusted_identifiers_and_progress_are_validated(function, arguments):
    with pytest.raises(ValueError):
        function(*arguments)


def test_persisted_request_data_must_be_json_and_bounded(monkeypatch):
    with pytest.raises(ValueError, match="JSON serializable"):
        generation_jobs.create_queued(
            CLIENT_A,
            "request",
            "rvc",
            "cpu",
            "Queued",
            request_data={"bad": object()},
        )

    monkeypatch.setattr(generation_jobs, "MAX_REQUEST_DATA_BYTES", 8)
    with pytest.raises(ValueError, match="at most 8 bytes"):
        generation_jobs.create_queued(
            CLIENT_A,
            "request",
            "rvc",
            "cpu",
            "Queued",
            request_data={"text": "too long"},
        )


def test_state_files_are_private_complete_json_documents():
    state = generation_jobs.create_queued(CLIENT_A.upper(), "request", "rvc", "cpu", "Queued")
    path = next((generation_jobs._root()).glob("*.json"))

    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == state
    assert state["client_id"] == CLIENT_A


def test_processes_never_observe_or_leave_partial_state():
    generation_jobs.create_queued(CLIENT_A, "request", "rvc", "cpu", "Queued")
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=_write_progress_in_process, args=(CLIENT_A, prefix))
        for prefix in ("one", "two", "three", "four")
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=5)

    assert [process.exitcode for process in processes] == [0, 0, 0, 0]
    state = generation_jobs.get(CLIENT_A)
    assert state["status"] == "queued"
    assert state["message"].split("-")[0] in {"one", "two", "three", "four"}
    assert state["progress"]["total"] == 25


def test_parallel_operations_are_independent_and_cancel_together():
    generation_jobs.create_queued(CLIENT_A, "request", "so_vits_svc_3", "cpu", "Queued")
    generation_jobs.mark_running(CLIENT_A, "request", "task")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(
            lambda index: generation_jobs.update_operation_progress(
                CLIENT_A,
                "request",
                f"cpu-{index}",
                f"CPU worker {index}",
                index,
                4,
                device="CPU",
            ),
            range(4),
        ))

    state = generation_jobs.get(CLIENT_A)
    assert set(state["operations"]) == {"cpu-0", "cpu-1", "cpu-2", "cpu-3"}
    assert state["operations"]["cpu-3"]["current"] == 3

    generation_jobs.request_cancel(CLIENT_A, "request")
    cancelling = generation_jobs.get(CLIENT_A)
    assert {operation["status"] for operation in cancelling["operations"].values()} == {
        "cancelling"
    }

    generation_jobs.mark_cancelled(CLIENT_A, "request")
    cancelled = generation_jobs.get(CLIENT_A)
    assert {operation["status"] for operation in cancelled["operations"].values()} == {
        "cancelled"
    }

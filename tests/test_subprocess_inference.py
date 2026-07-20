import sys
import threading
import time

import pytest

from hay_say_common.subprocess_inference import (
    InferenceCancelled,
    InferenceProcessRegistry,
    request_workspace,
)


def test_registry_cancels_every_process_for_one_request_without_stopping_registry():
    registry = InferenceProcessRegistry()
    errors = []

    def run_child():
        try:
            registry.run(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                request_id="browser-job",
            )
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=run_child) for _ in range(2)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 5
    while registry.state()["active_processes"] != 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    result = registry.cancel(["browser-job"])
    for thread in threads:
        thread.join(timeout=5)

    assert result == {
        "request_ids": ["browser-job"],
        "active_processes_signalled": 2,
        "runtime_preserved": True,
    }
    assert len(errors) == 2
    assert all(isinstance(error, InferenceCancelled) for error in errors)
    assert registry.state()["status"] == "ready-cold"
    with pytest.raises(InferenceCancelled):
        registry.run([sys.executable, "-c", "pass"], request_id="browser-job")
    with pytest.raises(InferenceCancelled):
        registry.raise_if_cancelled("browser-job")
    committed = []
    with pytest.raises(InferenceCancelled):
        registry.commit_if_active("browser-job", lambda: committed.append(True))
    assert not committed

    completed = registry.run([sys.executable, "-c", "pass"], request_id="next-job")
    assert completed.returncode == 0
    assert registry.state()["status"] == "ready-cold"


def test_request_workspace_removes_only_its_own_directory(tmp_path):
    sibling = tmp_path / "keep"
    sibling.mkdir()
    with request_workspace(tmp_path, "job-") as workspace:
        workspace_path = tmp_path / workspace.split("/")[-1]
        assert workspace_path.is_dir()
        (workspace_path / "audio.flac").write_bytes(b"audio")
    assert not workspace_path.exists()
    assert sibling.is_dir()

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]

BF16_ENVIRONMENTS = {
    "controllable_talknet": "HAY_SAY_TALKNET_CPU_BF16_AUTOCAST",
    "gpt_so_vits": "HAY_SAY_GPT_SOVITS_CPU_BF16_AUTOCAST",
    "rvc": "HAY_SAY_RVC_CPU_BF16_AUTOCAST",
    "styletts_2": "HAY_SAY_STYLETTS_CPU_BF16_AUTOCAST",
}


def load_server(runtime_id):
    path = REPOSITORY / "ubuntuserver/hay_say" / f"{runtime_id}_server/main.py"
    spec = importlib.util.spec_from_file_location(f"test_{runtime_id}_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_styletts_cancel_route_preserves_its_warm_worker_pool():
    module = load_server("styletts_2")
    module.runtime_manager.close()

    class FakeRuntime:
        def __init__(self):
            self.cancelled = []

        def cancel(self, request_ids):
            self.cancelled.append(request_ids)
            return {
                "request_ids": request_ids,
                "active_workers_signalled": 1,
                "runtime_preserved": True,
            }

        @staticmethod
        def state():
            return {
                "status": "warm-idle",
                "loaded_models": ["Fluttershy"],
                "active_jobs": 0,
                "queued_jobs": 0,
            }

    runtime = FakeRuntime()
    module.runtime_manager = runtime
    module.register_methods(object())

    client = module.app.test_client()
    response = client.post("/cancel", json={"Request IDs": ["cancel-this"]})

    assert response.status_code == 200
    assert json.loads(response.get_data(as_text=True))["runtime_preserved"] is True
    assert runtime.cancelled == [["cancel-this"]]
    runtime = client.get("/runtime")
    assert runtime.status_code == 200
    state = json.loads(runtime.get_data(as_text=True))
    assert state["status"] == "warm-idle"
    assert state["cpu_bf16"]["environment_variable"] == BF16_ENVIRONMENTS["styletts_2"]


@pytest.mark.parametrize("runtime_id", tuple(BF16_ENVIRONMENTS))
def test_legacy_runtime_endpoints_report_backend_bf16_policy(runtime_id):
    module = load_server(runtime_id)
    module.runtime_manager.close()

    class FakeRuntime:
        @staticmethod
        def state():
            return {"status": "ready-cold"}

    module.runtime_manager = FakeRuntime()
    module.register_methods(object())

    response = module.app.test_client().get("/runtime")

    assert response.status_code == 200
    policy = json.loads(response.get_data(as_text=True))["cpu_bf16"]
    assert policy["environment_variable"] == BF16_ENVIRONMENTS[runtime_id]
    assert policy["effective"] == (policy["requested"] and policy["amx_available"])

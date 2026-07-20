import importlib.util
import unittest

from ubuntuserver.runtime.supervisor import RuntimeNotFoundError


FLASK_AVAILABLE = importlib.util.find_spec("flask") is not None


class FakeSupervisor:
    def __init__(self):
        self.calls = []

    @staticmethod
    def _status(runtime_id, status):
        return {"id": runtime_id, "status": status}

    def list_statuses(self):
        return [self._status("rvc", "stopped")]

    def status(self, runtime_id):
        if runtime_id == "missing":
            raise RuntimeNotFoundError("Unknown runtime: missing")
        return self._status(runtime_id, "stopped")

    def start(self, runtime_id):
        self.calls.append(("start", runtime_id))
        return self._status(runtime_id, "starting")

    def stop(self, runtime_id):
        self.calls.append(("stop", runtime_id))
        return self._status(runtime_id, "stopped")

    def stop_all(self):
        self.calls.append(("stop-all", None))
        return [self._status("rvc", "stopped")]

    def restart(self, runtime_id):
        self.calls.append(("restart", runtime_id))
        return self._status(runtime_id, "starting")


@unittest.skipUnless(FLASK_AVAILABLE, "Flask is not installed in the test interpreter")
class RuntimeApiTests(unittest.TestCase):
    def setUp(self):
        from ubuntuserver.runtime.api import create_app

        self.supervisor = FakeSupervisor()
        self.client = create_app(self.supervisor).test_client()

    def test_health_list_and_get(self):
        self.assertEqual(self.client.get("/health").get_json(), {"status": "ok"})
        self.assertEqual(
            self.client.get("/runtimes").get_json(),
            {"runtimes": [{"id": "rvc", "status": "stopped"}]},
        )
        self.assertEqual(
            self.client.get("/runtimes/rvc").get_json(),
            {"id": "rvc", "status": "stopped"},
        )

    def test_lifecycle_endpoints_and_error_mapping(self):
        self.assertEqual(self.client.post("/runtimes/rvc/start").status_code, 202)
        self.assertEqual(self.client.post("/runtimes/rvc/stop").status_code, 200)
        self.assertEqual(self.client.post("/runtimes/rvc/restart").status_code, 202)
        self.assertEqual(
            self.client.post("/runtimes/stop-all").get_json(),
            {"runtimes": [{"id": "rvc", "status": "stopped"}]},
        )
        self.assertEqual(
            self.supervisor.calls,
            [("start", "rvc"), ("stop", "rvc"), ("restart", "rvc"), ("stop-all", None)],
        )
        missing = self.client.get("/runtimes/missing")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.get_json()["type"], "not_found")


if __name__ == "__main__":
    unittest.main()

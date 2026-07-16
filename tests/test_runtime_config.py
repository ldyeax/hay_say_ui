import json
import tempfile
import unittest
from pathlib import Path

from ubuntuserver.runtime.config import ConfigError, load_config


class RuntimeConfigTests(unittest.TestCase):
    def test_default_registry_has_all_unique_native_ports(self):
        config = load_config(environ={})

        expected = {
            "controllable_talknet": 6574,
            "so_vits_svc_3": 6575,
            "so_vits_svc_4": 6576,
            "so_vits_svc_5": 6577,
            "rvc": 6578,
            "styletts_2": 6580,
            "gpt_so_vits": 6581,
        }
        self.assertEqual(
            {runtime_id: runtime.port for runtime_id, runtime in config.runtimes.items()},
            expected,
        )
        self.assertEqual(len({runtime.port for runtime in config.runtimes.values()}), 7)
        self.assertEqual(config.manager.api_host, "127.0.0.1")

    def test_environment_expansion_and_runtime_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "runtimes.json"
            config_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "manager": {
                            "state_dir": "${ROOT}/state",
                            "log_dir": "${ROOT}/logs",
                            "api_host": "127.0.0.1",
                        },
                        "runtimes": [
                            {
                                "id": "rvc",
                                "label": "RVC",
                                "port": 6578,
                                "command": ["${ROOT}/python", "${ROOT}/main.py"],
                                "cwd": "${ROOT}",
                                "env": {"BASE": "${ROOT}"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            environment = {
                "ROOT": str(root),
                "ALT": str(root / "alternate"),
                "HAY_SAY_RUNTIME_STATE_DIR": "${ROOT}/override-state",
                "HAY_SAY_RUNTIME_RVC_PORT": "7001",
                "HAY_SAY_RUNTIME_RVC_DEVICE": "cuda:1",
                "HAY_SAY_RUNTIME_RVC_COMMAND_JSON": '["${ALT}/python", "${ALT}/server.py"]',
                "HAY_SAY_RUNTIME_RVC_ENV_JSON": '{"CUDA_VISIBLE_DEVICES":"1"}',
            }

            config = load_config(config_path, environ=environment)

        runtime = config.runtimes["rvc"]
        self.assertEqual(config.manager.state_dir, root / "override-state")
        self.assertEqual(runtime.port, 7001)
        self.assertEqual(runtime.device, "cuda:1")
        self.assertEqual(runtime.command, (str(root / "alternate/python"), str(root / "alternate/server.py")))
        self.assertEqual(runtime.env["BASE"], str(root))
        self.assertEqual(runtime.env["CUDA_VISIBLE_DEVICES"], "1")

    def test_rejects_duplicate_ports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "runtimes.json"
            config_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "manager": {"state_dir": str(root / "state"), "log_dir": str(root / "logs")},
                        "runtimes": [
                            {
                                "id": runtime_id,
                                "label": runtime_id,
                                "port": 7000,
                                "command": ["/usr/bin/python3", "/tmp/main.py"],
                                "cwd": "/tmp",
                            }
                            for runtime_id in ("first", "second")
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "Duplicate runtime port"):
                load_config(config_path, environ={})


if __name__ == "__main__":
    unittest.main()

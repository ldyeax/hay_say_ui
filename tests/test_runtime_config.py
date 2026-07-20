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
        for runtime_id, runtime in config.runtimes.items():
            expected_threads = "1" if runtime_id == "so_vits_svc_4" else "4"
            self.assertEqual(runtime.env["OMP_NUM_THREADS"], expected_threads)
            self.assertEqual(runtime.env["MKL_NUM_THREADS"], expected_threads)
        self.assertEqual(
            config.runtimes["so_vits_svc_4"].env["HAY_SAY_SVC4_CPU_THREADS_PER_WORKER"],
            "1",
        )
        self.assertEqual(config.runtimes["so_vits_svc_3"].env["HAY_SAY_SVC3_CPU_THREADS"], "4")
        for runtime_id in (
            "controllable_talknet", "so_vits_svc_3", "so_vits_svc_4",
            "so_vits_svc_5", "rvc", "styletts_2", "gpt_so_vits",
        ):
            self.assertEqual(
                config.runtimes[runtime_id].env["HAY_SAY_MODEL_IDLE_TTL_SECONDS"],
                "1800",
            )
        svc5 = config.runtimes["so_vits_svc_5"]
        self.assertEqual(svc5.env["HAY_SAY_SVC5_CPU_WORKERS"], "1")
        self.assertEqual(svc5.env["HAY_SAY_SVC5_CPU_THREADS_PER_WORKER"], "4")
        self.assertEqual(svc5.env["HAY_SAY_SVC5_GPU_WORKERS"], "1")
        self.assertEqual(svc5.env["HAY_SAY_SVC5_STARTUP_CONCURRENCY"], "1")
        styletts = config.runtimes["styletts_2"]
        self.assertEqual(styletts.env["HAY_SAY_STYLETTS_CPU_WORKERS"], "8")
        self.assertEqual(styletts.env["HAY_SAY_STYLETTS_GPU_WORKERS"], "1")
        expected_bf16 = {
            "controllable_talknet": ("HAY_SAY_TALKNET_CPU_BF16_AUTOCAST", "0"),
            "so_vits_svc_3": ("HAY_SAY_SVC3_CPU_BF16_AUTOCAST", "0"),
            "so_vits_svc_4": ("HAY_SAY_SVC4_CPU_BF16_AUTOCAST", "1"),
            "so_vits_svc_5": ("HAY_SAY_SVC5_CPU_BF16_AUTOCAST", "0"),
            "rvc": ("HAY_SAY_RVC_CPU_BF16_AUTOCAST", "0"),
            "styletts_2": ("HAY_SAY_STYLETTS_CPU_BF16_AUTOCAST", "1"),
            "gpt_so_vits": ("HAY_SAY_GPT_SOVITS_CPU_BF16_AUTOCAST", "0"),
        }
        for runtime_id, (variable, value) in expected_bf16.items():
            self.assertEqual(config.runtimes[runtime_id].env[variable], value)

    def test_svc4_uses_its_per_worker_thread_override(self):
        config = load_config(
            environ={
                "HAY_SAY_MODEL_CPU_THREADS": "12",
                "HAY_SAY_SVC4_CPU_THREADS_PER_WORKER": "3",
            }
        )

        svc4 = config.runtimes["so_vits_svc_4"]
        self.assertEqual(svc4.env["HAY_SAY_SVC4_CPU_THREADS_PER_WORKER"], "3")
        self.assertEqual(svc4.env["OMP_NUM_THREADS"], "3")
        self.assertEqual(svc4.env["OPENBLAS_NUM_THREADS"], "3")
        self.assertEqual(config.runtimes["so_vits_svc_3"].env["OMP_NUM_THREADS"], "4")

    def test_svc5_uses_replica_pool_and_idle_retention_overrides(self):
        config = load_config(
            environ={
                "HAY_SAY_MODEL_IDLE_TTL_SECONDS": "2400",
                "HAY_SAY_SVC5_CPU_WORKERS": "24",
                "HAY_SAY_SVC5_CPU_THREADS_PER_WORKER": "8",
                "HAY_SAY_SVC5_GPU_WORKERS": "2",
                "HAY_SAY_SVC5_STARTUP_CONCURRENCY": "8",
            }
        )

        svc5 = config.runtimes["so_vits_svc_5"]
        self.assertEqual(svc5.env["HAY_SAY_MODEL_IDLE_TTL_SECONDS"], "2400")
        self.assertEqual(svc5.env["HAY_SAY_SVC5_CPU_WORKERS"], "24")
        self.assertEqual(svc5.env["HAY_SAY_SVC5_CPU_THREADS_PER_WORKER"], "8")
        self.assertEqual(svc5.env["HAY_SAY_SVC5_GPU_WORKERS"], "2")
        self.assertEqual(svc5.env["HAY_SAY_SVC5_STARTUP_CONCURRENCY"], "8")
        self.assertEqual(svc5.env["OMP_NUM_THREADS"], "8")
        self.assertEqual(svc5.env["OPENBLAS_NUM_THREADS"], "8")

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

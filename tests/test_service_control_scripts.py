import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def _write_executable(path, contents):
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _run_script(tmp_path, script_name, user):
    trace = tmp_path / "trace"
    _write_executable(
        tmp_path / "id",
        """#!/bin/sh
if [ "$1" = "-un" ]; then
    printf '%s\\n' "$MOCK_USER"
elif [ "$1" = "-u" ]; then
    printf '1000\\n'
else
    exit 2
fi
""",
    )
    for command in ("sudo", "systemctl"):
        _write_executable(
            tmp_path / command,
            f"#!/bin/sh\nprintf '{command} %s\\n' \"$*\" > \"$TRACE\"\n",
        )

    environment = os.environ.copy()
    environment.update(
        {
            "MOCK_USER": user,
            "PATH": f"{tmp_path}:{environment['PATH']}",
            "TRACE": str(trace),
        }
    )
    result = subprocess.run(
        [REPOSITORY / "ubuntuserver" / script_name],
        capture_output=True,
        text=True,
        env=environment,
    )
    return result, trace.read_text(encoding="utf-8").strip() if trace.exists() else ""


class ServiceControlScriptTests(unittest.TestCase):
    scripts = (("start-hay-say.sh", "start"), ("stop-hay-say.sh", "stop"))

    def run_script(self, script_name, user):
        with tempfile.TemporaryDirectory() as directory:
            return _run_script(Path(directory), script_name, user)

    def test_runs_systemctl_directly_for_luna(self):
        for script_name, action in self.scripts:
            with self.subTest(script=script_name):
                result, trace = self.run_script(script_name, "luna")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    trace, f"systemctl --user {action} hay-say.target"
                )

    def test_switches_to_luna_for_root(self):
        for script_name, action in self.scripts:
            with self.subTest(script=script_name):
                result, trace = self.run_script(script_name, "root")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(trace.startswith("sudo -u luna bash -c "), trace)
                self.assertIn('XDG_RUNTIME_DIR="/run/user/$(id -u)"', trace)
                self.assertTrue(
                    trace.endswith(f"systemctl --user {action} hay-say.target"),
                    trace,
                )

    def test_rejects_other_users(self):
        for script_name, _action in self.scripts:
            with self.subTest(script=script_name):
                result, trace = self.run_script(script_name, "someone-else")
                self.assertEqual(result.returncode, 1)
                self.assertIn("run this script as root or luna", result.stderr)
                self.assertEqual(trace, "")


if __name__ == "__main__":
    unittest.main()

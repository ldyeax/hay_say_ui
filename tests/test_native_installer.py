import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def _write_patch_source(root: Path) -> None:
    patch_dir = root / "ubuntuserver/patches/gpt_so_vits_server"
    patch_dir.mkdir(parents=True)
    (patch_dir / "native-port.patch").write_text(
        "--- a/main.py\n"
        "+++ b/main.py\n"
        "@@ -1 +1 @@\n"
        "-original\n"
        "+patched\n",
        encoding="utf-8",
    )
    patch_dir = root / "ubuntuserver/patches/rvc"
    patch_dir.mkdir(parents=True)
    (patch_dir / "native-port.patch").write_text(
        "--- a/infer/lib/audio.py\n"
        "+++ b/infer/lib/audio.py\n"
        "@@ -1 +1 @@\n"
        "-original\n"
        "+patched\n",
        encoding="utf-8",
    )


def _run_patch_reconciliation(source: Path, data_root: Path):
    return subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
DATA_ROOT=$1
REPO_CONTENT_ROOT=$2
DRY_RUN=0
run() { "$@"; }
die() { printf 'test installer: %s\\n' "$*" >&2; return 1; }
log() { printf '[test installer] %s\\n' "$*"; }
source "$3"
apply_native_patches
""",
            "native-patch-test",
            str(data_root),
            str(source),
            str(REPOSITORY / "ubuntuserver/native-patches.sh"),
        ],
        capture_output=True,
        text=True,
    )


def _write_active_systemctl(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $* == '--user is-active --quiet hay-say.target' ]]; then\n"
        "    exit 0\n"
        "fi\n"
        "printf 'unexpected systemctl call: %s\\n' \"$*\" >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_root_installer_rejects_filesystem_root_as_data_directory():
    result = subprocess.run(
        [
            str(REPOSITORY / "ubuntuserver" / "install.sh"),
            "--dry-run",
            "--skip-apt",
            "--skip-images",
            "--data-root",
            "/",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unsafe system path" in result.stderr


def test_root_installer_propagates_no_ui_auth_option():
    result = subprocess.run(
        [
            str(REPOSITORY / "ubuntuserver" / "install.sh"),
            "--dry-run",
            "--skip-apt",
            "--skip-images",
            "--no-start",
            "--no-ui-auth",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "install-user.sh" in result.stdout
    assert "--no-ui-auth" in result.stdout


def test_selective_extraction_dry_run_creates_cache_and_uses_exact_sync(tmp_path):
    result = subprocess.run(
        [
            str(REPOSITORY / "ubuntuserver" / "extract-images.sh"),
            "--dry-run",
            "--runtime",
            "rvc",
            "--data-root",
            str(tmp_path / "data"),
            "--owner",
            os.environ.get("USER", "root"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert str(tmp_path / "data" / "cache") in result.stdout
    assert "rsync -a --delete --delete-excluded" in result.stdout
    assert "runtime-sources/rvc/" in result.stdout


def test_patch_reconciliation_is_quiet_and_independent_of_source_path():
    with tempfile.TemporaryDirectory(prefix="hay-say-installer-test-") as temporary:
        root = Path(temporary)
        source_a = root / "source-a"
        source_b = root / "source-b"
        data_root = root / "data"
        _write_patch_source(source_a)
        shutil.copytree(source_a, source_b)
        target = data_root / "runtime-sources/gpt_so_vits_server/main.py"
        target.parent.mkdir(parents=True)
        target.write_text("original\n", encoding="utf-8")
        rvc_target = data_root / "runtime-sources/rvc/infer/lib/audio.py"
        rvc_target.parent.mkdir(parents=True)
        rvc_target.write_bytes(b"original\r\n")

        first = _run_patch_reconciliation(source_a, data_root)
        assert first.returncode == 0, first.stderr
        assert target.read_text(encoding="utf-8") == "patched\n"
        assert rvc_target.read_text(encoding="utf-8") == "patched\n"
        assert "No such file or directory" not in first.stderr
        first_mtime = target.stat().st_mtime_ns
        first_rvc_mtime = rvc_target.stat().st_mtime_ns
        first_digest = (
            data_root / "provenance/native-patches/gpt_so_vits_server/patches.sha256"
        ).read_text(encoding="utf-8")
        time.sleep(0.01)

        second = _run_patch_reconciliation(source_b, data_root)
        assert second.returncode == 0, second.stderr
        assert "gpt_so_vits_server maintained patch set is already applied" in second.stdout
        assert "patching file" not in second.stdout
        assert target.read_text(encoding="utf-8") == "patched\n"
        assert rvc_target.read_text(encoding="utf-8") == "patched\n"
        assert target.stat().st_mtime_ns == first_mtime
        assert rvc_target.stat().st_mtime_ns == first_rvc_mtime
        assert (
            data_root / "provenance/native-patches/gpt_so_vits_server/patches.sha256"
        ).read_text(encoding="utf-8") == first_digest


def test_links_only_stops_active_stack_before_mutation_and_leaves_it_stopped(tmp_path):
    home = tmp_path / "home"
    install_root = home / "hay_say"
    (install_root / "hay_say_ui").mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_active_systemctl(fake_bin / "systemctl")
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "USER": environment.get("USER", "test-user"),
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_STATE_HOME": str(home / ".local/state"),
        }
    )

    result = subprocess.run(
        [
            str(REPOSITORY / "ubuntuserver/install-user.sh"),
            "--dry-run",
            "--links-only",
            "--source",
            str(REPOSITORY),
            "--install-root",
            str(install_root),
            "--data-root",
            str(tmp_path / "data"),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    stop = result.stdout.index("systemctl --user stop hay-say.target")
    mutation = result.stdout.index("configuring persistent data and runtime links")
    assert stop < mutation
    assert "systemctl --user restart hay-say.target" not in result.stdout

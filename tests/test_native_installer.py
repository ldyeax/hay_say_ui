import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
PATCH_HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


def test_native_patch_hunk_lengths_are_well_formed():
    for patch_file in sorted((REPOSITORY / "ubuntuserver/patches").glob("*/*.patch")):
        lines = patch_file.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            match = HUNK_HEADER.match(lines[index])
            if not match:
                index += 1
                continue

            expected_old = int(match.group(1) or 1)
            expected_new = int(match.group(2) or 1)
            old_count = 0
            new_count = 0
            index += 1
            while index < len(lines) and not lines[index].startswith(("@@ ", "--- a/")):
                line = lines[index]
                if line.startswith((" ", "-")):
                    old_count += 1
                if line.startswith((" ", "+")):
                    new_count += 1
                index += 1

            assert old_count == expected_old, f"malformed old hunk length: {patch_file}"
            assert new_count == expected_new, f"malformed new hunk length: {patch_file}"


def _reconstruct_patch_baseline(patch_file, root):
    targets = {}
    current_path = None
    lines = patch_file.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("--- a/"):
            current_path = line[len("--- a/"):]
            targets.setdefault(current_path, [])
            index += 1
            continue
        match = PATCH_HUNK_HEADER.match(line)
        if match and current_path is not None:
            old_start = int(match.group(1))
            target_lines = targets[current_path]
            while len(target_lines) < old_start - 1:
                target_lines.append(f"# baseline filler {len(target_lines) + 1}")
            index += 1
            while index < len(lines) and not lines[index].startswith(("@@ ", "--- a/")):
                hunk_line = lines[index]
                if hunk_line.startswith((" ", "-")):
                    target_lines.append(hunk_line[1:])
                index += 1
            continue
        index += 1

    for relative_path, target_lines in targets.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(target_lines) + "\n", encoding="utf-8")


def test_gpt_cooperative_cancel_patch_applies_without_fuzz(tmp_path):
    patch_file = REPOSITORY / "ubuntuserver/patches/gpt_so_vits/cooperative-cancel.patch"
    baseline = tmp_path / "gpt"
    _reconstruct_patch_baseline(patch_file, baseline)

    result = subprocess.run(
        [
            "patch", "--batch", "--forward", "--fuzz=0", "--dry-run",
            "-p1", "-i", str(patch_file),
        ],
        cwd=baseline,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "fuzz" not in result.stdout.lower()


def test_every_native_patch_target_has_a_runtime_mapping():
    patch_targets = sorted(
        path.name
        for path in (REPOSITORY / "ubuntuserver/patches").iterdir()
        if path.is_dir()
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
source "$1"
shift
for target in "$@"; do
    printf '%s=%s\n' "$target" "$(patch_runtime_id "$target")"
done
""",
            "native-patch-mapping-test",
            str(REPOSITORY / "ubuntuserver/native-patches.sh"),
            *patch_targets,
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == len(patch_targets)


def test_bf16_patch_sets_cover_all_extracted_model_source_trees():
    expected = {
        "controllable_talknet",
        "gpt_so_vits",
        "rvc",
        "so_vits_svc_4",
        "so_vits_svc_4_dot_1_stable",
        "so_vits_svc_5_v1",
        "so_vits_svc_5_v2",
        "styletts_2",
    }

    for target in expected:
        patch_file = REPOSITORY / "ubuntuserver/patches" / target / "bf16-numpy.patch"
        assert patch_file.is_file(), target
        contents = patch_file.read_text(encoding="utf-8")
        assert ".float()" in contents
        assert "--- a/" in contents
        assert "+++ b/" in contents


def test_talknet_bf16_patch_keeps_hifigan_in_fp32():
    contents = (
        REPOSITORY / "ubuntuserver/patches/controllable_talknet/bf16-numpy.patch"
    ).read_text(encoding="utf-8")

    assert "with torch.cpu.amp.autocast(enabled=False):" in contents
    assert contents.count("+    @_run_vocoder_in_fp32") == 3
    assert "+    def __init__" not in contents
    assert "+    def vocode" not in contents
    assert "+    def superres" not in contents
    assert "+        raise RuntimeError(message)" in contents


def test_legacy_server_overlays_isolate_outputs_and_preserve_runtime_on_cancel():
    for runtime_id in ("controllable_talknet", "gpt_so_vits", "rvc", "styletts_2"):
        contents = (
            REPOSITORY / "ubuntuserver/hay_say" / f"{runtime_id}_server/main.py"
        ).read_text(encoding="utf-8")
        assert "hsc.request_workspace" in contents
        assert '@app.route("/cancel", methods=["POST"])' in contents

    for runtime_id in ("controllable_talknet", "gpt_so_vits", "rvc", "styletts_2"):
        contents = (
            REPOSITORY / "ubuntuserver/hay_say" / f"{runtime_id}_server/main.py"
        ).read_text(encoding="utf-8")
        assert "PersistentModelRuntime" in contents
        assert "runtime_manager.cancel(request_ids)" in contents
    rvc = (REPOSITORY / "ubuntuserver/hay_say/rvc_server/main.py").read_text(encoding="utf-8")
    assert 'WORKER_SCRIPT_PATH = os.path.join(ARCHITECTURE_ROOT, "hay_say_worker.py")' in rvc
    assert (REPOSITORY / "ubuntuserver/hay_say/rvc/hay_say_worker.py").is_file()
    styletts_worker = REPOSITORY / "ubuntuserver/hay_say/styletts_2/hay_say_worker.py"
    styletts_runtime = REPOSITORY / "ubuntuserver/hay_say/styletts_2/hay_say_runtime.py"
    assert styletts_worker.is_file()
    assert styletts_runtime.is_file()
    assert "cpu_bf16_autocast(cpu_bf16)" in styletts_worker.read_text(encoding="utf-8")
    styletts_runtime_source = styletts_runtime.read_text(encoding="utf-8")
    assert "cancel_check" in styletts_runtime_source
    assert (
        "encoded_prosody = predictor_encoding.transpose(-1, -2) @ alignment"
        in styletts_runtime_source
    )

    talknet_server = (
        REPOSITORY / "ubuntuserver/hay_say/controllable_talknet_server/main.py"
    ).read_text(encoding="utf-8")
    talknet_worker = (
        REPOSITORY / "ubuntuserver/hay_say/controllable_talknet/hay_say_worker.py"
    ).read_text(encoding="utf-8")
    assert '"output_path": output_path' in talknet_server
    assert 'command["output_path"]' in talknet_worker


def test_installer_and_doctor_require_complete_upstream_sources_for_persistent_workers():
    installer = (REPOSITORY / "ubuntuserver/install-user.sh").read_text(encoding="utf-8")
    doctor = (REPOSITORY / "ubuntuserver/doctor.sh").read_text(encoding="utf-8")
    sentinels = (
        "controllable_talknet_cli.py",
        "models/vqgan32_universal_57000.ckpt",
        "GPT_SoVITS/inference_cli.py",
        "GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin",
        "infer/modules/vc/modules.py",
        "assets/hubert/hubert_base.pt",
        "models.py",
        "Utils/PLBERT/step_1000000.t7",
        "hubert/hubert-soft-0d54a1f4.pt",
    )

    for sentinel in sentinels:
        assert sentinel in installer
        assert sentinel in doctor
    assert "runtime_source_is_complete" in installer
    for runtime_id in ("controllable_talknet", "gpt_so_vits", "rvc", "styletts_2"):
        assert f"{runtime_id}/hay_say_worker.py" in doctor


def test_installer_discovers_the_user_systemd_bus_for_noninteractive_invocations():
    installer = (REPOSITORY / "ubuntuserver/install-user.sh").read_text(encoding="utf-8")

    assert 'local runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$EUID}"' in installer
    assert 'export XDG_RUNTIME_DIR="$runtime_dir"' in installer
    assert 'export DBUS_SESSION_BUS_ADDRESS="unix:path=$runtime_dir/bus"' in installer
    assert "configure_user_service_environment" in installer


def test_svc5_server_uses_persistent_replica_overlay():
    server_root = REPOSITORY / "ubuntuserver/hay_say/so_vits_svc_5_server"
    main = (server_root / "main.py").read_text(encoding="utf-8")
    runtime = (server_root / "svc5_runtime.py").read_text(encoding="utf-8")

    assert "from svc5_runtime import" in main
    assert '@application.route("/cancel", methods=["POST"])' in main
    assert "class ReplicaGroup" in runtime
    assert "class FeatureCache" in runtime
    assert "HAY_SAY_SVC5_CPU_THREADS_PER_WORKER" in runtime
    assert "HAY_SAY_MODEL_IDLE_TTL_SECONDS" in runtime


def test_svc5_install_and_doctor_validate_all_native_frontend_assets():
    installer = (REPOSITORY / "ubuntuserver/install-user.sh").read_text(encoding="utf-8")
    doctor = (REPOSITORY / "ubuntuserver/doctor.sh").read_text(encoding="utf-8")
    required_paths = (
        "so_vits_svc_5_server/version_determinator.py",
        "so_vits_svc_5_v1/svc_inference.py",
        "so_vits_svc_5_v1/configs/base.yaml",
        "so_vits_svc_5_v1/whisper_pretrain/medium.pt",
        "so_vits_svc_5_v2/svc_inference.py",
        "so_vits_svc_5_v2/configs/base.yaml",
        "so_vits_svc_5_v2/whisper_pretrain/large-v2.pt",
        "so_vits_svc_5_v2/hubert_pretrain/hubert-soft-0d54a1f4.pt",
        "so_vits_svc_5_v2/crepe/assets/full.pth",
    )

    for required_path in required_paths:
        assert required_path in installer
        assert required_path in doctor


def test_svc5_v2_patch_casts_crepe_probabilities_before_numpy():
    patch_file = REPOSITORY / "ubuntuserver/patches/so_vits_svc_5_v2/bf16-numpy.patch"
    contents = patch_file.read_text(encoding="utf-8")

    assert "--- a/crepe/decode.py" in contents
    assert "-    sequences = probs.cpu().numpy()" in contents
    assert "+    sequences = probs.float().cpu().numpy()" in contents


def test_cpu_heavy_services_have_native_thread_headroom():
    units = [
        REPOSITORY / "ubuntuserver/systemd/user/hay-say-runtime-manager.service",
        REPOSITORY / "ubuntuserver/systemd/user/hay-say-celery-cpu.service",
    ]

    for unit in units:
        contents = unit.read_text(encoding="utf-8")
        assert "TasksMax=4096" in contents
        assert "LimitNOFILE=65536" in contents


def _write_patch_source(root: Path) -> None:
    patch_dir = root / "ubuntuserver/patches/gpt_so_vits"
    patch_dir.mkdir(parents=True)
    (patch_dir / "native-port.patch").write_text(
        "--- a/GPT_SoVITS/inference_cli.py\n"
        "+++ b/GPT_SoVITS/inference_cli.py\n"
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


def _write_inactive_systemctl(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $* == '--user is-active --quiet hay-say.target' ]]; then\n"
        "    exit 3\n"
        "fi\n"
        "printf 'unexpected systemctl call: %s\\n' \"$*\" >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_sixty_core_lscpu(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "[[ $* == '-p=CORE,SOCKET' ]] || exit 1\n"
        "printf '# core,socket\\n'\n"
        "for ((core = 0; core < 60; core++)); do\n"
        "    printf '%s,0\\n' \"$core\"\n"
        "done\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_one_hundred_twenty_cpu_getconf(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "[[ $* == '_NPROCESSORS_ONLN' ]] || exit 1\n"
        "printf '120\\n'\n",
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


def _run_links_only_from_repository(tmp_path, *, install_svc3_base=False):
    home = tmp_path / "home"
    install_root = home / "hay_say"
    install_root.mkdir(parents=True)
    (install_root / "hay_say_ui").symlink_to(REPOSITORY, target_is_directory=True)
    data_root = tmp_path / "data"
    if install_svc3_base:
        hubert = data_root / "runtime-sources/so_vits_svc_3/hubert/hubert-soft-0d54a1f4.pt"
        hubert.parent.mkdir(parents=True)
        hubert.write_bytes(b"extracted-hubert")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_inactive_systemctl(fake_bin / "systemctl")
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
            "--links-only",
            "--source",
            str(REPOSITORY),
            "--install-root",
            str(install_root),
            "--data-root",
            str(data_root),
            "--no-ui-auth",
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    return result, install_root, data_root


@pytest.mark.skipif(os.geteuid() == 0, reason="the user-stage installer intentionally rejects root")
def test_fresh_links_do_not_treat_bundled_overlays_as_complete_runtimes(tmp_path):
    result, install_root, _data_root = _run_links_only_from_repository(tmp_path)

    assert result.returncode == 0, result.stderr
    architecture_roots = (
        "controllable_talknet",
        "so_vits_svc_3",
        "so_vits_svc_4",
        "so_vits_svc_4_dot_1_stable",
        "so_vits_svc_5_v1",
        "so_vits_svc_5_v2",
        "rvc",
        "styletts_2",
        "gpt_so_vits",
    )
    assert all(not (install_root / name).exists() for name in architecture_roots)
    assert all(
        (install_root / f"{runtime_id}_server").is_symlink()
        for runtime_id in (
            "controllable_talknet",
            "so_vits_svc_3",
            "so_vits_svc_4",
            "so_vits_svc_5",
            "rvc",
            "styletts_2",
            "gpt_so_vits",
        )
    )
    assert "runtime base source is not installed or is incomplete" in result.stdout


@pytest.mark.skipif(os.geteuid() == 0, reason="the user-stage installer intentionally rejects root")
def test_selective_extraction_links_only_the_complete_runtime_base(tmp_path):
    result, install_root, data_root = _run_links_only_from_repository(
        tmp_path, install_svc3_base=True
    )

    assert result.returncode == 0, result.stderr
    svc3 = install_root / "so_vits_svc_3"
    assert svc3.is_symlink()
    assert svc3.resolve() == data_root / "runtime-sources/so_vits_svc_3"
    assert (svc3 / "inference/infer_tool.py").is_file()
    assert (svc3 / "hubert/hubert-soft-0d54a1f4.pt").is_file()
    for name in ("controllable_talknet", "so_vits_svc_4", "rvc", "styletts_2", "gpt_so_vits"):
        assert not (install_root / name).exists()


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
        target = data_root / "runtime-sources/gpt_so_vits/GPT_SoVITS/inference_cli.py"
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
            data_root / "provenance/native-patches/gpt_so_vits/patches.sha256"
        ).read_text(encoding="utf-8")
        time.sleep(0.01)

        second = _run_patch_reconciliation(source_b, data_root)
        assert second.returncode == 0, second.stderr
        assert "gpt_so_vits maintained patch set is already applied" in second.stdout
        assert "patching file" not in second.stdout
        assert target.read_text(encoding="utf-8") == "patched\n"
        assert rvc_target.read_text(encoding="utf-8") == "patched\n"
        assert target.stat().st_mtime_ns == first_mtime
        assert rvc_target.stat().st_mtime_ns == first_rvc_mtime
        assert (
            data_root / "provenance/native-patches/gpt_so_vits/patches.sha256"
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
            "HAY_SAY_HOST_MEMORY_MIB": "262144",
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


@pytest.mark.skipif(os.geteuid() == 0, reason="the user-stage installer intentionally rejects root")
def test_links_only_derives_and_preserves_parallelism_settings(tmp_path):
    home = tmp_path / "home"
    install_root = home / "hay_say"
    (install_root / "hay_say_ui").mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_inactive_systemctl(fake_bin / "systemctl")
    _write_sixty_core_lscpu(fake_bin / "lscpu")
    _write_one_hundred_twenty_cpu_getconf(fake_bin / "getconf")
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "USER": environment.get("USER", "test-user"),
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_STATE_HOME": str(home / ".local/state"),
            "HAY_SAY_HOST_MEMORY_MIB": "253000",
        }
    )
    command = [
        str(REPOSITORY / "ubuntuserver/install-user.sh"),
        "--links-only",
        "--source",
        str(REPOSITORY),
        "--install-root",
        str(install_root),
        "--data-root",
        str(tmp_path / "data"),
    ]

    first = subprocess.run(command + ["--no-ui-auth"], capture_output=True, text=True, env=environment)
    assert first.returncode == 0, first.stderr
    environment_file = home / ".config/hay-say/environment"
    contents = environment_file.read_text(encoding="utf-8")
    assert "HAY_SAY_CPU_INFERENCE_SLOTS='15'" in contents
    assert "HAY_SAY_MODEL_CPU_THREADS='12'" in contents
    assert "HAY_SAY_MODEL_CPU_INTEROP_THREADS='1'" in contents
    assert "HAY_SAY_SVC3_CPU_PITCH_WORKERS='24'" in contents
    assert "HAY_SAY_SVC3_CPU_THREAD_BUDGET='240'" in contents
    assert "HAY_SAY_SVC3_CPU_THREADS='10'" in contents
    assert "HAY_SAY_AUTO_CPU_PITCH_VARIANTS='4'" in contents
    assert "HAY_SAY_SVC4_CPU_SLICE_WORKERS='64'" in contents
    assert "HAY_SAY_SVC4_CPU_THREADS_PER_WORKER='2'" in contents
    assert "HAY_SAY_SVC5_CPU_WORKERS='24'" in contents
    assert "HAY_SAY_SVC5_CPU_THREADS_PER_WORKER='8'" in contents
    assert "HAY_SAY_SVC5_GPU_WORKERS='1'" in contents
    assert "HAY_SAY_SVC5_STARTUP_CONCURRENCY='8'" in contents
    assert "HAY_SAY_RVC_CPU_WORKERS='24'" in contents
    assert "HAY_SAY_RVC_GPU_WORKERS='1'" in contents
    assert "HAY_SAY_TALKNET_CPU_WORKERS='12'" in contents
    assert "HAY_SAY_TALKNET_GPU_WORKERS='1'" in contents
    assert "HAY_SAY_GPT_SOVITS_CPU_WORKERS='12'" in contents
    assert "HAY_SAY_GPT_SOVITS_GPU_WORKERS='1'" in contents
    assert "HAY_SAY_STYLETTS_CPU_WORKERS='12'" in contents
    assert "HAY_SAY_STYLETTS_GPU_WORKERS='1'" in contents
    assert "HAY_SAY_TALKNET_CPU_BF16_AUTOCAST='0'" in contents
    assert "HAY_SAY_SVC3_CPU_BF16_AUTOCAST='0'" in contents
    assert "HAY_SAY_SVC4_CPU_BF16_AUTOCAST='1'" in contents
    assert "HAY_SAY_SVC5_CPU_BF16_AUTOCAST='0'" in contents
    assert "HAY_SAY_RVC_CPU_BF16_AUTOCAST='0'" in contents
    assert "HAY_SAY_STYLETTS_CPU_BF16_AUTOCAST='1'" in contents
    assert "HAY_SAY_GPT_SOVITS_CPU_BF16_AUTOCAST='0'" in contents
    assert "HAY_SAY_MODEL_IDLE_TTL_SECONDS='1800'" in contents
    assert "HAY_SAY_MAX_BATCH_DOWNLOAD_BYTES='268435456'" in contents
    assert "HAY_SAY_CPU_CONCURRENCY='20'" in contents
    assert "HAY_SAY_UI_AUTH_ENABLED='0'" in contents

    contents = contents.replace("HAY_SAY_CPU_INFERENCE_SLOTS='15'", "HAY_SAY_CPU_INFERENCE_SLOTS='3'")
    contents = contents.replace("HAY_SAY_MODEL_CPU_THREADS='12'", "HAY_SAY_MODEL_CPU_THREADS='10'")
    contents = contents.replace("HAY_SAY_CPU_CONCURRENCY='20'", "HAY_SAY_CPU_CONCURRENCY='5'")
    contents = contents.replace(
        "HAY_SAY_SVC3_CPU_THREAD_BUDGET='240'",
        "HAY_SAY_SVC3_CPU_THREAD_BUDGET='2304'",
    )
    contents = contents.replace("HAY_SAY_SVC3_CPU_THREADS='10'", "HAY_SAY_SVC3_CPU_THREADS='96'")
    contents = contents.replace(
        "HAY_SAY_SVC4_CPU_SLICE_WORKERS='64'",
        "HAY_SAY_SVC4_CPU_SLICE_WORKERS='17'",
    )
    contents = contents.replace(
        "HAY_SAY_SVC4_CPU_THREADS_PER_WORKER='2'",
        "HAY_SAY_SVC4_CPU_THREADS_PER_WORKER='3'",
    )
    contents = contents.replace("HAY_SAY_SVC5_CPU_WORKERS='24'", "HAY_SAY_SVC5_CPU_WORKERS='12'")
    contents = contents.replace(
        "HAY_SAY_SVC5_CPU_THREADS_PER_WORKER='8'",
        "HAY_SAY_SVC5_CPU_THREADS_PER_WORKER='6'",
    )
    contents = contents.replace("HAY_SAY_SVC5_STARTUP_CONCURRENCY='8'", "HAY_SAY_SVC5_STARTUP_CONCURRENCY='5'")
    contents = contents.replace("HAY_SAY_RVC_CPU_WORKERS='24'", "HAY_SAY_RVC_CPU_WORKERS='13'")
    contents = contents.replace("HAY_SAY_TALKNET_CPU_WORKERS='12'", "HAY_SAY_TALKNET_CPU_WORKERS='7'")
    contents = contents.replace(
        "HAY_SAY_GPT_SOVITS_CPU_WORKERS='12'",
        "HAY_SAY_GPT_SOVITS_CPU_WORKERS='6'",
    )
    contents = contents.replace(
        "HAY_SAY_STYLETTS_CPU_WORKERS='12'",
        "HAY_SAY_STYLETTS_CPU_WORKERS='5'",
    )
    contents = contents.replace(
        "HAY_SAY_SVC4_CPU_BF16_AUTOCAST='1'",
        "HAY_SAY_SVC4_CPU_BF16_AUTOCAST='0'",
    )
    contents = contents.replace("HAY_SAY_MODEL_IDLE_TTL_SECONDS='1800'", "HAY_SAY_MODEL_IDLE_TTL_SECONDS='2400'")
    environment_file.write_text(contents, encoding="utf-8")

    second = subprocess.run(command, capture_output=True, text=True, env=environment)
    assert second.returncode == 0, second.stderr
    contents = environment_file.read_text(encoding="utf-8")
    assert "HAY_SAY_CPU_INFERENCE_SLOTS='3'" in contents
    assert "HAY_SAY_MODEL_CPU_THREADS='10'" in contents
    assert "HAY_SAY_CPU_CONCURRENCY='5'" in contents
    assert "HAY_SAY_SVC3_CPU_THREAD_BUDGET='2304'" in contents
    assert "HAY_SAY_SVC3_CPU_THREADS='96'" in contents
    assert "HAY_SAY_SVC4_CPU_SLICE_WORKERS='17'" in contents
    assert "HAY_SAY_SVC4_CPU_THREADS_PER_WORKER='3'" in contents
    assert "HAY_SAY_SVC5_CPU_WORKERS='12'" in contents
    assert "HAY_SAY_SVC5_CPU_THREADS_PER_WORKER='6'" in contents
    assert "HAY_SAY_SVC5_STARTUP_CONCURRENCY='5'" in contents
    assert "HAY_SAY_RVC_CPU_WORKERS='13'" in contents
    assert "HAY_SAY_TALKNET_CPU_WORKERS='7'" in contents
    assert "HAY_SAY_GPT_SOVITS_CPU_WORKERS='6'" in contents
    assert "HAY_SAY_STYLETTS_CPU_WORKERS='5'" in contents
    assert "HAY_SAY_SVC4_CPU_BF16_AUTOCAST='0'" in contents
    assert "HAY_SAY_STYLETTS_CPU_BF16_AUTOCAST='1'" in contents
    assert "HAY_SAY_MODEL_IDLE_TTL_SECONDS='2400'" in contents
    assert "HAY_SAY_UI_AUTH_ENABLED='0'" in contents
    assert (home / ".config/hay-say/ui-auth").read_text(encoding="utf-8").startswith("enabled=0\n")

    contents = contents.replace(
        "HAY_SAY_SVC3_CPU_THREAD_BUDGET='2304'",
        "HAY_SAY_SVC3_CPU_THREAD_BUDGET='240'",
    )
    environment_file.write_text(contents, encoding="utf-8")
    third = subprocess.run(command, capture_output=True, text=True, env=environment)
    assert third.returncode == 0, third.stderr
    contents = environment_file.read_text(encoding="utf-8")
    assert "HAY_SAY_SVC3_CPU_THREAD_BUDGET='240'" in contents
    assert "HAY_SAY_SVC3_CPU_THREADS='10'" in contents
    assert "Reducing HAY_SAY_SVC3_CPU_THREADS from 96 to 10" in third.stderr

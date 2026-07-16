#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_SOURCE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

SOURCE_ROOT=$DEFAULT_SOURCE_ROOT
INSTALL_ROOT="${HAY_SAY_INSTALL_ROOT:-$HOME/hay_say}"
if [[ -n "${HAY_SAY_DATA_ROOT:-}" ]]; then
	DATA_ROOT=$HAY_SAY_DATA_ROOT
elif mountpoint -q /mnt/sanic 2>/dev/null && [[ -w /mnt/sanic ]]; then
	DATA_ROOT=/mnt/sanic/hay_say
else
	DATA_ROOT="$INSTALL_ROOT/data"
fi
DRY_RUN=0
NO_START=0
SKIP_DEPLOY=0
MODE=all
UI_AUTH_ENABLED=1
UI_AUTH_SET=0

usage() {
	cat <<'EOF'
Usage: install-user.sh [OPTIONS]

Idempotently deploy and configure Hay Say for the current non-root user.

  --source PATH          Repository worktree to deploy
  --install-root PATH    Installation root (default: ~/hay_say)
  --data-root PATH       Persistent/extracted data root
  --skip-deploy          Keep the deployed UI worktree unchanged
  --venvs-only           Only refresh UI and available runtime environments
  --links-only           Only refresh data/runtime links and environment file
  --ui-auth              Require generated HTTP Basic credentials
  --no-ui-auth           Allow access to the web UI without credentials
  --no-start             Install and enable units without starting them
  --dry-run              Print filesystem, network, and systemd changes
  -h, --help             Show this help
EOF
}

die() {
	printf 'install-user: %s\n' "$*" >&2
	exit 1
}

log() {
	printf '[hay-say] %s\n' "$*"
}

print_command() {
	printf ' +'
	printf ' %q' "$@"
	printf '\n'
}

run() {
	if ((DRY_RUN)); then
		print_command "$@"
	else
		"$@"
	fi
}

while (($#)); do
	case "$1" in
		--source)
			(($# >= 2)) || die '--source requires a path'
			SOURCE_ROOT=$2
			shift 2
			;;
		--install-root)
			(($# >= 2)) || die '--install-root requires a path'
			INSTALL_ROOT=$2
			shift 2
			;;
		--data-root)
			(($# >= 2)) || die '--data-root requires a path'
			DATA_ROOT=$2
			shift 2
			;;
		--skip-deploy)
			SKIP_DEPLOY=1
			shift
			;;
		--venvs-only)
			MODE=venvs
			SKIP_DEPLOY=1
			NO_START=1
			shift
			;;
		--links-only)
			MODE=links
			SKIP_DEPLOY=1
			NO_START=1
			shift
			;;
		--ui-auth)
			UI_AUTH_ENABLED=1
			UI_AUTH_SET=1
			shift
			;;
		--no-ui-auth)
			UI_AUTH_ENABLED=0
			UI_AUTH_SET=1
			shift
			;;
		--no-start)
			NO_START=1
			shift
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			die "unknown option: $1"
			;;
	esac
done

[[ "$SOURCE_ROOT" == /* ]] || die '--source must be absolute'
[[ "$INSTALL_ROOT" == /* ]] || die '--install-root must be absolute'
[[ "$DATA_ROOT" == /* ]] || die '--data-root must be absolute'
[[ -d "$SOURCE_ROOT" ]] || die "source repository does not exist: $SOURCE_ROOT"
[[ -r "$SOURCE_ROOT/ubuntuserver/config/images.tsv" ]] || die 'source is not a Hay Say repository'
if ((EUID == 0)) && ((!DRY_RUN)); then
	die 'run the user stage as the target user, not root'
fi

readonly UI_ROOT="$INSTALL_ROOT/hay_say_ui"
readonly VENV_ROOT="$INSTALL_ROOT/.venvs"
readonly UI_VENV="$VENV_ROOT/ui"
readonly CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/hay-say"
readonly STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}/hay-say"
readonly ENV_FILE="$CONFIG_HOME/environment"
readonly UI_AUTH_FILE="$CONFIG_HOME/ui-auth"
readonly USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
readonly RUNNER_DIR="$HOME/.local/lib/hay-say/bin"
REPO_CONTENT_ROOT=$SOURCE_ROOT

deploy_repository() {
	if ((SKIP_DEPLOY)); then
		[[ -d "$UI_ROOT" ]] || die "deployed repository is missing: $UI_ROOT"
		return
	fi
	if [[ "$SOURCE_ROOT" -ef "$UI_ROOT" ]]; then
		log 'source is already the deployed repository; skipping worktree sync'
		return
	fi

	log "deploying current worktree to $UI_ROOT"
	run mkdir -p "$INSTALL_ROOT"
	if [[ ! -e "$UI_ROOT" ]]; then
		if [[ -d "$SOURCE_ROOT/.git" ]]; then
			run git clone --no-hardlinks --no-checkout "$SOURCE_ROOT" "$UI_ROOT"
		else
			run mkdir -p "$UI_ROOT"
		fi
	elif [[ ! -d "$UI_ROOT" ]]; then
		die "deployment target is not a directory: $UI_ROOT"
	elif [[ ! -d "$UI_ROOT/.git" && -d "$SOURCE_ROOT/.git" ]]; then
		log 'installing missing Git metadata without replacing deployed files'
		run mkdir -p "$UI_ROOT/.git"
		run rsync -a "$SOURCE_ROOT/.git/" "$UI_ROOT/.git/"
	fi

	run rsync -a --delete-delay \
		--exclude '/.git/' \
		--exclude '/.planning/' \
		--exclude '/.agents/' \
		--exclude '/.codex/' \
		--exclude '**/.git/' \
		--exclude '__pycache__/' \
		--exclude '*.pyc' \
		--exclude '*.pyo' \
		--filter='dir-merge,- .gitignore' \
		"$SOURCE_ROOT/" "$UI_ROOT/"
	run rm -rf -- "$UI_ROOT/.planning" "$UI_ROOT/.agents" "$UI_ROOT/.codex"
}

link_persistent_directory() {
	local name=$1
	local data_path="$DATA_ROOT/$name"
	local install_path="$INSTALL_ROOT/$name"

	if [[ -L "$install_path" || ! -e "$install_path" ]]; then
		run mkdir -p "$data_path"
		run ln -sfn "$data_path" "$install_path"
		return
	fi
	if [[ ! -d "$install_path" ]]; then
		die "persistent path is not a directory: $install_path"
	fi

	# Existing installations may already contain large model/cache trees. Keep
	# them in place and point the new data layout at them rather than moving or
	# deleting user data implicitly.
	if [[ ! -e "$data_path" && ! -L "$data_path" ]]; then
		run ln -s "$install_path" "$data_path"
		log "preserved existing $install_path in place"
	elif [[ "$data_path" -ef "$install_path" ]]; then
		:
	else
		log "warning: both $install_path and $data_path exist; preserving $install_path"
	fi
}

link_runtime_sources() {
	local manifest="$REPO_CONTENT_ROOT/ubuntuserver/config/images.tsv"
	local runtime_id image container_path destination extra
	local name extracted bundled target
	declare -A linked=()

	while IFS=$'\t' read -r runtime_id image container_path destination extra; do
		[[ -z "$runtime_id" || "$runtime_id" == \#* ]] && continue
		[[ -z "${extra:-}" ]] || die "invalid runtime manifest row for $runtime_id"
		[[ "$destination" == runtime-sources/* ]] || continue
		name=${destination##*/}
		[[ -z "${linked[$name]:-}" ]] || continue
		linked[$name]=1
		extracted="$DATA_ROOT/$destination"
		bundled="$UI_ROOT/ubuntuserver/hay_say/$name"
		target="$INSTALL_ROOT/$name"

		if [[ -e "$extracted" ]]; then
			link_source=$extracted
		elif [[ -e "$bundled" ]]; then
			link_source=$bundled
		else
			log "runtime source is not installed yet: $name"
			continue
		fi

		if [[ -L "$target" || ! -e "$target" ]]; then
			run ln -sfn "$link_source" "$target"
		else
			log "warning: preserving non-symlink runtime path $target"
		fi
	done < "$manifest"

	# The published svc5 image names its active tree "_v2" while the runtime
	# registry and older integrations use the stable architecture name.
	if [[ -e "$INSTALL_ROOT/so_vits_svc_5_v2" ]]; then
		if [[ -L "$INSTALL_ROOT/so_vits_svc_5" || ! -e "$INSTALL_ROOT/so_vits_svc_5" ]]; then
			run ln -sfn "$INSTALL_ROOT/so_vits_svc_5_v2" "$INSTALL_ROOT/so_vits_svc_5"
		fi
	fi
}

overlay_bundled_native_sources() {
	local bundled_root="$REPO_CONTENT_ROOT/ubuntuserver/hay_say"
	local extracted_root="$DATA_ROOT/runtime-sources"
	local name
	for name in so_vits_svc_3 so_vits_svc_3_server; do
		[[ -d "$bundled_root/$name" ]] || die "bundled native source is missing: $name"
		if [[ -d "$extracted_root/$name" ]]; then
			log "overlaying native $name implementation without deleting extracted assets"
			run rsync -a --exclude .git/ --exclude __pycache__/ \
				"$bundled_root/$name/" "$extracted_root/$name/"
		fi
	done

	if [[ -d "$extracted_root/so_vits_svc_3" ]]; then
		run rm -f "$extracted_root/so_vits_svc_3/inference_main_template.py"
		if ((!DRY_RUN)); then
			[[ -f "$extracted_root/so_vits_svc_3/inference/runtime.py" ]] || \
				die 'SVC3 overlay is missing inference/runtime.py'
			[[ ! -e "$extracted_root/so_vits_svc_3/inference_main_template.py" ]] || \
				die 'obsolete SVC3 inference template remains after overlay'
		fi
	else
		[[ -f "$bundled_root/so_vits_svc_3/inference/runtime.py" ]] || \
			die 'bundled SVC3 source is missing inference/runtime.py'
		if [[ -e "$bundled_root/so_vits_svc_3/inference_main_template.py" ]]; then
			((DRY_RUN)) && log 'SVC3 template must be removed before a real installation' || \
				die 'bundled SVC3 still contains inference_main_template.py'
		fi
	fi
}

# shellcheck source=native-patches.sh
source "$SCRIPT_DIR/native-patches.sh"

write_environment() {
	local environment_tmp="$CONFIG_HOME/.environment.$$.tmp"
	local auth_tmp="$CONFIG_HOME/.ui-auth.$$.tmp"
	local value ui_username ui_password saved_auth_enabled
	for value in "$INSTALL_ROOT" "$DATA_ROOT" "$UI_ROOT" "$UI_VENV" "$CONFIG_HOME" "$STATE_HOME"; do
		[[ "$value" != *$'\n'* && "$value" != *"'"* ]] || die "unsupported quote or newline in path: $value"
	done

	run mkdir -p "$CONFIG_HOME" "$STATE_HOME/runtimes" "$STATE_HOME/logs" \
		"$DATA_ROOT/cache/huggingface" "$DATA_ROOT/cache/torch" \
		"$DATA_ROOT/cache/xdg" "$DATA_ROOT/cache/nltk"
	if ((DRY_RUN)); then
		printf ' + write environment configuration to %q\n' "$ENV_FILE"
		printf ' + create or preserve UI authentication config in %q\n' "$UI_AUTH_FILE"
		print_command install -m 600 "$REPO_CONTENT_ROOT/ubuntuserver/hosts" "$CONFIG_HOME/hosts"
		return
	fi

	umask 077
	if [[ -r "$UI_AUTH_FILE" ]]; then
		saved_auth_enabled=$(awk -F= '$1 == "enabled" { print substr($0, 9); exit }' "$UI_AUTH_FILE")
		ui_username=$(awk -F= '$1 == "username" { print substr($0, 10); exit }' "$UI_AUTH_FILE")
		ui_password=$(awk -F= '$1 == "password" { print substr($0, 10); exit }' "$UI_AUTH_FILE")
		if ((!UI_AUTH_SET)) && [[ "$saved_auth_enabled" == 0 || "$saved_auth_enabled" == 1 ]]; then
			UI_AUTH_ENABLED=$saved_auth_enabled
		fi
		[[ -n "$ui_username" && "$ui_password" =~ ^[a-f0-9]{48}$ ]] || \
			die "UI credential file is malformed: $UI_AUTH_FILE"
	else
		ui_username=$USER
		ui_password=$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n')
	fi
	{
		printf 'enabled=%s\n' "$UI_AUTH_ENABLED"
		printf 'username=%s\n' "$ui_username"
		printf 'password=%s\n' "$ui_password"
	} > "$auth_tmp"
	mv -f "$auth_tmp" "$UI_AUTH_FILE"
	chmod 600 "$UI_AUTH_FILE"
	{
		printf "HAY_SAY_INSTALL_ROOT='%s'\n" "$INSTALL_ROOT"
		printf "HAY_SAY_DATA_ROOT='%s'\n" "$DATA_ROOT"
		printf "HAY_SAY_HOME='%s'\n" "$INSTALL_ROOT"
		printf "HAY_SAY_UI='%s'\n" "$UI_ROOT"
		printf "HAY_SAY_UI_VENV='%s'\n" "$UI_VENV"
		printf "HAY_SAY_RUNTIME_CONFIG='%s'\n" "$UI_ROOT/ubuntuserver/runtime/runtimes.json"
		printf "HAY_SAY_RUNTIME_STATE_DIR='%s'\n" "$STATE_HOME/runtimes"
		printf "HAY_SAY_RUNTIME_LOG_DIR='%s'\n" "$STATE_HOME/logs"
		printf "HAY_SAY_STATE_DIR='%s'\n" "$STATE_HOME"
		printf "HAY_SAY_REDIS_SOCKET='%s'\n" "$HOME/redis.sock"
		printf "HAY_SAY_REDIS_PORT='7379'\n"
		printf "HAY_SAY_REDIS_URL='redis+socket://%s/redis.sock'\n" "$HOME"
		printf "HAY_SAY_RUNTIME_MANAGER_URL='http://127.0.0.1:6588'\n"
		printf "HAY_SAY_NATIVE='1'\n"
		printf "HAY_SAY_UI_AUTH_ENABLED='%s'\n" "$UI_AUTH_ENABLED"
		printf "HAY_SAY_UI_USERNAME='%s'\n" "$ui_username"
		printf "HAY_SAY_UI_PASSWORD='%s'\n" "$ui_password"
		printf "HAY_SAY_UI_BIND='0.0.0.0:6573'\n"
		printf "HAY_SAY_STYLETTS_2_HOST='127.0.0.1'\n"
		printf "HAY_SAY_STYLETTS_2_PORT='6580'\n"
		printf "weight_root='%s'\n" "$INSTALL_ROOT/rvc/assets/weights"
		printf "rmvpe_root='%s'\n" "$INSTALL_ROOT/rvc/assets/rmvpe"
		printf "HOSTALIASES='%s'\n" "$CONFIG_HOME/hosts"
		printf "HF_HOME='%s'\n" "$DATA_ROOT/cache/huggingface"
		printf "TORCH_HOME='%s'\n" "$DATA_ROOT/cache/torch"
		printf "XDG_CACHE_HOME='%s'\n" "$DATA_ROOT/cache/xdg"
		printf "NLTK_DATA='%s'\n" "$DATA_ROOT/cache/nltk"
		printf "PYTHONUNBUFFERED='1'\n"
	} > "$environment_tmp"
	mv -f "$environment_tmp" "$ENV_FILE"
	chmod 600 "$ENV_FILE"
	install -m 600 "$REPO_CONTENT_ROOT/ubuntuserver/hosts" "$CONFIG_HOME/hosts"
}

install_uv() {
	UV_BIN="$HOME/.local/bin/uv"
	if [[ -x "$UV_BIN" ]]; then
		return
	fi
	if command -v uv >/dev/null 2>&1; then
		UV_BIN=$(command -v uv)
		return
	fi

	log 'installing uv for the target user'
	if ((DRY_RUN)); then
		print_command curl -LsSf https://astral.sh/uv/install.sh -o /tmp/hay-say-uv-installer.sh
		print_command env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh /tmp/hay-say-uv-installer.sh
		return
	fi
	uv_installer=$(mktemp "${TMPDIR:-/tmp}/hay-say-uv.XXXXXX")
	trap 'rm -f -- "$uv_installer"' RETURN
	curl -LsSf https://astral.sh/uv/install.sh -o "$uv_installer"
	env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh "$uv_installer"
	rm -f -- "$uv_installer"
	trap - RETURN
	[[ -x "$UV_BIN" ]] || die 'uv installer completed without installing uv'
}

requirements_digest() {
	local python_version=$1
	shift
	{
		printf 'python_version=%s\n' "$python_version"
		printf 'environment_strategy=resolved-rebuild-v2\n'
		sha256sum "$@"
	} | sha256sum | cut -d ' ' -f 1
}

sync_venv() {
	local venv_path=$1
	local python_version=$2
	shift 2
	local -a requirements=("$@")
	local stamp="$venv_path/.hay-say-requirements.sha256"
	local digest actual_python=
	local recreate=0
	local requirement
	local -a install_args=()
	for requirement in "${requirements[@]}"; do
		[[ -r "$requirement" ]] || die "requirements file is missing: $requirement"
	done

	digest=$(requirements_digest "$python_version" "${requirements[@]}")
	if [[ -x "$venv_path/bin/python" ]]; then
		actual_python=$("$venv_path/bin/python" -c \
			'import platform; print(platform.python_version())' 2>/dev/null) || recreate=1
		if [[ "$python_version" == *.*.* ]]; then
			[[ "$actual_python" == "$python_version" ]] || recreate=1
		else
			[[ "$actual_python" == "$python_version".* ]] || recreate=1
		fi
		if [[ ! -r "$stamp" || "$(<"$stamp")" != "$digest" ]]; then
			recreate=1
		fi
	fi
	if ((!DRY_RUN)); then
		if ((!recreate)) && [[ -x "$venv_path/bin/python" && -r "$stamp" && "$(<"$stamp")" == "$digest" ]]; then
			log "environment is current: $venv_path"
			return
		fi
	fi

	run mkdir -p "$VENV_ROOT"
	if [[ ! -x "$venv_path/bin/python" ]] || ((recreate)); then
		run "$UV_BIN" python install "$python_version"
		if ((recreate)); then
			log "recreating $venv_path for Python $python_version (found ${actual_python:-unknown})"
			run "$UV_BIN" venv --clear --python "$python_version" --seed "$venv_path"
		else
			run "$UV_BIN" venv --python "$python_version" --seed "$venv_path"
		fi
	fi
	# These are top-level pinned manifests rather than fully compiled transitive
	# locks. Rebuild on changes, then resolve their dependency closure normally.
	install_args=("$UV_BIN" pip install --python "$venv_path/bin/python" --index-strategy unsafe-best-match)
	for requirement in "${requirements[@]}"; do
		install_args+=(-r "$requirement")
	done
	run "${install_args[@]}"
	if ((DRY_RUN)); then
		printf ' + update requirements stamp in %q\n' "$stamp"
	else
		printf '%s\n' "$digest" > "$stamp"
	fi
}

normalize_runtime_binaries() {
	local runtime_id=$1
	local venv_path=$2
	local python_version=$3
	local library

	case "$runtime_id" in
		controllable_talknet)
			# PESQ is an optional training metric and is not used during TalkNet
			# inference. Its legacy extension is not ABI-compatible with the
			# runtime's pinned NumPy, but TorchMetrics imports it when present.
			if ((DRY_RUN)); then
				print_command "$UV_BIN" pip uninstall --python "$venv_path/bin/python" pesq
			elif "$venv_path/bin/python" -c \
				'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pesq") else 1)'; then
				run "$UV_BIN" pip uninstall --python "$venv_path/bin/python" pesq
			fi
			library="$venv_path/lib/python$python_version/site-packages/torch/lib/libtorch_cpu.so"
			if ((!DRY_RUN)) && [[ ! -f "$library" ]]; then
				die "TalkNet Torch library is missing: $library"
			fi
			# The CUDA 11.3 wheel marks GNU_STACK executable, which current
			# Ubuntu kernels reject. Clearing that metadata does not alter code.
			run patchelf --clear-execstack "$library"
			;;
		gpt_so_vits)
			# LangSegment 0.2.0 and its 0.3.5 preservation package install
			# the same import directory. Remove the broken distribution first,
			# then restore the selected package so upgrades cannot mix files.
			if ((DRY_RUN)); then
				print_command "$UV_BIN" pip uninstall --python "$venv_path/bin/python" LangSegment
				print_command "$UV_BIN" pip install --python "$venv_path/bin/python" \
					--reinstall-package langsegment-backup 'langsegment-backup==0.3.5.post1'
			elif "$venv_path/bin/python" -c \
				'import importlib.metadata as m; m.distribution("LangSegment")' >/dev/null 2>&1; then
				run "$UV_BIN" pip uninstall --python "$venv_path/bin/python" LangSegment
				run "$UV_BIN" pip install --python "$venv_path/bin/python" \
					--reinstall-package langsegment-backup 'langsegment-backup==0.3.5.post1'
			fi

			# The image's ONNX Runtime wheel carries an executable GNU_STACK flag,
			# which current Ubuntu kernels reject before GPT-SoVITS can import it.
			if ((DRY_RUN)); then
				print_command patchelf --clear-execstack \
					"$venv_path/lib/python$python_version/site-packages/onnxruntime/capi/onnxruntime_pybind11_state*.so"
			else
				library=$(find \
					"$venv_path/lib/python$python_version/site-packages/onnxruntime/capi" \
					-maxdepth 1 -type f -name 'onnxruntime_pybind11_state*.so' -print -quit)
				[[ -n "$library" ]] || die "GPT-SoVITS ONNX Runtime library is missing"
				run patchelf --clear-execstack "$library"
			fi
			;;
	esac
}

install_ui_venv() {
	log 'checking the pinned UI environment'
	sync_venv "$UI_VENV" 3.10.16 \
		"$REPO_CONTENT_ROOT/ubuntuserver/config/ui-requirements.lock" \
		"$REPO_CONTENT_ROOT/ubuntuserver/runtime/requirements.txt"
}

install_runtime_venvs() {
	local registry="$REPO_CONTENT_ROOT/ubuntuserver/config/runtime-venvs.tsv"
	local runtime_id python_version requirements_path extra
	local requirements venv_path runtime_source server_source

	while IFS=$'\t' read -r runtime_id python_version requirements_path extra; do
		[[ -z "$runtime_id" || "$runtime_id" == \#* ]] && continue
		[[ -z "${extra:-}" ]] || die "invalid venv registry row for $runtime_id"
		requirements="$REPO_CONTENT_ROOT/ubuntuserver/$requirements_path"
		runtime_source="$INSTALL_ROOT/$runtime_id"
		server_source="$INSTALL_ROOT/${runtime_id}_server"
		if ((!DRY_RUN)) && [[ ! -e "$runtime_source" || ! -e "$server_source" ]]; then
			log "skipping $runtime_id environment until its image is extracted"
			continue
		fi
		venv_path="$VENV_ROOT/$runtime_id"
		log "checking runtime environment: $runtime_id (Python $python_version)"
		sync_venv "$venv_path" "$python_version" "$requirements"
		normalize_runtime_binaries "$runtime_id" "$venv_path" "$python_version"

	done < "$registry"
}

migrate_legacy_user_units() {
	local legacy_unit="$USER_UNIT_DIR/redis.service"
	local legacy_target=

	[[ -L "$legacy_unit" ]] || return 0
	legacy_target=$(readlink "$legacy_unit")
	case "$legacy_target" in
		"$UI_ROOT"/ubuntuserver/*redis.service)
			log 'removing the legacy Hay Say redis.service unit'
			if ((DRY_RUN)); then
				print_command systemctl --user disable --now redis.service
			else
				systemctl --user disable --now redis.service >/dev/null 2>&1 || true
			fi
			run rm -f "$legacy_unit" "$USER_UNIT_DIR/default.target.wants/redis.service"
			;;
	esac
}

wait_for_user_stack() {
	local deadline=$((SECONDS + 120))
	local service all_active
	local redis_socket="${HAY_SAY_REDIS_SOCKET:-$HOME/redis.sock}"
	local -a ui_curl_arguments=()
	local -a services=(
		hay-say.target
		hay-say-redis.service
		hay-say-runtime-manager.service
		hay-say-celery-download.service
		hay-say-celery-cpu.service
		hay-say-celery-gpu.service
		hay-say-ui.service
	)
	case "${HAY_SAY_UI_AUTH_ENABLED:-1}" in
		0|false|no|off) ;;
		*) ui_curl_arguments=(--user "${HAY_SAY_UI_USERNAME:-}:${HAY_SAY_UI_PASSWORD:-}") ;;
	esac

	if ((DRY_RUN)); then
		printf ' + wait for Hay Say services and loopback health endpoints (120 seconds)\n'
		return
	fi
	log 'waiting for user services and health endpoints'
	while ((SECONDS < deadline)); do
		all_active=1
		for service in "${services[@]}"; do
			systemctl --user is-active --quiet "$service" || all_active=0
		done
		if ((all_active)) && \
			curl -fsS --max-time 2 http://127.0.0.1:6588/health >/dev/null 2>&1 && \
			curl -fsS --max-time 2 "${ui_curl_arguments[@]}" \
				http://127.0.0.1:6573/ >/dev/null 2>&1 && \
			[[ -S "$redis_socket" ]] && \
			redis-cli -s "$redis_socket" ping 2>/dev/null | grep -qx PONG; then
			log 'user services are ready'
			return
		fi
		sleep 2
	done
	systemctl --user --no-pager --full status "${services[@]}" >&2 || true
	die 'Hay Say services did not become ready within 120 seconds'
}

install_user_units() {
	log 'installing user systemd units'
	migrate_legacy_user_units
	run mkdir -p "$USER_UNIT_DIR" "$RUNNER_DIR"
	run install -m 755 "$REPO_CONTENT_ROOT/ubuntuserver/bin/run-service.sh" "$RUNNER_DIR/run-service"
	for unit in "$REPO_CONTENT_ROOT"/ubuntuserver/systemd/user/*; do
		[[ -f "$unit" ]] || continue
		run install -m 644 "$unit" "$USER_UNIT_DIR/${unit##*/}"
	done

	if ! command -v systemctl >/dev/null 2>&1 && ((!DRY_RUN)); then
		die 'systemctl is required to install user services'
	fi
	run systemctl --user daemon-reload
	run systemctl --user enable hay-say.target
	if ((!NO_START)); then
		run systemctl --user restart hay-say.target
		wait_for_user_stack
	fi
}

stop_active_stack_for_update() {
	if ! command -v systemctl >/dev/null 2>&1; then
		((DRY_RUN)) || die 'systemctl is required to stop Hay Say before updating it'
		return
	fi
	if systemctl --user is-active --quiet hay-say.target; then
		log 'stopping active Hay Say services before updating files and environments'
		run systemctl --user stop hay-say.target
	fi
}

stop_active_stack_for_update
deploy_repository

if [[ "$MODE" == all || "$MODE" == links ]]; then
	log 'configuring persistent data and runtime links'
	run mkdir -p "$INSTALL_ROOT" "$DATA_ROOT"
	link_persistent_directory models
	link_persistent_directory audio_cache
	overlay_bundled_native_sources
	apply_native_patches
	link_runtime_sources
	write_environment
fi

if [[ "$MODE" == all || "$MODE" == venvs ]]; then
	if [[ -r "$ENV_FILE" ]]; then
		# shellcheck source=/dev/null
		source "$ENV_FILE"
	fi
	install_uv
	install_ui_venv
	install_runtime_venvs
fi

if [[ "$MODE" == all ]]; then
	install_user_units
fi

log "user installation complete ($UI_ROOT)"

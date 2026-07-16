#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${HAY_SAY_ENV_FILE:-$HOME/.config/hay-say/environment}"

usage() {
	printf 'usage: %s [--env PATH]\n' "$0"
}

while (($#)); do
	case "$1" in
		--env)
			(($# >= 2)) || { usage >&2; exit 2; }
			ENV_FILE=$2
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			usage >&2
			exit 2
			;;
	esac
done

FAILURES=0
WARNINGS=0

pass() { printf '[ok]   %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*"; WARNINGS=$((WARNINGS + 1)); }
fail() { printf '[fail] %s\n' "$*"; FAILURES=$((FAILURES + 1)); }

if [[ -r "$ENV_FILE" ]]; then
	# shellcheck source=/dev/null
	source "$ENV_FILE"
	pass "environment: $ENV_FILE"
else
	fail "environment is missing: $ENV_FILE"
fi

HAY_SAY_HOME="${HAY_SAY_HOME:-$HOME/hay_say}"
HAY_SAY_UI="${HAY_SAY_UI:-$HAY_SAY_HOME/hay_say_ui}"
HAY_SAY_UI_VENV="${HAY_SAY_UI_VENV:-$HAY_SAY_HOME/.venvs/ui}"
HAY_SAY_DATA_ROOT="${HAY_SAY_DATA_ROOT:-$HAY_SAY_HOME/data}"
HAY_SAY_REDIS_SOCKET="${HAY_SAY_REDIS_SOCKET:-$HOME/redis.sock}"
declare -a UI_CURL_ARGUMENTS=()
case "${HAY_SAY_UI_AUTH_ENABLED:-1}" in
	1|true|yes|on)
		UI_CURL_ARGUMENTS=(--user "${HAY_SAY_UI_USERNAME:-}:${HAY_SAY_UI_PASSWORD:-}")
		pass 'web UI authentication: enabled'
		;;
	0|false|no|off)
		pass 'web UI authentication: disabled'
		;;
	*)
		fail "invalid HAY_SAY_UI_AUTH_ENABLED value: $HAY_SAY_UI_AUTH_ENABLED"
		;;
esac

# A noninteractive `sudo -iu USER` login does not populate the user's systemd
# bus variables. Use the canonical per-user bus when it exists so the command
# documented in README reports the real unit state.
USER_RUNTIME_DIR="/run/user/$(id -u)"
if [[ -S "$USER_RUNTIME_DIR/bus" ]]; then
	export XDG_RUNTIME_DIR="$USER_RUNTIME_DIR"
	export DBUS_SESSION_BUS_ADDRESS="unix:path=$USER_RUNTIME_DIR/bus"
fi

for command in ffmpeg git patchelf redis-cli rsync sha256sum systemctl uv; do
	if command -v "$command" >/dev/null 2>&1 || [[ "$command" == uv && -x "$HOME/.local/bin/uv" ]]; then
		pass "command: $command"
	else
		fail "command is unavailable: $command"
	fi
done

for path in "$HAY_SAY_UI" "$HAY_SAY_HOME/models" "$HAY_SAY_HOME/audio_cache"; do
	[[ -e "$path" ]] && pass "path: $path" || fail "path is missing: $path"
done
[[ -x "$HAY_SAY_UI_VENV/bin/python" ]] && pass 'UI virtual environment' || fail 'UI virtual environment is missing'

declare -A expected_images=()
declare -A expected_digests=()
image_manifest="$HAY_SAY_UI/ubuntuserver/config/images.tsv"
digest_manifest="$HAY_SAY_UI/ubuntuserver/config/image-digests.tsv"
if [[ -r "$image_manifest" && -r "$digest_manifest" ]]; then
	while IFS=$'\t' read -r runtime_id image _; do
		[[ -z "$runtime_id" || "$runtime_id" == \#* ]] && continue
		expected_images[$runtime_id]=$image
	done < "$image_manifest"
	while IFS=$'\t' read -r runtime_id _ digest _; do
		[[ -z "$runtime_id" || "$runtime_id" == \#* ]] && continue
		expected_digests[$runtime_id]=$digest
	done < "$digest_manifest"
	manifest_sha256=$(sha256sum "$image_manifest")
	manifest_sha256=${manifest_sha256%% *}
else
	fail 'image or digest manifest is missing'
	manifest_sha256=
fi

provenance_value() {
	local key=$1
	local file=$2
	awk -v prefix="$key=" 'index($0, prefix) == 1 { print substr($0, length(prefix) + 1); exit }' "$file"
}

runtime_ids=(controllable_talknet so_vits_svc_3 so_vits_svc_4 so_vits_svc_5 rvc styletts_2 gpt_so_vits)
for runtime_id in "${runtime_ids[@]}"; do
	if [[ -x "$HAY_SAY_HOME/.venvs/$runtime_id/bin/python" ]]; then
		pass "runtime venv: $runtime_id"
	else
		fail "runtime venv is missing: $runtime_id"
	fi
	if [[ -e "$HAY_SAY_HOME/${runtime_id}_server/main.py" ]]; then
		pass "runtime server: $runtime_id"
	else
		fail "runtime server is missing: $runtime_id"
	fi
	provenance_file="$HAY_SAY_DATA_ROOT/provenance/$runtime_id.provenance"
	if [[ -r "$provenance_file" ]]; then
		provenance_runtime=$(provenance_value runtime_id "$provenance_file")
		provenance_image=$(provenance_value source_image "$provenance_file")
		provenance_digests=$(provenance_value repo_digests "$provenance_file")
		provenance_manifest=$(provenance_value manifest_sha256 "$provenance_file")
		provenance_format=$(provenance_value format_version "$provenance_file")
		provenance_image_id=$(provenance_value image_id "$provenance_file")
		if [[ "$provenance_format" == 1 && "$provenance_runtime" == "$runtime_id" && \
			"$provenance_image" == "${expected_images[$runtime_id]:-}" && \
			"$provenance_digests" == *"${expected_digests[$runtime_id]:-missing}"* && \
			"$provenance_manifest" == "$manifest_sha256" && \
			"$provenance_image_id" == sha256:* ]]; then
			pass "image provenance: $runtime_id"
		else
			fail "image provenance is stale or malformed: $runtime_id"
		fi
	else
		warn "image provenance is missing: $runtime_id"
	fi
done

if [[ -f "$HAY_SAY_HOME/so_vits_svc_3/inference/runtime.py" && \
	! -e "$HAY_SAY_HOME/so_vits_svc_3/inference_main_template.py" ]]; then
	pass 'SVC3 uses the native long-lived runtime'
else
	fail 'SVC3 native overlay is incomplete or the obsolete template remains'
fi

if systemctl --user is-enabled --quiet hay-say.target 2>/dev/null; then
	pass 'hay-say.target is enabled'
else
	fail 'hay-say.target is not enabled'
fi
if systemctl --user is-active --quiet hay-say.target 2>/dev/null; then
	pass 'hay-say.target is active'
else
	warn 'hay-say.target is not active'
fi

services=(
	hay-say-redis.service
	hay-say-runtime-manager.service
	hay-say-celery-download.service
	hay-say-celery-cpu.service
	hay-say-celery-gpu.service
	hay-say-ui.service
)
for service in "${services[@]}"; do
	if systemctl --user is-active --quiet "$service" 2>/dev/null; then
		pass "service: $service"
	else
		fail "service is not active: $service"
	fi
done

if [[ -S "$HAY_SAY_REDIS_SOCKET" ]] && redis-cli -s "$HAY_SAY_REDIS_SOCKET" ping 2>/dev/null | grep -qx PONG; then
	pass 'Redis socket responds'
else
	warn "Redis is not responding at $HAY_SAY_REDIS_SOCKET"
fi
if curl -fsS --max-time 2 http://127.0.0.1:6588/health >/dev/null 2>&1; then
	pass 'runtime manager API responds'
else
	warn 'runtime manager API is not responding on 127.0.0.1:6588'
fi
if curl -fsS --max-time 5 "${UI_CURL_ARGUMENTS[@]}" \
	http://127.0.0.1:6573/ >/dev/null 2>&1; then
	pass 'web UI responds'
else
	fail 'web UI is not responding on port 6573'
fi

for host_alias in redis so_vits_svc_3_server styletts_2_server gpt_so_vits_server; do
	if getent hosts "$host_alias" >/dev/null 2>&1; then
		pass "host alias: $host_alias"
	else
		fail "host alias does not resolve: $host_alias"
	fi
done

available_kib=$(df -Pk "$HAY_SAY_DATA_ROOT" 2>/dev/null | awk 'NR == 2 {print $4}')
if [[ "$available_kib" =~ ^[0-9]+$ ]]; then
	available_gib=$((available_kib / 1024 / 1024))
	((available_gib >= 20)) && pass "data free space: ${available_gib} GiB" || warn "low data free space: ${available_gib} GiB"
fi

printf '\nDoctor finished with %d failure(s) and %d warning(s).\n' "$FAILURES" "$WARNINGS"
((FAILURES == 0))

#!/usr/bin/env bash
set -euo pipefail

readonly COMPONENT="${1:-}"
readonly ENV_FILE="${HAY_SAY_ENV_FILE:-$HOME/.config/hay-say/environment}"

if [[ -z "$COMPONENT" ]]; then
	printf 'usage: %s COMPONENT\n' "$0" >&2
	exit 2
fi
if [[ ! -r "$ENV_FILE" ]]; then
	printf 'Hay Say environment file is missing: %s\n' "$ENV_FILE" >&2
	exit 1
fi

# The installer writes this file with shell- and systemd-compatible quoting.
# shellcheck source=/dev/null
set -a
source "$ENV_FILE"
set +a

: "${HAY_SAY_HOME:?missing HAY_SAY_HOME in $ENV_FILE}"
: "${HAY_SAY_UI:?missing HAY_SAY_UI in $ENV_FILE}"
: "${HAY_SAY_UI_VENV:?missing HAY_SAY_UI_VENV in $ENV_FILE}"
: "${HAY_SAY_REDIS_PORT:=7379}"
: "${HAY_SAY_REDIS_SOCKET:=$HOME/redis.sock}"

readonly -a ARCHITECTURES=(
	ControllableTalkNet
	SoVitsSvc3
	SoVitsSvc4
	SoVitsSvc5
	Rvc
	StyleTTS2
	GPTSoVITS
)

architecture_arguments=()
for architecture in "${ARCHITECTURES[@]}"; do
	architecture_arguments+=(--include_architecture "$architecture")
done

cd "$HAY_SAY_UI"
export PYTHONPATH="$HAY_SAY_UI${PYTHONPATH:+:$PYTHONPATH}"

case "$COMPONENT" in
	redis)
		mkdir -p "$(dirname "$HAY_SAY_REDIS_SOCKET")"
		exec /usr/bin/redis-server \
			--bind 127.0.0.1 \
			--port "$HAY_SAY_REDIS_PORT" \
			--protected-mode yes \
			--save '' \
			--appendonly no \
			--dir "${HAY_SAY_STATE_DIR:-$HOME/.local/state/hay-say}" \
			--unixsocket "$HAY_SAY_REDIS_SOCKET" \
			--unixsocketperm 700
		;;
	runtime-manager)
		exec "$HAY_SAY_UI_VENV/bin/python" -m ubuntuserver.runtime \
			--config "${HAY_SAY_RUNTIME_CONFIG:-$HAY_SAY_UI/ubuntuserver/runtime/runtimes.json}"
		;;
	celery-download)
		exec "$HAY_SAY_UI_VENV/bin/celery" \
			--workdir "$HAY_SAY_UI" \
			-A celery_download:celery_app worker \
			--loglevel "${HAY_SAY_LOG_LEVEL:-INFO}" \
			--concurrency "${HAY_SAY_DOWNLOAD_CONCURRENCY:-5}" \
			--hostname "hay-say-download@$(hostname)" \
			--cache_implementation file \
			"${architecture_arguments[@]}"
		;;
	celery-cpu)
		exec "$HAY_SAY_UI_VENV/bin/celery" \
			--workdir "$HAY_SAY_UI" \
			-A celery_generate_cpu:celery_app worker \
			--loglevel "${HAY_SAY_LOG_LEVEL:-INFO}" \
			--concurrency "${HAY_SAY_CPU_CONCURRENCY:-4}" \
			--prefetch-multiplier 1 \
			--hostname "hay-say-cpu@$(hostname)" \
			--cache_implementation file \
			"${architecture_arguments[@]}"
		;;
	celery-gpu)
		exec "$HAY_SAY_UI_VENV/bin/celery" \
			--workdir "$HAY_SAY_UI" \
			-A celery_generate_gpu:celery_app worker \
			--loglevel "${HAY_SAY_LOG_LEVEL:-INFO}" \
			--concurrency "${HAY_SAY_GPU_CONCURRENCY:-1}" \
			--prefetch-multiplier 1 \
			--hostname "hay-say-gpu@$(hostname)" \
			--cache_implementation file \
			"${architecture_arguments[@]}"
		;;
	ui)
		exec "$HAY_SAY_UI_VENV/bin/gunicorn" \
			--config "$HAY_SAY_UI/server_initialization.py" \
			--workers "${HAY_SAY_UI_WORKERS:-2}" \
			--bind "${HAY_SAY_UI_BIND:-0.0.0.0:6573}" \
			'wsgi:get_server(enable_model_management=True, update_model_lists_on_startup=False, enable_session_caches=False, migrate_models=True, cache_implementation="file", architectures=["ControllableTalkNet", "SoVitsSvc3", "SoVitsSvc4", "SoVitsSvc5", "Rvc", "StyleTTS2", "GPTSoVITS"])'
		;;
	*)
		printf 'unknown Hay Say component: %s\n' "$COMPONENT" >&2
		exit 2
		;;
esac

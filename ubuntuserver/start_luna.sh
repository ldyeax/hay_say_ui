#!/usr/bin/env bash
# start_luna.sh
set -euo pipefail

export LIMITED_USER=luna
export HOME_DIR="/home/$LIMITED_USER"
export HOSTALIASES="$HOME_DIR/hosts"
export REDIS_BIND="127.0.0.1"
export REDIS_PORT="7379"
export REDIS_PID="$HOME_DIR/redis.pid"
export REDIS_LOG="$HOME_DIR/redis.log"

RUN_DIR="$HOME_DIR"
#mkdir -p "$RUN_DIR"

export HAY_SAY_UI="$HOME_DIR/hay_say/hay_say_ui"
export SERVER_DIR="$HAY_SAY_UI/ubuntuserver"
chmod +x "$SERVER_DIR"/*.sh

CELERY_DL_PID="$RUN_DIR/celery_download.pid"
CELERY_GPU_PID="$RUN_DIR/celery_generate_gpu.pid"
CELERY_CPU_PID="$RUN_DIR/celery_generate_cpu.pid"
GUNICORN_PID="$RUN_DIR/gunicorn.pid"

PIDS=()

kill_pidfile() {
	local pidfile="$1"
	local sig="${2:-TERM}"

	if [[ -f "$pidfile" ]]; then
		local pid
		pid="$(cat "$pidfile" 2>/dev/null || true)"
		if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
			kill "-$sig" "$pid" 2>/dev/null || true
		fi
	fi
}

wait_pid_dead() {
	local pidfile="$1"
	local timeout="${2:-3}"

	if [[ ! -f "$pidfile" ]]; then
		return 0
	fi

	local pid
	pid="$(cat "$pidfile" 2>/dev/null || true)"
	if [[ -z "${pid:-}" ]]; then
		return 0
	fi

	local i
	for ((i=0; i<timeout*10; i++)); do
		if ! kill -0 "$pid" 2>/dev/null; then
			return 0
		fi
		sleep 0.1
	done
	return 1
}

cleanup() {
	# Stop app processes first (celery/gunicorn), then redis.
	# Try TERM, then KILL. Use pidfiles so we get the *real* process even if it detached.

	screen -S celery_download -X quit || true
	screen -S celery_generate_gpu -X quit || true
	screen -S celery_generate_cpu -X quit || true

	kill_pidfile "$GUNICORN_PID" TERM || true
	kill_pidfile "$CELERY_DL_PID" TERM || true
	kill_pidfile "$CELERY_GPU_PID" TERM || true
	kill_pidfile "$CELERY_CPU_PID" TERM || true

	wait_pid_dead "$GUNICORN_PID" 3 || true
	wait_pid_dead "$CELERY_DL_PID" 3 || true
	wait_pid_dead "$CELERY_GPU_PID" 3 || true
	wait_pid_dead "$CELERY_CPU_PID" 3 || true

	kill_pidfile "$GUNICORN_PID" KILL || true
	kill_pidfile "$CELERY_DL_PID" KILL || true
	kill_pidfile "$CELERY_GPU_PID" KILL || true
	kill_pidfile "$CELERY_CPU_PID" KILL || true

	# Extra safety: if anything still survived, kill matching processes for this user.
	# (Avoids nuking other users' services.)
	pkill -u "$LIMITED_USER" -f 'celery.*worker' >/dev/null 2>&1 || true
	pkill -u "$LIMITED_USER" -x gunicorn >/dev/null 2>&1 || true

	# Stop redis cleanly if possible
	if [[ -f "$REDIS_PID" ]]; then
		redis-cli -p "$REDIS_PORT" -h "$REDIS_BIND" shutdown nosave >/dev/null 2>&1 || true
		if kill -0 "$(cat "$REDIS_PID")" >/dev/null 2>&1; then
			kill "$(cat "$REDIS_PID")" >/dev/null 2>&1 || true
		fi
	fi
	pkill -u "$LIMITED_USER" -x redis-server >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

systemctl --user start redis

# Wait until it's up
while true; do
	if redis-cli -h "$REDIS_BIND" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
		break
	fi
	echo "Waiting for redis on $REDIS_BIND:$REDIS_PORT.."
	sleep 1
done

echo "Redis is up on $REDIS_BIND:$REDIS_PORT (log: $REDIS_LOG)"

while true; do
	if [[ -e $REDIS_SOCK ]]; then
		break
	fi
	echo "Waiting for redis sock at $REDIS_SOCK.."
done

cd "$HAY_SAY_UI"

screen_pids=()
SCREEN_DOWNLOAD_LOG="$HOME_DIR/celery_download.log"
SCREEN_GPU_LOG="$HOME_DIR/celery_generate_gpu.log"
SCREEN_CPU_LOG="$HOME_DIR/celery_generate_cpu.log"
rm -f "$SCREEN_DOWNLOAD_LOG"
echo "Starting celery_download in screen session 'celery_download' (log: $SCREEN_DOWNLOAD_LOG)"
screen -L -Logfile "$SCREEN_DOWNLOAD_LOG" -dmS celery_download "$SERVER_DIR/celery_download.sh"
rm -f "$SCREEN_GPU_LOG"
echo "Starting celery_generate_gpu in screen session 'celery_generate_gpu' (log: $SCREEN_GPU_LOG)"
screen -L -Logfile "$SCREEN_GPU_LOG" -dmS celery_generate_gpu "$SERVER_DIR/celery_generate_gpu.sh"
rm -f "$SCREEN_CPU_LOG"
echo "Starting celery_generate_cpu in screen session 'celery_generate_cpu' (log: $SCREEN_CPU_LOG)"
screen -L -Logfile "$SCREEN_CPU_LOG" -dmS celery_generate_cpu "$SERVER_DIR/celery_generate_cpu.sh"

gunicorn \
	--config=server_initialization.py \
	--workers 6 \
	--bind 0.0.0.0:6573 'wsgi:get_server(enable_model_management=True, update_model_lists_on_startup=False, enable_session_caches=False, migrate_models=True, cache_implementation="file", architectures=["ControllableTalkNet", "SoVitsSvc3", "SoVitsSvc4", "SoVitsSvc5", "Rvc", "StyleTTS2", "GPTSoVITS"])' &

# loop to make sure screen sessions all exist, if not call cleanup and exit

while true; do
	for session in celery_download celery_generate_gpu celery_generate_cpu; do
		if ! screen -list | grep -q "$session"; then
			echo "Screen session $session failed to start. Cleaning up and exiting."
			cleanup
			exit 1
		fi
	done
	sleep 1
done

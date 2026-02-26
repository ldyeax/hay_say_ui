#!/usr/bin/env bash
set -euo pipefail

export LIMITED_USER=luna
export HOME_DIR="/home/$LIMITED_USER"
export HOSTALIASES="$HOME_DIR/hosts"
export REDIS_BIND="127.0.0.1"
export REDIS_PORT="6379"
export REDIS_PID="$HOME_DIR/redis.pid"
export REDIS_LOG="$HOME_DIR/redis.log"

RUN_DIR="$HOME_DIR/.run_luna"
mkdir -p "$RUN_DIR"

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

	kill_pidfile "$GUNICORN_PID" TERM
	kill_pidfile "$CELERY_DL_PID" TERM
	kill_pidfile "$CELERY_GPU_PID" TERM
	kill_pidfile "$CELERY_CPU_PID" TERM

	wait_pid_dead "$GUNICORN_PID" 3 || true
	wait_pid_dead "$CELERY_DL_PID" 3 || true
	wait_pid_dead "$CELERY_GPU_PID" 3 || true
	wait_pid_dead "$CELERY_CPU_PID" 3 || true

	kill_pidfile "$GUNICORN_PID" KILL
	kill_pidfile "$CELERY_DL_PID" KILL
	kill_pidfile "$CELERY_GPU_PID" KILL
	kill_pidfile "$CELERY_CPU_PID" KILL

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

# Start redis (no persistence, local only)
redis-server \
	--bind "$REDIS_BIND" \
	--port "$REDIS_PORT" \
	--protected-mode yes \
	--save "" \
	--appendonly no \
	--daemonize yes \
	--pidfile "$REDIS_PID" \
	--logfile "$REDIS_LOG"

# Wait until it's up
while true; do
	if redis-cli -h "$REDIS_BIND" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
		break
	fi
	echo "Waiting for redis on $REDIS_BIND:$REDIS_PORT.."
	sleep 1
done

echo "Redis is up on $REDIS_BIND:$REDIS_PORT (log: $REDIS_LOG)"

cd "$HOME_DIR/hay_say/hay_say_ui"

# Start Celery workers and Gunicorn, each with a pidfile we can kill reliably.
celery --workdir "$HOME_DIR/hay_say/hay_say_ui/" -A celery_download:celery_app worker --loglevel=INFO --concurrency 5 --pidfile "$CELERY_DL_PID" &
PIDS+=("$!")
celery --workdir "$HOME_DIR/hay_say/hay_say_ui/" -A celery_generate_gpu:celery_app worker --loglevel=INFO --concurrency 1 --pidfile "$CELERY_GPU_PID" &
PIDS+=("$!")
celery --workdir "$HOME_DIR/hay_say/hay_say_ui/" -A celery_generate_cpu:celery_app worker --loglevel=INFO --concurrency 1 --pidfile "$CELERY_CPU_PID" &
PIDS+=("$!")
gunicorn --config=server_initialization.py --workers 1 --bind 0.0.0.0:6573 --pid "$GUNICORN_PID" 'wsgi:get_server()' &
PIDS+=("$!")

# Exit if ANY process exits non-zero.
# (Also, Ctrl-C will trigger trap and cleanup.)
while :; do
	if ! wait -n; then
		exit 1
	fi
done

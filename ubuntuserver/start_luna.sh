SCRIPT_DIR="/home/luna/hay_say/hay_say_ui/ubuntuserver"
ARCH_START_LUNA="$SCRIPT_DIR/arch_start_luna"
source "$SCRIPT_DIR/variables.sh"
source "$HOME_DIR/.venv/bin/activate"
# try to read list of arches from ARCH_START_LUNA, if it exists, into an array, otherwise default to empty array
if [ -f "$ARCH_START_LUNA" ]; then
	mapfile -t ARCH_START_LUNA_ARCHES < "$ARCH_START_LUNA"
else
	ARCH_START_LUNA_ARCHES=()
fi

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

#screens="celery_download celery_generate_gpu celery_generate_cpu gunicorn"
# make array
screens=(celery_download celery_generate_gpu celery_generate_cpu gunicorn)
for arch in "${ARCH_START_LUNA_ARCHES[@]}"; do
	screens+=("$arch")
done

screen_session_exists() {
	local session="$1"
	screen -list | grep -Fq ".$session"
}

for screen in "${screens[@]}"; do
	if screen_session_exists "$screen"; then
		echo "Screen session $screen already exists. Exiting."
		exit 1
	fi
done

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

	# screen -S celery_download -X quit || true
	# screen -S celery_generate_gpu -X quit || true
	# screen -S celery_generate_cpu -X quit || true
	for screen in "${screens[@]}"; do
		screen -S "$screen" -X quit || true
	done

	# kill_pidfile "$GUNICORN_PID" TERM || true
	# kill_pidfile "$CELERY_DL_PID" TERM || true
	# kill_pidfile "$CELERY_GPU_PID" TERM || true
	# kill_pidfile "$CELERY_CPU_PID" TERM || true
	for screen in "${screens[@]}"; do
		pidfile="$HOME_DIR/${screen}.pid"
		kill_pidfile "$pidfile" TERM || true
	done

	# wait_pid_dead "$GUNICORN_PID" 3 || true
	# wait_pid_dead "$CELERY_DL_PID" 3 || true
	# wait_pid_dead "$CELERY_GPU_PID" 3 || true
	# wait_pid_dead "$CELERY_CPU_PID" 3 || true
	for screen in "${screens[@]}"; do
		pidfile="$HOME_DIR/${screen}.pid"
		wait_pid_dead "$pidfile" 3 || true
	done

	# kill_pidfile "$GUNICORN_PID" KILL || true
	# kill_pidfile "$CELERY_DL_PID" KILL || true
	# kill_pidfile "$CELERY_GPU_PID" KILL || true
	# kill_pidfile "$CELERY_CPU_PID" KILL || true
	for screen in "${screens[@]}"; do
		pidfile="$HOME_DIR/${screen}.pid"
		kill_pidfile "$pidfile" KILL || true
	done

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
let 'rediswait = 1'
while true; do
	if redis-cli -h "$REDIS_BIND" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
		echo "Redis is up on $REDIS_BIND:$REDIS_PORT (log: $REDIS_LOG)"
		break
	fi
	printf "\rWaiting for redis on $REDIS_BIND:$REDIS_PORT.. $rediswait     "
	let 'rediswait++'
	sleep 1
done
printf "\n"


while true; do
	if [[ -e $REDIS_SOCK ]]; then
		echo "Found redis sock at $REDIS_SOCK"
		break
	fi
	echo "Waiting for redis sock at $REDIS_SOCK.."
done

cd "$HAY_SAY_UI"

# SCREEN_DOWNLOAD_LOG="$HOME_DIR/celery_download.log"
# SCREEN_GPU_LOG="$HOME_DIR/celery_generate_gpu.log"
# SCREEN_CPU_LOG="$HOME_DIR/celery_generate_cpu.log"
# rm -f "$SCREEN_DOWNLOAD_LOG"
# echo "Starting celery_download in screen session 'celery_download' (log: $SCREEN_DOWNLOAD_LOG)"
# screen -L -Logfile "$SCREEN_DOWNLOAD_LOG" -dmS celery_download "$SERVER_DIR/celery_download.sh"
# rm -f "$SCREEN_GPU_LOG"
# echo "Starting celery_generate_gpu in screen session 'celery_generate_gpu' (log: $SCREEN_GPU_LOG)"
# screen -L -Logfile "$SCREEN_GPU_LOG" -dmS celery_generate_gpu "$SERVER_DIR/celery_generate_gpu.sh"
# rm -f "$SCREEN_CPU_LOG"
# echo "Starting celery_generate_cpu in screen session 'celery_generate_cpu' (log: $SCREEN_CPU_LOG)"
# screen -L -Logfile "$SCREEN_CPU_LOG" -dmS celery_generate_cpu "$SERVER_DIR/celery_generate_cpu.sh"

monitored_screens=()
for screen in "${screens[@]}"; do
	scriptfile="$SERVER_DIR/${screen}.sh"
	if [[ ! -x "$scriptfile" ]]; then
		continue
	fi
	monitored_screens+=("$screen")
	logfile="$HOME_DIR/${screen}.log"
	rm -f "$logfile"
	echo "Starting $screen in screen session '$screen' (log: $logfile)"
	screen -L -Logfile "$logfile" -dmS "$screen" "$SERVER_DIR/${screen}.sh"
done

echo "Starting gunicorn"

# get architectures_arg from ARCH_START_LUNA_ARCHES, formatted like architectures=[\"arch1\", \"arch2\"]
architectures_arg="architectures=["
for arch in "${ARCH_START_LUNA_ARCHES[@]}"; do
	architectures_arg="$architectures_arg\"$arch\", "
done
architectures_arg="${architectures_arg%, }]" # remove trailing comma and space, add closing bracket

wsgi_args="enable_model_management=True"
wsgi_args="$wsgi_args, update_model_lists_on_startup=True"
wsgi_args="$wsgi_args, enable_session_caches=False"
wsgi_args="$wsgi_args, migrate_models=True"
wsgi_args="$wsgi_args, cache_implementation=\"file\""
wsgi_args="$wsgi_args, $architectures_arg"

gunicorn \
	--config=server_initialization.py \
	--workers 6 \
	--bind 0.0.0.0:6573 "wsgi:get_server($wsgi_args)" &

sleep 1

while true; do
	for session in "${monitored_screens[@]}"; do
		if ! screen_session_exists "$session"; then
			echo "Screen session $session failed to start. Cleaning up and exiting."
			cleanup
			exit 1
		fi
	done
	sleep 1
done

export LIMITED_USER=luna
export HOME_DIR="/home/$LIMITED_USER"
export HOSTALIASES="$HOME_DIR/hosts"
export REDIS_BIND="127.0.0.1"
export REDIS_PORT="7379"
export REDIS_PID="$HOME_DIR/redis.pid"
export REDIS_LOG="$HOME_DIR/redis.log"
export REDIS_SOCK="$HOME_DIR/redis.sock"
export HAY_SAY_UI="$HOME_DIR/hay_say/hay_say_ui"
export SERVER_DIR="$HAY_SAY_UI/ubuntuserver"
export MODELS_VENVS_CONFIG_DIR="$SERVER_DIR/model_venvs"
export HAY_SAY_SRC="$SERVER_DIR/hay_say"
export HAY_SAY_PYTHON_VERSION="3.12.13"
export HAY_SAY_SHARED_PYTHON_LIB_DIR="$HOME_DIR/.local/share/uv/python/cpython-$HAY_SAY_PYTHON_VERSION-linux-x86_64-gnu/lib"

if [ -d "$HAY_SAY_SHARED_PYTHON_LIB_DIR" ]; then
	export LD_LIBRARY_PATH="$HAY_SAY_SHARED_PYTHON_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

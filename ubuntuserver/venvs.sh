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
export HAY_SAY="$HOME_DIR/hay_say"
export HAY_SAY_SRC="$SERVER_DIR/hay_say"

cd $HOME_DIR/hay_say
mkdir .venvs
# uv venv --seed /home/luna/hay_say/.venvs/so_vits_svc_3
# loop through directory names in $MODELS_CONFIG_DIR and create a venv for each one
for dir in "$MODELS_VENVS_CONFIG_DIR"/*/; do
	dir="${dir%/}"
	echo dir=$dir
	name="$(basename "$dir")"
	venv_path="$HAY_SAY/.venvs/$name"
	if [ -d "$venv_path" ]; then
		echo "venv for $name exists"
	else
		uv venv --seed "$venv_path"
	fi
	source "$venv_path/bin/activate"
	pip install -r "$dir/requirements.txt"
	deactivate
done

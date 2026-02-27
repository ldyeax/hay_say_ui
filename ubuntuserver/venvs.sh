export LIMITED_USER=luna
export HOME_DIR="/home/$LIMITED_USER"
export HOSTALIASES="$HOME_DIR/hosts"
export REDIS_BIND="127.0.0.1"
export REDIS_PORT="7379"
export REDIS_PID="$HOME_DIR/redis.pid"
export REDIS_LOG="$HOME_DIR/redis.log"
export HAY_SAY="$HOME_DIR/hay_say"
export HAY_SAY_UI="$HOME_DIR/hay_say/hay_say_ui"
export SERVER_DIR="$HAY_SAY_UI/ubuntuserver"
export MODELS_CONFIG_DIR="$SERVER_DIR/models"

cd $HOME_DIR/hay_say
mkdir .venvs
# uv venv --seed /home/luna/hay_say/.venvs/so_vits_svc_3
# loop through directory names in $MODELS_CONFIG_DIR and create a venv for each one
for dir in "$MODELS_CONFIG_DIR"/*/; do
	dir="${dir%/}"
	name="$(basename "$dir")"
	venv_path="$HOME_DIR/.venvs/$name"
	if [ -d "$venv_path" ]; then
		echo "venv for $name exists"
	else
		uv venv --seed "$venv_path"
	fi
done
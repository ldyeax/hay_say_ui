export LIMITED_USER=luna
export HOME_DIR="/home/$LIMITED_USER"
export HOSTALIASES="$HOME_DIR/hosts"
export REDIS_BIND="127.0.0.1"
export REDIS_PORT="7379"
export REDIS_PID="$HOME_DIR/redis.pid"
export REDIS_LOG="$HOME_DIR/redis.log"
export HAY_SAY_UI="$HOME_DIR/hay_say/hay_say_ui"
export SERVER_DIR="$HAY_SAY_UI/ubuntuserver"
export MODELS_CONFIG_DIR="$SERVER_DIR/models"

cd $HOME_DIR
if [ -d ".venv" ]; then
	echo "venv exists"
else
	uv venv --seed
fi
source .venv/bin/activate


export PATH="$PATH:$HOME_DIR/.local/bin"

# Clone Hay Say
mkdir -p $HOME_DIR/hay_say
cd $HOME_DIR/hay_say
if [ -d "hay_say_ui" ]; then
	cd hay_say_ui
	git submodule update --init --recursive
	git pull
else
	git clone --recursive https://github.com/ldyeax/hay_say_ui
	cd hay_say_ui
fi

cd ubuntuserver

pip install -r requirements.txt

# install user systemd service redis.service and immediately activate

systemctl --user enable --now ./redis.service

source $SERVER_DIR/venvs.sh
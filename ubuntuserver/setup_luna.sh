SCRIPT_DIR="/home/luna/hay_say/hay_say_ui/ubuntuserver"

set -euo pipefail

echo "Installing UV"
curl -LsSf https://astral.sh/uv/install.sh | sh

export PATH="$PATH:$HOME_DIR/.local/bin"
echo Set PATH: $PATH

source "$SCRIPT_DIR/variables.sh"
echo Moving into HOME_DIR=$HOME_DIR
cd "$HOME_DIR"
echo HOME_DIR PWD=$PWD
if [ -d ".venv" ]; then
	echo "venv exists"
else
	uv venv --seed
fi
source .venv/bin/activate



# Clone Hay Say
hay_say="$HOME_DIR/hay_say"
echo Creating hay_say at $hay_say
mkdir -p $hay_say
cd "$hay_say"
echo PWD is $PWD
if [ -d "hay_say_ui" ]; then
	cd hay_say_ui
	echo Found existing hay_say_ui at $PWD
	git submodule update --init --recursive
	git pull
else
	git clone --recursive https://github.com/ldyeax/hay_say_ui
	cd hay_say_ui
	echo Cloned hay_say_ui into $PWD
fi

cd ubuntuserver

pip install -r requirements.txt

# install user systemd service redis.service and immediately activate

systemctl --user enable --now ./redis.service

#source $SERVER_DIR/venvs.sh

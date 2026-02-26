export LIMITED_USER=luna
export HOME_DIR=/home/$LIMITED_USER

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

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

#apt -y update && 
apt -y install git vim redis portaudio19-dev portaudio19-doc

#wget https://mega.nz/linux/repo/xUbuntu_25.10/amd64/megacmd-xUbuntu_25.10_amd64.deb && apt -y install "$PWD/megacmd-xUbuntu_25.10_amd64.deb"
if [[ $(which mega-cmd) ]] then
	echo "megacmd found"
else
	echo "Installing megacmd.."
	bash "$SCRIPT_DIR/megacmd.sh"
fi
#apt -y install megacmd

export LIMITED_USER=luna
export HOME_DIR=/home/$LIMITED_USER
if [ -d "$HOME_DIR" ]; then
	echo "Limited user already exists"
else
	useradd --create-home --shell /bin/bash $LIMITED_USER
fi
echo "Ensuring limited user is in video group"
usermod -aG video $LIMITED_USER

echo "Creating directories"

mkdir -p /home/luna/hay_say/models
mkdir -p /home/luna/hay_say/audio_cache

echo "Copying setup files for limited user"
#cp setup_luna.sh $HOME_DIR && chown $LIMITED_USER:$LIMITED_USER $HOME_DIR/setup_luna.sh

cat variables.sh > "$HOME_DIR/setup_luna.sh"
tail -n +2 setup_luna.sh >> "$HOME_DIR/setup_luna.sh"

#cp requirements.txt $HOME_DIR && chown $LIMITED_USER:$LIMITED_USER $HOME_DIR/requirements.txt
#rm -rf $HOME_DIR/.venv

chown -R $LIMITED_USER:$LIMITED_USER $HOME_DIR

echo "Done"

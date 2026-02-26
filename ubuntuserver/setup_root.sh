apt -y update && apt -y install git vim redis

apt -y install megacmd

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
mkdir /home/luna/hay_say && chown $LIMITED_USER:$LIMITED_USER /home/luna/hay_say && \
	mkdir /home/luna/hay_say/models && chown $LIMITED_USER:$LIMITED_USER /home/luna/hay_say/models && \
	mkdir /home/luna/hay_say/audio_cache && chown $LIMITED_USER:$LIMITED_USER /home/luna/hay_say/audio_cache

echo "Copying setup files for limited user"
cp setup_luna.sh $HOME_DIR && chown $LIMITED_USER:$LIMITED_USER $HOME_DIR/setup_luna.sh
cp requirements.txt $HOME_DIR && chown $LIMITED_USER:$LIMITED_USER $HOME_DIR/requirements.txt
rm -rf $HOME_DIR/.venv
echo "Done"

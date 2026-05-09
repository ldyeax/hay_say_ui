# if 24.04 in /etc/issue:
if grep -q "24.04" /etc/issue; then
	wget https://mega.nz/linux/repo/xUbuntu_24.04/amd64/megacmd-xUbuntu_24.04_amd64.deb && apt -y install "$PWD/megacmd-xUbuntu_24.04_amd64.deb"
elif grep -q "25.10" /etc/issue; then
	wget https://mega.nz/linux/repo/xUbuntu_25.10/amd64/megacmd-xUbuntu_25.10_amd64.deb && apt -y install "$PWD/megacmd-xUbuntu_25.10_amd64.deb"
else
	echo "Unknown Ubuntu version. Please install MegaCMD manually."
fi

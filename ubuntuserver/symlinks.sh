SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
source "$SCRIPT_DIR/variables.sh"

# loop directories in HAY_SAY_SRC, symlink from each directory to $HOME_DIR/hay_say
for dir in "$HAY_SAY_SRC"/*/; do
	dir="${dir%/}"
	name="$(basename "$dir")"
	target="$HOME_DIR/hay_say/$name"
	if [ -L "$target" ]; then
		echo "Symlink for $name already exists"
	elif [ -e "$target" ]; then
		echo "File or directory $target already exists and is not a symlink"
	else
		ln -s "$dir" "$target"
		echo "Symlinked $dir to $target"
	fi
done
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
source "$SCRIPT_DIR/variables.sh"

cd $HOME_DIR/hay_say
mkdir .venvs
# uv venv --seed /home/luna/hay_say/.venvs/so_vits_svc_3
# loop through directory names in $MODELS_CONFIG_DIR and create a venv for each one
for dir in "$MODELS_VENVS_CONFIG_DIR"/*/; do
	dir="${dir%/}"
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

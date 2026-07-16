#!/usr/bin/env bash

HAY_SAY_ENV_FILE="${HAY_SAY_ENV_FILE:-$HOME/.config/hay-say/environment}"
if [[ -r "$HAY_SAY_ENV_FILE" ]]; then
	# shellcheck source=/dev/null
	source "$HAY_SAY_ENV_FILE"
else
	export HAY_SAY_INSTALL_ROOT="${HAY_SAY_INSTALL_ROOT:-$HOME/hay_say}"
	export HAY_SAY_HOME="${HAY_SAY_HOME:-$HAY_SAY_INSTALL_ROOT}"
	export HAY_SAY_UI="${HAY_SAY_UI:-$HAY_SAY_HOME/hay_say_ui}"
fi

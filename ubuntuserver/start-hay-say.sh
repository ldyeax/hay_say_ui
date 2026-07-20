#!/usr/bin/env bash
set -euo pipefail

readonly SERVICE_USER=luna
readonly TARGET=hay-say.target

case "$(id -un)" in
	"$SERVICE_USER")
		exec systemctl --user start "$TARGET"
		;;
	root)
		exec sudo -u "$SERVICE_USER" bash -c \
			'export XDG_RUNTIME_DIR="/run/user/$(id -u)"; exec systemctl --user start hay-say.target'
		;;
	*)
		printf 'start-hay-say: run this script as root or %s\n' "$SERVICE_USER" >&2
		exit 1
		;;
esac

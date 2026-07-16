#!/usr/bin/env bash
set -euo pipefail

case "${1:-start}" in
	start) exec systemctl --user start hay-say.target ;;
	stop) exec systemctl --user stop hay-say.target ;;
	restart) exec systemctl --user restart hay-say.target ;;
	status) exec systemctl --user status hay-say.target ;;
	*) printf 'usage: %s [start|stop|restart|status]\n' "$0" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
set -euo pipefail

ARCH_ROOT="/home/luna/hay_say/so_vits_svc_3"

source /home/luna/hay_say/.venvs/so_vits_svc_3/bin/activate
cd "$ARCH_ROOT"
export PYTHONPATH="$ARCH_ROOT:${PYTHONPATH:-}"

exec python /home/luna/hay_say/so_vits_svc_3_server/main.py --ubuntuserver

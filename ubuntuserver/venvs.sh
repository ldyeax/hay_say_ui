#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
printf 'venvs.sh is deprecated; delegating to the idempotent user stage\n' >&2
exec "$SCRIPT_DIR/install-user.sh" --venvs-only "$@"

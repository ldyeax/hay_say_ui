#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
printf 'setup_root.sh is deprecated; delegating to install.sh\n' >&2
exec "$SCRIPT_DIR/install.sh" "$@"

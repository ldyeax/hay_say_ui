#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

TARGET_USER=luna
INSTALL_ROOT=
DATA_ROOT=
INSTALL_ROOT_SET=0
DATA_ROOT_SET=0
SKIP_APT=0
SKIP_IMAGES=0
NO_START=0
REMOVE_IMAGES=0
DRY_RUN=0
UI_AUTH_ARGUMENT=
declare -a SELECTED_RUNTIMES=()

usage() {
	cat <<'EOF'
Usage: sudo ./ubuntuserver/install.sh [OPTIONS]

Install or update the repository as native Ubuntu user services.

  --user USER            Service account (default: luna)
  --install-root PATH    Install root (default: /home/USER/hay_say)
  --data-root PATH       Persistent runtime/model/cache root. Defaults to
                         /mnt/sanic/hay_say when /mnt/sanic is mounted.
  --runtime ID           Extract only one runtime image; may be repeated
  --skip-apt             Do not update apt metadata or install packages
  --skip-images          Do not pull or extract runtime images
  --remove-images        Remove selected images after successful extraction
  --ui-auth              Require generated HTTP Basic credentials
  --no-ui-auth           Allow access to the web UI without credentials
  --no-start             Enable units without starting/restarting services
  --dry-run              Print all package, account, Docker, and service changes
  -h, --help             Show this help

The source repository is always discovered relative to this script, so the
installer is independent of the caller's working directory.
EOF
}

die() {
	printf 'install: %s\n' "$*" >&2
	exit 1
}

log() {
	printf '[hay-say root] %s\n' "$*"
}

print_command() {
	printf ' +'
	printf ' %q' "$@"
	printf '\n'
}

run() {
	if ((DRY_RUN)); then
		print_command "$@"
	else
		"$@"
	fi
}

while (($#)); do
	case "$1" in
		--user)
			(($# >= 2)) || die '--user requires a name'
			TARGET_USER=$2
			shift 2
			;;
		--install-root)
			(($# >= 2)) || die '--install-root requires a path'
			INSTALL_ROOT=$2
			INSTALL_ROOT_SET=1
			shift 2
			;;
		--data-root)
			(($# >= 2)) || die '--data-root requires a path'
			DATA_ROOT=$2
			DATA_ROOT_SET=1
			shift 2
			;;
		--runtime)
			(($# >= 2)) || die '--runtime requires an id'
			SELECTED_RUNTIMES+=("$2")
			shift 2
			;;
		--skip-apt)
			SKIP_APT=1
			shift
			;;
		--skip-images)
			SKIP_IMAGES=1
			shift
			;;
		--remove-images)
			REMOVE_IMAGES=1
			shift
			;;
		--ui-auth)
			UI_AUTH_ARGUMENT=--ui-auth
			shift
			;;
		--no-ui-auth)
			UI_AUTH_ARGUMENT=--no-ui-auth
			shift
			;;
		--no-start)
			NO_START=1
			shift
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			die "unknown option: $1"
			;;
	esac
done

[[ "$TARGET_USER" =~ ^[a-z_][a-z0-9_-]*$ ]] || die "invalid user name: $TARGET_USER"
if ((EUID != 0)) && ((!DRY_RUN)); then
	die 'run this entrypoint as root (or use --dry-run to preview)'
fi

if id "$TARGET_USER" >/dev/null 2>&1; then
	TARGET_HOME=$(getent passwd "$TARGET_USER" | cut -d: -f6)
else
	TARGET_HOME="/home/$TARGET_USER"
fi
[[ -n "$TARGET_HOME" && "$TARGET_HOME" == /* ]] || die "cannot determine home for $TARGET_USER"

validate_managed_root() {
	local label=$1
	local path=$2
	path=$(realpath -m -- "$path") || die "cannot normalize $label: $path"
	case "$path" in
		/|/bin|/bin/*|/boot|/boot/*|/dev|/dev/*|/etc|/etc/*|/lib|/lib/*|/lib32|/lib32/*|/lib64|/lib64/*|/proc|/proc/*|/root|/root/*|/run|/run/*|/sbin|/sbin/*|/sys|/sys/*|/tmp|/tmp/*|/usr|/usr/*)
			die "$label is an unsafe system path: $path"
			;;
		/home|/media|/mnt|/opt|/srv|/var)
			die "$label must not be a top-level shared directory: $path"
			;;
		/home/*)
			[[ "$path" == "$TARGET_HOME"/* ]] || \
				die "$label must not modify another account's home: $path"
			;;
	esac
	[[ "$path" != "$TARGET_HOME" ]] || die "$label must be below the target user's home, not the home itself"
	printf '%s\n' "$path"
}

if ((!INSTALL_ROOT_SET)); then
	INSTALL_ROOT="$TARGET_HOME/hay_say"
fi
if ((!DATA_ROOT_SET)); then
	if mountpoint -q /mnt/sanic 2>/dev/null; then
		DATA_ROOT=/mnt/sanic/hay_say
	else
		DATA_ROOT="$INSTALL_ROOT/data"
	fi
fi
[[ "$INSTALL_ROOT" == /* ]] || die '--install-root must be absolute'
[[ "$DATA_ROOT" == /* ]] || die '--data-root must be absolute'
INSTALL_ROOT=$(validate_managed_root '--install-root' "$INSTALL_ROOT")
DATA_ROOT=$(validate_managed_root '--data-root' "$DATA_ROOT")
[[ "$INSTALL_ROOT" != "$DATA_ROOT" ]] || die '--install-root and --data-root must be different directories'
case "$INSTALL_ROOT" in
	"$DATA_ROOT/runtime-sources"|"$DATA_ROOT/runtime-sources"/*|\
	"$DATA_ROOT/cache"|"$DATA_ROOT/cache"/*|\
	"$DATA_ROOT/models"|"$DATA_ROOT/models"/*|\
	"$DATA_ROOT/audio_cache"|"$DATA_ROOT/audio_cache"/*)
		die '--install-root must not be inside a managed data directory'
		;;
esac
[[ -r "$SCRIPT_DIR/config/apt-packages.txt" ]] || die 'apt package manifest is missing'

install_packages() {
	((SKIP_APT)) && return
	local package
	local -a packages=()
	while IFS= read -r package; do
		package=${package%%#*}
		package=${package//[[:space:]]/}
		[[ -n "$package" ]] && packages+=("$package")
	done < "$SCRIPT_DIR/config/apt-packages.txt"
	if ((!SKIP_IMAGES)) && ! command -v docker >/dev/null 2>&1; then
		packages+=(docker.io)
	fi
	((${#packages[@]})) || die 'apt package manifest is empty'

	log 'installing declared Ubuntu packages noninteractively'
	run env DEBIAN_FRONTEND=noninteractive apt-get update
	run env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}"
}

ensure_account() {
	local group
	if ! id "$TARGET_USER" >/dev/null 2>&1; then
		log "creating service account $TARGET_USER"
		run useradd --create-home --home-dir "$TARGET_HOME" --shell /bin/bash "$TARGET_USER"
	fi
	for group in video render; do
		if getent group "$group" >/dev/null 2>&1; then
			run usermod -aG "$group" "$TARGET_USER"
		fi
	done

	if ((DRY_RUN)) && ! id "$TARGET_USER" >/dev/null 2>&1; then
		TARGET_GROUP=$TARGET_USER
	else
		TARGET_GROUP=$(id -gn "$TARGET_USER")
	fi
	run install -d -m 0755 -o "$TARGET_USER" -g "$TARGET_GROUP" "$INSTALL_ROOT" "$DATA_ROOT"
	run loginctl enable-linger "$TARGET_USER"
}

install_host_aliases() {
	local marker='# hay-say native runtime aliases'
	local aliases='127.0.0.1 redis controllable_talknet_server so_vits_svc_3_server so_vits_svc_4_server so_vits_svc_5_server rvc_server styletts_2_server gpt_so_vits_server'
	if grep -Fqx "$aliases" /etc/hosts 2>/dev/null; then
		return
	fi
	if ((DRY_RUN)); then
		printf ' + append %q and loopback runtime aliases to /etc/hosts\n' "$marker"
	else
		{
			grep -Fq "$marker" /etc/hosts 2>/dev/null || printf '\n%s\n' "$marker"
			printf '%s\n' "$aliases"
		} >> /etc/hosts
	fi
}

extract_images() {
	((SKIP_IMAGES)) && return
	local -a arguments=(--data-root "$DATA_ROOT" --owner "$TARGET_USER")
	local runtime_id
	for runtime_id in "${SELECTED_RUNTIMES[@]}"; do
		arguments+=(--runtime "$runtime_id")
	done
	((REMOVE_IMAGES)) && arguments+=(--remove-images)
	((DRY_RUN)) && arguments+=(--dry-run)

	log 'pulling and extracting declared runtime source/assets'
	run systemctl start docker.service
	"$SCRIPT_DIR/extract-images.sh" "${arguments[@]}"
}

run_user_stage() {
	local -a arguments=(
		--source "$REPO_ROOT"
		--install-root "$INSTALL_ROOT"
		--data-root "$DATA_ROOT"
	)
	((NO_START)) && arguments+=(--no-start)
	[[ -n "$UI_AUTH_ARGUMENT" ]] && arguments+=("$UI_AUTH_ARGUMENT")
	((DRY_RUN)) && arguments+=(--dry-run)

	if ((DRY_RUN)); then
		print_command runuser -u "$TARGET_USER" -- env \
			HOME="$TARGET_HOME" USER="$TARGET_USER" LOGNAME="$TARGET_USER" \
			PATH="$TARGET_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
			XDG_RUNTIME_DIR="/run/user/UID" \
			bash "$SCRIPT_DIR/install-user.sh" "${arguments[@]}"
		# Run the user's dry-run as the current account too, so syntax and planned
		# actions remain visible even when the target account does not exist yet.
		env HOME="$TARGET_HOME" USER="$TARGET_USER" LOGNAME="$TARGET_USER" \
			PATH="$TARGET_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
			bash "$SCRIPT_DIR/install-user.sh" "${arguments[@]}"
		return
	fi

	local target_uid
	target_uid=$(id -u "$TARGET_USER")
	run systemctl start "user@${target_uid}.service"
	run runuser -u "$TARGET_USER" -- env \
		HOME="$TARGET_HOME" USER="$TARGET_USER" LOGNAME="$TARGET_USER" \
		PATH="$TARGET_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
		XDG_RUNTIME_DIR="/run/user/$target_uid" \
		DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$target_uid/bus" \
		bash "$SCRIPT_DIR/install-user.sh" "${arguments[@]}"
}

install_packages
ensure_account
install_host_aliases
extract_images
run_user_stage

log "installation complete for $TARGET_USER"
log "UI: http://$(hostname -f 2>/dev/null || hostname):6573"
log "UI authentication config: $TARGET_HOME/.config/hay-say/ui-auth"

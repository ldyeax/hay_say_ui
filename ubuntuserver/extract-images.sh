#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly DEFAULT_MANIFEST="$SCRIPT_DIR/config/images.tsv"
readonly DIGEST_MANIFEST="$SCRIPT_DIR/config/image-digests.tsv"

if mountpoint -q /mnt/sanic 2>/dev/null; then
	DATA_ROOT=/mnt/sanic/hay_say
else
	DATA_ROOT="${HOME:-/tmp}/hay_say-data"
fi
MANIFEST=$DEFAULT_MANIFEST
OWNER="${SUDO_USER:-$(id -un)}"
PULL=1
REMOVE_IMAGES=0
DRY_RUN=0
declare -a SELECTED_RUNTIMES=()

usage() {
	cat <<'EOF'
Usage: extract-images.sh [OPTIONS]

Extract only declared Hay Say runtime trees from the seven published images.

  --data-root PATH       Extraction destination (default: /mnt/sanic/hay_say
                         when /mnt/sanic is mounted)
  --manifest PATH        Alternate image manifest
  --owner USER           Owner for extracted files (default: calling user)
  --runtime ID           Extract one runtime; may be repeated
  --no-pull              Use locally cached images without pulling
  --remove-images        Remove the seven selected images after extraction
  --dry-run              Print planned Docker and filesystem changes
  -h, --help             Show this help

The script never runs image entrypoints, copies container virtual environments,
or invokes Docker prune. It removes only containers and staging directories it
created. Image removal is disabled unless --remove-images is supplied.
EOF
}

die() {
	printf 'extract-images: %s\n' "$*" >&2
	exit 1
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

is_selected() {
	local runtime_id=$1
	local selected
	if ((${#SELECTED_RUNTIMES[@]} == 0)); then
		return 0
	fi
	for selected in "${SELECTED_RUNTIMES[@]}"; do
		[[ "$selected" == "$runtime_id" ]] && return 0
	done
	return 1
}

validate_relative_destination() {
	local destination=$1
	[[ -n "$destination" ]] || return 1
	[[ "$destination" != /* ]] || return 1
	[[ "/$destination/" != *'/../'* ]] || return 1
	[[ "/$destination/" != *'/./'* ]] || return 1
}

while (($#)); do
	case "$1" in
		--data-root)
			(($# >= 2)) || die '--data-root requires a path'
			DATA_ROOT=$2
			shift 2
			;;
		--manifest)
			(($# >= 2)) || die '--manifest requires a path'
			MANIFEST=$2
			shift 2
			;;
		--owner)
			(($# >= 2)) || die '--owner requires a user'
			OWNER=$2
			shift 2
			;;
		--runtime)
			(($# >= 2)) || die '--runtime requires an id'
			SELECTED_RUNTIMES+=("$2")
			shift 2
			;;
		--no-pull)
			PULL=0
			shift
			;;
		--remove-images)
			REMOVE_IMAGES=1
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

[[ "$DATA_ROOT" == /* ]] || die '--data-root must be absolute'
DATA_ROOT=$(realpath -m -- "$DATA_ROOT") || die "cannot normalize data root: $DATA_ROOT"
[[ "$DATA_ROOT" != / ]] || die '--data-root must not be the filesystem root'
[[ -r "$MANIFEST" ]] || die "manifest is not readable: $MANIFEST"
if ((!DRY_RUN)) || id "$OWNER" >/dev/null 2>&1; then
	id "$OWNER" >/dev/null 2>&1 || die "owner does not exist: $OWNER"
fi

declare -A IMAGES=()
declare -A KNOWN_RUNTIMES=()
while IFS=$'\t' read -r runtime_id image container_path destination extra; do
	[[ -z "$runtime_id" || "$runtime_id" == \#* ]] && continue
	[[ -z "${extra:-}" ]] || die "too many fields in manifest row for $runtime_id"
	[[ "$runtime_id" =~ ^[a-z][a-z0-9_]*$ ]] || die "invalid runtime id: $runtime_id"
	[[ "$image" == hydrusbeta/hay_say:*_server ]] || die "unexpected image reference: $image"
	case "$container_path" in
		/home/luna/hay_say/*|/home/luna/.cache/torch/*|/home/luna/nltk_data/*) ;;
		*) die "unsafe container path: $container_path" ;;
	esac
	validate_relative_destination "$destination" || die "unsafe destination: $destination"
	if [[ -n "${IMAGES[$runtime_id]:-}" && "${IMAGES[$runtime_id]}" != "$image" ]]; then
		die "runtime $runtime_id maps to more than one image"
	fi
	IMAGES[$runtime_id]=$image
	KNOWN_RUNTIMES[$runtime_id]=1
done < "$MANIFEST"

((${#IMAGES[@]} == 7)) || die "manifest must declare exactly seven runtime images (found ${#IMAGES[@]})"
declare -A EXPECTED_DIGESTS=()
while IFS=$'\t' read -r digest_runtime digest_image digest extra; do
	[[ -z "$digest_runtime" || "$digest_runtime" == \#* ]] && continue
	[[ -z "${extra:-}" ]] || die "invalid digest manifest row for $digest_runtime"
	[[ "${IMAGES[$digest_runtime]:-}" == "$digest_image" ]] || die "digest image mismatch for $digest_runtime"
	[[ "$digest" =~ ^sha256:[a-f0-9]{64}$ ]] || die "invalid image digest for $digest_runtime"
	EXPECTED_DIGESTS[$digest_runtime]=$digest
done < "$DIGEST_MANIFEST"
((${#EXPECTED_DIGESTS[@]} == 7)) || die 'digest manifest must pin all seven images'
for selected in "${SELECTED_RUNTIMES[@]}"; do
	[[ -n "${KNOWN_RUNTIMES[$selected]:-}" ]] || die "runtime is not in the manifest: $selected"
done

if ((!DRY_RUN)); then
	command -v docker >/dev/null || die 'docker is required'
	command -v rsync >/dev/null || die 'rsync is required'
	command -v sha256sum >/dev/null || die 'sha256sum is required'
fi

readonly STAGING_PARENT="$DATA_ROOT/.extract"
readonly PROVENANCE_DIR="$DATA_ROOT/provenance"
CURRENT_CONTAINER=
CURRENT_STAGE=

cleanup() {
	local exit_code=$?
	trap - EXIT INT TERM
	if [[ -n "$CURRENT_CONTAINER" ]]; then
		docker rm -f "$CURRENT_CONTAINER" >/dev/null 2>&1 || true
	fi
	if [[ -n "$CURRENT_STAGE" && "$CURRENT_STAGE" == "$STAGING_PARENT"/* ]]; then
		rm -rf -- "$CURRENT_STAGE"
	fi
	exit "$exit_code"
}
trap cleanup EXIT INT TERM

run mkdir -p "$STAGING_PARENT" "$PROVENANCE_DIR" "$DATA_ROOT/runtime-sources" "$DATA_ROOT/cache"
if ((EUID == 0)) && id "$OWNER" >/dev/null 2>&1; then
	owner_group=$(id -gn "$OWNER")
	run chown "$OWNER:$owner_group" \
		"$DATA_ROOT" "$STAGING_PARENT" "$PROVENANCE_DIR" \
		"$DATA_ROOT/runtime-sources" "$DATA_ROOT/cache"
fi

mapfile -t RUNTIME_IDS < <(printf '%s\n' "${!IMAGES[@]}" | LC_ALL=C sort)
for runtime_id in "${RUNTIME_IDS[@]}"; do
	is_selected "$runtime_id" || continue
	image=${IMAGES[$runtime_id]}
	printf 'Extracting %s from %s\n' "$runtime_id" "$image"

	if ((PULL)); then
		run docker pull "$image"
	elif ((!DRY_RUN)); then
		docker image inspect "$image" >/dev/null 2>&1 || die "image is not cached: $image"
	else
		print_command docker image inspect "$image"
	fi
	if ((DRY_RUN)); then
		printf ' + verify %s resolves to %s\n' "$image" "${EXPECTED_DIGESTS[$runtime_id]}"
	else
		resolved_digests=$(docker image inspect --format '{{json .RepoDigests}}' "$image")
		[[ "$resolved_digests" == *"${EXPECTED_DIGESTS[$runtime_id]}"* ]] || \
			die "$image does not match pinned digest ${EXPECTED_DIGESTS[$runtime_id]} (resolved: $resolved_digests)"
	fi

	CURRENT_STAGE="$STAGING_PARENT/${runtime_id}.$$"
	container_name="hay-say-extract-${runtime_id}-$$"
	if ((DRY_RUN)); then
		print_command mkdir -p "$CURRENT_STAGE"
		print_command docker create --name "$container_name" "$image" /bin/true
	else
		mkdir -p "$CURRENT_STAGE"
		CURRENT_CONTAINER=$(docker create --name "$container_name" "$image" /bin/true)
	fi

	while IFS=$'\t' read -r row_runtime row_image container_path destination _; do
		[[ "$row_runtime" == "$runtime_id" ]] || continue
		[[ "$row_image" == "$image" ]] || die "manifest changed while reading $runtime_id"
		run mkdir -p "$CURRENT_STAGE/$destination"
		if ((DRY_RUN)); then
			print_command docker cp "$container_name:$container_path/." "$CURRENT_STAGE/$destination/"
		else
			docker cp "$CURRENT_CONTAINER:$container_path/." "$CURRENT_STAGE/$destination/"
		fi
	done < "$MANIFEST"

	if ((DRY_RUN)); then
		printf ' + remove nested .git and Python cache directories below %q\n' "$CURRENT_STAGE"
	else
		find "$CURRENT_STAGE" -type d \( -name .git -o -name __pycache__ \) -prune -exec rm -rf -- {} +
		find "$CURRENT_STAGE" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	fi
	while IFS=$'\t' read -r row_runtime _ _ destination _; do
		[[ "$row_runtime" == "$runtime_id" ]] || continue
		destination_path="$DATA_ROOT/$destination"
		resolved_destination=$(realpath -m -- "$destination_path")
		[[ "$resolved_destination" == "$DATA_ROOT"/* ]] || \
			die "destination escapes the data root: $destination_path"
		[[ ! -L "$destination_path" ]] || \
			die "refusing to replace a symlinked extraction destination: $destination_path"
		run mkdir -p "$(dirname "$destination_path")" "$destination_path"
		run rsync -a --delete --delete-excluded --no-owner --no-group \
			--exclude .git/ --exclude .venv/ --exclude venv/ \
			"$CURRENT_STAGE/$destination/" "$destination_path/"
		if ((EUID == 0)) && id "$OWNER" >/dev/null 2>&1; then
			run chown -R "$OWNER:$owner_group" "$destination_path"
		fi
	done < "$MANIFEST"

	if ((DRY_RUN)); then
		print_command docker rm "$container_name"
	else
		docker rm "$CURRENT_CONTAINER" >/dev/null
		CURRENT_CONTAINER=
	fi

	if ((DRY_RUN)); then
		printf ' + write digest and extraction provenance to %q\n' "$PROVENANCE_DIR/$runtime_id.provenance"
	else
		image_id=$(docker image inspect --format '{{.Id}}' "$image")
		repo_digests=$(docker image inspect --format '{{json .RepoDigests}}' "$image")
		created=$(docker image inspect --format '{{.Created}}' "$image")
		platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image")
		manifest_digest=$(sha256sum "$MANIFEST")
		manifest_digest=${manifest_digest%% *}
		provenance_tmp="$PROVENANCE_DIR/.${runtime_id}.$$.tmp"
		{
			printf 'format_version=1\n'
			printf 'runtime_id=%s\n' "$runtime_id"
			printf 'source_image=%s\n' "$image"
			printf 'image_id=%s\n' "$image_id"
			printf 'repo_digests=%s\n' "$repo_digests"
			printf 'image_created=%s\n' "$created"
			printf 'platform=%s\n' "$platform"
			printf 'manifest_sha256=%s\n' "$manifest_digest"
			printf 'extracted_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
			while IFS=$'\t' read -r row_runtime _ container_path destination _; do
				[[ "$row_runtime" == "$runtime_id" ]] || continue
				printf 'path=%s -> %s\n' "$container_path" "$destination"
			done < "$MANIFEST"
			} > "$provenance_tmp"
			mv -f "$provenance_tmp" "$PROVENANCE_DIR/$runtime_id.provenance"
			if ((EUID == 0)); then
				chown "$OWNER:$owner_group" "$PROVENANCE_DIR/$runtime_id.provenance"
			fi
		fi

	if ((DRY_RUN)); then
		print_command rm -rf "$CURRENT_STAGE"
	else
		rm -rf -- "$CURRENT_STAGE"
	fi
	CURRENT_STAGE=

	if ((REMOVE_IMAGES)); then
		run docker image rm "$image"
	fi
done

if ((DRY_RUN)); then
	printf ' + rewrite extracted absolute /home/luna/hay_say symlinks as relocatable links\n'
else
	while IFS= read -r -d '' link_path; do
		link_target=$(readlink "$link_path")
		[[ "$link_target" == /home/luna/hay_say/* ]] || continue
		relative_target=${link_target#/home/luna/hay_say/}
		new_target="$DATA_ROOT/runtime-sources/$relative_target"
		if [[ -e "$new_target" || -L "$new_target" ]]; then
			relative_link=$(realpath --relative-to="$(dirname "$link_path")" "$new_target")
			ln -sfn "$relative_link" "$link_path"
		else
			printf 'warning: unresolved container symlink %s -> %s\n' "$link_path" "$link_target" >&2
		fi
	done < <(find "$DATA_ROOT/runtime-sources" -type l -print0)
fi

printf 'Runtime extraction complete: %s\n' "$DATA_ROOT/runtime-sources"

#!/usr/bin/env bash

patch_paths() {
	awk '$1 == "---" || $1 == "+++" {
		path = $2
		sub(/^[ab]\//, "", path)
		if (path != "/dev/null") print path
	}' "$@" | LC_ALL=C sort -u
}

patch_set_digest() {
	local patch_file digest
	for patch_file in "$@"; do
		digest=$(sha256sum "$patch_file")
		printf '%s  %s\n' "${digest%% *}" "${patch_file##*/}"
	done | sha256sum | cut -d ' ' -f 1
}

copy_patch_paths() {
	local source=$1
	local destination=$2
	shift 2
	(($#)) || return 0
	(cd "$source" && cp -a --parents -- "$@" "$destination")
}

patch_set_matches() {
	local target=$1
	local direction=$2
	shift 2
	local -a patches=("$@") paths=()
	local temporary patch_file index status=0
	mapfile -t paths < <(patch_paths "${patches[@]}")
	temporary=$(mktemp -d "${TMPDIR:-/tmp}/hay-say-patch-check.XXXXXX")
	if ! copy_patch_paths "$target" "$temporary" "${paths[@]}" 2>/dev/null; then
		rm -rf -- "$temporary"
		return 1
	fi
	if [[ "$direction" == forward ]]; then
		for patch_file in "${patches[@]}"; do
			patch --forward --batch --silent -d "$temporary" -p1 < "$patch_file" >/dev/null 2>&1 || {
				status=1
				break
			}
		done
	else
		for ((index=${#patches[@]} - 1; index >= 0; index--)); do
			patch --reverse --batch --silent -d "$temporary" -p1 < "${patches[index]}" >/dev/null 2>&1 || {
				status=1
				break
			}
		done
	fi
	rm -rf -- "$temporary"
	return "$status"
}

apply_patch_set() {
	local target=$1
	local direction=$2
	shift 2
	local -a patches=("$@")
	local patch_file index
	if [[ "$direction" == forward ]]; then
		for patch_file in "${patches[@]}"; do
			patch --forward --batch -d "$target" -p1 < "$patch_file"
		done
	else
		for ((index=${#patches[@]} - 1; index >= 0; index--)); do
			patch --reverse --batch -d "$target" -p1 < "${patches[index]}"
		done
	fi
}

patch_runtime_id() {
	case "$1" in
		controllable_talknet_server) printf 'controllable_talknet\n' ;;
		so_vits_svc_4_server) printf 'so_vits_svc_4\n' ;;
		so_vits_svc_5_server) printf 'so_vits_svc_5\n' ;;
		rvc|rvc_server) printf 'rvc\n' ;;
		styletts_2|styletts_2_server) printf 'styletts_2\n' ;;
		gpt_so_vits_server) printf 'gpt_so_vits\n' ;;
		*) return 1 ;;
	esac
}

apply_native_patches() {
	local server target patch_dir patch_file patch_digest runtime_id provenance_file provenance_digest
	local state_dir base_dir saved_patch_digest saved_provenance_digest
	local index paths_changed target_has_current_patch
	local -a servers=(
		controllable_talknet_server
		so_vits_svc_4_server
		so_vits_svc_5_server
		rvc
		rvc_server
		styletts_2
		styletts_2_server
		gpt_so_vits_server
	)
	local -a patches=() current_paths=() saved_paths=() managed_paths=()
	for server in "${servers[@]}"; do
		target="$DATA_ROOT/runtime-sources/$server"
		patch_dir="$REPO_CONTENT_ROOT/ubuntuserver/patches/$server"
		[[ -d "$target" ]] || continue
		[[ -d "$patch_dir" ]] || die "native patch directory is missing: $patch_dir"
		if [[ "$server" == rvc ]] && grep -q $'\r$' "$target/infer/lib/audio.py"; then
			# The published RVC image stores this source as CRLF. Normalize the
			# maintained patch target so GNU patch behaves consistently.
			run sed -i 's/\r$//' "$target/infer/lib/audio.py"
		fi
		patches=("$patch_dir"/*.patch)
		[[ -e "${patches[0]}" ]] || die "native patch directory is empty: $patch_dir"
		for patch_file in "${patches[@]}"; do
			[[ -r "$patch_file" ]] || die "native patch is unreadable: $patch_file"
		done
		if ((DRY_RUN)); then
			printf ' + reconcile maintained patch set in %q from %q\n' "$target" "$patch_dir"
			continue
		fi

		mapfile -t current_paths < <(patch_paths "${patches[@]}")
		((${#current_paths[@]})) || die "native patch set has no file paths: $patch_dir"
		patch_digest=$(patch_set_digest "${patches[@]}")
		runtime_id=$(patch_runtime_id "$server") || die "patch runtime mapping is missing: $server"
		provenance_file="$DATA_ROOT/provenance/$runtime_id.provenance"
		provenance_digest=missing
		if [[ -r "$provenance_file" ]]; then
			provenance_digest=$(sha256sum "$provenance_file")
			provenance_digest=${provenance_digest%% *}
		fi
		state_dir="$DATA_ROOT/provenance/native-patches/$server"
		base_dir="$state_dir/base"
		[[ ! -L "$state_dir" ]] || die "native patch state must not be a symlink: $state_dir"
		mkdir -p "$state_dir"
		saved_patch_digest=
		saved_provenance_digest=
		[[ -r "$state_dir/patches.sha256" ]] && saved_patch_digest=$(<"$state_dir/patches.sha256")
		[[ -r "$state_dir/provenance.sha256" ]] && \
			saved_provenance_digest=$(<"$state_dir/provenance.sha256")
		saved_paths=()
		[[ -r "$state_dir/paths.txt" ]] && mapfile -t saved_paths < "$state_dir/paths.txt"
		mapfile -t managed_paths < <(printf '%s\n' "${saved_paths[@]}" "${current_paths[@]}" | sed '/^$/d' | LC_ALL=C sort -u)
		paths_changed=0
		if ((${#saved_paths[@]} != ${#current_paths[@]})); then
			paths_changed=1
		else
			for ((index=0; index < ${#current_paths[@]}; index++)); do
				if [[ "${saved_paths[index]}" != "${current_paths[index]}" ]]; then
					paths_changed=1
					break
				fi
			done
		fi
		target_has_current_patch=0
		if patch_set_matches "$target" reverse "${patches[@]}"; then
			target_has_current_patch=1
		fi

		if [[ -d "$base_dir" && "$saved_provenance_digest" == "$provenance_digest" && \
			"$saved_patch_digest" != "$patch_digest" ]] && \
			((!target_has_current_patch || paths_changed)); then
			for patch_file in "${managed_paths[@]}"; do
				if [[ ! -e "$base_dir/$patch_file" ]]; then
					copy_patch_paths "$target" "$base_dir" "$patch_file" || \
						die "cannot extend native patch baseline for $server: $patch_file"
				fi
			done
			copy_patch_paths "$base_dir" "$target" "${managed_paths[@]}"
		fi

		if patch_set_matches "$target" forward "${patches[@]}"; then
			rm -rf -- "$base_dir"
			mkdir -p "$base_dir"
			copy_patch_paths "$target" "$base_dir" "${current_paths[@]}"
			apply_patch_set "$target" forward "${patches[@]}"
		elif ((target_has_current_patch)); then
			if [[ ! -d "$base_dir" || "$saved_provenance_digest" != "$provenance_digest" ]]; then
				apply_patch_set "$target" reverse "${patches[@]}"
				rm -rf -- "$base_dir"
				mkdir -p "$base_dir"
				copy_patch_paths "$target" "$base_dir" "${current_paths[@]}"
				apply_patch_set "$target" forward "${patches[@]}"
			else
				log "$server maintained patch set is already applied"
			fi
		else
			die "$server cannot be reconciled with its maintained patches; re-extract runtime $runtime_id"
		fi
		printf '%s\n' "$patch_digest" > "$state_dir/patches.sha256"
		printf '%s\n' "$provenance_digest" > "$state_dir/provenance.sha256"
		printf '%s\n' "${current_paths[@]}" > "$state_dir/paths.txt"
	done
}

#!/usr/bin/env bash
# celery_generate_cpu.sh
cd $HAY_SAY_UI

SCRIPT_DIR="/home/luna/hay_say/hay_say_ui/ubuntuserver"
ARCH_START_LUNA="$SCRIPT_DIR/arch_start_luna"
ARCHITECTURE_ARGS=()
if [[ -f "$ARCH_START_LUNA" ]]; then
	while IFS= read -r arch || [[ -n "$arch" ]]; do
		[[ -n "$arch" ]] && ARCHITECTURE_ARGS+=(--include_architecture "$arch")
	done < "$ARCH_START_LUNA"
fi

celery \
	--workdir ~/hay_say/hay_say_ui/ \
	-A celery_generate_cpu:celery_app \
	worker \
	--loglevel=INFO \
	--concurrency 1 \
	--cache_implementation file \
	"${ARCHITECTURE_ARGS[@]}"

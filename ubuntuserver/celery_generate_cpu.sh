# celery_generate_cpu.sh
cd $HAY_SAY_UI
celery \
	--workdir ~/hay_say/hay_say_ui/ \
	-A celery_generate_cpu:celery_app \
	worker \
	--loglevel=INFO \
	--concurrency 1 \
	--cache_implementation file \
	--include_architecture ControllableTalkNet \
	--include_architecture SoVitsSvc3 \
	--include_architecture SoVitsSvc4 \
	--include_architecture SoVitsSvc5 \
	--include_architecture Rvc \
	--include_architecture StyleTTS2 \
	--include_architecture GPTSoVITS 

celery \
	--workdir ~/hay_say/hay_say_ui/ \
	-A celery_download:celery_app \
	worker \
	--loglevel=INFO \
	--concurrency 5 \
	--include_architecture ControllableTalkNet \
	--include_architecture SoVitsSvc3 \
	--include_architecture SoVitsSvc4 \
	--include_architecture SoVitsSvc5 \
	--include_architecture Rvc \
	--include_architecture StyleTTS2 \
	--include_architecture GPTSoVITS & 
celery \
	--workdir ~/hay_say/hay_say_ui/ \
	-A celery_generate_gpu:celery_app \
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
	--include_architecture GPTSoVITS &
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
	--include_architecture GPTSoVITS &
gunicorn \
	--config=server_initialization.py \
	--workers 6 \
	--bind 0.0.0.0:6573 'wsgi:get_server(enable_model_management=True, update_model_lists_on_startup=False, enable_session_caches=False, migrate_models=True, cache_implementation=\"file\", architectures=[\"ControllableTalkNet\", \"SoVitsSvc3\", \"SoVitsSvc4\", \"SoVitsSvc5\", \"Rvc\", \"StyleTTS2\", \"GPTSoVITS\"])'
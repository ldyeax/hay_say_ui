export HOME_DIR=/home/$LIMITED_USER

cd $HOME_DIR
rm -rf ./.venv
uv venv --seed

pip install -r requirements.txt

export PATH="$PATH:$HOME_DIR/.local/bin"

# Clone Hay Say
git clone -b main --single-branch -q https://github.com/ldyeax/hay_say_ui ~/hay_say/hay_say_ui/

# Expose port 6573, the port that Hay Say uses
#EXPOSE 6573

# Start Celery workers for background callbacks and run Hay Say on a gunicorn server
#CMD ["/bin/sh", "-c", " \
celery --workdir ~/hay_say/hay_say_ui/ -A celery_download:celery_app worker --loglevel=INFO --concurrency 5 & \
celery --workdir ~/hay_say/hay_say_ui/ -A celery_generate_gpu:celery_app worker --loglevel=INFO --concurrency 1 & \
celery --workdir ~/hay_say/hay_say_ui/ -A celery_generate_cpu:celery_app worker --loglevel=INFO --concurrency 1 & \
gunicorn --config=server_initialization.py --workers 1 --bind 0.0.0.0:6573 'wsgi:get_server()'`

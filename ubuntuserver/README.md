The ubuntuserver directory is a project to run the Hay Say UI and its model endpoints directly on Ubuntu server instead of in Docker containers. I find this makes them easier to iterate on and update.

setup_luna.sh should be run as root and sets up the limited user (luna).
start_luna.sh is run by the limited user and runs the celery workers, UI, and model servers in screen sessions.
	- Scripts are placed in the same directory as start_luna.sh named after the name of the model that gets passed to the architectures argument in gunicorn, such as SoVitsSvc3.sh
	- These scripts take care of:
		- Activating that architecture's virtual environment (e.g. /home/luna/.venvs/so_vits_svc_3)
		- Running that architecture's server (e.g. running /home/luna/hay_say/so_vits_svc3_server/main.py from /home/luna/hay_say/so_vits_svc_3/)
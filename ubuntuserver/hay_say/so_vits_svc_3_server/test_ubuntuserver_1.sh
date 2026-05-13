source /home/luna/hay_say/.venvs/so_vits_svc_3/bin/activate
cd /home/luna/hay_say/hay_say_ui/ubuntuserver/hay_say/so_vits_svc_3
export PYTHONPATH="/home/luna/hay_say/hay_say_ui/ubuntuserver/hay_say/so_vits_svc_3:/home/luna/hay_say/hay_say_ui/ubuntuserver/hay_say/so_vits_svc_3_server:${PYTHONPATH:-}"
python ../so_vits_svc_3_server/main.py --ubuntuserver --test1
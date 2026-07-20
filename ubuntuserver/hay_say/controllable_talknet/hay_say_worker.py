"""Persistent, cooperatively cancellable Controllable TalkNet worker."""

import argparse
import json
import socket
import threading
import traceback
from types import SimpleNamespace

import soundfile
from hay_say_common.worker_control import WorkerControl


class WorkerCancelled(RuntimeError):
    pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--character", required=True)
    return parser.parse_args()


def main():
    args = parse_arguments()
    control = socket.socket(fileno=args.control_fd)
    reader = control.makefile("r", encoding="utf-8")
    writer = control.makefile("w", encoding="utf-8")
    write_lock = threading.Lock()
    worker_control = WorkerControl()

    def respond(payload):
        with write_lock:
            writer.write(json.dumps(payload, sort_keys=True) + "\n")
            writer.flush()

    def read_commands():
        for line in reader:
            try:
                command = json.loads(line)
            except (TypeError, ValueError):
                continue
            worker_control.submit(command)
        worker_control.submit({"action": "stop"})

    threading.Thread(target=read_commands, name="talknet-worker-control", daemon=True).start()

    import controllable_talknet
    from controllable_talknet_cli import amend_pitch_options_if_needed, generate_audio

    respond({"status": "ready", "device": str(controllable_talknet.DEVICE)})

    while True:
        command = worker_control.commands.get()
        if command.get("action") == "stop":
            return
        if command.get("action") != "generate":
            continue
        request_id = command.get("request_id")
        worker_control.begin(request_id)

        def check_cancelled():
            if worker_control.cancelled:
                raise WorkerCancelled("Controllable TalkNet inference was cancelled")

        try:
            options = SimpleNamespace(
                reference_audio=command.get("reference_audio"),
                custom_model=args.character,
                character=None,
                text=command["user_text"],
                pitch_factor=command["pitch_factor"],
                pitch_options=list(command.get("pitch_options", [])),
            )
            options.pitch_options = amend_pitch_options_if_needed(options)
            audio, sample_rate = generate_audio(options, cancel_check=check_cancelled)
            check_cancelled()
            soundfile.write(command["output_path"], audio, sample_rate, format="FLAC")
            check_cancelled()
            respond({"status": "completed", "request_id": request_id})
        except WorkerCancelled:
            respond({"status": "cancelled", "request_id": request_id})
        except Exception:
            if worker_control.cancelled:
                respond({"status": "cancelled", "request_id": request_id})
            else:
                respond({
                    "status": "failed",
                    "request_id": request_id,
                    "error": traceback.format_exc(),
                })
        finally:
            worker_control.finish(request_id)


if __name__ == "__main__":
    main()

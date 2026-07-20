"""Persistent, cooperatively cancellable StyleTTS2 model worker."""

import argparse
import json
import os
import socket
import threading
import traceback

import hay_say_common as hsc
from hay_say_common.worker_control import WorkerControl
from hay_say_runtime import StyleTTS2Session
from hay_say_torch_bootstrap import cpu_bf16_autocast


ARCHITECTURE_NAME = "styletts_2"
CONFIG_EXTENSIONS = (".yml", ".yaml")


class WorkerCancelled(RuntimeError):
    pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--character", required=True)
    return parser.parse_args()


def config_path(character_dir):
    for extension in CONFIG_EXTENSIONS:
        paths = hsc.get_files_with_extension(character_dir, extension)
        if paths:
            return paths[0]
    raise FileNotFoundError(f"No StyleTTS2 config found in {character_dir}")


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

    threading.Thread(
        target=read_commands, name="styletts-worker-control", daemon=True
    ).start()

    character_dir = hsc.character_dir(ARCHITECTURE_NAME, args.character)
    cpu_bf16 = os.environ.get("HAY_SAY_CPU_BF16_AUTOCAST") == "1"
    with cpu_bf16_autocast(cpu_bf16):
        session = StyleTTS2Session(
            hsc.get_single_file_with_extension(character_dir, ".pth"),
            config_path(character_dir),
        )
    respond({"status": "ready", "device": str(session.device)})

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
                raise WorkerCancelled("StyleTTS2 inference was cancelled")

        try:
            with cpu_bf16_autocast(cpu_bf16):
                session.generate(cancel_check=check_cancelled, **command["options"])
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

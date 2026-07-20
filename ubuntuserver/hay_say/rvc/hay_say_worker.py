"""Persistent, cooperatively cancellable RVC model worker."""

import argparse
import json
import socket
import sys
import threading
import traceback

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

    threading.Thread(target=read_commands, name="rvc-worker-control", daemon=True).start()

    # RVC parses argv while importing its configuration module.
    sys.argv = ["", "--commandlinemode"]
    from configs.config import Config
    from infer.modules.vc.modules import VC
    from infer.modules.vc.utils import load_hubert

    config = Config()
    voice = VC(config)
    voice.get_vc(args.character + ".pth", None, None)
    voice.hubert_model = load_hubert(config)
    respond({"status": "ready", "device": str(config.device)})

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
                raise WorkerCancelled("RVC inference was cancelled")

        try:
            check_cancelled()
            info, result = voice.vc_single(
                0,
                command["input_path"],
                command["pitch_shift"],
                None,
                command["f0_method"],
                command.get("index_path", ""),
                "",
                command["index_ratio"],
                command.get("filter_radius", 0),
                0,
                command["rms_mix_ratio"],
                command["protect"],
                cancel_check=check_cancelled,
            )
            check_cancelled()
            sample_rate, audio = result
            if sample_rate is None or audio is None:
                raise RuntimeError(info)
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

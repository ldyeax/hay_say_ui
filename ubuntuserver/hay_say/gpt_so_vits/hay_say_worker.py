"""Persistent, cooperatively cancellable GPT-SoVITS model worker."""

import argparse
import json
import socket
import threading
import traceback

import hay_say_common as hsc
from hay_say_common.worker_control import WorkerControl


ARCHITECTURE_NAME = "gpt_so_vits"


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

    threading.Thread(target=read_commands, name="gpt-sovits-worker-control", daemon=True).start()

    from GPT_SoVITS.inference_cli import synthesize
    from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, device

    character_dir = hsc.character_dir(ARCHITECTURE_NAME, args.character)
    gpt_model = hsc.get_single_file_with_extension(character_dir, ".ckpt")
    sovits_model = hsc.get_single_file_with_extension(character_dir, ".pth")
    change_gpt_weights(gpt_path=gpt_model)
    change_sovits_weights(sovits_path=sovits_model)
    respond({"status": "ready", "device": str(device)})

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
                raise WorkerCancelled("GPT-SoVITS inference was cancelled")

        try:
            synthesize(
                gpt_model,
                sovits_model,
                command.get("precomputed_traits_file"),
                command.get("reference_audio"),
                command.get("reference_text_file"),
                command["reference_language"],
                command["target_text_file"],
                command["target_language"],
                command["workspace"],
                command["cutting_strategy"],
                command["top_k"],
                command["top_p"],
                command["temperature"],
                command.get("ref_free"),
                command["speed"],
                command.get("additional_refs", []),
                command.get("trait"),
                load_weights=False,
                cancel_check=check_cancelled,
            )
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

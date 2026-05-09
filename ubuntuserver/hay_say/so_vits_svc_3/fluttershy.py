import io
import logging
import queue
import threading
import time
from pathlib import Path
import os

import librosa
import numpy as np
import soundfile
import maad
import torch

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")

model_path = "/home/luna/hay_say/models/so_vits_svc_3/characters/Fluttershy/G_58000.pth"
config_path = "/home/luna/hay_say/models/so_vits_svc_3/characters/Fluttershy/config.json"
infer_tool.mkdir(["flutteraw", "flutteresults"])

# 支持多个wav文件，放在flutteraw文件夹下
clean_names = ["eruption.flac"]
trans = [i for i in range(-12, 13)]  # 一次性合成-12到+12的音高
spk_list = ["Fluttershy (speaking)"]
slice_db = -40  # 默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50
wav_format = 'flac'  # 音频输出格式
clip = 0     # 音频自动切片，0为不切片，单位为秒/s
lr = 1    # 交叉淡入时间，单位为秒/s

# Detect all available devices: CPU + every CUDA GPU
devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
print(f"Using devices: {devices}")

# Load one independent model instance per device
svc_models = {}
for device in devices:
    print(f"Loading model on {device} ...")
    svc_models[device] = Svc(model_path, config_path, device=device)

infer_tool.fill_a_to_b(trans, clean_names)

# Build the shared job queue
job_queue = queue.Queue()

for clean_name in clean_names:
    for tran in trans:
        flutteraw_audio_path = f"flutteraw/{clean_name}"
        if "." not in flutteraw_audio_path:
            flutteraw_audio_path += ".wav"
        infer_tool.format_wav(flutteraw_audio_path)
        wav_path = Path(flutteraw_audio_path).with_suffix('.wav')
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
        per_size = clip * audio_sr
        lg_size = int(lr * audio_sr)
        job_queue.put((audio_data, audio_sr, per_size, lg_size, spk_list, tran, clean_name))
        print(f'job added: {clean_name}, tran: {tran}, per_size: {per_size}, lg_size: {lg_size}')


def do_job(svc_model, job):
    audio_data, audio_sr, per_size, lg_size, spk_list, tran, clean_name = job
    for spk in spk_list:
        audio = []
        for (slice_tag, data) in audio_data:
            print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
            length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
            if slice_tag:
                print('jump empty segment')
                _audio = np.zeros(length)
                audio.extend(list(_audio))
                continue
            if per_size != 0:
                datas = infer_tool.split_list_by_n(data, per_size, pre=lg_size)
            else:
                datas = [data]
            for i, dat in enumerate(datas):
                if clip != 0:
                    print(f'###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======')
                flutteraw_path = io.BytesIO()
                soundfile.write(flutteraw_path, dat, audio_sr, format="wav")
                flutteraw_path.seek(0)
                out_audio, out_sr = svc_model.infer(spk, tran, flutteraw_path)
                _audio = out_audio.cpu().numpy()
                if clip != 0 and i != 0 and lr != 0:
                    audio = list(maad.util.crossfade(np.array(audio), _audio, lg_size))
                else:
                    audio.extend(list(_audio))
        res_path = f'./flutteresults/{clean_name}_{tran}key_{spk}.{wav_format}'
        soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)


def worker(device):
    svc_model = svc_models[device]
    print(f"[{device}] worker started")
    while True:
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            break
        _, _, _, _, _, tran, clean_name = job
        print(f"[{device}] processing {clean_name}, tran: {tran}")
        try:
            do_job(svc_model, job)
        except Exception as e:
            print(f"[{device}] ERROR on {clean_name} tran={tran}: {e}")
        finally:
            job_queue.task_done()
    print(f"[{device}] worker done")


threads = [threading.Thread(target=worker, args=(device,), daemon=True) for device in devices]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("All jobs completed.")

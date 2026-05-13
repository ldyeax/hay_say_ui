import os
import io
import logging
from pathlib import Path

import maad
import numpy as np
import soundfile

# (model_path, config_path) -> Svc instance
tup_to_svc = {}


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def infer_ubuntuserver(model_path = "logs/32k/G_174000-Copy1.pth", config_path = "configs/config.json", clean_names = ["sample-src"], trans = [0], spk_list = ['yunhao'], cpu = False):
    logging.getLogger('numba').setLevel(logging.WARNING)
    chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")
    clean_names = _as_list(clean_names)
    trans = _as_list(trans)
    spk_list = _as_list(spk_list)
 
    global tup_to_svc
    if (model_path, config_path) in tup_to_svc:
        svc_model = tup_to_svc[(model_path, config_path)]
    else:
        svc_model = Svc(model_path, config_path)
        tup_to_svc[(model_path, config_path)] = svc_model

    slice_db = -40  # Default is -40; use -30 for noisy audio and -50 to preserve breaths in clean vocal tracks
    wav_format = 'flac'  # Audio output format
    clip = 0     # Automatic audio slicing; 0 means no slicing, unit: seconds
    lr = 1    # Crossfade duration, unit: seconds
    
    infer_tool.mkdir(["raw", "results"])
    infer_tool.fill_a_to_b(trans, clean_names)

    jobs = []

    for clean_name, tran in zip(clean_names, trans):
        raw_audio_path = f"raw/{clean_name}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        wav_path = Path(raw_audio_path).with_suffix('.wav')
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)
        per_size = clip*audio_sr
        lg_size = int(lr*audio_sr)
        
        jobs.append((audio_data, audio_sr, per_size, lg_size, spk_list, tran, clean_name, clip, lr, wav_format))
    for job in jobs:
        do_job(svc_model, -1, *job)

cpu = False  # Whether to use CPU inference; set to True if needed

if cpu : os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc


    
def take_job():
    global jobs
    return jobs.pop(0)

# gpu_num < 0 = cpu, gpu_num >= 0 = gpu id
def do_job(svc_model, gpu_num, audio_data, audio_sr, per_size, lg_size, spk_list, tran, clean_name, clip, lr, wav_format):
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
                datas = infer_tool.split_list_by_n(data, per_size,pre=lg_size)
            else:
                datas = [data]
            for i,dat in enumerate(datas):
                if clip!=0: print(f'###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======')
                raw_path = io.BytesIO()
                soundfile.write(raw_path, dat, audio_sr, format="wav")
                raw_path.seek(0)
                out_audio, out_sr = svc_model.infer(spk, tran, raw_path)
                _audio = out_audio.cpu().numpy()
                if clip!=0 and i!=0 and lr!=0 : audio = list(maad.util.crossfade(np.array(audio), _audio, lg_size))
                else: audio.extend(list(_audio))
        res_path = f'./results/{clean_name}_{tran}key_{spk}.{wav_format}'
        soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)

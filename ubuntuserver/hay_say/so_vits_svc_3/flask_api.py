import io
import logging

import soundfile
import torch
import torchaudio
from flask import Flask, request, send_file
from flask_cors import CORS

from inference.infer_tool import Svc, RealTimeVC

app = Flask(__name__)

CORS(app)

logging.getLogger('numba').setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # Pitch shift information
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # Sample rate required by the DAW
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    speaker_id = int(float(request_form.get("sSpeakId", 0)))
    # Receive the WAV file over HTTP and convert it
    input_wav_path = io.BytesIO(wave_file.read())

    # Model inference
    if raw_infer:
        out_audio, out_sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path)
        tar_audio = torchaudio.functional.resample(out_audio, svc_model.target_sample, daw_sample)
    else:
        out_audio = svc.process(svc_model, speaker_id, f_pitch_change, input_wav_path)
        tar_audio = torchaudio.functional.resample(torch.from_numpy(out_audio), svc_model.target_sample, daw_sample)
    # Return the audio
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio.cpu().numpy(), daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == '__main__':
    # When enabled, use direct slice synthesis; False uses crossfade mode
    # In the VST plugin, adjusting slice time to 0.3-0.5s can reduce latency. Direct slicing can produce pops at joins, while crossfade may cause slight overlap artifacts
    # Choose the method you can tolerate, or set the VST max slice time to 1s. This is set to True here for greater stability at the cost of higher latency
    raw_infer = True
    # Each model corresponds uniquely to its config
    model_name = "logs/32k/G_174000-Copy1.pth"
    config_name = "configs/config.json"
    svc_model = Svc(model_name, config_name)
    svc = RealTimeVC()
    # This matches the VST plugin and is not recommended to change
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)

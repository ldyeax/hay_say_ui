# SoftVC VITS Singing Voice Conversion

## Important!!!!!!!!!!
SoVits is a voice conversion (speaker conversion) tool. It changes the timbre of speech in an audio clip to match a target speaker, and it is not TTS (text-to-speech). Although SoVits is based on VITS, they are different projects, so please do not confuse them. If you want to train TTS, please go to [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://github.com/jaywalnut310/vits)

## Usage Policy
1. Please resolve dataset licensing on your own. Any problems caused by training with an unauthorized dataset are entirely your responsibility and have nothing to do with sovits.
2. Any video made with sovits and published to a video platform must clearly state in the description the source singing voice or audio used for conversion. For example, if you convert audio extracted from someone else's published video or audio, you must provide the original video and music links. If you use your own voice or audio synthesized by another singing synthesis engine as the source, that must also be stated in the description.
3. Any infringement caused by the input source is entirely your responsibility. If you use other commercial singing synthesis software as the source, make sure to follow that software's terms of use. Note that many singing synthesis engines explicitly prohibit using their output as a conversion source!

## English docs
[Check here](Eng_docs.md)

## Updates
> Updated the 4.0-v2 model. The workflow is the same as 4.0. See the [4.0-v2 branch](https://github.com/innnky/so-vits-svc/tree/4.0-v2). This is the last update for sovits. \
> **The 4.0 model and Colab scripts have been updated**: in the [4.0 branch](https://github.com/innnky/so-vits-svc/tree/4.0), the unified sampling rate is 44100 Hz (while inference VRAM usage is still lower than 3.0's 32 kHz version), and the feature extractor has been changed to contentvec. Stability has not yet been broadly tested.
>
> Based on incomplete statistics, training with multiple speakers seems to worsen **timbre leakage**. It is not recommended to train models with more than 5 speakers. If you want the result to sound closer to the target timbre, the current recommendation is to **train single-speaker models whenever possible**.\
> The staccato issue has been solved, and audio quality has improved significantly.\
> Version 2.0 has been moved to the sovits_2.0 branch.\
> Version 3.0 uses FreeVC's code structure and is not compatible with older versions.\
> Compared with [DiffSVC](https://github.com/prophesier/diff-svc), diffsvc performs better when the training data is extremely high quality. For lower-quality datasets, this repository may perform better. In addition, this repository is much faster at inference.

## Model Overview
A singing voice conversion model that uses the SoftVC content encoder to extract speech features from the source audio and feeds them into VITS together with F0, replacing the original text input to achieve singing voice conversion. The vocoder is replaced with [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan) to solve the staccato issue.

## Notice
+ The current branch is the 32 kHz version. 32 kHz models infer faster, use much less VRAM, and require much less disk space for datasets, so this version is recommended.
+ If you want to train a 48 kHz model, switch to the [main branch](https://github.com/innnky/so-vits-svc/tree/main).

## Pre-downloaded Model Files
+ soft vc hubert: [hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
  + Place it in the `hubert` directory.
+ Pretrained base model files [G_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/G_0.pth) and [D_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/D_0.pth)
  + Place them in `logs/32k`.
  + The pretrained base model is required, because experiments show that training from scratch may fail to converge, and the base model also speeds up training.
  + The pretrained base model was trained on Yunhao, Jishuang, Huiyu Star AI, Paimon, and Ayachi Nene, covering common male and female vocal ranges, so it can be considered a relatively universal base model.
  + The base model has removed unrelated weights such as `optimizer speaker_embedding` and can only be used to initialize training, not for inference.
  + This base model is compatible with the 48 kHz base model.
```shell
# One-click download
# hubert
wget -P hubert/ https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt
# G and D pretrained models
wget -P logs/32k/ https://huggingface.co/innnky/sovits_pretrained/resolve/main/G_0.pth
wget -P logs/32k/ https://huggingface.co/innnky/sovits_pretrained/resolve/main/D_0.pth

```

## One-click Colab dataset creation and training script
[one-click colab](https://colab.research.google.com/drive/1_-gh9i-wCPNlRZw6pYF-9UufetcVrGBX?usp=sharing)

## Dataset Preparation
Just place the dataset into the `dataset_raw` directory using the file structure below.
```shell
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

## Data Pre-processing
1. Resample to 32 kHz

```shell
python resample.py
 ```
2. Automatically split the training, validation, and test sets, and generate the config file
```shell
python preprocess_flist_config.py
# Notice
# In the generated config file, the number of speakers `n_speakers` is automatically set according to the number of speakers in the dataset
# To leave room for adding speakers later, `n_speakers` is automatically set to twice the current number of speakers in the dataset
# If you want more empty slots, you can manually modify `n_speakers` in the generated `config.json` after this step
# Once training starts, this value cannot be changed
```
3. Generate HuBERT and F0 features
```shell
python preprocess_hubert_f0.py
```
After completing the steps above, the `dataset` directory contains the preprocessed data and you can delete the `dataset_raw` folder.

## Training
```shell
python train.py -c configs/config.json -m 32k
```

## Inference

Use [inference_main.py](inference_main.py)
+ Change `model_path` to your latest trained checkpoint
+ Put the audio to be converted in the `raw` folder
+ Set `clean_names` to the audio filenames to convert
+ Set `trans` to the number of semitones for pitch shifting
+ Set `spk_list` to the speaker names to synthesize

## Onnx Export
### Important: When exporting Onnx, clone the entire repository again!!! When exporting Onnx, clone the entire repository again!!! When exporting Onnx, clone the entire repository again!!!
Use [onnx_export.py](onnx_export.py)
+ Create a new folder named `checkpoints` and open it
+ Create a new folder inside `checkpoints` as the project folder, using your project name, for example `aziplayer`
+ Rename your model to `model.pth` and your config file to `config.json`, then place them in the `aziplayer` folder you just created
+ In [onnx_export.py](onnx_export.py), change `"NyaruTaffy"` in `path = "NyaruTaffy"` to your project name, for example `path = "aziplayer"`
+ Run [onnx_export.py](onnx_export.py)
+ Wait for it to finish; a `model.onnx` file will be generated in your project folder as the exported model
+ Note: If you want to export a 48K model, follow the steps below or use `model_onnx_48k.py` directly
   + Open [model_onnx.py](model_onnx.py) and change `sampling_rate` in the last `SynthesizerTrn` class's `hps` from 32000 to 48000
   + Open [nvSTFT](/vdecoder/hifigan/nvSTFT.py) and change every 32000 to 48000
    ### UIs that support the Onnx model
    + [MoeSS](https://github.com/NaruseMioShirakana/MoeSS)
+ I removed all training-only functions and every complicated transpose, leaving nothing behind, because only after removing those things can you tell that you are using Onnx

## Gradio (WebUI)
Use [sovits_gradio.py](sovits_gradio.py)
+ Create a `checkpoints` folder and open it
+ Create a new folder inside `checkpoints` as the project folder, using your project name
+ Rename your model to `model.pth` and your config to `config.json`, then place them in the folder you just created
+ Run [sovits_gradio.py](sovits_gradio.py)

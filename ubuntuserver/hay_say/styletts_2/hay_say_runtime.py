"""Reusable StyleTTS2 inference session for native persistent workers."""

import json
import re
from collections import OrderedDict

import gruut
import librosa
import models
import nltk.data
import numpy as np
import soundfile
import torch
import torchaudio
import yaml
from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert
from arpabetandipaconvertor.arpabet2phoneticalphabet import (
    ARPAbet2PhoneticAlphabetConvertor,
)
from munch import Munch
from text_utils import TextCleaner


INTERNAL_SAMPLE_RATE = 24000


def recursive_munch(value):
    if isinstance(value, dict):
        return Munch((key, recursive_munch(item)) for key, item in value.items())
    if isinstance(value, list):
        return [recursive_munch(item) for item in value]
    return value


def locate_pronunciation_spans(text):
    arpabet_spans = [match.span(0) for match in re.finditer(r"{(.*?)}", text)]
    ipa_spans = [match.span(0) for match in re.finditer(r"<(.*?)>", text)]
    pronunciation_spans = sorted(arpabet_spans + ipa_spans)
    if pronunciation_spans:
        plaintext_spans = [
            (left[1], right[0])
            for left, right in zip(pronunciation_spans[:-1], pronunciation_spans[1:])
        ]
        plaintext_spans = (
            [(0, pronunciation_spans[0][0])]
            + plaintext_spans
            + [(pronunciation_spans[-1][1], len(text))]
        )
    else:
        plaintext_spans = [(0, len(text))]
    plaintext_spans = [span for span in plaintext_spans if span[0] != span[1]]
    return sorted(pronunciation_spans + plaintext_spans), arpabet_spans, ipa_spans


def word_to_ipa(word):
    if not word.is_spoken:
        return word.text
    return "".join(word.phonemes or ())


def text_to_ipa(text):
    spans, arpabet_spans, ipa_spans = locate_pronunciation_spans(text)
    converter = ARPAbet2PhoneticAlphabetConvertor()
    words = []
    for span in spans:
        span_text = (
            text[span[0]:span[1]]
            .replace("{", "")
            .replace("}", "")
            .replace("<", "")
            .replace(">", "")
            .strip()
        )
        if span in ipa_spans:
            words.append(span_text)
        elif span in arpabet_spans:
            words.append(converter.convert_to_international_phonetic_alphabet(span_text))
        else:
            words.extend(
                word_to_ipa(word)
                for sentence in gruut.sentences(span_text)
                for word in sentence
            )
    return " ".join(words)


class StyleTTS2Session:
    """Own one loaded character model and perform serial inference requests."""

    def __init__(self, weights_file, config_file, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(config_file, "r", encoding="utf-8") as config_stream:
            self.config = yaml.safe_load(config_stream)

        text_aligner = models.load_ASR_models(
            self.config.get("ASR_path", False),
            self.config.get("ASR_config", False),
        )
        pitch_extractor = models.load_F0_models(self.config.get("F0_path", False))
        plbert = load_plbert(self.config.get("PLBERT_dir", False))
        self.model_params = recursive_munch(self.config["model_params"])
        self.model = models.build_model(
            self.model_params, text_aligner, pitch_extractor, plbert
        )
        for component in self.model.values():
            component.eval().to(self.device)

        checkpoint = torch.load(weights_file, map_location=self.device)
        parameters = checkpoint["net"]
        for name, component in self.model.items():
            if name not in parameters:
                continue
            try:
                component.load_state_dict(parameters[name])
            except RuntimeError:
                unwrapped = OrderedDict(
                    (key[7:] if key.startswith("module.") else key, value)
                    for key, value in parameters[name].items()
                )
                component.load_state_dict(unwrapped, strict=False)
            component.eval()

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001, sigma_max=3.0, rho=9.0
            ),
            clamp=False,
        )
        self.text_cleaner = TextCleaner()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self._sentence_tokenizer = None

    @staticmethod
    def _check(cancel_check):
        if cancel_check is not None:
            cancel_check()

    def _preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        return (torch.log(1e-5 + mel_tensor.unsqueeze(0)) + 4) / 4

    def compute_style(self, path, cancel_check=None):
        self._check(cancel_check)
        wave, sample_rate = librosa.load(path)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if sample_rate != INTERNAL_SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=INTERNAL_SAMPLE_RATE
            )
        mel_tensor = self._preprocess(audio).to(self.device)
        self._check(cancel_check)
        with torch.no_grad():
            reference_style = self.model.style_encoder(mel_tensor.unsqueeze(1))
            reference_prosody = self.model.predictor_encoder(mel_tensor.unsqueeze(1))
        self._check(cancel_check)
        return torch.cat([reference_style, reference_prosody], dim=1)

    def precomputed_style(self, path, model_name, character, trait):
        with open(path, "r", encoding="utf-8") as style_stream:
            entries = json.load(style_stream)
        for model_entry in entries:
            if model_entry.get("Model") != model_name:
                continue
            for character_entry in model_entry.get("Characters", []):
                if character_entry.get("Character") != character:
                    continue
                for style in character_entry.get("Pre-computed Styles", []):
                    if style.get("Trait") == trait:
                        return torch.tensor(
                            [style["Style Vector"]],
                            dtype=torch.float32,
                            device=self.device,
                        )
        raise ValueError(
            f"No precomputed style found for {model_name}/{character}/{trait}"
        )

    @staticmethod
    def _length_mask(lengths):
        positions = torch.arange(lengths.max()).unsqueeze(0).expand(
            lengths.shape[0], -1
        ).type_as(lengths)
        return positions + 1 > lengths.unsqueeze(1)

    def _sample_style(
        self, noise, bert_duration, diffusion_steps, embedding_scale,
        reference_style, cancel_check,
    ):
        sigmas = self.sampler.sigma_schedule(diffusion_steps, noise.device)
        inference_arguments = {
            "embedding": bert_duration,
            "embedding_scale": embedding_scale,
            "features": reference_style,
        }

        def denoise(*arguments, **keywords):
            self._check(cancel_check)
            return self.sampler.denoise_fn(
                *arguments, **{**keywords, **inference_arguments}
            )

        sampled = self.sampler.sampler(
            noise, fn=denoise, sigmas=sigmas, num_steps=diffusion_steps
        )
        self._check(cancel_check)
        return sampled.clamp(-1.0, 1.0) if self.sampler.clamp else sampled

    def infer(
        self, text, previous_style, noise, diffusion_steps, embedding_scale,
        reference_style=None, timbre_blend=0.25, prosody_blend=0.25,
        previous_blend=0.7, speed=1.0, cancel_check=None,
    ):
        self._check(cancel_check)
        phonemes = text_to_ipa(text.strip().replace('"', ""))
        tokens = self.text_cleaner(phonemes)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        self._check(cancel_check)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = self._length_mask(input_lengths).to(tokens.device)
            encoded_text = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_duration = self.model.bert(
                tokens, attention_mask=(~text_mask).int()
            )
            duration_encoding = self.model.bert_encoder(bert_duration).transpose(-1, -2)
            self._check(cancel_check)
            predicted_style = self._sample_style(
                noise, bert_duration, diffusion_steps, embedding_scale,
                reference_style, cancel_check,
            ).squeeze(0)
            if previous_style is not None:
                predicted_style = (
                    previous_blend * previous_style
                    + (1 - previous_blend) * predicted_style
                )
            prosody = predicted_style[:, 128:]
            timbre = predicted_style[:, :128]
            if reference_style is not None:
                timbre = (
                    timbre_blend * timbre
                    + (1 - timbre_blend) * reference_style[:, :128]
                )
                prosody = (
                    prosody_blend * prosody
                    + (1 - prosody_blend) * reference_style[:, 128:]
                )
                predicted_style = torch.cat([timbre, prosody], dim=-1)

            predictor_encoding = self.model.predictor.text_encoder(
                duration_encoding, prosody, input_lengths, text_mask
            )
            duration, _ = self.model.predictor.lstm(predictor_encoding)
            duration = self.model.predictor.duration_proj(duration)
            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            predicted_duration = torch.round(duration.squeeze()).clamp(min=1)
            alignment = torch.zeros(
                input_lengths, int(predicted_duration.sum().data)
            )
            frame = 0
            for index in range(alignment.size(0)):
                alignment[index, frame:frame + int(predicted_duration[index].data)] = 1
                frame += int(predicted_duration[index].data)
            alignment = alignment.unsqueeze(0).to(self.device)
            encoded_prosody = predictor_encoding.transpose(-1, -2) @ alignment
            if self.model_params.decoder.type == "hifigan":
                shifted = torch.zeros_like(encoded_prosody)
                shifted[:, :, 0] = encoded_prosody[:, :, 0]
                shifted[:, :, 1:] = encoded_prosody[:, :, :-1]
                encoded_prosody = shifted
            f0_prediction, noise_prediction = self.model.predictor.F0Ntrain(
                encoded_prosody, prosody
            )
            encoded_asr = encoded_text @ alignment
            if self.model_params.decoder.type == "hifigan":
                shifted = torch.zeros_like(encoded_asr)
                shifted[:, :, 0] = encoded_asr[:, :, 0]
                shifted[:, :, 1:] = encoded_asr[:, :, :-1]
                encoded_asr = shifted
            self._check(cancel_check)
            output = self.model.decoder(
                encoded_asr,
                f0_prediction,
                noise_prediction,
                timbre.squeeze().unsqueeze(0),
            )
        self._check(cancel_check)
        return output.squeeze().float().cpu().numpy()[..., :-100], predicted_style

    def generate(
        self, *, text, output_path, noise, style_blend, diffusion_steps,
        embedding_scale, use_long_form, reference_audio=None,
        reference_style_json=None, precomputed_style_model=None,
        precomputed_style_character=None, precomputed_style_trait=None,
        timbre_blend=0.3, prosody_blend=0.1, speed=1.0,
        cancel_check=None,
    ):
        self._check(cancel_check)
        if reference_audio:
            reference_style = self.compute_style(reference_audio, cancel_check)
        elif reference_style_json:
            reference_style = self.precomputed_style(
                reference_style_json,
                precomputed_style_model,
                precomputed_style_character,
                precomputed_style_trait,
            )
        else:
            reference_style = None
        scaled_noise = noise * torch.randn(1, 1, 256).to(self.device)

        if use_long_form:
            if self._sentence_tokenizer is None:
                self._sentence_tokenizer = nltk.data.load(
                    "tokenizers/punkt/english.pickle"
                )
            outputs = []
            previous_style = None
            for sentence in self._sentence_tokenizer.tokenize(text):
                if not sentence.strip():
                    continue
                self._check(cancel_check)
                output, previous_style = self.infer(
                    sentence,
                    previous_style,
                    scaled_noise,
                    diffusion_steps,
                    embedding_scale,
                    reference_style,
                    timbre_blend,
                    prosody_blend,
                    style_blend,
                    speed,
                    cancel_check,
                )
                outputs.append(output)
            if not outputs:
                raise ValueError("Text did not contain a speakable sentence")
            audio = np.concatenate(outputs).ravel()
        else:
            audio, _ = self.infer(
                text,
                None,
                scaled_noise,
                diffusion_steps,
                embedding_scale,
                reference_style,
                timbre_blend,
                prosody_blend,
                style_blend,
                speed,
                cancel_check,
            )

        self._check(cancel_check)
        soundfile.write(output_path, audio, INTERNAL_SAMPLE_RATE, format="FLAC")
        self._check(cancel_check)

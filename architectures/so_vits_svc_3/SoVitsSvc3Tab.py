import json
import os

from dash import html, dcc, Input, Output, callback

import hay_say_common as hsc
from architectures.AbstractTab import AbstractTab


class SoVitsSvc3Tab(AbstractTab):
    @property
    def id(self):
        return 'so_vits_svc_3'

    @property
    def port(self):
        return 6575

    @property
    def label(self):
        return 'so-vits-svc 3.0'

    @property
    def description(self):
        return [html.P('so-vits-svc achieves a voice conversion effect by extracting "soft speech" features from '
                       'reference audio and passing them to a variational autoencoder.'),
                html.P(
                    html.A('https://github.com/svc-develop-team/so-vits-svc',
                           href='https://github.com/svc-develop-team/so-vits-svc')
                ),
                html.P('Thank you to Vul Traz and various unknown/anonymous users for providing the character models')]

    @property
    def requirements(self):
        return html.P(
            html.Em('This architecture requires a voice recording input. Text inputs are ignored.')
        )

    def meets_requirements(self, user_text, user_audio, selected_character):
        return user_audio is not None and selected_character is not None

    @property
    def options(self):
        return html.Table([
            html.Tr([
                html.Td("Note: \"TFH\" = Them's Fightin' Herds characters", colSpan=2, className='centered')
            ]),
            html.Tr([
                html.Td(html.Label('Character', htmlFor=self.input_ids[0]), className='option-label'),
                html.Td(self.character_dropdown)
            ]),
            html.Tr([
                html.Td(html.Label('Shift Pitch (semitones)', htmlFor=self.input_ids[1]), className='option-label'),
                html.Td(dcc.Input(id=self.input_ids[1], type='number', min=-36, max=36, step=1, value=0))
            ], title='Adjusts the pitch of the generated audio in semitones'),
            html.Tr([
                html.Td(html.Label('Speaker', htmlFor=self.input_ids[2]), className='option-label'),
                html.Td(dcc.Dropdown(id=self.input_ids[2], clearable=False))
            ], title='Selects a voice from a multi-speaker checkpoint.'),
            html.Tr([
                html.Td(html.Label('Silence slice threshold (dB)', htmlFor=self.input_ids[3]),
                        className='option-label'),
                html.Td(dcc.Input(id=self.input_ids[3], type='number', min=-100, max=0, step=1, value=-40))
            ], title='Audio below this level is treated as silence when splitting long inputs.'),
        ], className='spaced-table')

    @property
    def input_ids(self):
        return [self.id+'-character', self.id+'-semitone-pitch', self.id+'-speaker', self.id+'-slice-threshold']

    def available_speakers(self, character):
        if not character:
            return [], None
        character_dir = hsc.character_dir(self.id, character)
        try:
            with open(os.path.join(character_dir, 'config.json'), encoding='utf-8') as config_file:
                speakers = sorted(json.load(config_file)['spk'])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return [], None
        selected = speakers[0] if speakers else None
        speaker_file = os.path.join(character_dir, 'speaker.json')
        if os.path.isfile(speaker_file):
            try:
                with open(speaker_file, encoding='utf-8') as source:
                    configured = json.load(source).get('speaker')
                if configured in speakers:
                    selected = configured
            except (OSError, AttributeError, json.JSONDecodeError):
                pass
        return speakers, selected

    def register_callbacks(self, enable_model_management):
        super().register_callbacks(enable_model_management)

        @callback(
            Output(self.input_ids[2], 'options'),
            Output(self.input_ids[2], 'value'),
            Input(self.input_ids[0], 'value'),
        )
        def update_speakers(character):
            return self.available_speakers(character)

    @property
    def pitch_batch_key(self):
        return 'Pitch Shift'

    @property
    def supports_native_pitch_batch(self):
        return True

    def construct_input_dict(self, session_data, *args):
        return {
            'Architecture': self.id,
            'Character': args[0],
            'Pitch Shift': args[1],
            'Speaker': args[2],
            'Slice Threshold': args[3],
        }

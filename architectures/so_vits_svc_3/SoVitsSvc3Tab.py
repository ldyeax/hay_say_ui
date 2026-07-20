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

    @property
    def supports_parallel_requests(self):
        return True

    @property
    def supports_mixed_device_pitch_batch(self):
        return True

    def mixed_device_caches_are_warm(self, runtime_state, options, cpu_device, gpu_device):
        if not isinstance(runtime_state, dict) or cpu_device != '':
            return False
        character = options.get('Character') if isinstance(options, dict) else None
        if not isinstance(character, str) or not character:
            return False
        character_dir = os.path.realpath(hsc.character_dir(self.id, character))
        config_path = os.path.realpath(os.path.join(character_dir, 'config.json'))
        try:
            model_paths = sorted(
                os.path.realpath(os.path.join(character_dir, name))
                for name in os.listdir(character_dir)
                if name.startswith('G_') and name.endswith('.pth')
                and os.path.isfile(os.path.join(character_dir, name))
            )
        except OSError:
            return False
        if len(model_paths) != 1 or not os.path.isfile(config_path):
            return False
        try:
            model_revision = self._file_revision(model_paths[0])
            config_revision = self._file_revision(config_path)
        except OSError:
            return False

        details = runtime_state.get('loaded_model_details')
        if not isinstance(details, list):
            return False
        required_devices = {'cpu', f'cuda:{int(gpu_device)}'}
        matched_devices = set()
        for entry in details:
            if not isinstance(entry, dict) or entry.get('character') != character:
                continue
            model_path = entry.get('model_path')
            loaded_config = entry.get('config_path')
            device = entry.get('device')
            if not all(isinstance(value, str) for value in (model_path, loaded_config, device)):
                continue
            if (
                os.path.realpath(model_path) == model_paths[0]
                and os.path.realpath(loaded_config) == config_path
                and entry.get('model_revision') == model_revision
                and entry.get('config_revision') == config_revision
                and device in required_devices
            ):
                matched_devices.add(device)
        return matched_devices == required_devices

    @staticmethod
    def _file_revision(path):
        stat = os.stat(path)
        return {
            'device': int(stat.st_dev),
            'inode': int(stat.st_ino),
            'size': int(stat.st_size),
            'modified_ns': int(stat.st_mtime_ns),
        }

    @property
    def serializes_device_requests(self):
        return True

    def construct_input_dict(self, session_data, *args):
        return {
            'Architecture': self.id,
            'Character': args[0],
            'Pitch Shift': args[1],
            'Speaker': args[2],
            'Slice Threshold': args[3],
        }

import io
import json
import zipfile

import numpy
import pytest
from plotly.utils import PlotlyJSONEncoder

import generator
import output_download
from hay_say_common.cache import Stage
from postprocessed_display import prepare_postprocessed_display


DISPLAY_SESSION = '0123456789abcdef0123456789abcdef'
DISPLAY_OUTPUT = '0123456789abcdef0123'
DISPLAY_ANCHOR = 'fedcba9876543210fedc'


def _metadata(pitch, batch_id='batch-1234567890'):
    return {
        'Inputs': {'User File': 'Lead Vocal.wav', 'User Text': None},
        'Processing Options': {
            'Architecture': 'so_vits_svc_3',
            'Character': 'Fluttershy (speaking)',
            'Pitch Shift': pitch,
        },
        'Postprocessing Options': {'Adjust Output Speed': 1.0},
        'Batch ID': batch_id,
        'Pitch Variant': pitch,
    }


def test_descriptive_audio_filename_includes_character_architecture_source_and_pitch():
    filename = output_download.descriptive_audio_filename(_metadata(2), 'output-hash', 'FLAC')

    assert filename == 'Fluttershy-speaking_so-vits-svc-3_Lead-Vocal_pitch_+2.flac'


def test_descriptive_audio_filename_reads_pitch_from_legacy_processing_options():
    metadata = _metadata(-5)
    metadata.pop('Pitch Variant')

    assert output_download.descriptive_audio_filename(metadata, 'output-hash', 'wav').endswith(
        '_pitch_-5.wav'
    )


def test_descriptive_audio_filename_keeps_pitch_when_other_fields_are_long():
    metadata = _metadata(12)
    metadata['Processing Options']['Character'] = 'Fluttershy' * 40

    filename = output_download.descriptive_audio_filename(metadata, 'output-hash', 'flac')

    assert filename.endswith('_pitch_+12.flac')
    assert len(filename) <= 185


def test_create_pitch_batch_archive_contains_every_pitch_with_descriptive_names():
    class Cache:
        metadata = {
            'high': _metadata(2),
            'low': _metadata(-2),
            'middle': _metadata(0),
            'other-settings': {
                **_metadata(0),
                'Postprocessing Options': {'Adjust Output Speed': 1.25},
            },
        }

        @classmethod
        def read_metadata(cls, stage, session_id):
            assert (stage, session_id) == (Stage.POSTPROCESSED, 'session')
            return cls.metadata

        @staticmethod
        def read_audio_from_cache(stage, session_id, output_id):
            assert (stage, session_id) == (Stage.POSTPROCESSED, 'session')
            return numpy.full(32, {'low': -0.25, 'middle': 0.0, 'high': 0.25}[output_id]), 16000

    archive, filename = output_download.create_pitch_batch_archive(Cache, 'session', 'middle', 'wav')

    assert filename == 'Fluttershy-speaking_so-vits-svc-3_pitch-batch_batch-12.zip'
    with zipfile.ZipFile(io.BytesIO(archive)) as contents:
        assert contents.namelist() == [
            'Fluttershy-speaking_so-vits-svc-3_Lead-Vocal_pitch_-2.wav',
            'Fluttershy-speaking_so-vits-svc-3_Lead-Vocal_pitch_0.wav',
            'Fluttershy-speaking_so-vits-svc-3_Lead-Vocal_pitch_+2.wav',
        ]
        assert all(contents.read(name).startswith(b'RIFF') for name in contents.namelist())


def test_postprocessed_metadata_preserves_pitch_batch_lineage():
    class Cache:
        metadata = {
            (Stage.OUTPUT, 'session'): {
                'output': {
                    'Inputs': {'Preprocessed File': None, 'User Text': 'hello'},
                    'Options': {
                        'Architecture': 'rvc',
                        'Character': 'Fluttershy',
                        'Pitch Shift': 4,
                    },
                    'Batch ID': 'batch-id',
                    'Pitch Batches': {
                        'batch-id': {'Output IDs': ['output', 'other'], 'Pitches': [4, 6]},
                    },
                    'Pitch Variant': 4,
                    'Model Identity': 'model-fingerprint',
                },
            },
        }

        @classmethod
        def read_metadata(cls, stage, session_id):
            return cls.metadata.get((stage, session_id), {})

        @classmethod
        def update_metadata(cls, stage, session_id, updater):
            metadata = cls.metadata.setdefault((stage, session_id), {})
            updater(metadata)

    generator.write_postprocessed_metadata(Cache, 'output', 'postprocessed', False, False, 1.0, {'id': 'session'})
    metadata = Cache.metadata[(Stage.POSTPROCESSED, 'session')]['postprocessed']

    assert metadata['Batch ID'] == 'batch-id'
    assert metadata['Pitch Batches']['batch-id']['Output IDs'] == ['output', 'other']
    assert metadata['Pitch Variant'] == 4
    assert metadata['Output ID'] == 'output'
    assert metadata['Model Identity'] == 'model-fingerprint'


def test_cached_postprocessed_audio_refreshes_pitch_batch_lineage(monkeypatch):
    class Cache:
        metadata = {
            (Stage.OUTPUT, 'session'): {
                'output': {
                    'Inputs': {'Preprocessed File': None, 'User Text': None},
                    'Options': {'Character': 'Fluttershy', 'Pitch Shift': -3},
                    'Batch ID': 'new-batch',
                    'Pitch Variant': -3,
                    'Model Identity': 'weights',
                },
            },
            (Stage.POSTPROCESSED, 'session'): {'postprocessed': {'Batch ID': None}},
        }

        @staticmethod
        def file_is_already_cached(stage, session_id, output_id):
            return True

        @classmethod
        def read_metadata(cls, stage, session_id):
            return cls.metadata.get((stage, session_id), {})

        @classmethod
        def update_metadata(cls, stage, session_id, updater):
            metadata = cls.metadata.setdefault((stage, session_id), {})
            updater(metadata)

        @staticmethod
        def read_audio_from_cache(*_args):
            raise AssertionError('cached postprocessed audio should not be re-encoded')

    monkeypatch.setattr(generator.pcc, 'compute_next_hash', lambda *_args: 'postprocessed')

    result = generator.postprocess(Cache, 'output', False, False, 1.0, {'id': 'session'})

    assert result == 'postprocessed'
    assert Cache.metadata[(Stage.POSTPROCESSED, 'session')]['postprocessed']['Batch ID'] == 'new-batch'
    assert Cache.metadata[(Stage.POSTPROCESSED, 'session')]['postprocessed']['Pitch Variant'] == -3


def test_postprocessed_display_offers_zip_download_for_pitch_batch():
    metadata = {
        **_metadata(2),
        'Preprocessing Options': None,
        'Postprocessing Options': {
            'Reduce Metallic Noise': False,
            'Auto Tune Output': False,
            'Adjust Output Speed': 1.0,
        },
        'Time of Creation': '2026/07/15 12:00:00.000000',
    }

    class Cache:
        @staticmethod
        def read_file_bytes(stage, session_id, output_id):
            return b'audio'

        @staticmethod
        def read_metadata(stage, session_id):
            assert (stage, session_id) == (Stage.POSTPROCESSED, DISPLAY_SESSION)
            return {DISPLAY_OUTPUT: metadata}

    display = prepare_postprocessed_display(Cache, DISPLAY_OUTPUT, {'id': DISPLAY_SESSION})

    ids = _component_ids(display)
    assert {'type': 'batch-download-button', 'index': DISPLAY_OUTPUT} in ids
    assert {'type': 'batch-download', 'index': DISPLAY_OUTPUT} in ids


def test_postprocessed_display_uses_a_lazy_audio_url_instead_of_embedding_bytes():
    metadata = {
        **_metadata(2),
        'Preprocessing Options': None,
        'Postprocessing Options': {
            'Reduce Metallic Noise': False,
            'Auto Tune Output': False,
            'Adjust Output Speed': 1.0,
        },
        'Time of Creation': '2026/07/15 12:00:00.000000',
    }

    class Cache:
        @staticmethod
        def read_file_bytes(*_args):
            raise AssertionError('rendering output must not load or base64-encode audio bytes')

        @staticmethod
        def read_metadata(stage, session_id):
            assert (stage, session_id) == (Stage.POSTPROCESSED, DISPLAY_SESSION)
            return {DISPLAY_OUTPUT: metadata}

    display = prepare_postprocessed_display(Cache, DISPLAY_OUTPUT, {'id': DISPLAY_SESSION})
    audio = _components_by_type(display, 'Audio')

    assert len(audio) == 1
    properties = audio[0].to_plotly_json()['props']
    assert properties['src'].startswith('/')
    assert not properties['src'].startswith('data:')
    assert properties['preload'] == 'none'


def test_large_output_history_serializes_to_a_bounded_metadata_payload():
    metadata = {
        **_metadata(2),
        'Preprocessing Options': None,
        'Postprocessing Options': {
            'Reduce Metallic Noise': False,
            'Auto Tune Output': False,
            'Adjust Output Speed': 1.0,
        },
        'Time of Creation': '2026/07/15 12:00:00.000000',
    }
    output_ids = [f'{index:020x}' for index in range(100)]

    class Cache:
        @staticmethod
        def read_file_bytes(*_args):
            raise AssertionError('output history must never serialize cached audio bytes')

        @staticmethod
        def read_metadata(stage, session_id):
            return {output_id: metadata for output_id in output_ids}

    displays = [
        prepare_postprocessed_display(Cache, output_id, {'id': DISPLAY_SESSION})
        for output_id in output_ids
    ]
    payload = json.dumps(displays, cls=PlotlyJSONEncoder)

    assert len(payload.encode('utf-8')) < 500_000
    assert 'data:audio' not in payload
    assert payload.count('/cache-audio/postprocessed/') == len(output_ids)


def test_postprocessed_display_offers_batch_download_on_every_pitch_result():
    def metadata(output_id, pitch):
        return {
            **_metadata(pitch, 'batch-id'),
            'Output ID': output_id,
            'Pitch Batches': {
                'batch-id': {'Output IDs': ['a-output', 'b-output'], 'Pitches': [-2, 2]},
            },
            'Preprocessing Options': None,
            'Postprocessing Options': {
                'Reduce Metallic Noise': False,
                'Auto Tune Output': False,
                'Adjust Output Speed': 1.0,
            },
            'Time of Creation': '2026/07/15 12:00:00.000000',
        }

    metadata_by_cache_id = {
        DISPLAY_ANCHOR: metadata('a-output', -2),
        DISPLAY_OUTPUT: metadata('b-output', 2),
    }

    class Cache:
        @staticmethod
        def read_metadata(stage, session_id):
            assert (stage, session_id) == (Stage.POSTPROCESSED, DISPLAY_SESSION)
            return metadata_by_cache_id

    for cache_id in metadata_by_cache_id:
        display = prepare_postprocessed_display(Cache, cache_id, {'id': DISPLAY_SESSION})
        key = output_download.batch_download_key('batch-id', cache_id)
        ids = _component_ids(display)

        assert {'type': 'batch-download-button', 'index': key} in ids
        assert {'type': 'batch-download', 'index': key} in ids


def test_postprocessed_display_offers_each_overlapping_batch_once():
    metadata = {
        **_metadata(0, 'new-batch'),
        'Output ID': 'shared-output',
        'Pitch Batches': {
            'old-batch': {
                'Output IDs': ['old-output', 'shared-output'],
                'Pitches': [-2, 0],
            },
            'new-batch': {
                'Output IDs': ['shared-output', 'new-output'],
                'Pitches': [0, 2],
            },
        },
        'Preprocessing Options': None,
        'Postprocessing Options': {
            'Reduce Metallic Noise': False,
            'Auto Tune Output': False,
            'Adjust Output Speed': 1.0,
        },
        'Time of Creation': '2026/07/15 12:00:00.000000',
    }

    class Cache:
        @staticmethod
        def read_file_bytes(stage, session_id, output_id):
            return b'audio'

        @staticmethod
        def read_metadata(stage, session_id):
            assert (stage, session_id) == (Stage.POSTPROCESSED, DISPLAY_SESSION)
            return {DISPLAY_ANCHOR: metadata}

    display = prepare_postprocessed_display(Cache, DISPLAY_ANCHOR, {'id': DISPLAY_SESSION})
    ids = _component_ids(display)

    for batch_id in ('old-batch', 'new-batch'):
        key = output_download.batch_download_key(batch_id, DISPLAY_ANCHOR)
        assert ids.count({'type': 'batch-download-button', 'index': key}) == 1
        assert ids.count({'type': 'batch-download', 'index': key}) == 1


def test_pitch_batch_control_explicitly_says_it_downloads_every_pitch():
    metadata = {
        **_metadata(0, 'batch-id'),
        'Output ID': 'middle-output',
        'Pitch Batches': {
            'batch-id': {
                'Output IDs': ['low-output', 'middle-output', 'high-output'],
                'Pitches': [-2, 0, 2],
            },
        },
    }

    controls = output_download.pitch_batch_controls(metadata, 'middle-cache-id')

    assert controls == [(
        output_download.batch_download_key('batch-id', 'middle-cache-id'),
        'Download all 3 pitches (-2 to +2) (.zip)',
    )]


def test_overlapping_pitch_batches_keep_complete_historical_archives():
    first_id = 'batch-first'
    second_id = 'batch-second'
    first_manifest = {
        'Output IDs': ['out-low', 'out-middle', 'out-high'],
        'Pitches': [-2, 0, 2],
    }
    second_manifest = {
        'Output IDs': ['out-lower', 'out-middle', 'out-higher'],
        'Pitches': [-4, 0, 4],
    }

    def entry(output_id, pitch, current_batch, memberships):
        metadata = _metadata(pitch, current_batch)
        metadata['Output ID'] = output_id
        metadata['Pitch Batches'] = memberships
        return metadata

    class Cache:
        metadata = {
            'low': entry('out-low', -2, first_id, {first_id: first_manifest}),
            'middle': entry(
                'out-middle',
                0,
                second_id,
                {first_id: first_manifest, second_id: second_manifest},
            ),
            'high': entry('out-high', 2, first_id, {first_id: first_manifest}),
            'lower': entry('out-lower', -4, second_id, {second_id: second_manifest}),
            'higher': entry('out-higher', 4, second_id, {second_id: second_manifest}),
        }

        @classmethod
        def read_metadata(cls, stage, session_id):
            return cls.metadata

        @staticmethod
        def read_audio_from_cache(stage, session_id, output_id):
            return numpy.full(32, 0.1), 16000

    selection = output_download.batch_download_key(first_id, 'low')
    archive, _ = output_download.create_pitch_batch_archive(Cache, 'session', selection, 'wav')

    with zipfile.ZipFile(io.BytesIO(archive)) as contents:
        assert [output_download.pitch_variant(Cache.metadata[name]) for name in ('low', 'middle', 'high')] == [-2, 0, 2]
        assert len(contents.namelist()) == 3
        assert any('pitch_0.wav' in name for name in contents.namelist())


def test_pitch_batch_archive_rejects_excessive_encoded_audio(monkeypatch):
    class Cache:
        metadata = {'low': _metadata(-1), 'high': _metadata(1)}

        @classmethod
        def read_metadata(cls, stage, session_id):
            return cls.metadata

        @staticmethod
        def read_audio_from_cache(stage, session_id, output_id):
            return numpy.ones(1024, dtype=numpy.float32), 16000

    monkeypatch.setenv('HAY_SAY_MAX_BATCH_DOWNLOAD_BYTES', '100')

    with pytest.raises(ValueError, match='too large'):
        output_download.create_pitch_batch_archive(Cache, 'session', 'low', 'wav')


def _component_ids(component):
    if component is None:
        return []
    if isinstance(component, (list, tuple)):
        return [component_id for child in component for component_id in _component_ids(child)]
    if not hasattr(component, 'to_plotly_json'):
        return []
    properties = component.to_plotly_json()['props']
    component_ids = [properties['id']] if properties.get('id') is not None else []
    return component_ids + _component_ids(properties.get('children'))


def _components_by_type(component, component_type):
    if component is None:
        return []
    if isinstance(component, (list, tuple)):
        return [
            match
            for child in component
            for match in _components_by_type(child, component_type)
        ]
    if not hasattr(component, 'to_plotly_json'):
        return []
    serialized = component.to_plotly_json()
    matches = [component] if serialized.get('type') == component_type else []
    return matches + _components_by_type(serialized['props'].get('children'), component_type)

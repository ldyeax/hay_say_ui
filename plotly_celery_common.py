import base64
import datetime
import hashlib
import json

from hay_say_common.cache import Stage

import hay_say_common as hsc
from architectures.controllable_talknet.ControllableTalknetTab import ControllableTalknetTab
from architectures.rvc.RvcTab import RvcTab
from architectures.so_vits_svc_3.SoVitsSvc3Tab import SoVitsSvc3Tab
from architectures.so_vits_svc_4.SoVitsSvc4Tab import SoVitsSvc4Tab
from architectures.so_vits_svc_5.SoVitsSvc5Tab import SoVitsSvc5Tab
from architectures.styletts_2.StyleTTS2Tab import StyleTTS2Tab
from architectures.gpt_so_vits.GPTSoVITSTab import GPTSoVITSTab

_memoized_architecture_map = None

_architecture_classes = {
    'ControllableTalkNet': ControllableTalknetTab,
    'SoVitsSvc3': SoVitsSvc3Tab,
    'SoVitsSvc4': SoVitsSvc4Tab,
    'SoVitsSvc5': SoVitsSvc5Tab,
    'Rvc': RvcTab,
    'StyleTTS2': StyleTTS2Tab,
    'GPTSoVITS': GPTSoVITSTab,
}


def architecture_map(cache=None, choices=None):
    global _memoized_architecture_map
    if _memoized_architecture_map is None:
        _memoized_architecture_map = dict()
    if choices:
        if cache is None:
            raise ValueError('A cache is required when initializing architecture tabs')
        for choice in choices:
            if choice not in _architecture_classes:
                raise ValueError(f'Unknown architecture: {choice}')
            if choice not in _memoized_architecture_map:
                _memoized_architecture_map[choice] = _architecture_classes[choice](cache)
            else:
                _memoized_architecture_map[choice].cache = cache
    elif cache is None and not _memoized_architecture_map:
        raise ValueError('The architecture map has not been initialized')
    return _memoized_architecture_map


def construct_architecture_tabs(choices, cache_type):
    cache = hsc.select_cache_implementation(cache_type)
    return [architecture_map(cache, choices)[choice] for choice in choices]


def select_architecture_tabs(choices):
    return [architecture_map(None)[choice] for choice in choices]


def convert_to_bools(*args):
    # intended for converting a list of checklist items from ['']/None to True/False, respectively
    return [bool(arg) for arg in args]


def compute_next_hash(current_hash, *args):
    payload = json.dumps([current_hash, *args], sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]


def prepare_src_attribute(src_bytes, mimetype):
    binary_contents = b'data:' + mimetype.encode('utf-8') + b',' + base64.b64encode(src_bytes)
    return binary_contents.decode('utf-8')


def lookup_filehash(cache, selected_file, session_data):
    raw_metadata = cache.read_metadata(Stage.RAW, session_data['id'])
    reverse_lookup = {raw_metadata[key]['User File']: key for key in raw_metadata}
    return reverse_lookup.get(selected_file)


def preprocess(cache, filename, semitone_pitch, debug_pitch, reduce_noise, crop_silence, session_data):
    # Get hashes and determine file locations. Delegate actual preprocessing work to preprocess_file
    # filename must not be None.

    debug_pitch, reduce_noise, crop_silence = convert_to_bools(debug_pitch, reduce_noise, crop_silence)

    hash_raw = lookup_filehash(cache, filename, session_data)
    hash_preprocessed = compute_next_hash(hash_raw, semitone_pitch, debug_pitch, reduce_noise, crop_silence)
    preprocess_file(cache, hash_raw, hash_preprocessed, semitone_pitch, debug_pitch, reduce_noise,
                    crop_silence, session_data)
    return hash_preprocessed


def preprocess_file(cache, hash_raw, hash_preprocessed, semitone_pitch, debug_pitch, reduce_noise, crop_silence,
                    session_data):
    # Handle file operations and write to metadata file. Delegate actual preprocessing work to preprocess_bytes

    if cache.file_is_already_cached(Stage.PREPROCESSED, session_data['id'], hash_preprocessed):
        return

    data_raw, sr_raw = cache.read_audio_from_cache(Stage.RAW, session_data['id'], hash_raw)
    data_preprocessed, sr_preprocessed = preprocess_bytes(data_raw, sr_raw, semitone_pitch, debug_pitch,
                                                          reduce_noise, crop_silence)
    cache.save_audio_to_cache(Stage.PREPROCESSED, session_data['id'], hash_preprocessed, data_preprocessed,
                              sr_preprocessed)
    write_preprocessed_metadata(cache, hash_raw, hash_preprocessed, semitone_pitch, debug_pitch, reduce_noise,
                                crop_silence, session_data)


def preprocess_bytes(bytes_raw, sr_raw, semitone_pitch, debug_pitch, reduce_noise, crop_silence):
    # todo: implement this
    return bytes_raw, sr_raw


def write_preprocessed_metadata(cache, hash_raw, hash_preprocessed, semitone_pitch, debug_pitch,
                                reduce_noise, crop_silence, session_data):
    entry = {
        'Raw File': hash_raw,
        'Options':
            {
                'Semitone Pitch': semitone_pitch,
                'Debug Pitch': debug_pitch,
                'Reduce Noise': reduce_noise,
                'Crop Silence': crop_silence
            },
        'Time of Creation': datetime.datetime.now().strftime(hsc.cache.TIMESTAMP_FORMAT)
    }
    cache.update_metadata(
        Stage.PREPROCESSED,
        session_data['id'],
        lambda metadata: metadata.update({hash_preprocessed: entry}),
    )

import base64
import datetime
import hashlib
import json
import os
import traceback
import uuid
from http.client import HTTPConnection
from pathlib import Path

from hay_say_common.cache import Stage

import hay_say_common as hsc
import plotly_celery_common as pcc
from pitch_batch import parse_pitch_spec
from postprocessed_display import prepare_postprocessed_display
import runtime_client


# todo: That's a lot of inputs, and most of them get passed down to the generate() method. Is there a cleaner way to
#  pass all these arguments?
def generate_and_prepare_postprocessed_display(clicks, set_progress, message, cache_type, gpu_id, session_data,
                                               selected_architectures, user_text, selected_file, semitone_pitch,
                                               debug_pitch, reduce_noise, crop_silence, reduce_metallic_noise,
                                               auto_tune_output, output_speed_adjustment, pitch_batch_enabled,
                                               pitch_batch_values, args):
    if clicks is not None:
        highlight_first = True
        try:
            set_progress(message)
            generate(cache_type, gpu_id, session_data, selected_architectures, user_text,
                     selected_file, semitone_pitch, debug_pitch, reduce_noise, crop_silence,
                     reduce_metallic_noise, auto_tune_output, output_speed_adjustment,
                     pitch_batch_enabled, pitch_batch_values, args, set_progress)
        except Exception:
            return 'An error has occurred. Please send the software maintainers the following information as ' \
                   'well as any recent output in the Command Prompt/terminal (please review and remove any ' \
                   'private info before sending!): \n\n' + \
                   traceback.format_exc(), 'Generate!'
    else:
        highlight_first = False
    cache = hsc.select_cache_implementation(cache_type)
    sorted_hashes = cache.get_hashes_sorted_by_timestamp(Stage.POSTPROCESSED, session_data['id'])
    first_output = [
        prepare_postprocessed_display(cache, sorted_hashes[0], session_data,
                                      highlight=highlight_first)] if sorted_hashes else []
    remaining_outputs = [prepare_postprocessed_display(cache, hash_postprocessed, session_data)
                         for hash_postprocessed in reversed(sorted_hashes[1:])]
    return remaining_outputs + first_output, 'Generate!'


def generate(cache_type, gpu_id, session_data, selected_architectures, user_text, selected_file, semitone_pitch,
             debug_pitch, reduce_noise, crop_silence, reduce_metallic_noise, auto_tune_output, output_speed_adjustment,
             pitch_batch_enabled, pitch_batch_values, args, set_progress=None):
    print('generating on ' + ('CPU' if gpu_id == '' else ('GPU #' + str(gpu_id))), flush=True)
    cache = hsc.select_cache_implementation(cache_type)
    selected_tab_object = get_selected_tab_object(selected_architectures, args[0:len(selected_architectures)])
    relevant_inputs = get_inputs_for_selected_tab(selected_architectures, selected_tab_object,
                                                  args[len(selected_architectures):])
    hash_preprocessed = preprocess_if_needed(cache, selected_file, semitone_pitch, debug_pitch, reduce_noise,
                                             crop_silence, session_data)
    hash_outputs = process_batch(
        cache,
        user_text,
        hash_preprocessed,
        selected_tab_object,
        relevant_inputs,
        session_data,
        gpu_id,
        pitch_batch_enabled,
        pitch_batch_values,
        set_progress,
    )
    for hash_output in hash_outputs:
        postprocess(cache, hash_output, reduce_metallic_noise, auto_tune_output,
                    output_speed_adjustment, session_data)


def get_selected_tab_object(selected_architectures, hidden_states):
    if len(hidden_states) != len(selected_architectures):
        raise ValueError('The web server and generation worker registered different architecture lists')
    visible_tabs = [tab for hidden, tab in zip(hidden_states, selected_architectures) if hidden is False]
    if len(visible_tabs) != 1:
        raise ValueError(f'Expected one selected architecture, found {len(visible_tabs)}')
    return visible_tabs[0]


def get_inputs_for_selected_tab(selected_architectures, tab_object, args):
    all_inputs = [item for sublist in [tab.input_ids for tab in selected_architectures] for item in sublist]
    if len(args) != len(all_inputs):
        raise ValueError('The web server and generation worker registered different architecture inputs')
    indices_of_relevant_inputs = [index for index, item in enumerate(all_inputs) if
                                  item in tab_object.input_ids]
    return [args[i] for i in indices_of_relevant_inputs]


def preprocess_if_needed(cache, selected_file, semitone_pitch, debug_pitch, reduce_noise, crop_silence, session_data):
    if selected_file is None:
        hash_preprocessed = None
    else:
        hash_preprocessed = pcc.preprocess(cache, selected_file, semitone_pitch, debug_pitch, reduce_noise,
                                       crop_silence, session_data)
    return hash_preprocessed


def process_batch(cache, user_text, hash_preprocessed, tab_object, relevant_inputs, session_data, gpu_id,
                  pitch_batch_enabled=False, pitch_batch_values=None, set_progress=None):
    options = tab_object.construct_input_dict(session_data, *relevant_inputs)
    model_identity = _model_identity(tab_object.id, options.get('Character'))
    generation_nonce = None if tab_object.cache_generated_output else uuid.uuid4().hex
    pitch_key = tab_object.pitch_batch_key
    if pitch_batch_enabled:
        if pitch_key is None:
            raise ValueError(f'{tab_object.label} does not expose a pitch parameter')
        minimum, maximum = tab_object.pitch_batch_bounds
        pitches = parse_pitch_spec(pitch_batch_values, minimum, maximum)
    else:
        pitches = [options.get(pitch_key)] if pitch_key else [None]

    variants = []
    for pitch in pitches:
        variant_options = dict(options)
        if pitch_key is not None:
            variant_options[pitch_key] = int(pitch)
        output_hash = pcc.compute_next_hash(
            hash_preprocessed,
            user_text,
            tab_object.id,
            variant_options,
            model_identity,
            generation_nonce,
        )
        variants.append((pitch, variant_options, output_hash))

    batch_id = pcc.compute_next_hash(*(variant[2] for variant in variants))
    uncached = _uncached_variants(cache, variants, session_data)

    if uncached:
        runtime_client.ensure_runtime_started(tab_object.id, tab_object.port)
    host, port = resolve_service_endpoint(tab_object)
    with runtime_client.generation_lock(tab_object.id):
        uncached = _uncached_variants(cache, variants, session_data)
        if len(uncached) > 1 and tab_object.supports_native_pitch_batch:
            batch_options = dict(uncached[0][1])
            batch_options['Pitch Shifts'] = [variant[0] for variant in uncached]
            payload = construct_payload(
                user_text,
                hash_preprocessed,
                batch_options,
                uncached[0][2],
                session_data,
                gpu_id,
            )
            payload['Output Files'] = [variant[2] for variant in uncached]
            if set_progress:
                set_progress(f'Generating {len(uncached)} pitch variants with {tab_object.label}...')
            send_payload(payload, host, port)
        else:
            for index, (_, variant_options, output_hash) in enumerate(uncached, start=1):
                if set_progress:
                    set_progress(f'Generating variant {index} of {len(uncached)} with {tab_object.label}...')
                payload = construct_payload(
                    user_text,
                    hash_preprocessed,
                    variant_options,
                    output_hash,
                    session_data,
                    gpu_id,
                )
                send_payload(payload, host, port)

        for pitch, variant_options, output_hash in variants:
            verify_output_exists(cache, output_hash, session_data)
            write_output_metadata(
                cache,
                hash_preprocessed,
                user_text,
                output_hash,
                variant_options,
                session_data,
                batch_id=batch_id if len(variants) > 1 else None,
                pitch=pitch,
                model_identity=model_identity,
            )
    return [variant[2] for variant in variants]


def _uncached_variants(cache, variants, session_data):
    return [variant for variant in variants
            if not cache.file_is_already_cached(Stage.OUTPUT, session_data['id'], variant[2])]


def _model_identity(runtime_id, character):
    """Fingerprint the selected weights without reading multi-gigabyte checkpoints."""
    if not character:
        return 'no-model'
    character_root = Path(hsc.character_dir(runtime_id, character))
    architecture_root = character_root.parent.parent
    candidates = list(character_root.rglob('*')) if character_root.is_dir() else []
    if architecture_root.is_dir():
        candidates.extend(path for path in architecture_root.iterdir() if path.is_file())
    digest = hashlib.sha256()
    for path in sorted({path for path in candidates if path.is_file()}, key=lambda item: str(item)):
        stat = path.stat()
        try:
            relative = path.relative_to(architecture_root)
        except ValueError:
            relative = path
        digest.update(str(relative).encode('utf-8', errors='surrogateescape'))
        digest.update(f'\0{stat.st_size}\0{stat.st_mtime_ns}\0'.encode('ascii'))
    return digest.hexdigest()


def process(cache, user_text, hash_preprocessed, tab_object, relevant_inputs, session_data, gpu_id):
    """Compatibility wrapper for a single deterministic generation."""
    return process_batch(
        cache,
        user_text,
        hash_preprocessed,
        tab_object,
        relevant_inputs,
        session_data,
        gpu_id,
    )[0]


def construct_payload(user_text, hash_preprocessed, options, hash_output, session_data, gpu_id):
    return {
        'Inputs': {
            'User Text': user_text,
            'User Audio': hash_preprocessed
        },
        'Options': options,
        'Output File': hash_output,
        'GPU ID': gpu_id,
        'Session ID': session_data['id']
    }


def resolve_service_endpoint(tab_object):
    return runtime_client.service_endpoint(tab_object.id, tab_object.port)


def send_payload(payload, host, port, timeout=None):
    timeout = timeout or float(os.environ.get('HAY_SAY_MODEL_REQUEST_TIMEOUT', '900'))
    connection = HTTPConnection(host, port, timeout=timeout)
    headers = {'Content-type': 'application/json'}
    connection.request('POST', '/generate', json.dumps(payload), headers)
    try:
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(extract_message(response))
        response.read()
    finally:
        connection.close()


def extract_message(response):
    body = response.read().decode('utf-8', errors='replace')
    try:
        json_response = json.loads(body)
    except json.JSONDecodeError:
        return body or f'Model service returned HTTP {response.status}'
    message = json_response.get('message')
    if message:
        try:
            return base64.b64decode(message).decode('utf-8')
        except (ValueError, UnicodeDecodeError):
            return str(message)
    return str(json_response.get('error') or json_response.get('detail') or json_response)


def verify_output_exists(cache, hash_output, session_data):
    try:
        cache.read_audio_from_cache(Stage.OUTPUT, session_data['id'], hash_output)
    except Exception as e:
        raise Exception("Payload was sent, but output file was not produced.") from e


def write_output_metadata(cache, hash_preprocessed, user_text, hash_output, options, session_data,
                          batch_id=None, pitch=None, model_identity=None):
    entry = {
        'Inputs': {
            'Preprocessed File': hash_preprocessed,
            'User Text': user_text
        },
        'Options': options,
        'Batch ID': batch_id,
        'Pitch Variant': pitch,
        'Model Identity': model_identity,
        'Time of Creation': datetime.datetime.now().strftime(hsc.cache.TIMESTAMP_FORMAT)
    }
    cache.update_metadata(
        Stage.OUTPUT,
        session_data['id'],
        lambda metadata: metadata.update({hash_output: entry}),
    )


def postprocess(cache, hash_output, reduce_metallic_noise, auto_tune_output, output_speed_adjustment, session_data):
    # Convert data types to something more digestible
    reduce_metallic_noise, auto_tune_output = pcc.convert_to_bools(reduce_metallic_noise, auto_tune_output)
    output_speed_adjustment = float(
        output_speed_adjustment)  # Dash's Range Input supplies a string, so cast to float

    # Check whether the postprocessed file already exists
    hash_postprocessed = pcc.compute_next_hash(hash_output, reduce_metallic_noise, auto_tune_output,
                                           output_speed_adjustment)
    if cache.file_is_already_cached(Stage.POSTPROCESSED, session_data['id'], hash_postprocessed):
        _refresh_metadata_timestamp(cache, Stage.POSTPROCESSED, session_data['id'], hash_postprocessed)
        return hash_postprocessed

    # Perform postprocessing
    data_output, sr_output = cache.read_audio_from_cache(Stage.OUTPUT, session_data['id'], hash_output)
    data_postprocessed, sr_postprocessed = postprocess_bytes(data_output, sr_output, reduce_metallic_noise,
                                                             auto_tune_output, output_speed_adjustment)

    # write the postprocessed data to file
    cache.save_audio_to_cache(Stage.POSTPROCESSED, session_data['id'], hash_postprocessed, data_postprocessed,
                              sr_postprocessed)

    # write metadata file
    write_postprocessed_metadata(cache, hash_output, hash_postprocessed, reduce_metallic_noise, auto_tune_output,
                                 output_speed_adjustment, session_data)

    return hash_postprocessed


def _refresh_metadata_timestamp(cache, stage, session_id, cache_key):
    now = datetime.datetime.now().strftime(hsc.cache.TIMESTAMP_FORMAT)

    def refresh(metadata):
        if cache_key in metadata:
            metadata[cache_key]['Time of Creation'] = now

    cache.update_metadata(stage, session_id, refresh)


def postprocess_bytes(bytes_output, sr_output, reduce_metallic_noise, auto_tune_output, output_speed_adjustment):
    # todo: implement this
    return bytes_output, sr_output


def write_postprocessed_metadata(cache, hash_output, hash_postprocessed, reduce_metallic_noise, auto_tune_output,
                                 output_speed_adjustment, session_data):
    processing_options, user_text, hash_preprocessed = get_process_info(cache, hash_output, session_data)
    selected_file, preprocess_options = get_preprocess_info(cache, hash_preprocessed, session_data)

    entry = {
        'Inputs': {
            'User File': selected_file,
            'User Text': user_text
        },
        'Preprocessing Options': preprocess_options,
        'Processing Options': processing_options,
        'Postprocessing Options': {
            'Reduce Metallic Noise': reduce_metallic_noise,
            'Auto Tune Output': auto_tune_output,
            'Adjust Output Speed': output_speed_adjustment
        },
        'Time of Creation': datetime.datetime.now().strftime(hsc.cache.TIMESTAMP_FORMAT)
    }
    cache.update_metadata(
        Stage.POSTPROCESSED,
        session_data['id'],
        lambda metadata: metadata.update({hash_postprocessed: entry}),
    )


def get_process_info(cache, hash_output, session_data):
    output_metadata = cache.read_metadata(Stage.OUTPUT, session_data['id'])
    processing_options = output_metadata.get(hash_output).get('Options')
    user_text = output_metadata.get(hash_output).get('Inputs').get('User Text')
    hash_preprocessed = output_metadata.get(hash_output).get('Inputs').get('Preprocessed File')
    return processing_options, user_text, hash_preprocessed


def get_preprocess_info(cache, hash_preprocessed, session_data):
    if hash_preprocessed is None:
        selected_file = None
        preprocess_options = None
    else:
        preprocess_metadata = cache.read_metadata(Stage.PREPROCESSED, session_data['id'])
        preprocess_options = preprocess_metadata.get(hash_preprocessed).get('Options')
        hash_raw = preprocess_metadata.get(hash_preprocessed).get('Raw File')

        raw_metadata = cache.read_metadata(Stage.RAW, session_data['id'])
        selected_file = raw_metadata.get(hash_raw).get('User File')
    return selected_file, preprocess_options

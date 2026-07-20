"""Names and archive creation for generated-audio downloads."""

import io
import json
import os
import re
import zipfile
from numbers import Number

import soundfile

from hay_say_common.cache import Stage


_UNSAFE_FILENAME_CHARACTERS = re.compile(r"[^A-Za-z0-9.+-]+")


def descriptive_audio_filename(metadata, output_id, file_format):
    """Build a stable, human-readable name for one generated audio file."""
    extension = _normalize_format(file_format).lower()
    processing = metadata.get('Processing Options') or {}
    inputs = metadata.get('Inputs') or {}
    parts = [
        _filename_part(processing.get('Character')),
        _filename_part(processing.get('Architecture')),
        _source_description(inputs),
    ]
    pitch = pitch_variant(metadata)
    pitch_part = None
    if pitch is not None:
        pitch_part = 'pitch_' + _format_pitch(pitch)
        parts.append(pitch_part)
    parts = [part for part in parts if part]
    if not parts:
        parts = ['generated-audio', _filename_part(str(output_id)[:8])]
    stem = '_'.join(parts)
    if len(stem) > 180:
        if pitch_part:
            prefix = '_'.join(parts[:-1])[:179 - len(pitch_part)].rstrip('._-')
            stem = prefix + '_' + pitch_part
        else:
            stem = stem[:180].rstrip('._-')
    return stem + '.' + extension


def pitch_variant(metadata):
    """Read pitch lineage, including metadata written before pitch batches existed."""
    pitch = metadata.get('Pitch Variant')
    if pitch is not None:
        return pitch
    for key, value in (metadata.get('Processing Options') or {}).items():
        if 'pitch' in key.lower() and isinstance(value, Number) and not isinstance(value, bool):
            return value
    return None


def batch_download_key(batch_id, anchor_id):
    return json.dumps([str(batch_id), str(anchor_id)], separators=(',', ':'))


def pitch_batch_controls(metadata, anchor_id):
    """Return stable download keys/labels for every batch containing this output."""
    memberships = metadata.get('Pitch Batches')
    output_id = metadata.get('Output ID')
    if not isinstance(memberships, dict) or not isinstance(output_id, str):
        batch_id = metadata.get('Batch ID')
        return [(str(anchor_id), 'Download pitch batch (.zip)')] if batch_id else []

    controls = []
    for batch_id, manifest in sorted(memberships.items()):
        output_ids = manifest.get('Output IDs') if isinstance(manifest, dict) else None
        if (
            not isinstance(output_ids, list)
            or len(output_ids) < 2
            or output_id not in map(str, output_ids)
        ):
            continue
        pitches = manifest.get('Pitches')
        label = _batch_label(pitches)
        controls.append((batch_download_key(batch_id, anchor_id), label))
    return controls


def batch_members(metadata_by_id, selection_id):
    """Return the postprocessed outputs belonging to the anchor's pitch batch."""
    batch_id, anchor_id = _parse_batch_selection(selection_id)
    anchor = metadata_by_id.get(anchor_id)
    if anchor is None:
        raise ValueError('The selected output is no longer available')
    batch_id = batch_id or anchor.get('Batch ID')
    if not batch_id:
        raise ValueError('The selected output is not part of a pitch batch')
    postprocessing = anchor.get('Postprocessing Options')
    manifest = (anchor.get('Pitch Batches') or {}).get(batch_id)
    output_ids = manifest.get('Output IDs') if isinstance(manifest, dict) else None
    expected_ids = {str(value) for value in output_ids} if isinstance(output_ids, list) else None
    members = [
        (output_id, entry)
        for output_id, entry in metadata_by_id.items()
        if (
            (expected_ids is not None and str(entry.get('Output ID')) in expected_ids)
            or (expected_ids is None and entry.get('Batch ID') == batch_id)
        )
        and entry.get('Postprocessing Options') == postprocessing
    ]
    members.sort(key=lambda item: (_pitch_sort_key(item[1]), item[0]))
    if expected_ids is not None and {str(entry.get('Output ID')) for _, entry in members} != expected_ids:
        raise ValueError('Some outputs from this pitch batch are no longer available')
    return members


def create_pitch_batch_archive(cache, session_id, selection_id, file_format):
    """Encode a complete pitch batch and return ``(zip_bytes, download_name)``."""
    metadata_by_id = cache.read_metadata(Stage.POSTPROCESSED, session_id)
    members = batch_members(metadata_by_id, selection_id)
    if len(members) < 2:
        raise ValueError('Fewer than two outputs from this pitch batch are available')

    archive_buffer = io.BytesIO()
    encoded_bytes = 0
    maximum_bytes = _positive_environment_int('HAY_SAY_MAX_BATCH_DOWNLOAD_BYTES', 256 * 1024 * 1024)
    used_names = set()
    with zipfile.ZipFile(archive_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
        for output_id, metadata in members:
            filename = _unique_filename(
                descriptive_audio_filename(metadata, output_id, file_format),
                used_names,
            )
            audio, sample_rate = cache.read_audio_from_cache(Stage.POSTPROCESSED, session_id, output_id)
            audio_buffer = io.BytesIO()
            soundfile.write(audio_buffer, audio, sample_rate, format=_normalize_format(file_format))
            encoded = audio_buffer.getvalue()
            encoded_bytes += len(encoded)
            if encoded_bytes > maximum_bytes:
                raise ValueError(
                    'This pitch batch is too large to download as one ZIP; '
                    'use a compressed format or individual downloads'
                )
            archive.writestr(filename, encoded)

    _, anchor_id = _parse_batch_selection(selection_id)
    anchor = metadata_by_id[anchor_id]
    processing = anchor.get('Processing Options') or {}
    archive_parts = [
        _filename_part(processing.get('Character')),
        _filename_part(processing.get('Architecture')),
        'pitch-batch',
        _filename_part(str(_parse_batch_selection(selection_id)[0] or anchor.get('Batch ID'))[:8]),
    ]
    archive_name = '_'.join(part for part in archive_parts if part) + '.zip'
    return archive_buffer.getvalue(), archive_name


def _source_description(inputs):
    user_file = inputs.get('User File')
    if user_file:
        return _filename_part(os.path.splitext(os.path.basename(str(user_file)))[0])
    user_text = inputs.get('User Text')
    if user_text:
        return _filename_part(str(user_text)[:48])
    return None


def _parse_batch_selection(selection_id):
    try:
        value = json.loads(selection_id)
    except (TypeError, json.JSONDecodeError):
        return None, selection_id
    if isinstance(value, list) and len(value) == 2 and all(isinstance(item, str) for item in value):
        return value[0], value[1]
    return None, selection_id


def _batch_label(pitches):
    numeric = sorted(float(value) for value in pitches) if (
        isinstance(pitches, list)
        and pitches
        and all(isinstance(value, Number) and not isinstance(value, bool) for value in pitches)
    ) else []
    if not numeric:
        return 'Download all pitch variants (.zip)'
    low = _format_pitch(numeric[0])
    high = _format_pitch(numeric[-1])
    return f'Download all {len(numeric)} pitches ({low} to {high}) (.zip)'


def _positive_environment_int(name, default):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _filename_part(value):
    if value is None:
        return None
    result = _UNSAFE_FILENAME_CHARACTERS.sub('-', str(value).strip()).strip('.-_')
    return result or None


def _format_pitch(value):
    if isinstance(value, Number) and float(value).is_integer():
        value = int(value)
    if isinstance(value, Number) and value > 0:
        return '+' + str(value)
    return str(value)


def _pitch_sort_key(metadata):
    pitch = pitch_variant(metadata)
    if isinstance(pitch, Number):
        return 0, float(pitch)
    return 1, str(pitch)


def _normalize_format(file_format):
    value = str(file_format or '').strip().lstrip('.').upper()
    if not value or not re.fullmatch(r'[A-Z0-9]+', value):
        raise ValueError('Select a valid audio file format')
    return value


def _unique_filename(filename, used_names):
    if filename not in used_names:
        used_names.add(filename)
        return filename
    stem, extension = os.path.splitext(filename)
    index = 2
    while f'{stem}_{index}{extension}' in used_names:
        index += 1
    filename = f'{stem}_{index}{extension}'
    used_names.add(filename)
    return filename

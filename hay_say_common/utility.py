"""General audio and filesystem helpers used across Hay Say services."""

import base64
import io
import os

import numpy
import soundfile


def create_link(existing_path, desired_link_path):
    """Create or repair a symlink without replacing an unrelated directory."""
    existing_path = os.path.abspath(existing_path)
    desired_link_path = os.path.abspath(desired_link_path)
    if not os.path.exists(existing_path):
        raise FileNotFoundError(existing_path)
    if os.path.islink(desired_link_path):
        if os.path.realpath(desired_link_path) == os.path.realpath(existing_path):
            return
        os.unlink(desired_link_path)
    elif os.path.lexists(desired_link_path):
        raise FileExistsError(f"Refusing to replace non-symlink path: {desired_link_path}")
    os.symlink(existing_path, desired_link_path)


def get_audio_from_src_attribute(src, encoding):
    _, raw = src.split(",", 1)
    buffer = io.BytesIO(base64.b64decode(raw.encode(encoding)))
    return _read_audio_source(buffer)


def read_audio(path):
    return _read_audio_source(path)


def _read_audio_source(source):
    audio, sample_rate = soundfile.read(source, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = numpy.mean(audio, axis=1, dtype=numpy.float32)
    return audio, sample_rate


def get_singleton_file(folder):
    files = [os.path.join(folder, name) for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]
    if len(files) != 1:
        raise ValueError(f"Expected one file in {folder}, found {len(files)}")
    return files[0]


def get_single_file_with_extension(directory, extension):
    files = get_files_with_extension(directory, extension)
    if len(files) != 1:
        raise ValueError(f"Expected one {extension} file in {directory}, found {len(files)}")
    return files[0]


def get_files_with_extension(directory, extension):
    extension = extension if extension.startswith(".") else f".{extension}"
    return get_files_ending_with(directory, extension)


def get_files_ending_with(directory, suffix):
    return [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(suffix) and os.path.isfile(os.path.join(directory, name))
    ]


def get_full_file_path(folder, filename_sans_extension):
    files = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.splitext(name)[0] == filename_sans_extension
    ]
    if len(files) != 1:
        raise ValueError(
            f"Expected one file named {filename_sans_extension} in {folder}, found {len(files)}"
        )
    return files[0]

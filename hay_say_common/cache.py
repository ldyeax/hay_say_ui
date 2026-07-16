"""Process-safe file cache used by the UI, workers, and model services.

The PyPI implementation used unlocked JSON read/modify/write sequences and
non-atomic audio writes. This native implementation coordinates every process
with `flock` and exposes `update_metadata` for atomic mutations.
"""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import soundfile

from .file_integration import ROOT_DIR, guarantee_directory
from .utility import read_audio

CACHE_FORMAT = "FLAC"
CACHE_EXTENSION = ".flac"
CACHE_MIMETYPE = "audio/flac;base64"
MAX_FILES_PER_STAGE = int(os.environ.get("HAY_SAY_MAX_CACHE_FILES", "100"))
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


class Stage(Enum):
    RAW = auto()
    PREPROCESSED = auto()
    OUTPUT = auto()
    POSTPROCESSED = auto()


class FileImpl:
    METADATA_FILENAME = "metadata.json"
    ROOT_DIR = ROOT_DIR
    AUDIO_FOLDER = os.environ.get("HAY_SAY_AUDIO_CACHE_DIR", os.path.join(ROOT_DIR, "audio_cache"))
    folder_map = {
        Stage.RAW: "raw",
        Stage.PREPROCESSED: "preprocessed",
        Stage.OUTPUT: "output",
        Stage.POSTPROCESSED: "postprocessed",
    }

    @classmethod
    def map_folder(cls, stage, session_id):
        parts = [cls.AUDIO_FOLDER]
        if session_id:
            parts.append(str(session_id))
        parts.append(cls.folder_map[stage])
        return os.path.join(*parts)

    @classmethod
    def _metadata_path(cls, stage, session_id):
        return os.path.join(cls.map_folder(stage, session_id), cls.METADATA_FILENAME)

    @classmethod
    @contextmanager
    def _session_lock(cls, session_id, exclusive):
        if not session_id:
            yield
            return
        lock_root = Path(cls.AUDIO_FOLDER) / ".session-locks"
        lock_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()
        with (lock_root / f"{key}.lock").open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @classmethod
    def _touch_session(cls, session_id):
        if session_id:
            session_root = Path(cls.AUDIO_FOLDER) / str(session_id)
            session_root.mkdir(parents=True, exist_ok=True)
            (session_root / ".last-access").touch()

    @classmethod
    @contextmanager
    def _lock(cls, stage, session_id, exclusive):
        with cls._session_lock(session_id, exclusive=False):
            cls._touch_session(session_id)
            folder = guarantee_directory(cls.map_folder(stage, session_id))
            lock_path = os.path.join(folder, ".cache.lock")
            with open(lock_path, "a+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @classmethod
    def _read_metadata_unlocked(cls, stage, session_id):
        path = cls._metadata_path(stage, session_id)
        if not os.path.isfile(path):
            return {}
        with open(path, encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    @classmethod
    def _write_metadata_unlocked(cls, stage, session_id, contents):
        folder = guarantee_directory(cls.map_folder(stage, session_id))
        target = cls._metadata_path(stage, session_id)
        descriptor, temporary = tempfile.mkstemp(prefix=".metadata-", suffix=".json", dir=folder)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as metadata_file:
                json.dump(contents, metadata_file, sort_keys=True, indent=4)
                metadata_file.flush()
                os.fsync(metadata_file.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def read_metadata(cls, stage, session_id):
        with cls._lock(stage, session_id, exclusive=False):
            return cls._read_metadata_unlocked(stage, session_id)

    @classmethod
    def write_metadata(cls, stage, session_id, dict_contents):
        with cls._lock(stage, session_id, exclusive=True):
            cls._write_metadata_unlocked(stage, session_id, dict_contents)

    @classmethod
    def update_metadata(cls, stage, session_id, updater):
        """Atomically mutate metadata and return the updated dictionary.

        `updater` receives the live dictionary while an exclusive process lock
        is held. It may mutate in place or return a replacement dictionary.
        """
        with cls._lock(stage, session_id, exclusive=True):
            metadata = cls._read_metadata_unlocked(stage, session_id)
            replacement = updater(metadata)
            if replacement is not None:
                metadata = replacement
            cls._write_metadata_unlocked(stage, session_id, metadata)
            return metadata

    @classmethod
    @contextmanager
    def metadata_transaction(cls, stage, session_id):
        with cls._lock(stage, session_id, exclusive=True):
            metadata = cls._read_metadata_unlocked(stage, session_id)
            yield metadata
            cls._write_metadata_unlocked(stage, session_id, metadata)

    @classmethod
    def _audio_path(cls, stage, session_id, filename_sans_extension):
        return os.path.join(cls.map_folder(stage, session_id), filename_sans_extension + CACHE_EXTENSION)

    @classmethod
    def read_audio_from_cache(cls, stage, session_id, filename_sans_extension):
        with cls._lock(stage, session_id, exclusive=False):
            return read_audio(cls._audio_path(stage, session_id, filename_sans_extension))

    @classmethod
    def _write_audio_unlocked(cls, stage, session_id, filename_sans_extension, array, samplerate):
        folder = guarantee_directory(cls.map_folder(stage, session_id))
        target = cls._audio_path(stage, session_id, filename_sans_extension)
        descriptor, temporary = tempfile.mkstemp(prefix=".audio-", suffix=CACHE_EXTENSION, dir=folder)
        os.close(descriptor)
        try:
            soundfile.write(temporary, array, samplerate, format=CACHE_FORMAT)
            with open(temporary, "rb") as audio_file:
                os.fsync(audio_file.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def save_audio_to_cache(cls, stage, session_id, filename_sans_extension, array, samplerate):
        with cls._lock(stage, session_id, exclusive=True):
            target = cls._audio_path(stage, session_id, filename_sans_extension)
            existing = cls._audio_files_unlocked(stage, session_id)
            if target not in existing and len(existing) >= MAX_FILES_PER_STAGE:
                cls._delete_oldest_unlocked(stage, session_id)
            cls._write_audio_unlocked(stage, session_id, filename_sans_extension, array, samplerate)

    @classmethod
    def write_audio_file(cls, stage, session_id, filename_sans_extension, array, samplerate):
        with cls._lock(stage, session_id, exclusive=True):
            cls._write_audio_unlocked(stage, session_id, filename_sans_extension, array, samplerate)

    @classmethod
    def _audio_files_unlocked(cls, stage, session_id):
        folder = guarantee_directory(cls.map_folder(stage, session_id))
        return [str(path) for path in Path(folder).glob(f"*{CACHE_EXTENSION}") if path.is_file()]

    @classmethod
    def count_audio_cache_files(cls, stage, session_id):
        with cls._lock(stage, session_id, exclusive=False):
            return len(cls._audio_files_unlocked(stage, session_id))

    @classmethod
    def _delete_oldest_unlocked(cls, stage, session_id):
        files = cls._audio_files_unlocked(stage, session_id)
        if not files:
            return None
        oldest = min(files, key=os.path.getmtime)
        key = os.path.splitext(os.path.basename(oldest))[0]
        os.remove(oldest)
        metadata = cls._read_metadata_unlocked(stage, session_id)
        if key in metadata:
            del metadata[key]
            cls._write_metadata_unlocked(stage, session_id, metadata)
        return key

    @classmethod
    def delete_oldest_cache_file(cls, stage, session_id):
        with cls._lock(stage, session_id, exclusive=True):
            return cls._delete_oldest_unlocked(stage, session_id)

    @classmethod
    def get_hashes_sorted_by_timestamp(cls, stage, session_id):
        with cls._lock(stage, session_id, exclusive=False):
            metadata = cls._read_metadata_unlocked(stage, session_id)

            def timestamp(key):
                value = metadata[key].get("Time of Creation")
                try:
                    return datetime.strptime(value, TIMESTAMP_FORMAT).timestamp()
                except (TypeError, ValueError):
                    path = cls._audio_path(stage, session_id, key)
                    return os.path.getmtime(path) if os.path.exists(path) else 0

            return sorted(metadata, key=timestamp, reverse=True)

    @classmethod
    def file_is_already_cached(cls, stage, session_id, filename_sans_extension):
        with cls._lock(stage, session_id, exclusive=False):
            metadata = cls._read_metadata_unlocked(stage, session_id)
            path = cls._audio_path(stage, session_id, filename_sans_extension)
            return filename_sans_extension in metadata and os.path.isfile(path)

    @classmethod
    def delete_all_files_at_stage(cls, stage, session_id):
        with cls._lock(stage, session_id, exclusive=True):
            folder = guarantee_directory(cls.map_folder(stage, session_id))
            for name in os.listdir(folder):
                if name == ".cache.lock":
                    continue
                path = os.path.join(folder, name)
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)

    @classmethod
    def read_file_bytes(cls, stage, session_id, filename_sans_extension):
        with cls._lock(stage, session_id, exclusive=False):
            with open(cls._audio_path(stage, session_id, filename_sans_extension), "rb") as audio_file:
                return audio_file.read()

    @classmethod
    def delete_old_session_data(cls, cutoff_in_seconds=24 * 3600):
        root = Path(cls.AUDIO_FOLDER)
        if not root.exists():
            return
        stage_names = set(cls.folder_map.values())
        for path in root.iterdir():
            if path.is_dir() and path.name not in stage_names and not path.name.startswith("."):
                cls._delete_session_if_older(path.name, cutoff_in_seconds)

    @classmethod
    def _session_last_access(cls, path):
        marker = path / ".last-access"
        if marker.exists():
            return marker.stat().st_mtime
        timestamps = [path.stat().st_mtime]
        timestamps.extend(item.stat().st_mtime for item in path.rglob("*") if item.exists())
        return max(timestamps)

    @classmethod
    def _delete_session_if_older(cls, session_id, cutoff_in_seconds):
        with cls._session_lock(session_id, exclusive=True):
            path = Path(cls.AUDIO_FOLDER) / str(session_id)
            if path.is_dir() and time.time() - cls._session_last_access(path) > cutoff_in_seconds:
                shutil.rmtree(path)

    @classmethod
    def delete_session_data(cls, session_id):
        with cls._session_lock(session_id, exclusive=True):
            path = os.path.join(cls.AUDIO_FOLDER, str(session_id))
            if os.path.isdir(path):
                shutil.rmtree(path)


class MongoImpl:
    """Reserved for compatibility with old command lines."""

    def __getattr__(self, name):
        raise NotImplementedError("The Mongo cache backend has never been implemented; use the file cache")


cache_implementation_map = {"file": FileImpl}


def select_cache_implementation(choice):
    try:
        return cache_implementation_map[choice]
    except KeyError as error:
        raise ValueError(f"Unsupported cache implementation: {choice}") from error

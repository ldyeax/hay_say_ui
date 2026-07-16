import multiprocessing
import os
import threading
import time

import numpy

from hay_say_common.cache import FileImpl, Stage


def _update_metadata_worker(folder, key):
    FileImpl.AUDIO_FOLDER = folder
    FileImpl.update_metadata(Stage.OUTPUT, None, lambda metadata: metadata.update({key: {"value": key}}))


def test_metadata_updates_are_process_safe(tmp_path):
    folder = str(tmp_path / "cache")
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_update_metadata_worker, args=(folder, str(index))) for index in range(12)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(5)
        assert process.exitcode == 0

    FileImpl.AUDIO_FOLDER = folder
    assert set(FileImpl.read_metadata(Stage.OUTPUT, None)) == {str(index) for index in range(12)}


def test_cached_audio_survives_successive_lookups(tmp_path):
    FileImpl.AUDIO_FOLDER = str(tmp_path / "cache")
    samples = numpy.zeros(128, dtype=numpy.float32)
    FileImpl.save_audio_to_cache(Stage.OUTPUT, None, "stable", samples, 32000)
    FileImpl.update_metadata(Stage.OUTPUT, None, lambda metadata: metadata.update({"stable": {}}))

    assert FileImpl.file_is_already_cached(Stage.OUTPUT, None, "stable")
    assert FileImpl.file_is_already_cached(Stage.OUTPUT, None, "stable")
    restored, sample_rate = FileImpl.read_audio_from_cache(Stage.OUTPUT, None, "stable")
    assert sample_rate == 32000
    assert len(restored) == len(samples)


def test_recent_session_activity_prevents_age_cleanup(tmp_path):
    FileImpl.AUDIO_FOLDER = str(tmp_path / "cache")
    FileImpl.write_metadata(Stage.OUTPUT, "active", {})
    marker = tmp_path / "cache" / "active" / ".last-access"
    old = time.time() - 3600
    os.utime(marker, (old, old))

    FileImpl.read_metadata(Stage.OUTPUT, "active")
    FileImpl.delete_old_session_data(cutoff_in_seconds=60)

    assert (tmp_path / "cache" / "active").is_dir()


def test_cleanup_waits_for_active_session_lock(tmp_path):
    FileImpl.AUDIO_FOLDER = str(tmp_path / "cache")
    FileImpl.write_metadata(Stage.OUTPUT, "locked", {})
    marker = tmp_path / "cache" / "locked" / ".last-access"
    old = time.time() - 3600
    os.utime(marker, (old, old))
    finished = threading.Event()

    with FileImpl._session_lock("locked", exclusive=False):
        cleanup = threading.Thread(
            target=lambda: (FileImpl.delete_old_session_data(cutoff_in_seconds=60), finished.set())
        )
        cleanup.start()
        time.sleep(0.05)
        assert not finished.is_set()
        assert marker.parent.is_dir()

    cleanup.join(2)
    assert finished.is_set()
    assert not marker.parent.exists()

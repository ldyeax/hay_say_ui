"""Persistent, process-isolated runtime for So-VITS-SVC 4.0 and 4.1."""

from __future__ import annotations

import contextlib
import gc
import io
import json
import logging
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

if __package__:
    from .slice_runtime import (
        IsolatedWorkerPool,
        WorkerSpec,
        assemble_crossfaded,
        plan_forced_clips,
        silence_for_input_length,
    )
else:  # The installed native service places this directory on sys.path.
    from slice_runtime import (
        IsolatedWorkerPool,
        WorkerSpec,
        assemble_crossfaded,
        plan_forced_clips,
        silence_for_input_length,
    )


def normalize_device(gpu_id: Any) -> str:
    if isinstance(gpu_id, bool):
        raise ValueError("GPU ID must be an integer, an empty string, or 'cpu'")
    if gpu_id is None:
        return "cpu"
    if isinstance(gpu_id, int):
        return "cpu" if gpu_id < 0 else "cuda:{}".format(gpu_id)
    if not isinstance(gpu_id, str):
        raise ValueError("GPU ID must be an integer, an empty string, or 'cpu'")
    value = gpu_id.strip().lower()
    if value in ("", "cpu", "-1"):
        return "cpu"
    if value.startswith("cuda:"):
        value = value.split(":", 1)[1]
    try:
        index = int(value)
    except ValueError as exc:
        raise ValueError("GPU ID must be an integer, an empty string, or 'cpu'") from exc
    if index < 0:
        return "cpu"
    return "cuda:{}".format(index)


def file_revision(path: str) -> Tuple[int, int, int, int]:
    stat = os.stat(path)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def config_details(config_path: str) -> Tuple[str, int]:
    with open(config_path, encoding="utf-8") as source:
        config = json.load(source)
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("SVC4 config is missing its data section")
    sample_rate = data.get("sampling_rate")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("SVC4 config has an invalid sampling rate")
    version = "4.1" if "contentvec_final_proj" in data else "4.0"
    return version, sample_rate


@dataclass(frozen=True)
class ModelSpec:
    character: str
    version: str
    source_root: str
    model_path: str
    config_path: str
    cluster_path: str
    target_sample: int
    model_revision: Tuple[int, int, int, int]
    config_revision: Tuple[int, int, int, int]
    cluster_revision: Optional[Tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class SegmentJob:
    index: int
    audio: np.ndarray = field(repr=False, compare=False)
    sample_rate: int
    threshold: float


@dataclass(frozen=True)
class ClipJob:
    index: int
    audio: np.ndarray = field(repr=False, compare=False)
    sample_rate: int
    speaker: str
    pitch: int
    cluster_ratio: float
    predict_pitch: bool
    noise_scale: float
    reduce_hoarseness: bool
    cpu_bf16: bool
    pad_seconds: float = 0.5


def _mono_float_audio(audio: Any) -> np.ndarray:
    value = np.asarray(audio)
    if value.ndim == 2:
        channel_axis = 1 if value.shape[0] >= value.shape[1] else 0
        value = value.mean(axis=channel_axis)
    if value.ndim != 1:
        raise ValueError("input audio must be mono or a two-dimensional channel array")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError("input audio must contain numeric samples")
    return np.ascontiguousarray(value, dtype=np.float32)


def _worker_error(message: str) -> Tuple[str, str]:
    return "error", message


def configure_worker_threads(device: str = "cpu") -> Any:
    """Configure Torch before any upstream SVC module can import it."""

    from hay_say_torch_bootstrap import configure_torch_threads

    if device == "cpu":
        configured = os.environ.get("HAY_SAY_SVC4_CPU_THREADS_PER_WORKER", "1")
        try:
            threads = int(configured)
        except ValueError as exc:
            raise RuntimeError(
                "HAY_SAY_SVC4_CPU_THREADS_PER_WORKER must be a positive integer"
            ) from exc
        if threads < 1:
            raise RuntimeError(
                "HAY_SAY_SVC4_CPU_THREADS_PER_WORKER must be a positive integer"
            )
        return configure_torch_threads(force=True, intraop_threads=threads)
    return configure_torch_threads(force=True)


def _call_model_infer(model: Any, spec: ModelSpec, device: str, payload: ClipJob, source: Any) -> Any:
    """Call the version-specific upstream API under request-scoped autocast."""

    try:
        from hay_say_torch_bootstrap import cpu_bf16_autocast
    except ImportError:
        cpu_bf16_autocast = lambda _enabled: contextlib.nullcontext()  # noqa: E731
    with cpu_bf16_autocast(payload.cpu_bf16 and device == "cpu"):
        common = {
            "cluster_infer_ratio": payload.cluster_ratio,
            "auto_predict_f0": payload.predict_pitch,
            "noice_scale": payload.noise_scale,
        }
        if spec.version == "4.1":
            return model.infer(
                payload.speaker,
                payload.pitch,
                source,
                f0_predictor="crepe" if payload.reduce_hoarseness else "pm",
                **common,
            )
        return model.infer(
            payload.speaker,
            payload.pitch,
            source,
            F0_mean_pooling=payload.reduce_hoarseness,
            **common,
        )


def _model_worker_main(connection: Any, spec: ModelSpec, device: str, enhance: bool) -> None:
    """Own exactly one upstream ``Svc`` instance in a clean import namespace."""

    try:
        configure_worker_threads(device)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("numba").setLevel(logging.WARNING)
        os.chdir(spec.source_root)
        source_root = os.path.realpath(spec.source_root)
        sys.path[:] = [source_root] + [
            path for path in sys.path if os.path.realpath(path or os.curdir) != source_root
        ]
        from inference import slicer  # type: ignore
        from inference.infer_tool import Svc  # type: ignore
        import soundfile

        model = Svc(
            spec.model_path,
            spec.config_path,
            device,
            spec.cluster_path,
            enhance,
        )
        connection.send(("ready", {"sample_rate": int(model.target_sample)}))
    except BaseException:
        try:
            connection.send(_worker_error(traceback.format_exc()))
        finally:
            connection.close()
        return

    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send(("closed", None))
                break
            try:
                if command == "segment":
                    descriptor, path = tempfile.mkstemp(prefix="hay-say-svc4-", suffix=".wav")
                    os.close(descriptor)
                    try:
                        soundfile.write(path, payload.audio, payload.sample_rate)
                        chunks = slicer.cut(path, db_thresh=payload.threshold)
                        segments, sample_rate = slicer.chunks2audio(path, chunks)
                    finally:
                        try:
                            os.unlink(path)
                        except FileNotFoundError:
                            pass
                    value = tuple(
                        (bool(is_silence), np.ascontiguousarray(audio, dtype=np.float32))
                        for is_silence, audio in segments
                    ), int(sample_rate)
                    connection.send(("result", value))
                    continue
                if command != "infer":
                    raise ValueError("unknown worker command: {}".format(command))

                pad_samples = int(round(payload.sample_rate * payload.pad_seconds))
                padded = np.pad(payload.audio, (pad_samples, pad_samples))
                source = io.BytesIO()
                soundfile.write(source, padded, payload.sample_rate, format="WAV")
                source.seek(0)
                result = _call_model_infer(model, spec, device, payload, source)
                output = result[0]
                detach = getattr(output, "detach", None)
                if callable(detach):
                    output = detach()
                cast_float = getattr(output, "float", None)
                if callable(cast_float):
                    output = cast_float()
                cpu = getattr(output, "cpu", None)
                if callable(cpu):
                    output = cpu()
                numpy = getattr(output, "numpy", None)
                if callable(numpy):
                    output = numpy()
                output = np.asarray(output, dtype=np.float32).reshape(-1)
                target_pad = int(round(spec.target_sample * payload.pad_seconds))
                if target_pad:
                    if output.shape[0] <= target_pad * 2:
                        raise RuntimeError("SVC4 produced less audio than its inference padding")
                    output = output[target_pad:-target_pad]
                connection.send(("result", np.ascontiguousarray(output, dtype=np.float32)))
            except BaseException:
                connection.send(_worker_error(traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        try:
            unload = getattr(model, "unload_model", None)
            if callable(unload):
                unload()
        except BaseException:
            pass
        connection.close()


class ProcessModelWorker:
    """Synchronous proxy for one persistent spawned model process."""

    def __init__(
        self,
        spec: ModelSpec,
        device: str,
        enhance: bool,
        name: str,
        process_target: Callable[..., None] = _model_worker_main,
    ):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._connection = parent
        self._process = context.Process(
            target=process_target,
            args=(child, spec, device, enhance),
            name="svc4-{}".format(name),
            daemon=True,
        )
        self._process.start()
        child.close()
        try:
            status, value = self._receive()
        except BaseException:
            self.close()
            raise
        if status != "ready":
            self.close()
            raise RuntimeError("SVC4 worker failed to start:\n{}".format(value))
        if int(value["sample_rate"]) != spec.target_sample:
            self.close()
            raise RuntimeError("SVC4 worker loaded an unexpected target sample rate")

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid

    def _receive(self) -> Tuple[str, Any]:
        try:
            return self._connection.recv()
        except EOFError as exc:
            raise RuntimeError(
                "SVC4 worker exited unexpectedly with code {}".format(self._process.exitcode)
            ) from exc

    def _request(self, command: str, payload: Any) -> Any:
        if not self._process.is_alive():
            raise RuntimeError(
                "SVC4 worker is not running (exit code {})".format(self._process.exitcode)
            )
        self._connection.send((command, payload))
        status, value = self._receive()
        if status == "error":
            raise RuntimeError("SVC4 worker inference failed:\n{}".format(value))
        if status != "result":
            raise RuntimeError("SVC4 worker returned an unexpected response: {}".format(status))
        return value

    def segment(self, job: SegmentJob) -> Any:
        return self._request("segment", job)

    def infer(self, job: ClipJob) -> np.ndarray:
        return self._request("infer", job)

    def close(self) -> None:
        process = getattr(self, "_process", None)
        connection = getattr(self, "_connection", None)
        if process is None:
            return
        if process.is_alive() and connection is not None:
            try:
                connection.send(("close", None))
                if connection.poll(5):
                    connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if connection is not None:
            connection.close()
        self._process = None
        self._connection = None


class ModelGroup:
    def __init__(
        self,
        spec: ModelSpec,
        device: str,
        enhance: bool,
        cpu_workers: int,
        worker_factory: Callable[[ModelSpec, str, bool, str], Any],
    ):
        self.spec = spec
        self.device = device
        self.enhance = enhance
        worker_count = cpu_workers if device == "cpu" else 1
        self._max_workers = worker_count
        worker_specs = tuple(
            WorkerSpec(
                name="{}-{}".format(device.replace(":", "-"), ordinal),
                device=device,
                factory=lambda ordinal=ordinal: worker_factory(
                    spec,
                    device,
                    enhance,
                    "{}-{}".format(device.replace(":", "-"), ordinal),
                ),
            )
            for ordinal in range(worker_count)
        )
        self.pool = IsolatedWorkerPool(worker_specs)

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def warm(self, workers: int = 1) -> None:
        self.pool.start(workers)

    def segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold: float,
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> Any:
        job = SegmentJob(0, audio, sample_rate, threshold)
        result = self.pool.map_indexed(
            (job,),
            lambda worker, current: worker.segment(current),
            max_workers=1,
            cancel_check=cancel_check,
        )
        return result[0].value

    def infer(
        self,
        jobs: Iterable[ClipJob],
        workers: int,
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> Any:
        return self.pool.map_indexed(
            jobs,
            lambda worker, current: worker.infer(current),
            max_workers=workers,
            cancel_check=cancel_check,
        )

    def close(self) -> None:
        self.pool.close()

    def state(self) -> dict:
        return {
            "workers": self.pool.started_workers,
            "worker_limit": self.max_workers,
            "worker_processes": list(self.pool.state()),
        }


@dataclass
class _CacheEntry:
    spec: ModelSpec
    device: str
    enhance: bool
    group: ModelGroup
    loaded_at: float
    last_used: float
    in_use: int = 0


MIN_MODEL_IDLE_TTL_SECONDS = 30 * 60
CANCELLATION_RETENTION_SECONDS = 60 * 60
MAX_CANCELLATION_TOMBSTONES = 4096


class GenerationCancelled(RuntimeError):
    """Raised at a safe inference boundary after a request is cancelled."""


@dataclass
class _CancellationEntry:
    event: threading.Event = field(default_factory=threading.Event)
    active_requests: int = 0
    cancelled_at: Optional[float] = None


class CancellationRegistry:
    """Share cancellation tokens across concurrent calls with one request ID."""

    def __init__(
        self,
        retention_seconds: float = CANCELLATION_RETENTION_SECONDS,
        max_tombstones: int = MAX_CANCELLATION_TOMBSTONES,
    ):
        self.retention_seconds = float(retention_seconds)
        self.max_tombstones = int(max_tombstones)
        self._entries: OrderedDict[str, _CancellationEntry] = OrderedDict()
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def track(self, request_id: Optional[str]):
        if request_id is None:
            yield None
            return
        with self._lock:
            self._prune_locked(time.time())
            entry = self._entries.get(request_id)
            if entry is None:
                entry = _CancellationEntry()
                self._entries[request_id] = entry
            entry.active_requests += 1
            self._entries.move_to_end(request_id)
        try:
            yield entry.event
        finally:
            with self._lock:
                entry.active_requests -= 1
                if entry.active_requests == 0 and not entry.event.is_set():
                    self._entries.pop(request_id, None)
                self._prune_locked(time.time())

    def cancel(self, request_ids: Iterable[str]) -> dict:
        now = time.time()
        active = []
        normalized = tuple(dict.fromkeys(request_ids))
        with self._lock:
            self._prune_locked(now)
            for request_id in normalized:
                entry = self._entries.get(request_id)
                if entry is None:
                    entry = _CancellationEntry()
                    self._entries[request_id] = entry
                if entry.active_requests:
                    active.append(request_id)
                entry.cancelled_at = entry.cancelled_at or now
                entry.event.set()
                self._entries.move_to_end(request_id)
            self._prune_locked(now)
        return {
            "cancelled_request_ids": list(normalized),
            "active_request_ids": active,
        }

    def snapshot(self) -> dict:
        with self._lock:
            self._prune_locked(time.time())
            return {
                "active_requests": sum(entry.active_requests for entry in self._entries.values()),
                "cancelled_request_ids": sum(
                    1 for entry in self._entries.values() if entry.event.is_set()
                ),
            }

    def commit_if_active(
        self,
        cancellation: Optional[threading.Event],
        callback: Callable[[], Any],
    ) -> Any:
        """Linearize output commit against cancellation for this runtime."""

        if cancellation is None:
            return callback()
        with self._lock:
            if cancellation.is_set():
                raise GenerationCancelled("Generation cancelled")
            return callback()

    def _prune_locked(self, now: float) -> None:
        expired = [
            request_id
            for request_id, entry in self._entries.items()
            if (
                entry.active_requests == 0
                and entry.cancelled_at is not None
                and now - entry.cancelled_at >= self.retention_seconds
            )
        ]
        for request_id in expired:
            self._entries.pop(request_id, None)
        tombstones = [
            request_id
            for request_id, entry in self._entries.items()
            if entry.active_requests == 0 and entry.event.is_set()
        ]
        for request_id in tombstones[:-self.max_tombstones]:
            self._entries.pop(request_id, None)


def model_idle_ttl_seconds(configured: Optional[int] = None) -> int:
    raw = os.environ.get("HAY_SAY_MODEL_IDLE_TTL_SECONDS", str(MIN_MODEL_IDLE_TTL_SECONDS)) \
        if configured is None else configured
    if isinstance(raw, bool):
        raise ValueError("HAY_SAY_MODEL_IDLE_TTL_SECONDS must be a non-negative integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("HAY_SAY_MODEL_IDLE_TTL_SECONDS must be a non-negative integer") from exc
    if value < 0:
        raise ValueError("HAY_SAY_MODEL_IDLE_TTL_SECONDS must be a non-negative integer")
    return max(MIN_MODEL_IDLE_TTL_SECONDS, value)


class ModelCache:
    def __init__(
        self,
        cpu_workers: int,
        max_models_per_device: int = 1,
        worker_factory: Callable[[ModelSpec, str, bool, str], Any] = ProcessModelWorker,
        idle_ttl_seconds: Optional[int] = None,
    ):
        if cpu_workers < 1 or cpu_workers > 64:
            raise ValueError("cpu_workers must be between one and 64")
        if max_models_per_device < 1:
            raise ValueError("max_models_per_device must be at least one")
        self.cpu_workers = int(cpu_workers)
        self.max_models_per_device = int(max_models_per_device)
        self.idle_ttl_seconds = model_idle_ttl_seconds(idle_ttl_seconds)
        self.worker_factory = worker_factory
        self._entries: OrderedDict[Any, _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _key(spec: ModelSpec, device: str, enhance: bool) -> tuple:
        return (
            os.path.realpath(spec.model_path),
            spec.model_revision,
            os.path.realpath(spec.config_path),
            spec.config_revision,
            os.path.realpath(spec.cluster_path) if spec.cluster_path else "",
            spec.cluster_revision,
            device,
            bool(enhance),
        )

    @contextlib.contextmanager
    def acquire(self, spec: ModelSpec, device: str, enhance: bool):
        key = self._key(spec, device, enhance)
        evicted = []
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                now = time.time()
                entry = _CacheEntry(
                    spec,
                    device,
                    enhance,
                    ModelGroup(spec, device, enhance, self.cpu_workers, self.worker_factory),
                    now,
                    now,
                )
                self._entries[key] = entry
            entry.in_use += 1
            entry.last_used = time.time()
            self._entries.move_to_end(key)
            evicted = self._take_evictions(device, key)
        self._close(evicted)
        try:
            yield entry.group
        finally:
            with self._lock:
                current = self._entries.get(key)
                if current is entry:
                    current.in_use -= 1
                    current.last_used = time.time()
                    evicted = self._take_evictions(device)
                else:
                    evicted = []
            self._close(evicted)

    def _take_evictions(self, device: str, protected: Any = None) -> list:
        evicted = []
        now = time.time()
        while sum(entry.device == device for entry in self._entries.values()) > self.max_models_per_device:
            candidate = next(
                (
                    key
                    for key, entry in self._entries.items()
                    if (
                        key != protected
                        and entry.device == device
                        and entry.in_use == 0
                        and now - entry.last_used >= self.idle_ttl_seconds
                    )
                ),
                None,
            )
            if candidate is None:
                break
            evicted.append(self._entries.pop(candidate))
        return evicted

    @staticmethod
    def _close(entries: Sequence[_CacheEntry]) -> None:
        for entry in entries:
            entry.group.close()
        if entries:
            gc.collect()

    def unload(self, character: Optional[str] = None, device: Optional[str] = None) -> dict:
        removed = []
        busy = []
        with self._lock:
            for key, entry in list(self._entries.items()):
                if character is not None and entry.spec.character != character:
                    continue
                if device is not None and entry.device != device:
                    continue
                if entry.in_use:
                    busy.append(self._state(entry))
                else:
                    removed.append(self._entries.pop(key))
        self._close(removed)
        return {
            "unloaded_models": [self._state(entry) for entry in removed],
            "busy_models": busy,
        }

    def close(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        self._close(entries)

    def snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            return {
                "max_models_per_device": self.max_models_per_device,
                "idle_ttl_seconds": self.idle_ttl_seconds,
                "loaded_models": [self._state(entry, now) for entry in self._entries.values()],
            }

    def _state(self, entry: _CacheEntry, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        return {
            "character": entry.spec.character,
            "version": entry.spec.version,
            "device": entry.device,
            "model_path": entry.spec.model_path,
            "config_path": entry.spec.config_path,
            "model_revision": revision_state(entry.spec.model_revision),
            "config_revision": revision_state(entry.spec.config_revision),
            "cluster_revision": revision_state(entry.spec.cluster_revision),
            "enhance": entry.enhance,
            "loaded_at": entry.loaded_at,
            "last_used": entry.last_used,
            "warm_until": entry.last_used + self.idle_ttl_seconds,
            "idle_seconds": max(0.0, now - entry.last_used),
            "idle_ttl_remaining_seconds": max(
                0.0, entry.last_used + self.idle_ttl_seconds - now
            ),
            "active_leases": entry.in_use,
            **entry.group.state(),
        }


class SVC4Runtime:
    def __init__(self, model_cache: ModelCache):
        self.model_cache = model_cache
        self._state_lock = threading.RLock()
        self._device_locks = {}
        self._active_jobs = {}
        self._queued_jobs = {}
        self._cancellations = CancellationRegistry()

    @contextlib.contextmanager
    def _job(self, device: str, cancellation: Optional[threading.Event] = None):
        with self._state_lock:
            lock = self._device_locks.setdefault(device, threading.Lock())
            self._queued_jobs[device] = self._queued_jobs.get(device, 0) + 1
        acquired = False
        queued = True
        active = False
        try:
            while not acquired:
                self.raise_if_cancelled(cancellation)
                acquired = lock.acquire(timeout=0.05)
            self.raise_if_cancelled(cancellation)
            with self._state_lock:
                self._queued_jobs[device] -= 1
                queued = False
                self._active_jobs[device] = self._active_jobs.get(device, 0) + 1
                active = True
            yield
        finally:
            with self._state_lock:
                if queued:
                    self._queued_jobs[device] -= 1
                if active:
                    self._active_jobs[device] -= 1
            if acquired:
                lock.release()

    @contextlib.contextmanager
    def cancellation_scope(self, request_id: Optional[str]):
        with self._cancellations.track(request_id) as cancellation:
            self.raise_if_cancelled(cancellation)
            yield cancellation

    def cancel(self, request_ids: Iterable[str]) -> dict:
        return self._cancellations.cancel(request_ids)

    def commit_if_active(
        self,
        cancellation: Optional[threading.Event],
        callback: Callable[[], Any],
    ) -> Any:
        return self._cancellations.commit_if_active(cancellation, callback)

    @staticmethod
    def raise_if_cancelled(cancellation: Optional[threading.Event]) -> None:
        if cancellation is not None and cancellation.is_set():
            raise GenerationCancelled("Generation cancelled")

    def _workers(self, requested: int, device: str, jobs: int) -> int:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
            raise ValueError("Slice Workers must be a non-negative integer")
        if device != "cpu":
            return 1
        limit = self.model_cache.cpu_workers
        return min(jobs, requested or limit, limit) if jobs else 1

    def warm(self, spec: ModelSpec, device: str, enhance: bool = False, workers: int = 1) -> dict:
        effective = self._workers(workers, device, max(1, workers))
        with self._job(device):
            with self.model_cache.acquire(spec, device, enhance) as group:
                group.warm(effective)
        return {
            "character": spec.character,
            "version": spec.version,
            "device": device,
            "sample_rate": spec.target_sample,
            "slice_workers": effective,
        }

    def generate(
        self,
        spec: ModelSpec,
        speaker: str,
        source_audio: Any,
        sample_rate: int,
        device: str,
        pitch: int,
        slice_seconds: float,
        crossfade_seconds: float,
        slice_workers: int,
        cluster_ratio: float,
        predict_pitch: bool,
        reduce_hoarseness: bool,
        enhance: bool,
        noise_scale: float,
        cpu_bf16: bool = False,
        slice_threshold: float = -40.0,
        cancellation: Optional[threading.Event] = None,
    ) -> Tuple[np.ndarray, int, int]:
        audio = _mono_float_audio(source_audio)
        if not audio.size:
            raise ValueError("input audio is empty")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("input sample rate must be a positive integer")
        if not isinstance(cpu_bf16, bool):
            raise ValueError("CPU BF16 Autocast must be a boolean")
        # Validate forced-slice settings even when automatic slicing finds only silence.
        plan_forced_clips(
            1,
            sample_rate,
            spec.target_sample,
            float(slice_seconds),
            float(crossfade_seconds),
        )

        self.raise_if_cancelled(cancellation)

        def cancel_check() -> None:
            self.raise_if_cancelled(cancellation)

        with self._job(device, cancellation):
            with self.model_cache.acquire(spec, device, enhance) as group:
                self.raise_if_cancelled(cancellation)
                segments, segmented_rate = group.segment(
                    audio, sample_rate, slice_threshold, cancel_check=cancel_check
                )
                self.raise_if_cancelled(cancellation)
                if segmented_rate != sample_rate:
                    raise RuntimeError("SVC4 slicer changed the input sample rate")
                output_plan = []
                jobs = []
                next_job = 0
                for is_silence, segment in segments:
                    self.raise_if_cancelled(cancellation)
                    segment = _mono_float_audio(segment)
                    if is_silence:
                        output_plan.append(("silence", len(segment), None, None))
                        continue
                    clips = plan_forced_clips(
                        len(segment),
                        sample_rate,
                        spec.target_sample,
                        float(slice_seconds),
                        float(crossfade_seconds),
                    )
                    job_indexes = []
                    for clip in clips:
                        self.raise_if_cancelled(cancellation)
                        job_indexes.append(next_job)
                        jobs.append(
                            ClipJob(
                                next_job,
                                np.ascontiguousarray(clip.extract(segment), dtype=np.float32),
                                sample_rate,
                                speaker,
                                int(pitch),
                                float(cluster_ratio),
                                bool(predict_pitch),
                                float(noise_scale),
                                bool(reduce_hoarseness),
                                cpu_bf16 and device == "cpu",
                            )
                        )
                        next_job += 1
                    output_plan.append(("clips", None, clips, tuple(job_indexes)))

                requested_workers = slice_workers if float(slice_seconds) > 0 else 1
                workers = self._workers(requested_workers, device, len(jobs))
                results = group.infer(jobs, workers, cancel_check=cancel_check) if jobs else ()
                self.raise_if_cancelled(cancellation)
                by_job = {result.index: result.value for result in results}
                assembled = []
                for kind, silence_length, clips, indexes in output_plan:
                    self.raise_if_cancelled(cancellation)
                    if kind == "silence":
                        assembled.append(
                            silence_for_input_length(
                                silence_length, sample_rate, spec.target_sample
                            )
                        )
                        continue
                    clip_outputs = {
                        clip.index: by_job[job_index]
                        for clip, job_index in zip(clips, indexes)
                    }
                    assembled.append(assemble_crossfaded(clips, clip_outputs, retain_ratio=0.75))
                self.raise_if_cancelled(cancellation)
                output = np.concatenate(assembled).astype(np.float32, copy=False)
                return output, spec.target_sample, workers

    def unload(self, character: Optional[str] = None, device: Optional[str] = None) -> dict:
        return self.model_cache.unload(character, device)

    def close(self) -> None:
        self.model_cache.close()

    def state(self) -> dict:
        cache = self.model_cache.snapshot()
        with self._state_lock:
            active_by_device = {key: value for key, value in self._active_jobs.items() if value}
            queued_by_device = {key: value for key, value in self._queued_jobs.items() if value}
            active_jobs = sum(self._active_jobs.values())
            queued_jobs = sum(self._queued_jobs.values())
        models = cache["loaded_models"]
        devices = []
        known_devices = sorted(
            set(active_by_device) | set(queued_by_device) | {model["device"] for model in models}
        )
        for device in known_devices:
            device_models = [model for model in models if model["device"] == device]
            devices.append({
                "device": device,
                "active_jobs": active_by_device.get(device, 0),
                "queued_jobs": queued_by_device.get(device, 0),
                "busy": active_by_device.get(device, 0) > 0,
                "warm": any(model["workers"] for model in device_models),
                "loaded_models": len(device_models),
            })
        warm = any(model["workers"] for model in models)
        busy = active_jobs > 0
        loaded_devices = sorted({model["device"] for model in models if model["workers"]})
        current_device = (
            loaded_devices[0] if len(loaded_devices) == 1
            else ("multiple" if loaded_devices else None)
        )
        return {
            "status": "busy" if busy else ("warm-idle" if warm else "ready-cold"),
            "device": current_device,
            "warm": warm,
            "busy": busy,
            "active_jobs": active_jobs,
            "queued_jobs": queued_jobs,
            "active_jobs_by_device": active_by_device,
            "queued_jobs_by_device": queued_by_device,
            "cpu_slice_worker_limit": self.model_cache.cpu_workers,
            "loaded_model_details": models,
            "loaded_models": [model["character"] for model in models if model["workers"]],
            "devices": devices,
            "max_models_per_device": cache["max_models_per_device"],
            "idle_ttl_seconds": cache["idle_ttl_seconds"],
            "cancellation": self._cancellations.snapshot(),
            "resources": {"process_rss_bytes": process_rss_bytes(), "worker_rss_bytes": worker_rss(models)},
        }


def build_runtime(
    cpu_workers: Optional[int] = None,
    max_models_per_device: int = 1,
    worker_factory: Callable[[ModelSpec, str, bool, str], Any] = ProcessModelWorker,
) -> SVC4Runtime:
    configured = cpu_workers
    if configured is None:
        configured = int(os.environ.get("HAY_SAY_SVC4_CPU_SLICE_WORKERS", "4"))
    return SVC4Runtime(ModelCache(configured, max_models_per_device, worker_factory))


def revision_state(revision: Optional[Tuple[int, int, int, int]]) -> Optional[dict]:
    if revision is None:
        return None
    return {
        "device": int(revision[0]),
        "inode": int(revision[1]),
        "size": int(revision[2]),
        "modified_ns": int(revision[3]),
    }


def process_rss_bytes(pid: Optional[int] = None) -> Optional[int]:
    try:
        path = "/proc/{}/statm".format(pid if pid is not None else "self")
        with open(path, encoding="ascii") as source:
            resident_pages = int(source.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, TypeError, ValueError):
        return None


def worker_rss(models: Sequence[dict]) -> int:
    total = 0
    for model in models:
        for worker in model.get("worker_processes", ()):
            value = process_rss_bytes(worker.get("pid"))
            if value is not None:
                total += value
    return total

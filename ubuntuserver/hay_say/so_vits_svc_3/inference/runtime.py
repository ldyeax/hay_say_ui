"""Long-lived, thread-safe inference runtime for so-vits-svc 3."""

import contextlib
import gc
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from hay_say_torch_bootstrap import configure_torch_threads, cpu_bf16_autocast


MIN_MODEL_IDLE_TTL_SECONDS = 30 * 60
CANCELLATION_RETENTION_SECONDS = 60 * 60
MAX_CANCELLATION_TOMBSTONES = 4096


class GenerationCancelled(RuntimeError):
    """Raised at a safe inference boundary after a request is cancelled."""


@dataclass
class _CancellationEntry:
    event: threading.Event = field(default_factory=threading.Event)
    active_requests: int = 0
    cancelled_at: float | None = None


class CancellationRegistry:
    """Share cancellation tokens across concurrent calls with one request ID."""

    def __init__(self, retention_seconds=CANCELLATION_RETENTION_SECONDS,
                 max_tombstones=MAX_CANCELLATION_TOMBSTONES):
        self.retention_seconds = float(retention_seconds)
        self.max_tombstones = int(max_tombstones)
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def track(self, request_id):
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

    def cancel(self, request_ids):
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

    def snapshot(self):
        with self._lock:
            self._prune_locked(time.time())
            return {
                "active_requests": sum(entry.active_requests for entry in self._entries.values()),
                "cancelled_request_ids": sum(
                    1 for entry in self._entries.values() if entry.event.is_set()
                ),
            }

    def commit_if_active(self, cancellation, callback):
        """Linearize output commit against cancellation for this runtime."""

        if cancellation is None:
            return callback()
        with self._lock:
            if cancellation.is_set():
                raise GenerationCancelled("Generation cancelled")
            return callback()

    def _prune_locked(self, now):
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


def model_idle_ttl_seconds(configured=None):
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


def _positive_environment_int(name, default):
    raw = os.environ.get(name, str(default))
    if isinstance(raw, bool):
        raise ValueError("{} must be a positive integer".format(name))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a positive integer".format(name)) from exc
    if value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def svc3_cpu_thread_settings(cpu_pitch_workers=None):
    """Bound every replica so the full pitch pool stays within one host-wide budget."""
    logical_cpus = max(1, os.cpu_count() or 1)
    default_budget = logical_cpus * 2
    workers = _positive_environment_int(
        "HAY_SAY_SVC3_CPU_PITCH_WORKERS", 5
    ) if cpu_pitch_workers is None else int(cpu_pitch_workers)
    if workers < 1:
        raise ValueError("HAY_SAY_SVC3_CPU_PITCH_WORKERS must be a positive integer")
    budget = _positive_environment_int("HAY_SAY_SVC3_CPU_THREAD_BUDGET", default_budget)
    if budget < workers:
        raise ValueError(
            "HAY_SAY_SVC3_CPU_THREAD_BUDGET must be at least "
            "HAY_SAY_SVC3_CPU_PITCH_WORKERS"
        )
    requested = _positive_environment_int(
        "HAY_SAY_SVC3_CPU_THREADS",
        os.environ.get("HAY_SAY_MODEL_CPU_THREADS", "4"),
    )
    budgeted_per_worker = budget // workers
    return {
        "requested_threads_per_worker": requested,
        "threads_per_worker": min(requested, budgeted_per_worker),
        "thread_budget": budget,
        "pitch_workers": workers,
    }


def configure_svc3_cpu_threads():
    settings = svc3_cpu_thread_settings()
    return configure_torch_threads(
        force=True,
        intraop_threads=settings["threads_per_worker"],
    )


@dataclass(frozen=True)
class ModelSpec:
    character: str
    model_path: str
    config_path: str
    model_revision: tuple | None = None
    config_revision: tuple | None = None


@dataclass
class _CacheEntry:
    spec: ModelSpec
    device: str
    model: object
    loaded_at: float
    last_used: float
    in_use: int = 0


def normalize_device(gpu_id):
    """Map Hay Say's legacy GPU IDs to explicit torch device strings."""

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


class ModelCache:
    """A device-aware LRU cache that pins models while they are in use."""

    def __init__(self, hubert_path, max_models_per_device=2, model_loader=None, hubert_loader=None,
                 idle_ttl_seconds=None):
        if int(max_models_per_device) < 1:
            raise ValueError("max_models_per_device must be at least 1")
        self.hubert_path = os.path.realpath(hubert_path)
        self.max_models_per_device = int(max_models_per_device)
        self.idle_ttl_seconds = model_idle_ttl_seconds(idle_ttl_seconds)
        self._model_loader = model_loader or self._default_model_loader
        self._hubert_loader = hubert_loader or self._default_hubert_loader
        self._entries = OrderedDict()
        self._huberts = {}
        self._lock = threading.RLock()

    @staticmethod
    def _default_hubert_loader(hubert_path, device):
        torch = configure_svc3_cpu_threads()

        from .infer_tool import hubert_model

        torch_device = torch.device(device)
        if torch_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU")
            if torch_device.index is None or torch_device.index >= torch.cuda.device_count():
                raise RuntimeError("CUDA device {} is not available".format(device))
        hubert = hubert_model.hubert_soft(hubert_path).to(torch_device)
        if hasattr(hubert, "eval"):
            hubert.eval()
        return hubert

    @staticmethod
    def _default_model_loader(spec, device, hubert):
        from .infer_tool import Svc

        return Svc(
            spec.model_path,
            spec.config_path,
            device=device,
            hubert=hubert,
        )

    @staticmethod
    def _key(spec, device):
        return (
            os.path.realpath(spec.model_path),
            spec.model_revision,
            os.path.realpath(spec.config_path),
            spec.config_revision,
            str(device),
        )

    @contextlib.contextmanager
    def acquire(self, spec, device):
        device = str(device)
        key = self._key(spec, device)
        evicted = []
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                hubert = self._huberts.get(device)
                if hubert is None:
                    hubert = self._hubert_loader(self.hubert_path, device)
                    self._huberts[device] = hubert
                model = self._model_loader(spec, device, hubert)
                now = time.time()
                entry = _CacheEntry(spec, device, model, now, now, in_use=0)
                self._entries[key] = entry
            entry.in_use += 1
            entry.last_used = time.time()
            self._entries.move_to_end(key)
            evicted = self._take_lru_evictions_locked(device, protected_key=key)
        self._release_entries(evicted)

        try:
            yield entry.model
        finally:
            with self._lock:
                current = self._entries.get(key)
                if current is entry:
                    current.in_use -= 1
                    current.last_used = time.time()
                    evicted = self._take_lru_evictions_locked(device)
                else:
                    evicted = []
            self._release_entries(evicted)

    def _take_lru_evictions_locked(self, device, protected_key=None):
        evicted = []
        now = time.time()
        while sum(entry.device == device for entry in self._entries.values()) > self.max_models_per_device:
            candidate_key = None
            for key, entry in self._entries.items():
                if (
                    key != protected_key
                    and entry.device == device
                    and entry.in_use == 0
                    and now - entry.last_used >= self.idle_ttl_seconds
                ):
                    candidate_key = key
                    break
            if candidate_key is None:
                break
            evicted.append(self._entries.pop(candidate_key))
        return evicted

    def unload(self, character=None, device=None, release_hubert=True):
        """Unload matching idle models and return names that were busy."""

        if device is not None:
            device = str(device)
        removed = []
        busy = []
        released_huberts = []
        with self._lock:
            target_devices = set()
            for key, entry in list(self._entries.items()):
                if character is not None and entry.spec.character != character:
                    continue
                if device is not None and entry.device != device:
                    continue
                target_devices.add(entry.device)
                if entry.in_use:
                    busy.append(self._entry_state(entry))
                    continue
                removed.append(self._entries.pop(key))

            if device is not None:
                target_devices.add(device)
            elif character is None:
                target_devices.update(self._huberts.keys())

            if release_hubert:
                for target_device in target_devices:
                    has_models = any(entry.device == target_device for entry in self._entries.values())
                    if not has_models and target_device in self._huberts:
                        released_huberts.append((target_device, self._huberts.pop(target_device)))

        self._release_entries(removed)
        for _, hubert in released_huberts:
            self._release_object(hubert)
        self._empty_cuda_cache()
        return {
            "unloaded_models": [self._entry_state(entry) for entry in removed],
            "busy_models": busy,
            "released_hubert_devices": [item[0] for item in released_huberts],
        }

    def snapshot(self):
        with self._lock:
            now = time.time()
            return {
                "max_models_per_device": self.max_models_per_device,
                "idle_ttl_seconds": self.idle_ttl_seconds,
                "loaded_models": [self._entry_state(entry, now) for entry in self._entries.values()],
                "hubert_devices": sorted(self._huberts.keys()),
            }

    def _entry_state(self, entry, now=None):
        now = time.time() if now is None else now
        replicas = getattr(entry.model, "_inference_replicas", ())
        return {
            "character": entry.spec.character,
            "device": entry.device,
            "model_path": entry.spec.model_path,
            "config_path": entry.spec.config_path,
            "model_revision": revision_state(entry.spec.model_revision),
            "config_revision": revision_state(entry.spec.config_revision),
            "loaded_at": entry.loaded_at,
            "last_used": entry.last_used,
            "warm_until": entry.last_used + self.idle_ttl_seconds,
            "idle_seconds": max(0.0, now - entry.last_used),
            "idle_ttl_remaining_seconds": max(
                0.0, entry.last_used + self.idle_ttl_seconds - now
            ),
            "active_leases": entry.in_use,
            "inference_replicas": max(1, len(replicas)),
        }

    @classmethod
    def _release_entries(cls, entries):
        for entry in entries:
            cls._release_object(entry.model)
        if entries:
            cls._empty_cuda_cache()

    @staticmethod
    def _release_object(value):
        if value is None:
            return
        close = getattr(value, "close", None)
        if callable(close):
            close()
        else:
            to = getattr(value, "to", None)
            if callable(to):
                to("cpu")
        del value
        gc.collect()

    @staticmethod
    def _empty_cuda_cache():
        try:
            torch = configure_torch_threads(force=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


class SvcRuntime:
    """Schedules device work and converts in-memory source audio."""

    def __init__(self, model_cache, segmenter=None, cpu_pitch_workers=None):
        self.model_cache = model_cache
        self._segmenter = segmenter or self._default_segmenter
        self.cpu_pitch_workers = self._positive_int(
            cpu_pitch_workers if cpu_pitch_workers is not None
            else os.environ.get("HAY_SAY_SVC3_CPU_PITCH_WORKERS", "5"),
            "HAY_SAY_SVC3_CPU_PITCH_WORKERS",
        )
        self._state_lock = threading.RLock()
        self._device_locks = {}
        self._active_jobs = {}
        self._queued_jobs = {}
        self._cancellations = CancellationRegistry()

    @contextlib.contextmanager
    def _job(self, device, cancellation=None):
        with self._state_lock:
            device_lock = self._device_locks.setdefault(device, threading.Lock())
            self._queued_jobs[device] = self._queued_jobs.get(device, 0) + 1
        acquired = False
        queued = True
        active = False
        try:
            while not acquired:
                self.raise_if_cancelled(cancellation)
                acquired = device_lock.acquire(timeout=0.05)
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
                device_lock.release()

    @contextlib.contextmanager
    def cancellation_scope(self, request_id):
        with self._cancellations.track(request_id) as cancellation:
            self.raise_if_cancelled(cancellation)
            yield cancellation

    def cancel(self, request_ids):
        return self._cancellations.cancel(request_ids)

    def commit_if_active(self, cancellation, callback):
        return self._cancellations.commit_if_active(cancellation, callback)

    @staticmethod
    def raise_if_cancelled(cancellation):
        if cancellation is not None and cancellation.is_set():
            raise GenerationCancelled("Generation cancelled")

    def warm(self, spec, device):
        with self._job(device):
            with self.model_cache.acquire(spec, device) as model:
                if device == "cpu" and hasattr(model, "ensure_inference_replicas"):
                    model.ensure_inference_replicas(self.cpu_pitch_workers)
                return {
                    "character": spec.character,
                    "device": device,
                    "sample_rate": int(model.target_sample),
                }

    def generate(self, spec, speaker, pitches, source_audio, sample_rate, device, slice_db=-40.0,
                 cpu_bf16=False, cancellation=None):
        if not pitches:
            raise ValueError("At least one pitch shift is required")
        if not isinstance(cpu_bf16, bool):
            raise ValueError("CPU BF16 Autocast must be a boolean")
        self.raise_if_cancelled(cancellation)
        with self._job(device, cancellation):
            with self.model_cache.acquire(spec, device) as model:
                self.raise_if_cancelled(cancellation)
                segments = list(self._segmenter(source_audio, sample_rate, slice_db))
                self.raise_if_cancelled(cancellation)
                if not segments:
                    raise ValueError("Input audio is empty")

                prepared = []
                for is_silence, audio, segment_rate in segments:
                    self.raise_if_cancelled(cancellation)
                    if is_silence:
                        prepared.append((True, len(audio), None, segment_rate))
                    else:
                        with cpu_bf16_autocast(cpu_bf16 and device == "cpu"):
                            features = model.prepare_features(audio, segment_rate)
                        self.raise_if_cancelled(cancellation)
                        prepared.append((False, len(audio), features, segment_rate))

                replicas = (model,)
                if device == "cpu" and len(pitches) > 1 and hasattr(model, "ensure_inference_replicas"):
                    replicas = model.ensure_inference_replicas(min(self.cpu_pitch_workers, len(pitches)))
                    self.raise_if_cancelled(cancellation)
                indexed = list(enumerate(pitches))
                if len(replicas) == 1:
                    generated = self._run_pitch_lane(
                        replicas[0], speaker, indexed, prepared, cpu_bf16 and device == "cpu",
                        cancellation,
                    )
                else:
                    lanes = [indexed[index::len(replicas)] for index in range(len(replicas))]
                    generated = []
                    with ThreadPoolExecutor(
                        max_workers=len(replicas), thread_name_prefix="svc3-pitch"
                    ) as executor:
                        futures = [
                            executor.submit(
                                self._run_pitch_lane,
                                replica,
                                speaker,
                                lane,
                                prepared,
                                cpu_bf16 and device == "cpu",
                                cancellation,
                            )
                            for replica, lane in zip(replicas, lanes) if lane
                        ]
                        for future in futures:
                            generated.extend(future.result())
                self.raise_if_cancelled(cancellation)
                generated.sort(key=lambda item: item[0])
                return [(pitch, audio) for _, pitch, audio in generated], int(model.target_sample)

    @classmethod
    def _run_pitch_lane(cls, model, speaker, indexed_pitches, prepared, cpu_bf16,
                        cancellation=None):
        generated = []
        with cpu_bf16_autocast(cpu_bf16):
            for output_index, pitch in indexed_pitches:
                cls.raise_if_cancelled(cancellation)
                output_segments = []
                for is_silence, source_length, features, segment_rate in prepared:
                    cls.raise_if_cancelled(cancellation)
                    if is_silence:
                        output_length = int(np.ceil(
                            source_length / float(segment_rate) * int(model.target_sample)
                        ))
                        output_segments.append(np.zeros(output_length, dtype=np.float32))
                        continue
                    audio, _ = model.infer_from_features(speaker, pitch, features)
                    cls.raise_if_cancelled(cancellation)
                    output_segments.append(cls._to_numpy(audio))
                output = np.concatenate(output_segments).astype(np.float32, copy=False)
                generated.append((output_index, pitch, output))
        return generated

    def unload(self, character=None, device=None):
        return self.model_cache.unload(character=character, device=device, release_hubert=True)

    def state(self):
        cache_state = self.model_cache.snapshot()
        loaded_model_details = cache_state["loaded_models"]
        with self._state_lock:
            devices = set(self._device_locks)
            devices.update(cache_state["hubert_devices"])
            devices.update(model["device"] for model in loaded_model_details)
            device_states = []
            for device in sorted(devices):
                active = self._active_jobs.get(device, 0)
                queued = self._queued_jobs.get(device, 0)
                models = [
                    item for item in loaded_model_details if item["device"] == device
                ]
                device_states.append({
                    "device": device,
                    "active_jobs": active,
                    "queued_jobs": queued,
                    "busy": active > 0,
                    "warm": bool(models),
                    "hubert_loaded": device in cache_state["hubert_devices"],
                    "loaded_models": len(models),
                })
            active_jobs = sum(self._active_jobs.values())
            queued_jobs = sum(self._queued_jobs.values())

        warm = bool(loaded_model_details)
        busy = active_jobs > 0
        loaded_devices = sorted({item["device"] for item in loaded_model_details})
        if len(loaded_devices) == 1:
            current_device = loaded_devices[0]
        elif loaded_devices:
            current_device = "multiple"
        else:
            current_device = None
        return {
            "status": "busy" if busy else ("warm-idle" if warm else "ready-cold"),
            "device": current_device,
            "warm": warm,
            "busy": busy,
            "active_jobs": active_jobs,
            "queued_jobs": queued_jobs,
            "loaded_models": [item["character"] for item in loaded_model_details],
            "loaded_model_details": loaded_model_details,
            "hubert_devices": cache_state["hubert_devices"],
            "devices": device_states,
            "max_models_per_device": cache_state["max_models_per_device"],
            "idle_ttl_seconds": cache_state["idle_ttl_seconds"],
            "cpu_pitch_workers": self.cpu_pitch_workers,
            "cpu_thread_settings": svc3_cpu_thread_settings(self.cpu_pitch_workers),
            "cancellation": self._cancellations.snapshot(),
            "resources": self.resource_usage(loaded_devices),
        }

    @staticmethod
    def _default_segmenter(source_audio, sample_rate, slice_db):
        import torchaudio

        from .infer_tool import audio_to_mono_tensor
        from .slicer import Slicer

        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer")
        source = audio_to_mono_tensor(source_audio).squeeze(0)
        if sample_rate != 32000:
            source = torchaudio.functional.resample(source, sample_rate, 32000)
        waveform = source.detach().float().cpu().numpy().astype(np.float32, copy=False)
        chunks = Slicer(sr=32000, threshold=float(slice_db)).slice(waveform)
        for chunk in chunks.values():
            start_text, end_text = chunk["split_time"].split(",", 1)
            start, end = int(start_text), int(end_text)
            if end > start:
                yield bool(chunk["slice"]), waveform[start:end], 32000

    @staticmethod
    def _to_numpy(audio):
        if hasattr(audio, "detach"):
            audio = audio.detach()
        if hasattr(audio, "float"):
            audio = audio.float()
        if hasattr(audio, "cpu"):
            audio = audio.cpu()
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    @staticmethod
    def _positive_int(value, name):
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("{} must be a positive integer".format(name)) from exc
        if parsed < 1:
            raise ValueError("{} must be a positive integer".format(name))
        return parsed

    @staticmethod
    def gpu_info():
        try:
            torch = configure_torch_threads(force=True)
        except ImportError:
            return []
        result = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free_memory, total_memory = torch.cuda.mem_get_info(index)
            result.append({
                "Index": index,
                "Name": properties.name,
                "Free Memory": int(free_memory),
                "Total Memory": int(total_memory),
                "Allocated Memory": int(torch.cuda.memory_allocated(index)),
                "Reserved Memory": int(torch.cuda.memory_reserved(index)),
            })
        return result

    @classmethod
    def resource_usage(cls, loaded_devices=()):
        usage = {"process_rss_bytes": cls._process_rss_bytes(), "gpus": []}
        cuda_devices = [device for device in loaded_devices if str(device).startswith("cuda:")]
        if not cuda_devices:
            return usage
        try:
            torch = configure_torch_threads(force=True)

            for device in cuda_devices:
                index = int(str(device).split(":", 1)[1])
                usage["gpus"].append({
                    "Index": index,
                    "Allocated Memory": int(torch.cuda.memory_allocated(index)),
                    "Reserved Memory": int(torch.cuda.memory_reserved(index)),
                })
        except (ImportError, RuntimeError, ValueError):
            pass
        return usage

    @staticmethod
    def _process_rss_bytes():
        try:
            with open("/proc/self/statm", "r") as statm:
                resident_pages = int(statm.read().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (IndexError, OSError, ValueError):
            return None


def file_revision(path):
    stat = os.stat(path)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def revision_state(revision):
    if revision is None:
        return None
    device, inode, size, modified_ns = revision
    return {
        "device": int(device),
        "inode": int(inode),
        "size": int(size),
        "modified_ns": int(modified_ns),
    }

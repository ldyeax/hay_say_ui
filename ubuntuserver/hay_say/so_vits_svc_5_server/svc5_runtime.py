"""Persistent, process-isolated So-VITS-SVC 5 runtime."""

from __future__ import annotations

import contextlib
import gc
import math
import multiprocessing
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


MIN_MODEL_IDLE_TTL_SECONDS = 1800.0


class GenerationCancelled(RuntimeError):
    """Raised when a browser generation request is cancelled cooperatively."""


class ReplicaUnavailableError(RuntimeError):
    """Raised when a persistent replica process can no longer accept work."""


def raise_if_cancelled(cancel_event: Any, message: str) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled(message)


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
    return "cpu" if index < 0 else "cuda:{}".format(index)


def file_revision(path: str) -> Tuple[int, int, int, int]:
    stat = os.stat(path)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def revision_state(revision: Optional[Tuple[int, int, int, int]]) -> Optional[dict]:
    if revision is None:
        return None
    return {
        "device": revision[0],
        "inode": revision[1],
        "size": revision[2],
        "modified_ns": revision[3],
    }


@dataclass(frozen=True)
class ModelSpec:
    character: str
    version: int
    source_root: str
    checkpoint_path: str
    config_path: str
    speaker_path: str
    checkpoint_revision: Tuple[int, int, int, int]
    config_revision: Tuple[int, int, int, int]
    speaker_revision: Tuple[int, int, int, int]
    sample_rate: int = 32000


@dataclass(frozen=True)
class FeatureJob:
    audio: np.ndarray = field(repr=False, compare=False)
    sample_rate: int


@dataclass(frozen=True)
class PreparedFeatures:
    ppg: np.ndarray = field(repr=False, compare=False)
    pitch: np.ndarray = field(repr=False, compare=False)
    vector: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def bytes(self) -> int:
        total = self.ppg.nbytes + self.pitch.nbytes
        if self.vector is not None:
            total += self.vector.nbytes
        return int(total)


@dataclass(frozen=True)
class InferJob:
    features: PreparedFeatures = field(repr=False, compare=False)
    pitch_shift: int
    cpu_bf16: bool = False


def mono_float_audio(audio: Any) -> np.ndarray:
    value = np.asarray(audio)
    if value.ndim == 2:
        channel_axis = 1 if value.shape[0] >= value.shape[1] else 0
        value = value.mean(axis=channel_axis)
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.number):
        raise ValueError("input audio must be a mono or stereo numeric array")
    return np.ascontiguousarray(value, dtype=np.float32)


def configure_worker_threads(device: str) -> Any:
    from hay_say_torch_bootstrap import configure_torch_threads

    if device == "cpu":
        threads = int(os.environ.get("HAY_SAY_SVC5_CPU_THREADS_PER_WORKER", "4"))
        if threads < 1:
            raise RuntimeError("HAY_SAY_SVC5_CPU_THREADS_PER_WORKER must be positive")
        return configure_torch_threads(force=True, intraop_threads=threads)
    return configure_torch_threads(force=True)


def _as_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cast = getattr(value, "float", None)
    if callable(cast):
        value = cast()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    return np.ascontiguousarray(value)


class _LoadedPipeline:
    """One warm voice model; feature encoders load lazily in this process."""

    def __init__(self, spec: ModelSpec, device: str):
        self.spec = spec
        self.device_name = device
        self.torch = configure_worker_threads(device)
        self.device = self.torch.device(device)
        if spec.source_root not in sys.path:
            sys.path.insert(0, spec.source_root)
        os.chdir(spec.source_root)

        from omegaconf import OmegaConf
        from vits.models import SynthesizerInfer
        import svc_inference

        self.hp = OmegaConf.load(spec.config_path)
        configured_sample_rate = self.hp.data.sampling_rate
        if (
            isinstance(configured_sample_rate, bool)
            or not isinstance(configured_sample_rate, int)
            or configured_sample_rate != spec.sample_rate
        ):
            raise RuntimeError(
                "SVC5 configuration sample rate changed after request validation"
            )
        self.model = SynthesizerInfer(
            self.hp.data.filter_length // 2 + 1,
            self.hp.data.segment_size // self.hp.data.hop_length,
            self.hp,
        )
        svc_inference.load_svc_model(spec.checkpoint_path, self.model)
        self.model.eval()
        self.model.to(self.device)
        self.speaker = self.torch.as_tensor(
            np.load(spec.speaker_path), dtype=self.torch.float32
        )
        self._whisper = None
        self._hubert = None
        self._whisper_module = None
        self._hubert_module = None
        self._pitch_module = None

    def _load_frontend(self, cancel_event=None) -> None:
        if self._whisper is not None:
            return
        raise_if_cancelled(cancel_event, "SVC5 generation was cancelled before loading frontends")
        import pitch.inference as pitch_inference
        import whisper.inference as whisper_inference

        self._pitch_module = pitch_inference
        self._whisper_module = whisper_inference
        if self.spec.version == 1:
            checkpoint = self.torch.load(
                os.path.join(self.spec.source_root, "whisper_pretrain", "medium.pt"),
                map_location="cpu",
            )
            dimensions = whisper_inference.ModelDimensions(**checkpoint["dims"])
            whisper = whisper_inference.Whisper(dimensions)
            whisper.load_state_dict(checkpoint["model_state_dict"])
            whisper.eval()
            self._whisper = whisper.to(self.device)
        else:
            self._whisper = whisper_inference.load_model(
                os.path.join(self.spec.source_root, "whisper_pretrain", "large-v2.pt"),
                self.device_name,
            )
            raise_if_cancelled(
                cancel_event, "SVC5 generation was cancelled while loading frontends"
            )
            import hubert.inference as hubert_inference

            self._hubert_module = hubert_inference
            self._hubert = hubert_inference.load_model(
                os.path.join(
                    self.spec.source_root,
                    "hubert_pretrain",
                    "hubert-soft-0d54a1f4.pt",
                ),
                self.device_name,
            )
        raise_if_cancelled(
            cancel_event, "SVC5 generation was cancelled while loading frontends"
        )

    def prepare(self, job: FeatureJob, cancel_event=None) -> PreparedFeatures:
        self._load_frontend(cancel_event)
        import soundfile

        with tempfile.TemporaryDirectory(prefix="hay-say-svc5-features-") as temporary:
            wave_path = os.path.join(temporary, "input.wav")
            ppg_path = os.path.join(temporary, "content.npy")
            vector_path = os.path.join(temporary, "hubert.npy")
            soundfile.write(wave_path, job.audio, job.sample_rate)
            if self.spec.version == 1:
                self._whisper_module.pred_ppg(self._whisper, wave_path, ppg_path)
                raise_if_cancelled(
                    cancel_event, "SVC5 generation was cancelled after Whisper extraction"
                )
                pitch = self._pitch_module.compute_f0_nn(wave_path, self.device)
                vector = None
            else:
                self._whisper_module.pred_ppg(
                    self._whisper, wave_path, ppg_path, self.device_name
                )
                raise_if_cancelled(
                    cancel_event, "SVC5 generation was cancelled after Whisper extraction"
                )
                # CREPE and HuBERT contain CPU operations that are not BF16-safe.
                pitch = self._pitch_module.compute_f0_sing(wave_path, self.device_name)
                raise_if_cancelled(
                    cancel_event, "SVC5 generation was cancelled after pitch extraction"
                )
                self._hubert_module.pred_vec(
                    self._hubert, wave_path, vector_path, self.device_name
                )
                vector = np.asarray(np.load(vector_path), dtype=np.float32)
            raise_if_cancelled(
                cancel_event, "SVC5 generation was cancelled during feature extraction"
            )
            ppg = np.asarray(np.load(ppg_path), dtype=np.float32)
        return PreparedFeatures(
            np.ascontiguousarray(ppg),
            np.ascontiguousarray(_as_numpy(pitch), dtype=np.float32).reshape(-1),
            None if vector is None else np.ascontiguousarray(vector),
        )

    def infer(self, job: InferJob, cancel_event=None) -> np.ndarray:
        raise_if_cancelled(cancel_event, "SVC5 generation was cancelled before inference")
        torch = self.torch
        ppg = torch.as_tensor(
            np.repeat(job.features.ppg, 2, axis=0), dtype=torch.float32
        )
        vector = None
        if self.spec.version == 2:
            if job.features.vector is None:
                raise RuntimeError("SVC5 v2 inference requires HuBERT features")
            vector = torch.as_tensor(
                np.repeat(job.features.vector, 2, axis=0), dtype=torch.float32
            )
        pitch = np.asarray(job.features.pitch, dtype=np.float32)
        if job.pitch_shift:
            pitch = pitch * float(2 ** (job.pitch_shift / 12.0))
        pitch = torch.as_tensor(pitch, dtype=torch.float32)

        lengths = [pitch.shape[0], ppg.shape[0]]
        if vector is not None:
            lengths.append(vector.shape[0])
        frame_count = min(lengths)
        pitch = pitch[:frame_count]
        ppg = ppg[:frame_count]
        if vector is not None:
            vector = vector[:frame_count]
        if frame_count == 0:
            return np.empty(0, dtype=np.float32)

        from hay_say_torch_bootstrap import cpu_bf16_autocast

        autocast = cpu_bf16_autocast(job.cpu_bf16 and self.device_name == "cpu")
        with torch.no_grad(), autocast:
            speaker = self.speaker.unsqueeze(0).to(self.device)
            source_pitch = pitch.unsqueeze(0).to(self.device)
            source = self.model.pitch2source(source_pitch)
            raise_if_cancelled(
                cancel_event, "SVC5 generation was cancelled after source generation"
            )
            if self.spec.version == 1:
                output = self._infer_v1(
                    ppg, pitch, speaker, source, frame_count, cancel_event
                )
            else:
                output = self._infer_v2(
                    ppg, vector, pitch, speaker, source, frame_count, cancel_event
                )
        return np.ascontiguousarray(output, dtype=np.float32)

    def _infer_v1(
        self, ppg, pitch, speaker, source, frame_count: int, cancel_event=None
    ) -> np.ndarray:
        torch = self.torch
        hop_size = int(self.hp.data.hop_length)
        hop_frames = 10
        chunk_frames = 2500
        output_index = 0
        output = []
        produced_chunk = False
        while output_index + chunk_frames < frame_count:
            raise_if_cancelled(
                cancel_event, "SVC5 generation was cancelled between inference chunks"
            )
            produced_chunk = True
            start = 0 if output_index == 0 else output_index - hop_frames
            trim_start = 0 if output_index == 0 else hop_frames * hop_size
            if output_index + chunk_frames + hop_frames > frame_count:
                stop = output_index + chunk_frames
                trim_stop = None
            else:
                stop = output_index + chunk_frames + hop_frames
                trim_stop = -hop_frames * hop_size
            current = self.model.inference(
                ppg[start:stop].unsqueeze(0).to(self.device),
                pitch[start:stop].unsqueeze(0).to(self.device),
                speaker,
                torch.LongTensor([stop - start]).to(self.device),
                source[:, :, start * hop_size:stop * hop_size].to(self.device),
            )[0, 0]
            raise_if_cancelled(
                cancel_event, "SVC5 generation was cancelled after an inference chunk"
            )
            current = _as_numpy(current).reshape(-1)
            output.extend(current[trim_start:trim_stop])
            output_index += chunk_frames
        if output_index < frame_count:
            raise_if_cancelled(
                cancel_event, "SVC5 generation was cancelled before the final inference chunk"
            )
            start = output_index - hop_frames if produced_chunk else 0
            trim_start = hop_frames * hop_size if produced_chunk else 0
            current = self.model.inference(
                ppg[start:].unsqueeze(0).to(self.device),
                pitch[start:].unsqueeze(0).to(self.device),
                speaker,
                torch.LongTensor([frame_count - start]).to(self.device),
                source[:, :, start * hop_size:].to(self.device),
            )[0, 0]
            raise_if_cancelled(
                cancel_event, "SVC5 generation was cancelled after the final inference chunk"
            )
            output.extend(_as_numpy(current).reshape(-1)[trim_start:])
        return np.asarray(output, dtype=np.float32)

    def _infer_v2(
        self, ppg, vector, pitch, speaker, source, frame_count: int, cancel_event=None
    ) -> np.ndarray:
        torch = self.torch
        hop_size = int(self.hp.data.hop_length)
        hop_frames = 10
        chunk_frames = 2500
        output_index = 0
        output = []
        while output_index < frame_count:
            raise_if_cancelled(
                cancel_event, "SVC5 generation was cancelled between inference chunks"
            )
            start = 0 if output_index == 0 else output_index - hop_frames
            trim_start = 0 if output_index == 0 else hop_frames * hop_size
            if output_index + chunk_frames + hop_frames > frame_count:
                stop = frame_count
                trim_stop = None
            else:
                stop = output_index + chunk_frames + hop_frames
                trim_stop = -hop_frames * hop_size
            current = self.model.inference(
                ppg[start:stop].unsqueeze(0).to(self.device),
                vector[start:stop].unsqueeze(0).to(self.device),
                pitch[start:stop].unsqueeze(0).to(self.device),
                speaker,
                torch.LongTensor([stop - start]).to(self.device),
                source[:, :, start * hop_size:stop * hop_size].to(self.device),
            )[0, 0]
            raise_if_cancelled(
                cancel_event, "SVC5 generation was cancelled after an inference chunk"
            )
            current = _as_numpy(current).reshape(-1)
            output.extend(current[trim_start:trim_stop])
            output_index += chunk_frames
        return np.asarray(output, dtype=np.float32)


def _worker_error() -> Tuple[str, str]:
    return "error", traceback.format_exc()


def _model_worker_main(connection, spec: ModelSpec, device: str, cancel_event) -> None:
    pipeline = None
    try:
        pipeline = _LoadedPipeline(spec, device)
        connection.send(("ready", {"sample_rate": spec.sample_rate}))
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send(("closed", None))
                break
            try:
                if command == "prepare":
                    value = pipeline.prepare(payload, cancel_event)
                elif command == "infer":
                    value = pipeline.infer(payload, cancel_event)
                else:
                    raise RuntimeError("unknown SVC5 worker command: {}".format(command))
                connection.send(("result", value))
            except GenerationCancelled as exc:
                connection.send(("cancelled", str(exc)))
            except BaseException:
                connection.send(_worker_error())
    except BaseException:
        try:
            connection.send(_worker_error())
        except BaseException:
            pass
    finally:
        pipeline = None
        gc.collect()
        try:
            connection.close()
        except BaseException:
            pass


class ProcessReplica:
    """Synchronous proxy for one warm spawned SVC5 model process."""

    def __init__(
        self,
        spec: ModelSpec,
        device: str,
        name: str,
        process_target: Callable[..., None] = _model_worker_main,
    ):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self.name = name
        self.device = device
        self.prepared_inputs = 0
        self.completed_inferences = 0
        self._connection = parent
        self._cancel_event = context.Event()
        self._process = context.Process(
            target=process_target,
            args=(child, spec, device, self._cancel_event),
            name="svc5-{}".format(name),
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
            raise RuntimeError("SVC5 replica failed to start:\n{}".format(value))

    @property
    def pid(self) -> Optional[int]:
        return None if self._process is None else self._process.pid

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def begin_request(self) -> None:
        self._cancel_event.clear()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _receive(self) -> Tuple[str, Any]:
        try:
            return self._connection.recv()
        except (EOFError, OSError) as exc:
            code = None if self._process is None else self._process.exitcode
            raise ReplicaUnavailableError(
                "SVC5 replica exited unexpectedly with code {}".format(code)
            ) from exc

    def _request(self, command: str, payload: Any) -> Any:
        if self._process is None or not self._process.is_alive():
            raise ReplicaUnavailableError("SVC5 replica is not running")
        try:
            self._connection.send((command, payload))
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise ReplicaUnavailableError("SVC5 replica connection is unavailable") from exc
        status, value = self._receive()
        if status == "cancelled":
            raise GenerationCancelled(str(value))
        if status == "error":
            raise RuntimeError("SVC5 replica operation failed:\n{}".format(value))
        if status != "result":
            raise RuntimeError("unexpected SVC5 replica response: {}".format(status))
        return value

    def prepare(self, job: FeatureJob) -> PreparedFeatures:
        value = self._request("prepare", job)
        self.prepared_inputs += 1
        return value

    def infer(self, job: InferJob) -> np.ndarray:
        value = self._request("infer", job)
        self.completed_inferences += 1
        return value

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
        self._cancel_event = None


@dataclass
class _ReplicaSlot:
    ordinal: int
    replica: Any
    started_at: float
    last_used: float
    busy: bool = False
    uses: int = 0
    active_request_id: Optional[str] = None


class ReplicaGroup:
    """A dynamically growing pool shared by simultaneous HTTP requests."""

    def __init__(
        self,
        spec: ModelSpec,
        device: str,
        limit: int,
        startup_semaphore: threading.Semaphore,
        replica_factory: Callable[[ModelSpec, str, str], Any] = ProcessReplica,
    ):
        self.spec = spec
        self.device = device
        self.limit = int(limit)
        self._startup_semaphore = startup_semaphore
        self._replica_factory = replica_factory
        self._slots = []
        self._available = []
        self._starting = 0
        self._waiting = 0
        self._next_ordinal = 0
        self._closed = False
        self._condition = threading.Condition()

    def _construct(self, ordinal: int) -> _ReplicaSlot:
        with self._startup_semaphore:
            replica = self._replica_factory(
                self.spec,
                self.device,
                "{}-{}".format(self.device.replace(":", "-"), ordinal),
            )
        now = time.time()
        return _ReplicaSlot(ordinal, replica, now, now)

    def _checkout(self, cancelled: Callable[[], bool]) -> _ReplicaSlot:
        while True:
            with self._condition:
                if self._closed:
                    raise RuntimeError("SVC5 replica group is closed")
                if cancelled():
                    raise GenerationCancelled("SVC5 generation was cancelled while queued")
                if self._available:
                    slot = self._available.pop()
                    slot.busy = True
                    return slot
                if len(self._slots) + self._starting < self.limit:
                    ordinal = self._next_ordinal
                    self._next_ordinal += 1
                    self._starting += 1
                    break
                self._waiting += 1
                try:
                    self._condition.wait(timeout=0.2)
                finally:
                    self._waiting -= 1
        try:
            slot = self._construct(ordinal)
        except BaseException:
            with self._condition:
                self._starting -= 1
                self._condition.notify_all()
            raise
        with self._condition:
            self._starting -= 1
            self._slots.append(slot)
            if cancelled():
                self._available.append(slot)
                self._condition.notify_all()
                raise GenerationCancelled("SVC5 generation was cancelled while warming")
            slot.busy = True
            self._condition.notify_all()
        return slot

    def _release(self, slot: _ReplicaSlot) -> None:
        with self._condition:
            slot.busy = False
            slot.active_request_id = None
            slot.last_used = time.time()
            slot.uses += 1
            if not self._closed:
                self._available.append(slot)
            self._condition.notify_all()

    @staticmethod
    def _replica_is_alive(replica: Any) -> bool:
        value = getattr(replica, "is_alive", True)
        try:
            return bool(value() if callable(value) else value)
        except BaseException:
            return False

    @staticmethod
    def _signal_replica_cancel(replica: Any) -> None:
        callback = getattr(replica, "cancel", None)
        if callable(callback):
            callback()

    def _retire(self, slot: _ReplicaSlot) -> None:
        with self._condition:
            slot.busy = False
            slot.active_request_id = None
            if slot in self._available:
                self._available.remove(slot)
            if slot in self._slots:
                self._slots.remove(slot)
            self._condition.notify_all()
        try:
            slot.replica.close()
        except BaseException:
            pass

    def _run(
        self,
        operation: Callable[[Any], Any],
        cancelled: Callable[[], bool],
        request_id: Optional[str],
    ) -> Any:
        slot = self._checkout(cancelled)
        retire = False
        try:
            begin_request = getattr(slot.replica, "begin_request", None)
            if callable(begin_request):
                begin_request()
            with self._condition:
                slot.active_request_id = request_id
            if cancelled():
                self._signal_replica_cancel(slot.replica)
                raise GenerationCancelled("SVC5 generation was cancelled before inference")
            value = operation(slot.replica)
            if cancelled():
                raise GenerationCancelled("SVC5 generation was cancelled during inference")
            return value
        except BaseException as exc:
            retire = isinstance(exc, ReplicaUnavailableError) or not self._replica_is_alive(
                slot.replica
            )
            raise
        finally:
            if retire:
                self._retire(slot)
            else:
                self._release(slot)

    def prepare(
        self,
        job: FeatureJob,
        cancelled: Callable[[], bool],
        request_id: Optional[str] = None,
    ) -> PreparedFeatures:
        return self._run(lambda replica: replica.prepare(job), cancelled, request_id)

    def infer(
        self,
        job: InferJob,
        cancelled: Callable[[], bool],
        request_id: Optional[str] = None,
    ) -> np.ndarray:
        return self._run(lambda replica: replica.infer(job), cancelled, request_id)

    def cancel(self, request_ids: Sequence[str]) -> int:
        wanted = set(request_ids)
        interrupted = 0
        with self._condition:
            for slot in self._slots:
                if slot.busy and slot.active_request_id in wanted:
                    self._signal_replica_cancel(slot.replica)
                    interrupted += 1
        return interrupted

    def warm(self, workers: int) -> None:
        target = min(self.limit, max(1, int(workers)))
        with self._condition:
            needed = max(0, target - len(self._slots) - self._starting)
            ordinals = range(self._next_ordinal, self._next_ordinal + needed)
            self._next_ordinal += needed
            self._starting += needed
        if not needed:
            return

        def create(ordinal: int):
            return self._construct(ordinal)

        created = []
        failure = None
        with ThreadPoolExecutor(max_workers=needed, thread_name_prefix="svc5-warm") as executor:
            futures = [executor.submit(create, ordinal) for ordinal in ordinals]
            for future in as_completed(futures):
                try:
                    created.append(future.result())
                except BaseException as exc:
                    failure = failure or exc
        with self._condition:
            self._starting -= needed
            self._slots.extend(created)
            self._available.extend(created)
            self._condition.notify_all()
        if failure is not None:
            raise failure

    def close(self) -> None:
        with self._condition:
            self._closed = True
            slots = list(self._slots)
            self._slots.clear()
            self._available.clear()
            self._condition.notify_all()
        for slot in slots:
            slot.replica.close()

    def state(self, idle_ttl_seconds: float) -> dict:
        now = time.time()
        with self._condition:
            slots = list(self._slots)
            starting = self._starting
            waiting = self._waiting
        return {
            "workers": len(slots),
            "worker_limit": self.limit,
            "starting_workers": starting,
            "busy_workers": sum(slot.busy for slot in slots),
            "queued_jobs": waiting,
            "worker_processes": [
                {
                    "name": getattr(slot.replica, "name", str(slot.ordinal)),
                    "device": self.device,
                    "pid": getattr(slot.replica, "pid", None),
                    "busy": slot.busy,
                    "uses": slot.uses,
                    "started_at": slot.started_at,
                    "last_used": slot.last_used,
                    "minimum_residency_remaining_seconds": max(
                        0.0, idle_ttl_seconds - (now - slot.last_used)
                    ),
                    "prepared_inputs": getattr(slot.replica, "prepared_inputs", 0),
                    "completed_inferences": getattr(
                        slot.replica, "completed_inferences", 0
                    ),
                }
                for slot in slots
            ],
        }


@dataclass
class _ModelEntry:
    spec: ModelSpec
    device: str
    group: ReplicaGroup
    loaded_at: float
    last_used: float
    leases: int = 0


class ModelCache:
    def __init__(
        self,
        cpu_workers: int,
        gpu_workers: int,
        startup_concurrency: int,
        idle_ttl_seconds: float,
        max_models_per_device: int = 1,
        replica_factory: Callable[[ModelSpec, str, str], Any] = ProcessReplica,
    ):
        for name, value in (
            ("cpu_workers", cpu_workers),
            ("gpu_workers", gpu_workers),
            ("startup_concurrency", startup_concurrency),
            ("max_models_per_device", max_models_per_device),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("{} must be a positive integer".format(name))
        if (
            isinstance(idle_ttl_seconds, bool)
            or not isinstance(idle_ttl_seconds, (int, float))
            or not math.isfinite(idle_ttl_seconds)
            or idle_ttl_seconds < 0
        ):
            raise ValueError("idle_ttl_seconds must be a finite non-negative number")
        self.cpu_workers = cpu_workers
        self.gpu_workers = gpu_workers
        self.startup_concurrency = startup_concurrency
        self.idle_ttl_seconds = max(
            MIN_MODEL_IDLE_TTL_SECONDS, float(idle_ttl_seconds)
        )
        self.max_models_per_device = max_models_per_device
        self.replica_factory = replica_factory
        self._startup_semaphore = threading.Semaphore(startup_concurrency)
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _key(spec: ModelSpec, device: str) -> tuple:
        return (
            os.path.realpath(spec.checkpoint_path),
            spec.checkpoint_revision,
            os.path.realpath(spec.config_path),
            spec.config_revision,
            os.path.realpath(spec.speaker_path),
            spec.speaker_revision,
            device,
        )

    @contextlib.contextmanager
    def acquire(self, spec: ModelSpec, device: str):
        key = self._key(spec, device)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                now = time.time()
                limit = self.cpu_workers if device == "cpu" else self.gpu_workers
                entry = _ModelEntry(
                    spec,
                    device,
                    ReplicaGroup(
                        spec,
                        device,
                        limit,
                        self._startup_semaphore,
                        self.replica_factory,
                    ),
                    now,
                    now,
                )
                self._entries[key] = entry
            entry.leases += 1
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
                    current.leases -= 1
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
                    if key != protected
                    and entry.device == device
                    and entry.leases == 0
                    and now - entry.last_used >= self.idle_ttl_seconds
                ),
                None,
            )
            if candidate is None:
                break
            evicted.append(self._entries.pop(candidate))
        return evicted

    @staticmethod
    def _close(entries: Iterable[_ModelEntry]) -> None:
        for entry in entries:
            entry.group.close()
        if entries:
            gc.collect()

    def unload(self, character: Optional[str], device: Optional[str]) -> dict:
        removed = []
        busy = []
        with self._lock:
            for key, entry in list(self._entries.items()):
                if character is not None and entry.spec.character != character:
                    continue
                if device is not None and entry.device != device:
                    continue
                if entry.leases:
                    busy.append(self._state(entry))
                else:
                    removed.append(self._entries.pop(key))
        self._close(removed)
        return {
            "unloaded_models": [self._state(entry) for entry in removed],
            "busy_models": busy,
        }

    def cancel(self, request_ids: Sequence[str]) -> int:
        with self._lock:
            groups = [entry.group for entry in self._entries.values()]
        return sum(group.cancel(request_ids) for group in groups)

    def close(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        self._close(entries)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "cpu_worker_limit": self.cpu_workers,
                "gpu_worker_limit": self.gpu_workers,
                "startup_concurrency": self.startup_concurrency,
                "idle_ttl_seconds": self.idle_ttl_seconds,
                "loaded_models": [self._state(entry) for entry in self._entries.values()],
            }

    def _state(self, entry: _ModelEntry) -> dict:
        now = time.time()
        return {
            "character": entry.spec.character,
            "version": entry.spec.version,
            "device": entry.device,
            "checkpoint_path": entry.spec.checkpoint_path,
            "checkpoint_revision": revision_state(entry.spec.checkpoint_revision),
            "loaded_at": entry.loaded_at,
            "last_used": entry.last_used,
            "active_leases": entry.leases,
            "minimum_residency_remaining_seconds": max(
                0.0, self.idle_ttl_seconds - (now - entry.last_used)
            ),
            **entry.group.state(self.idle_ttl_seconds),
        }


@dataclass
class _FeatureEntry:
    condition: threading.Condition
    value: Optional[PreparedFeatures] = None
    error: Optional[BaseException] = None
    loading: bool = True
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class FeatureCache:
    """Shares pitch-independent frontend features between pitch requests."""

    def __init__(self, capacity: int = 8):
        if capacity < 1:
            raise ValueError("feature cache capacity must be positive")
        self.capacity = int(capacity)
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    def get(
        self,
        key: Any,
        loader: Callable[[], PreparedFeatures],
        cancelled: Callable[[], bool],
    ) -> PreparedFeatures:
        while True:
            with self._lock:
                entry = self._entries.get(key)
                owner = entry is None
                if owner:
                    entry = _FeatureEntry(threading.Condition(self._lock))
                    self._entries[key] = entry
                else:
                    self._entries.move_to_end(key)
                while not owner and entry.loading:
                    if cancelled():
                        raise GenerationCancelled(
                            "SVC5 generation was cancelled awaiting shared features"
                        )
                    entry.condition.wait(timeout=0.2)
                if owner:
                    break
                if cancelled():
                    raise GenerationCancelled(
                        "SVC5 generation was cancelled awaiting shared features"
                    )
                if isinstance(entry.error, GenerationCancelled):
                    # The cancelled owner removes its failed entry. A different
                    # browser request can become the new owner instead of failing
                    # along with work it did not cancel.
                    continue
                if entry.error is not None:
                    raise RuntimeError("shared SVC5 feature extraction failed") from entry.error
                entry.last_used = time.time()
                return entry.value
        try:
            if cancelled():
                raise GenerationCancelled("SVC5 generation was cancelled before feature extraction")
            value = loader()
        except BaseException as exc:
            with self._lock:
                entry.error = exc
                entry.loading = False
                entry.condition.notify_all()
                self._entries.pop(key, None)
            raise
        with self._lock:
            entry.value = value
            entry.loading = False
            entry.last_used = time.time()
            entry.condition.notify_all()
            while len(self._entries) > self.capacity:
                candidate = next(
                    (candidate for candidate, item in self._entries.items() if not item.loading),
                    None,
                )
                if candidate is None:
                    break
                self._entries.pop(candidate)
        return value

    def state(self) -> dict:
        with self._lock:
            entries = list(self._entries.values())
        return {
            "capacity": self.capacity,
            "entries": len(entries),
            "loading": sum(entry.loading for entry in entries),
            "bytes": sum(entry.value.bytes for entry in entries if entry.value is not None),
        }


class SVC5Runtime:
    def __init__(self, model_cache: ModelCache, feature_cache: Optional[FeatureCache] = None):
        self.model_cache = model_cache
        self.feature_cache = feature_cache or FeatureCache()
        self._state_lock = threading.RLock()
        self._active_by_device: Dict[str, int] = {}
        self._active_request_ids: Dict[str, Dict[str, int]] = {}
        self._cancelled: Dict[str, float] = {}

    def _is_cancelled(self, request_id: Optional[str]) -> bool:
        if request_id is None:
            return False
        with self._state_lock:
            return request_id in self._cancelled

    @contextlib.contextmanager
    def _job(self, request_id: Optional[str], device: str):
        with self._state_lock:
            if request_id is not None and request_id in self._cancelled:
                raise GenerationCancelled("SVC5 generation was cancelled")
            self._active_by_device[device] = self._active_by_device.get(device, 0) + 1
            if request_id is not None:
                devices = self._active_request_ids.setdefault(request_id, {})
                devices[device] = devices.get(device, 0) + 1
        try:
            yield
        finally:
            with self._state_lock:
                self._active_by_device[device] -= 1
                if request_id is not None:
                    devices = self._active_request_ids.get(request_id)
                    if devices is not None:
                        devices[device] -= 1
                        if devices[device] <= 0:
                            devices.pop(device, None)
                        if not devices:
                            self._active_request_ids.pop(request_id, None)

    def generate(
        self,
        spec: ModelSpec,
        input_key: str,
        source_audio: Any,
        sample_rate: int,
        device: str,
        pitch_shift: int,
        cpu_bf16: bool = False,
        request_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        audio = mono_float_audio(source_audio)
        if not audio.size:
            raise ValueError("input audio is empty")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("input sample rate must be a positive integer")
        if isinstance(pitch_shift, bool) or not isinstance(pitch_shift, int):
            raise ValueError("Pitch Shift must be an integer")
        if not -36 <= pitch_shift <= 36:
            raise ValueError("Pitch Shift must be between -36 and 36")
        if not isinstance(cpu_bf16, bool):
            raise ValueError("CPU BF16 Autocast must be a boolean")
        cancelled = lambda: self._is_cancelled(request_id)
        feature_key = (spec.version, input_key, sample_rate, len(audio))
        with self._job(request_id, device):
            with self.model_cache.acquire(spec, device) as group:
                features = self.feature_cache.get(
                    feature_key,
                    lambda: group.prepare(
                        FeatureJob(audio, sample_rate), cancelled, request_id
                    ),
                    cancelled,
                )
                output = group.infer(
                    InferJob(features, pitch_shift, cpu_bf16 and device == "cpu"),
                    cancelled,
                    request_id,
                )
            if cancelled():
                raise GenerationCancelled("SVC5 generation was cancelled; output discarded")
        return output, spec.sample_rate

    def warm(self, spec: ModelSpec, device: str, workers: int = 1) -> dict:
        with self.model_cache.acquire(spec, device) as group:
            group.warm(workers)
        return {
            "character": spec.character,
            "version": spec.version,
            "device": device,
            "workers": min(group.limit, max(1, workers)),
        }

    def cancel(self, request_ids: Sequence[str]) -> dict:
        now = time.time()
        with self._state_lock:
            # Request IDs are UUID-like and bounded by the server. Keeping one hour
            # prevents a delayed queued task from escaping cancellation.
            self._cancelled = {
                key: timestamp
                for key, timestamp in self._cancelled.items()
                if now - timestamp < 3600
            }
            for request_id in request_ids:
                self._cancelled[request_id] = now
            active = sorted(
                request_id for request_id in request_ids if request_id in self._active_request_ids
            )
        self.model_cache.cancel(request_ids)
        return {"cancelled": sorted(set(request_ids)), "active": active}

    def commit_if_active(self, request_id: Optional[str], callback: Callable[[], Any]) -> Any:
        """Serialize the final cache commit against cancellation tombstones."""

        with self._state_lock:
            if request_id is not None and request_id in self._cancelled:
                raise GenerationCancelled("SVC5 generation was cancelled; output discarded")
            return callback()

    def unload(self, character: Optional[str], device: Optional[str]) -> dict:
        return self.model_cache.unload(character, device)

    def close(self) -> None:
        self.model_cache.close()

    def state(self) -> dict:
        cache = self.model_cache.snapshot()
        with self._state_lock:
            active_by_device = {
                device: count for device, count in self._active_by_device.items() if count
            }
            active_request_ids = sorted(self._active_request_ids)
        models = cache["loaded_models"]
        started = sum(model["workers"] for model in models)
        starting = sum(model["starting_workers"] for model in models)
        busy_workers = sum(model["busy_workers"] for model in models)
        queued = sum(model["queued_jobs"] for model in models)
        active = sum(active_by_device.values())
        devices = []
        known_devices = sorted(set(active_by_device) | {model["device"] for model in models})
        for device in known_devices:
            device_models = [model for model in models if model["device"] == device]
            devices.append(
                {
                    "device": device,
                    "active_jobs": active_by_device.get(device, 0),
                    "queued_jobs": sum(model["queued_jobs"] for model in device_models),
                    "workers": sum(model["workers"] for model in device_models),
                    "starting_workers": sum(
                        model["starting_workers"] for model in device_models
                    ),
                    "busy_workers": sum(model["busy_workers"] for model in device_models),
                }
            )
        busy = active > 0 or starting > 0 or busy_workers > 0
        loaded_devices = sorted({model["device"] for model in models if model["workers"]})
        current_device = (
            loaded_devices[0]
            if len(loaded_devices) == 1
            else ("multiple" if loaded_devices else None)
        )
        return {
            "status": "busy" if busy else ("warm-idle" if started else "ready-cold"),
            "device": current_device,
            "warm": started > 0,
            "busy": busy,
            "active_jobs": active,
            "queued_jobs": queued,
            "active_jobs_by_device": active_by_device,
            "active_request_ids": active_request_ids,
            "workers": started,
            "starting_workers": starting,
            "busy_workers": busy_workers,
            "loaded_models": [model["character"] for model in models if model["workers"]],
            "loaded_model_details": models,
            "devices": devices,
            "feature_cache": self.feature_cache.state(),
            **{key: value for key, value in cache.items() if key != "loaded_models"},
        }


def build_runtime(
    cpu_workers: Optional[int] = None,
    gpu_workers: Optional[int] = None,
    startup_concurrency: Optional[int] = None,
    idle_ttl_seconds: Optional[float] = None,
    feature_cache_entries: Optional[int] = None,
) -> SVC5Runtime:
    if cpu_workers is None:
        cpu_workers = int(os.environ.get("HAY_SAY_SVC5_CPU_WORKERS", "4"))
    if gpu_workers is None:
        gpu_workers = int(os.environ.get("HAY_SAY_SVC5_GPU_WORKERS", "1"))
    if startup_concurrency is None:
        startup_concurrency = int(
            os.environ.get("HAY_SAY_SVC5_STARTUP_CONCURRENCY", "4")
        )
    if idle_ttl_seconds is None:
        idle_ttl_seconds = float(os.environ.get("HAY_SAY_MODEL_IDLE_TTL_SECONDS", "1800"))
    if feature_cache_entries is None:
        feature_cache_entries = int(os.environ.get("HAY_SAY_SVC5_FEATURE_CACHE_ENTRIES", "8"))
    return SVC5Runtime(
        ModelCache(
            int(cpu_workers),
            int(gpu_workers),
            int(startup_concurrency),
            float(idle_ttl_seconds),
        ),
        FeatureCache(int(feature_cache_entries)),
    )

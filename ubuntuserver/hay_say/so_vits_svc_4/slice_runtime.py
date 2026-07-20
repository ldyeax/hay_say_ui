"""Rate-correct forced slicing and isolated parallel clip execution.

The upstream SVC4 command line script combines silence slicing, forced
slicing, model inference, and crossfading in one loop.  This module keeps the
forced-slice portion independent from Torch so it can be tested and reused by
the native server.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np


T = TypeVar("T")


def _sample_boundary(input_sample: int, input_sample_rate: int, output_sample_rate: int) -> int:
    """Map an absolute sample boundary without accumulating rounding error."""

    return int(round(input_sample * output_sample_rate / input_sample_rate))


def _validate_sample_rate(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ClipPlan:
    """One independently inferable forced clip on an absolute timeline."""

    index: int
    input_start: int
    input_stop: int
    new_input_start: int
    output_start: int
    output_stop: int
    new_output_start: int

    @property
    def input_samples(self) -> int:
        return self.input_stop - self.input_start

    @property
    def expected_output_samples(self) -> int:
        return self.output_stop - self.output_start

    @property
    def crossfade_input_samples(self) -> int:
        return self.new_input_start - self.input_start

    @property
    def crossfade_output_samples(self) -> int:
        return self.new_output_start - self.output_start

    def extract(self, audio: Sequence[float]) -> Any:
        """Return this clip from an array or another sliceable sequence."""

        return audio[self.input_start:self.input_stop]


def plan_forced_clips(
    total_input_samples: int,
    input_sample_rate: int,
    output_sample_rate: int,
    slice_seconds: float,
    crossfade_seconds: float,
) -> Tuple[ClipPlan, ...]:
    """Plan SVC4-style forced clips with rate-correct overlap boundaries.

    ``slice_seconds`` is the amount of new input consumed by each clip.  Every
    clip after the first also includes up to ``crossfade_seconds`` of preceding
    input.  The overlap therefore provides independent inference context and
    can be crossfaded after clips complete in any order.
    """

    input_sample_rate = _validate_sample_rate("input_sample_rate", input_sample_rate)
    output_sample_rate = _validate_sample_rate("output_sample_rate", output_sample_rate)
    if isinstance(total_input_samples, bool) or not isinstance(total_input_samples, int):
        raise ValueError("total_input_samples must be a non-negative integer")
    if total_input_samples < 0:
        raise ValueError("total_input_samples must be a non-negative integer")
    if not np.isfinite(slice_seconds) or slice_seconds < 0:
        raise ValueError("slice_seconds must be finite and non-negative")
    if not np.isfinite(crossfade_seconds) or crossfade_seconds < 0:
        raise ValueError("crossfade_seconds must be finite and non-negative")
    if slice_seconds == 0 and crossfade_seconds != 0:
        raise ValueError("crossfade_seconds requires forced slicing")
    if slice_seconds != 0 and crossfade_seconds > slice_seconds:
        raise ValueError("crossfade_seconds must not exceed slice_seconds")
    if total_input_samples == 0:
        return ()

    if slice_seconds == 0:
        slice_samples = total_input_samples
        crossfade_samples = 0
    else:
        slice_samples = int(round(slice_seconds * input_sample_rate))
        crossfade_samples = int(round(crossfade_seconds * input_sample_rate))
        if slice_samples <= 0:
            raise ValueError("slice_seconds is shorter than one input sample")

    clips = []
    for index, new_start in enumerate(range(0, total_input_samples, slice_samples)):
        input_start = max(0, new_start - crossfade_samples)
        input_stop = min(total_input_samples, new_start + slice_samples)
        output_start = _sample_boundary(input_start, input_sample_rate, output_sample_rate)
        output_stop = _sample_boundary(input_stop, input_sample_rate, output_sample_rate)
        new_output_start = _sample_boundary(new_start, input_sample_rate, output_sample_rate)
        clips.append(
            ClipPlan(
                index=index,
                input_start=input_start,
                input_stop=input_stop,
                new_input_start=new_start,
                output_start=output_start,
                output_stop=output_stop,
                new_output_start=new_output_start,
            )
        )
    return tuple(clips)


def _fit_mono_output(audio: Any, expected_samples: int) -> np.ndarray:
    output = np.asarray(audio)
    if output.ndim != 1:
        raise ValueError("clip inference output must be mono")
    difference = expected_samples - output.shape[0]
    if difference > 0:
        left = difference // 2
        output = np.pad(output, (left, difference - left))
    elif difference < 0:
        trim = -difference
        left = trim // 2
        output = output[left:output.shape[0] - (trim - left)]
    return output


def assemble_crossfaded(
    clips: Sequence[ClipPlan],
    outputs: Mapping[int, Any],
    retain_ratio: float = 1.0,
) -> np.ndarray:
    """Assemble indexed clip outputs in timeline order with a linear fade.

    ``retain_ratio`` mirrors SVC4's linear-gradient-retain setting.  Ratios
    below one discard the least reliable outer edges of each overlap and fade
    only its centered retained portion.
    """

    if not np.isfinite(retain_ratio) or retain_ratio <= 0 or retain_ratio > 1:
        raise ValueError("retain_ratio must be finite and in the range (0, 1]")

    if not clips:
        if outputs:
            raise ValueError("outputs were provided for an empty clip plan")
        return np.empty(0, dtype=np.float32)

    expected_indexes = {clip.index for clip in clips}
    if set(outputs) != expected_indexes:
        missing = sorted(expected_indexes - set(outputs))
        unexpected = sorted(set(outputs) - expected_indexes)
        raise ValueError(f"clip outputs do not match plan; missing={missing}, unexpected={unexpected}")

    ordered = sorted(clips, key=lambda clip: clip.index)
    if [clip.index for clip in ordered] != list(range(len(ordered))):
        raise ValueError("clip indexes must be contiguous and start at zero")

    assembled = _fit_mono_output(outputs[0], ordered[0].expected_output_samples)
    if ordered[0].output_start != 0:
        raise ValueError("the first clip must begin at output sample zero")

    for clip in ordered[1:]:
        current = _fit_mono_output(outputs[clip.index], clip.expected_output_samples)
        overlap = clip.crossfade_output_samples
        if assembled.shape[0] != clip.new_output_start:
            raise ValueError("clip plan is not contiguous on the output timeline")
        if overlap > assembled.shape[0] or overlap > current.shape[0]:
            raise ValueError("crossfade is longer than an adjacent clip output")
        if overlap:
            mix_dtype = np.result_type(assembled.dtype, current.dtype, np.float32)
            assembled = assembled.astype(mix_dtype, copy=False)
            current = current.astype(mix_dtype, copy=False)
            retained = int(overlap * retain_ratio)
            discarded = overlap - retained
            discard_left = discarded // 2
            discard_right = discarded - discard_left
            if retained == 1:
                incoming_weight = np.array([0.5], dtype=mix_dtype)
            elif retained:
                incoming_weight = np.linspace(0.0, 1.0, retained, dtype=mix_dtype)
            else:
                incoming_weight = np.empty(0, dtype=mix_dtype)
            outgoing_weight = 1.0 - incoming_weight
            old_start = assembled.shape[0] - retained - discard_right
            new_start = discard_left
            mixed = (
                assembled[old_start:old_start + retained] * outgoing_weight
                + current[new_start:new_start + retained] * incoming_weight
            )
            assembled = np.concatenate(
                (assembled[:old_start], mixed, current[new_start + retained:])
            )
        else:
            assembled = np.concatenate((assembled, current))
        if assembled.shape[0] != clip.output_stop:
            raise ValueError("assembled clip length does not match the output timeline")
    return assembled


def silence_for_input_length(
    input_samples: int,
    input_sample_rate: int,
    output_sample_rate: int,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Create a target-rate silence segment matching an input-rate duration."""

    input_sample_rate = _validate_sample_rate("input_sample_rate", input_sample_rate)
    output_sample_rate = _validate_sample_rate("output_sample_rate", output_sample_rate)
    if isinstance(input_samples, bool) or not isinstance(input_samples, int) or input_samples < 0:
        raise ValueError("input_samples must be a non-negative integer")
    length = _sample_boundary(input_samples, input_sample_rate, output_sample_rate)
    return np.zeros(length, dtype=dtype)


@dataclass(frozen=True)
class WorkerSpec:
    """Factory for one model object that may run at most one clip at a time."""

    name: str
    device: str
    factory: Callable[[], Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class IndexedResult(Generic[T]):
    index: int
    value: T
    worker_name: str
    device: str


def build_worker_specs(
    cpu_workers: int,
    cpu_factory: Callable[[int], Any],
    gpu_factory: Optional[Callable[[], Any]] = None,
) -> Tuple[WorkerSpec, ...]:
    """Build configurable CPU worker specs and, optionally, one GPU spec."""

    if isinstance(cpu_workers, bool) or not isinstance(cpu_workers, int) or cpu_workers < 0:
        raise ValueError("cpu_workers must be a non-negative integer")
    specs = []
    for ordinal in range(cpu_workers):
        specs.append(
            WorkerSpec(
                name=f"cpu-{ordinal}",
                device="cpu",
                factory=lambda ordinal=ordinal: cpu_factory(ordinal),
            )
        )
    if gpu_factory is not None:
        specs.append(WorkerSpec(name="gpu-0", device="cuda", factory=gpu_factory))
    if not specs:
        raise ValueError("at least one CPU or GPU worker is required")
    return tuple(specs)


@dataclass
class _WorkerSlot:
    spec: WorkerSpec
    worker: Any
    in_use: threading.Lock = field(default_factory=threading.Lock)


class IsolatedWorkerPool:
    """Run indexed clips concurrently without sharing a live model instance."""

    def __init__(self, worker_specs: Iterable[WorkerSpec], startup_concurrency: int = 4):
        self._specs = tuple(worker_specs)
        if not self._specs:
            raise ValueError("at least one worker spec is required")
        if len({spec.name for spec in self._specs}) != len(self._specs):
            raise ValueError("worker names must be unique")
        gpu_workers = [spec for spec in self._specs if spec.device.startswith("cuda")]
        if len(gpu_workers) > 1:
            raise ValueError("SVC4 supports at most one GPU worker per pool")
        if (
            isinstance(startup_concurrency, bool)
            or not isinstance(startup_concurrency, int)
            or startup_concurrency < 1
        ):
            raise ValueError("startup_concurrency must be a positive integer")
        self._startup_concurrency = startup_concurrency
        self._slots: Optional[Tuple[_WorkerSlot, ...]] = None
        self._available: Optional[queue.Queue[_WorkerSlot]] = None
        self._map_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()

    def _create_slots(self, specs: Sequence[WorkerSpec]) -> Tuple[_WorkerSlot, ...]:
        if len(specs) == 1:
            spec = specs[0]
            return (_WorkerSlot(spec=spec, worker=spec.factory()),)

        created: list[Optional[_WorkerSlot]] = [None] * len(specs)
        failures = []

        def create(index: int, spec: WorkerSpec) -> Tuple[int, _WorkerSlot]:
            return index, _WorkerSlot(spec=spec, worker=spec.factory())

        concurrency = min(self._startup_concurrency, len(specs))
        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="svc4-start",
        ) as executor:
            futures = {
                executor.submit(create, index, spec): index
                for index, spec in enumerate(specs)
            }
            for future in as_completed(futures):
                try:
                    index, slot = future.result()
                    created[index] = slot
                except BaseException as exc:
                    failures.append((futures[future], exc))

        completed = tuple(slot for slot in created if slot is not None)
        if failures:
            try:
                self._close_slots(completed)
            except BaseException:
                pass
            raise min(failures, key=lambda failure: failure[0])[1]
        return completed

    def start(self, max_workers: Optional[int] = None) -> "IsolatedWorkerPool":
        """Start up to ``max_workers`` lanes, growing an existing pool lazily."""

        if max_workers is None:
            max_workers = len(self._specs)
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise ValueError("max_workers must be a positive integer")
        if max_workers < 1 or max_workers > len(self._specs):
            raise ValueError(
                "max_workers must be between 1 and {}".format(len(self._specs))
            )
        with self._lifecycle_lock:
            existing = self._slots or ()
            if len(existing) >= max_workers:
                return self
            slots = list(existing)
            created = self._create_slots(self._specs[len(existing):max_workers])
            slots.extend(created)
            available = self._available or queue.Queue()
            for slot in created:
                available.put(slot)
            self._slots = tuple(slots)
            self._available = available
        return self

    @property
    def started_workers(self) -> int:
        with self._lifecycle_lock:
            return len(self._slots or ())

    def state(self) -> Tuple[Mapping[str, Any], ...]:
        """Return lightweight lane telemetry without exposing worker objects."""

        with self._lifecycle_lock:
            slots = tuple(self._slots or ())
        return tuple(
            {
                "name": slot.spec.name,
                "device": slot.spec.device,
                "pid": getattr(slot.worker, "pid", None),
            }
            for slot in slots
        )

    @staticmethod
    def _close_slots(slots: Iterable[_WorkerSlot]) -> None:
        first_error = None
        for slot in slots:
            close = getattr(slot.worker, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:  # close every worker before reporting one failure
                    if first_error is None:
                        first_error = exc
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        with self._map_lock:
            with self._lifecycle_lock:
                if self._slots is None:
                    return
                slots = self._slots
                self._slots = None
                self._available = None
            self._close_slots(slots)

    def __enter__(self) -> "IsolatedWorkerPool":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.close()
        except BaseException:
            if exc_type is None:
                raise

    def map_indexed(
        self,
        jobs: Iterable[Any],
        infer: Callable[[Any, Any], T],
        max_workers: Optional[int] = None,
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> Tuple[IndexedResult[T], ...]:
        """Run jobs concurrently and return results sorted by ``job.index``."""

        job_list = tuple(jobs)
        indexes = [getattr(job, "index", None) for job in job_list]
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indexes):
            raise ValueError("every job must have an integer index attribute")
        if len(set(indexes)) != len(indexes):
            raise ValueError("job indexes must be unique")
        if not job_list:
            return ()

        if max_workers is None:
            max_workers = min(len(job_list), len(self._specs))
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise ValueError("max_workers must be a positive integer")
        if max_workers < 1 or max_workers > len(self._specs):
            raise ValueError(
                "max_workers must be between 1 and {}".format(len(self._specs))
            )

        with self._map_lock:
            if cancel_check is not None:
                cancel_check()
            self.start(max_workers)
            assert self._slots is not None
            assert self._available is not None
            available = self._available

            def execute(job: Any) -> IndexedResult[T]:
                if cancel_check is not None:
                    cancel_check()
                while True:
                    if cancel_check is not None:
                        cancel_check()
                    try:
                        slot = available.get(timeout=0.05)
                        break
                    except queue.Empty:
                        continue
                acquired = slot.in_use.acquire(blocking=False)
                if not acquired:  # The queue is the isolation mechanism; retain a hard invariant check.
                    available.put(slot)
                    raise RuntimeError(f"worker {slot.spec.name} was scheduled concurrently")
                try:
                    if cancel_check is not None:
                        cancel_check()
                    value = infer(slot.worker, job)
                    if cancel_check is not None:
                        cancel_check()
                    return IndexedResult(job.index, value, slot.spec.name, slot.spec.device)
                finally:
                    slot.in_use.release()
                    available.put(slot)

            concurrency = min(max_workers, len(job_list))
            with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="svc4-clip") as executor:
                futures = [executor.submit(execute, job) for job in job_list]
                completed = [future.result() for future in as_completed(futures)]
        return tuple(sorted(completed, key=lambda result: result.index))

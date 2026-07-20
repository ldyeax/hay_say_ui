import base64
import datetime
import hashlib
import json
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from http.client import HTTPConnection
from pathlib import Path

from hay_say_common.cache import Stage

import hay_say_common as hsc
import generation_jobs
import inference_scheduler
import plotly_celery_common as pcc
from pitch_batch import parse_pitch_spec
import runtime_client


_GPU_CAPABILITY_CACHE = {}


class GenerationCancelled(RuntimeError):
    pass


class GenerationRequestUnavailable(RuntimeError):
    """Raised when a stale or duplicate background task cannot own a job."""


def _failure_message(error):
    detail = str(error).strip()
    message = type(error).__name__ if not detail else f'{type(error).__name__}: {detail}'
    return message[:4096]


GENERATION_SNAPSHOT_FIELDS = (
    'user_text',
    'selected_file',
    'semitone_pitch',
    'debug_pitch',
    'reduce_noise',
    'crop_silence',
    'reduce_metallic_noise',
    'auto_tune_output',
    'output_speed_adjustment',
    'pitch_batch_enabled',
    'pitch_batch_values',
)


def generate_and_prepare_postprocessed_display(request_data, message, cache_type, gpu_id,
                                               selected_architectures):
    """Claim and execute a self-contained request persisted by the browser."""
    if not isinstance(request_data, dict):
        raise GenerationRequestUnavailable('Generation request data is required')
    request_id = request_data.get('request_id')
    client_id = request_data.get('client_id')
    session_data = {'id': request_data.get('session_id'), 'client_id': client_id}
    try:
        from celery import current_task

        task_id = getattr(current_task.request, 'id', None)
    except (ImportError, RuntimeError, AttributeError):
        task_id = None

    claimed = generation_jobs.claim_running(client_id, request_id, task_id or request_id)
    if claimed is None:
        raise GenerationRequestUnavailable('Generation request is no longer queued')

    def set_progress(progress_message, *, operation_id=None, operation_label=None, current=None, total=None,
                     status='running', device=None):
        if operation_id is None:
            generation_jobs.update_progress(client_id, request_id, str(progress_message))
            return
        generation_jobs.update_operation_progress(
            client_id,
            request_id,
            operation_id,
            operation_label or operation_id,
            current,
            total,
            status=status,
            message=str(progress_message),
            device=device,
        )

    tracked_session = dict(session_data)
    tracked_session['request_id'] = request_id
    try:
        snapshot = request_data.get('snapshot')
        if not isinstance(snapshot, dict):
            raise ValueError('Generation request snapshot is missing')
        missing_fields = [field for field in GENERATION_SNAPSHOT_FIELDS if field not in snapshot]
        if missing_fields:
            raise ValueError(f'Generation request snapshot is missing {", ".join(missing_fields)}')
        hidden_states = snapshot.get('hidden_states')
        architecture_inputs = snapshot.get('architecture_inputs')
        if not isinstance(hidden_states, list) or not isinstance(architecture_inputs, list):
            raise ValueError('Generation request architecture inputs are invalid')
        args = tuple(hidden_states + architecture_inputs)
        selected_tab = get_selected_tab_object(selected_architectures, args[0:len(selected_architectures)])
        if claimed.get('runtime_id') != selected_tab.id:
            raise ValueError('Generation request architecture does not match the queued job')
        if generation_jobs.is_cancel_requested(client_id, request_id):
            raise GenerationCancelled('Generation was stopped before it started')
        set_progress(message)
        output_hashes = generate(
            cache_type, gpu_id, tracked_session, selected_architectures, snapshot['user_text'],
            snapshot['selected_file'], snapshot['semitone_pitch'], snapshot['debug_pitch'],
            snapshot['reduce_noise'], snapshot['crop_silence'], snapshot['reduce_metallic_noise'],
            snapshot['auto_tune_output'], snapshot['output_speed_adjustment'],
            snapshot['pitch_batch_enabled'], snapshot['pitch_batch_values'],
            args, set_progress,
        )
        if generation_jobs.is_cancel_requested(client_id, request_id):
            raise GenerationCancelled('Generation was stopped')
    except GenerationCancelled:
        generation_jobs.mark_cancelled(client_id, request_id)
        raise
    except Exception as error:
        if generation_jobs.is_cancel_requested(client_id, request_id):
            generation_jobs.mark_cancelled(client_id, request_id)
            raise GenerationCancelled('Generation was stopped') from error
        failed = generation_jobs.mark_failed(client_id, request_id, _failure_message(error))
        if isinstance(failed, dict) and failed.get('status') in generation_jobs.CANCEL_STATUSES:
            generation_jobs.mark_cancelled(client_id, request_id)
            raise GenerationCancelled('Generation was stopped') from error
        raise
    completed = generation_jobs.mark_completed(client_id, request_id)
    if completed is None or completed.get('status') != 'completed':
        if (
            isinstance(completed, dict)
            and completed.get('status') in generation_jobs.CANCEL_STATUSES
        ) or generation_jobs.is_cancel_requested(client_id, request_id):
            generation_jobs.mark_cancelled(client_id, request_id)
            raise GenerationCancelled('Generation was stopped')
        raise GenerationRequestUnavailable('Generation request changed before completion')
    return {'request_id': request_id, 'runtime_id': selected_tab.id, 'outputs': output_hashes}, 'Generate!'


def generate(cache_type, gpu_id, session_data, selected_architectures, user_text, selected_file, semitone_pitch,
             debug_pitch, reduce_noise, crop_silence, reduce_metallic_noise, auto_tune_output, output_speed_adjustment,
             pitch_batch_enabled, pitch_batch_values, args, set_progress=None):
    if inference_scheduler.is_auto_device(gpu_id):
        device_description = 'the first available CPU/GPU slot'
    else:
        device_description = inference_scheduler.device_label(gpu_id)
    print('generating on ' + device_description, flush=True)
    cache = hsc.select_cache_implementation(cache_type)
    selected_tab_object = get_selected_tab_object(selected_architectures, args[0:len(selected_architectures)])
    relevant_inputs = get_inputs_for_selected_tab(selected_architectures, selected_tab_object,
                                                  args[len(selected_architectures):])
    _raise_if_cancelled(session_data)
    hash_preprocessed = preprocess_if_needed(cache, selected_file, semitone_pitch, debug_pitch, reduce_noise,
                                             crop_silence, session_data)
    _raise_if_cancelled(session_data)
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
    if set_progress:
        noun = 'output' if len(hash_outputs) == 1 else 'outputs'
        set_progress(f'Post-processing {len(hash_outputs)} {noun} from {selected_tab_object.label}...')
    hash_postprocessed = []
    for hash_output in hash_outputs:
        _raise_if_cancelled(session_data)
        processed = postprocess(
            cache,
            hash_output,
            reduce_metallic_noise,
            auto_tune_output,
            output_speed_adjustment,
            session_data,
        )
        _raise_if_cancelled(session_data)
        hash_postprocessed.append(processed)
    _raise_if_cancelled(session_data)
    return hash_postprocessed


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
    cpu_precision_policy_identity = _backend_cpu_precision_policy_identity(tab_object.id)
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
            cpu_precision_policy_identity,
            generation_nonce,
        )
        variants.append((pitch, variant_options, output_hash))

    batch_id = pcc.compute_next_hash(*(variant[2] for variant in variants))
    batch_manifest = None
    if len(variants) > 1:
        batch_manifest = {
            'Output IDs': [variant[2] for variant in variants],
            'Pitches': [variant[0] for variant in variants],
        }
    uncached = _uncached_variants(cache, variants, session_data)

    if uncached:
        _raise_if_cancelled(session_data)
        runtime_client.ensure_runtime_started(tab_object.id, tab_object.port)
        _raise_if_cancelled(session_data)
    host, port = resolve_service_endpoint(tab_object)
    output_ids = [variant[2] for variant in variants]
    cancel_check = lambda: _raise_if_cancelled(session_data)
    with runtime_client.output_locks(
        tab_object.id,
        session_data['id'],
        output_ids,
        cancel_check=cancel_check,
    ):
        runtime_guard = nullcontext() if getattr(tab_object, 'supports_parallel_requests', False) \
            else runtime_client.generation_lock(tab_object.id, cancel_check=cancel_check)
        with runtime_guard:
            _raise_if_cancelled(session_data)
            uncached = _uncached_variants(cache, variants, session_data)
            if uncached:
                _dispatch_uncached_variants(
                    user_text,
                    hash_preprocessed,
                    tab_object,
                    uncached,
                    session_data,
                    gpu_id,
                    host,
                    port,
                    set_progress,
                )

            for pitch, variant_options, output_hash in variants:
                _raise_if_cancelled(session_data)
                verify_output_exists(cache, output_hash, session_data)
                _raise_if_cancelled(session_data)
                _commit_if_active(
                    session_data,
                    lambda: write_output_metadata(
                        cache,
                        hash_preprocessed,
                        user_text,
                        output_hash,
                        variant_options,
                        session_data,
                        batch_id=batch_id if len(variants) > 1 else None,
                        batch_manifest=batch_manifest,
                        pitch=pitch,
                        model_identity=model_identity,
                    ),
                )
    _raise_if_cancelled(session_data)
    return [variant[2] for variant in variants]


def _dispatch_uncached_variants(user_text, hash_preprocessed, tab_object, variants, session_data, requested_device,
                                host, port, set_progress):
    operation_details = _initialize_variant_operations(variants, set_progress)
    allow_auto_gpu = _auto_gpu_supported(tab_object) if inference_scheduler.is_auto_device(requested_device) else True
    serial_device_key = tab_object.id if getattr(tab_object, 'serializes_device_requests', False) else None
    mixed_minimum = _positive_environment_int('HAY_SAY_MIXED_PITCH_MIN_VARIANTS', 3)
    can_mix = (
        inference_scheduler.is_auto_device(requested_device)
        and allow_auto_gpu
        and getattr(tab_object, 'supports_mixed_device_pitch_batch', False)
        and len(variants) >= mixed_minimum
    )
    if can_mix:
        with inference_scheduler.mixed_inference_reservations(
            allow_gpu=allow_auto_gpu,
            serial_device_key=serial_device_key,
            cancel_check=lambda: _raise_if_cancelled(session_data),
        ) as (cpu_reservation, gpu_reservation):
            if cpu_reservation is not None and gpu_reservation is not None:
                work_queue = _VariantWorkQueue(variants, balance_cpu_refills=True)
                gpu_seeds = _seed_variant_lanes(
                    work_queue,
                    _device_pitch_request_workers(tab_object, gpu_reservation.device, len(variants)),
                    _device_pitch_claim_size(tab_object, gpu_reservation.device),
                )
                cpu_seeds = _seed_variant_lanes(
                    work_queue,
                    _device_pitch_request_workers(tab_object, cpu_reservation.device, len(variants)),
                    _device_pitch_claim_size(tab_object, cpu_reservation.device),
                    reserve=max(1, len(variants) // 2) if len(variants) > 3 else 0,
                )
                if set_progress:
                    set_progress(
                        f'Generating {len(variants)} pitch variants with {tab_object.label} on '
                        f'{inference_scheduler.device_label(gpu_reservation.device)} + CPU...'
                    )
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix='hay-say-pitch') as executor:
                    futures = [
                        executor.submit(
                            _drain_variant_queue_group_with_reservation,
                            gpu_reservation,
                            work_queue,
                            gpu_seeds,
                            user_text,
                            hash_preprocessed,
                            tab_object,
                            session_data,
                            host,
                            port,
                            set_progress,
                            operation_details,
                        ),
                        executor.submit(
                            _drain_variant_queue_group_with_reservation,
                            cpu_reservation,
                            work_queue,
                            cpu_seeds,
                            user_text,
                            hash_preprocessed,
                            tab_object,
                            session_data,
                            host,
                            port,
                            set_progress,
                            operation_details,
                        ),
                    ]
                    first_error = None
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as error:
                            work_queue.abort()
                            if first_error is None:
                                first_error = error
                    if first_error is not None:
                        raise first_error
                return

            selected_reservation = gpu_reservation if gpu_reservation is not None else cpu_reservation
            with selected_reservation as selected_device:
                _send_variants_with_parallel_workers(
                    user_text,
                    hash_preprocessed,
                    tab_object,
                    variants,
                    session_data,
                    selected_device,
                    host,
                    port,
                    set_progress,
                    operation_details,
                )
            return

    with inference_scheduler.inference_device(
        requested_device,
        allow_gpu=allow_auto_gpu,
        serial_device_key=serial_device_key,
        cancel_check=lambda: _raise_if_cancelled(session_data),
    ) as selected_device:
        _send_variants_with_parallel_workers(
            user_text,
            hash_preprocessed,
            tab_object,
            variants,
            session_data,
            selected_device,
            host,
            port,
            set_progress,
            operation_details,
        )


class _VariantWorkQueue:
    """A request-local queue whose device lanes claim work only when ready."""

    def __init__(self, variants, balance_cpu_refills=False):
        self._pending = deque(variants)
        self._lock = threading.Lock()
        self._aborted = False
        self._balance_cpu_refills = bool(balance_cpu_refills)

    def claim(self, limit, reserve=0):
        if limit < 1:
            raise ValueError('Variant claim limit must be positive')
        if reserve < 0:
            raise ValueError('Variant reserve must be non-negative')
        with self._lock:
            return self._claim_locked(limit, reserve)

    def claim_for_device(self, limit, selected_device):
        if limit < 1:
            raise ValueError('Variant claim limit must be positive')
        with self._lock:
            reserve = 0
            if self._balance_cpu_refills and selected_device == inference_scheduler.CPU_DEVICE:
                reserve = (len(self._pending) + 1) // 2
            return self._claim_locked(limit, reserve)

    def _claim_locked(self, limit, reserve):
        if self._aborted:
            return ()
        claimed = []
        while len(self._pending) > reserve and len(claimed) < limit:
            claimed.append(self._pending.popleft())
        return tuple(claimed)

    def abort(self):
        with self._lock:
            self._aborted = True


def _seed_variant_lanes(work_queue, lane_count, claim_size, reserve=0):
    seeds = []
    for _ in range(lane_count):
        claimed = work_queue.claim(claim_size, reserve=reserve)
        if not claimed:
            break
        seeds.append(claimed)
    return tuple(seeds)


def _drain_variant_queue_group_with_reservation(reservation, work_queue, initial_claims, user_text,
                                                hash_preprocessed, tab_object, session_data, host, port,
                                                set_progress, operation_details):
    with reservation as selected_device:
        _drain_variant_queue_group(
            selected_device,
            work_queue,
            initial_claims,
            user_text,
            hash_preprocessed,
            tab_object,
            session_data,
            host,
            port,
            set_progress,
            operation_details,
        )


def _drain_variant_queue_group(selected_device, work_queue, initial_claims, user_text, hash_preprocessed,
                               tab_object, session_data, host, port, set_progress, operation_details):
    if not initial_claims:
        return
    if len(initial_claims) == 1:
        _drain_variant_queue(
            selected_device,
            work_queue,
            initial_claims[0],
            user_text,
            hash_preprocessed,
            tab_object,
            session_data,
            host,
            port,
            set_progress,
            operation_details,
        )
        return
    with ThreadPoolExecutor(max_workers=len(initial_claims), thread_name_prefix='hay-say-pitch-lane') as executor:
        futures = [
            executor.submit(
                _drain_variant_queue,
                selected_device,
                work_queue,
                initial_variants,
                user_text,
                hash_preprocessed,
                tab_object,
                session_data,
                host,
                port,
                set_progress,
                operation_details,
            )
            for initial_variants in initial_claims
        ]
        first_error = None
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                work_queue.abort()
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def _drain_variant_queue(selected_device, work_queue, initial_variants, user_text, hash_preprocessed,
                         tab_object, session_data, host, port, set_progress, operation_details):
    claim_size = _device_pitch_claim_size(tab_object, selected_device)
    claimed = initial_variants
    while claimed:
        try:
            _send_claimed_variants(
                user_text,
                hash_preprocessed,
                tab_object,
                claimed,
                session_data,
                selected_device,
                host,
                port,
                set_progress,
                operation_details,
            )
        except Exception:
            work_queue.abort()
            raise
        _raise_if_cancelled(session_data)
        claimed = work_queue.claim_for_device(claim_size, selected_device)


def _send_variants_with_parallel_workers(user_text, hash_preprocessed, tab_object, variants, session_data,
                                         selected_device, host, port, set_progress, operation_details):
    lane_count = _device_pitch_request_workers(tab_object, selected_device, len(variants))
    claim_size = _device_pitch_claim_size(tab_object, selected_device)
    chunk_single_lane = (
        selected_device == inference_scheduler.CPU_DEVICE
        and getattr(tab_object, 'supports_native_pitch_batch', False)
        and len(variants) > claim_size
    )
    if lane_count == 1 and not chunk_single_lane:
        _send_claimed_variants(
            user_text,
            hash_preprocessed,
            tab_object,
            variants,
            session_data,
            selected_device,
            host,
            port,
            set_progress,
            operation_details,
        )
        return
    work_queue = _VariantWorkQueue(variants)
    initial_claims = _seed_variant_lanes(
        work_queue,
        lane_count,
        claim_size,
    )
    _drain_variant_queue_group(
        selected_device,
        work_queue,
        initial_claims,
        user_text,
        hash_preprocessed,
        tab_object,
        session_data,
        host,
        port,
        set_progress,
        operation_details,
    )


def _device_pitch_request_workers(tab_object, selected_device, variant_count):
    if not getattr(tab_object, 'supports_parallel_requests', False):
        return 1
    worker_selector = getattr(tab_object, 'pitch_batch_request_workers', None)
    worker_count = worker_selector(selected_device) if callable(worker_selector) else 1
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise ValueError('pitch_batch_request_workers must return a positive integer')
    return min(worker_count, max(1, variant_count))


def _device_pitch_claim_size(tab_object, selected_device):
    if not getattr(tab_object, 'supports_native_pitch_batch', False):
        return 1
    if selected_device == inference_scheduler.CPU_DEVICE:
        active_cpu_lanes = _positive_environment_int('HAY_SAY_SVC3_CPU_PITCH_WORKERS', 5)
        configured_batch_limit = _positive_environment_int(
            'HAY_SAY_AUTO_CPU_PITCH_VARIANTS',
            min(4, active_cpu_lanes),
        )
        return min(configured_batch_limit, active_cpu_lanes)
    return _positive_environment_int('HAY_SAY_AUTO_GPU_PITCH_VARIANTS', 1)


def _send_claimed_variants(user_text, hash_preprocessed, tab_object, variants, session_data, selected_device, host,
                           port, set_progress, operation_details):
    if not variants:
        return
    if len(variants) > 1 and not getattr(tab_object, 'supports_native_pitch_batch', False):
        for variant in variants:
            _send_claimed_variants(
                user_text,
                hash_preprocessed,
                tab_object,
                (variant,),
                session_data,
                selected_device,
                host,
                port,
                set_progress,
                operation_details,
            )
        return
    device = inference_scheduler.device_label(selected_device)
    for variant in variants:
        _update_variant_operation(
            set_progress,
            variant,
            status='running',
            current=0,
            device=device,
            message=f'Generating pitch {variant[0]} with {tab_object.label} on {device}...',
            operation_details=operation_details,
        )
    try:
        _send_variants(
            user_text,
            hash_preprocessed,
            tab_object,
            variants,
            session_data,
            selected_device,
            host,
            port,
            None,
        )
    except Exception as error:
        cancellation = error if isinstance(error, GenerationCancelled) else None
        if cancellation is None:
            try:
                _raise_if_cancelled(session_data)
            except GenerationCancelled as cancelled:
                cancellation = cancelled
        status = 'cancelled' if cancellation is not None else 'failed'
        for variant in variants:
            _update_variant_operation(
                set_progress,
                variant,
                status=status,
                current=0,
                device=device,
                message=f'Pitch {variant[0]} {status} on {device}.',
                operation_details=operation_details,
            )
        if cancellation is not None and cancellation is not error:
            raise cancellation from error
        raise
    for variant in variants:
        _update_variant_operation(
            set_progress,
            variant,
            status='completed',
            current=1,
            device=device,
            message=f'Completed pitch {variant[0]} with {tab_object.label} on {device}.',
            operation_details=operation_details,
        )


def _initialize_variant_operations(variants, set_progress):
    operation_details = {
        variant[2]: (f'pitch:{index:03d}:{variant[2]}', index)
        for index, variant in enumerate(variants)
    }
    for variant in variants:
        _update_variant_operation(
            set_progress,
            variant,
            status='pending',
            current=0,
            device=None,
            message=f'Pitch {variant[0]} is waiting for an inference device.',
            operation_details=operation_details,
        )
    return operation_details


def _update_variant_operation(set_progress, variant, *, status, current, device, message, operation_details):
    if set_progress is None:
        return
    pitch, _, output_hash = variant
    operation_id, _ = operation_details[output_hash]
    label = 'Generation' if pitch is None else f'Pitch {int(pitch):+d}'
    set_progress(
        message,
        operation_id=operation_id,
        operation_label=label,
        current=current,
        total=1,
        status=status,
        device=device,
    )


def _send_variants(user_text, hash_preprocessed, tab_object, variants, session_data, selected_device, host, port,
                   set_progress):
    _raise_if_cancelled(session_data)
    if len(variants) > 1 and tab_object.supports_native_pitch_batch:
        batch_options = dict(variants[0][1])
        batch_options['Pitch Shifts'] = [variant[0] for variant in variants]
        payload = construct_payload(
            user_text,
            hash_preprocessed,
            batch_options,
            variants[0][2],
            session_data,
            selected_device,
        )
        payload['Output Files'] = [variant[2] for variant in variants]
        if set_progress:
            set_progress(
                f'Generating {len(variants)} pitch variants with {tab_object.label} on '
                f'{inference_scheduler.device_label(selected_device)}...'
            )
        send_payload(payload, host, port)
        _raise_if_cancelled(session_data)
        return

    for index, (_, variant_options, output_hash) in enumerate(variants, start=1):
        _raise_if_cancelled(session_data)
        if set_progress:
            set_progress(
                f'Generating variant {index} of {len(variants)} with {tab_object.label} on '
                f'{inference_scheduler.device_label(selected_device)}...'
            )
        payload = construct_payload(
            user_text,
            hash_preprocessed,
            variant_options,
            output_hash,
            session_data,
            selected_device,
        )
        send_payload(payload, host, port)
        _raise_if_cancelled(session_data)


def _raise_if_cancelled(session_data):
    client_id = session_data.get('client_id') if isinstance(session_data, dict) else None
    request_id = session_data.get('request_id') if isinstance(session_data, dict) else None
    if client_id and request_id and generation_jobs.is_cancel_requested(client_id, request_id):
        raise GenerationCancelled('Generation was stopped')


def _commit_if_active(session_data, callback):
    client_id = session_data.get('client_id') if isinstance(session_data, dict) else None
    request_id = session_data.get('request_id') if isinstance(session_data, dict) else None
    if not client_id or not request_id:
        return callback()
    committed, value = generation_jobs.commit_if_active(client_id, request_id, callback)
    if not committed:
        raise GenerationCancelled('Generation was stopped before committing output')
    return value


def _positive_environment_int(name, default):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _auto_gpu_supported(tab_object):
    """Cache stable model capability while live GPU health remains scheduler-owned."""
    now = time.monotonic()
    ttl = float(os.environ.get('HAY_SAY_GPU_CAPABILITY_TTL', '300'))
    if ttl < 0:
        raise ValueError('HAY_SAY_GPU_CAPABILITY_TTL must be non-negative')
    cached = _GPU_CAPABILITY_CACHE.get(tab_object.id)
    if cached is not None and now - cached[0] <= ttl:
        return cached[1]
    available = bool(tab_object.is_gpu_available)
    _GPU_CAPABILITY_CACHE[tab_object.id] = (now, available)
    return available


def _uncached_variants(cache, variants, session_data):
    return [variant for variant in variants
            if not cache.file_is_already_cached(Stage.OUTPUT, session_data['id'], variant[2])]


def _backend_cpu_precision_policy_identity(runtime_id):
    """Keep backend precision changes from reusing outputs made under the old policy."""
    if runtime_id not in hsc.server_utility.MODEL_CPU_BF16_ENVIRONMENTS:
        return 'cpu-fp32'
    enabled = hsc.model_cpu_bf16_enabled(runtime_id)
    return 'cpu-bf16-amx' if enabled else 'cpu-fp32'


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
        digest.update(f'\0{stat.st_dev}\0{stat.st_ino}\0{stat.st_size}\0{stat.st_mtime_ns}\0'.encode('ascii'))
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
    payload = {
        'Inputs': {
            'User Text': user_text,
            'User Audio': hash_preprocessed
        },
        'Options': options,
        'Output File': hash_output,
        'GPU ID': gpu_id,
        'Session ID': session_data['id']
    }
    request_id = session_data.get('request_id') if isinstance(session_data, dict) else None
    if request_id:
        payload['Request ID'] = request_id
    return payload


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
                          batch_id=None, batch_manifest=None, pitch=None, model_identity=None):
    entry = {
        'Inputs': {
            'Preprocessed File': hash_preprocessed,
            'User Text': user_text
        },
        'Options': options,
        'Batch ID': batch_id,
        'Pitch Batches': {},
        'Pitch Variant': pitch,
        'Model Identity': model_identity,
        'Time of Creation': datetime.datetime.now().strftime(hsc.cache.TIMESTAMP_FORMAT)
    }
    def update(metadata):
        existing = metadata.get(hash_output) or {}
        memberships = existing.get('Pitch Batches')
        if isinstance(memberships, dict):
            entry['Pitch Batches'] = dict(memberships)
        if batch_id and isinstance(batch_manifest, dict):
            entry['Pitch Batches'][batch_id] = batch_manifest
        metadata[hash_output] = entry

    cache.update_metadata(Stage.OUTPUT, session_data['id'], update)


def postprocess(cache, hash_output, reduce_metallic_noise, auto_tune_output, output_speed_adjustment, session_data):
    _raise_if_cancelled(session_data)
    # Convert data types to something more digestible
    reduce_metallic_noise, auto_tune_output = pcc.convert_to_bools(reduce_metallic_noise, auto_tune_output)
    output_speed_adjustment = float(
        output_speed_adjustment)  # Dash's Range Input supplies a string, so cast to float

    # Check whether the postprocessed file already exists
    hash_postprocessed = pcc.compute_next_hash(hash_output, reduce_metallic_noise, auto_tune_output,
                                           output_speed_adjustment)
    if cache.file_is_already_cached(Stage.POSTPROCESSED, session_data['id'], hash_postprocessed):
        _raise_if_cancelled(session_data)
        # Refresh the full entry so cached outputs gain the current pitch-batch lineage as well as a new timestamp.
        _commit_if_active(
            session_data,
            lambda: write_postprocessed_metadata(
                cache,
                hash_output,
                hash_postprocessed,
                reduce_metallic_noise,
                auto_tune_output,
                output_speed_adjustment,
                session_data,
            ),
        )
        _raise_if_cancelled(session_data)
        return hash_postprocessed

    # Perform postprocessing
    data_output, sr_output = cache.read_audio_from_cache(Stage.OUTPUT, session_data['id'], hash_output)
    _raise_if_cancelled(session_data)
    data_postprocessed, sr_postprocessed = postprocess_bytes(data_output, sr_output, reduce_metallic_noise,
                                                             auto_tune_output, output_speed_adjustment)

    def commit_postprocessed_output():
        cache.save_audio_to_cache(
            Stage.POSTPROCESSED,
            session_data['id'],
            hash_postprocessed,
            data_postprocessed,
            sr_postprocessed,
        )
        write_postprocessed_metadata(
            cache,
            hash_output,
            hash_postprocessed,
            reduce_metallic_noise,
            auto_tune_output,
            output_speed_adjustment,
            session_data,
        )

    _raise_if_cancelled(session_data)
    _commit_if_active(session_data, commit_postprocessed_output)

    _raise_if_cancelled(session_data)
    return hash_postprocessed


def postprocess_bytes(bytes_output, sr_output, reduce_metallic_noise, auto_tune_output, output_speed_adjustment):
    # todo: implement this
    return bytes_output, sr_output


def write_postprocessed_metadata(cache, hash_output, hash_postprocessed, reduce_metallic_noise, auto_tune_output,
                                 output_speed_adjustment, session_data):
    processing_options, user_text, hash_preprocessed, batch_id, pitch_batches, pitch, model_identity = get_process_info(
        cache,
        hash_output,
        session_data,
    )
    selected_file, preprocess_options = get_preprocess_info(cache, hash_preprocessed, session_data)

    entry = {
        'Inputs': {
            'User File': selected_file,
            'User Text': user_text
        },
        'Preprocessing Options': preprocess_options,
        'Processing Options': processing_options,
        'Batch ID': batch_id,
        'Pitch Batches': pitch_batches,
        'Pitch Variant': pitch,
        'Output ID': hash_output,
        'Model Identity': model_identity,
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
    entry = output_metadata.get(hash_output)
    processing_options = entry.get('Options')
    user_text = entry.get('Inputs').get('User Text')
    hash_preprocessed = entry.get('Inputs').get('Preprocessed File')
    return (
        processing_options,
        user_text,
        hash_preprocessed,
        entry.get('Batch ID'),
        entry.get('Pitch Batches') if isinstance(entry.get('Pitch Batches'), dict) else {},
        entry.get('Pitch Variant'),
        entry.get('Model Identity'),
    )


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

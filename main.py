import argparse
import datetime
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid

import dash_bootstrap_components as dbc
import soundfile
from dash import (
    Dash,
    html,
    dcc,
    Input,
    Output,
    State,
    ctx,
    callback,
    clientside_callback,
    MATCH,
    no_update,
)
from dash.exceptions import PreventUpdate
from hay_say_common.cache import Stage

import hay_say_common as hsc
import cache_media
import output_download
import plotly_celery_common as pcc
import runtime_client
import generation_jobs
from deletion_scheduler import register_cache_cleanup_callback
from postprocessed_display import prepare_postprocessed_display

# todo: so-vits output is much louder than controllable talknet. Should the output volume be equalized?

SHOW_INPUT_OPTIONS_LABEL = 'Show pre-processing options'
SHOW_OUTPUT_OPTIONS_LABEL = 'Show post-processing options'
TAB_BUTTON_PREFIX = '-tab-button'
TAB_CELL_SUFFIX = '-tab-cell'
ANNOUNCEMENT_CHECK_INTERVAL = 60000  # milliseconds
CLIENT_ID_PATTERN = re.compile(r'^[a-f0-9]{32}$')
CANCEL_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0)
_cancel_retry_lock = threading.Lock()
_cancel_retries = set()


def tab_visibility_and_classes(triggered_id, available_tabs):
    """Select the first architecture when the page has no tab trigger yet."""
    button_ids = [tab.id + TAB_BUTTON_PREFIX for tab in available_tabs]
    selected_button_id = triggered_id if triggered_id in button_ids else (
        button_ids[0] if button_ids else None
    )
    hidden_states = [button_id != selected_button_id for button_id in button_ids]
    tab_classes = [
        'tab-cell-selected' if button_id == selected_button_id else 'tab-cell'
        for button_id in button_ids
    ]
    return hidden_states + tab_classes


def normalize_session_data(existing_data, enable_session_caches):
    """Preserve the browser identity that connects a refreshed page to its job."""
    existing_data = existing_data if isinstance(existing_data, dict) else {}
    client_id = existing_data.get('client_id')
    if not isinstance(client_id, str) or CLIENT_ID_PATTERN.fullmatch(client_id) is None:
        client_id = uuid.uuid4().hex
    session_id = existing_data.get('id')
    if enable_session_caches and not isinstance(session_id, str):
        session_id = uuid.uuid4().hex
    if not enable_session_caches:
        session_id = None
    return {'id': session_id, 'client_id': client_id}


def browser_job(session_data):
    client_id = session_data.get('client_id') if isinstance(session_data, dict) else None
    if not isinstance(client_id, str) or CLIENT_ID_PATTERN.fullmatch(client_id) is None:
        return None
    return generation_jobs.get(client_id)


def create_generation_request(queue, session_data, selected_tab, snapshot):
    """Persist a job before returning the self-contained Celery trigger."""
    if not isinstance(session_data, dict) or selected_tab is None or not isinstance(snapshot, dict):
        raise PreventUpdate
    existing = browser_job(session_data)
    if generation_jobs.active(existing):
        raise PreventUpdate
    client_id = session_data.get('client_id')
    request_id = uuid.uuid4().hex
    request_data = {
        'request_id': request_id,
        'client_id': client_id,
        'session_id': session_data.get('id'),
        'snapshot': snapshot,
    }
    try:
        queued = generation_jobs.create_queued(
            client_id,
            request_id,
            selected_tab.id,
            queue,
            'Waiting in queue...',
            request_data=request_data,
        )
    except generation_jobs.GenerationJobConflict:
        # CPU and GPU buttons can race before the next poll disables both.
        raise PreventUpdate from None
    return queued['request_data']


def recover_generation_triggers(state, cpu_request, gpu_request):
    """Recover a queued request whose foreground callback response was lost."""
    if not isinstance(state, dict) or state.get('status') != 'queued':
        return no_update, no_update
    request_data = state.get('request_data')
    if (
        not isinstance(request_data, dict)
        or request_data.get('request_id') != state.get('request_id')
        or request_data.get('client_id') != state.get('client_id')
    ):
        return no_update, no_update
    queue = state.get('queue')
    current = cpu_request if queue == 'cpu' else gpu_request if queue == 'gpu' else None
    if isinstance(current, dict) and current.get('request_id') == state.get('request_id'):
        return no_update, no_update
    if queue == 'cpu':
        return request_data, no_update
    if queue == 'gpu':
        return no_update, request_data
    return no_update, no_update


def generation_job_view(state):
    """Return the compact lifecycle state needed by browser controls."""
    if not isinstance(state, dict):
        return None
    return {
        key: state.get(key)
        for key in (
            'request_id',
            'runtime_id',
            'status',
            'message',
            'progress',
            'operations',
            'updated_at',
        )
    }


def _progress_number(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def generation_progress_components(state):
    """Build small progress rows without placing audio or request data in Dash state."""
    if not isinstance(state, dict):
        return []
    operations = state.get('operations')
    rows = []
    if isinstance(operations, dict):
        for operation_id, operation in sorted(operations.items()):
            if not isinstance(operation, dict):
                continue
            label = operation.get('label') or operation_id
            device = operation.get('device')
            if device:
                label = f'{label} - {device}'
            current = operation.get('current')
            total = operation.get('total')
            status = operation.get('status') or 'pending'
            has_fraction = current is not None and total is not None
            indeterminate = status in {'running', 'cancelling'} and (
                not has_fraction or current == 0
            )
            if indeterminate:
                counter = 'Stopping' if status == 'cancelling' else 'Running'
            elif status == 'pending' and has_fraction and current == 0:
                counter = 'Waiting'
            elif has_fraction:
                counter = f'{_progress_number(current)} / {_progress_number(total)}'
            else:
                counter = status.capitalize()
            progress_options = {'max': total or 1, 'className': 'generation-progress-bar'}
            if has_fraction and not indeterminate:
                progress_options['value'] = current
            rows.append(html.Div([
                html.Div([
                    html.Span(label, className='generation-progress-label'),
                    html.Span(counter, className='generation-progress-counter'),
                ], className='generation-progress-header'),
                html.Progress(**progress_options),
            ], className=f'generation-progress-row generation-progress-{status}'))
    if rows:
        return rows

    progress = state.get('progress')
    if not isinstance(progress, dict):
        return []
    current = progress.get('current')
    total = progress.get('total')
    if current is None or total is None:
        return []
    return [html.Div([
        html.Div([
            html.Span('Overall', className='generation-progress-label'),
            html.Span(
                f'{_progress_number(current)} / {_progress_number(total)}',
                className='generation-progress-counter',
            ),
        ], className='generation-progress-header'),
        html.Progress(value=current, max=total, className='generation-progress-bar'),
    ], className='generation-progress-row generation-progress-running')]


def generation_controls_view(state):
    """Render status controls and make job activity the sole poll switch."""
    active = generation_jobs.active(state)
    progress_rows = generation_progress_components(state) if active else []
    progress_hidden = not progress_rows
    if active:
        message = state.get('message') or 'Generation is running...'
        return message, False, False, False, progress_rows, progress_hidden, False
    if isinstance(state, dict) and state.get('status') == 'failed':
        message = state.get('message') or 'Generation failed.'
        return message, False, True, True, [], True, True
    if isinstance(state, dict) and state.get('status') == 'cancelled':
        return 'Generation stopped.', False, True, True, [], True, True
    return '', True, True, True, [], True, True


def generation_trigger_updates(state, cpu_request, gpu_request):
    """Recover queued work, then clear persisted triggers once it is terminal."""
    if generation_jobs.active(state):
        return recover_generation_triggers(state, cpu_request, gpu_request)
    return (
        None if cpu_request is not None else no_update,
        None if gpu_request is not None else no_update,
    )


def _retry_runtime_cancel(client_id, runtime_id, request_id, runtime_cancel, *,
                          delays=CANCEL_RETRY_DELAYS, sleeper=time.sleep):
    """Retry only the cooperative endpoint while the same request is cancelling."""
    for delay in delays:
        sleeper(delay)
        state = generation_jobs.get(client_id)
        if (
            not isinstance(state, dict)
            or state.get('request_id') != request_id
            or state.get('status') != 'cancelling'
        ):
            return False
        try:
            runtime_cancel(runtime_id, [request_id])
        except runtime_client.RuntimeManagerError as error:
            print(f'Could not retry {runtime_id} cancellation endpoint: {error}', flush=True)
            continue
        return True
    return False


def _schedule_runtime_cancel_retry(client_id, runtime_id, request_id, runtime_cancel):
    key = (client_id, request_id)
    with _cancel_retry_lock:
        if key in _cancel_retries:
            return
        _cancel_retries.add(key)

    def retry():
        try:
            _retry_runtime_cancel(client_id, runtime_id, request_id, runtime_cancel)
        finally:
            with _cancel_retry_lock:
                _cancel_retries.discard(key)

    threading.Thread(
        target=retry,
        name=f'cancel-retry-{request_id[:12]}',
        daemon=True,
    ).start()


def cancel_browser_generation(session_data, runtime_cancel=None, celery_apps=None):
    """Cancel one job cooperatively while preserving the warm model runtime."""
    state = browser_job(session_data)
    if not generation_jobs.active(state):
        return None
    affected_state = generation_jobs.request_cancel(state['client_id'], state['request_id'])
    if not affected_state:
        return None
    runtime_id = affected_state['runtime_id']
    runtime_cancel = runtime_cancel or runtime_client.cancel_generation
    try:
        runtime_cancel(runtime_id, [affected_state['request_id']])
    except runtime_client.RuntimeManagerError as error:
        print(f'Could not signal {runtime_id} cancellation endpoint: {error}', flush=True)
        _schedule_runtime_cancel_retry(
            affected_state['client_id'], runtime_id, affected_state['request_id'], runtime_cancel
        )

    if celery_apps is None:
        from celery_generate_cpu import celery_app as cpu_celery_app
        from celery_generate_gpu import celery_app as gpu_celery_app
        celery_apps = {'cpu': cpu_celery_app, 'gpu': gpu_celery_app}

    task_id = affected_state.get('task_id')
    celery_app = celery_apps.get(affected_state.get('queue'))
    if task_id and celery_app is not None:
        try:
            celery_app.control.revoke(task_id, terminate=False)
        except Exception as error:
            # The durable cancellation marker remains authoritative.
            print(f'Could not revoke generation task {task_id}: {error}', flush=True)
    else:
        # A task that never started cannot observe the durable marker itself.
        generation_jobs.mark_cancelled(affected_state['client_id'], affected_state['request_id'])
    return {'request_id': affected_state['request_id'], 'runtime_id': runtime_id}


def construct_main_interface(tab_buttons, tabs_contents, enable_session_caches):
    return [
        html.Div([
            html.Div(id='dummy'),
            dcc.Store(id='session', storage_type='session', data={'id': None, 'client_id': None}),
            dcc.Store(id='cpu-generation-request', storage_type='session'),
            dcc.Store(id='gpu-generation-request', storage_type='session'),
            dcc.Store(id='generation-result-signal', storage_type='memory'),
            dcc.Store(id='generation-stop-signal', storage_type='memory'),
            dcc.Store(id='generation-render-signature', storage_type='memory'),
            dcc.Store(id='generation-job-state', storage_type='session'),
            dcc.Store(id='generation-polled-job-state', storage_type='memory'),
            dcc.Interval(id='generation-poll', interval=1000, n_intervals=0),
            dcc.Interval(interval=ANNOUNCEMENT_CHECK_INTERVAL, id='announcement-checker'),
            html.Div(
                dcc.Markdown('Banner Announcements will appear here', id='banner-announcement'),
                id='banner-announcement-wrapper', hidden=True),
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Announcement"), close_button=False),
                dbc.ModalBody("Modal Announcements will appear here", id='modal-announcement-body'),
                dbc.ModalFooter(
                    dbc.Button("Close", id='modal-announcement-close'))
            ], id='modal-announcement', is_open=False),
            html.H1('Hay Say'),
            html.H2('A Unified Interface for Pony Voice Generation', className='subtitle'),
            html.H2('Input'),
            dcc.Textarea(id='text-input', placeholder="Enter some text you would like a pony to say."),
            html.P('And/or provide a voice recording that you would like to ponify:'),
            dcc.Upload([html.Div('Drag and drop file or click to Upload...')], id='file-picker', multiple=True),
            dcc.ConfirmDialog(id='confirm-delete-inputs', message='Are you sure you want to delete all uploaded audio?'),
            dcc.Loading(
                html.Table([
                    html.Tr(
                        html.Td(
                            html.Div('Selected File:')
                        )
                    ),
                    html.Tr(
                        html.Td(html.Div([
                            dbc.Select(id='file-dropdown', className='file-dropdown'),
                            html.Button('delete all uploaded files', id='delete-raw-through-output',
                                        style={'margin-top': '20px'})],
                            id='dropdown-container',
                        )),
                    ),
                    html.Tr(
                        html.Td(
                            html.Div(html.Audio(
                                src=None,
                                controls=True,
                                preload='none',
                                id='input-playback',
                                hidden=True,
                            ))
                        )
                    )],
                    className='spaced-table'
                ),
                type='default',
                parent_className='dropdown-container-loader'
            ),
            html.H2('Preprocessing'),
            dcc.Checklist([SHOW_INPUT_OPTIONS_LABEL], id='show-preprocessing-options', value=[],
                          inputStyle={'margin-right': '10px'}),
            # For future ref, this is how you do a vertical checklist with spacing, in case I want to try that again:
            # dcc.Checklist(['Debug pitch', 'Reduce noise', 'Crop Silence'], ['Debug pitch'], id='test',
            #                labelStyle={'display': 'block', 'margin': '20px'}),
            html.Table([
                html.Tr(
                    html.Td('Note: There are currently no options available. ' +
                            'Adding pre-processing options is on the to-do list!',
                            colSpan=2, className='centered')
                ),
                html.Tr([
                    html.Td('Adjust pitch of voice recording (semitones)', className='option-label'),
                    html.Td(dcc.Input(id='semitone-pitch', type='number', min=-25, max=25, step=1, value=0))
                ], hidden=True),
                html.Tr([
                    html.Td('Debug pitch', className='option-label'),
                    html.Td(dcc.Checklist([''], id='debug-pitch'))
                ], hidden=True),
                html.Tr([
                    html.Td('Reduce noise', className='option-label'),
                    html.Td(dcc.Checklist([''], id='reduce-noise'))
                ], hidden=True),
                html.Tr([
                    html.Td('Crop silence at beginning and end', className='option-label'),
                    html.Td(dcc.Checklist([''], id='crop-silence'))
                ], hidden=True),
                html.Tr(
                    html.Td(
                        html.Button("Preview", id='preview'),
                        colSpan=2, className='centered'),
                    hidden=True
                ),
                html.Tr(
                    html.Td(
                        html.Div(
                            html.Audio(src=None, controls=True, preload='none', id='preprocess_playback'),
                            className='centered'),
                        colSpan=2
                    ), hidden=True
                )], id='preprocessing-options', className='spaced-table'
            ),
            html.Br(),
            html.Hr(),
            html.H2('AI Architecture'),
            html.P("Now pick an AI architecture and tweak its settings to your liking:"),
            html.Div(
                html.Table(
                    html.Tr(tab_buttons), className='tab-table-header'
                ),
                className='architecture-tab-strip',
            ),
            html.Div([
                html.Table(
                    tabs_contents,
                    className='tab-table spaced-table'
                ),
                html.Table([
                    html.H2('Postprocessing'),
                    dcc.Checklist([SHOW_OUTPUT_OPTIONS_LABEL], id='show-output-options', value=[],
                                  inputStyle={'margin-right': '10px'})
                    ],
                ),
                html.Table([
                    html.Tr(
                        html.Td('Note: There are currently no options available. ' +
                                'Adding post-processing options is on the to-do list!',
                                colSpan=2, className='centered')
                    ),
                    html.Tr([
                        html.Td('Reduce Metallic Sound', className='option-label'),
                        html.Td(dcc.Checklist([''], id='reduce-metallic-sound'), colSpan=2)
                    ], hidden=True),
                    html.Tr([
                        html.Td('Auto-tune output', className='option-label'),
                        html.Td(dcc.Checklist([''], id='auto-tune-output'), colSpan=2)
                    ], hidden=True),
                    html.Tr([
                        html.Td('Adjust speed of output', className='option-label'),
                        html.Td(html.Div('20', id='output-speed-adjustment')),
                        html.Td(dcc.Input(type='range', min=0.25, max=4, value="1", id='adjust-output-speed',
                                          step='0.01')),
                    ], hidden=True)],
                    id='postprocessing-options',
                    className='spaced-table'
                ),
                html.Table([
                    html.Tr(
                        html.Td(
                            html.Div([
                                dbc.Switch(
                                    id='pitch-batch-enabled',
                                    label='Pitch variants',
                                    value=False,
                                ),
                                dbc.Input(
                                    id='pitch-batch-values',
                                    type='text',
                                    placeholder='-12,-7,0,7,12 or -12:12:2',
                                    disabled=True,
                                ),
                            ], id='pitch-batch-controls', className='pitch-batch-controls', hidden=True),
                            className='generate-cell',
                        ),
                    ),
                    html.Tr(
                        html.Td(
                            html.Div('Generate with:'), className='centered'
                        ),
                    ),
                    html.Tr(
                        html.Td(
                            dcc.Loading(
                                dcc.RadioItems(id='hardware-selector'),
                                type='default'  # circle, graph, cube, circle, dot, default,
                            ),
                            className='centered'
                        ),
                    ),
                    html.Tr(
                        html.Td(
                            html.Button('Generate!', id='generate-button-gpu', className='generate-button'),
                            className='no-padding'
                        ),
                    ),
                    html.Tr(
                        html.Td(
                            html.Button('Generate!', id='generate-button-cpu', className='generate-button'),
                            className='no-padding'
                        ),
                    ),
                    html.Tr(
                        html.Td(
                            html.Div([
                                html.Span('Waiting in queue...', id='generate-message', hidden=True),
                                html.Div(
                                    id='generation-progress',
                                    className='generation-progress',
                                    hidden=True,
                                ),
                                html.Button('Stop', id='cancel-generation', hidden=True,
                                            className='cancel-generation-button'),
                            ], className='generation-status'),
                            className='centered'
                        ),
                    )],
                    className='generate-table'
                ),
            ], className='box-div'),
            html.Br(),
            html.Hr(),
            html.H2('Output'),
            html.Table(
                html.Tr([
                    # todo: hide this delete button if there's nothing to delete?
                    html.Td(
                        html.Button('Delete all generated audio', id='delete-postprocessed'),
                        className='output-delete-cell',
                    ),
                    html.Td(
                        html.Div([
                            html.Label("Download file format:", htmlFor='output-file-format'),
                            dbc.Select(options=sorted([item.lower() for item in soundfile.available_formats().keys() if item.lower() != 'raw']), value='flac', id='output-file-format',
                                       className='file-format-dropdown'),
                        ], className='output-format-control'),
                        className='output-format-cell',
                    )
                ], className='output-controls-row'),
                className='output-controls',
            ),
            html.Div(id='message'),
        ], id='hay-say-outer-div', className='outer-div')
    ]


def register_generate_callbacks(cache_type, architectures):
    import celery_generate_gpu
    import celery_generate_cpu
    celery_generate_gpu.CacheSelection(None, cache_type, architectures)
    celery_generate_cpu.CacheSelection(None, cache_type, architectures)


def register_main_callbacks(enable_session_caches, cache_type, architectures):
    cache = hsc.select_cache_implementation(cache_type)
    available_tabs = pcc.select_architecture_tabs(architectures)

    @callback(
        [Output('pitch-batch-controls', 'hidden'),
         Output('pitch-batch-values', 'disabled')],
        [Input(tab.id, 'hidden') for tab in available_tabs] +
        [Input('pitch-batch-enabled', 'value')]
    )
    def configure_pitch_batch(*hidden_states_and_enabled):
        hidden_states = hidden_states_and_enabled[:-1]
        enabled = hidden_states_and_enabled[-1]
        selected_tab = get_selected_tab_object(hidden_states)
        unsupported = selected_tab is None or selected_tab.pitch_batch_key is None
        return unsupported, unsupported or not enabled

    @callback(
        Output('pitch-batch-enabled', 'value'),
        [Input(tab.id, 'hidden') for tab in available_tabs],
    )
    def reset_pitch_batch_for_unsupported_tab(*hidden_states):
        selected_tab = get_selected_tab_object(hidden_states)
        return False if selected_tab is None or selected_tab.pitch_batch_key is None else no_update

    @callback(
        Output('session', 'data'),
        Input('dummy', 'n_clicks'),
        State('session', 'data')
    )
    def initialize_session_data(n_clicks, existing_data):
        if n_clicks is None:
            return normalize_session_data(existing_data, enable_session_caches)
        else:
            print('Warning! initialize_session_data was called outside of initialization. Ignoring request.',
                  flush=True)
            return existing_data

    snapshot_states = [
        State('session', 'data'),
        State('text-input', 'value'),
        State('file-dropdown', 'value'),
        State('semitone-pitch', 'value'),
        State('debug-pitch', 'value'),
        State('reduce-noise', 'value'),
        State('crop-silence', 'value'),
        State('reduce-metallic-sound', 'value'),
        State('auto-tune-output', 'value'),
        State('adjust-output-speed', 'value'),
        State('pitch-batch-enabled', 'value'),
        State('pitch-batch-values', 'value'),
    ] + [State(tab.id, 'hidden') for tab in available_tabs] + [
        State(item, 'value') for tab in available_tabs for item in tab.input_ids
    ]

    def queue_generation(queue, hardware_selection, session_data, user_text, selected_file,
                         semitone_pitch, debug_pitch, reduce_noise, crop_silence,
                         reduce_metallic_noise, auto_tune_output, output_speed_adjustment,
                         pitch_batch_enabled, pitch_batch_values, tab_values):
        hidden_states = list(tab_values[:len(available_tabs)])
        architecture_inputs = list(tab_values[len(available_tabs):])
        selected_tab = get_selected_tab_object(hidden_states)
        snapshot = {
            'hardware_selection': hardware_selection,
            'user_text': user_text,
            'selected_file': selected_file,
            'semitone_pitch': semitone_pitch,
            'debug_pitch': debug_pitch,
            'reduce_noise': reduce_noise,
            'crop_silence': crop_silence,
            'reduce_metallic_noise': reduce_metallic_noise,
            'auto_tune_output': auto_tune_output,
            'output_speed_adjustment': output_speed_adjustment,
            'pitch_batch_enabled': pitch_batch_enabled,
            'pitch_batch_values': pitch_batch_values,
            'hidden_states': hidden_states,
            'architecture_inputs': architecture_inputs,
        }
        return create_generation_request(queue, session_data, selected_tab, snapshot)

    @callback(
        [Output('cpu-generation-request', 'data'),
         Output('generation-job-state', 'data', allow_duplicate=True)],
        [Input('generate-button-cpu', 'n_clicks'), State('hardware-selector', 'value')] + snapshot_states,
        prevent_initial_call=True,
    )
    def queue_cpu_generation(n_clicks, hardware_selection, session_data, user_text, selected_file,
                             semitone_pitch, debug_pitch, reduce_noise, crop_silence,
                             reduce_metallic_noise, auto_tune_output, output_speed_adjustment,
                             pitch_batch_enabled, pitch_batch_values, *tab_values):
        if not n_clicks:
            raise PreventUpdate
        request_data = queue_generation(
            'cpu', hardware_selection, session_data, user_text, selected_file, semitone_pitch,
            debug_pitch, reduce_noise, crop_silence, reduce_metallic_noise, auto_tune_output,
            output_speed_adjustment, pitch_batch_enabled, pitch_batch_values, tab_values,
        )
        return request_data, generation_job_view(browser_job(session_data))

    @callback(
        [Output('gpu-generation-request', 'data'),
         Output('generation-job-state', 'data', allow_duplicate=True)],
        [Input('generate-button-gpu', 'n_clicks')] + snapshot_states,
        prevent_initial_call=True,
    )
    def queue_gpu_generation(n_clicks, session_data, user_text, selected_file, semitone_pitch,
                             debug_pitch, reduce_noise, crop_silence, reduce_metallic_noise,
                             auto_tune_output, output_speed_adjustment, pitch_batch_enabled,
                             pitch_batch_values, *tab_values):
        if not n_clicks:
            raise PreventUpdate
        request_data = queue_generation(
            'gpu', 'GPU', session_data, user_text, selected_file, semitone_pitch, debug_pitch,
            reduce_noise, crop_silence, reduce_metallic_noise, auto_tune_output,
            output_speed_adjustment, pitch_batch_enabled, pitch_batch_values, tab_values,
        )
        return request_data, generation_job_view(browser_job(session_data))

    @callback(
        [Output('banner-announcement', 'children'),
         Output('banner-announcement-wrapper', 'hidden'),
         Output('modal-announcement-body', 'children'),
         Output('modal-announcement', 'is_open')],
        [State('modal-announcement', 'is_open'),
         Input('announcement-checker', 'n_intervals')],
    )
    def display_announcements(current_modal_open, _):
        time_now_utc = datetime.datetime.now(datetime.timezone.utc)
        announcements_file = os.path.join(os.path.dirname(__file__), 'running as server', 'announcements.json')
        print('announcements_file', flush=True)
        print(announcements_file, flush=True)
        banner_announcements, banner_hidden, modal_announcements, modal_open = [], True, [], current_modal_open
        if os.path.isfile(announcements_file):
            try:
                with open(announcements_file, 'r') as file:
                    json_contents = json.load(file)
                for announcement in json_contents:
                    time_format = '%Y-%m-%d %H:%M:%S%z'  # e.g. "2023-11-25 19:01:11+0000"
                    effective_time = datetime.datetime.strptime(announcement['Effective Time'], time_format)
                    expiration_time = datetime.datetime.strptime(announcement['Expiration Time'], time_format)
                    message = announcement['Message']
                    if effective_time <= time_now_utc < expiration_time:
                        # Announcement is active.
                        if announcement.get('Modal') and time_now_utc < effective_time + \
                                datetime.timedelta(milliseconds=ANNOUNCEMENT_CHECK_INTERVAL-1):
                            # A modal announcements is only displayed if it just went active within the last
                            # ANNOUNCEMENT_CHECK_INTERVAL milliseconds.
                            modal_announcements += [message]
                        if announcement.get('Banner'):
                            banner_announcements += [message]
                if banner_announcements:
                    banner_hidden = False
                if modal_announcements:
                    modal_open = True
            except:
                # If there's any problem with parsing the announcements file, don't display any announcements.
                banner_announcements, banner_hidden, modal_announcements, modal_open = [], True, [], current_modal_open
        return '\n\n'.join(banner_announcements), banner_hidden, '\n\n'.join(modal_announcements), modal_open

    @callback(
        Output('modal-announcement', 'is_open', allow_duplicate=True),
        Input('modal-announcement-close', 'n_clicks'),
        prevent_initial_call=True
    )
    def close_modal_announcements(_):
        return False

    @callback(
        [Output('generate-button-gpu', 'hidden'),
         Output('generate-button-cpu', 'hidden')],
        Input('hardware-selector', 'value')
    )
    def select_generate_button(hardware_selection):
        return hardware_selection != 'GPU', hardware_selection not in ('Auto', 'CPU')

    @callback(
        [Output('message', 'children'),
         Output('generation-render-signature', 'data'),
         Output('cpu-generation-request', 'data', allow_duplicate=True),
         Output('gpu-generation-request', 'data', allow_duplicate=True),
         Output('generation-polled-job-state', 'data')],
        [Input('generation-poll', 'n_intervals'),
         Input('generation-result-signal', 'data'),
         Input('generation-stop-signal', 'data')],
        [State('session', 'data'),
         State('generation-render-signature', 'data'),
         State('cpu-generation-request', 'data'),
         State('gpu-generation-request', 'data'),
         State('generation-job-state', 'data')],
        prevent_initial_call='initial_duplicate',
    )
    def poll_generation(_n_intervals, _result_signal, _stop_signal, session_data, rendered_signature,
                        cpu_request, gpu_request, displayed_job_state):
        session_data = session_data if isinstance(session_data, dict) else {'id': None}
        hashes = cache.get_hashes_sorted_by_timestamp(Stage.POSTPROCESSED, session_data.get('id'))
        signature = list(hashes)
        if signature == rendered_signature:
            outputs = no_update
            signature_output = no_update
        else:
            newest = [prepare_postprocessed_display(cache, hashes[0], session_data)] if hashes else []
            older = [
                prepare_postprocessed_display(cache, output_hash, session_data)
                for output_hash in reversed(hashes[1:])
            ]
            outputs = older + newest
            signature_output = signature

        state = browser_job(session_data)
        state_view = generation_job_view(state)
        state_output = no_update if state_view == displayed_job_state else state_view
        cpu_trigger, gpu_trigger = generation_trigger_updates(state, cpu_request, gpu_request)

        return outputs, signature_output, cpu_trigger, gpu_trigger, state_output

    clientside_callback(
        """
        function(polled, current) {
            const noUpdate = window.dash_clientside.no_update;
            if (!polled || JSON.stringify(polled) === JSON.stringify(current)) {
                return noUpdate;
            }
            if (!current) {
                return polled;
            }
            const active = value =>
                value && ["queued", "running", "cancelling"].includes(value.status);
            if (polled.request_id !== current.request_id) {
                if (active(current) && !active(polled)) {
                    return noUpdate;
                }
                if (active(polled) && !active(current)) {
                    return polled;
                }
            } else {
                const rank = value => {
                    if (!value) return -1;
                    if (["completed", "failed", "cancelled"].includes(value.status)) return 3;
                    if (value.status === "cancelling") return 2;
                    if (value.status === "running") return 1;
                    if (value.status === "queued") return 0;
                    return -1;
                };
                const polledRank = rank(polled);
                const currentRank = rank(current);
                if (polledRank > currentRank) {
                    return polled;
                }
                if (polledRank < currentRank) {
                    return noUpdate;
                }
            }
            return (polled.updated_at || "") >= (current.updated_at || "")
                ? polled
                : noUpdate;
        }
        """,
        Output('generation-job-state', 'data', allow_duplicate=True),
        Input('generation-polled-job-state', 'data'),
        State('generation-job-state', 'data'),
        prevent_initial_call=True,
    )

    @callback(
        [Output('generate-message', 'children'),
         Output('generate-message', 'hidden'),
         Output('cancel-generation', 'hidden'),
         Output('cancel-generation', 'disabled'),
         Output('generation-progress', 'children'),
         Output('generation-progress', 'hidden'),
         Output('generation-poll', 'disabled')],
        Input('generation-job-state', 'data'),
    )
    def render_generation_controls(job_state):
        return generation_controls_view(job_state)

    @callback(
        Output('generation-stop-signal', 'data'),
        Input('cancel-generation', 'n_clicks'),
        State('session', 'data'),
        prevent_initial_call=True,
    )
    def stop_generation(n_clicks, session_data):
        if not n_clicks or not isinstance(session_data, dict):
            raise PreventUpdate
        result = cancel_browser_generation(session_data)
        if result is None:
            raise PreventUpdate
        return result

    @callback(
        Output('message', 'children', allow_duplicate=True),
        Input('delete-postprocessed', 'n_clicks'),
        State('session', 'data'),
        prevent_initial_call=True
    )
    def delete_all_postprocessed(_, session_data):
        cache.delete_all_files_at_stage(Stage.POSTPROCESSED, session_data['id'])
        return ''

    gpt_so_vits_tab = pcc.architecture_map().get('GPTSoVITS', None)
    if gpt_so_vits_tab is not None:
        @callback(
            [Output('file-dropdown', 'options', allow_duplicate=True),
             Output('file-dropdown', 'value', allow_duplicate=True),
             Output('dropdown-container', 'hidden', allow_duplicate=True),
             Output(gpt_so_vits_tab.input_ids[1], 'options', allow_duplicate=True),
             Output(gpt_so_vits_tab.input_ids[1], 'value', allow_duplicate=True),
             Output(gpt_so_vits_tab.input_ids[4], 'options', allow_duplicate=True),
             Output(gpt_so_vits_tab.input_ids[4], 'value', allow_duplicate=True)],
            Input('confirm-delete-inputs', 'submit_n_clicks'),
            State('session', 'data'),
            prevent_initial_call=True
        )
        def delete_all_raw_through_output(_, session_data):
            cache.delete_all_files_at_stage(Stage.RAW, session_data['id'])
            cache.delete_all_files_at_stage(Stage.PREPROCESSED, session_data['id'])
            cache.delete_all_files_at_stage(Stage.OUTPUT, session_data['id'])
            return [], None, True, [], None, [], []

        @callback(
            [Output('file-dropdown', 'options'),
             Output('file-dropdown', 'value'),
             Output('dropdown-container', 'hidden'),
             Output('file-picker', 'contents'),
             Output(gpt_so_vits_tab.input_ids[1], 'options'),
             Output(gpt_so_vits_tab.input_ids[1], 'value'),
             Output(gpt_so_vits_tab.input_ids[4], 'options'),
             Output(gpt_so_vits_tab.input_ids[4], 'value')],
            # work around for issue #816 (https://github.com/plotly/dash-core-components/issues/816)
            Input('file-picker', 'contents'),
            State('file-picker', 'filename'),
            State('session', 'data'),
            State(gpt_so_vits_tab.input_ids[1], 'value'),
            State(gpt_so_vits_tab.input_ids[4], 'value')
        )
        def upload_file(file_contents_list, filename_list, session_data, current_gpt_file, current_additional_files):
            if file_contents_list is None:  # initial load of page
                filenames, currently_selected_file, hidden = update_dropdown(cache, None, session_data)
                additional = [name for name in (current_additional_files or []) if name in filenames]
                return (filenames, currently_selected_file, hidden, None, filenames, currently_selected_file,
                        filenames, additional)
            else:
                for file_contents, filename in zip(file_contents_list, filename_list):
                    filename = append_index_if_needed(filename, session_data)
                    raw_array, raw_samplerate = hsc.get_audio_from_src_attribute(file_contents, 'utf-8')
                    save_raw_audio_to_cache(filename, raw_array, raw_samplerate, session_data)
                filenames, currently_selected_file, hidden = update_dropdown(cache, filename_list[0], session_data)
                selected_gptsovits_file = current_gpt_file if current_gpt_file in filenames else currently_selected_file
                additional = [name for name in (current_additional_files or []) if name in filenames]
                return (filenames, currently_selected_file, hidden, None, filenames, selected_gptsovits_file,
                        filenames, additional)
    else:
        @callback(
            [Output('file-dropdown', 'options', allow_duplicate=True),
             Output('file-dropdown', 'value', allow_duplicate=True),
             Output('dropdown-container', 'hidden', allow_duplicate=True)],
            Input('confirm-delete-inputs', 'submit_n_clicks'),
            State('session', 'data'),
            prevent_initial_call=True
        )
        def delete_all_raw_through_output(_, session_data):
            cache.delete_all_files_at_stage(Stage.RAW, session_data['id'])
            cache.delete_all_files_at_stage(Stage.PREPROCESSED, session_data['id'])
            cache.delete_all_files_at_stage(Stage.OUTPUT, session_data['id'])
            return [], None, True

        @callback(
            [Output('file-dropdown', 'options'),
             Output('file-dropdown', 'value'),
             Output('dropdown-container', 'hidden'),
             Output('file-picker', 'contents')],  # work around for issue #816 (https://github.com/plotly/dash-core-components/issues/816)
            Input('file-picker', 'contents'),
            State('file-picker', 'filename'),
            State('session', 'data'),
        )
        def upload_file(file_contents_list, filename_list, session_data):
            if file_contents_list is None:  # initial load of page
                return *update_dropdown(cache, None, session_data), None
            else:
                for file_contents, filename in zip(file_contents_list, filename_list):
                    filename = append_index_if_needed(filename, session_data)
                    raw_array, raw_samplerate = hsc.get_audio_from_src_attribute(file_contents, 'utf-8')
                    save_raw_audio_to_cache(filename, raw_array, raw_samplerate, session_data)
                return *update_dropdown(cache, filename_list[0], session_data), None

    @callback(
        Output('confirm-delete-inputs', 'displayed'),
        Input('delete-raw-through-output', 'n_clicks'),
        prevent_initial_call=True
    )
    def display_confirm_for_deleting_inputs(_):
        return True

    @callback(
        [Output(tab.id, 'hidden') for tab in available_tabs] +
        [Output(tab.id + TAB_CELL_SUFFIX, 'className') for tab in available_tabs] +
        [Input(tab.id + TAB_BUTTON_PREFIX, 'n_clicks') for tab in available_tabs],
    )
    def hide_unused_tabs(*_):
        return tab_visibility_and_classes(ctx.triggered_id, available_tabs)

    @callback(
        Output('output-speed-adjustment', 'children'),
        Input('adjust-output-speed', 'value')
    )
    def adjust_output_speed(adjustment):
        # cast to float first, then round to 2 decimal places
        return "{:3.2f}".format(float(adjustment))

    @callback(
        Output('preprocessing-options', 'hidden'),
        Input('show-preprocessing-options', 'value')
    )
    def show_preprocessing_options(value):
        return SHOW_INPUT_OPTIONS_LABEL not in value

    def append_index_if_needed(filename, session_data):
        # Appends an index to the end of the filename, like 'my file.wav (2)', if the file already exists.
        # todo: I think putting something after the extension might break stuff. Do this instead: 'my file (2).wav'
        raw_metadata = cache.read_metadata(Stage.RAW, session_data['id'])
        similar_filenames = [value['User File']
                             for value in raw_metadata.values()
                             if value['User File'].startswith(filename)
                             and (re.match(r' \([0-9]+\)', value['User File'][
                                                           len(filename):])  # file with same name but ending with ' (#)'
                                  or not value['User File'][len(filename):])]  # file with exactly the same name
        index = 1
        while filename in similar_filenames:
            index += 1
            filename = filename + ' (' + str(index) + ')'
        return filename

    def save_raw_audio_to_cache(filename, raw_array, raw_samplerate, session_data):
        hash_raw = hashlib.sha256(raw_array).hexdigest()[:20]
        if cache.file_is_already_cached(Stage.RAW, session_data['id'], hash_raw):
            pass
        else:
            cache.save_audio_to_cache(Stage.RAW, session_data['id'], hash_raw, raw_array, raw_samplerate)
            write_raw_metadata(hash_raw, filename, session_data)

    def write_raw_metadata(hash_80_bits, filename, session_data):
        entry = {
            'User File': filename,
            'Time of Creation': datetime.datetime.now().strftime(hsc.cache.TIMESTAMP_FORMAT)
        }
        cache.update_metadata(
            Stage.RAW,
            session_data['id'],
            lambda metadata: metadata.update({hash_80_bits: entry}),
        )

    @callback(
        [Output('input-playback', 'src'),
         Output('input-playback', 'hidden')],
        Input('file-dropdown', 'value'),
        State('session', 'data'),
    )
    def update_playback(selected_file, session_data):
        if selected_file is None:
            return None, True
        metadata = cache.read_metadata(Stage.RAW, session_data['id'])
        reverse_lookup = {metadata[key]['User File']: key for key in metadata}
        hash_raw = reverse_lookup[selected_file]
        return cache_media.cache_audio_url(Stage.RAW, session_data['id'], hash_raw), False

    @callback(
        Output('postprocessing-options', 'hidden'),
        Input('show-output-options', 'value')
    )
    def show_postprocessing_options(value):
        return SHOW_OUTPUT_OPTIONS_LABEL not in value

    @callback(
        [Output('generate-button-gpu', 'disabled'),
         Output('generate-button-cpu', 'disabled')],
        [Input('text-input', 'value'),
         Input('file-dropdown', 'value'),
         Input('generation-job-state', 'data')] +
        [Input(tab.id, 'hidden') for tab in available_tabs] +
        [Input(item, 'value') for tab in available_tabs for item in tab.input_ids]
    )
    def disable_generate_button(user_text, selected_file, job_state, *hidden_states_and_options):
        # todo: don't disable the generate button. Instead, highlight the requirements text and whatever the user is
        #  missing in red.
        hidden_states = hidden_states_and_options[:len(available_tabs)]
        option_values = hidden_states_and_options[len(available_tabs):]
        job_active = generation_jobs.active(job_state)
        tab_object = get_selected_tab_object(hidden_states)
        if tab_object is None:
            return True, True
        else:
            index = hidden_states.index(False)
            start = sum(len(tab.input_ids) for tab in available_tabs[:index])
            selected_options = option_values[start:start + len(tab_object.input_ids)]
            selected_character = selected_options[0]
            hidden = not tab_object.meets_all_requirements(
                user_text, selected_file, selected_character, selected_options
            )
            disabled = hidden or job_active
            return disabled, disabled

    # todo: disable the preview button if no audio file is selected.
    @callback(
        Output('preprocess_playback', 'src'),
        Input('preview', 'n_clicks'),
        State('session', 'data'),
        State('file-dropdown', 'value'),
        State('semitone-pitch', 'value'),
        State('debug-pitch', 'value'),
        State('reduce-noise', 'value'),
        State('crop-silence', 'value'),
    )
    def generate_preview(_, session_data, selected_file, semitone_pitch, debug_pitch, reduce_noise, crop_silence):
        if selected_file is None:
            raise PreventUpdate

        hash_preprocessed = pcc.preprocess(
            cache,
            selected_file,
            semitone_pitch,
            debug_pitch,
            reduce_noise,
            crop_silence,
            session_data,
        )

        # return src
        return cache_media.cache_audio_url(
            Stage.PREPROCESSED,
            session_data['id'],
            hash_preprocessed,
        )

    def get_selected_tab_object(hidden_states):
        # Get the tab that is *not* hidden (i.e. hidden == False)
        return {hidden: tab for hidden, tab in zip(hidden_states, available_tabs)}.get(False)

    @callback(
        Output({'type': 'output-download', 'index': MATCH}, 'data'),
        State('session', 'data'),
        State('output-file-format', 'value'),
        Input({'type': 'output-download-button', 'index': MATCH}, 'n_clicks'),
        prevent_initial_call=True
    )
    def download_postprocessed_audio(session_data, output_file_format, n_clicks):
        if n_clicks:
            # A download button was actually clicked, so return a download.
            hash_postprocessed = ctx.triggered_id['index']
            metadata = cache.read_metadata(Stage.POSTPROCESSED, session_data['id'])[hash_postprocessed]
            filename = output_download.descriptive_audio_filename(
                metadata,
                hash_postprocessed,
                output_file_format,
            )
            with tempfile.TemporaryDirectory() as tempdir:
                path = os.path.join(tempdir, filename)
                data, sr = cache.read_audio_from_cache(Stage.POSTPROCESSED, session_data['id'], hash_postprocessed)
                soundfile.write(path, data, sr)
                return dcc.send_file(path, filename=filename)
        else:
            return None

    @callback(
        Output({'type': 'batch-download', 'index': MATCH}, 'data'),
        State('session', 'data'),
        State('output-file-format', 'value'),
        Input({'type': 'batch-download-button', 'index': MATCH}, 'n_clicks'),
        prevent_initial_call=True
    )
    def download_pitch_batch(session_data, output_file_format, n_clicks):
        if not n_clicks:
            return None
        archive, filename = output_download.create_pitch_batch_archive(
            cache,
            session_data['id'],
            ctx.triggered_id['index'],
            output_file_format,
        )
        return dcc.send_bytes(archive, filename, type='application/zip')

    def update_dropdown(cache, filename, session_data):
        raw_metadata = cache.read_metadata(Stage.RAW, session_data['id'])
        filenames = [value['User File'] for value in raw_metadata.values()]
        currently_selected_file = filename if filename else filenames[0] if filenames else None
        hidden = currently_selected_file is None
        return filenames, currently_selected_file, hidden

        # raw_metadata = cache.read_metadata(Stage.RAW, session_data['id']) if ctx.triggered_id == 'confirm-delete-inputs' else dict()
        # filenames = [value['User File'] for value in raw_metadata.values()]
        # filenames = filenames + uploaded_filename_list if uploaded_filename_list is not None else filenames
        # currently_selected_file = currently_selected_file if currently_selected_file in filenames \
        #     else uploaded_filename_list[0] if uploaded_filename_list is not None and len(uploaded_filename_list) > 0 \
        #     else filenames[0] if len(filenames) > 0 \
        #     else None
        # return filenames, currently_selected_file


def construct_tab_buttons(available_tabs):
    return [html.Td(
        html.Button(tab.label, id=tab.id + TAB_BUTTON_PREFIX, className='tab-button'),
        className='tab-cell', id=tab.id + TAB_CELL_SUFFIX)
        for tab in available_tabs]


def construct_tabs_interface(architectures, enable_model_management, cache_type):
    available_tabs = pcc.construct_architecture_tabs(architectures, cache_type)
    tab_buttons = construct_tab_buttons(available_tabs)
    tabs_contents = [tab.tab_contents(enable_model_management) for tab in available_tabs]
    return tab_buttons, tabs_contents


def register_tab_callbacks(architectures, enable_model_management):
    available_tabs = pcc.select_architecture_tabs(architectures)
    for tab in available_tabs:
        tab.register_callbacks(enable_model_management)


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(prog='wsgi.py', description='A Unified Interface for Pony Voice Generation.')
    parser.add_argument('--update_model_lists_on_startup', action='store_true', default=False, help='Causes Hay Say to download the latest model lists so that all the latest models appear in the character download menus.')
    parser.add_argument('--enable_model_management', action='store_true', default=False, help='Enables the user to download and delete models.')
    parser.add_argument('--enable_runtime_admin', action='store_true', default=None,
                        help='Enables the native model runtime status and lifecycle panel.')
    parser.add_argument('--enable_session_caches', action='store_true', default=False, help='Maintain separate caches for each session. If not enabled, a single cache is used for all sessions.')
    parser.add_argument('--cache_implementation', default='file', choices=hsc.cache_implementation_map.keys(), help='Selects an implementation for the audio cache, e.g. saving them to files or to a database.')
    parser.add_argument('--migrate_models', action='store_true', default=False, help='Automatically move models from the model pack directories and custom model directory to the new models directory when Hay Say starts.')
    # todo: this is hardcoded. fix it.
    parser.add_argument('--architectures', nargs='*', choices=['ControllableTalkNet', 'SoVitsSvc3', 'SoVitsSvc4', 'SoVitsSvc5', 'Rvc', 'StyleTTS2', 'GPTSoVITS'], default=['ControllableTalkNet', 'SoVitsSvc3', 'SoVitsSvc4', 'SoVitsSvc5', 'Rvc', 'StyleTTS2', 'GPTSoVITS'], help='Selects which architectures are shown in the Hay Say UI')
    return parser.parse_args(arguments)


def construct_app_layout(enable_model_management, cache_type, architectures, enable_session_caches):
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.SLATE],
        meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
    )
    tab_buttons, tabs_contents = construct_tabs_interface(architectures, enable_model_management, cache_type)
    app.layout = html.Div(construct_main_interface(tab_buttons, tabs_contents, enable_session_caches))
    app.title = 'Hay Say'
    return app


def register_app_callbacks(architectures, enable_model_management, enable_session_caches, cache_type):
    register_tab_callbacks(architectures, enable_model_management)
    register_main_callbacks(enable_session_caches, cache_type, architectures)
    register_generate_callbacks(cache_type, architectures)


def register_download_callbacks(cache_type, architectures):
    # The import statement is located here, to make sure that the download callbacks are not loaded at all unless this
    # method is called.
    from celery_download import ArchitectureSelection
    # Instantiate ArchitectureSelection to instantiate the download callbacks.
    ArchitectureSelection(None, cache_type, architectures)


def add_model_manager_page(app, available_tabs):
    # The import statement is located here, to make sure that the model_manager module is not loaded at all unless this
    # method is called.
    from model_manager import construct_model_manager, register_model_manager_callbacks
    app.layout.children.append(construct_model_manager(available_tabs))
    register_model_manager_callbacks(available_tabs)


def add_toolbar(app, enable_model_management, enable_runtime_admin):
    # The import statement is located here, to make sure that the toolbar module is not loaded at all unless this method
    # is called.
    from toolbar import construct_toolbar, register_toolbar_callbacks
    app.layout.children.append(construct_toolbar(enable_model_management, enable_runtime_admin))
    register_toolbar_callbacks(enable_model_management, enable_runtime_admin)


def add_runtime_admin_page(app):
    from runtime_admin import construct_runtime_admin, register_runtime_admin_callbacks
    app.layout.children.append(construct_runtime_admin())
    register_runtime_admin_callbacks()


def add_management_components(cache_type, enable_model_management, enable_runtime_admin, architectures, app):
    available_tabs = pcc.select_architecture_tabs(architectures)
    if enable_model_management:
        register_download_callbacks(cache_type, architectures)
        add_model_manager_page(app, available_tabs)
    else:
        app.layout.children.append(html.Div(id='model-manager-outer-div', hidden=True))
    if enable_runtime_admin:
        add_runtime_admin_page(app)
    else:
        app.layout.children.append(html.Div(id='runtime-admin-outer-div', hidden=True))
    if enable_model_management or enable_runtime_admin:
        add_toolbar(app, enable_model_management, enable_runtime_admin)


def register_cache_cleanup_callback_if_needed(enable_session_caches, cache_type):
    if enable_session_caches:
        register_cache_cleanup_callback(cache_type)


def build_app(architectures, update_model_lists_on_startup=False, enable_model_management=False, enable_session_caches=False,
              cache_type='file', migrate_models=False, enable_runtime_admin=None):
    if enable_runtime_admin is None:
        enable_runtime_admin = runtime_client.admin_enabled()
    app = construct_app_layout(enable_model_management, cache_type, architectures, enable_session_caches)
    cache_media.register_cache_audio_route(app.server, hsc.select_cache_implementation(cache_type))
    register_app_callbacks(architectures, enable_model_management, enable_session_caches, cache_type)
    add_management_components(cache_type, enable_model_management, enable_runtime_admin, architectures, app)
    register_cache_cleanup_callback_if_needed(enable_session_caches, cache_type)

    # Save some of the command-line options to the server object so that the server hook methods can get to them:
    app.server.update_model_lists_on_startup = update_model_lists_on_startup
    app.server.migrate_models = migrate_models
    app.server.architectures = architectures
    return app

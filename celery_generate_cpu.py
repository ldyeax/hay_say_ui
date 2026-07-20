import click
from celery import Celery, bootsteps
from click import Option
from dash import Input, Output, State, callback, CeleryManager, ctx
from dash.exceptions import PreventUpdate

import hay_say_common as hsc
import main
import plotly_celery_common as pcc
from generator import (
    GenerationCancelled,
    GenerationRequestUnavailable,
    generate_and_prepare_postprocessed_display,
)
from celery_config import redis_url

# Set up a background callback manager
REDIS_URL = redis_url(2)
celery_app = Celery(__name__, broker=REDIS_URL, backend=REDIS_URL)
background_callback_manager = CeleryManager(celery_app)

# Add a command-line argument for selecting the cache implementation
celery_app.user_options['worker'].add(
    Option(('--cache_implementation',), default='file', show_default=True,
           type=click.Choice(hsc.cache.cache_implementation_map.keys(), case_sensitive=False),
           help='Selects an implementation for the audio cache, e.g. saving them to files or to a database.'))

# Add a command-line argument that lets the user select specific architectures to register with the celery worker
celery_app.user_options['worker'].add(
    Option(('--include_architecture',), multiple=True, default=[], show_default=True,
           help='Add an architecture for which the download callback will be registered'))


# Add a boot step to use the command-line argument
class CacheSelection(bootsteps.Step):
    def __init__(self, parent, cache_implementation, include_architecture, **options):
        super().__init__(parent, **options)
        selected_architectures = pcc.construct_architecture_tabs(include_architecture, cache_implementation)

        @callback(
            output=Output('generation-result-signal', 'data'),
            inputs=[Input('cpu-generation-request', 'data')],
            background=True,
            manager=background_callback_manager,
            prevent_initial_call='initial_duplicate',
        )
        def generate_with_cpu(request_data):
            if not isinstance(request_data, dict):
                raise PreventUpdate
            snapshot = request_data.get('snapshot')
            hardware_selection = snapshot.get('hardware_selection') if isinstance(snapshot, dict) else None
            gpu_id = requested_device(hardware_selection)
            message = 'selecting an available CPU/GPU slot...' if gpu_id == 'auto' else 'generating on CPU...'
            try:
                result, _button_label = generate_and_prepare_postprocessed_display(
                    request_data, message, cache_implementation, gpu_id, selected_architectures,
                )
                return result
            except (GenerationCancelled, GenerationRequestUnavailable):
                raise PreventUpdate

        @callback(
            [Output('hardware-selector', 'options')] +
            [Output('hardware-selector', 'value')],
            [State('hardware-selector', 'value')] +
            [Input(tab.id + main.TAB_BUTTON_PREFIX, 'n_clicks') for tab in selected_architectures],
        )
        def hide_unused_tabs(current_hardware_selection, *_):
            hidden_states = [
                not (tab.id + main.TAB_BUTTON_PREFIX == ctx.triggered_id)
                for tab in selected_architectures
            ]
            selected_tab = get_selected_tab_object(hidden_states) if ctx.triggered_id else (
                selected_architectures[0] if selected_architectures else None
            )
            hardware_options, hardware_selection = select_hardware(
                current_hardware_selection,
                selected_tab,
            )
            return [hardware_options, hardware_selection]

        def get_selected_tab_object(hidden_states):
            # Get the tab that is *not* hidden (i.e. hidden == False)
            return {hidden: tab for hidden, tab in zip(hidden_states, selected_architectures)}.get(False)


celery_app.steps['worker'].add(CacheSelection)


def requested_device(hardware_selection):
    if hardware_selection in (None, 'Auto'):
        return 'auto'
    if hardware_selection == 'CPU':
        return ''
    raise ValueError(f'CPU dispatcher received unsupported hardware selection: {hardware_selection!r}')


def select_hardware(current_hardware_selection, selected_tab=None):
    options = selected_tab.hardware_options if selected_tab is not None else ['Auto', 'CPU']
    selection = current_hardware_selection if current_hardware_selection in options else 'Auto'
    return options, selection

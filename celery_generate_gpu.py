import click
from billiard.process import current_process
from celery import Celery, bootsteps
from click import Option
from dash import Input, Output, callback, CeleryManager
from dash.exceptions import PreventUpdate

import hay_say_common as hsc
import plotly_celery_common as pcc
from generator import (
    GenerationCancelled,
    GenerationRequestUnavailable,
    generate_and_prepare_postprocessed_display,
)
from gpu_selection import gpu_id_for_worker
from celery_config import redis_url

# Set up a background callback manager
REDIS_URL = redis_url(1)
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
            output=Output('generation-result-signal', 'data', allow_duplicate=True),
            inputs=[Input('gpu-generation-request', 'data')],
            background=True,
            manager=background_callback_manager,
            prevent_initial_call='initial_duplicate',
        )
        def generate_with_gpu(request_data):
            if not isinstance(request_data, dict):
                raise PreventUpdate
            gpu_id = gpu_id_for_worker(current_process().index)
            message = 'generating on GPU #' + str(gpu_id) + '...'
            try:
                result, _button_label = generate_and_prepare_postprocessed_display(
                    request_data, message, cache_implementation, gpu_id, selected_architectures,
                )
                return result
            except (GenerationCancelled, GenerationRequestUnavailable):
                raise PreventUpdate


celery_app.steps['worker'].add(CacheSelection)

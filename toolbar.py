from dash import html, Input, Output, ctx, callback


def construct_toolbar(enable_model_management=True, enable_runtime_admin=True):
    return html.Div([
        html.Button('Manage Models', id='manage-models', hidden=not enable_model_management),
        html.Button('Model Runtimes', id='manage-runtimes', hidden=not enable_runtime_admin),
        html.Button('Return to Hay Say', id='hay-say', hidden=True)
    ], id='toolbar', className='toolbar')


def register_toolbar_callbacks(enable_model_management=True, enable_runtime_admin=True):
    @callback(
        [Output('manage-models', 'hidden'),
         Output('manage-runtimes', 'hidden'),
         Output('hay-say', 'hidden'),
         Output('hay-say-outer-div', 'hidden'),
         Output('model-manager-outer-div', 'hidden'),
         Output('runtime-admin-outer-div', 'hidden')],
        [Input('hay-say', 'n_clicks'),
         Input('manage-models', 'n_clicks'),
         Input('manage-runtimes', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_tools_menu(*_):
        triggered = ctx.triggered_id
        view = {'manage-models': 'models', 'manage-runtimes': 'runtimes'}.get(triggered, 'hay-say')
        return (
            not enable_model_management or view == 'models',
            not enable_runtime_admin or view == 'runtimes',
            view == 'hay-say',
            view != 'hay-say',
            view != 'models',
            view != 'runtimes',
        )

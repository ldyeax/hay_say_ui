"""Dash components for inspecting and controlling native model runtimes."""

from __future__ import annotations

import datetime

from dash import ALL, Input, Output, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

import runtime_client


def _bytes(value):
    if value is None:
        return "-"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{amount:.0f} {unit}"
        amount /= 1024


def _duration(value):
    if value is None:
        return "-"
    return str(datetime.timedelta(seconds=int(value)))


def _runtime_row(item):
    runtime_id = item["id"]
    status = item.get("status", "unknown")
    running = status in runtime_client.RUNNING_STATES or status == "starting" or item.get("pid") is not None
    models = item.get("loaded_models") or []
    return html.Tr([
        html.Td([html.Strong(item.get("label", runtime_id)), html.Small(runtime_id)]),
        html.Td(status, className=f"runtime-status runtime-status-{status}"),
        html.Td(item.get("device") or "-"),
        html.Td(", ".join(models) if models else "-"),
        html.Td("-" if item.get("total_cpu_percent") is None else f"{item['total_cpu_percent']:.1f}%"),
        html.Td(_bytes(item.get("total_rss_bytes"))),
        html.Td(_bytes(item.get("gpu_memory_bytes"))),
        html.Td(_duration(item.get("uptime_seconds"))),
        html.Td(f"{item.get('active_jobs', 0)} / {item.get('queued_jobs', 0)}"),
        html.Td(html.Div([
            html.Button(
                "Start",
                id={"type": "runtime-action", "runtime": runtime_id, "action": "start"},
                disabled=running or not item.get("enabled", True),
                title=f"Start {item.get('label', runtime_id)}",
            ),
            html.Button(
                "Stop",
                id={"type": "runtime-action", "runtime": runtime_id, "action": "stop"},
                disabled=not running,
                title=f"Stop {item.get('label', runtime_id)}",
            ),
            html.Button(
                "Restart",
                id={"type": "runtime-action", "runtime": runtime_id, "action": "restart"},
                disabled=not running,
                title=f"Restart {item.get('label', runtime_id)}",
            ),
        ], className="runtime-actions")),
    ])


def construct_runtime_admin():
    return html.Div([
        dcc.Interval(id="runtime-admin-interval", interval=3000, n_intervals=0),
        dcc.Store(id="runtime-admin-refresh", data=0),
        html.H1("Model Runtimes"),
        html.Div(id="runtime-admin-message", role="status"),
        html.Div(id="runtime-admin-table", className="runtime-table-wrap"),
    ], id="runtime-admin-outer-div", className="outer-div runtime-admin", hidden=True)


def register_runtime_admin_callbacks():
    @callback(
        Output("runtime-admin-table", "children"),
        Input("runtime-admin-interval", "n_intervals"),
        Input("runtime-admin-refresh", "data"),
    )
    def refresh_runtime_table(*_):
        try:
            runtimes = runtime_client.list_runtimes()
        except runtime_client.RuntimeManagerError as exc:
            return html.Div(str(exc), className="runtime-admin-error")
        headings = ["Runtime", "State", "Device", "Loaded models", "CPU", "RAM", "GPU RAM", "Uptime",
                    "Jobs active / queued", "Actions"]
        return html.Table([
            html.Thead(html.Tr([html.Th(heading) for heading in headings])),
            html.Tbody([_runtime_row(item) for item in runtimes]),
        ], className="runtime-table")

    @callback(
        Output("runtime-admin-refresh", "data"),
        Output("runtime-admin-message", "children"),
        Input({"type": "runtime-action", "runtime": ALL, "action": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def perform_runtime_action(clicks):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or not any(clicks or []):
            raise PreventUpdate
        runtime_id = triggered["runtime"]
        action = triggered["action"]
        try:
            status = runtime_client.runtime_action(runtime_id, action)
            message = f"{status.get('label', runtime_id)}: {status.get('status', action)}"
        except runtime_client.RuntimeManagerError as exc:
            message = str(exc)
        return datetime.datetime.now().timestamp(), message

"""Loopback-only Flask control API for the model runtime supervisor."""

from __future__ import annotations

import argparse
from typing import Any

from .config import load_config
from .supervisor import (
    RuntimeDisabledError,
    RuntimeNotFoundError,
    RuntimeOperationError,
    RuntimeSupervisor,
    RuntimeSupervisorError,
    UnsafeProcessError,
)


def create_app(supervisor: RuntimeSupervisor) -> Any:
    # Import lazily so lifecycle tooling and unit tests do not need Flask merely
    # to inspect configuration or supervise processes.
    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.get("/runtimes")
    def list_runtimes() -> Any:
        return jsonify({"runtimes": supervisor.list_statuses()})

    @app.get("/runtimes/<runtime_id>")
    def get_runtime(runtime_id: str) -> Any:
        return jsonify(supervisor.status(runtime_id))

    @app.post("/runtimes/<runtime_id>/start")
    def start_runtime(runtime_id: str) -> Any:
        result = supervisor.start(runtime_id)
        return jsonify(result), 202 if result["status"] == "starting" else 200

    @app.post("/runtimes/<runtime_id>/stop")
    def stop_runtime(runtime_id: str) -> Any:
        return jsonify(supervisor.stop(runtime_id))

    @app.post("/runtimes/<runtime_id>/restart")
    def restart_runtime(runtime_id: str) -> Any:
        result = supervisor.restart(runtime_id)
        return jsonify(result), 202 if result["status"] == "starting" else 200

    @app.post("/runtimes/stop-all")
    def stop_all_runtimes() -> Any:
        return jsonify({"runtimes": supervisor.stop_all()})

    @app.errorhandler(RuntimeNotFoundError)
    def not_found(error: RuntimeNotFoundError) -> Any:
        return jsonify({"error": str(error), "type": "not_found"}), 404

    @app.errorhandler(RuntimeDisabledError)
    def disabled(error: RuntimeDisabledError) -> Any:
        return jsonify({"error": str(error), "type": "disabled"}), 409

    @app.errorhandler(UnsafeProcessError)
    def unsafe_process(error: UnsafeProcessError) -> Any:
        return jsonify({"error": str(error), "type": "unsafe_process"}), 409

    @app.errorhandler(RuntimeOperationError)
    def operation_failed(error: RuntimeOperationError) -> Any:
        return jsonify({"error": str(error), "type": "operation_failed"}), 500

    @app.errorhandler(RuntimeSupervisorError)
    def supervisor_failed(error: RuntimeSupervisorError) -> Any:
        return jsonify({"error": str(error), "type": "supervisor_error"}), 500

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Hay Say native runtime control API")
    parser.add_argument("--config", help="Path to a runtime registry JSON file")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    supervisor = RuntimeSupervisor(config)
    app = create_app(supervisor)
    app.run(
        host="127.0.0.1",
        port=config.manager.api_port,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()

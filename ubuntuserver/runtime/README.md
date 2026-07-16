# Native model runtime supervisor

This package starts each model web service as a native Ubuntu process. The
registry is `runtimes.json`; commands are argument arrays and are never passed
through a shell.

Install the small supervisor environment and start the loopback API:

```bash
python3 -m pip install -r ubuntuserver/runtime/requirements.txt
python3 -m ubuntuserver.runtime
```

The API listens on `127.0.0.1:6588` and exposes:

- `GET /health`
- `GET /runtimes`
- `GET /runtimes/<id>`
- `POST /runtimes/<id>/start`
- `POST /runtimes/<id>/stop`
- `POST /runtimes/<id>/restart`

## Configuration

Strings in the JSON file support `$NAME`, `${NAME}`, and
`${NAME:-default}` expansion. `HAY_SAY_HOME` relocates all model source and
virtual-environment paths together.

Manager fields can be overridden with `HAY_SAY_RUNTIME_STATE_DIR`,
`HAY_SAY_RUNTIME_LOG_DIR`, `HAY_SAY_RUNTIME_API_PORT`,
`HAY_SAY_RUNTIME_REQUEST_TIMEOUT`, `HAY_SAY_RUNTIME_START_GRACE`,
`HAY_SAY_RUNTIME_STOP_TIMEOUT`, and
`HAY_SAY_RUNTIME_COLLECT_GPU_MEMORY`.

Runtime-specific overrides use the upper-case runtime id. For example:

```bash
export HAY_SAY_RUNTIME_RVC_DEVICE=cuda:1
export HAY_SAY_RUNTIME_RVC_ENABLED=false
export HAY_SAY_RUNTIME_RVC_COMMAND_JSON='["/opt/rvc/bin/python", "/opt/rvc/main.py"]'
export HAY_SAY_RUNTIME_RVC_ENV_JSON='{"CUDA_VISIBLE_DEVICES":"1"}'
```

Each model server may implement `GET /runtime` to report warm-cache state:

```json
{
  "status": "warm-idle",
  "device": "cuda:0",
  "loaded_models": ["model-name"],
  "active_jobs": 0,
  "queued_jobs": 0,
  "last_error": null
}
```

Older servers without this endpoint are reported as `ready-cold` once their
HTTP listener responds. A runtime endpoint can report `starting`,
`ready-cold`, `warm-idle`, `busy`, or `error`.

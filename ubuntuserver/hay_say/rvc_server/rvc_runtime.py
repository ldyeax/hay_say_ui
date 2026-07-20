"""Compatibility import for the maintained persistent RVC runtime."""

from hay_say_common.persistent_inference import (
    PersistentModelRuntime,
    PersistentModelWorker,
    WorkerSpec,
    device_key,
    positive_environment_int,
)

RvcRuntime = PersistentModelRuntime
PersistentRvcWorker = PersistentModelWorker

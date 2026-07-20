"""Shared filesystem, cache, audio, and model-server utilities.

This local package intentionally shadows the old PyPI package. Keeping the
implementation in this repository ensures every native runtime uses the same
locking and path semantics.
"""

from . import cache, file_integration, server_utility, utility
from .cache import cache_implementation_map, select_cache_implementation
from .file_integration import (
    MODELS_DIR,
    ROOT_DIR,
    character_dir,
    characters_dir,
    guarantee_directory,
    model_pack_dirs,
    multispeaker_model_dir,
)
from .server_utility import (
    clean_up,
    construct_error_message,
    construct_full_error_message,
    get_file_list,
    get_gpu_info_from_another_venv,
    model_cpu_bf16_enabled,
    model_cpu_bf16_policy,
    runtime_state_with_cpu_bf16,
    select_hardware,
)
from .subprocess_inference import (
    InferenceCancelled,
    InferenceProcessRegistry,
    request_workspace,
)
from .persistent_inference import PersistentModelRuntime, PersistentModelWorker, WorkerSpec
from .worker_control import WorkerControl
from .utility import (
    create_link,
    get_audio_from_src_attribute,
    get_files_ending_with,
    get_files_with_extension,
    get_full_file_path,
    get_single_file_with_extension,
    get_singleton_file,
    read_audio,
)

__version__ = "1.1.0-native"

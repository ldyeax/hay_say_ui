"""Filesystem locations shared by the UI and native model runtimes."""

import os
from pathlib import Path


def _configured_path(variable, default):
    return str(Path(os.environ.get(variable, default)).expanduser().resolve())


ROOT_DIR = _configured_path("HAY_SAY_HOME", Path.home() / "hay_say")
MODELS_DIR = _configured_path("HAY_SAY_MODELS_DIR", Path(ROOT_DIR) / "models")


def guarantee_directory(directory):
    """Create *directory* when needed and return its absolute string path."""
    path = Path(directory).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def possible_model_pack_dirs(architecture_name):
    return [os.path.join(ROOT_DIR, f"{architecture_name}_model_pack_{index}") for index in range(100)]


def model_pack_dirs(architecture_name):
    """Return existing legacy model-pack directories for an architecture."""
    return [directory for directory in possible_model_pack_dirs(architecture_name) if os.path.isdir(directory)]


def characters_dir(architecture_name):
    return guarantee_directory(os.path.join(MODELS_DIR, architecture_name, "characters"))


def character_dir(architecture_name, character_name):
    return os.path.join(MODELS_DIR, architecture_name, "characters", character_name)


def multispeaker_model_dir(architecture_name, model_name):
    return os.path.join(MODELS_DIR, architecture_name, "multispeaker_models", model_name)

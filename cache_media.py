"""HTTP access to cached audio without embedding it in Dash callback JSON."""

import os
import re

from flask import abort, send_file

from hay_say_common.cache import Stage


CACHE_AUDIO_ROUTE = "/cache-audio/<stage_token>/<session_token>/<cache_key>.flac"
CACHE_AUDIO_ENDPOINT = "hay_say_cache_audio"
SHARED_SESSION_TOKEN = "shared"
CACHE_MAX_AGE_SECONDS = 3600

_SESSION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_CACHE_KEY_PATTERN = re.compile(r"^[0-9a-f]{20}$")
_STAGES_BY_TOKEN = {stage.name.lower(): stage for stage in Stage}


def _stage_token(stage):
    if isinstance(stage, Stage):
        return stage.name.lower()
    if isinstance(stage, str) and stage in _STAGES_BY_TOKEN:
        return stage
    raise ValueError("stage must be a cache Stage or canonical stage token")


def _session_token(session_id):
    if session_id is None:
        return SHARED_SESSION_TOKEN
    if isinstance(session_id, str) and _SESSION_ID_PATTERN.fullmatch(session_id):
        return session_id
    raise ValueError("session_id must be None or exactly 32 lowercase hexadecimal characters")


def _validated_cache_key(cache_key):
    if isinstance(cache_key, str) and _CACHE_KEY_PATTERN.fullmatch(cache_key):
        return cache_key
    raise ValueError("cache_key must be exactly 20 lowercase hexadecimal characters")


def cache_audio_url(stage, session_id, cache_key):
    """Build the canonical URL for one cached FLAC file."""
    return (
        f"/cache-audio/{_stage_token(stage)}/{_session_token(session_id)}/"
        f"{_validated_cache_key(cache_key)}.flac"
    )


def register_cache_audio_route(server, cache):
    """Register the validated, range-capable cached-audio endpoint."""
    if CACHE_AUDIO_ENDPOINT in server.view_functions:
        return

    def serve_cache_audio(stage_token, session_token, cache_key):
        stage = _STAGES_BY_TOKEN.get(stage_token)
        if stage is None or not _CACHE_KEY_PATTERN.fullmatch(cache_key):
            abort(404)

        if session_token == SHARED_SESSION_TOKEN:
            session_id = None
        elif _SESSION_ID_PATTERN.fullmatch(session_token):
            session_id = session_token
        else:
            abort(404)

        path = cache.audio_path(stage, session_id, cache_key)
        if not os.path.isfile(path):
            abort(404)

        try:
            response = send_file(
                path,
                mimetype="audio/flac",
                conditional=True,
                etag=True,
                max_age=CACHE_MAX_AGE_SECONDS,
            )
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            abort(404)

        response.cache_control.public = False
        response.cache_control.private = True
        return response

    server.add_url_rule(
        CACHE_AUDIO_ROUTE,
        endpoint=CACHE_AUDIO_ENDPOINT,
        view_func=serve_cache_audio,
        methods=("GET", "HEAD"),
    )

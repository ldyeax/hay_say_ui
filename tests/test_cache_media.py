import numpy
import pytest
from flask import Flask

from cache_media import cache_audio_url, register_cache_audio_route
from hay_say_common.cache import FileImpl, Stage


SESSION_ID = "0123456789abcdef0123456789abcdef"
CACHE_KEY = "0123456789abcdef0123"


@pytest.fixture
def cache_client(tmp_path, monkeypatch):
    monkeypatch.setattr(FileImpl, "AUDIO_FOLDER", str(tmp_path / "audio-cache"))
    FileImpl.write_audio_file(
        Stage.POSTPROCESSED,
        SESSION_ID,
        CACHE_KEY,
        numpy.linspace(-0.5, 0.5, 4096, dtype=numpy.float32),
        32000,
    )
    app = Flask(__name__)
    register_cache_audio_route(app, FileImpl)
    return app.test_client()


def test_cache_audio_url_supports_session_and_shared_cache():
    assert cache_audio_url(Stage.POSTPROCESSED, SESSION_ID, CACHE_KEY) == (
        f"/cache-audio/postprocessed/{SESSION_ID}/{CACHE_KEY}.flac"
    )
    assert cache_audio_url("raw", None, CACHE_KEY) == (
        f"/cache-audio/raw/shared/{CACHE_KEY}.flac"
    )


@pytest.mark.parametrize(
    "stage,session_id,cache_key",
    [
        ("unknown", SESSION_ID, CACHE_KEY),
        (Stage.RAW, "not-a-session", CACHE_KEY),
        (Stage.RAW, SESSION_ID, "too-short"),
        (Stage.RAW, SESSION_ID.upper(), CACHE_KEY),
        (Stage.RAW, SESSION_ID, CACHE_KEY.upper()),
    ],
)
def test_cache_audio_url_rejects_noncanonical_identifiers(stage, session_id, cache_key):
    with pytest.raises(ValueError):
        cache_audio_url(stage, session_id, cache_key)


def test_cache_audio_route_serves_flac_with_private_caching(cache_client):
    response = cache_client.get(cache_audio_url(Stage.POSTPROCESSED, SESSION_ID, CACHE_KEY))

    assert response.status_code == 200
    assert response.mimetype == "audio/flac"
    assert response.data.startswith(b"fLaC")
    assert response.headers["ETag"]
    assert "private" in response.headers["Cache-Control"]
    assert "max-age=3600" in response.headers["Cache-Control"]


def test_cache_audio_route_honors_byte_ranges(cache_client):
    url = cache_audio_url(Stage.POSTPROCESSED, SESSION_ID, CACHE_KEY)
    complete = cache_client.get(url)
    response = cache_client.get(url, headers={"Range": "bytes=0-15"})

    assert response.status_code == 206
    assert response.data == complete.data[:16]
    assert response.headers["Accept-Ranges"] == "bytes"
    assert response.headers["Content-Range"] == f"bytes 0-15/{len(complete.data)}"
    assert "private" in response.headers["Cache-Control"]


@pytest.mark.parametrize(
    "url",
    [
        f"/cache-audio/unknown/{SESSION_ID}/{CACHE_KEY}.flac",
        f"/cache-audio/postprocessed/not-a-session/{CACHE_KEY}.flac",
        f"/cache-audio/postprocessed/{SESSION_ID}/too-short.flac",
        f"/cache-audio/postprocessed/{SESSION_ID}/{CACHE_KEY}.wav",
        f"/cache-audio/postprocessed/{SESSION_ID}/ffffffffffffffffffff.flac",
    ],
)
def test_cache_audio_route_returns_404_for_malformed_or_missing_files(cache_client, url):
    assert cache_client.get(url).status_code == 404

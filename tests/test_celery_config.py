from celery_config import redis_url


def test_redis_url_adds_virtual_host(monkeypatch):
    monkeypatch.setenv("HAY_SAY_REDIS_URL", "redis+socket:///tmp/redis.sock")
    assert redis_url(2) == "redis+socket:///tmp/redis.sock?virtual_host=2"


def test_redis_url_preserves_other_query_parameters(monkeypatch):
    monkeypatch.setenv("HAY_SAY_REDIS_URL", "redis+socket:///tmp/redis.sock?socket_timeout=3&virtual_host=9")
    assert redis_url(1) == "redis+socket:///tmp/redis.sock?socket_timeout=3&virtual_host=1"

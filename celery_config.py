"""Shared Celery broker configuration for native and container deployments."""

import os
from urllib.parse import parse_qsl, urlencode


def redis_url(virtual_host):
    base = os.environ.get("HAY_SAY_REDIS_URL", "redis+socket:///home/luna/redis.sock")
    address, _, raw_query = base.partition("?")
    query = dict(parse_qsl(raw_query, keep_blank_values=True))
    query["virtual_host"] = str(int(virtual_host))
    return address + "?" + urlencode(query)

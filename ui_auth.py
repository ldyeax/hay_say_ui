"""Optional HTTP Basic authentication for the externally bound web UI."""

import hmac
import os

from flask import Response, request


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def _auth_enabled(username, password):
    configured = os.environ.get("HAY_SAY_UI_AUTH_ENABLED")
    if configured is None:
        return bool(username or password)
    normalized = configured.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise RuntimeError(
        "HAY_SAY_UI_AUTH_ENABLED must be one of: 1, 0, true, false, yes, no, on, off"
    )


def configure_basic_auth(server):
    username = os.environ.get("HAY_SAY_UI_USERNAME")
    password = os.environ.get("HAY_SAY_UI_PASSWORD")
    if not _auth_enabled(username, password):
        return
    if not username or not password:
        raise RuntimeError(
            "UI authentication is enabled but HAY_SAY_UI_USERNAME or HAY_SAY_UI_PASSWORD is missing"
        )

    @server.before_request
    def require_basic_auth():
        supplied = request.authorization
        valid = (
            supplied is not None
            and hmac.compare_digest(supplied.username or "", username)
            and hmac.compare_digest(supplied.password or "", password)
        )
        if valid:
            return None
        return Response(
            "Authentication required",
            401,
            {
                "WWW-Authenticate": 'Basic realm="Hay Say", charset="UTF-8"',
                "Cache-Control": "no-store",
            },
        )

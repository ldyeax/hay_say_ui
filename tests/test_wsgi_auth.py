import base64

from flask import Flask

from ui_auth import configure_basic_auth


def test_basic_auth_protects_ui(monkeypatch):
    monkeypatch.delenv("HAY_SAY_UI_AUTH_ENABLED", raising=False)
    monkeypatch.setenv("HAY_SAY_UI_USERNAME", "luna")
    monkeypatch.setenv("HAY_SAY_UI_PASSWORD", "secret")
    app = Flask(__name__)
    app.add_url_rule("/", view_func=lambda: "ok")
    configure_basic_auth(app)
    client = app.test_client()

    assert client.get("/").status_code == 401
    token = base64.b64encode(b"luna:secret").decode()
    response = client.get("/", headers={"Authorization": f"Basic {token}"})
    assert response.status_code == 200


def test_basic_auth_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.setenv("HAY_SAY_UI_AUTH_ENABLED", "0")
    monkeypatch.setenv("HAY_SAY_UI_USERNAME", "luna")
    monkeypatch.setenv("HAY_SAY_UI_PASSWORD", "secret")
    app = Flask(__name__)
    app.add_url_rule("/", view_func=lambda: "ok")

    configure_basic_auth(app)

    assert app.test_client().get("/").status_code == 200


def test_explicitly_enabled_auth_requires_credentials(monkeypatch):
    monkeypatch.setenv("HAY_SAY_UI_AUTH_ENABLED", "true")
    monkeypatch.delenv("HAY_SAY_UI_USERNAME", raising=False)
    monkeypatch.delenv("HAY_SAY_UI_PASSWORD", raising=False)
    app = Flask(__name__)

    try:
        configure_basic_auth(app)
    except RuntimeError as error:
        assert "is missing" in str(error)
    else:
        raise AssertionError("explicit authentication without credentials must fail")

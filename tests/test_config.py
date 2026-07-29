import pytest

from app.core.config import _required_secret


def test_required_secret_rejects_missing_value(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SECRET_KEY must be set"):
        _required_secret("SECRET_KEY")


def test_required_secret_rejects_placeholder(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "change-me-" + ("x" * 64))

    with pytest.raises(RuntimeError, match="strong random secret"):
        _required_secret("SECRET_KEY")


def test_required_secret_accepts_long_value(monkeypatch):
    secret = "s" * 64
    monkeypatch.setenv("SECRET_KEY", secret)

    assert _required_secret("SECRET_KEY") == secret

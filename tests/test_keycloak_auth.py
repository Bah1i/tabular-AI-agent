import asyncio

import pytest

from app.auth import keycloak
from app.auth.keycloak import (
    UserContext,
    create_session_token,
    decode_session_token,
    get_current_user_optional,
    require_admin_user,
    require_authenticated_user,
)
from app.core.config import settings


class DummyRequest:
    cookies: dict = {}


def test_keycloak_disabled_returns_dev_admin(monkeypatch):
    monkeypatch.setattr(settings, "keycloak_enabled", False)

    user = asyncio.run(get_current_user_optional(DummyRequest(), None))

    assert user is not None
    assert user.sub == "dev-anonymous"
    assert "admin" in user.roles


def test_extract_roles_from_realm_and_client(monkeypatch):
    monkeypatch.setattr(settings, "keycloak_client_id", "tabular-ai-agent")
    monkeypatch.setattr(settings, "keycloak_audience", "tabular-ai-agent")
    payload = {
        "realm_access": {"roles": ["offline_access"]},
        "resource_access": {"tabular-ai-agent": {"roles": ["admin", "viewer"]}},
    }

    assert keycloak._extract_roles(payload) == ["admin", "offline_access", "viewer"]


def test_admin_dependency_rejects_non_admin_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "keycloak_enabled", True)

    with pytest.raises(Exception):
        asyncio.run(require_admin_user(UserContext(sub="u1", roles=["viewer"])))


def test_authenticated_dependency_accepts_user_role_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "keycloak_enabled", True)

    user = asyncio.run(require_authenticated_user(UserContext(sub="u1", roles=["user"])))

    assert user.sub == "u1"


def test_signed_session_token_round_trips_user_context(monkeypatch):
    monkeypatch.setattr(settings, "session_secret_key", "test-secret")
    user = UserContext(sub="u1", email="u1@example.test", roles=["user"])

    restored = decode_session_token(create_session_token(user))

    assert restored is not None
    assert restored.sub == "u1"
    assert restored.email == "u1@example.test"
    assert restored.roles == ["user"]

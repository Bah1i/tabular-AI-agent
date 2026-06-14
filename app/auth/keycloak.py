from functools import lru_cache
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from pydantic import BaseModel, Field

from app.core.config import settings


class UserContext(BaseModel):
    sub: str
    email: str | None = None
    roles: list[str] = Field(default_factory=list)


bearer_scheme = HTTPBearer(auto_error=False)


def keycloak_browser_issuer() -> str:
    return (settings.keycloak_browser_issuer or settings.keycloak_issuer or "").rstrip("/")


def keycloak_backchannel_issuer() -> str:
    return (settings.keycloak_backchannel_issuer or settings.keycloak_issuer or "").rstrip("/")


@lru_cache(maxsize=1)
def get_jwks_client() -> PyJWKClient:
    issuer = keycloak_backchannel_issuer()
    if not issuer:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="KEYCLOAK_ISSUER is not configured")
    jwks_url = issuer + "/protocol/openid-connect/certs"
    return PyJWKClient(jwks_url)


def _extract_roles(payload: dict) -> list[str]:
    roles: set[str] = set(payload.get("realm_access", {}).get("roles", []))
    resource_access = payload.get("resource_access", {})
    client_ids = [settings.keycloak_client_id, settings.keycloak_audience]
    for client_id in [c for c in client_ids if c]:
        roles.update(resource_access.get(client_id, {}).get("roles", []))
    return sorted(roles)


def _decode_token(token: str) -> dict:
    if not settings.keycloak_issuer:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="KEYCLOAK_ISSUER is not configured")

    signing_key = get_jwks_client().get_signing_key_from_jwt(token)
    decode_options = {"verify_aud": bool(settings.keycloak_audience)}
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        issuer=settings.keycloak_issuer.rstrip("/"),
        audience=settings.keycloak_audience,
        options=decode_options,
    )


def create_session_token(user: UserContext) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.sub,
        "email": user.email,
        "roles": user.roles,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=settings.session_max_age_seconds)).timestamp()),
        "typ": "tabular-ai-session",
    }
    return jwt.encode(payload, settings.session_secret_key, algorithm="HS256")


def decode_session_token(token: str) -> UserContext | None:
    try:
        payload = jwt.decode(token, settings.session_secret_key, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    if payload.get("typ") != "tabular-ai-session":
        return None
    return UserContext(
        sub=str(payload.get("sub", "")),
        email=payload.get("email"),
        roles=[str(role) for role in payload.get("roles", [])],
    )


async def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> UserContext | None:
    if not settings.keycloak_enabled:
        return UserContext(sub="dev-anonymous", email=None, roles=["admin", "dev"])

    if credentials is not None:
        try:
            payload = _decode_token(credentials.credentials)
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token") from exc
        return UserContext(sub=str(payload.get("sub", "")), email=payload.get("email"), roles=_extract_roles(payload))

    session_token = request.cookies.get(settings.session_cookie_name)
    if session_token:
        return decode_session_token(session_token)
    return None


async def require_admin_user(user: UserContext | None = Depends(get_current_user_optional)) -> UserContext:
    if not settings.keycloak_enabled:
        return user or UserContext(sub="dev-anonymous", email=None, roles=["admin", "dev"])

    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    if "admin" not in user.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")

    return user


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept or request.url.path == "/" or request.url.path.startswith("/ui/")


async def require_authenticated_user(
    user: UserContext | None = Depends(get_current_user_optional),
    request: Request = None,
) -> UserContext:
    if not settings.keycloak_enabled:
        return user or UserContext(sub="dev-anonymous", email=None, roles=["admin", "user", "dev"])

    if user is None:
        if request is not None and _wants_html(request):
            next_url = request.url.path
            if request.url.query:
                next_url += f"?{request.url.query}"
            login_url = f"/auth/login?next={quote(next_url, safe='')}"
            raise HTTPException(
                status_code=status.HTTP_303_SEE_OTHER,
                headers={"Location": login_url},
                detail="Authentication required",
            )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    if not ({"admin", "user"} & set(user.roles)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User role required")

    return user

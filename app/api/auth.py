from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from app.auth.keycloak import (
    UserContext,
    _decode_token,
    _extract_roles,
    create_session_token,
    get_current_user_optional,
    keycloak_backchannel_issuer,
    keycloak_browser_issuer,
)
from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["auth"])

OAUTH_STATE_COOKIE = "tabular_ai_oauth_state"


def _safe_next(value: str | None) -> str:
    if not value or not value.startswith("/") or value.startswith("//"):
        return "/"
    return value


def _callback_url(request: Request) -> str:
    return str(request.url_for("auth_callback"))


def _create_state(next_url: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "next": _safe_next(next_url),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
        "typ": "tabular-ai-oauth-state",
    }
    return jwt.encode(payload, settings.session_secret_key, algorithm="HS256")


def _decode_state(value: str) -> dict:
    try:
        payload = jwt.decode(value, settings.session_secret_key, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OAuth state") from exc
    if payload.get("typ") != "tabular-ai-oauth-state":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OAuth state")
    return payload


def _set_session_cookie(response: RedirectResponse, user: UserContext) -> None:
    response.set_cookie(
        settings.session_cookie_name,
        create_session_token(user),
        max_age=settings.session_max_age_seconds,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
    )


@router.get("/login")
def login(request: Request, next: str = "/"):
    if not settings.keycloak_enabled:
        return RedirectResponse(_safe_next(next), status_code=303)
    issuer = keycloak_browser_issuer()
    if not issuer or not settings.keycloak_client_id:
        raise HTTPException(status_code=500, detail="Keycloak login is not configured")
    state = _create_state(next)
    params = urlencode(
        {
            "client_id": settings.keycloak_client_id,
            "response_type": "code",
            "scope": "openid email profile",
            "redirect_uri": _callback_url(request),
            "state": state,
        }
    )
    response = RedirectResponse(f"{issuer}/protocol/openid-connect/auth?{params}", status_code=303)
    response.set_cookie(
        OAUTH_STATE_COOKIE,
        state,
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=settings.session_cookie_secure,
    )
    return response


@router.get("/callback", name="auth_callback")
async def callback(request: Request, code: str | None = None, state: str | None = None):
    if not settings.keycloak_enabled:
        return RedirectResponse("/", status_code=303)
    expected_state = request.cookies.get(OAUTH_STATE_COOKIE)
    if not code or not state or not expected_state or state != expected_state:
        raise HTTPException(status_code=400, detail="Invalid OAuth callback")
    state_payload = _decode_state(state)
    issuer = keycloak_backchannel_issuer()
    if not issuer or not settings.keycloak_client_id:
        raise HTTPException(status_code=500, detail="Keycloak callback is not configured")

    async with httpx.AsyncClient(timeout=10.0) as client:
        token_response = await client.post(
            f"{issuer}/protocol/openid-connect/token",
            data={
                "grant_type": "authorization_code",
                "client_id": settings.keycloak_client_id,
                "code": code,
                "redirect_uri": _callback_url(request),
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if token_response.status_code >= 400:
        raise HTTPException(status_code=401, detail="Could not exchange authorization code")
    tokens = token_response.json()
    access_token = tokens.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Keycloak did not return an access token")

    payload = _decode_token(access_token)
    user = UserContext(sub=str(payload.get("sub", "")), email=payload.get("email"), roles=_extract_roles(payload))
    response = RedirectResponse(_safe_next(state_payload.get("next")), status_code=303)
    response.delete_cookie(OAUTH_STATE_COOKIE)
    _set_session_cookie(response, user)
    return response


@router.post("/logout")
def logout():
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(settings.session_cookie_name)
    response.delete_cookie(OAUTH_STATE_COOKIE)
    return response


@router.get("/me")
def me(user: UserContext | None = Depends(get_current_user_optional)):
    return {"authenticated": user is not None, "user": user.model_dump() if user else None}

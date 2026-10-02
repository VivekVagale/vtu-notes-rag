"""Who is calling, and are they the owner.

Local driver signs its own compact tokens with HMAC-SHA256 - no PyJWT, no
account, works offline. The Supabase driver later verifies that project's JWTs
instead; everything downstream only ever sees a Principal, so no endpoint
changes when the driver does.

Owner resolution fails closed: the owner is whoever signs in with OWNER_EMAIL,
compared case-insensitively. Never a header, a query parameter, or any claim the
caller can set themselves. OWNER_EMAIL unset means nobody is owner and every
moderation route refuses.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

from .config import Settings

TOKEN_TTL_S = 12 * 3600


class AuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class Principal:
    user_id: str | None = None
    email: str | None = None
    is_owner: bool = False

    @property
    def is_anonymous(self) -> bool:
        return self.user_id is None


ANONYMOUS = Principal()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class LocalTokens:
    """Signed, expiring, tamper-evident. Deliberately small."""

    def __init__(self, secret: str | None = None) -> None:
        # No secret configured: generate one per process. Tokens then stop
        # working on restart, which is correct for a dev default - it must not
        # silently behave like a stable production secret.
        self.secret = (secret or secrets.token_hex(32)).encode("utf-8")
        self.ephemeral = secret is None

    def issue(self, *, user_id: str, email: str, now: int | None = None) -> str:
        payload = {
            "sub": user_id,
            "email": email,
            "exp": int(now if now is not None else time.time()) + TOKEN_TTL_S,
        }
        body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        signature = _b64(hmac.new(self.secret, body.encode("ascii"), hashlib.sha256).digest())
        return f"{body}.{signature}"

    def verify(self, token: str, *, now: int | None = None) -> dict[str, object]:
        try:
            body, signature = token.split(".", 1)
        except ValueError as exc:
            raise AuthError("malformed token") from exc
        expected = _b64(hmac.new(self.secret, body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(expected, signature):
            raise AuthError("bad signature")
        try:
            payload = json.loads(_unb64(body))
        except Exception as exc:
            raise AuthError("malformed token") from exc
        if int(payload.get("exp", 0)) < int(now if now is not None else time.time()):
            raise AuthError("token expired")
        return payload


class LocalAuthDriver:
    def __init__(self, settings: Settings, tokens: LocalTokens | None = None) -> None:
        self.settings = settings
        self.tokens = tokens or LocalTokens(settings.auth_secret)

    def is_owner_email(self, email: str) -> bool:
        owner = (self.settings.owner_email or "").strip().lower()
        return bool(owner) and email.strip().lower() == owner

    def principal_from_header(self, header: str | None) -> Principal:
        if not header:
            return ANONYMOUS
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthError("expected a bearer token")
        payload = self.tokens.verify(token.strip())
        email = str(payload.get("email", ""))
        return Principal(
            user_id=str(payload.get("sub")),
            email=email,
            is_owner=self.is_owner_email(email),
        )


def get_auth(settings: Settings) -> LocalAuthDriver:
    return LocalAuthDriver(settings)

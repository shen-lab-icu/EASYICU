"""Password verification, purpose-bound signed cookies and bounded lockouts."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections import deque

SESSION_COOKIE = "__Host-easyicu_session"
CSRF_COOKIE = "__Host-easyicu_csrf"
SESSION_SECONDS = 12 * 60 * 60
LOGIN_WINDOW = 15 * 60


def _derive(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=32768, r=8, p=3,
                          dklen=32, maxmem=64 * 1024 * 1024)


def hash_password(password: str) -> dict[str, str]:
    if not password or len(password.encode("utf-8")) > 4096:
        raise ValueError("password must contain 1 to 4096 UTF-8 bytes")
    salt = secrets.token_bytes(16)
    return {"algorithm": "scrypt-n32768-r8-p3", "salt": salt.hex(),
            "hash": _derive(password, salt).hex()}


def verify_password(password: str, record: dict | None) -> bool:
    # Unknown accounts pay the same scrypt work as existing ones.
    salt = bytes.fromhex(record["salt"]) if record else bytes(16)
    expected = bytes.fromhex(record["hash"]) if record else bytes(32)
    derived = _derive(password, salt)
    return hmac.compare_digest(derived, expected) and record is not None


class Signer:
    def __init__(self, secret: bytes, clock=time.time):
        self.secret = secret
        self.clock = clock

    def issue(self, purpose: str, claims: dict, lifetime: int) -> str:
        payload = {**claims, "exp": int(self.clock()) + lifetime}
        body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
        signature = hmac.new(self.secret, f"{purpose}:{body}".encode(), "sha256").hexdigest()
        return f"{body}.{signature}"

    def verify(self, purpose: str, token: str) -> dict | None:
        try:
            if len(token) > 4096:
                return None
            body, signature = token.split(".")
            expected = hmac.new(self.secret, f"{purpose}:{body}".encode(), "sha256").hexdigest()
            if not hmac.compare_digest(signature, expected):
                return None
            claims = json.loads(base64.b64decode(body, altchars=b"-_", validate=True))
            if not isinstance(claims, dict) or type(claims.get("exp")) is not int:
                return None
            return claims if claims["exp"] > self.clock() else None
        except (ValueError, TypeError, UnicodeError):
            return None


class LoginLimiter:
    """Single-process rolling counters. Auth holds a lock across check + verify."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self.failures: dict[tuple[str, str], deque] = {}
        self.locked_until: dict[tuple[str, str], float] = {}

    def _prune(self) -> None:
        now = self.clock()
        for key in list(self.failures):
            queue = self.failures[key]
            while queue and queue[0] <= now - LOGIN_WINDOW:
                queue.popleft()
            if not queue:
                del self.failures[key]
        for key, until in list(self.locked_until.items()):
            if until <= now:
                del self.locked_until[key]

    def blocked(self, user: str, ip: str) -> bool:
        self._prune()
        return (any(key in self.locked_until for key in (("user", user), ("ip", ip)))
                or len(self.failures) >= 10000)

    def failure(self, user: str, ip: str) -> None:
        for key in (("user", user), ("ip", ip)):
            queue = self.failures.setdefault(key, deque())
            queue.append(self.clock())
            if len(queue) >= 5:
                self.locked_until[key] = self.clock() + LOGIN_WINDOW

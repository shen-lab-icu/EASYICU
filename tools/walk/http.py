"""Bounded HTTP JSON transport, pinned to one origin with no redirects/proxy."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request

from .privacy import token

MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class HTTPFailure(Exception):
    def __init__(self, status: int, code: str):
        self.status, self.code = status, token(code, "http_error")
        super().__init__(self.code)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def normalize_base(base: str) -> str:
    parsed = urllib.parse.urlsplit(base)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("base_must_be_http_origin_without_credentials")
    try:
        _ = parsed.port
    except ValueError:
        raise ValueError("base_port_invalid") from None
    return base.rstrip("/")


class HttpAPI:
    def __init__(self, base: str, *, exchange=None):
        self.base = normalize_base(base)
        self.exchange = exchange or self._exchange
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _exchange(self, method, url, body, timeout):
        encoded = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = urllib.request.Request(url, data=encoded, method=method,
                                         headers={"Accept": "application/json", "Content-Type": "application/json"})
        try:
            response = self._opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        except (TimeoutError, socket.timeout):
            raise TimeoutError("http_timeout") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError("http_timeout") from None
            raise HTTPFailure(0, "http_connection_failed") from None
        with response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            return response.code, response.headers, raw

    def request(self, method: str, path: str, body=None, *, params=None, timeout=30):
        if (not path.startswith("/api/") or "\\" in path or ".." in path
                or "%" in path or "?" in path or "#" in path or urllib.parse.urlsplit(path).netloc):
            raise ValueError("api_path_required")
        if timeout <= 0:
            raise TimeoutError("http_timeout")
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        status, headers, raw = self.exchange(method, url, body, timeout)
        if 300 <= status < 400:
            raise HTTPFailure(status, "http_redirect_refused")
        if len(raw) > MAX_RESPONSE_BYTES:
            raise HTTPFailure(status, "http_response_too_large")
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise HTTPFailure(status, "http_invalid_json") from None
        if not 200 <= status < 300:
            detail = payload.get("detail", {}) if isinstance(payload, dict) else {}
            code = detail.get("error") if isinstance(detail, dict) else None
            raise HTTPFailure(status, token(code, f"http_{status}"))
        if not isinstance(payload, dict):
            raise HTTPFailure(status, "http_object_required")
        return payload

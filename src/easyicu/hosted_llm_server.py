"""Hosted EasyICU LLM relay service.

This module exposes an OpenAI-compatible `/v1/chat/completions` endpoint so
EasyICU clients can talk to a project-owned backend instead of requiring each
end user to bring an API key.

The relay is intentionally minimal and fail-closed:
- Accept OpenAI-style chat completion requests
- Resolve local model aliases such as `hosted-default`
- Forward requests to OpenRouter with the server's own API key
- Require a shared bearer token unless explicit loopback-only development mode is set
- Apply a simple in-memory per-IP rate limit

Environment variables:
    OPENROUTER_API_KEY              Required. Server-side OpenRouter key.
    OPENROUTER_BASE_URL             Optional. Defaults to OpenRouter API v1.
    EASYICU_HOSTED_DEFAULT_MODEL    Optional. Default upstream model alias target.
    EASYICU_HOSTED_SERVER_TOKEN     Shared bearer token expected from clients.
    EASYICU_HOSTED_ALLOWED_MODELS   Optional direct model allowlist; aliases remain allowed.
    EASYICU_HOSTED_ALLOWED_ORIGINS  Optional exact browser-origin allowlist.
    EASYICU_HOSTED_TRUSTED_PROXIES Optional trusted proxy IP/CIDR list for X-Forwarded-For.
    EASYICU_HOSTED_RATE_LIMIT       Optional. Requests/minute per IP. Default 20.
    EASYICU_HOSTED_HOST             Optional. Bind host. Default 127.0.0.1.
    EASYICU_HOSTED_PORT             Optional. Bind port. Default 8787.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import ipaddress
import json
import os
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from typing import Any, Iterator, Sequence

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from easyicu.webserver.host_security import AllowedHostsMiddleware

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
HOSTED_DEFAULT_MODEL = os.getenv(
    "EASYICU_HOSTED_DEFAULT_MODEL",
    "openai/gpt-oss-120b:free",
).strip()
HOSTED_FALLBACK_MODELS = [
    item.strip()
    for item in os.getenv(
        "EASYICU_HOSTED_FALLBACK_MODELS",
        (
            "openai/gpt-oss-120b:free,"
            "google/gemma-4-31b-it:free,"
            "z-ai/glm-4.5-air:free,"
            "openrouter/free"
        ),
    ).split(",")
    if item.strip()
]
HOSTED_SERVER_TOKEN = os.getenv("EASYICU_HOSTED_SERVER_TOKEN", "").strip()
HOSTED_RATE_LIMIT = int(os.getenv("EASYICU_HOSTED_RATE_LIMIT", "20") or "20")
HOSTED_ALLOW_UNAUTHENTICATED_LOCAL = os.getenv(
    "EASYICU_HOSTED_ALLOW_UNAUTHENTICATED_LOCAL", ""
).strip().lower() in {"1", "true", "yes", "on"}
HOSTED_ALLOW_WILDCARD_ORIGIN = os.getenv(
    "EASYICU_HOSTED_ALLOW_WILDCARD_ORIGIN", ""
).strip().lower() in {"1", "true", "yes", "on"}
HOSTED_ALLOW_ANY_HOST = os.getenv(
    "EASYICU_HOSTED_ALLOW_ANY_HOST", ""
).strip().lower() in {"1", "true", "yes", "on"}
_RAW_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("EASYICU_HOSTED_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
HOSTED_ALLOWED_ORIGINS = [
    origin
    for origin in _RAW_ALLOWED_ORIGINS
    if origin != "*" or HOSTED_ALLOW_WILDCARD_ORIGIN
]
HOSTED_ALLOWED_MODELS = {
    model.strip()
    for model in os.getenv("EASYICU_HOSTED_ALLOWED_MODELS", "").split(",")
    if model.strip()
}
_RAW_ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("EASYICU_HOSTED_ALLOWED_HOSTS", "").split(",")
    if host.strip()
]
HOSTED_ALLOWED_HOSTS = [
    host for host in _RAW_ALLOWED_HOSTS if host != "*" or HOSTED_ALLOW_ANY_HOST
] or ["127.0.0.1", "localhost", "[::1]", "testserver"]


def _trusted_proxy_networks() -> (
    tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]
):
    networks = []
    for value in os.getenv("EASYICU_HOSTED_TRUSTED_PROXIES", "").split(","):
        value = value.strip()
        if value:
            networks.append(ipaddress.ip_network(value, strict=False))
    return tuple(networks)


HOSTED_TRUSTED_PROXY_NETWORKS = _trusted_proxy_networks()

MODEL_ALIASES = {
    "hosted-default": HOSTED_DEFAULT_MODEL,
}

# Per-request size ceilings. A requests/minute limit alone does not bound cost:
# 20 requests can each carry a 50 MB body, 5,000 messages and max_tokens=200000.
# These bound the *shape* of a single request; the rate limit bounds frequency.
HOSTED_MAX_BODY_BYTES = int(
    os.getenv("EASYICU_HOSTED_MAX_BODY_BYTES", str(2 * 1024 * 1024)) or 0
)
HOSTED_MAX_MESSAGES = int(os.getenv("EASYICU_HOSTED_MAX_MESSAGES", "200") or 0)
HOSTED_MAX_OUTPUT_TOKENS = int(
    os.getenv("EASYICU_HOSTED_MAX_OUTPUT_TOKENS", "8192") or 0
)
# Separate from the ceiling: a request that does not ask for a length should not
# be billed as if it asked for the maximum.
HOSTED_DEFAULT_OUTPUT_TOKENS = int(
    os.getenv("EASYICU_HOSTED_DEFAULT_OUTPUT_TOKENS", "2048") or 0
)
HOSTED_MAX_COMPLETIONS = int(os.getenv("EASYICU_HOSTED_MAX_COMPLETIONS", "1") or 0)

# Fields forwarded upstream. An allowlist (rather than copying the whole client
# payload) keeps a caller from smuggling provider-side options — routing,
# provider preferences, unbounded sampling knobs — through the relay.
HOSTED_FORWARDED_FIELDS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "seed",
        "n",
        "response_format",
        "tools",
        "tool_choice",
        "presence_penalty",
        "frequency_penalty",
    }
)

# Cap on distinct client IPs tracked for rate limiting, so the state dict cannot
# grow without bound under spoofed or rotating source addresses.
HOSTED_RATE_LIMIT_MAX_TRACKED_IPS = int(
    os.getenv("EASYICU_HOSTED_RATE_LIMIT_MAX_IPS", "4096") or 0
)

#: Upper bound on simultaneous upstream calls (see _upstream_slot).
HOSTED_MAX_CONCURRENT_UPSTREAM = int(
    os.getenv("EASYICU_HOSTED_MAX_CONCURRENT_UPSTREAM", "8") or 1
)
_UPSTREAM_SEMAPHORE: "asyncio.Semaphore | None" = None

_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_STATE: "OrderedDict[str, deque[float]]" = OrderedDict()


def _require_openrouter_key() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required to run the hosted LLM server."
        )


def _is_loopback_address(value: str) -> bool:
    if value.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value.strip()).is_loopback
    except ValueError:
        return False


def _validate_security_configuration(bind_host: str) -> None:
    if "*" in _RAW_ALLOWED_ORIGINS and not HOSTED_ALLOW_WILDCARD_ORIGIN:
        raise RuntimeError(
            "Wildcard CORS requires EASYICU_HOSTED_ALLOW_WILDCARD_ORIGIN=true."
        )
    if "*" in _RAW_ALLOWED_HOSTS and not HOSTED_ALLOW_ANY_HOST:
        raise RuntimeError(
            "Wildcard Host access requires EASYICU_HOSTED_ALLOW_ANY_HOST=true."
        )
    if HOSTED_SERVER_TOKEN:
        return
    if HOSTED_ALLOW_UNAUTHENTICATED_LOCAL and _is_loopback_address(bind_host):
        return
    raise RuntimeError(
        "EASYICU_HOSTED_SERVER_TOKEN is required. For explicit loopback-only "
        "development, set EASYICU_HOSTED_ALLOW_UNAUTHENTICATED_LOCAL=true."
    )


def _peer_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _peer_is_trusted_proxy(peer: str) -> bool:
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(address in network for network in HOSTED_TRUSTED_PROXY_NETWORKS)


def _client_ip(request: Request) -> str:
    peer = _peer_ip(request)
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded and _peer_is_trusted_proxy(peer):
        chain = []
        for candidate in [*forwarded.split(","), peer]:
            try:
                chain.append(str(ipaddress.ip_address(candidate.strip())))
            except ValueError:
                continue
        for candidate in reversed(chain):
            if not _peer_is_trusted_proxy(candidate):
                return candidate
        if chain:
            return chain[0]
    return peer


def _check_rate_limit(client_ip: str) -> None:
    if HOSTED_RATE_LIMIT <= 0:
        return

    now = time.time()
    window_start = now - 60
    with _RATE_LIMIT_LOCK:
        # Evict IPs whose window has fully expired, then bound the table by LRU.
        # Without this the dict grows one entry per distinct source address seen
        # since boot.
        stale = [
            ip
            for ip, seen in _RATE_LIMIT_STATE.items()
            if not seen or seen[-1] < window_start
        ]
        for ip in stale:
            if ip != client_ip:
                _RATE_LIMIT_STATE.pop(ip, None)

        bucket = _RATE_LIMIT_STATE.setdefault(client_ip, deque())
        _RATE_LIMIT_STATE.move_to_end(client_ip)
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= HOSTED_RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for {client_ip}. Limit={HOSTED_RATE_LIMIT}/min",
            )
        bucket.append(now)

        while (
            HOSTED_RATE_LIMIT_MAX_TRACKED_IPS > 0
            and len(_RATE_LIMIT_STATE) > HOSTED_RATE_LIMIT_MAX_TRACKED_IPS
        ):
            _RATE_LIMIT_STATE.popitem(last=False)


def _check_auth(request: Request) -> None:
    if not HOSTED_SERVER_TOKEN:
        if HOSTED_ALLOW_UNAUTHENTICATED_LOCAL and _is_loopback_address(
            _peer_ip(request)
        ):
            return
        raise HTTPException(
            status_code=503,
            detail="Hosted service authentication is not configured.",
        )

    auth_header = request.headers.get("authorization", "")
    expected = f"Bearer {HOSTED_SERVER_TOKEN}"
    if not hmac.compare_digest(auth_header, expected):
        raise HTTPException(status_code=401, detail="Invalid hosted service token.")


def _resolve_model(model_name: str | None) -> str:
    candidate = str(model_name or "").strip() or "hosted-default"
    if candidate in MODEL_ALIASES:
        return MODEL_ALIASES[candidate]
    if candidate in HOSTED_ALLOWED_MODELS:
        return candidate
    raise HTTPException(status_code=400, detail="Requested model is not allowed.")


def _build_upstream_headers(request: Request) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://easyicu.local",
        "X-Title": "EasyICU Hosted LLM",
    }
    return headers


async def _read_bounded_body(request: Request) -> bytes:
    """Read the request body, refusing anything over the configured ceiling.

    Streams and stops at the limit so an oversized (or lying Content-Length)
    request never gets fully buffered in memory.
    """

    limit = HOSTED_MAX_BODY_BYTES
    declared = request.headers.get("content-length")
    if limit > 0 and declared:
        try:
            if int(declared) > limit:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body exceeds {limit} bytes.",
                )
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid Content-Length header."
            ) from None

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if limit > 0 and total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Request body exceeds {limit} bytes.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@asynccontextmanager
async def _upstream_slot():
    """Bound how many upstream calls are in flight at once.

    Without this the threadpool is the only limit, so a burst of slow upstream
    calls exhausts it and starves every other route.
    """

    global _UPSTREAM_SEMAPHORE
    if _UPSTREAM_SEMAPHORE is None:
        _UPSTREAM_SEMAPHORE = asyncio.Semaphore(HOSTED_MAX_CONCURRENT_UPSTREAM)
    async with _UPSTREAM_SEMAPHORE:
        yield


def _strict_int(value: Any, field: str) -> int:
    """Accept only a real integer — not 8192.9, not "8192", not True.

    ``int(8192.9)`` silently truncates, which let a value pass the ceiling check
    in one form and reach the provider in another.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPException(
            status_code=400,
            detail=f"{field!r} must be an integer, got {type(value).__name__}.",
        )
    return value


def _validate_request_shape(payload: dict[str, Any]) -> dict[str, Any]:
    """Reject oversized requests and return a NORMALISED payload.

    Validating a coerced copy while forwarding the caller's raw value let
    ``max_tokens: 8192.9`` (or the string ``"8192"``) pass the ceiling check and
    still reach the provider un-normalised. The normalised values are written
    back so what was checked is what is sent.
    """

    payload = dict(payload)
    messages = payload.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="'messages' must be a list.")
        if HOSTED_MAX_MESSAGES > 0 and len(messages) > HOSTED_MAX_MESSAGES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Too many messages: {len(messages)} > " f"{HOSTED_MAX_MESSAGES}."
                ),
            )

    max_tokens = payload.get("max_tokens")
    if max_tokens is not None:
        max_tokens = _strict_int(max_tokens, "max_tokens")
        payload["max_tokens"] = max_tokens
        if max_tokens < 1:
            raise HTTPException(
                status_code=400, detail="'max_tokens' must be positive."
            )
        if HOSTED_MAX_OUTPUT_TOKENS > 0 and max_tokens > HOSTED_MAX_OUTPUT_TOKENS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"'max_tokens' exceeds the hosted ceiling: {max_tokens} > "
                    f"{HOSTED_MAX_OUTPUT_TOKENS}."
                ),
            )

    completions = payload.get("n")
    if completions is not None:
        completions = _strict_int(completions, "n")
        payload["n"] = completions
        if completions < 1 or (
            HOSTED_MAX_COMPLETIONS > 0 and completions > HOSTED_MAX_COMPLETIONS
        ):
            raise HTTPException(
                status_code=413,
                detail=f"'n' exceeds the hosted ceiling of {HOSTED_MAX_COMPLETIONS}.",
            )

    stream_value = payload.get("stream")
    if stream_value is not None and not isinstance(stream_value, bool):
        raise HTTPException(status_code=400, detail="'stream' must be a boolean.")

    return payload


def _build_upstream_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # Forward an allowlist rather than the caller's whole payload, so
    # provider-side options the relay has not vetted cannot ride along.
    upstream_payload = {
        key: value for key, value in payload.items() if key in HOSTED_FORWARDED_FIELDS
    }
    upstream_payload["model"] = _resolve_model(payload.get("model"))
    # A ceiling is not a default. Defaulting an omitted max_tokens to the
    # maximum allowed made every unspecified request bill at the ceiling.
    if HOSTED_DEFAULT_OUTPUT_TOKENS > 0:
        upstream_payload.setdefault("max_tokens", HOSTED_DEFAULT_OUTPUT_TOKENS)
    return upstream_payload


def _should_retry_with_fallback(response: requests.Response) -> bool:
    if response.status_code not in {429, 500, 502, 503, 504}:
        return False
    data = _json_or_text(response)
    message = json.dumps(data, ensure_ascii=False).lower()
    return any(
        token in message
        for token in (
            "rate",
            "limit",
            "temporarily",
            "overloaded",
            "provider returned error",
        )
    )


def _fallback_models_for(requested_model: str) -> list[str]:
    resolved_default = MODEL_ALIASES["hosted-default"]
    current = MODEL_ALIASES.get(requested_model, requested_model)
    candidates = []
    for model in HOSTED_FALLBACK_MODELS:
        resolved = MODEL_ALIASES.get(model, model)
        if resolved and resolved not in {current} and resolved not in candidates:
            candidates.append(resolved)
    if (
        current != resolved_default
        and resolved_default not in {current}
        and resolved_default not in candidates
    ):
        candidates.insert(0, resolved_default)
    return candidates


def _post_upstream(
    request: Request,
    upstream_payload: dict[str, Any],
    *,
    stream: bool,
) -> requests.Response:
    headers = _build_upstream_headers(request)
    upstream_url = f"{OPENROUTER_BASE_URL}/chat/completions"

    try:
        upstream_response = requests.post(
            upstream_url,
            headers=headers,
            json=upstream_payload,
            timeout=180,
            stream=stream,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502, detail=f"Upstream request failed: {exc}"
        ) from exc

    if stream or not _should_retry_with_fallback(upstream_response):
        return upstream_response

    fallback_payload = dict(upstream_payload)
    for fallback_model in _fallback_models_for(str(upstream_payload.get("model", ""))):
        fallback_payload["model"] = fallback_model
        upstream_response.close()
        try:
            upstream_response = requests.post(
                upstream_url,
                headers=headers,
                json=fallback_payload,
                timeout=180,
                stream=False,
            )
        except requests.RequestException:
            continue
        if upstream_response.status_code < 400 or not _should_retry_with_fallback(
            upstream_response
        ):
            return upstream_response

    return upstream_response


def _json_or_text(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"error": {"message": response.text or "Upstream request failed."}}


def _stream_upstream(response: requests.Response) -> Iterator[bytes]:
    try:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                yield chunk
    finally:
        response.close()


app = FastAPI(title="EasyICU Hosted LLM", version="1.0.0")
app.add_middleware(AllowedHostsMiddleware, allowed_hosts=HOSTED_ALLOWED_HOSTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=HOSTED_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "provider": "openrouter",
        "default_model": HOSTED_DEFAULT_MODEL,
        "rate_limit_per_minute": HOSTED_RATE_LIMIT,
        "auth_required": bool(HOSTED_SERVER_TOKEN),
        "unauthenticated_local_development": HOSTED_ALLOW_UNAUTHENTICATED_LOCAL,
        "allowed_model_aliases": sorted(MODEL_ALIASES),
        "max_body_bytes": HOSTED_MAX_BODY_BYTES,
        "max_messages": HOSTED_MAX_MESSAGES,
        "max_output_tokens": HOSTED_MAX_OUTPUT_TOKENS,
        "default_output_tokens": HOSTED_DEFAULT_OUTPUT_TOKENS,
        "max_completions": HOSTED_MAX_COMPLETIONS,
        "max_concurrent_upstream": HOSTED_MAX_CONCURRENT_UPSTREAM,
    }


@app.get("/v1/models")
def list_models(request: Request) -> dict[str, Any]:
    _check_auth(request)
    return {
        "object": "list",
        "data": [
            {
                "id": alias,
                "object": "model",
                "owned_by": "easyicu-hosted",
                "upstream_model": target,
            }
            for alias, target in MODEL_ALIASES.items()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Hosted service upstream provider is not configured.",
        )
    _check_rate_limit(_client_ip(request))

    body = await _read_bounded_body(request)
    try:
        payload = json.loads(body)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="Request body must be JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object.")
    payload = _validate_request_shape(payload)
    upstream_payload = _build_upstream_payload(payload)
    stream = bool(upstream_payload.get("stream", False))
    # _post_upstream uses a synchronous client with a 180 s timeout. Awaiting it
    # directly on the event loop would let one slow upstream call stall every
    # other request on this worker — including /health. Run it on the threadpool
    # and bound how many can be in flight at once.
    async with _upstream_slot():
        upstream_response = await run_in_threadpool(
            _post_upstream, request, upstream_payload, stream=stream
        )

    if stream:
        if upstream_response.status_code >= 400:
            data = _json_or_text(upstream_response)
            upstream_response.close()
            return JSONResponse(status_code=upstream_response.status_code, content=data)
        return StreamingResponse(
            _stream_upstream(upstream_response),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    data = _json_or_text(upstream_response)
    return JSONResponse(status_code=upstream_response.status_code, content=data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the EasyICU hosted LLM relay.")
    parser.add_argument("--host", default=os.getenv("EASYICU_HOSTED_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("EASYICU_HOSTED_PORT", "8787"))
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _require_openrouter_key()
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_security_configuration(args.host)

    import uvicorn

    uvicorn.run(
        "easyicu.hosted_llm_server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

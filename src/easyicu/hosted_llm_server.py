"""Hosted EasyICU LLM relay service.

This module exposes an OpenAI-compatible `/v1/chat/completions` endpoint so
the Streamlit webapp can talk to a project-owned backend instead of requiring
each end user to bring an API key.

The relay is intentionally minimal:
- Accept OpenAI-style chat completion requests
- Resolve local model aliases such as `hosted-default`
- Forward requests to OpenRouter with the server's own API key
- Optionally require a shared bearer token from the frontend
- Apply a simple in-memory per-IP rate limit

Environment variables:
    OPENROUTER_API_KEY              Required. Server-side OpenRouter key.
    OPENROUTER_BASE_URL             Optional. Defaults to OpenRouter API v1.
    EASYICU_HOSTED_DEFAULT_MODEL    Optional. Default upstream model alias target.
    EASYICU_HOSTED_SERVER_TOKEN     Optional. Shared bearer token expected from clients.
    EASYICU_HOSTED_RATE_LIMIT       Optional. Requests/minute per IP. Default 20.
    EASYICU_HOSTED_HOST             Optional. Bind host. Default 0.0.0.0.
    EASYICU_HOSTED_PORT             Optional. Bind port. Default 8787.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
from typing import Any, Iterator, Sequence

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse


OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
HOSTED_DEFAULT_MODEL = os.getenv("EASYICU_HOSTED_DEFAULT_MODEL", "openrouter/free").strip()
HOSTED_FALLBACK_MODELS = [
    item.strip()
    for item in os.getenv(
        "EASYICU_HOSTED_FALLBACK_MODELS",
        "openrouter/free,deepseek/deepseek-chat-v3-0324:free",
    ).split(",")
    if item.strip()
]
HOSTED_SERVER_TOKEN = os.getenv("EASYICU_HOSTED_SERVER_TOKEN", "").strip()
HOSTED_RATE_LIMIT = int(os.getenv("EASYICU_HOSTED_RATE_LIMIT", "20") or "20")
HOSTED_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("EASYICU_HOSTED_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

MODEL_ALIASES = {
    "hosted-default": HOSTED_DEFAULT_MODEL,
}

_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_STATE: dict[str, deque[float]] = {}


def _require_openrouter_key() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required to run the hosted LLM server.")


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _check_rate_limit(client_ip: str) -> None:
    if HOSTED_RATE_LIMIT <= 0:
        return

    now = time.time()
    window_start = now - 60
    with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMIT_STATE.setdefault(client_ip, deque())
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= HOSTED_RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for {client_ip}. Limit={HOSTED_RATE_LIMIT}/min",
            )
        bucket.append(now)


def _check_auth(request: Request) -> None:
    if not HOSTED_SERVER_TOKEN:
        return

    auth_header = request.headers.get("authorization", "")
    expected = f"Bearer {HOSTED_SERVER_TOKEN}"
    if auth_header != expected:
        raise HTTPException(status_code=401, detail="Invalid hosted service token.")


def _resolve_model(model_name: str | None) -> str:
    candidate = (model_name or "").strip() or "hosted-default"
    return MODEL_ALIASES.get(candidate, candidate)


def _build_upstream_headers(request: Request) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": request.headers.get("origin", "https://easyicu.local"),
        "X-Title": "EasyICU Hosted LLM",
    }
    return headers


def _build_upstream_payload(payload: dict[str, Any]) -> dict[str, Any]:
    upstream_payload = dict(payload)
    upstream_payload["model"] = _resolve_model(payload.get("model"))
    return upstream_payload


def _should_retry_with_fallback(response: requests.Response) -> bool:
    if response.status_code not in {429, 500, 502, 503, 504}:
        return False
    data = _json_or_text(response)
    message = json.dumps(data, ensure_ascii=False).lower()
    return any(token in message for token in ("rate", "limit", "temporarily", "overloaded", "provider returned error"))


def _fallback_models_for(requested_model: str) -> list[str]:
    resolved_default = _resolve_model("hosted-default")
    current = _resolve_model(requested_model)
    candidates = []
    for model in HOSTED_FALLBACK_MODELS:
        resolved = _resolve_model(model)
        if resolved and resolved not in {current} and resolved not in candidates:
            candidates.append(resolved)
    if current != resolved_default and resolved_default not in {current} and resolved_default not in candidates:
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
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

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
        if upstream_response.status_code < 400 or not _should_retry_with_fallback(upstream_response):
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=HOSTED_ALLOWED_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "provider": "openrouter",
        "default_model": HOSTED_DEFAULT_MODEL,
        "rate_limit_per_minute": HOSTED_RATE_LIMIT,
        "auth_required": bool(HOSTED_SERVER_TOKEN),
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
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
    _require_openrouter_key()
    _check_auth(request)
    _check_rate_limit(_client_ip(request))

    payload = await request.json()
    upstream_payload = _build_upstream_payload(payload)
    stream = bool(upstream_payload.get("stream"))
    upstream_response = _post_upstream(request, upstream_payload, stream=stream)

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
    parser.add_argument("--host", default=os.getenv("EASYICU_HOSTED_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("EASYICU_HOSTED_PORT", "8787")))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _require_openrouter_key()
    parser = build_parser()
    args = parser.parse_args(argv)

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

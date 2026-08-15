"""Dependency-neutral capability probes for configured LLM clients.

This module owns provider capability discovery without importing a concrete
transport implementation. Gates can therefore ask whether an injected client
supports an optional capability without acquiring the production provider (or
its offline mock fallback) as a transitive dependency.
"""

from __future__ import annotations

from typing import Any


def model_looks_vision_capable(model: str) -> bool:
    """Return the conservative name-based default for vision support."""

    lowered = (model or "").strip().lower()
    if not lowered:
        return False
    positive_tokens = (
        "gpt-4o",
        "omni",
        "vision",
        "gemini",
        "qwen-vl",
        "qwen2.5-vl",
        "vl-",
        "pixtral",
        "llava",
        "molmo",
        "internvl",
    )
    negative_tokens = (
        "coder",
        "instruct",
        "reasoner",
        "embedding",
        "rerank",
        "whisper",
        "audio",
    )
    if any(token in lowered for token in negative_tokens):
        return False
    return any(token in lowered for token in positive_tokens)


def llm_supports_vision(client: Any) -> bool:
    """Best-effort capability probe for optional figure-VLM review.

    Unknown clients fail closed. Wrappers and routers may expose their children
    through ``for_role`` or ``iter_clients``; the probe follows those public
    seams without importing or naming any concrete provider class.
    """

    if client is None:
        return False
    if hasattr(client, "supports_vision"):
        advertised = getattr(client, "supports_vision")
        try:
            return bool(advertised() if callable(advertised) else advertised)
        except Exception:
            return False
    if hasattr(client, "for_role"):
        try:
            analyzer_client = client.for_role("analyzer")
        except Exception:
            analyzer_client = None
        if analyzer_client is not None:
            return llm_supports_vision(analyzer_client)
    if hasattr(client, "iter_clients"):
        try:
            return any(llm_supports_vision(child) for child in client.iter_clients())
        except Exception:
            return False
    if hasattr(client, "complete_with_images"):
        model = getattr(client, "_model", None)
        if model is None:
            return True
        return model_looks_vision_capable(str(model))
    return False


__all__ = ["llm_supports_vision", "model_looks_vision_capable"]

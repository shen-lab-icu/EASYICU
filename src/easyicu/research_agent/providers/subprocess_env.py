"""Reviewed environment boundary for external provider subprocesses."""

from __future__ import annotations

import os
from typing import Mapping, Sequence


_SAFE_PROVIDER_ENV_KEYS = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
)

_BACKEND_ENV_KEYS = {
    "codex": ("CODEX_HOME",),
    "claude": (
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CONFIG_DIR",
    ),
}


def external_llm_opted_in(environment: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environment is None else environment
    raw = str(source.get("EASYICU_ALLOW_EXTERNAL_LLM", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def build_provider_subprocess_env(
    backend: str,
    *,
    environment: Mapping[str, str] | None = None,
    required_keys: Sequence[str] = (),
) -> dict[str, str]:
    """Return only reviewed runtime, backend-auth, and explicit input keys."""

    normalized_backend = str(backend or "").strip().lower()
    if normalized_backend not in _BACKEND_ENV_KEYS:
        raise ValueError(f"unsupported provider subprocess backend: {backend!r}")
    source = os.environ if environment is None else environment
    keys = (*_SAFE_PROVIDER_ENV_KEYS, *_BACKEND_ENV_KEYS[normalized_backend])
    selected = {
        key: str(source[key]) for key in keys if key in source and str(source[key])
    }
    for raw_key in required_keys:
        key = str(raw_key or "").strip()
        if not key or not key.replace("_", "a").isalnum():
            raise ValueError("provider subprocess required env keys must be names")
        if key in source and str(source[key]):
            selected[key] = str(source[key])
    return selected


__all__ = ["build_provider_subprocess_env", "external_llm_opted_in"]

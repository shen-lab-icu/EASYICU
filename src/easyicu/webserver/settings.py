"""Local settings store for the native web server.

The native server has no browser-session state, so settings persist to a local
JSON file under the user's home directory. **Local-first**: nothing leaves the
machine.

Invariant: ``ai_enabled`` (the external-LLM opt-in gate) defaults to
**False**. Any code path about to make an external LLM call must check it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import easyicu

_CONFIG_DIR = Path.home() / ".easyicu"
_CONFIG_PATH = _CONFIG_DIR / "webserver_settings.json"

# Only these keys are accepted from the client; unknown keys are ignored.
DEFAULTS: Dict[str, Any] = {
    "ai_enabled": False,          # external-LLM opt-in gate — OFF by default
    "language": "en",             # "en" | "zh"
    "data_mode": "demo",          # "demo" | "real"
    "evidence_gate": "strict",    # "strict" | "standard"
    "demo_patients": 20,
    "demo_duration": "24h",
    "working_dir": str(Path.home() / "easyicu" / "workspace"),
    "export_dir": str(Path.home() / "easyicu" / "exports"),
}

_COERCE = {
    "ai_enabled": bool,
    "demo_patients": int,
}


def _read_raw() -> Dict[str, Any]:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_settings() -> Dict[str, Any]:
    """Return defaults merged with whatever is persisted on disk."""
    merged = dict(DEFAULTS)
    for k, v in _read_raw().items():
        if k in DEFAULTS:
            merged[k] = v
    return merged


def update_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge-update known keys, persist, and return the full settings."""
    current = load_settings()
    for k, v in patch.items():
        if k not in DEFAULTS:
            continue
        if k in _COERCE:
            try:
                v = _COERCE[k](v)
            except (TypeError, ValueError):
                continue
        current[k] = v
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def _easyicu_version() -> str:
    version = getattr(easyicu, "__version__", None)
    if version:
        return str(version)
    try:
        from importlib.metadata import version as _v

        return _v("easyicu")
    except Exception:
        return "unknown"


def about() -> Dict[str, Any]:
    """Read-only environment facts for the Settings → About panel."""
    return {
        "version": _easyicu_version(),
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
        "config_path": str(_CONFIG_PATH),
    }

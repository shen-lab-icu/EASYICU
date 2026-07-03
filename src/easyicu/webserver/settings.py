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
import threading
from pathlib import Path
from typing import Any, Dict

import easyicu

_CONFIG_DIR = Path.home() / ".easyicu"
_CONFIG_PATH = _CONFIG_DIR / "webserver_settings.json"
_LOCK = threading.RLock()

# Only these keys are accepted from the client; unknown keys are ignored.
DEFAULTS: Dict[str, Any] = {
    "ai_enabled": False,  # external-LLM opt-in gate — OFF by default
    "language": "en",  # "en" | "zh"
    "data_mode": "demo",  # "demo" | "real"
    "evidence_gate": "strict",  # "strict" | "standard"
    "demo_patients": 20,
    "demo_duration": "24h",
    "working_dir": None,
    "export_dir": None,
    "module_folder_mode": True,
    "telemetry_enabled": False,
    "cache_cohort_frames": True,
    "agent_model_mode": "local",
    "token_budget": 120000,
    "auto_repair": True,
    "science_skills_enabled": True,
    "connector_pubmed_enabled": True,
    "connector_zotero_enabled": False,
    "mcp_tools_enabled": False,
    "prompt_contracts_enabled": True,
    "tool_audit_enabled": True,
    "remote_compute_enabled": False,
    "density": "comfortable",
    "reduce_motion": False,
}


def _optional_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_CHOICES = {
    "language": {"en", "zh"},
    "data_mode": {"demo", "real"},
    "evidence_gate": {"strict", "standard"},
    "demo_duration": {"24h", "48h", "168h"},
    "agent_model_mode": {"local", "external"},
    "density": {"comfortable", "compact"},
}


def _int_range(min_value: int, max_value: int):
    def _coerce(value: Any) -> int:
        return max(min_value, min(max_value, int(value)))

    return _coerce


def _choice(key: str):
    allowed = _CHOICES[key]

    def _coerce(value: Any) -> str:
        text = str(value).strip()
        if text not in allowed:
            raise ValueError(f"unsupported {key}: {text}")
        return text

    return _coerce


_COERCE = {
    "ai_enabled": bool,
    "language": _choice("language"),
    "data_mode": _choice("data_mode"),
    "evidence_gate": _choice("evidence_gate"),
    "demo_patients": _int_range(10, 50),
    "demo_duration": _choice("demo_duration"),
    "working_dir": _optional_path,
    "export_dir": _optional_path,
    "module_folder_mode": bool,
    "telemetry_enabled": bool,
    "cache_cohort_frames": bool,
    "agent_model_mode": _choice("agent_model_mode"),
    "token_budget": int,
    "auto_repair": bool,
    "science_skills_enabled": bool,
    "connector_pubmed_enabled": bool,
    "connector_zotero_enabled": bool,
    "mcp_tools_enabled": bool,
    "prompt_contracts_enabled": bool,
    "tool_audit_enabled": bool,
    "remote_compute_enabled": bool,
    "density": _choice("density"),
    "reduce_motion": bool,
}


def _read_raw() -> Dict[str, Any]:
    try:
        text = _CONFIG_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload, _end = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _write_settings(payload: Dict[str, Any]) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CONFIG_PATH.with_name(_CONFIG_PATH.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(_CONFIG_PATH)


def load_settings() -> Dict[str, Any]:
    """Return defaults merged with whatever is persisted on disk."""
    with _LOCK:
        merged = dict(DEFAULTS)
        for k, v in _read_raw().items():
            if k in DEFAULTS:
                merged[k] = v
        return merged


def update_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge-update known keys, persist, and return the full settings."""
    with _LOCK:
        current = dict(DEFAULTS)
        for key, value in _read_raw().items():
            if key in DEFAULTS:
                current[key] = value
        for k, v in patch.items():
            if k not in DEFAULTS:
                continue
            if k in _COERCE:
                try:
                    v = _COERCE[k](v)
                except (TypeError, ValueError):
                    continue
            current[k] = v
        _write_settings(current)
        return current


def reset_settings() -> Dict[str, Any]:
    """Reset local settings to the current backend defaults."""
    with _LOCK:
        current = dict(DEFAULTS)
        _write_settings(current)
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

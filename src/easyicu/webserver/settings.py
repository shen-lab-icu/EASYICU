"""Local settings store for the native web server.

The native server has no browser-session state, so settings persist to a local
JSON file under the user's home directory. **Local-first**: nothing leaves the
machine.

Invariant: ``ai_enabled`` (the external-LLM opt-in gate) defaults to
**False**. Any code path about to make an external LLM call must check it.

Invariant: **every key in ``DEFAULTS`` has a consumer outside this module.**
A setting the API accepts, coerces and persists but nothing ever reads is an
API that lies to its client — the write returns 200 and changes nothing. When
a feature is retired, its key moves to ``RETIRED_KEYS`` so a stale client is
told so, instead of getting a silent no-op. ``test_webserver_settings_contract``
holds both halves of that line.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List

import easyicu
from easyicu.webserver.host_security import local_access_policy
from easyicu.webserver.input_validation import parse_bool

_CONFIG_DIR = Path.home() / ".easyicu"
_CONFIG_PATH = _CONFIG_DIR / "webserver_settings.json"
_LOCK = threading.RLock()

# Only these keys are accepted from the client. Each one must have a reader
# somewhere in ``easyicu`` outside this module — see the invariant above.
DEFAULTS: Dict[str, Any] = {
    "ai_enabled": False,  # external-LLM opt-in gate — OFF by default
    "language": "en",  # "en" | "zh"
    "data_mode": "demo",  # "demo" | "real"
    "export_dir": None,
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

#: Keys this store used to accept. None of them ever had a reader: the API
#: coerced and persisted them, the Settings screen drew them as inert placards,
#: and no product code looked at the stored value. They are named here so a
#: stale client gets a 400 that says the setting is gone, rather than a 200
#: that pretends the write landed. Re-adding one is allowed — in the same
#: change that adds its consumer.
RETIRED_KEYS: Dict[str, str] = {
    "evidence_gate": "evidence enforcement is decided per agent run, not globally",
    "demo_patients": "demo screens use bounded seeded fixtures with a fixed size",
    "demo_duration": "demo time windows are illustrative and not configurable",
    "working_dir": "project roots are chosen per workflow, not globally",
    "module_folder_mode": "export layout is owned by the source registry",
    "telemetry_enabled": "this build has no telemetry collector to enable",
    "cache_cohort_frames": "cache policy is owned by each job/export",
    "agent_model_mode": "the provider is chosen per run in Agent Projects",
    "token_budget": "the provider adapter enforces its own bounded output contract",
    "auto_repair": "repair policy is recorded per run, not toggled globally",
}


def _optional_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_CHOICES = {
    "language": {"en", "zh"},
    "data_mode": {"demo", "real"},
    "density": {"comfortable", "compact"},
}


def _choice(key: str):
    allowed = _CHOICES[key]

    def _coerce(value: Any) -> str:
        text = str(value).strip()
        if text not in allowed:
            raise ValueError(f"unsupported {key}: {text}")
        return text

    return _coerce


_COERCE = {
    "ai_enabled": parse_bool,
    "language": _choice("language"),
    "data_mode": _choice("data_mode"),
    "export_dir": _optional_path,
    "science_skills_enabled": parse_bool,
    "connector_pubmed_enabled": parse_bool,
    "connector_zotero_enabled": parse_bool,
    "mcp_tools_enabled": parse_bool,
    "prompt_contracts_enabled": parse_bool,
    "tool_audit_enabled": parse_bool,
    "remote_compute_enabled": parse_bool,
    "density": _choice("density"),
    "reduce_motion": parse_bool,
}


class SettingsValidationError(ValueError):
    """Raised when a settings patch cannot be applied as written.

    A rejected write must fail loudly. The previous store skipped bad keys and
    still answered 200, so a client could not tell a stored value from a
    dropped one — and the Settings screen cheerfully reported "saved" for a
    key the backend had thrown away.
    """

    def __init__(self, rejected: List[Dict[str, str]]) -> None:
        self.rejected = rejected
        reasons = "; ".join(f"{item['key']}: {item['reason']}" for item in rejected)
        super().__init__(reasons or "settings_patch_rejected")

    @property
    def detail(self) -> Dict[str, Any]:
        return {
            "error": "settings_patch_rejected",
            "reason": str(self),
            "rejected": self.rejected,
        }


def _validate_patch(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce every key or raise, so a 200 always means the write landed."""
    accepted: Dict[str, Any] = {}
    rejected: List[Dict[str, str]] = []
    for key, value in patch.items():
        if key in RETIRED_KEYS:
            rejected.append(
                {
                    "key": key,
                    "code": "retired_setting",
                    "reason": f"'{key}' is no longer a setting: {RETIRED_KEYS[key]}",
                }
            )
            continue
        if key not in DEFAULTS:
            rejected.append(
                {
                    "key": key,
                    "code": "unknown_setting",
                    "reason": f"'{key}' is not a known setting",
                }
            )
            continue
        coerce = _COERCE.get(key)
        if coerce is None:
            accepted[key] = value
            continue
        try:
            accepted[key] = coerce(value)
        except (TypeError, ValueError) as exc:
            rejected.append(
                {
                    "key": key,
                    "code": "invalid_value",
                    "reason": f"'{key}' rejected: {exc}",
                }
            )
    if rejected:
        raise SettingsValidationError(rejected)
    return accepted


_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_POLL_SECONDS = 0.02


@contextlib.contextmanager
def _file_lock() -> Iterator[None]:
    """Serialise read-modify-write across processes, not just threads.

    ``_LOCK`` is a ``threading.RLock``, so it orders writers inside one uvicorn
    but not between two of them. ``easyicu-webapp --background`` plus a
    foreground server is enough to have both read the same file, apply
    different patches and have the second write erase the first.

    An ``O_CREAT | O_EXCL`` sidecar is the portable primitive here: it works on
    the macfuse mounts this project runs on, where ``flock`` is unreliable. On
    timeout the lock is taken anyway rather than failing the request — a stale
    lockfile from a killed process must not make Settings permanently
    unwritable, and the loser of that race is a lost preference, not data.
    """
    lock_path = _CONFIG_PATH.with_name(_CONFIG_PATH.name + ".lock")
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    handle = None
    while True:
        try:
            handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                break
            time.sleep(_LOCK_POLL_SECONDS)
        except OSError as exc:  # read-only home, exotic filesystem
            if exc.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
                break
            raise
    try:
        yield
    finally:
        if handle is not None:
            os.close(handle)
            with contextlib.suppress(OSError):
                lock_path.unlink()


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


def _merged_from_disk() -> Dict[str, Any]:
    """Defaults overlaid with whatever survives coercion on disk.

    The read path stays tolerant on purpose: a file written by an older build
    carries retired keys, and refusing to start over them would strand the
    user. Unreadable and retired values are dropped; the write path is where
    a bad value is reported.
    """
    merged = dict(DEFAULTS)
    for key, value in _read_raw().items():
        if key not in DEFAULTS:
            continue
        coerce = _COERCE.get(key)
        if coerce is not None:
            try:
                value = coerce(value)
            except (TypeError, ValueError):
                continue
        merged[key] = value
    return merged


def load_settings() -> Dict[str, Any]:
    """Return defaults merged with whatever is persisted on disk."""
    with _LOCK:
        return _merged_from_disk()


def update_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge-update known keys, persist, and return the full settings.

    Raises ``SettingsValidationError`` if any key in ``patch`` is unknown,
    retired, or fails coercion. Nothing is written in that case — a partial
    apply would leave the client's view and the file disagreeing.
    """
    accepted = _validate_patch(patch)
    with _LOCK, _file_lock():
        current = _merged_from_disk()
        current.update(accepted)
        _write_settings(current)
        return current


def reset_settings() -> Dict[str, Any]:
    """Reset local settings to the current backend defaults."""
    with _LOCK, _file_lock():
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
    """Read-only environment facts for the Settings → About panel.

    ``local_access`` is read from the live host policy rather than restated in
    the UI. The Settings screen used to print "local-only: enforced" as a
    literal string, so it would have shown a green tick even with the host
    policy widened — a privacy page grading its own homework.
    """
    return {
        "version": _easyicu_version(),
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
        "config_path": str(_CONFIG_PATH),
        "local_access": local_access_policy(),
    }

"""Hosted-instance policy at the HTTP and filesystem owner boundaries.

Only EASYICU_HOSTED=1 enables this profile. Explicit data coordinates require
absolute, symlink-resolved EASYICU_DATA_ROOTS. Host-owned project/run receipts
and generated exports remain accessible inside this instance's EASYICU_HOME;
that exception never grants the folder picker access to the home directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request


def is_hosted() -> bool:
    return os.environ.get("EASYICU_HOSTED") == "1"


def data_roots() -> tuple[Path, ...]:
    """A malformed allowlist grants nothing, including empty path segments."""
    raw = os.environ.get("EASYICU_DATA_ROOTS", "")
    if not raw:
        return ()
    roots = []
    try:
        for entry in raw.split(os.pathsep):
            path = Path(entry)
            if not entry or not path.is_absolute() or not path.is_dir():
                return ()
            resolved = path.resolve()
            if resolved == Path(resolved.anchor):
                return ()  # A hosted picker must never expose the filesystem root.
            roots.append(resolved)
    except (OSError, RuntimeError, ValueError):
        return ()
    return tuple(roots)


def path_allowed(value: Any, *, instance_owned: bool = False) -> bool:
    if not is_hosted():
        return True
    try:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            return False
        path = path.resolve()
        roots = data_roots()
        home = os.environ.get("EASYICU_HOME", "")
        if instance_owned and home and Path(home).is_absolute():
            roots = (*roots, Path(home).resolve())
        return any(path.is_relative_to(root) for root in roots)
    except (OSError, RuntimeError, ValueError):
        return False


def require_path(value: Any, *, instance_owned: bool = False) -> None:
    if not path_allowed(value, instance_owned=instance_owned):
        raise HTTPException(status_code=403, detail={
            "error": "path_outside_data_roots",
            "reason": "This path is outside the instance's permitted directories.",
        })


def unavailable(*, status_code: int = 403) -> None:
    raise HTTPException(status_code=status_code, detail={
        "error": "unavailable_when_hosted",
        "reason": "This action is managed by the instance administrator.",
    })


_ADMIN_ROUTES = frozenset({
    "/api/settings/reset",
    "/api/extensions/skills/install", "/api/extensions/mcp/install",
    "/api/extensions/state", "/api/extensions/remove", "/api/extensions/mcp/test",
    "/api/copilot/pi/provider-config", "/api/copilot/pi/provider-config/models",
    "/api/agent-runs/provider-config", "/api/agent-runs/codex-auth/login",
    "/api/agent-runs/codex-auth/cancel", "/api/agent-runs/codex-auth/logout",
    "/api/copilot/pi/research-provider/model", "/api/copilot/pi/research-provider/codex/login",
    "/api/copilot/pi/research-provider/codex/cancel", "/api/copilot/pi/research-provider/codex/logout",
})
_DISPLAY_SETTINGS = frozenset({"language", "density", "reduce_motion"})
_DATA_FIELDS = frozenset({
    "path", "paths", "source_path", "source_paths", "data_path", "data_root", "root",
    "raw_root", "out_dir", "output_dir", "export_dir", "export_path", "export_root",
    "parent_dir", "active_path", "crossdb_paths", "registered_export_path",
})
_PROJECT_FIELDS = frozenset({
    "project_dir", "project_root", "project_path", "project_parent_dir",
    "project_seed_dir", "project_seed_path",
})
_INSTANCE_READ_ROUTES = frozenset({
    "/api/workspace/summary", "/api/workspaces/register", "/api/workspaces/rename",
    "/api/workspaces/remove", "/api/workspaces/registry", "/api/workspaces/crossdb-summary",
    "/api/jobs/crossdb-summary", "/api/crossdb-review/summary",
    "/api/patient-review/drilldown", "/api/patient-review/entities", "/api/patient-review/entity",
    "/api/patient-review/table-preview", "/api/patient-review/feature", "/api/patient-review/sources",
    "/api/cohort-review/summary",
})


def _check_coordinates(payload: Any, *, instance_reads: bool = False) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in _DATA_FIELDS | _PROJECT_FIELDS and value not in (None, "", []):
                values = value if isinstance(value, list) else [value]
                for coordinate in values:
                    require_path(coordinate, instance_owned=key in _PROJECT_FIELDS or instance_reads)
            else:
                _check_coordinates(value, instance_reads=instance_reads)
    elif isinstance(payload, list):
        for value in payload:
            _check_coordinates(value, instance_reads=instance_reads)


async def guard_request(request: Request) -> None:
    """Shared API dependency: reject before any feature owner is invoked."""
    if not is_hosted():
        return
    path = request.url.path.rstrip("/")
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        if path in _ADMIN_ROUTES:
            unavailable()
        if path.startswith("/api/jobs/") and path.endswith("/open-output"):
            unavailable(status_code=503)
    instance_reads = path in _INSTANCE_READ_ROUTES
    _check_coordinates(dict(request.query_params), instance_reads=instance_reads)
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return  # FastAPI owns invalid/missing-body validation.
    if path == "/api/settings" and request.method == "POST":
        if isinstance(payload, dict) and set(payload) - _DISPLAY_SETTINGS:
            unavailable()
    _check_coordinates(payload, instance_reads=instance_reads)

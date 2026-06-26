"""Local export-source registry for the native EasyICU web UI.

The registry is the shared contract between Data Extraction, Patient/Cohort
Review, Cross-DB, Copilot, and Agent Projects. It records only local paths and
bounded metadata; patient rows are not persisted here.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from easyicu.webserver import dataio
from easyicu.webserver import settings as settings_store

_CONFIG_DIR = Path.home() / ".easyicu"
_CONFIG_PATH = _CONFIG_DIR / "webserver_sources.json"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _read_raw() -> Dict[str, Any]:
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_raw(data: Dict[str, Any]) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _norm_path(raw_path: str) -> str:
    path = Path(str(raw_path or "").strip()).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    return str(path)


def _source_id(path: str) -> str:
    return "src_" + hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]


def _source_from_path(
    path: str, label: str | None = None, registered_at: str | None = None
) -> Dict[str, Any]:
    norm = _norm_path(path)
    desc = dataio.describe_export_source(norm)
    item: Dict[str, Any] = {
        "id": _source_id(norm),
        "path": norm,
        "label": label or desc.get("label") or Path(norm).name or "local",
        "ok": bool(desc.get("ok")),
        "registered_at": registered_at or _now(),
    }
    if desc.get("ok"):
        item.update(
            {
                "database": desc.get("database"),
                "generated": desc.get("generated"),
                "modules": desc.get("modules", []),
                "summary": desc.get("summary", {}),
                "file_count": len(desc.get("files", [])),
            }
        )
    else:
        item["error"] = desc.get("error", "invalid_export")
    return item


def _dedup_paths(paths: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in paths:
        if not raw:
            continue
        path = _norm_path(raw)
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _raw_removed_paths(raw: Dict[str, Any]) -> List[str]:
    return _dedup_paths(raw.get("removed_paths") or [])


def _autodiscovered_paths() -> List[str]:
    settings = settings_store.load_settings()
    bases = [
        Path(
            str(settings.get("export_dir") or settings_store.DEFAULTS["export_dir"])
        ).expanduser(),
        Path.home() / "easyicu" / "exports",
    ]
    paths: List[str] = []
    for base in bases:
        try:
            base = base.resolve()
        except OSError:
            pass
        if not base.is_dir():
            continue
        try:
            children = sorted(base.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for child in children:
            try:
                if child.is_dir() and dataio.describe_export_source(str(child)).get(
                    "ok"
                ):
                    paths.append(str(child))
            except (
                Exception
            ):  # noqa: BLE001 - arbitrary local folders must not break registry boot.
                continue
    return _dedup_paths(paths)


def load_registry() -> Dict[str, Any]:
    raw = _read_raw()
    removed_paths = set(_raw_removed_paths(raw))
    stored = raw.get("sources") if isinstance(raw.get("sources"), list) else []
    by_path: Dict[str, Dict[str, Any]] = {}
    for item in stored:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        if _norm_path(str(item.get("path"))) in removed_paths:
            continue
        source = _source_from_path(
            str(item.get("path")),
            label=item.get("label"),
            registered_at=item.get("registered_at"),
        )
        by_path[source["path"]] = source
    for path in _autodiscovered_paths():
        if _norm_path(path) in removed_paths:
            continue
        by_path.setdefault(path, _source_from_path(path))

    sources = sorted(
        by_path.values(),
        key=lambda s: (not s.get("ok"), str(s.get("label") or "").lower()),
    )
    valid_paths = [str(s["path"]) for s in sources if s.get("ok")]
    active_path = (
        _norm_path(raw.get("active_path") or "") if raw.get("active_path") else None
    )
    if active_path not in valid_paths:
        active_path = valid_paths[0] if valid_paths else None

    crossdb_paths = [
        p for p in _dedup_paths(raw.get("crossdb_paths") or []) if p in valid_paths
    ]
    if len(crossdb_paths) < 2 and len(valid_paths) >= 2:
        crossdb_paths = valid_paths[:2]
    elif not crossdb_paths and active_path:
        crossdb_paths = [active_path]

    return {
        "ok": True,
        "sources": sources,
        "active_path": active_path,
        "crossdb_paths": crossdb_paths,
        "config_path": str(_CONFIG_PATH),
    }


def save_registry(patch: Dict[str, Any]) -> Dict[str, Any]:
    raw = _read_raw()
    removed_paths = set(_raw_removed_paths(raw))
    current = load_registry()
    existing = {s["path"]: s for s in current["sources"]}
    for item in patch.get("sources") or []:
        if isinstance(item, str):
            path, label = item, None
        elif isinstance(item, dict):
            path, label = item.get("path"), item.get("label")
        else:
            continue
        if path:
            source = _source_from_path(str(path), label=label)
            existing[source["path"]] = source
            removed_paths.discard(source["path"])

    active_path = patch.get("active_path", current.get("active_path"))
    active_path = _norm_path(active_path) if active_path else None
    crossdb_paths = patch.get("crossdb_paths", current.get("crossdb_paths") or [])
    crossdb_paths = _dedup_paths(crossdb_paths)

    persist_sources = [
        {
            "path": s["path"],
            "label": s.get("label"),
            "registered_at": s.get("registered_at"),
        }
        for s in existing.values()
    ]
    _write_raw(
        {
            "sources": persist_sources,
            "active_path": active_path,
            "crossdb_paths": crossdb_paths,
            "removed_paths": sorted(removed_paths),
            "updated_at": _now(),
        }
    )
    return load_registry()


def register_source(
    path: str, label: str | None = None, active: bool = True, crossdb: bool = True
) -> Dict[str, Any]:
    source = _source_from_path(path, label=label)
    if not source.get("ok"):
        return {"ok": False, "error": source.get("error"), "source": source}
    registry = load_registry()
    next_paths = list(registry.get("crossdb_paths") or [])
    if crossdb and source["path"] not in next_paths:
        next_paths.append(source["path"])
    return save_registry(
        {
            "sources": [source],
            "active_path": source["path"] if active else registry.get("active_path"),
            "crossdb_paths": next_paths,
        }
    )


def rename_source(path: str, label: str) -> Dict[str, Any]:
    """Rename a registered source in metadata only; never touches export files."""
    norm = _norm_path(path)
    clean_label = str(label or "").strip()
    if not clean_label:
        return {"ok": False, "error": "label_required", "path": norm}

    registry = load_registry()
    current = next(
        (s for s in registry.get("sources", []) if s.get("path") == norm), None
    )
    if current is None:
        return {"ok": False, "error": "source_not_registered", "path": norm}

    raw = _read_raw()
    stored = raw.get("sources") if isinstance(raw.get("sources"), list) else []
    updated = False
    persist_sources: List[Dict[str, Any]] = []
    for item in stored:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        item_path = _norm_path(str(item.get("path")))
        if item_path == norm:
            persist_sources.append(
                {
                    "path": norm,
                    "label": clean_label,
                    "registered_at": item.get("registered_at")
                    or current.get("registered_at"),
                }
            )
            updated = True
        else:
            persist_sources.append(
                {
                    "path": item_path,
                    "label": item.get("label"),
                    "registered_at": item.get("registered_at"),
                }
            )
    if not updated:
        persist_sources.append(
            {
                "path": norm,
                "label": clean_label,
                "registered_at": current.get("registered_at") or _now(),
            }
        )

    _write_raw(
        {
            "sources": persist_sources,
            "active_path": raw.get("active_path") or registry.get("active_path"),
            "crossdb_paths": raw.get("crossdb_paths")
            or registry.get("crossdb_paths")
            or [],
            "removed_paths": [p for p in _raw_removed_paths(raw) if p != norm],
            "updated_at": _now(),
        }
    )
    result = load_registry()
    result.update(
        {
            "action": "renamed_source_metadata",
            "path": norm,
            "label": clean_label,
            "disk_touched": False,
        }
    )
    return result


def remove_source(path: str) -> Dict[str, Any]:
    """Unregister one source without deleting or modifying its export folder."""
    norm = _norm_path(path)
    registry = load_registry()
    current = next(
        (s for s in registry.get("sources", []) if s.get("path") == norm), None
    )
    if current is None:
        return {"ok": False, "error": "source_not_registered", "path": norm}

    raw = _read_raw()
    stored = raw.get("sources") if isinstance(raw.get("sources"), list) else []
    persist_sources = [
        {
            "path": _norm_path(str(item.get("path"))),
            "label": item.get("label"),
            "registered_at": item.get("registered_at"),
        }
        for item in stored
        if isinstance(item, dict)
        and item.get("path")
        and _norm_path(str(item.get("path"))) != norm
    ]
    remaining_valid = [
        str(s["path"])
        for s in registry.get("sources", [])
        if s.get("ok") and s.get("path") != norm
    ]
    active_path = raw.get("active_path") or registry.get("active_path")
    active_path = _norm_path(active_path) if active_path else None
    if active_path == norm or active_path not in remaining_valid:
        active_path = remaining_valid[0] if remaining_valid else None

    raw_crossdb = raw.get("crossdb_paths") or registry.get("crossdb_paths") or []
    crossdb_paths = [
        p for p in _dedup_paths(raw_crossdb) if p != norm and p in remaining_valid
    ]
    removed_paths = set(_raw_removed_paths(raw))
    removed_paths.add(norm)
    _write_raw(
        {
            "sources": persist_sources,
            "active_path": active_path,
            "crossdb_paths": crossdb_paths,
            "removed_paths": sorted(removed_paths),
            "updated_at": _now(),
        }
    )
    result = load_registry()
    result.update(
        {
            "action": "unregistered_source_only",
            "removed_path": norm,
            "disk_deleted": False,
        }
    )
    return result

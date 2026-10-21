"""Which registered export the conversation recommends for a named database.

A database the researcher names may have more than one validated EasyICU
export.  The most complete one (the most ICU stays and the most modules) is
recommended.  Equally complete exports are told apart by the one the
workspace has active, then by the latest generation time.  When none of these
tells them apart, or no export has both the most stays and the most modules,
nothing is recommended and the exports are left for the researcher to choose:
a database with registered exports is never described as unregistered.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

from easyicu.databases.profiles import get_database_profile
from easyicu.webserver import sources

SelectionReason = Literal[
    "most_complete_local_dataset",
    "active_local_dataset",
    "latest_local_dataset",
]


@dataclass(frozen=True)
class RegisteredExportRecommendation:
    """The export to recommend and why, or the exports left to choose from."""

    source: Optional[Dict[str, Any]] = None
    reason: Optional[SelectionReason] = None
    choices: tuple[Dict[str, Any], ...] = ()


def _stays(row: Mapping[str, Any]) -> int:
    return int((row.get("aggregate") or {}).get("stays") or 0)


def _modules(row: Mapping[str, Any]) -> int:
    return int(row.get("module_count") or 0)


def _generated_at(row: Mapping[str, Any]) -> Optional[datetime]:
    text = str(row.get("generated") or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def database_reader_label(database: Any) -> str:
    """The name a reader knows a database by: its display name and the
    release EasyICU's catalog references ("eICU v2.0"); ``""`` for a
    database the catalog does not name.

    An export's registered label can carry an export id or a commit; a reader
    sentence names the database instead.
    """

    try:
        profile = get_database_profile(str(database or "").strip())
    except (KeyError, ValueError):
        return ""
    if profile.reference_release:
        return f"{profile.display_name} v{profile.reference_release}"
    return profile.display_name


def generated_date(row: Mapping[str, Any]) -> Optional[str]:
    """The export's generation date (``YYYY-MM-DD``), when it states one."""

    generated = _generated_at(row)
    return generated.date().isoformat() if generated is not None else None


def recommend_registered_export(
    choices: Sequence[Mapping[str, Any]],
) -> RegisteredExportRecommendation:
    """Recommend one registered export of ``choices``, or leave them to choose."""

    local = [dict(row) for row in choices if row.get("source_scope") == "registered_export"]
    if not local:
        return RegisteredExportRecommendation()
    complete = [
        row
        for row in local
        if all(
            _stays(row) >= _stays(other) and _modules(row) >= _modules(other)
            for other in local
        )
    ]
    if len(complete) == 1:
        return RegisteredExportRecommendation(complete[0], "most_complete_local_dataset")
    if not complete:
        # One export has more stays and another more modules: which is more
        # complete is the researcher's call, among those no other surpasses.
        return RegisteredExportRecommendation(
            choices=tuple(
                row
                for row in local
                if not any(
                    _stays(other) >= _stays(row)
                    and _modules(other) >= _modules(row)
                    and (_stays(other), _modules(other)) != (_stays(row), _modules(row))
                    for other in local
                )
            )
        )
    active = [row for row in complete if row.get("active") is True]
    if len(active) == 1:
        return RegisteredExportRecommendation(active[0], "active_local_dataset")
    times = [_generated_at(row) for row in complete]
    if all(time is not None for time in times):
        stamps = [time.timestamp() for time in times]
        latest = [row for row, stamp in zip(complete, stamps) if stamp == max(stamps)]
        if len(latest) == 1:
            return RegisteredExportRecommendation(latest[0], "latest_local_dataset")
    return RegisteredExportRecommendation(choices=tuple(complete))


def _registered_source_projection(
    source: Mapping[str, Any], *, active_path: Any, database_label: str
) -> Dict[str, Any]:
    """Return one path-free registered-export choice for conversational setup."""

    summary = source.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    label = str(source.get("label") or "EasyICU export").strip()[:240]
    if "/" in label or "\\" in label:
        label = f"{str(source.get('database') or 'EasyICU').upper()} export"
    modules = [
        str(item)[:120] for item in (source.get("modules") or []) if str(item).strip()
    ][:64]
    source_scope = (
        "official_demo"
        if sources.is_official_demo_source(source)
        else "registered_export"
    )
    return {
        "source_id": str(source.get("id") or "")[:80],
        # Registry labels are execution identity and may contain internal run
        # names such as "full6".  Conversational setup gets the canonical
        # database/release label instead; exact source_id remains authoritative.
        "label": label if source_scope == "official_demo" else database_label,
        "database": str(source.get("database") or "")[:80],
        "source_scope": source_scope,
        "availability": (
            "official_demo"
            if source_scope == "official_demo"
            else "available_in_easyicu"
        ),
        "generated": str(source.get("generated") or "")[:80] or None,
        "active": bool(
            str(source.get("path") or "").strip()
            and str(source.get("path") or "").strip() == str(active_path or "").strip()
        ),
        "module_count": len(modules),
        "modules": modules,
        "aggregate": {
            key: summary.get(key)
            for key in ("stays", "modules", "file_count", "total_rows")
            if isinstance(summary.get(key), (int, float))
        },
    }


def registered_sources_for_database(
    registry: Mapping[str, Any], database: Optional[str],
) -> tuple[list[Dict[str, Any]], RegisteredExportRecommendation]:
    """One catalog policy for the Copilot tool and the session source offer."""

    choices: list[Dict[str, Any]] = []
    if database:
        profile = get_database_profile(database)
        if not profile.is_public:
            raise KeyError(database)
        label = database_reader_label(profile.key)
        for row in registry.get("sources") or []:
            if not isinstance(row, Mapping) or not row.get("ok") or not row.get("id"):
                continue
            try:
                row_profile = get_database_profile(str(row.get("database") or ""))
            except KeyError:
                continue
            if (row_profile.parent_key or row_profile.key) != profile.key:
                continue
            choices.append(_registered_source_projection(
                row, active_path=registry.get("active_path"), database_label=label,
            ))
            if len(choices) >= 40:
                break
    return choices, recommend_registered_export(choices)


def registered_source_choice(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Bounded card facts, also used for the tool's numbered export choices."""

    return {
        "source_id": row.get("source_id"),
        "label": row.get("label"),
        "generated_date": generated_date(row),
        "stays": (row.get("aggregate") or {}).get("stays"),
        "module_count": row.get("module_count"),
    }


__all__ = [
    "RegisteredExportRecommendation",
    "SelectionReason",
    "database_reader_label",
    "generated_date",
    "recommend_registered_export",
    "registered_source_choice",
    "registered_sources_for_database",
]

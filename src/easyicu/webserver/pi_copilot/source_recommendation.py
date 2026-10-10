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


__all__ = [
    "RegisteredExportRecommendation",
    "SelectionReason",
    "generated_date",
    "recommend_registered_export",
]

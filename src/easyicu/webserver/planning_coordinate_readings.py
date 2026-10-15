"""Source receipts for a candidate's host-selected outcome and exposure.

Display metadata only: a keyword reading is shown only while its exact column
remains in the context the run used. Configured coordinates are never shown
as unconfirmed keyword selections. Missing legacy records add no claim;
unverifiable records explicitly remain unavailable.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from easyicu.concept.catalog import CONCEPT_DICTIONARY

PLANNING_COORDINATE_READINGS_SCHEMA = "easyicu.planning_coordinate_readings/1"
PLANNING_COORDINATE_READINGS_FILENAME = "planning_coordinate_readings.json"
_MAX_BYTES = 64_000
_ROLES = ("outcome", "exposure")
_AGGREGATIONS = {"min", "max", "mean", "median", "first", "last", "sum", "count"}
_COORDINATE = re.compile(r"[\w][\w:.+-]{0,159}")


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def context_readings(
    proposed: Iterable[Mapping[str, Any]], *,
    columns: Mapping[str, Any], configured: Mapping[str, bool],
) -> list[dict[str, Any]]:
    """Bind provenance to actual planning coordinates without guessing it."""
    by_role = {row.get("role"): row for row in proposed
               if isinstance(row, Mapping) and isinstance(row.get("role"), str)
               and row["role"] in _ROLES}
    rows = []
    for role in _ROLES:
        column = _text(columns.get(role), 160)
        if not column or _COORDINATE.fullmatch(column) is None:
            continue
        reading = by_role.get(role) or {}
        source_column = _text(reading.get("column"), 160)
        suffix = column.removeprefix(source_column + "_") if source_column else ""
        matched = column == source_column or role == "exposure" and suffix in _AGGREGATIONS
        if configured.get(role):
            rows.append({
                "role": role, "column": column, "basis": "configured",
                "concept": _text(reading.get("concept") or source_column, 160) if matched else column,
                "category": _text(reading.get("category"), 40) or None if matched else None,
                "evidence": None,
            })
        elif matched and reading.get("basis") == "keyword" and _text(reading.get("evidence"), 160):
            rows.append({
                "role": role, "column": column, "basis": "keyword",
                "concept": _text(reading.get("concept") or source_column, 160),
                "category": _text(reading.get("category"), 40) or None,
                "evidence": _text(reading.get("evidence"), 160) or None,
            })
    return rows


def write_planning_coordinate_readings(run_dir: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write the launch receipt, including a known empty reading."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
        "schema_version": PLANNING_COORDINATE_READINGS_SCHEMA,
        "readings": list(rows),
    }, ensure_ascii=False, sort_keys=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=run_dir,
                                         prefix=".coordinate-readings-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, run_dir / PLANNING_COORDINATE_READINGS_FILENAME)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def planned_context_columns(run_dir: Path) -> dict[str, Any] | None:
    """Read the run's actual context, without a mutable plan-file fallback."""
    from easyicu.research_agent.research_context.typed import parse_research_context_json

    path = run_dir / "research_context.json"
    try:
        if path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
            return None
        context = parse_research_context_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    return {"outcome": context.target_outcome, "exposure": context.primary_exposure}


def _valid_row(row: Any) -> bool:
    if not isinstance(row, Mapping) or set(row) != {
        "role", "concept", "column", "category", "basis", "evidence",
    }:
        return False
    if row.get("role") not in _ROLES or row.get("basis") not in ("keyword", "configured"):
        return False
    for key in ("concept", "column"):
        if not isinstance(row.get(key), str) or _COORDINATE.fullmatch(row[key]) is None:
            return False
    category, evidence = row.get("category"), row.get("evidence")
    return (
        (category is None or isinstance(category, str) and len(category) <= 40)
        and (
            evidence is None if row["basis"] == "configured"
            else isinstance(evidence, str) and 0 < len(evidence) <= 160
        )
    )


def load_planning_coordinate_readings(
    wrapper_dir: Path, *, context_columns: Mapping[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Return matching keyword rows, or None for an unverifiable receipt."""
    path = wrapper_dir / PLANNING_COORDINATE_READINGS_FILENAME
    if not path.exists() and not path.is_symlink():
        return []
    try:
        if path.is_symlink() or path.stat().st_size > _MAX_BYTES:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    if (
        not isinstance(record, Mapping)
        or record.get("schema_version") != PLANNING_COORDINATE_READINGS_SCHEMA
        or not isinstance(record.get("readings"), list)
        or len(record["readings"]) > len(_ROLES)
        or any(not _valid_row(row) for row in record["readings"])
        or len({row["role"] for row in record["readings"]}) != len(record["readings"])
        or context_columns is None
    ):
        return None
    return [
        dict(row) for row in record["readings"]
        if row["basis"] == "keyword" and context_columns.get(row["role"]) == row["column"]
    ]


def project_coordinate_readings(rows: Any) -> dict[str, Any] | None:
    """Bound and label the host receipt without treating it as confirmation."""
    if isinstance(rows, (list, tuple)) and not rows:
        return None
    if rows is None or not isinstance(rows, (list, tuple)) or len(rows) > len(_ROLES):
        return {"status": "unavailable", "items": []}
    if (any(not _valid_row(row) or row["basis"] != "keyword" for row in rows)
            or len({row["role"] for row in rows}) != len(rows)):
        return {"status": "unavailable", "items": []}
    items = []
    for row in rows:
        names = CONCEPT_DICTIONARY.get(row["concept"]) or ()
        items.append({
            **dict(row),
            "label_en": _text(names[0], 80) if names else None,
            "label_zh": _text(names[1], 80) if len(names) > 1 else None,
        })
    return {"status": "shown", "items": items}

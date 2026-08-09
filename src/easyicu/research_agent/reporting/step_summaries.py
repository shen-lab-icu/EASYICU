"""Authoritative step-summary lookup for reporting gates.

This module owns the small compatibility boundary between modern, ledger-bound
run artifacts and the pre-ledger filesystem layout.  Reporting gates consume
the returned summaries but must not infer authority independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from ..authority.runtime_artifacts import (
    current_successful_step_records,
    load_run_artifact_authority,
)


def step_authority_records(
    run_dir: Path,
    per_step_records: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[Sequence[Mapping[str, Any]]]:
    """Return the current ledger records, or ``None`` for legacy runs."""

    if per_step_records is not None:
        return per_step_records
    authority = load_run_artifact_authority(run_dir)
    if authority is None:
        return None
    records = authority.get("per_step_records")
    return records if isinstance(records, list) else []


def authoritative_step_summaries(
    run_dir: Path,
    per_step_records: Optional[Sequence[Mapping[str, Any]]],
) -> List[tuple[str, Mapping[str, Any]]]:
    """Return current successful summaries, or legacy filesystem summaries."""

    per_step_records = step_authority_records(run_dir, per_step_records)
    if per_step_records is not None:
        return [
            (str(record.get("step_id") or ""), summary)
            for record in current_successful_step_records(per_step_records)
            if isinstance((summary := record.get("step_summary")), Mapping)
        ]
    summaries: List[tuple[str, Mapping[str, Any]]] = []
    for path in sorted(run_dir.glob("steps/*/outputs/step_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, Mapping):
            summaries.append((path.parents[1].name, payload))
    return summaries


__all__ = ["authoritative_step_summaries", "step_authority_records"]

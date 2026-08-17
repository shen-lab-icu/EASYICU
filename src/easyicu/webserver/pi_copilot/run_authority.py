"""Resolve the run history owned by one Guided Copilot project.

Research Agent pipeline projections live inside the bounded Copilot project
workspace, while historical native Web runs live under the legacy projects
root.  This module is the single owner for merging those two read-only
histories so a newer pipeline run cannot be hidden by an older native run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from easyicu.webserver import agent_runs


def list_bound_run_history(
    *,
    study_context_id: Optional[str],
    project_root: Optional[Path | str] = None,
    limit: int = 50,
) -> Sequence[Dict[str, Any]]:
    """Return one newest-first, run-id-deduplicated project history."""

    bounded_limit = max(1, min(int(limit or 50), 200))
    histories = [
        agent_runs.list_run_history(
            study_id=study_context_id,
            limit=bounded_limit,
        )
    ]
    if project_root is not None:
        histories.append(
            agent_runs.list_run_history(
                study_id=study_context_id,
                project_root=str(Path(project_root).expanduser()),
                limit=bounded_limit,
            )
        )

    selected: dict[str, Dict[str, Any]] = {}
    for history in histories:
        for raw in history.get("runs") or []:
            if not isinstance(raw, dict):
                continue
            run_id = str(raw.get("run_id") or "").strip()
            if not run_id:
                continue
            incumbent = selected.get(run_id)
            if incumbent is None or _updated_epoch(raw) > _updated_epoch(incumbent):
                selected[run_id] = dict(raw)

    rows = sorted(
        selected.values(),
        key=lambda row: (_updated_epoch(row), str(row.get("run_id") or "")),
        reverse=True,
    )
    return rows[:bounded_limit]


def latest_bound_run_id(
    *,
    study_context_id: Optional[str],
    project_root: Optional[Path | str] = None,
) -> Optional[str]:
    """Return the latest exact run coordinate for one Copilot project."""

    rows = list_bound_run_history(
        study_context_id=study_context_id,
        project_root=project_root,
        limit=1,
    )
    if not rows:
        return None
    return str(rows[0].get("run_id") or "") or None


def _updated_epoch(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("updated_at_epoch") or 0.0)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["latest_bound_run_id", "list_bound_run_history"]

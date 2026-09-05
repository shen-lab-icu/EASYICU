"""Resolve the run history owned by one Guided Copilot project.

Research Agent pipeline projections live inside the bounded Copilot project
workspace, while historical native Web runs live under the legacy projects
root.  This module is the single owner for merging those two read-only
histories so a newer pipeline run cannot be hidden by an older native run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from easyicu.webserver import agent_runs, state_paths, study_contexts

from .contracts import PLANNER_CHECKPOINT_GATE_REASONS
from .workspace import ProjectWorkspace

_MAX_FAILURE_PROJECTION_BYTES = 64 * 1024


def _normalized_planner_gate_reason(row: Mapping[str, Any]) -> str:
    """Upgrade one legacy planner-only Provider 5xx projection on read.

    Older wrappers used the generic execution-failure code even though their
    immutable source manifest recorded that analysis never started and the
    Planner stopped on a Provider HTTP error.  Reading that closed projection
    is enough to restore the same checkpoint-resume route used by new runs;
    every other historical execution failure remains untouched.
    """

    reason = str(row.get("gate_reason") or "")
    if reason != "research_pipeline_execution_failed":
        return reason
    project_dir = Path(str(row.get("project_dir") or "")).expanduser()
    manifest = project_dir / "source_run_manifest.json"
    try:
        if manifest.is_symlink() or not manifest.is_file():
            return reason
        raw = manifest.read_bytes()
        if not raw or len(raw) > _MAX_FAILURE_PROJECTION_BYTES:
            return reason
        payload = json.loads(raw)
    except (OSError, ValueError, TypeError):
        return reason
    if not isinstance(payload, Mapping):
        return reason
    if (
        payload.get("schema_version") == "easyicu.web-research-pipeline-projection/1"
        and payload.get("status") == "failed"
        and payload.get("failure_code") == "research_pipeline_execution_failed"
        and payload.get("failure_type") == "provider_http"
        and payload.get("analysis_started") is False
    ):
        return "research_pipeline_planner_provider_unavailable"
    return reason


def research_pipeline_workspace() -> ProjectWorkspace:
    """Return the single host-owned workspace for pipeline run projections."""

    return ProjectWorkspace(state_paths.state_root() / "pi-agent" / "workspace")


def research_pipeline_project_root(study_context_id: Optional[str]) -> Path:
    """Resolve the pipeline run root from the scientific project identity."""

    return research_pipeline_workspace().project_root(
        str(study_context_id or "unbound-study")
    )


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
            row = dict(raw)
            row["gate_reason"] = _normalized_planner_gate_reason(row)
            row["development_planner_checkpoint_available"] = (
                _development_planner_checkpoint_available(row)
            )
            incumbent = selected.get(run_id)
            if incumbent is None or _updated_epoch(raw) > _updated_epoch(incumbent):
                selected[run_id] = row

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
    """Return the latest workflow-owning run coordinate for one project.

    A failed preparation attempt remains immutable history, but it does not
    replace an unchanged reviewable candidate plan. Session rebinding and the
    workflow card must select the same authority or the next system-owned
    revision loses the very plan it is meant to revise.
    """

    rows = list_bound_run_history(
        study_context_id=study_context_id,
        project_root=project_root,
        limit=50,
    )
    owner = workflow_authoritative_run(rows)
    if owner is None:
        return None
    return str(owner.get("run_id") or "") or None


def workflow_authoritative_run(
    rows: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Return the run that currently owns the user-facing workflow.

    A package-bound preparation attempt can fail before producing a plan or
    starting analysis. When it was launched from an unchanged, review-pending
    candidate plan, the failed attempt remains audit history but must not erase
    that candidate or force the researcher to spend another Planner run.
    """

    if not rows:
        return None
    latest = rows[0]
    if len(rows) < 2:
        return latest
    digest = str(latest.get("scientific_configuration_sha256") or "")
    for index, candidate in enumerate(rows[1:], start=1):
        same_configuration = bool(
            digest
            and digest
            == str(candidate.get("scientific_configuration_sha256") or "")
        )
        candidate_artifacts = {
            str(name or "").strip()
            for name in candidate.get("artifact_names") or []
        }
        candidate_waits_for_execution_upgrade = bool(
            str(candidate.get("run_status") or "") == "human_review_pending"
            and str(candidate.get("gate_reason") or "")
            == "human_plan_review_required"
            and {
                "operator_plan_approval_required",
                "plan_scientific_changes_required",
            }
            & {
                str(code or "").strip()
                for code in candidate.get("pending_review_reason_codes") or []
            }
            and "agent_plan.json" in candidate_artifacts
        )
        preceding_attempts_failed_before_plan = all(
            str(row.get("run_status") or "") == "failed"
            and str(row.get("run_type") or "") == "full"
            and "agent_plan.json"
            not in {
                str(name or "").strip()
                for name in row.get("artifact_names") or []
            }
            and str(row.get("scientific_configuration_sha256") or "") == digest
            for row in rows[:index]
        )
        if (
            same_configuration
            and candidate_waits_for_execution_upgrade
            and preceding_attempts_failed_before_plan
        ):
            return candidate
        if not preceding_attempts_failed_before_plan:
            break
    return latest


def _updated_epoch(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("updated_at_epoch") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _development_planner_checkpoint_available(row: Dict[str, Any]) -> bool:
    """Project presence only; the run owner validates chain integrity later."""

    if not _checkpoint_can_seed_planning(row):
        return False
    project_dir = Path(str(row.get("project_dir") or "")).expanduser()
    pipeline_root = project_dir / "pipeline"
    try:
        run_dirs = [
            path
            for path in pipeline_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and path.name.startswith("run_")
        ]
    except OSError:
        return False
    if len(run_dirs) != 1:
        return False
    try:
        return any(
            path.is_file()
            and not path.is_symlink()
            and path.name.startswith("progressive_planner_checkpoint_")
            and path.name.endswith(".json")
            for path in run_dirs[0].iterdir()
        )
    except OSError:
        return False


def _checkpoint_can_seed_planning(row: Mapping[str, Any]) -> bool:
    """Return whether this terminal failure stopped before scientific execution.

    Planner-owned failures are closed by their reason codes.  A Pydantic schema
    failure is broader and may also happen during analysis, so it is eligible
    only when the immutable wrapper manifest says analysis never started.  This
    lets a late plan-schema repair reuse validated Planner checkpoints without
    treating an execution failure as planning authority.
    """

    reason = str(row.get("gate_reason") or "")
    if reason in PLANNER_CHECKPOINT_GATE_REASONS:
        return True
    if reason != "research_pipeline_schema_validation_failed":
        return False
    project_dir = Path(str(row.get("project_dir") or "")).expanduser()
    manifest = project_dir / "source_run_manifest.json"
    try:
        if manifest.is_symlink() or not manifest.is_file():
            return False
        raw = manifest.read_bytes()
        if not raw or len(raw) > _MAX_FAILURE_PROJECTION_BYTES:
            return False
        payload = json.loads(raw)
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        isinstance(payload, Mapping)
        and payload.get("schema_version")
        == "easyicu.web-research-pipeline-projection/1"
        and payload.get("status") == "failed"
        and payload.get("failure_code") == reason
        and payload.get("analysis_started") is False
    )


def resumable_planner_checkpoint_job_id(
    *,
    study: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    project_root: Path | str,
) -> str:
    """Select one unchanged, bounded Planner checkpoint from run history.

    Preparation-only projections and user-cancelled retries do not own a new
    plan, so they remain visible without masking the newest valid checkpoint.
    Every other newer terminal result is an authority boundary and prevents a
    silent jump back to older planning state.
    """

    resumable_reasons = PLANNER_CHECKPOINT_GATE_REASONS
    transparent_attempt_reasons = {
        "data_foundation_blocked",
        "research_pipeline_cancelled",
    }
    candidate: Mapping[str, Any] | None = None
    for row in rows:
        reason = str(row.get("gate_reason") or "")
        if reason in resumable_reasons or _checkpoint_can_seed_planning(row):
            candidate = row
            break
        if reason in transparent_attempt_reasons:
            continue
        return ""
    if candidate is None:
        return ""
    if str(candidate.get("run_status") or "") != "failed":
        return ""
    if candidate.get("development_planner_checkpoint_available") is not True:
        return ""
    study_id = str(study.get("id") or "").strip()
    if not study_id or str(candidate.get("study_id") or "") != study_id:
        return ""
    planned_digest = str(
        candidate.get("scientific_configuration_sha256") or ""
    ).strip()
    if (
        len(planned_digest) != 64
        or planned_digest
        != study_contexts.scientific_configuration_sha256(dict(study))
    ):
        return ""
    match = re.fullmatch(
        r"run_([A-Za-z0-9][A-Za-z0-9_-]{0,79})",
        str(candidate.get("run_id") or ""),
    )
    if match is None:
        return ""
    candidate_dir = Path(str(candidate.get("project_dir") or "")).expanduser()
    try:
        expected_root = Path(project_root).expanduser().resolve(strict=True)
        resolved_candidate = candidate_dir.resolve(strict=True)
        resolved_candidate.relative_to(expected_root)
    except (FileNotFoundError, OSError, ValueError):
        return ""
    if resolved_candidate.parent.parent != expected_root:
        return ""
    return match.group(1)


__all__ = [
    "latest_bound_run_id",
    "list_bound_run_history",
    "resumable_planner_checkpoint_job_id",
    "research_pipeline_project_root",
    "research_pipeline_workspace",
]

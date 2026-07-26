"""Pure projections for execution, scientific, and paper completion state."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

from ..plan_utils import _output_declares_figure


def count_missing_evidence_markers(text: str) -> int:
    """Count unresolved evidence placeholders in manuscript text."""

    return len(
        re.findall(
            r"(?:\[evidence missing:\s*[^\]]+\]|<!--\s*evidence missing:\s*[^>]+-->)",
            text or "",
            flags=re.IGNORECASE,
        )
    )


def count_writer_attempts(run_dir: Path) -> int | None:
    """Count writer passes from the append-only run audit stream."""

    audit_path = run_dir / "audit_log.jsonl"
    if not audit_path.exists():
        return None
    count = 0
    try:
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if str(event.get("event", "")).startswith("Drafting manuscript scaffold"):
                count += 1
    except Exception:
        return None
    return count


def has_figure_only_output_contract(step: Any) -> bool:
    """Return whether every declared output is a figure-like product."""

    outputs = [
        str(output or "").strip()
        for output in (getattr(step, "expected_outputs", None) or [])
        if str(output or "").strip()
    ]
    if not outputs:
        return False
    for output in outputs:
        kind, separator, _product = output.lower().partition(":")
        if separator:
            if kind.strip() not in {"figure", "plot", "chart", "fig", "heatmap"}:
                return False
        elif not _output_declares_figure(output):
            return False
    return True


def step_completion_projection(
    *,
    required_step_ids: Sequence[str],
    record_by_step: Mapping[str, Mapping[str, Any]],
    status_by_step: Mapping[str, str],
    step_ok: Callable[[str], bool],
) -> dict[str, object]:
    """Project outer execution and closed scientific terminal states."""

    incomplete: list[dict[str, str]] = []
    states: list[dict[str, Any]] = []
    for step_id in required_step_ids:
        record = record_by_step.get(step_id, {})
        summary = record.get("step_summary")
        summary_status = (
            str(summary.get("status") or "").strip().lower()
            if isinstance(summary, Mapping)
            else ""
        )
        # Closed, host-recognised scientific terminal. Generated prose or an
        # arbitrary status-like token cannot create or clear this state.
        scientific_complete = summary_status != "completed_feasibility_failure"
        if not scientific_complete:
            incomplete.append({"step_id": step_id, "summary_status": summary_status})
        states.append(
            {
                "schema_version": "easyicu.step_completion_state/1",
                "step_id": step_id,
                "execution_ok": step_ok(step_id),
                "outer_status": status_by_step.get(step_id),
                "summary_status": summary_status or None,
                "scientific_requirement_complete": scientific_complete,
            }
        )
    return {
        "scientific_incomplete_steps": incomplete,
        "step_completion_states": states,
    }


def publication_authorized(
    *,
    manuscript_ready: bool,
    publication_figure_bundle_ready: bool,
    publication_provenance_ready: bool,
    display_suite_complete: bool,
    article_contract_complete: bool,
    article_figure_strategy_complete: bool,
    plan_not_truncated: bool = True,
) -> bool:
    """Return the existing fail-closed publication conjunction.

    ``plan_not_truncated`` closes a gap the other terms cannot see. They all
    ask whether what the run *did* is sound; none asks whether the run did
    what it planned. When a plan exceeds ``max_total_steps`` the host drops
    steps and records a warning, and every remaining step can then complete,
    bind its evidence and verify its numerics — so a run that quietly lost its
    calibration figure or its PH diagnostic reaches this conjunction looking
    exactly like one that never needed them. The dropped products are named in
    the truncation finding; this is what makes naming them binding.

    It stays out of ``manuscript_ready`` on purpose: a truncated run is still
    worth reading, iterating on, and diagnosing. It is not a paper.
    """

    return bool(
        manuscript_ready
        and publication_figure_bundle_ready
        and publication_provenance_ready
        and display_suite_complete
        and article_contract_complete
        and article_figure_strategy_complete
        and plan_not_truncated
    )


def run_completion_axes(
    *,
    execution_ok: bool,
    artifact_valid: bool,
    scientific_requirement_complete: bool,
    paper_authorized: bool,
) -> dict[str, object]:
    """Expose existing authoritative gates under four explicit user axes."""

    return {
        "completion_schema_version": "easyicu.run_completion_axes/1",
        "execution_ok": bool(execution_ok),
        "artifact_valid": bool(artifact_valid),
        "scientific_requirement_complete": bool(scientific_requirement_complete),
        "paper_authorized": bool(paper_authorized),
    }


__all__ = [
    "count_missing_evidence_markers",
    "count_writer_attempts",
    "has_figure_only_output_contract",
    "publication_authorized",
    "run_completion_axes",
    "step_completion_projection",
]

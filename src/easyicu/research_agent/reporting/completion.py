"""Pure projections for execution, scientific, and paper completion state."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
import json
from pathlib import Path
import re
from typing import Any

from ..planning.figure_step_contract import _output_declares_figure


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
    """Legacy name for the publication-content conjunction, not paper authority.

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


@dataclass(frozen=True, slots=True)
class RunCompletionFacts:
    """Verdicts supplied by the existing validators, never inferred from prose."""

    execution_complete: bool
    evidence_complete: bool
    numeric_verified: bool
    analysis_validated: bool
    publication_figure_bundle_ready: bool
    publication_provenance_ready: bool
    display_suite_complete: bool
    article_contract_complete: bool
    article_figure_strategy_complete: bool
    scientific_maturity_article_grade: bool
    plan_truncated: bool
    replan_budget_exhausted: bool
    administrative_metadata_verified: bool

    def __post_init__(self) -> None:
        for field in fields(self):
            if type(getattr(self, field.name)) is not bool:
                raise TypeError(f"completion_fact_requires_boolean: {field.name}")


def _content_status(
    *, diagnostic: bool, publication: bool, manuscript: bool, execution: bool
) -> str:
    if diagnostic:
        return "diagnostic_only"
    if publication:
        return "publication_ready"
    if manuscript:
        return "manuscript_ready"
    return "analysis_only" if execution else "diagnostic_only"


@dataclass(frozen=True, slots=True)
class RunCompletionDecision:
    """One final completion decision, including content and execution authority.

    The default execution identity cannot authorize a paper. Reporting hosts
    consume the finished projection; there is no intermediate paper permission
    for an artifact writer to correct later. This composes existing verdicts,
    and does not replace plan approval, validation, or human sign-off owners.
    """

    facts: RunCompletionFacts
    forced_diagnostic_only: bool = False
    execution_paper_eligible: bool = False
    plan_authority_verified: bool = False
    plan_authority_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.facts, RunCompletionFacts):
            raise TypeError("completion_requires_typed_facts")
        for name in (
            "forced_diagnostic_only",
            "execution_paper_eligible",
            "plan_authority_verified",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"completion_authority_requires_boolean: {name}")
        digest = str(self.plan_authority_sha256 or "").strip().lower()
        bound = bool(
            self.plan_authority_verified and re.fullmatch(r"[0-9a-f]{64}", digest)
        )
        object.__setattr__(self, "plan_authority_verified", bound)
        object.__setattr__(self, "plan_authority_sha256", digest if bound else None)

    @property
    def manuscript_ready(self) -> bool:
        return (
            self.facts.execution_complete
            and self.facts.evidence_complete
            and self.facts.numeric_verified
            and self.facts.analysis_validated
        )

    @property
    def publication_ready(self) -> bool:
        return (
            publication_authorized(
                manuscript_ready=self.manuscript_ready,
                publication_figure_bundle_ready=self.facts.publication_figure_bundle_ready,
                publication_provenance_ready=self.facts.publication_provenance_ready,
                display_suite_complete=self.facts.display_suite_complete,
                article_contract_complete=self.facts.article_contract_complete,
                article_figure_strategy_complete=self.facts.article_figure_strategy_complete,
                plan_not_truncated=not self.facts.plan_truncated,
            )
            and self.facts.scientific_maturity_article_grade
        )

    @property
    def status(self) -> str:
        return _content_status(
            diagnostic=self.forced_diagnostic_only
            or self.facts.replan_budget_exhausted,
            publication=self.publication_ready,
            manuscript=self.manuscript_ready,
            execution=self.facts.execution_complete,
        )

    @property
    def publication_artifacts_ready(self) -> bool:
        return self.publication_ready and self.status == "publication_ready"

    @property
    def paper_authorized(self) -> bool:
        return (
            self.publication_artifacts_ready
            and self.execution_paper_eligible
            and self.plan_authority_verified
        )

    def to_gates(self) -> dict[str, object]:
        return {
            **{
                field.name: getattr(self.facts, field.name)
                for field in fields(self.facts)
            },
            "completion_schema_version": "easyicu.run_completion_axes/1",
            "completion_status": self.status,
            "execution_ok": self.facts.execution_complete,
            "artifact_valid": self.facts.evidence_complete,
            "scientific_requirement_complete": self.facts.analysis_validated,
            "manuscript_ready": self.manuscript_ready,
            "publication_ready": self.publication_ready,
            # Existing content + administrative-metadata readiness; final
            # scientific permission is exposed separately as paper_authorized.
            "submission_ready": self.publication_ready
            and self.facts.administrative_metadata_verified,
            "publication_artifacts_ready": self.publication_artifacts_ready,
            "execution_paper_eligible": self.execution_paper_eligible,
            "plan_authority_verified": self.plan_authority_verified,
            "plan_authority_sha256": self.plan_authority_sha256,
            "paper_authorized": self.paper_authorized,
            "forced_diagnostic_only": self.forced_diagnostic_only,
        }


def readiness_status(gates: Mapping[str, Any]) -> str:
    """Read a completed projection, with the same ladder for legacy reports."""
    expected = _content_status(
        diagnostic=bool(
            gates.get("forced_diagnostic_only") or gates.get("replan_budget_exhausted")
        ),
        publication=bool(gates.get("publication_ready")),
        manuscript=bool(gates.get("manuscript_ready")),
        execution=bool(gates.get("execution_complete")),
    )
    if "completion_status" in gates and gates["completion_status"] != expected:
        raise ValueError("completion_status_projection_mismatch")
    return expected


__all__ = [
    "count_missing_evidence_markers",
    "count_writer_attempts",
    "has_figure_only_output_contract",
    "RunCompletionDecision",
    "RunCompletionFacts",
    "publication_authorized",
    "readiness_status",
    "step_completion_projection",
]

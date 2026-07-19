"""GateEvaluator — the execute-phase figure Visual-QA gate as a real module.

Extracted from ``pipeline_execute.py`` (batch: real cross-file GateEvaluator
module). This owns the typed Visual-QA gate: the cosmetic-demotion predicate,
the ``VisualGateResult`` collection, and the ``VisualRepairDecision`` — all
side-effect-free w.r.t. pipeline runtime state (they read figure files / call
the auditor but mutate no step_record / findings / budget / evidence / lock and
drive no control flow). ``pipeline_execute`` re-exports every public name here so
existing imports keep working; the AST contract in
``tests/research_agent/test_gate_evaluator_contract.py`` locks the boundary.

Imports only leaf modules (contracts / repairs.reasons / scalar_utils /
visual_qa) so there is no import cycle with pipeline_execute.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..contracts import ValidationFinding
from ..repairs.reasons import typed_repair_ticket
from ..scalar_utils import _expected_numeric_annotations_for_step
from .visual_qa import VisualQAAuditor


def _visual_repair_request_log(
    findings: Sequence[ValidationFinding],
) -> str:
    """Keep visual-repair scope and structured collision details together."""

    payload = json.dumps(
        [
            {
                "validator": finding.validator,
                "severity": finding.severity,
                "message": finding.message,
                "detail": finding.detail,
            }
            for finding in findings
        ],
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    return (
        "STRUCTURED VISUAL FINDINGS (diagnostic mirror; not routing authority):\n"
        + payload
    )


_COSMETIC_VISUAL_REASON = "svg_text_overlap_spacing"
_LEGACY_COSMETIC_VISUAL_MESSAGE = re.compile(
    r"^svg figure '[^']+' has overlapping text elements; "
    r"multi-panel labels, annotations or axis text need more spacing\.?$",
    re.IGNORECASE,
)
_HARD_VISUAL_MESSAGE = re.compile(
    r"\b(?:blank|clip(?:ped|ping)?|crop(?:ped|ping)?|missing|absent|"
    r"unreadable|overflow|truncat(?:ed|ion)|numeric|mismatch|disagree)\b",
    re.IGNORECASE,
)


def _is_cosmetic_visual_finding(finding: ValidationFinding) -> bool:
    """Return true only for the closed SVG spacing reason safe to demote."""

    if finding.severity != "error" or finding.validator != "visual_qa":
        return False
    message = str(finding.message or "").strip()
    if _HARD_VISUAL_MESSAGE.search(message):
        return False
    detail = finding.detail if isinstance(finding.detail, Mapping) else {}
    if str(detail.get("reason") or "").strip() == _COSMETIC_VISUAL_REASON:
        return True
    # Preserve readiness for historical deterministic findings that predate
    # the reason enum, but require the complete canonical message rather than
    # two permissive substrings.
    return _LEGACY_COSMETIC_VISUAL_MESSAGE.fullmatch(message) is not None


def _demote_cosmetic_visual_findings(
    findings: Sequence[ValidationFinding],
) -> tuple[List[ValidationFinding], List[ValidationFinding]]:
    """Demote cosmetic visual errors and return remaining hard errors."""

    demoted: List[ValidationFinding] = []
    for finding in findings:
        if _is_cosmetic_visual_finding(finding):
            demoted.append(finding.model_copy(update={"severity": "warning"}))
        else:
            demoted.append(finding)
    blocking_errors = [f for f in demoted if f.severity == "error"]
    return demoted, blocking_errors


@dataclass(frozen=True)
class VisualGateResult:
    """Inert, typed outcome of the figure Visual-QA audit for one step.

    Batch 1a-2 — the first typed slice of the execute-phase GateEvaluator. This
    frozen value holds the raw audit findings, the error subset, and the
    cosmetic-demotion projection, and NOTHING about control flow. (Producing it
    does read figure files + invoke the auditor — see ``collect_visual_gate_result``
    — but the value itself carries no side effects.) ``_execute_one_step`` maps
    it onto continue / return / step-status / budget / lock / evidence; the gate
    only reports.
    """

    ran: bool
    findings: Tuple[ValidationFinding, ...]
    error_findings: Tuple[ValidationFinding, ...]
    demoted_findings: Tuple[ValidationFinding, ...]
    blocking_errors: Tuple[ValidationFinding, ...]

    @property
    def has_errors(self) -> bool:
        return bool(self.error_findings)

    @property
    def has_blocking_errors(self) -> bool:
        return bool(self.blocking_errors)

    @property
    def was_demoted(self) -> bool:
        """True when at least one hard error was demoted to a warning."""
        return any(
            original.severity == "error" and demoted.severity == "warning"
            for original, demoted in zip(self.findings, self.demoted_findings)
        )


def collect_visual_gate_result(
    *,
    enabled: bool,
    step_figures: Sequence[Path],
    step: Any,
    step_summary: Mapping[str, Any],
) -> VisualGateResult:
    """Run the figure Visual-QA audit and classify it.

    This is NOT a pure function: it reads the step's figure files and invokes
    ``VisualQAAuditor``. What it has is no *pipeline runtime-state* side
    effects — it mutates no ``step_record`` / ``findings`` / budget / evidence /
    lock and drives no control flow; the orchestrator owns all of that.

    Mirrors the original inline ``VisualQAAuditor().audit_with_expected`` +
    ``visual_errors`` + ``_demote_cosmetic_visual_findings`` sequence from
    ``_execute_one_step``. Returns ``ran=False`` with empty finding tuples when
    Visual QA is disabled or the step produced no figures — exactly the guard
    the orchestrator used to spell inline. The cosmetic-demotion projection is
    computed eagerly; on the clean (no-error) path it is a no-op that leaves
    ``demoted_findings == findings``.
    """

    if not (enabled and step_figures):
        return VisualGateResult(
            ran=False,
            findings=(),
            error_findings=(),
            demoted_findings=(),
            blocking_errors=(),
        )

    expected_numeric = _expected_numeric_annotations_for_step(
        step=step,
        step_summary=step_summary,
    )
    numeric_expectations = (
        {
            str(path): expected_numeric
            for path in step_figures
            if path.suffix.lower() == ".svg"
        }
        if expected_numeric
        else None
    )
    visual_findings = VisualQAAuditor().audit_with_expected(
        figure_paths=step_figures,
        expected_numeric_by_path=numeric_expectations,
    )
    error_findings = [f for f in visual_findings if f.severity == "error"]
    demoted_findings, blocking_errors = _demote_cosmetic_visual_findings(
        visual_findings
    )
    return VisualGateResult(
        ran=True,
        findings=tuple(visual_findings),
        error_findings=tuple(error_findings),
        demoted_findings=tuple(demoted_findings),
        blocking_errors=tuple(blocking_errors),
    )


class VisualRepairAction(str, Enum):
    """Which of the three top-level visual-error branches the step should take.

    Only meaningful when the gate found blocking visual errors; the concrete
    side effects (demote / terminal / fallback-retry / capsule repair) are the
    orchestrator's, driven by these actions plus its own budget/attempt state.
    """

    SEALED_SUPPRESS = "sealed_suppress"  # verified sealed renderer — never rewrite
    EXHAUSTED = "exhausted"  # out of layout-repair attempts or LLM budget
    LLM_REPAIR = "llm_repair"  # request one bounded LLM layout repair


@dataclass(frozen=True)
class VisualRepairDecision:
    """Side-effect-free recommendation for how to handle visual errors.

    Batch 1a-2 — the decision half of the typed Visual-QA gate. It selects the
    branch and, for ``LLM_REPAIR``, carries the repair RECOMMENDATION (the base
    typed repair ticket, the layout-only host guidance, and the operator repair
    log). It never calls an LLM, consumes budget, builds a RepairPromptAuthority,
    touches evidence, or decides control flow — the orchestration layer does all
    of that. The monotonic-concept regression constraints are appended to the
    authority by the orchestrator, not here.
    """

    action: VisualRepairAction
    reason: str
    repair_ticket: Tuple[Dict[str, Any], ...] = ()
    host_guidance: Optional[Dict[str, Any]] = None
    repair_log: str = ""


def decide_visual_repair(
    result: VisualGateResult,
    *,
    sealed: bool,
    attempts_exhausted: bool,
    budget_available: bool,
) -> Optional[VisualRepairDecision]:
    """Classify the visual-error branch and build the repair recommendation.

    Mirrors the original ``if sealed / elif exhausted-or-no-budget / else``
    selection inside ``_execute_one_step`` plus the ``LLM_REPAIR`` payload
    construction. Returns ``None`` when the gate found no errors (no decision to
    make). Pure: ``sealed`` / ``attempts_exhausted`` / ``budget_available`` are
    the caller's already-evaluated state reads.
    """

    if not result.has_errors:
        return None

    if sealed:
        return VisualRepairDecision(
            action=VisualRepairAction.SEALED_SUPPRESS,
            reason="sealed_renderer_authorized_code",
        )

    if attempts_exhausted or not budget_available:
        if attempts_exhausted and not budget_available:
            reason = "max_visual_repair_attempts_and_no_llm_budget"
        elif attempts_exhausted:
            reason = "max_visual_repair_attempts"
        else:
            reason = "no_llm_repair_budget"
        return VisualRepairDecision(
            action=VisualRepairAction.EXHAUSTED,
            reason=reason,
        )

    host_guidance = {
        "layout_only": True,
        "preserve": [
            "source_data_values_and_rows",
            "step_summary_numeric_and_statistical_values",
            "figure_contract_claims_evidence_and_panel_roles",
        ],
        "forbid": [
            "source_resolution_changes",
            "cohort_or_data_transformations",
            "estimate_or_scientific_label_changes",
        ],
    }
    repair_log = (
        "Visual QA rejected one or more figure outputs "
        "before evidence registration. Fix the figure "
        "layout, preserve all tables/statistics, save PNG "
        "and editable SVG with the same stem, include "
        "publication figure exports when requested, and rerun.\n\n"
        + _visual_repair_request_log(list(result.findings))
    )
    return VisualRepairDecision(
        action=VisualRepairAction.LLM_REPAIR,
        reason="llm_layout_repair",
        repair_ticket=tuple(typed_repair_ticket(list(result.findings))),
        host_guidance=host_guidance,
        repair_log=repair_log,
    )

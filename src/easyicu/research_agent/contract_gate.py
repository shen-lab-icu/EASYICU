"""Contract gates — deterministic contract / figure-contract findings, as a module.

Batch: real cross-file contract-gate extraction (Codex-ordered, after the visual
GateEvaluator move). This is the home for the deterministic-contract family of
gates that only READ step state and RETURN findings; control flow, repair, and
evidence authority stay in the execution layer (``pipeline_execute``).

It starts with the figure-contract gate (``_post_canonicalization_figure_findings``),
which is dependency-clean. The larger ``_step_deterministic_contract_findings``
gate is NOT moved yet: it is entangled with a ~10-function primary-exposure /
contract cluster (``_primary_exposure_*`` / ``_primary_model_leakage_findings`` /
``_step_contract_findings`` + module constants) and needs its own dedicated
untangling batch before it can join here without a cascade. See
``task_logs/20260718_contract_gate_extraction.md``.

Imports only leaf modules so there is no import cycle with ``pipeline_execute``;
``pipeline_execute`` re-exports every public name here for back-compat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .audits.validators import (
    FigureContractQualityValidator,
    FigureSourceDataValidator,
)
from .contracts import ValidationFinding
from .ordered_stratified_contract import ordered_stratified_numeric_findings
from .schema import AnalysisStep


def _post_canonicalization_figure_findings(
    *,
    step: AnalysisStep,
    out_dir: Path,
    run_dir: Path,
    step_summary: Dict[str, Any],
    completed_step_records: Sequence[Mapping[str, Any]],
    resolved_input_bindings: Mapping[str, Mapping[str, Any]],
    execution_cohort_path: Path,
    figure_contract_validator: FigureContractQualityValidator,
    figure_source_validator: FigureSourceDataValidator,
) -> List[ValidationFinding]:
    """Figure-contract / figure-source / ordered-stratified findings evaluated
    AFTER the early figure-contract canonicalization repair.

    These are kept OUT of ``_step_deterministic_contract_findings`` on purpose:
    the early pre-registration gate must interleave the figure-contract
    canonicalization repair BETWEEN the shared contract sequence and these figure
    audits (the audits must see the already-canonicalized contracts). So the
    early gate calls the shared contract sequence, then runs the canonicalization
    repair inline, then calls this — preserving that hard ordering while still
    lifting the figure-audit block out of the execution loop.
    """

    findings: List[ValidationFinding] = figure_contract_validator.audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary=step_summary,
    )
    findings += figure_source_validator.audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary=step_summary,
        completed_step_records=completed_step_records,
        resolved_input_bindings=resolved_input_bindings,
    )
    # For the controlled ordered-stratified method, replay the agent-authored
    # tables from the locked cohort before evidence registration. Numeric/method
    # errors therefore return to the existing coder repair loop instead of
    # becoming a late warning.
    findings += ordered_stratified_numeric_findings(
        cohort_path=execution_cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary=step_summary,
    )
    return findings


__all__ = ["_post_canonicalization_figure_findings"]

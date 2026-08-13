"""Run-level robustness-panel transaction.

This module owns the execute-finalisation boundary for the pre-specified
robustness panel: recover the plan-time lock, collect only already-authorised
estimates, persist the panel, and project its run-manifest fields.  It never
chooses an exposure, outcome, cohort, estimator, or primary estimand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from ..schema import ValidationFinding
from .estimators import fit_robustness_rows_from_records
from .panel import (
    build_robustness_panel_from_records,
    robustness_specs_for_execution,
    unexecuted_locked_spec_ids,
    write_robustness_panel,
)

if TYPE_CHECKING:
    from ..authority.evidence_store import EvidenceStore


@dataclass(frozen=True)
class RunRobustnessPanelResult:
    """Immutable host handoff from robustness finalisation."""

    findings: tuple[ValidationFinding, ...]
    panel_path: Optional[str] = None
    n_variants: Optional[int] = None
    range_low: Optional[float] = None
    range_high: Optional[float] = None

    def manifest_update(self) -> Dict[str, Any]:
        """Return a fresh manifest projection for a successfully written panel."""

        if self.panel_path is None:
            return {}
        return {
            "robustness_panel_path": self.panel_path,
            "robustness_n_variants": self.n_variants,
            "robustness_range_low": self.range_low,
            "robustness_range_high": self.range_high,
        }


def finalize_run_robustness_panel(
    *,
    run_dir: Path,
    plan: Any,
    per_step_records: Sequence[Dict[str, Any]],
    cohort_path: Optional[Path],
    context: Any,
    evidence: "EvidenceStore",
    prompt_pack_version: Optional[str],
) -> RunRobustnessPanelResult:
    """Build and persist the locked run-level robustness disclosure panel.

    Finalisation deliberately disables implicit cohort refitting.  Variant
    rows must come from an exact registered primary-script replay; otherwise a
    locked-but-unexecuted specification is surfaced as a fail-closed finding.
    Artifact-construction failures remain isolated warnings so the caller can
    report the precise robustness failure alongside other run findings.
    """

    findings: list[ValidationFinding] = []
    try:
        robustness_specs = robustness_specs_for_execution(run_dir=run_dir, plan=plan)
        if robustness_specs and not list(getattr(plan, "robustness_specs", []) or []):
            findings.append(
                ValidationFinding(
                    validator="robustness_panel",
                    severity="warning",
                    message=(
                        "Recovered robustness_specs from the plan-time lock because "
                        "the active replanned AnalysisPlan no longer carried them."
                    ),
                )
            )

        adapter_rows, adapter_warnings = fit_robustness_rows_from_records(
            specs=robustness_specs,
            per_step_records=per_step_records,
            primary_cohort=getattr(plan, "cohort", None),
            cohort_path=cohort_path,
            context=context,
            run_dir=run_dir,
            allow_implicit_cohort_refit=False,
        )
        findings.extend(
            ValidationFinding(
                validator="robustness_estimator",
                severity="warning",
                message=warning,
            )
            for warning in adapter_warnings
        )

        panel = build_robustness_panel_from_records(
            specs=robustness_specs,
            per_step_records=per_step_records,
            adapter_rows=adapter_rows,
        )
        write_robustness_panel(
            run_dir=run_dir,
            panel=panel,
            evidence=evidence,
            prompt_pack_version=prompt_pack_version,
        )

        unexecuted_locked_specs = unexecuted_locked_spec_ids(panel)
        if unexecuted_locked_specs:
            findings.append(
                ValidationFinding(
                    validator="robustness_panel",
                    severity="error",
                    message=(
                        "The run locked robustness specifications that no step "
                        "estimated, so the panel carries them as blank rows: "
                        + ", ".join(unexecuted_locked_specs)
                        + ". A step re-estimating the locked grid must declare "
                        "robustness_replay_spec to reach the deterministic "
                        "replay owner; generic refitting stays disabled here "
                        "because it would choose an exposure, outcome or "
                        "method on the plan's behalf."
                    ),
                    detail={
                        "unexecuted_spec_ids": unexecuted_locked_specs,
                        "primary_spec_id": panel.primary_spec_id,
                        "locked_spec_count": len(robustness_specs),
                        "panel_path": "robustness_panel.json",
                    },
                )
            )

        return RunRobustnessPanelResult(
            findings=tuple(findings),
            panel_path="robustness_panel.json",
            n_variants=panel.n_variants,
            range_low=panel.range_low,
            range_high=panel.range_high,
        )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="robustness_panel",
                severity="warning",
                message=f"Robustness panel artifact could not be built: {exc}",
            )
        )
        return RunRobustnessPanelResult(findings=tuple(findings))


__all__ = ["RunRobustnessPanelResult", "finalize_run_robustness_panel"]

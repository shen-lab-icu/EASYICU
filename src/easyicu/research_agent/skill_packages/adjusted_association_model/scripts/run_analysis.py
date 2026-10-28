"""Call the exact host model owner, retaining its matrix and covariance rules."""

import json
from pathlib import Path

from ....contracts.association_execution import association_estimator_kind
from ....execution.runners.adjusted_association_executor import run_adjusted_association_from_env
from ...reference_io import LoadedCohort
from ..spec import AdjustedAssociationReferenceSpec


def run_analysis(loaded: LoadedCohort, spec: AdjustedAssociationReferenceSpec, out_dir: str | Path) -> dict:
    requirement = spec.model
    retention = requirement.baseline_missing_handling
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = run_adjusted_association_from_env(
        frame=loaded.frame, cohort_path=loaded.source_path, output_dir=out,
        requirement_id=requirement.requirement_id, exposure=requirement.exposure_source,
        outcome=requirement.outcome, covariates=list(requirement.covariates or []),
        model_terms=list(requirement.model_terms or []),
        estimator_kind=association_estimator_kind(requirement),
        method_family=requirement.method_family, analysis_set=requirement.analysis_set,
        analysis_role=requirement.analysis_role, primary_contrast_level=requirement.primary_contrast_level,
        dependence=requirement.dependence,
        missing_category_covariates=list(retention.covariates) if retention is not None else (),
        emit_step_summary=False,
    )
    (out / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    return summary

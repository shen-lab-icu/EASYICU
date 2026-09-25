"""Measure, before approval, how much of its population the primary model keeps.

The scientific review used to judge missing-data handling from labels alone:
a plan whose primary model would silently fit a minority of its cohort scored
as well as one that fitted all of it.  This host probe reads the cohort file
execution will bind and applies the primary owner's own population and row
rules (``model_matrix.primary_model_rows`` and the landmark runners), so it
lives with those owners; the pipeline entry calls it and hands the review
typed evidence (``contracts.model_retention``), which the review turns into
findings.  It never raises for a data problem: an unreadable or unevaluable
cohort is itself evidence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..authority.current_case_scientific_runtime import (
    LandmarkCategoricalAssociationRuntimeAuthority,
    LandmarkSplineRuntimeAuthority,
)
from ..authority.declared_levels import execution_model_requirement
from ..contracts.model_retention import (
    MISSING_CATEGORY_MIN_ROWS,
    CovariateRetention,
    PrimaryModelRetentionEvidence,
    RequirementRetention,
)
from ..contracts.model_terms import PlannedModelRequirement
from .model_matrix import (
    ModelTermCompilationError,
    compile_model_terms,
    primary_model_rows,
)
from ..schema import AnalysisPlan, AnalysisStep

_ANALYSIS_COHORT_INPUTS = frozenset({"artifact:analysis_cohort", "dataset:analysis_cohort"})


def resolve_review_cohort_path(
    *,
    run_dir: Path,
    plan: AnalysisPlan,
    universe_path: Optional[Path],
    materialization: Optional[Mapping[str, Any]],
    cohort_concept_ids: Sequence[str] = (),
) -> Optional[Path]:
    """Return the cohort file execution binds as ``COHORT_PARQUET``.

    A fresh plan passes its materialization result; a reused plan recovers the
    one its plan phase closed, exactly as cohort adoption does, and otherwise
    reads the universe (``no_definition``).
    """

    if materialization is not None:
        status = materialization.get("status")
        if status == "applied":
            return Path(materialization["path"])
        return universe_path if status == "no_definition" else None
    from ..cohort.schema import load_materialized_analysis_cohort_result

    recovered = load_materialized_analysis_cohort_result(
        run_dir=Path(run_dir), plan=plan, cohort_concept_ids=cohort_concept_ids
    )
    if recovered is not None:
        return Path(recovered["path"])
    return universe_path


def _metadata_only(context: Any) -> bool:
    provenance = getattr(getattr(context, "cohort", None), "provenance", None) or {}
    return str(provenance.get("evidence_stage") or "") == "metadata_only_planning"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sole_primary(step: AnalysisStep) -> Optional[PlannedModelRequirement]:
    primary = [
        item
        for item in (step.model_requirements or ())
        if item.analysis_role == "primary"
    ]
    return primary[0] if len(primary) == 1 else None


def _generic_primary(plan: AnalysisPlan) -> Optional[tuple[AnalysisStep, PlannedModelRequirement]]:
    candidates = []
    for step in plan.steps:
        if step.planned_analysis_role != "primary":
            continue
        requirement = _sole_primary(step)
        if requirement is not None and requirement.model_terms:
            candidates.append((step, requirement))
    return candidates[0] if len(candidates) == 1 else None


def _rate(outcome: Any, mask: Any) -> Optional[float]:
    n = int(mask.sum())
    if n < MISSING_CATEGORY_MIN_ROWS:
        return None
    return round(float(outcome.loc[mask].mean()), 4)


def _requirement_retention(
    *,
    step_id: str,
    requirement: PlannedModelRequirement,
    population: Any,
) -> RequirementRetention:
    import pandas as pd

    terms = list(requirement.model_terms or ())
    exposure = requirement.exposure_source
    outcome_column = requirement.outcome
    listed = requirement.missing_category_covariates()
    compiled = compile_model_terms(population, terms=terms, exposure=exposure)
    design = compiled.design
    outcome = pd.to_numeric(population[outcome_column], errors="coerce").astype(float)
    evaluable = outcome.notna() & design[list(compiled.exposure_columns)].notna().all(axis=1)
    rows = primary_model_rows(
        population,
        terms=terms,
        exposure=exposure,
        outcome=outcome_column,
        missing_category_covariates=listed,
    )
    kept_rows = population.index.isin(rows.design.index)
    columns_by_source: dict[str, list[str]] = {}
    for column, source in compiled.source_by_design_column.items():
        columns_by_source.setdefault(source, []).append(column)
    kept_states = {item.covariate for item in rows.missing_category_terms}
    dropped_states = {item.covariate: item.reason_code for item in rows.unmeasured_rows_dropped}
    binary = bool(outcome.loc[evaluable].dropna().isin([0.0, 1.0]).all())
    evaluable_n = int(evaluable.sum())
    complete = evaluable.copy()
    covariates: list[CovariateRetention] = []
    for name in requirement.covariates or ():
        if name not in columns_by_source:
            continue
        observed = design[columns_by_source[name]].notna().all(axis=1)
        complete &= observed
        missing = evaluable & ~observed
        n_missing = int(missing.sum())
        handling = (
            "unmeasured_category"
            if name in kept_states
            else "unmeasured_rows_dropped"
            if name in dropped_states
            else "drop_row"
        )
        covariates.append(
            CovariateRetention(
                name=name,
                handling=handling,
                n_missing=n_missing,
                missing_share=round(n_missing / evaluable_n, 4) if evaluable_n else 0.0,
                outcome_rate_missing=_rate(outcome, missing) if binary else None,
                outcome_rate_observed=(
                    _rate(outcome, evaluable & observed) if binary else None
                ),
                reason_code=dropped_states.get(name),
            )
        )
    model_n = int(kept_rows.sum())
    complete_case_n = int(complete.sum())
    dropped = evaluable & ~kept_rows
    return RequirementRetention(
        step_id=step_id,
        requirement_id=requirement.requirement_id,
        policy="explicit_missing_category" if listed else "drop_missing_baseline",
        population_n=int(len(population)),
        evaluable_n=evaluable_n,
        model_n=model_n,
        complete_case_n=complete_case_n,
        retention=round(model_n / evaluable_n, 4) if evaluable_n else None,
        complete_case_retention=(
            round(complete_case_n / evaluable_n, 4) if evaluable_n else None
        ),
        outcome_rate_retained=_rate(outcome, evaluable & kept_rows) if binary else None,
        outcome_rate_dropped=_rate(outcome, dropped) if binary else None,
        covariates=tuple(covariates),
    )


def _spline_retention(
    *,
    step_id: str,
    frame: Any,
    authority: LandmarkSplineRuntimeAuthority,
) -> RequirementRetention:
    import pandas as pd

    from .runners.landmark_spline_fit import prepare_landmark_model_population

    population = prepare_landmark_model_population(frame, authority)
    outcome = pd.to_numeric(population.working[authority.outcome_column], errors="coerce")
    evaluable = population.primary_mask & outcome.notna()
    kept_rows = population.working.index.isin(population.model_frame.index)
    evaluable_n = int(evaluable.sum())
    covariates: list[CovariateRetention] = []
    for name in authority.required_adjustment_columns:
        missing = evaluable & population.working[name].isna()
        n_missing = int(missing.sum())
        covariates.append(
            CovariateRetention(
                name=name,
                handling="drop_row",
                n_missing=n_missing,
                missing_share=round(n_missing / evaluable_n, 4) if evaluable_n else 0.0,
                outcome_rate_missing=_rate(outcome, missing),
                outcome_rate_observed=_rate(outcome, evaluable & ~missing),
            )
        )
    model_n = int((evaluable & kept_rows).sum())
    return RequirementRetention(
        step_id=step_id,
        requirement_id=f"{step_id}_primary",
        policy="drop_missing_baseline",
        population_n=int(
            (population.alive_at_landmark & population.under_observation).sum()
        ),
        evaluable_n=evaluable_n,
        model_n=model_n,
        complete_case_n=model_n,
        retention=round(model_n / evaluable_n, 4) if evaluable_n else None,
        complete_case_retention=round(model_n / evaluable_n, 4) if evaluable_n else None,
        outcome_rate_retained=_rate(outcome, evaluable & kept_rows),
        outcome_rate_dropped=_rate(outcome, evaluable & ~kept_rows),
        covariates=tuple(covariates),
    )


def measure_primary_model_retention(
    *,
    context: Any,
    plan: AnalysisPlan,
    cohort_path: Optional[Path],
    runtime_authority: Any = None,
) -> PrimaryModelRetentionEvidence:
    """Return typed retention evidence for the plan's primary adjusted model."""

    def evidence(status: str, reason: Optional[str] = None, **extra: Any):
        return PrimaryModelRetentionEvidence(status=status, reason_code=reason, **extra)

    if _metadata_only(context):
        # Candidate planning reads no patient rows; the package-bound review counts them.
        return evidence("rows_unavailable", "metadata_only_planning")
    categorical = isinstance(runtime_authority, LandmarkCategoricalAssociationRuntimeAuthority)
    spline = isinstance(runtime_authority, LandmarkSplineRuntimeAuthority)
    try:
        if categorical:
            step = runtime_authority.governed_primary_step(plan)
            requirement = _sole_primary(step)
            if requirement is None:
                return evidence("not_evaluable", "primary_model_requirement_absent")
            requirement = execution_model_requirement(step, requirement)
            population_source = "landmark_eligible_rows"
        elif spline:
            step = runtime_authority.governed_step(plan)
            requirement = None
            population_source = "landmark_model_population"
        else:
            found = _generic_primary(plan)
            if found is None:
                return evidence("not_applicable", "no_primary_adjusted_model")
            step, requirement = found
            requirement = execution_model_requirement(step, requirement)
            typed_inputs = [
                item for item in step.inputs if item.split(":", 1)[0] in {"artifact", "dataset"}
            ]
            if any(item not in _ANALYSIS_COHORT_INPUTS for item in typed_inputs):
                return evidence("rows_unavailable", "typed_cohort_product_not_materialized")
            population_source = "analysis_cohort"
    except Exception as exc:  # the owner itself refused the plan
        return evidence("not_evaluable", f"primary_owner_refused_{type(exc).__name__}")
    if cohort_path is None or not Path(cohort_path).is_file():
        return evidence("rows_unavailable", "cohort_not_materialized")

    try:
        import pandas as pd
        import pyarrow.parquet as pq

        if spline:
            needed = set(runtime_authority.required_columns)
        else:
            needed = {
                requirement.outcome,
                requirement.exposure_source,
                *(term.name for term in requirement.model_terms or ()),
            }
            if categorical:
                needed |= {
                    runtime_authority.outcome_column,
                    runtime_authority.event_time_column,
                    runtime_authority.observation_duration_column,
                }
        available = set(pq.ParquetFile(cohort_path).schema_arrow.names)
        if needed - available:
            return evidence("not_evaluable", "primary_model_columns_absent")
        # Only the owner's columns: a plan-phase read must not load the export.
        frame = pd.read_parquet(cohort_path, columns=sorted(needed))
        if frame.empty:
            return evidence("rows_unavailable", "zero_rows")
        if spline:
            measured = _spline_retention(
                step_id=step.step_id, frame=frame, authority=runtime_authority
            )
        else:
            population = frame
            if categorical:
                from .runners.landmark_categorical_association_executor import (
                    landmark_eligibility_mask,
                )

                nonnegative, alive, observed = landmark_eligibility_mask(
                    frame,
                    outcome_column=runtime_authority.outcome_column,
                    event_time_column=runtime_authority.event_time_column,
                    observation_duration_column=runtime_authority.observation_duration_column,
                    observation_duration_unit=runtime_authority.observation_duration_unit,
                    landmark_hours=runtime_authority.landmark_hours,
                )
                population = frame.loc[nonnegative & alive & observed]
            measured = _requirement_retention(
                step_id=step.step_id, requirement=requirement, population=population
            )
    except ModelTermCompilationError as exc:
        return evidence("not_evaluable", f"model_term_{exc.reason_code}")
    except Exception as exc:
        from .runners.landmark_categorical_association_executor import (
            LandmarkCategoricalExecutionError,
        )

        if isinstance(exc, LandmarkCategoricalExecutionError):
            return evidence("not_evaluable", "landmark_eligibility_refused")
        return evidence("probe_failed", f"unexpected_{type(exc).__name__}")
    return evidence(
        "measured",
        population_source=population_source,
        cohort_source_sha256=_sha256(Path(cohort_path)),
        requirements=(measured,),
    )


__all__ = ["measure_primary_model_retention", "resolve_review_cohort_path"]

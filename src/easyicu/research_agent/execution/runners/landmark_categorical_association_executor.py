"""Deterministic fixed-landmark cohort and categorical association adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path
import textwrap
from typing import Any, Mapping

from ...authority.current_case_scientific_runtime import (
    LandmarkCategoricalAssociationRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ...authority.declared_levels import execution_model_requirement
from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...contracts.association_execution import sole_primary_model_requirement
from ...contracts.capability_ids import LANDMARK_CATEGORICAL_ANALYSIS_KIND
from ...schema import AnalysisPlan, AnalysisStep
from .adjusted_association_executor import run_adjusted_association_from_env
from .plausibility_receipt import render_standard_plausibility_receipt_code

LANDMARK_CATEGORICAL_COHORT_ANALYSIS_KIND = "signed_landmark_analysis_cohort"
LANDMARK_CATEGORICAL_PRIMARY_ANALYSIS_KIND = LANDMARK_CATEGORICAL_ANALYSIS_KIND


class LandmarkCategoricalExecutionError(RuntimeError):
    """The signed landmark categorical contract could not be executed."""


def _sealed(
    authority: LandmarkCategoricalAssociationRuntimeAuthority
    | Mapping[str, Any]
    | None,
) -> LandmarkCategoricalAssociationRuntimeAuthority | None:
    if authority is None:
        return None
    value = load_current_case_scientific_runtime_authority(authority)
    return (
        value
        if isinstance(value, LandmarkCategoricalAssociationRuntimeAuthority)
        else None
    )


def landmark_categorical_cohort_executor_owns_step(
    step: AnalysisStep | Mapping[str, Any],
    *,
    plan: AnalysisPlan,
    authority: LandmarkCategoricalAssociationRuntimeAuthority
    | Mapping[str, Any]
    | None,
) -> bool:
    sealed = _sealed(authority)
    if sealed is None:
        return False
    try:
        return sealed.governed_cohort_step(plan) == step
    except ValueError:
        return False


def landmark_categorical_primary_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: LandmarkCategoricalAssociationRuntimeAuthority
    | Mapping[str, Any]
    | None,
) -> bool:
    sealed = _sealed(authority)
    if sealed is None:
        return False
    try:
        return sealed.governed_primary_step(plan) == step
    except ValueError:
        return False


def landmark_eligibility_mask(
    frame: Any,
    *,
    outcome_column: str,
    event_time_column: str,
    observation_duration_column: str,
    observation_duration_unit: str,
    landmark_hours: float,
):
    """Return the exact alive-and-observed landmark eligibility mask."""

    import pandas as pd

    missing = sorted(
        {
            outcome_column,
            event_time_column,
            observation_duration_column,
        }
        - set(frame.columns)
    )
    if missing:
        raise LandmarkCategoricalExecutionError(
            "landmark cohort lacks required columns: " + ", ".join(missing)
        )
    outcome_source = frame[outcome_column]
    outcome = pd.to_numeric(outcome_source, errors="coerce")
    if bool((outcome_source.notna() & outcome.isna()).any()) or bool(
        outcome.isna().any()
    ):
        raise LandmarkCategoricalExecutionError(
            "landmark outcome is missing or non-numeric"
        )
    if not bool(outcome.isin((0.0, 1.0)).all()):
        raise LandmarkCategoricalExecutionError(
            "landmark outcome is not binary 0/1"
        )
    event_source = frame[event_time_column]
    event_time = pd.to_numeric(event_source, errors="coerce")
    if bool((event_source.notna() & event_time.isna()).any()):
        raise LandmarkCategoricalExecutionError(
            "landmark event time is non-numeric"
        )
    if bool((outcome.eq(1.0) & event_time.isna()).any()):
        raise LandmarkCategoricalExecutionError(
            "landmark eligibility cannot time every outcome event"
        )
    duration_source = frame[observation_duration_column]
    duration = pd.to_numeric(duration_source, errors="coerce")
    if bool((duration_source.notna() & duration.isna()).any()):
        raise LandmarkCategoricalExecutionError(
            "landmark observation duration is non-numeric"
        )
    if bool((duration.dropna() < 0.0).any()):
        raise LandmarkCategoricalExecutionError(
            "landmark observation duration is negative"
        )
    threshold = float(landmark_hours)
    if observation_duration_unit == "days":
        threshold /= 24.0
    elif observation_duration_unit != "hours":
        raise LandmarkCategoricalExecutionError(
            "landmark observation duration unit is unsupported"
        )
    nonnegative_event = outcome.eq(0.0) | event_time.ge(0.0)
    alive = outcome.eq(0.0) | event_time.gt(float(landmark_hours))
    observed = duration.notna() & duration.ge(threshold)
    return nonnegative_event, alive, observed


def run_landmark_categorical_cohort(
    *,
    frame: Any,
    source_path: Path,
    authority: LandmarkCategoricalAssociationRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    out_dir: Path,
    plausibility_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize the one signed primary landmark cohort and attrition ledger."""

    import pandas as pd

    sealed = _sealed(authority)
    if sealed is None:
        raise TypeError("landmark cohort executor received the wrong authority kind")
    if len(str(runtime_projection_sha256)) != 64:
        raise LandmarkCategoricalExecutionError(
            "runtime projection digest is required"
        )
    nonnegative, alive, observed = landmark_eligibility_mask(
        frame,
        outcome_column=sealed.outcome_column,
        event_time_column=sealed.event_time_column,
        observation_duration_column=sealed.observation_duration_column,
        observation_duration_unit=sealed.observation_duration_unit,
        landmark_hours=sealed.landmark_hours,
    )
    stages = [
        ("source_cohort", pd.Series(True, index=frame.index)),
        ("nonnegative_event_time", nonnegative),
        ("alive_at_landmark", alive),
        ("observed_at_landmark", observed),
    ]
    mask = pd.Series(True, index=frame.index)
    flow: list[dict[str, Any]] = []
    for order, (predicate, condition) in enumerate(stages):
        before = int(mask.sum())
        if order:
            mask &= condition
        remaining = int(mask.sum())
        flow.append(
            {
                "step_order": order,
                "predicate_kind": predicate,
                "n_before": before,
                "n_excluded": before - remaining,
                "n_remaining": remaining,
            }
        )
    if not bool(mask.any()):
        raise LandmarkCategoricalExecutionError(
            "the signed landmark cohort contains no eligible rows"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = out_dir / "analysis_cohort.parquet"
    flow_path = out_dir / "cohort_flow.csv"
    frame.loc[mask].copy().to_parquet(cohort_path, index=False)
    pd.DataFrame(flow).to_csv(flow_path, index=False)
    mask_sha256 = hashlib.sha256(
        bytes(mask.astype("uint8").tolist())
    ).hexdigest()
    summary: dict[str, Any] = {
        "status": "ok",
        "deterministic_standard_analysis": (
            LANDMARK_CATEGORICAL_COHORT_ANALYSIS_KIND
        ),
        "n_source": int(len(frame)),
        "n_analysis_cohort": int(mask.sum()),
        "analysis_cohort_n": int(mask.sum()),
        "landmark_hours": float(sealed.landmark_hours),
        "eligibility_mask_sha256": mask_sha256,
        "cohort_binding": {
            "source": "COHORT_PARQUET",
            "selection_mode": "signed_alive_and_observed_at_landmark",
            "n_full": int(len(frame)),
            "n_complete_case": int(mask.sum()),
            "n_dropped": int(len(frame) - mask.sum()),
            "final_cohort_n": int(mask.sum()),
            "row_order_preserved": True,
        },
        "landmark_runtime_receipt": {
            "schema_version": (
                "easyicu.landmark_categorical_cohort_runtime_receipt/1"
            ),
            "execution_contract_sha256": sealed.execution_contract_sha256,
            "runtime_projection_sha256": runtime_projection_sha256,
            "source_sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
            "eligibility_mask_sha256": mask_sha256,
            "event_time_column": sealed.event_time_column,
            "observation_duration_column": sealed.observation_duration_column,
            "observation_duration_unit": sealed.observation_duration_unit,
            "landmark_hours": float(sealed.landmark_hours),
            "require_alive_at_landmark": True,
            "exclude_negative_event_times": True,
        },
        "output_files": {
            sealed.cohort_product: cohort_path.name,
            sealed.cohort_flow_product: flow_path.name,
        },
    }
    if plausibility_audit is not None:
        summary["plausibility_audit"] = dict(plausibility_audit)
    return summary


def landmark_categorical_cohort_executor_code(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: LandmarkCategoricalAssociationRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    sealed = _sealed(authority)
    if sealed is None or not landmark_categorical_cohort_executor_owns_step(
        step, plan=plan, authority=sealed
    ):
        raise ValueError("step is not owned by the landmark cohort executor")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    receipt_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope,
            frame_name="frame",
        )
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        import pandas as pd

        from easyicu.research_agent.execution.runners.landmark_categorical_association_executor import (
            run_landmark_categorical_cohort,
        )

        source = Path(os.environ["COHORT_PARQUET"]).resolve()
        frame = pd.read_parquet(source)
        {receipt_code}
        summary = run_landmark_categorical_cohort(
            frame=frame,
            source_path=source,
            authority={sealed.model_dump(mode="json")!r},
            runtime_projection_sha256={runtime_projection_sha256!r},
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            plausibility_audit={"plausibility_audit" if receipt_code else "None"},
        )
        out_dir = Path(os.environ["STEP_OUT_DIR"])
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        """
    ).strip()


def run_landmark_categorical_primary(
    *,
    frame: Any,
    cohort_path: Path,
    step: AnalysisStep,
    authority: LandmarkCategoricalAssociationRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Fit the signed categorical model through the existing statsmodels owner."""

    sealed = _sealed(authority)
    if sealed is None:
        raise TypeError("landmark primary executor received the wrong authority kind")
    parsed_step = (
        step if isinstance(step, AnalysisStep) else AnalysisStep.model_validate(step)
    )
    requirement = sole_primary_model_requirement(parsed_step)
    if requirement is None:
        raise LandmarkCategoricalExecutionError(
            "landmark primary has no single model requirement"
        )
    requirement = execution_model_requirement(parsed_step, requirement)
    nonnegative, alive, observed = landmark_eligibility_mask(
        frame,
        outcome_column=sealed.outcome_column,
        event_time_column=sealed.event_time_column,
        observation_duration_column=sealed.observation_duration_column,
        observation_duration_unit=sealed.observation_duration_unit,
        landmark_hours=sealed.landmark_hours,
    )
    if not bool((nonnegative & alive & observed).all()):
        raise LandmarkCategoricalExecutionError(
            "the bound analysis cohort contains rows outside landmark eligibility"
        )
    summary = run_adjusted_association_from_env(
        requirement_id=requirement.requirement_id,
        exposure=requirement.exposure_source,
        outcome=requirement.outcome,
        covariates=requirement.covariates or (),
        model_terms=requirement.model_terms or (),
        estimator_kind="logistic",
        analysis_set=requirement.analysis_set,
        analysis_role=requirement.analysis_role,
        method_family=requirement.method_family,
        primary_contrast_level=requirement.primary_contrast_level,
        dependence=requirement.dependence,
        typed_cohort_input=sealed.cohort_product,
        frame=frame,
        cohort_path=cohort_path,
        emit_step_summary=False,
        output_dir=out_dir,
    )
    summary["landmark_runtime_receipt"] = {
        "schema_version": (
            "easyicu.landmark_categorical_association_runtime_receipt/1"
        ),
        "execution_contract_sha256": sealed.execution_contract_sha256,
        "runtime_projection_sha256": runtime_projection_sha256,
        "landmark_hours": float(sealed.landmark_hours),
        "exposure_kind": sealed.exposure_kind,
        "exposure_levels": list(sealed.exposure_levels),
        "reference_level": sealed.exposure_reference_level,
        "primary_contrast_level": sealed.primary_contrast_level,
        "interpretation": sealed.interpretation,
    }
    return summary


def landmark_categorical_primary_executor_code(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: LandmarkCategoricalAssociationRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    sealed = _sealed(authority)
    if sealed is None or not landmark_categorical_primary_executor_owns_step(
        step, plan=plan, authority=sealed
    ):
        raise ValueError("step is not owned by the landmark categorical executor")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    receipt_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope,
            frame_name="frame",
        )
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.landmark_categorical_association_executor import (
            run_landmark_categorical_primary,
        )
        from easyicu.research_agent.execution.runners.typed_input_binding import (
            load_step_cohort_frame,
        )

        frame, cohort_path = load_step_cohort_frame(
            typed_cohort_input={sealed.cohort_product!r},
        )
        {receipt_code}
        summary = run_landmark_categorical_primary(
            frame=frame,
            cohort_path=cohort_path,
            step={step.model_dump(mode="json")!r},
            authority={sealed.model_dump(mode="json")!r},
            runtime_projection_sha256={runtime_projection_sha256!r},
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
        )
        {"summary['plausibility_audit'] = plausibility_audit" if receipt_code else ""}
        out_dir = Path(os.environ["STEP_OUT_DIR"])
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        """
    ).strip()


__all__ = [
    "LANDMARK_CATEGORICAL_COHORT_ANALYSIS_KIND",
    "LANDMARK_CATEGORICAL_PRIMARY_ANALYSIS_KIND",
    "LandmarkCategoricalExecutionError",
    "landmark_categorical_cohort_executor_code",
    "landmark_categorical_cohort_executor_owns_step",
    "landmark_categorical_primary_executor_code",
    "landmark_categorical_primary_executor_owns_step",
    "landmark_eligibility_mask",
    "run_landmark_categorical_cohort",
    "run_landmark_categorical_primary",
]

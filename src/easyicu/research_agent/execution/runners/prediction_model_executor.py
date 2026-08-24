"""Deterministic analysis-only adapter for a closed static prediction workflow.

The Planner owns the outcome and predictor roster.  The materialized cohort
owns patient grouping.  This adapter fixes only leakage-safe splitting,
training-only preprocessing, one regularized logistic model, and deterministic
evaluation mechanics.  It deliberately grants no paper authorization.
"""

from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ...contracts.dependence import PlannedDependenceRequirement, resolve_patient_groups
from ...contracts.prediction_execution import (
    PREDICTION_MODEL_ANALYSIS_KIND,
    PREDICTION_PERFORMANCE_PRODUCT,
    PREDICTION_PRIMARY_ACTION,
    PREDICTION_SCORES_PRODUCT,
    static_prediction_execution_verdict,
    static_prediction_model_columns,
)
from ...contracts.prediction_validation import PredictionValidationSpec
from ...methods.delong_auc import delong_auc_ci
from ...planning.dependence_authority import context_patient_group_authority
from ...prediction_validation_owner import (
    run_prediction_validation,
    run_prediction_validation_csv,
)
from ...robustness.panel import load_locked_robustness_specs
from ...intake.materialized_metadata import (
    MaterializedCohortAuthority,
    MaterializedCohortAuthorityRef,
    VerifiedMaterializedCohortAuthority,
    load_verified_materialized_cohort_authority,
)
from ...research_context.typed import (
    ResearchContextAuthority,
    parse_research_context_json,
)
from ...schema import AnalysisStep
from .typed_input_binding import (
    load_typed_input,
    sha256_file,
    sole_typed_cohort_input,
)

PREDICTION_INTERNAL_VALIDATION_PRODUCT = "table:validation"
PREDICTION_CALIBRATION_PRODUCT = "table:calibration"
PREDICTION_CLINICAL_UTILITY_PRODUCT = "table:clinical_utility"

_ACTION_OUTPUTS = {
    "prediction.discrimination_calibration": (
        PREDICTION_SCORES_PRODUCT,
        PREDICTION_PERFORMANCE_PRODUCT,
    ),
    "prediction.internal_validation": (PREDICTION_INTERNAL_VALIDATION_PRODUCT,),
    "prediction.calibration_metrics": (PREDICTION_CALIBRATION_PRODUCT,),
    "prediction.decision_curve": (PREDICTION_CLINICAL_UTILITY_PRODUCT,),
}
_PRIMARY_ACTION = PREDICTION_PRIMARY_ACTION
_SCORE_COLUMNS = ("unit_id", "subject_id", "split", "outcome", "probability")
_THRESHOLDS = tuple(float(value) for value in np.linspace(0.05, 0.50, 10))
_PRIMARY_SPLIT_SEED = 1729
_REPEATED_SPLIT_SEEDS = tuple(range(1730, 1740))


def prediction_model_executor_owns_step(step: AnalysisStep) -> bool:
    """Own only one exact action/product/input shape."""

    action = str(step.scientific_action_id or "")
    expected = _ACTION_OUTPUTS.get(action)
    if expected is None or tuple(step.expected_outputs) != expected:
        return False
    if action == _PRIMARY_ACTION:
        return static_prediction_execution_verdict(step).claimed
    else:
        typed_inputs = tuple(value for value in step.inputs if ":" in value)
        if (
            step.planned_analysis_role not in {"secondary", "auxiliary"}
            or typed_inputs != (PREDICTION_SCORES_PRODUCT,)
        ):
            return False
    return bool(
        step.table_one_spec is None
        and step.cohort_definition_spec is None
        and step.measurement_audit_spec is None
        and step.robustness_replay_spec is None
        and step.trajectory_stability_spec is None
        and not step.model_requirements
    )


def prediction_model_consumed_input_keys(step: AnalysisStep) -> tuple[str, ...]:
    if step.scientific_action_id == _PRIMARY_ACTION:
        cohort = sole_typed_cohort_input(step)
        return (cohort,) if cohort else ()
    return (PREDICTION_SCORES_PRODUCT,)


def prediction_model_executor_code(step: AnalysisStep) -> str:
    if not prediction_model_executor_owns_step(step):
        raise ValueError("step is not owned by the static prediction adapter")
    action = str(step.scientific_action_id)
    if action == _PRIMARY_ACTION:
        cohort = sole_typed_cohort_input(step)
        return textwrap.dedent(
            f"""
            import json
            import os
            from pathlib import Path

            from easyicu.research_agent.execution.runners.prediction_model_executor import (
                run_prediction_model,
            )
            from easyicu.research_agent.execution.runners.typed_input_binding import (
                load_step_cohort_frame,
            )

            frame, cohort_path = load_step_cohort_frame(
                typed_cohort_input={cohort!r},
            )
            summary = run_prediction_model(
                frame=frame,
                declared_columns={static_prediction_model_columns(step)!r},
                typed_cohort_input={cohort!r},
                source_cohort=cohort_path,
                out_dir=Path(os.environ["STEP_OUT_DIR"]),
                run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
                step_id={step.step_id!r},
            )
            print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
            """
        ).strip()
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.prediction_model_executor import (
            run_prediction_score_analysis,
        )

        summary = run_prediction_score_analysis(
            action_id={action!r},
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        """
    ).strip()


def _load_context(run_dir: Path) -> ResearchContextAuthority:
    return parse_research_context_json(
        (Path(run_dir) / "research_context.json").read_text("utf-8")
    )


def _binary_outcome(values: pd.Series, *, column: str) -> pd.Series:
    if values.isna().any() or pd.api.types.is_bool_dtype(values.dtype):
        raise RuntimeError(f"prediction outcome {column!r} must be complete numeric 0/1")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise RuntimeError(f"prediction outcome {column!r} must use exact numeric 0/1")
    return numeric.astype(int)


def _patient_group_authority(
    *,
    context: ResearchContextAuthority,
    source_cohort: Path,
    run_dir: Path,
) -> PlannedDependenceRequirement | None:
    """Resolve only a context- or verified-ancestry-issued patient grouping."""

    direct = context_patient_group_authority(context)
    if direct is not None:
        return direct

    verified = load_verified_materialized_cohort_authority(Path(source_cohort))
    authority_root = Path(source_cohort).parent
    if verified is None:
        materialized_inputs = getattr(context, "materialized_inputs", None)
        context_cohort = getattr(materialized_inputs, "cohort", None)
        if context_cohort is not None:
            cohort_file = str(context_cohort.cohort_file)
            if Path(cohort_file).name != cohort_file:
                raise RuntimeError("typed context cohort file is not run-local")
            expected = MaterializedCohortAuthorityRef.from_dict(
                context_cohort.authority_ref
            )
            authority_root = Path(run_dir)
            verified = load_verified_materialized_cohort_authority(
                authority_root / cohort_file,
                expected_authority=expected,
            )
    return _patient_group_from_verified(verified, authority_root=authority_root)


def _patient_group_from_verified(
    verified: VerifiedMaterializedCohortAuthority | None,
    *,
    authority_root: Path,
) -> PlannedDependenceRequirement | None:
    """Resolve an exact grouping rule from one verified authority ancestry."""

    if verified is None or verified.authority.parent_authority_sha256 is None:
        return None
    parent_sha256 = verified.authority.parent_authority_sha256
    parent_path = Path(authority_root) / (
        f"cohort_authority.sha256-{parent_sha256}.json"
    )
    if not parent_path.is_file() or sha256_file(parent_path) != parent_sha256:
        return None
    payload = json.loads(parent_path.read_text("utf-8"))
    if not isinstance(payload, Mapping):
        return None
    parent = MaterializedCohortAuthority.from_dict(payload)
    if (
        parent.cohort_sha256 != verified.authority.cohort_sha256
        or parent.row_identity_sha256 != verified.authority.row_identity_sha256
        or parent.identity_column != verified.authority.identity_column
    ):
        return None
    replacement = parent.producer_parameters.get("replacement_row_identity")
    if not isinstance(replacement, Mapping):
        return None
    derivation = replacement.get("patient_group_derivation")
    if not (
        replacement.get("output_identity_column") == parent.identity_column
        and isinstance(derivation, Mapping)
        and derivation.get("algorithm") == "prefix_before_:s"
        and derivation.get("delimiter") == ":s"
    ):
        return None
    return PlannedDependenceRequirement(
        group_source=parent.identity_column,
        group_derivation="prefix_before_delimiter",
        delimiter=":s",
    )


def _unit_ids(frame: pd.DataFrame, source: str) -> pd.Series:
    values = frame[source]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise RuntimeError("prediction row identity contains missing values")
    text = values.map(lambda value: f"{type(value).__name__}:{value!r}")
    if text.duplicated().any():
        text = text + ":row" + pd.Series(np.arange(len(text)), index=text.index).astype(str)
    if text.duplicated().any():  # pragma: no cover - defensive
        raise RuntimeError("prediction unit identity is not unique")
    return text


def _split_labels(
    groups: pd.Series,
    outcome: pd.Series,
    *,
    seed: int = _PRIMARY_SPLIT_SEED,
) -> np.ndarray:
    if groups.nunique() < 10:
        raise RuntimeError("prediction requires at least 10 patient groups")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_index, validation_index = next(
        splitter.split(np.zeros(len(groups)), outcome.to_numpy(), groups.to_numpy())
    )
    labels = np.full(len(groups), "development", dtype=object)
    labels[validation_index] = "validation"
    train_groups = set(groups.iloc[train_index])
    validation_groups = set(groups.iloc[validation_index])
    if train_groups & validation_groups:
        raise RuntimeError("patient groups cross development and validation splits")
    for label in ("development", "validation"):
        if outcome.iloc[np.flatnonzero(labels == label)].nunique() != 2:
            raise RuntimeError(f"{label} split does not contain both outcome classes")
    return labels


def _repeated_group_split_validation(
    *,
    frame: pd.DataFrame,
    outcome: pd.Series,
    groups: pd.Series,
    unit_ids: pd.Series,
    features: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refit ten fixed patient-group splits and return auditable uncertainty."""

    rows: list[dict[str, Any]] = []
    for seed in _REPEATED_SPLIT_SEEDS:
        split = _split_labels(groups, outcome, seed=seed)
        development = split == "development"
        validation = split == "validation"
        probabilities = _fit_probabilities(
            frame=frame,
            outcome=outcome,
            features=features,
            development=development,
            prediction_rows=validation,
        )
        scores = pd.DataFrame(
            {
                "unit_id": unit_ids.loc[validation].to_numpy(),
                "subject_id": groups.loc[validation].to_numpy(),
                "split": "validation",
                "outcome": outcome.loc[validation].to_numpy(),
                "probability": probabilities,
            },
            columns=_SCORE_COLUMNS,
        )
        result = run_prediction_validation(scores, _prediction_validation_spec())
        auc = delong_auc_ci(scores["outcome"], scores["probability"])
        rows.append(
            {
                "split_seed": seed,
                "development_n": int(development.sum()),
                "validation_n": int(validation.sum()),
                "development_subject_n": int(groups.loc[development].nunique()),
                "validation_subject_n": int(groups.loc[validation].nunique()),
                "patient_overlap_n": 0,
                "validation_event_n": int(scores["outcome"].sum()),
                "auroc": auc.auc,
                "average_precision": float(
                    average_precision_score(scores["outcome"], scores["probability"])
                ),
                "brier_score": result.summary.brier_score,
                "calibration_status": result.summary.calibration_status,
                "calibration_intercept": result.summary.calibration_intercept,
                "calibration_slope": result.summary.calibration_slope,
            }
        )
    metrics = ("auroc", "average_precision", "brier_score")
    summary = {
        "method": "repeated_patient_group_split_refit",
        "n_repeats": len(rows),
        "validation_fraction": 0.20,
        "split_seeds": list(_REPEATED_SPLIT_SEEDS),
        "all_patient_overlap_zero": all(row["patient_overlap_n"] == 0 for row in rows),
        "metrics": {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])),
                "standard_deviation": float(
                    np.std([row[metric] for row in rows], ddof=1)
                ),
                "minimum": float(min(row[metric] for row in rows)),
                "maximum": float(max(row[metric] for row in rows)),
            }
            for metric in metrics
        },
        "authority_scope": "analysis_only",
        "external_validation_established": False,
    }
    return rows, summary


def _model_pipeline(frame: pd.DataFrame, features: tuple[str, ...]) -> Pipeline:
    numeric = tuple(
        column for column in features if pd.api.types.is_numeric_dtype(frame[column])
    )
    categorical = tuple(column for column in features if column not in numeric)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(numeric),
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                        ),
                    ]
                ),
                list(categorical),
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers, remainder="drop")),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=1729,
                ),
            ),
        ]
    )


def _fit_probabilities(
    *,
    frame: pd.DataFrame,
    outcome: pd.Series,
    features: tuple[str, ...],
    development: np.ndarray,
    prediction_rows: np.ndarray,
) -> np.ndarray:
    """Fit once on the declared development rows and score declared rows."""

    if not bool(development.any()) or not bool(prediction_rows.any()):
        raise RuntimeError("prediction fit requires non-empty development and scoring rows")
    if outcome.loc[development].nunique() != 2:
        raise RuntimeError("prediction development rows do not contain both outcome classes")
    model = _model_pipeline(frame.loc[development], features)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(frame.loc[development, list(features)], outcome.loc[development])
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("prediction logistic model did not converge")
    probabilities = model.predict_proba(frame.loc[prediction_rows, list(features)])[:, 1]
    if not np.isfinite(probabilities).all() or not (
        (0 <= probabilities) & (probabilities <= 1)
    ).all():
        raise RuntimeError("prediction model produced invalid probabilities")
    return probabilities


def _prediction_validation_spec() -> PredictionValidationSpec:
    return PredictionValidationSpec(
        unit_id_column="unit_id",
        subject_id_column="subject_id",
        split_column="split",
        outcome_column="outcome",
        probability_column="probability",
        evaluation_split="validation",
        analysis_unit="encounter",
        thresholds=_THRESHOLDS,
        calibration_bins=10,
    )


def run_prediction_robustness_specs(
    *,
    frame: pd.DataFrame,
    outcome: pd.Series,
    groups: pd.Series,
    unit_ids: pd.Series,
    split: np.ndarray,
    features: tuple[str, ...],
    specs: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute only plan-locked complete-case variants this owner understands.

    The model, split, outcome, and feature roster are inherited unchanged from
    the primary owner.  A spec that changes any other coordinate is left
    unexecuted so the run-level robustness gate retains its fail-closed error.
    """

    panel_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    expected_variables = set((*features, outcome.name))
    for spec in specs:
        missing = getattr(spec, "missing_override", None)
        if (
            getattr(spec, "axis", None) != "missing"
            or getattr(spec, "cohort_override", None) is not None
            or getattr(spec, "outcome_override", None) is not None
            or not isinstance(missing, Mapping)
            or str(missing.get("strategy") or "") != "complete_case"
        ):
            continue
        variables = tuple(str(value or "").strip() for value in missing.get("variables", ()))
        if (
            not variables
            or len(variables) != len(set(variables))
            or set(variables) != expected_variables
            or any(variable not in frame.columns for variable in variables)
        ):
            continue
        complete = frame.loc[:, list(variables)].notna().all(axis=1).to_numpy()
        development = complete & (split == "development")
        validation = complete & (split == "validation")
        if (
            int(development.sum()) == 0
            or int(validation.sum()) == 0
            or outcome.loc[development].nunique() != 2
            or outcome.loc[validation].nunique() != 2
        ):
            continue
        probabilities = _fit_probabilities(
            frame=frame,
            outcome=outcome,
            features=features,
            development=development,
            prediction_rows=complete,
        )
        variant_scores = pd.DataFrame(
            {
                "unit_id": unit_ids.loc[complete].to_numpy(),
                "subject_id": groups.loc[complete].to_numpy(),
                "split": split[complete],
                "outcome": outcome.loc[complete].to_numpy(),
                "probability": probabilities,
            },
            columns=_SCORE_COLUMNS,
        )
        validation_result = run_prediction_validation(
            variant_scores, _prediction_validation_spec()
        )
        evaluated = variant_scores.loc[variant_scores["split"].eq("validation")]
        auc = delong_auc_ci(evaluated["outcome"], evaluated["probability"])
        average_precision = float(
            average_precision_score(evaluated["outcome"], evaluated["probability"])
        )
        summary = validation_result.summary
        result = {
            "spec_id": str(getattr(spec, "spec_id", "")),
            "axis": "missing",
            "analysis": "complete_case_refit_same_patient_split",
            "predictors": list(features),
            "development_n": int(development.sum()),
            "validation_n": int(validation.sum()),
            "validation_subject_n": int(evaluated["subject_id"].nunique()),
            "validation_event_n": int(evaluated["outcome"].sum()),
            "auroc": auc.auc,
            "auroc_se": auc.se,
            "auroc_ci_low": auc.ci_low,
            "auroc_ci_high": auc.ci_high,
            "auroc_ci_method": "delong_logit_normal_95pct",
            "average_precision": average_precision,
            "brier_score": summary.brier_score,
            "calibration_status": summary.calibration_status,
            "calibration_intercept": summary.calibration_intercept,
            "calibration_slope": summary.calibration_slope,
            "authority_scope": "analysis_only",
        }
        results.append(result)
        panel_rows.append(
            {
                "spec_id": result["spec_id"],
                "axis": "missing",
                "n": result["validation_n"],
                "point_estimate": auc.auc,
                "ci_low": auc.ci_low,
                "ci_high": auc.ci_high,
                "se": auc.se,
                "evidence_id": "",
                "converged": True,
                "notes": (
                    "metric=AUROC; complete-case refit with the primary model, "
                    "outcome, predictor roster, and patient split unchanged; "
                    "95% CI uses the deterministic DeLong logit interval"
                ),
            }
        )
    return panel_rows, results


def run_prediction_model(
    *,
    frame: pd.DataFrame,
    declared_columns: tuple[str, ...],
    typed_cohort_input: str,
    source_cohort: Path,
    out_dir: Path,
    run_dir: Path,
    step_id: str,
) -> dict[str, Any]:
    """Fit the exact Planner roster with one fixed analysis-only pipeline."""

    context = _load_context(Path(run_dir))
    outcome_column = str(context.target_outcome or "").strip()
    group_authority = _patient_group_authority(
        context=context,
        source_cohort=Path(source_cohort),
        run_dir=Path(run_dir),
    )
    if not outcome_column or group_authority is None:
        raise RuntimeError("prediction requires typed outcome and patient-group authority")
    required = {outcome_column, group_authority.group_source, *declared_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"prediction cohort is missing declared columns: {missing!r}")
    features = tuple(
        column
        for column in declared_columns
        if column not in {outcome_column, group_authority.group_source}
    )
    if not features or len(features) != len(set(features)):
        raise RuntimeError("prediction requires a unique non-empty predictor roster")
    for column in features:
        if frame[column].notna().sum() == 0:
            raise RuntimeError(f"prediction feature {column!r} is entirely missing")
    outcome = _binary_outcome(frame[outcome_column], column=outcome_column)
    groups = pd.Series(
        resolve_patient_groups(
            frame[group_authority.group_source], requirement=group_authority
        ).groups,
        index=frame.index,
    )
    split = _split_labels(groups, outcome)
    development = split == "development"
    all_rows = np.ones(len(frame), dtype=bool)
    probabilities = _fit_probabilities(
        frame=frame,
        outcome=outcome,
        features=features,
        development=development,
        prediction_rows=all_rows,
    )

    unit_ids = _unit_ids(frame, group_authority.group_source)
    scores = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "subject_id": groups,
            "split": split,
            "outcome": outcome,
            "probability": probabilities,
        },
        columns=_SCORE_COLUMNS,
    )
    validation = scores.loc[scores["split"].eq("validation")]
    validation_result = run_prediction_validation(scores, _prediction_validation_spec())
    validation_summary = validation_result.summary
    auc = delong_auc_ci(validation["outcome"], validation["probability"])
    repeated_split_rows, repeated_split_summary = _repeated_group_split_validation(
        frame=frame,
        outcome=outcome,
        groups=groups,
        unit_ids=unit_ids,
        features=features,
    )
    performance = pd.DataFrame(
        [
            {
                "model": "logistic_regression_l2",
                "authority_scope": "analysis_only",
                "paper_authorization_allowed": False,
                "split_seed": _PRIMARY_SPLIT_SEED,
                "validation_fraction": 0.20,
                "predictor_n": len(features),
                "predictors": "|".join(features),
                "development_n": int(development.sum()),
                "validation_n": int((~development).sum()),
                "development_subject_n": int(groups.loc[development].nunique()),
                "validation_subject_n": int(groups.loc[~development].nunique()),
                "patient_overlap_n": 0,
                "validation_event_n": int(validation["outcome"].sum()),
                "validation_event_rate": float(validation["outcome"].mean()),
                "auroc": auc.auc,
                "auroc_se": auc.se,
                "auroc_ci_low": auc.ci_low,
                "auroc_ci_high": auc.ci_high,
                "auroc_ci_method": "delong_logit_normal_95pct",
                "average_precision": float(
                    average_precision_score(
                        validation["outcome"], validation["probability"]
                    )
                ),
                "brier_score": float(
                    brier_score_loss(
                        validation["outcome"], validation["probability"]
                    )
                ),
                "calibration_status": validation_summary.calibration_status,
                "calibration_intercept": validation_summary.calibration_intercept,
                "calibration_slope": validation_summary.calibration_slope,
                "preprocessing_fit_scope": "development_partition_only",
                "patient_group_source": group_authority.group_source,
                "patient_group_derivation": group_authority.group_derivation,
                "repeated_split_n": repeated_split_summary["n_repeats"],
                "repeated_split_seeds": json.dumps(
                    repeated_split_summary["split_seeds"], separators=(",", ":")
                ),
                "repeated_split_auroc_mean": repeated_split_summary["metrics"][
                    "auroc"
                ]["mean"],
                "repeated_split_auroc_sd": repeated_split_summary["metrics"][
                    "auroc"
                ]["standard_deviation"],
                "repeated_split_average_precision_mean": repeated_split_summary[
                    "metrics"
                ]["average_precision"]["mean"],
                "repeated_split_average_precision_sd": repeated_split_summary[
                    "metrics"
                ]["average_precision"]["standard_deviation"],
                "repeated_split_brier_mean": repeated_split_summary["metrics"][
                    "brier_score"
                ]["mean"],
                "repeated_split_brier_sd": repeated_split_summary["metrics"][
                    "brier_score"
                ]["standard_deviation"],
                "repeated_split_results": json.dumps(
                    repeated_split_rows, separators=(",", ":"), allow_nan=False
                ),
            }
        ]
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "prediction_scores.csv", index=False)
    performance.to_csv(out_dir / "prediction_performance.csv", index=False)
    validation_receipt = run_prediction_validation_csv(
        source_path=out_dir / "prediction_scores.csv",
        expected_source_sha256=sha256_file(out_dir / "prediction_scores.csv"),
        spec=_prediction_validation_spec(),
    )
    robustness_rows, prediction_robustness = run_prediction_robustness_specs(
        frame=frame,
        outcome=outcome,
        groups=groups,
        unit_ids=unit_ids,
        split=split,
        features=features,
        specs=load_locked_robustness_specs(Path(run_dir)),
    )
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_static_prediction_model_with_repeated_split_validation",
        "analysis_family": "prediction",
        "deterministic_standard_analysis": PREDICTION_MODEL_ANALYSIS_KIND,
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "predictor_roster": list(features),
        "predictor_roster_contract": "raw_input_prefix_before_typed_cohort",
        "scientific_validation_owner": "prediction_validation_owner",
        "scientific_validation_contract": "PredictionValidationReceipt",
        "prediction_validation_receipt": validation_receipt.model_dump(mode="json"),
        "resampling_validation": {
            **repeated_split_summary,
            "repeats": repeated_split_rows,
        },
        "robustness_rows": robustness_rows,
        "prediction_robustness_results": prediction_robustness,
        "source_cohort": str(Path(source_cohort).resolve()),
        "source_cohort_sha256": sha256_file(Path(source_cohort)),
        "source_inputs": [typed_cohort_input],
        "input_bindings": [{"input_key": typed_cohort_input, "loaded": True}],
        "output_files": {
            PREDICTION_SCORES_PRODUCT: "prediction_scores.csv",
            PREDICTION_PERFORMANCE_PRODUCT: "prediction_performance.csv",
        },
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _validation_result(frame: pd.DataFrame):
    return run_prediction_validation(
        frame,
        PredictionValidationSpec(
            unit_id_column="unit_id",
            subject_id_column="subject_id",
            split_column="split",
            outcome_column="outcome",
            probability_column="probability",
            evaluation_split="validation",
            analysis_unit="encounter",
            thresholds=_THRESHOLDS,
            calibration_bins=10,
        ),
    )


def run_prediction_score_analysis(
    *,
    action_id: str,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
) -> dict[str, Any]:
    """Compute one exact downstream validation product from sealed scores."""

    if action_id not in _ACTION_OUTPUTS or action_id == _PRIMARY_ACTION:
        raise RuntimeError("unsupported downstream prediction action")
    bound = load_typed_input(
        input_key=PREDICTION_SCORES_PRODUCT,
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        expected_columns=_SCORE_COLUMNS,
        require_consumption_contract=True,
        minimum_row_count=1,
    )
    result = _validation_result(bound.frame)
    if action_id == "prediction.internal_validation":
        table = pd.DataFrame(
            [
                {
                    **result.summary.model_dump(mode="json"),
                    "patient_overlap_n": 0,
                    "split_rule": "patient_group_shuffle_80_20_seed_1729",
                    "authority_scope": "analysis_only",
                }
            ]
        )
        filename = "internal_validation.csv"
        product = PREDICTION_INTERNAL_VALIDATION_PRODUCT
    elif action_id == "prediction.calibration_metrics":
        summary_row = {
            "row_role": "summary",
            "bin_index": 0,
            "n": result.summary.evaluation_n,
            "event_n": result.summary.event_n,
            "mean_predicted_probability": result.summary.mean_predicted_probability,
            "observed_event_rate": result.summary.event_rate,
            "minimum_predicted_probability": np.nan,
            "maximum_predicted_probability": np.nan,
            "brier_score": result.summary.brier_score,
            "calibration_status": result.summary.calibration_status,
            "calibration_intercept": result.summary.calibration_intercept,
            "calibration_slope": result.summary.calibration_slope,
        }
        rows = [summary_row]
        for item in result.calibration_bins:
            rows.append(
                {
                    "row_role": "calibration_bin",
                    **item.model_dump(mode="json"),
                    "brier_score": np.nan,
                    "calibration_status": result.summary.calibration_status,
                    "calibration_intercept": np.nan,
                    "calibration_slope": np.nan,
                }
            )
        table = pd.DataFrame(rows)
        filename = "calibration_assessment.csv"
        product = PREDICTION_CALIBRATION_PRODUCT
    else:
        evaluation = bound.frame.loc[bound.frame["split"].eq("validation")]
        outcomes = evaluation["outcome"].to_numpy(dtype=int)
        probabilities = evaluation["probability"].to_numpy(dtype=float)
        prevalence = float(outcomes.mean())
        rows = []
        for threshold in _THRESHOLDS:
            positive = probabilities >= threshold
            true_positive = int(np.count_nonzero(positive & (outcomes == 1)))
            false_positive = int(np.count_nonzero(positive & (outcomes == 0)))
            odds = threshold / (1.0 - threshold)
            rows.append(
                {
                    "threshold": threshold,
                    "n": len(outcomes),
                    "net_benefit_model": true_positive / len(outcomes)
                    - false_positive / len(outcomes) * odds,
                    "net_benefit_all": prevalence - (1.0 - prevalence) * odds,
                    "net_benefit_none": 0.0,
                }
            )
        table = pd.DataFrame(rows)
        filename = "clinical_utility.csv"
        product = PREDICTION_CLINICAL_UTILITY_PRODUCT
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / filename, index=False)
    if sha256_file(bound.path) != bound.sha256:
        raise RuntimeError("prediction scores changed during downstream evaluation")
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": f"deterministic_{action_id.replace('.', '_')}",
        "analysis_family": "prediction",
        "deterministic_standard_analysis": PREDICTION_MODEL_ANALYSIS_KIND,
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "source_inputs": [PREDICTION_SCORES_PRODUCT],
        "input_bindings": [
            {
                "input_key": PREDICTION_SCORES_PRODUCT,
                "evidence_id": bound.evidence_id,
                "sha256": bound.sha256,
                "loaded": True,
                "row_count": bound.row_count,
            }
        ],
        "output_files": {product: filename},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "PREDICTION_CALIBRATION_PRODUCT",
    "PREDICTION_CLINICAL_UTILITY_PRODUCT",
    "PREDICTION_INTERNAL_VALIDATION_PRODUCT",
    "PREDICTION_MODEL_ANALYSIS_KIND",
    "PREDICTION_PERFORMANCE_PRODUCT",
    "PREDICTION_SCORES_PRODUCT",
    "prediction_model_consumed_input_keys",
    "prediction_model_executor_code",
    "prediction_model_executor_owns_step",
    "run_prediction_model",
    "run_prediction_robustness_specs",
    "run_prediction_score_analysis",
]

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

from ...contracts.concept_values import columns_without_values
from ...contracts.dependence import resolve_patient_groups
from ...contracts.executor_stop import ExecutorStop, write_executor_stop_record
from ...contracts.prediction_execution import (
    PREDICTION_BENCHMARK_ACTION,
    PREDICTION_BENCHMARK_PRODUCT,
    PREDICTION_CALIBRATION_PRODUCT,
    PREDICTION_CLINICAL_UTILITY_PRODUCT,
    PREDICTION_INTERNAL_VALIDATION_PRODUCT,
    PREDICTION_MODEL_ANALYSIS_KIND,
    PREDICTION_PERFORMANCE_PRODUCT,
    PREDICTION_PRIMARY_ACTION,
    PREDICTION_SCORES_PRODUCT,
    STATIC_PREDICTION_ACTION_OUTPUTS as _ACTION_OUTPUTS,
    static_prediction_benchmark_cohort_input,
    static_prediction_benchmark_columns,
    static_prediction_executes_robustness_spec,
    static_prediction_features,
    static_prediction_model_columns,
    static_prediction_owns_step,
)
from ...contracts.prediction_validation import PredictionValidationSpec
from ...methods.auc_interval import AUCInterval, auc_interval, paired_auc_difference
from ...methods.delong_auc import delong_auc_ci
from ...planning.benchmark_comparator import (
    benchmark_comparator_facts,
    calibration_reason,
    comparator_information_window,
    information_window_relation,
)
from ...prediction_validation_owner import (
    run_prediction_validation,
    run_prediction_validation_csv,
)
from ...robustness.panel import load_locked_robustness_specs
from ...research_context.materialization_window import bound_feature_window_end_hours
from ...research_context.typed import (
    ResearchContextAuthority,
    parse_research_context_json,
)
from ...schema import AnalysisStep
from .patient_groups import step_patient_group_authority
from .typed_input_binding import (
    load_typed_input,
    sha256_file,
    sole_typed_cohort_input,
)

_PRIMARY_ACTION = PREDICTION_PRIMARY_ACTION
_SCORE_COLUMNS = ("unit_id", "subject_id", "split", "outcome", "probability")
_THRESHOLDS = tuple(float(value) for value in np.linspace(0.05, 0.50, 10))
_PRIMARY_SPLIT_SEED = 1729
_REPEATED_SPLIT_SEEDS = tuple(range(1730, 1740))


def prediction_model_executor_owns_step(step: AnalysisStep) -> bool:
    """Own only one exact action/product/input shape.

    The shape is a dependency-neutral contract so the plan reviewer asks the
    same question the executor answers.
    """

    return static_prediction_owns_step(step)


def prediction_model_consumed_input_keys(step: AnalysisStep) -> tuple[str, ...]:
    if step.scientific_action_id == _PRIMARY_ACTION:
        cohort = sole_typed_cohort_input(step)
        return (cohort,) if cohort else ()
    if step.scientific_action_id == PREDICTION_BENCHMARK_ACTION:
        cohort = static_prediction_benchmark_cohort_input(step)
        return (cohort, PREDICTION_SCORES_PRODUCT) if cohort else (PREDICTION_SCORES_PRODUCT,)
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
    if action == PREDICTION_BENCHMARK_ACTION:
        cohort = static_prediction_benchmark_cohort_input(step)
        return textwrap.dedent(
            f"""
            import json
            import os
            from pathlib import Path

            from easyicu.research_agent.execution.runners.prediction_model_executor import (
                run_prediction_benchmark_comparison,
            )
            from easyicu.research_agent.execution.runners.typed_input_binding import (
                load_step_cohort_frame,
            )

            frame, cohort_path = load_step_cohort_frame(
                typed_cohort_input={cohort!r},
            )
            summary = run_prediction_benchmark_comparison(
                frame=frame,
                comparator_columns={static_prediction_benchmark_columns(step)!r},
                typed_cohort_input={cohort!r},
                source_cohort=cohort_path,
                out_dir=Path(os.environ["STEP_OUT_DIR"]),
                run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
                resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
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
    """Return the complete 0/1 outcome every later prediction table carries.

    A logical column is the same closed binary outcome spelled True/False
    (data preparation may write an event flag that way); it becomes 0/1 here,
    once, so the scores and every downstream validator see one numeric outcome.
    """

    if values.isna().any():
        raise RuntimeError(f"prediction outcome {column!r} must be complete numeric 0/1")
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(int)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise RuntimeError(f"prediction outcome {column!r} must use exact numeric 0/1")
    return numeric.astype(int)


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


class _PredictorsWithoutValues(RuntimeError):
    """Predictors no development row holds a value of.

    The imputer would drop them with a warning, and the model fitted would
    not be the one the plan states (``contracts.concept_values``).
    """

    def __init__(self, columns: tuple[str, ...], development_n: int) -> None:
        super().__init__(
            ", ".join(repr(column) for column in columns)
            + f" hold no value in any of the {development_n} development stays"
        )
        self.columns = columns


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
    unobserved = columns_without_values(frame.loc[development], features)
    if unobserved:
        raise _PredictorsWithoutValues(unobserved, int(np.count_nonzero(development)))
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


def _interval_words(interval: AUCInterval) -> str:
    """How an AUROC interval was computed, in the panel's words."""

    if interval.bootstrap_n:
        return (
            f"percentile interval of {interval.bootstrap_n:,} patient resamples "
            "(repeat ICU stays of a patient are resampled together)"
        )
    return "the deterministic DeLong logit interval"


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
    for spec in specs:
        if not static_prediction_executes_robustness_spec(
            spec, features=features, outcome=str(outcome.name)
        ):
            continue
        variables = tuple(
            str(value or "").strip() for value in spec.missing_override["variables"]
        )
        if any(variable not in frame.columns for variable in variables):
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
        auc = auc_interval(
            evaluated["outcome"], evaluated["probability"], evaluated["subject_id"]
        )
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
            "auroc_ci_method": auc.method,
            "auroc_bootstrap_n": auc.bootstrap_n,
            "auroc_bootstrap_skipped_n": auc.bootstrap_skipped_n,
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
                    f"95% CI: {_interval_words(auc)}"
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
    group_authority = step_patient_group_authority(
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
    features = static_prediction_features(
        declared_columns,
        outcome=outcome_column,
        group_source=group_authority.group_source,
    )
    if not features or len(features) != len(set(features)):
        raise RuntimeError("prediction requires a unique non-empty predictor roster")
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
    try:
        probabilities = _fit_probabilities(
            frame=frame,
            outcome=outcome,
            features=features,
            development=development,
            prediction_rows=all_rows,
        )
    except _PredictorsWithoutValues as found:
        stop = ExecutorStop("prediction_predictor_unobserved", detail=str(found))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        write_executor_stop_record(out_dir, stop)
        raise stop from found

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
    auc = auc_interval(
        validation["outcome"], validation["probability"], validation["subject_id"]
    )
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
                "auroc_ci_method": auc.method,
                "auroc_bootstrap_n": auc.bootstrap_n,
                "auroc_bootstrap_skipped_n": auc.bootstrap_skipped_n,
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

    if action_id not in _ACTION_OUTPUTS or action_id in {
        _PRIMARY_ACTION,
        PREDICTION_BENCHMARK_ACTION,
    }:
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


#: The benchmark comparison table's columns, in order (one row per comparator
#: and metric; the interval, difference and method columns are the AUROC
#: row's only).
_BENCHMARK_COLUMNS = (
    "comparator_column",
    "comparator_concept",
    "comparator_kind",
    "metric",
    "model_value",
    "model_ci_low",
    "model_ci_high",
    "comparator_value",
    "comparator_ci_low",
    "comparator_ci_high",
    "difference",
    "difference_se",
    "difference_ci_low",
    "difference_ci_high",
    "z",
    "p_value",
    "interval_method",
    "bootstrap_n",
    "bootstrap_skipped_n",
    "validation_n",
    "comparator_missing_n",
    "comparison_n",
    "comparison_event_n",
    "comparison_subject_n",
    "calibration_status",
    "calibration_reason",
    "comparator_predicts",
    "outcome_concept",
    "comparator_information_window",
    "prediction_time_hours",
    "information_window_relation",
    "information_window_differs",
)
_CALIBRATION_METRICS = ("brier_score", "calibration_intercept", "calibration_slope")


class BenchmarkComparisonError(RuntimeError):
    """A benchmark comparison this owner refuses; ``code`` says why."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _calibration_summary(
    *, unit_ids: np.ndarray, subjects: np.ndarray, outcome: np.ndarray, probability: np.ndarray
):
    frame = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "subject_id": subjects,
            "split": "validation",
            "outcome": outcome,
            "probability": probability,
        },
        columns=_SCORE_COLUMNS,
    )
    return run_prediction_validation(frame, _prediction_validation_spec()).summary


def run_prediction_benchmark_comparison(
    *,
    frame: pd.DataFrame,
    comparator_columns: Sequence[str],
    typed_cohort_input: str,
    source_cohort: Path,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
) -> dict[str, Any]:
    """Compare the sealed model scores with existing scores on the same stays.

    The comparison rows are the validation stays whose comparator is
    recorded; the model's scores are the primary's, read from its sealed
    table and aligned to the cohort by the primary's own row identity.  The
    comparator's facts -- probability or oriented score, the outcome it
    predicts, the window it was computed over -- are the dictionary's
    (:mod:`...planning.benchmark_comparator`).  Every interval follows how the
    comparison rows depend (:mod:`...methods.auc_interval`).  Calibration is
    compared only for a probability of the study's own outcome, and never by
    recalibrating the comparator.
    """

    context = _load_context(Path(run_dir))
    outcome_column = str(context.target_outcome or "").strip()
    group_authority = step_patient_group_authority(
        context=context,
        source_cohort=Path(source_cohort),
        run_dir=Path(run_dir),
    )
    if not outcome_column or group_authority is None:
        raise RuntimeError("prediction requires typed outcome and patient-group authority")
    comparators = tuple(str(column) for column in comparator_columns)
    missing = sorted({group_authority.group_source, *comparators} - set(frame.columns))
    if missing:
        raise RuntimeError(f"benchmark comparison cohort is missing declared columns: {missing!r}")
    prediction_time = bound_feature_window_end_hours(context)
    if prediction_time is None:
        raise BenchmarkComparisonError(
            "benchmark_prediction_time_unstated",
            "the run binds no feature window, so the model's prediction time is unknown",
        )
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
        text_columns=("unit_id", "subject_id"),
    )
    groups = resolve_patient_groups(
        frame[group_authority.group_source], requirement=group_authority
    ).groups
    cohort_rows = pd.DataFrame(
        {
            "unit_id": _unit_ids(frame, group_authority.group_source).to_numpy(),
            "cohort_subject_id": pd.Series(groups).astype(str).to_numpy(),
            **{column: frame[column].to_numpy() for column in comparators},
        }
    )
    scores = bound.frame
    if len(scores) != len(cohort_rows) or set(scores["unit_id"]) != set(cohort_rows["unit_id"]):
        raise BenchmarkComparisonError(
            "benchmark_rows_do_not_align",
            "the model's scores and the cohort hold different stays",
        )
    merged = scores.merge(cohort_rows, on="unit_id", how="left", validate="one_to_one")
    if not merged["subject_id"].astype(str).eq(merged["cohort_subject_id"]).all():
        raise BenchmarkComparisonError(
            "benchmark_rows_do_not_align",
            "a stay's patient differs between the model's scores and the cohort",
        )
    validation = merged.loc[merged["split"].eq("validation")].reset_index(drop=True)
    outcome_variable = context.variable(outcome_column)
    outcome_concept = str(
        getattr(outcome_variable, "source_concept", None) or outcome_column
    )
    rows: list[dict[str, Any]] = []
    reportable: list[dict[str, Any]] = []
    for column in comparators:
        variable = context.variable(column)
        concept = str(getattr(variable, "source_concept", None) or column)
        facts = benchmark_comparator_facts(concept)
        if facts is None or facts.kind is None:
            raise BenchmarkComparisonError(
                "benchmark_comparator_unsupported",
                f"{column!r} is neither a probability nor a score whose direction "
                "is stated (planning.benchmark_comparator)",
            )
        # What the column is, the whole cohort's rows say: one value a
        # probability cannot take, in any split, means it is not one.
        raw = merged[column]
        values = pd.to_numeric(raw, errors="coerce")
        if (values.isna() & raw.notna()).any() or not np.isfinite(values.dropna()).all():
            raise BenchmarkComparisonError(
                "benchmark_comparator_not_numeric",
                f"{column!r} holds values that are not finite numbers",
            )
        if facts.kind == "probability" and (
            (values < 0.0) | (values > 1.0)
        ).any():
            raise BenchmarkComparisonError(
                "benchmark_probability_out_of_range",
                f"{column!r} is a probability but holds values outside [0, 1]",
            )
        if columns_without_values(merged.loc[merged["split"].eq("validation")], [column]):
            # Nothing to compare the model with (``contracts.concept_values``).
            stop = ExecutorStop(
                "benchmark_comparator_unobserved",
                detail=f"{column!r} holds no value in any of the "
                f"{len(validation)} validation stays",
            )
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            write_executor_stop_record(out_dir, stop)
            raise stop
        validation_values = values.loc[merged["split"].eq("validation")].reset_index(drop=True)
        present = validation_values.notna().to_numpy()
        comparator_values = validation_values.to_numpy(dtype=float)[present]
        compared = validation.loc[present]
        outcome = compared["outcome"].to_numpy(dtype=int)
        if np.unique(outcome).size != 2:
            raise BenchmarkComparisonError(
                "benchmark_comparison_one_class",
                f"the validation stays with {column!r} recorded hold one outcome class",
            )
        subjects = compared["subject_id"].astype(str).to_numpy()
        model_values = compared["probability"].to_numpy(dtype=float)
        difference = paired_auc_difference(outcome, model_values, comparator_values, subjects)
        model_auc = auc_interval(outcome, model_values, subjects)
        comparator_auc = auc_interval(outcome, comparator_values, subjects)
        reason = calibration_reason(facts, outcome_concept=outcome_concept)
        window = comparator_information_window(
            getattr(variable, "analysis_window", None), facts
        )
        relation = information_window_relation(
            window, prediction_time_hours=float(prediction_time)
        )
        shared = {
            "comparator_column": column,
            "comparator_concept": concept,
            "comparator_kind": facts.kind,
            "validation_n": int(len(validation)),
            "comparator_missing_n": int((~present).sum()),
            "comparison_n": int(present.sum()),
            "comparison_event_n": int(outcome.sum()),
            "comparison_subject_n": int(np.unique(subjects).size),
            "calibration_status": "compared" if reason is None else "calibration_not_compared",
            "calibration_reason": reason or "",
            "comparator_predicts": facts.predicts or "",
            "outcome_concept": outcome_concept,
            "comparator_information_window": window or "",
            "prediction_time_hours": float(prediction_time),
            "information_window_relation": relation,
            "information_window_differs": relation != "same",
        }
        rows.append(
            {
                **shared,
                "metric": "auroc",
                "model_value": model_auc.auc,
                "model_ci_low": model_auc.ci_low,
                "model_ci_high": model_auc.ci_high,
                "comparator_value": comparator_auc.auc,
                "comparator_ci_low": comparator_auc.ci_low,
                "comparator_ci_high": comparator_auc.ci_high,
                "difference": difference.difference,
                "difference_se": difference.se,
                "difference_ci_low": difference.ci_low,
                "difference_ci_high": difference.ci_high,
                "z": difference.z,
                "p_value": difference.p_value,
                "interval_method": difference.method,
                "bootstrap_n": difference.bootstrap_n,
                "bootstrap_skipped_n": difference.bootstrap_skipped_n,
            }
        )
        # The reportable block names each model's AUROC and calibration as the
        # manuscript audit and the numeric binder read them (``*.auroc``,
        # ``*.auroc_ci_low``, ``*.brier_score``); the table holds the same values.
        model_block: dict[str, float | None] = {
            "auroc": model_auc.auc,
            "auroc_ci_low": model_auc.ci_low,
            "auroc_ci_high": model_auc.ci_high,
        }
        comparator_block: dict[str, float | None] = {
            "auroc": comparator_auc.auc,
            "auroc_ci_low": comparator_auc.ci_low,
            "auroc_ci_high": comparator_auc.ci_high,
        }
        if reason is None:
            unit_ids = compared["unit_id"].to_numpy()
            model_calibration = _calibration_summary(
                unit_ids=unit_ids, subjects=subjects, outcome=outcome, probability=model_values
            )
            comparator_calibration = _calibration_summary(
                unit_ids=unit_ids,
                subjects=subjects,
                outcome=outcome,
                probability=comparator_values,
            )
            for metric in _CALIBRATION_METRICS:
                model_value = _finite_or_none(getattr(model_calibration, metric))
                comparator_value = _finite_or_none(getattr(comparator_calibration, metric))
                model_block[metric] = model_value
                comparator_block[metric] = comparator_value
                rows.append(
                    {
                        **shared,
                        "metric": metric,
                        "model_value": model_value,
                        "comparator_value": comparator_value,
                    }
                )
        reportable.append(
            {
                **shared,
                "model": model_block,
                "comparator": comparator_block,
                "auroc_difference": difference.difference,
                "auroc_difference_ci_low": difference.ci_low,
                "auroc_difference_ci_high": difference.ci_high,
                "auroc_difference_p_value": difference.p_value,
                "interval_method": difference.method,
            }
        )
    table = pd.DataFrame(rows, columns=_BENCHMARK_COLUMNS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "benchmark_comparison.csv", index=False)
    if sha256_file(bound.path) != bound.sha256:
        raise RuntimeError("prediction scores changed during the benchmark comparison")
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_prediction_benchmark_comparison",
        "analysis_family": "prediction",
        "deterministic_standard_analysis": PREDICTION_MODEL_ANALYSIS_KIND,
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "comparator_columns": list(comparators),
        "reportable_benchmark_comparison": {
            "prediction_time_hours": float(prediction_time),
            "outcome_concept": outcome_concept,
            "comparisons": reportable,
        },
        "source_cohort": str(Path(source_cohort).resolve()),
        "source_cohort_sha256": sha256_file(Path(source_cohort)),
        "source_inputs": [typed_cohort_input, PREDICTION_SCORES_PRODUCT],
        "input_bindings": [
            {"input_key": typed_cohort_input, "loaded": True},
            {
                "input_key": PREDICTION_SCORES_PRODUCT,
                "evidence_id": bound.evidence_id,
                "sha256": bound.sha256,
                "loaded": True,
                "row_count": bound.row_count,
            },
        ],
        "output_files": {PREDICTION_BENCHMARK_PRODUCT: "benchmark_comparison.csv"},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "BenchmarkComparisonError",
    "PREDICTION_CALIBRATION_PRODUCT",
    "PREDICTION_CLINICAL_UTILITY_PRODUCT",
    "PREDICTION_INTERNAL_VALIDATION_PRODUCT",
    "PREDICTION_MODEL_ANALYSIS_KIND",
    "PREDICTION_PERFORMANCE_PRODUCT",
    "PREDICTION_SCORES_PRODUCT",
    "prediction_model_consumed_input_keys",
    "prediction_model_executor_code",
    "prediction_model_executor_owns_step",
    "run_prediction_benchmark_comparison",
    "run_prediction_model",
    "run_prediction_robustness_specs",
    "run_prediction_score_analysis",
]

"""Deterministic owner for a fully typed ordered-stratified analysis.

One step answers "does the outcome change steadily across the ordered
exposure levels" three ways, each on the scale it can honestly claim:

* per-level risks with Wilson intervals and a crude Cochran-Armitage test of
  the binary outcome (scale-free);
* the primary model refitted with the exposure as one ``ordinal_linear`` term
  (one unit per declared level), with the primary's own estimator, adjustment
  roster, dependence contract and missing-data policy, for the adjusted odds
  or risk ratio per level increment, its Wald interval and trend p-value;
* when the plan names one, a continuous outcome by level (median, IQR) with a
  Jonckheere-Terpstra test, Holm-adjusted with the Cochran-Armitage test.
"""

from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ...contracts.ordered_stratified import (
    ADJUSTED_TREND_TEST_ID,
    CONTRACT_KEY,
    CONTRACT_SCHEMA_VERSION,
    CONTROLLED_METHOD,
    ORDERED_STRATIFIED_INTERPRETATION_CLASS,
    PARENT_PRODUCT,
    SCIENTIFIC_ACTION_ID,
    SECONDARY_REPORTING_SCHEMA_VERSION,
    TREND_REPORTING_KEY,
    TREND_REPORTING_SCHEMA_VERSION,
)
from ...methods.ordered_trends import (
    OrderedTrendResult,
    cochran_armitage_trend,
    jonckheere_terpstra_trend,
    normal_tail_probability,
    wilson_interval,
)
from ...authority.declared_levels import execution_model_requirement
from ...contracts.model_terms import level_spelling
from ...contracts.ordered_stratified_spec import (
    OrderedStratifiedSpec,
    OrderedTrendUnsupported,
    ordered_trend_exposure_term,
    ordered_trend_model_for,
)
from ...contracts.cohort_product_keys import is_closed_cohort_product_key
from ...schema import AnalysisPlan, AnalysisStep
from .typed_input_binding import (
    load_typed_cohort,
    load_typed_input,
    run_dir_from_env,
)

ORDERED_STRATIFIED_ANALYSIS_KIND = "ordered_stratified_analysis"
ORDERED_STRATIFIED_PARENT_PRODUCT = PARENT_PRODUCT
#: The kernel's own working directory for the trend refit; its files are
#: audit detail beside the declared trend table, never step evidence.
_TREND_MODEL_DIR = "ordinal_trend_model"


def _typed_cohort_input(step: AnalysisStep) -> str:
    values = [value for value in step.inputs if is_closed_cohort_product_key(value)]
    return values[0] if len(values) == 1 else ""


def ordered_stratified_consumed_input_keys(step: AnalysisStep) -> tuple[str, ...]:
    """Return every resolved authority consumed by the typed adapter.

    The cohort supplies values.  The adjusted-estimate product supplies the
    upstream identity that binds this secondary analysis to its one primary
    model; the selector verifies that relationship before execution, so the
    host must issue an identity receipt for both authorities.
    """

    cohort = _typed_cohort_input(step)
    if not cohort or ORDERED_STRATIFIED_PARENT_PRODUCT not in step.inputs:
        return ()
    return (cohort, ORDERED_STRATIFIED_PARENT_PRODUCT)


def ordered_stratified_spec_for_step(
    step: AnalysisStep, *, plan: AnalysisPlan
) -> OrderedStratifiedSpec | None:
    """Project one closed adapter spec from same-plan typed declarations."""

    if (
        str(step.method or "").strip().casefold() != CONTROLLED_METHOD
        or step.scientific_action_id != SCIENTIFIC_ACTION_ID
        or not _typed_cohort_input(step)
    ):
        return None
    try:
        position = next(index for index, item in enumerate(plan.steps) if item is step)
    except StopIteration:
        positions = [
            index for index, item in enumerate(plan.steps) if item.step_id == step.step_id
        ]
        if len(positions) != 1:
            return None
        position = positions[0]
    parents = [
        item
        for item in plan.steps[:position]
        if item.planned_analysis_role == "primary"
        and ORDERED_STRATIFIED_PARENT_PRODUCT in item.expected_outputs
        and ORDERED_STRATIFIED_PARENT_PRODUCT in step.inputs
    ]
    if len(parents) != 1:
        return None
    requirements = [
        requirement
        for requirement in parents[0].model_requirements
        if requirement.analysis_role == "primary"
        and requirement.outcome_type == "binary"
        and requirement.required_for_step_success
    ]
    if len(requirements) != 1:
        return None
    # The levels as the primary executes them: host placeholders resolved.
    requirement = execution_model_requirement(parents[0], requirements[0])
    try:
        # The compiler applies the same predicate and stops the plan with
        # progressive_ordered_trend_primary_binding_invalid; only a plan it
        # never judged reaches here with a declaration it refuses.
        exposure_term = ordered_trend_exposure_term(requirement)
        trend_model = ordered_trend_model_for(requirement)
    except OrderedTrendUnsupported:
        return None
    exposure = requirement.exposure_source
    binary_outcome = requirement.outcome
    bare_inputs = [value for value in step.inputs if ":" not in value]
    remaining = [
        value for value in bare_inputs if value not in {exposure, binary_outcome}
    ]
    trend_products = [
        value for value in step.expected_outputs if value.startswith("table:")
    ]
    if (
        exposure not in bare_inputs
        or binary_outcome not in bare_inputs
        or len(remaining) > 1
        or len(trend_products) != 1
    ):
        return None
    levels = list(exposure_term.levels or [])
    return OrderedStratifiedSpec(
        ordered_exposure=exposure,
        ordered_levels=levels,
        cochran_armitage_scores=list(range(len(levels))),
        binary_outcome=binary_outcome,
        continuous_outcome=remaining[0] if remaining else None,
        trend_model=trend_model,
        trend_product=trend_products[0],
    )


def ordered_stratified_executor_owns_step(
    step: AnalysisStep, *, plan: AnalysisPlan
) -> bool:
    return ordered_stratified_spec_for_step(step, plan=plan) is not None


def ordered_stratified_executor_code(step: AnalysisStep, *, plan: AnalysisPlan) -> str:
    spec = ordered_stratified_spec_for_step(step, plan=plan)
    if spec is None:
        raise ValueError("step is not owned by the ordered-stratified executor")
    return textwrap.dedent(
        f"""
        from easyicu.research_agent.execution.runners.ordered_stratified_executor import (
            run_ordered_stratified_from_env,
        )

        run_ordered_stratified_from_env(
            spec_payload={spec.model_dump(mode="json")!r},
            typed_cohort_input={_typed_cohort_input(step)!r},
            analysis_role={step.planned_analysis_role!r},
        )
        """
    ).strip()


def _level_mask(series: pd.Series, level: Any) -> pd.Series:
    if isinstance(level, (int, float)) and not isinstance(level, bool):
        return pd.to_numeric(series, errors="coerce") == float(level)
    return series.notna() & (series.astype(str) == str(level))


def _resolved_ordered_levels(series: pd.Series, declared: Sequence[Any]) -> list[Any]:
    observed = []
    for value in pd.unique(series.dropna()):
        observed.append(value.item() if isinstance(value, np.generic) else value)
    by_spelling: dict[str, list[Any]] = {}
    for value in observed:
        by_spelling.setdefault(level_spelling(value), []).append(value)
    resolved: list[Any] = []
    for token in declared:
        matches = by_spelling.get(level_spelling(token), [])
        if len(matches) != 1:
            raise RuntimeError(
                f"declared ordered level {token!r} does not resolve uniquely "
                "against the bound cohort"
            )
        resolved.append(matches[0])
    undeclared = sorted(set(by_spelling) - {level_spelling(value) for value in declared})
    if undeclared:
        raise RuntimeError(
            "ordered exposure contains non-missing values outside the declared "
            f"level set: {undeclared!r}"
        )
    return resolved


def _holm(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p, kind="mergesort")
    adjusted_sorted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * float(p[original_index])))
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def _trend_row(
    result: OrderedTrendResult,
    *,
    outcome: str,
    outcome_type: str,
    test_id: str,
    levels_with_data: int,
    family_size: int,
) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "outcome_type": outcome_type,
        "test_id": test_id,
        "test_name": result.test_name,
        "alternative": result.alternative,
        "n": result.n,
        "levels_with_data": levels_with_data,
        "statistic": result.statistic,
        "statistic_name": result.statistic_type,
        "expected_statistic": result.expected_statistic,
        "variance": result.variance,
        "z_statistic": result.z_statistic,
        "chi_square": result.chi_square,
        "effect_size": result.effect_size,
        "effect_size_name": result.effect_size_name,
        "p_value": result.p_value,
        "adjusted_p": None,
        "p_value_reporting": result.p_value_reporting,
        "log_p_value": result.log_p_value,
        "negative_log10_p": result.negative_log10_p,
        "p_value_bounded": result.p_value_bounded,
        "tie_correction": result.tie_correction,
        "continuity_correction": result.continuity_correction,
        "implementation": result.implementation,
        "score_scheme": result.score_scheme,
        "family_id": "ordered_trend_outcomes",
        "family_size": family_size,
        "prespecified": True,
        "multiplicity_policy": "holm_familywise",
        "status": "ok",
    }


def _product_filename(product: str) -> str:
    return product.split(":", 1)[1] + ".csv"


def _primary_model_rows(primary: pd.DataFrame, *, spec: OrderedStratifiedSpec) -> dict[str, Any]:
    """The analysed rows, events and fit coordinates the primary model reported.

    Every contrast row of one model shares them.  The refit must reproduce
    them exactly: the exposure coding is the only coordinate it may change.
    """

    model = spec.trend_model
    if "model_id" not in primary.columns:
        raise RuntimeError("the primary estimates table does not name its model")
    rows = primary.loc[primary["model_id"].astype(str) == model.primary_requirement_id]

    def text(value: Any) -> str:
        return "" if pd.isna(value) else str(value)

    try:
        coordinates = {
            (
                int(row["n"]), int(row["n_events"]), text(row["estimator_kind"]),
                text(row["covariates"]), text(row["variance_estimator"]),
            )
            for _, row in rows.iterrows()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "the primary estimates table does not report its analysed rows and events"
        ) from exc
    if len(coordinates) != 1:
        raise RuntimeError(
            f"the primary estimates table reports {len(rows)} rows of model "
            f"{model.primary_requirement_id!r} without one shared analysed population"
        )
    n, events, kind, covariates, variance = next(iter(coordinates))
    return {
        "model_id": model.primary_requirement_id, "n": n, "n_events": events,
        "estimator_kind": kind, "covariates": covariates, "variance_estimator": variance,
    }


def _adjusted_trend(
    *,
    frame: pd.DataFrame,
    spec: OrderedStratifiedSpec,
    work_dir: Path,
    primary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refit the primary model with the exposure as one per-level term.

    The adjusted-association owner fits it -- the same function, estimator,
    roster, dependence contract and missing-data policy as the primary -- so
    the trend and the level contrasts can only differ by the exposure coding.
    ``primary`` is what the primary model reported; a refit over other rows,
    events, estimator, roster or variance fails the step.  Returns the
    trend-table row and the contract record of the fit.
    """

    from .adjusted_association_executor import (
        AdjustedAssociationError,
        run_adjusted_association_from_env,
    )

    model = spec.trend_model
    try:
        fitted = run_adjusted_association_from_env(
            requirement_id=model.requirement_id,
            exposure=spec.ordered_exposure,
            outcome=spec.binary_outcome,
            covariates=list(model.covariates),
            model_terms=list(model.model_terms),
            estimator_kind=model.estimator_kind,
            analysis_set=model.analysis_set,
            analysis_role="secondary",
            method_family=model.method_family,
            primary_contrast_level=None,
            dependence=model.dependence,
            missing_category_covariates=list(model.missing_category_covariates),
            typed_cohort_input=None,
            frame=frame,
            cohort_path=None,
            emit_step_summary=False,
            output_dir=work_dir,
        )
    except AdjustedAssociationError as exc:
        raise RuntimeError(f"the adjusted ordered-trend refit failed: {exc}") from exc
    estimates = pd.read_csv(work_dir / "adjusted_association_estimates.csv")
    if len(estimates) != 1:
        raise RuntimeError(
            "the adjusted ordered-trend refit must report one per-level estimate"
        )
    fitted_row = estimates.iloc[0]
    values = {
        key: float(fitted_row[key])
        for key in ("estimate", "ci_low", "ci_high", "standard_error")
    }
    if not all(math.isfinite(value) for value in values.values()) or not (
        0.0 < values["ci_low"] <= values["estimate"] <= values["ci_high"]
        and values["standard_error"] > 0.0
    ):
        raise RuntimeError(
            "the adjusted ordered-trend refit returned no usable per-level estimate"
        )
    measure = model.effect_measure
    z_statistic = math.log(values["estimate"]) / values["standard_error"]
    tail = normal_tail_probability(z_statistic, "two-sided")
    n = int(fitted["n_total"])
    events = int(fitted["n_events"])
    refit = {
        "n": n, "n_events": events, "estimator_kind": model.estimator_kind,
        "covariates": ";".join(model.covariates),
        "variance_estimator": str(fitted.get("variance_estimator") or ""),
    }
    differing = sorted(key for key, value in refit.items() if primary.get(key) != value)
    if differing:
        raise RuntimeError(
            "the adjusted ordered-trend refit is not the primary model with one "
            f"per-level term: {', '.join(differing)} differ from primary model "
            f"{primary.get('model_id')!r} ("
            + "; ".join(f"{key} {refit[key]!r} vs {primary.get(key)!r}" for key in differing)
            + ")"
        )
    row = {
        "outcome": spec.binary_outcome,
        "outcome_type": "binary",
        "test_id": ADJUSTED_TREND_TEST_ID,
        "test_name": "Adjusted ordinal-linear trend (Wald test of the per-level term)",
        "alternative": "two-sided",
        "n": n,
        "levels_with_data": len(spec.ordered_levels),
        "statistic": z_statistic,
        "statistic_name": "wald_z",
        "expected_statistic": 0.0,
        "variance": values["standard_error"] ** 2,
        "z_statistic": z_statistic,
        "chi_square": z_statistic**2,
        "effect_size": values["estimate"],
        "effect_size_name": f"adjusted_{measure}_per_level",
        **tail,
        # Model-based, outside the crude test family: never Holm-adjusted.
        "adjusted_p": None,
        "tie_correction": False,
        "continuity_correction": False,
        "implementation": (
            "easyicu.execution.runners.adjusted_association_executor"
            ".ordinal_linear_refit.v1"
        ),
        "score_scheme": spec.score_scheme,
        "family_id": "adjusted_ordered_trend",
        "family_size": 1,
        "prespecified": True,
        "multiplicity_policy": "single_model_based_test",
        "status": "ok",
        "effect_scale": measure,
        "estimate": values["estimate"],
        "ci_low": values["ci_low"],
        "ci_high": values["ci_high"],
        "standard_error": values["standard_error"],
        "n_events": events,
        "model_id": model.requirement_id,
        "estimator_kind": model.estimator_kind,
        "covariates": ";".join(model.covariates),
        "variance_estimator": fitted.get("variance_estimator"),
        "cluster_count": fitted.get("cluster_count"),
    }
    record = {
        "test_id": ADJUSTED_TREND_TEST_ID,
        "model_id": model.requirement_id,
        "estimator_kind": model.estimator_kind,
        "method_family": model.method_family,
        "effect_scale": measure,
        "increment": "one_declared_level",
        "level_scores": list(range(len(spec.ordered_levels))),
        "covariates": list(model.covariates),
        "analysis_set": model.analysis_set,
        "n": n,
        "n_events": events,
        "estimate": values["estimate"],
        "ci_low": values["ci_low"],
        "ci_high": values["ci_high"],
        "standard_error": values["standard_error"],
        "z_statistic": z_statistic,
        "p_value": tail["p_value"],
        "p_value_reporting": tail["p_value_reporting"],
        "variance_estimator": fitted.get("variance_estimator"),
        "cluster_count": fitted.get("cluster_count"),
        "design_columns": list(fitted.get("design_columns") or []),
        "model_terms": [term.model_dump(mode="json") for term in model.model_terms],
        "primary_model_id": primary["model_id"],
        "primary_n": primary["n"],
        "primary_n_events": primary["n_events"],
    }
    return row, record


def run_ordered_stratified_from_env(
    *,
    spec_payload: Mapping[str, Any],
    typed_cohort_input: str,
    analysis_role: str,
) -> dict[str, Any]:
    spec = OrderedStratifiedSpec.model_validate(spec_payload)
    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, cohort_path = load_typed_cohort(
        input_key=typed_cohort_input,
        run_dir=run_dir_from_env(),
        resolved_inputs_path=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]).resolve(),
    )
    continuous_name = spec.continuous_outcome
    required = {spec.ordered_exposure, spec.binary_outcome}
    if continuous_name is not None:
        required.add(continuous_name)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError("ordered-stratified cohort is missing: " + ", ".join(missing))

    resolved_levels = _resolved_ordered_levels(
        frame[spec.ordered_exposure], spec.ordered_levels
    )
    spec = OrderedStratifiedSpec.model_validate(
        {**spec.model_dump(mode="python"), "ordered_levels": resolved_levels}
    )

    masks = [_level_mask(frame[spec.ordered_exposure], level) for level in spec.ordered_levels]
    membership = sum(mask.astype(int) for mask in masks)
    if int((membership > 1).sum()):
        raise RuntimeError("declared ordered levels overlap under execution matching")
    exposure_nonmissing = frame[spec.ordered_exposure].notna()
    invalid_exposure_n = int((exposure_nonmissing & (membership != 1)).sum())
    if invalid_exposure_n:
        raise RuntimeError(
            f"ordered exposure contains {invalid_exposure_n} non-missing values "
            "outside the declared level set"
        )
    valid_exposure_n = int((membership == 1).sum())
    if valid_exposure_n <= 0:
        raise RuntimeError("ordered-stratified analysis set is empty")

    binary = pd.to_numeric(frame[spec.binary_outcome], errors="coerce")
    invalid_binary_n = int((frame[spec.binary_outcome].notna() & ~binary.isin([0, 1])).sum())
    if invalid_binary_n:
        raise RuntimeError(
            f"binary outcome contains {invalid_binary_n} non-missing values outside 0/1"
        )
    continuous = (
        pd.to_numeric(frame[continuous_name], errors="coerce")
        if continuous_name is not None
        else None
    )
    continuous_finite = (
        pd.Series(np.isfinite(continuous.to_numpy(dtype=float)), index=frame.index)
        if continuous is not None
        else None
    )
    if continuous is not None:
        invalid_continuous_n = int(
            (frame[continuous_name].notna() & ~continuous_finite).sum()
        )
        if invalid_continuous_n:
            raise RuntimeError(
                f"continuous outcome contains {invalid_continuous_n} non-finite values"
            )

    rows: list[dict[str, Any]] = []
    events: list[int] = []
    totals: list[int] = []
    for order, (level, level_mask) in enumerate(zip(spec.ordered_levels, masks)):
        level_n = int(level_mask.sum())
        binary_mask = level_mask & binary.isin([0, 1])
        binary_n = int(binary_mask.sum())
        continuous_mask = (
            level_mask & continuous_finite if continuous_finite is not None else None
        )
        continuous_n = int(continuous_mask.sum()) if continuous_mask is not None else None
        if binary_n <= 0 or (continuous_n is not None and continuous_n <= 0):
            raise RuntimeError(
                f"ordered level {level!r} has no valid binary or continuous outcome"
            )
        event_n = int(binary.loc[binary_mask].sum())
        events.append(event_n)
        totals.append(binary_n)
        ci = wilson_interval(event_n, binary_n, alpha=0.05)
        row: dict[str, Any] = {
            # Level values and their zero-based order are identifiers, not
            # additive counts.  Mark every record explicitly so the generic
            # aggregate-row gate cannot mistake a coincidental arithmetic
            # equality (for example 3 == 0 + 1 + 2) for an overall row.
            "row_role": "exposure_level",
            "level_value": level,
            "level_order": order,
            "level_n": level_n,
            "level_percentage": 100.0 * level_n / valid_exposure_n,
            "binary_outcome": spec.binary_outcome,
            "binary_n": binary_n,
            "binary_missing_n": level_n - binary_n,
            "binary_event_n": event_n,
            "binary_risk": event_n / binary_n,
            "binary_percentage": 100.0 * event_n / binary_n,
            "binary_ci_low": ci.ci_low,
            "binary_ci_high": ci.ci_high,
            "binary_ci_method": "wilson_score",
            "binary_ci_alpha": 0.05,
        }
        if continuous is not None and continuous_mask is not None:
            values = continuous.loc[continuous_mask]
            row.update(
                {
                    "continuous_outcome": continuous_name,
                    "continuous_n": continuous_n,
                    "continuous_missing_n": level_n - int(continuous_n or 0),
                    "continuous_median": float(values.median()),
                    "continuous_q25": float(values.quantile(0.25, interpolation="linear")),
                    "continuous_q75": float(values.quantile(0.75, interpolation="linear")),
                }
            )
        rows.append(row)

    ca = cochran_armitage_trend(
        events,
        totals,
        scores=spec.cochran_armitage_scores,
        group_order=spec.ordered_levels,
    )
    tests: list[tuple[OrderedTrendResult, str, str, str]] = [
        (ca, spec.binary_outcome, "binary", "cochran_armitage"),
    ]
    if continuous is not None and continuous_finite is not None:
        jt_mask = (membership == 1) & continuous_finite
        jt = jonckheere_terpstra_trend(
            continuous.loc[jt_mask].tolist(),
            frame.loc[jt_mask, spec.ordered_exposure].tolist(),
            group_order=spec.ordered_levels,
        )
        tests.append((jt, str(continuous_name), "continuous", "jonckheere_terpstra"))
    family_size = len(tests)
    trend_rows = [
        _trend_row(
            result,
            outcome=outcome,
            outcome_type=outcome_type,
            test_id=test_id,
            levels_with_data=len(spec.ordered_levels),
            family_size=family_size,
        )
        for result, outcome, outcome_type, test_id in tests
    ]
    adjusted = _holm([float(result.p_value) for result, *_rest in tests])
    for row, value in zip(trend_rows, adjusted):
        row["adjusted_p"] = value
    primary_table = load_typed_input(
        input_key=PARENT_PRODUCT,
        run_dir=run_dir_from_env(),
        resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]).resolve(),
        expected_declared_kind="table",
        minimum_row_count=1,
    ).frame
    adjusted_row, adjusted_record = _adjusted_trend(
        frame=frame, spec=spec, work_dir=out_dir / _TREND_MODEL_DIR,
        primary=_primary_model_rows(primary_table, spec=spec),
    )
    trend_rows.append(adjusted_row)

    stratified_path = out_dir / _product_filename(spec.stratified_product)
    trend_path = out_dir / _product_filename(spec.trend_product)
    pd.DataFrame(rows).to_csv(stratified_path, index=False)
    trend_frame = pd.DataFrame(trend_rows)
    trend_frame.to_csv(trend_path, index=False)

    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "execution_owner": "ordered_stratified_executor_v1",
        "ordered_exposure_column": spec.ordered_exposure,
        "ordered_levels": list(spec.ordered_levels),
        "cochran_armitage_scores": list(spec.cochran_armitage_scores),
        "score_scheme": spec.score_scheme,
        "binary_outcome_column": spec.binary_outcome,
        "continuous_outcome_column": continuous_name,
        "locked_cohort_n": int(len(frame)),
        "valid_ordered_exposure_n": valid_exposure_n,
        "ci_method": "wilson_score",
        "ci_alpha": 0.05,
        "continuous_summary": "median_iqr" if continuous_name is not None else None,
        "quantile_method": "linear" if continuous_name is not None else None,
        "stratified_table": stratified_path.name,
        "trend_table": trend_path.name,
        "tests": {
            "binary": {"test_id": "cochran_armitage", "alternative": "two-sided"},
            **(
                {"continuous": {"test_id": "jonckheere_terpstra", "alternative": "two-sided"}}
                if continuous_name is not None
                else {}
            ),
        },
        "multiplicity_policy": "holm_familywise",
        "multiplicity_family_size": family_size,
        "adjusted_trend": adjusted_record,
    }
    binary_trend = trend_rows[0]
    reportable_secondary_results: dict[str, Any] = {
        "schema_version": SECONDARY_REPORTING_SCHEMA_VERSION,
        "analysis_role": analysis_role,
        # It carries crude tests beside an adjusted per-level ratio; each
        # part names its own adjustment, and none is causal.
        "interpretation_ceiling": "secondary_association_not_causal",
        "ordered_exposure": spec.ordered_exposure,
        "ordered_levels": list(spec.ordered_levels),
        "binary_outcome": spec.binary_outcome,
        "binary_level_summaries": [
            {
                "level": row["level_value"],
                "n": row["binary_n"],
                "events": row["binary_event_n"],
                "risk": row["binary_risk"],
                "ci_low": row["binary_ci_low"],
                "ci_high": row["binary_ci_high"],
            }
            for row in rows
        ],
        "binary_trend": {
            "test_id": binary_trend["test_id"],
            "adjustment": "crude",
            "n": binary_trend["n"],
            # Holm across the crude tests only; not a covariate adjustment.
            "holm_adjusted_p": binary_trend["adjusted_p"],
            "p_value_reporting": binary_trend["p_value_reporting"],
        },
        "adjusted_trend": {
            "interpretation_ceiling": "secondary_adjusted_association_not_causal",
            "effect_scale": adjusted_record["effect_scale"],
            "increment": adjusted_record["increment"],
            "estimate": adjusted_record["estimate"],
            "ci_low": adjusted_record["ci_low"],
            "ci_high": adjusted_record["ci_high"],
            "p_value_reporting": adjusted_record["p_value_reporting"],
            "n": adjusted_record["n"],
            "n_events": adjusted_record["n_events"],
            "covariates": list(adjusted_record["covariates"]),
        },
        "continuous_outcome": continuous_name,
    }
    if continuous_name is not None:
        continuous_trend = trend_rows[1]
        reportable_secondary_results["continuous_level_summaries"] = [
            {
                "level": row["level_value"],
                "n": row["continuous_n"],
                "median": row["continuous_median"],
                "q25": row["continuous_q25"],
                "q75": row["continuous_q75"],
            }
            for row in rows
        ]
        reportable_secondary_results["continuous_trend"] = {
            "test_id": continuous_trend["test_id"],
            "adjustment": "crude",
            "n": continuous_trend["n"],
            "holm_adjusted_p": continuous_trend["adjusted_p"],
            "p_value_reporting": continuous_trend["p_value_reporting"],
            "effect_size": continuous_trend["effect_size"],
            "effect_size_name": continuous_trend["effect_size_name"],
        }
    summary = {
        "status": "ok",
        "analysis_family": "association",
        "interpretation_class": ORDERED_STRATIFIED_INTERPRETATION_CLASS,
        "interpretation_ceiling": "secondary_association_not_causal",
        "analysis_role": analysis_role,
        "analysis_set": "exposure_observed_per_outcome_available_case",
        "source_cohort": cohort_path.name,
        "source_row_count_reconciliation": {
            "source_rows": int(len(frame)),
            "valid_ordered_exposure_rows": valid_exposure_n,
            "excluded_missing_exposure_rows": int(len(frame) - valid_exposure_n),
            "filtering_performed": valid_exposure_n != len(frame),
        },
        "reportable_secondary_results": reportable_secondary_results,
        # The host compiles one secondary association claim from this envelope.
        TREND_REPORTING_KEY: {
            "schema_version": TREND_REPORTING_SCHEMA_VERSION,
            "exposure": spec.ordered_exposure,
            "ordered_levels": [level_spelling(level) for level in spec.ordered_levels],
            "outcome": spec.binary_outcome,
            "analysis_set": adjusted_record["analysis_set"],
            "adjustment_covariates": list(adjusted_record["covariates"]),
            "effect_scale": adjusted_record["effect_scale"],
            "estimate": adjusted_record["estimate"],
            "lower": adjusted_record["ci_low"],
            "upper": adjusted_record["ci_high"],
            "p_value": adjusted_record["p_value"],
            "n": adjusted_record["n"],
            "events": adjusted_record["n_events"],
        },
        CONTRACT_KEY: contract,
        "output_files": {
            spec.trend_product: trend_path.name,
        },
        "audit_files": {
            "ordered_stratified_table": stratified_path.name,
        },
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary

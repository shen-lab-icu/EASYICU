"""Deterministic owner for a fully typed ordered-stratified analysis."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ...contracts.ordered_stratified import (
    CONTRACT_KEY,
    CONTRACT_SCHEMA_VERSION,
    CONTROLLED_METHOD,
)
from ...methods.ordered_trends import (
    OrderedTrendResult,
    cochran_armitage_trend,
    jonckheere_terpstra_trend,
    wilson_interval,
)
from ...contracts.model_terms import level_spelling
from ...contracts.ordered_stratified_spec import OrderedStratifiedSpec
from ...contracts.cohort_product_keys import is_closed_cohort_product_key
from ...schema import AnalysisPlan, AnalysisStep
from .typed_input_binding import (
    load_typed_cohort,
    run_dir_from_env,
)

ORDERED_STRATIFIED_ANALYSIS_KIND = "ordered_stratified_analysis"


def _typed_cohort_input(step: AnalysisStep) -> str:
    values = [value for value in step.inputs if is_closed_cohort_product_key(value)]
    return values[0] if len(values) == 1 else ""


def ordered_stratified_spec_for_step(
    step: AnalysisStep, *, plan: AnalysisPlan
) -> OrderedStratifiedSpec | None:
    """Project one closed adapter spec from same-plan typed declarations."""

    if (
        str(step.method or "").strip().casefold() != CONTROLLED_METHOD
        or step.scientific_action_id != "association.ordinal_trend"
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
        and "table:adjusted_association_estimates" in item.expected_outputs
        and "table:adjusted_association_estimates" in step.inputs
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
    requirement = requirements[0]
    terms = [
        term
        for term in requirement.model_terms or []
        if term.role == "exposure"
        and term.name == requirement.exposure_source
        and term.coding == "ordinal_linear"
        and term.transform == "declared_level_index"
        and len(term.levels or []) >= 3
    ]
    if len(terms) != 1:
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
        or len(remaining) != 1
        or len(trend_products) != 1
    ):
        return None
    levels = list(terms[0].levels or [])
    return OrderedStratifiedSpec(
        ordered_exposure=exposure,
        ordered_levels=levels,
        cochran_armitage_scores=list(range(len(levels))),
        binary_outcome=binary_outcome,
        continuous_outcome=remaining[0],
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
        "family_size": 2,
        "prespecified": True,
        "multiplicity_policy": "holm_familywise",
        "status": "ok",
    }


def _product_filename(product: str) -> str:
    return product.split(":", 1)[1] + ".csv"


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
    required = {spec.ordered_exposure, spec.binary_outcome, spec.continuous_outcome}
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
    continuous = pd.to_numeric(frame[spec.continuous_outcome], errors="coerce")
    continuous_finite = pd.Series(
        np.isfinite(continuous.to_numpy(dtype=float)), index=frame.index
    )
    invalid_continuous_n = int(
        (frame[spec.continuous_outcome].notna() & ~continuous_finite).sum()
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
        continuous_mask = level_mask & continuous_finite
        continuous_n = int(continuous_mask.sum())
        if binary_n <= 0 or continuous_n <= 0:
            raise RuntimeError(
                f"ordered level {level!r} has no valid binary or continuous outcome"
            )
        event_n = int(binary.loc[binary_mask].sum())
        events.append(event_n)
        totals.append(binary_n)
        ci = wilson_interval(event_n, binary_n, alpha=0.05)
        values = continuous.loc[continuous_mask]
        rows.append(
            {
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
                "continuous_outcome": spec.continuous_outcome,
                "continuous_n": continuous_n,
                "continuous_missing_n": level_n - continuous_n,
                "continuous_median": float(values.median()),
                "continuous_q25": float(values.quantile(0.25, interpolation="linear")),
                "continuous_q75": float(values.quantile(0.75, interpolation="linear")),
            }
        )

    ca = cochran_armitage_trend(
        events,
        totals,
        scores=spec.cochran_armitage_scores,
        group_order=spec.ordered_levels,
    )
    jt_mask = (membership == 1) & continuous_finite
    jt = jonckheere_terpstra_trend(
        continuous.loc[jt_mask].tolist(),
        frame.loc[jt_mask, spec.ordered_exposure].tolist(),
        group_order=spec.ordered_levels,
    )
    trend_rows = [
        _trend_row(
            ca,
            outcome=spec.binary_outcome,
            outcome_type="binary",
            test_id="cochran_armitage",
            levels_with_data=len(spec.ordered_levels),
        ),
        _trend_row(
            jt,
            outcome=spec.continuous_outcome,
            outcome_type="continuous",
            test_id="jonckheere_terpstra",
            levels_with_data=len(spec.ordered_levels),
        ),
    ]
    adjusted = _holm([float(ca.p_value), float(jt.p_value)])
    for row, value in zip(trend_rows, adjusted):
        row["adjusted_p"] = value

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
        "continuous_outcome_column": spec.continuous_outcome,
        "locked_cohort_n": int(len(frame)),
        "valid_ordered_exposure_n": valid_exposure_n,
        "ci_method": "wilson_score",
        "ci_alpha": 0.05,
        "continuous_summary": "median_iqr",
        "quantile_method": "linear",
        "stratified_table": stratified_path.name,
        "trend_table": trend_path.name,
        "tests": {
            "binary": {"test_id": "cochran_armitage", "alternative": "two-sided"},
            "continuous": {"test_id": "jonckheere_terpstra", "alternative": "two-sided"},
        },
        "multiplicity_policy": "holm_familywise",
        "multiplicity_family_size": 2,
    }
    summary = {
        "status": "ok",
        "analysis_family": "association",
        "interpretation_class": "ordered_stratified_secondary",
        "interpretation_ceiling": "secondary_unadjusted_not_causal",
        "analysis_role": analysis_role,
        "analysis_set": "exposure_observed_per_outcome_available_case",
        "source_cohort": cohort_path.name,
        "source_row_count_reconciliation": {
            "source_rows": int(len(frame)),
            "valid_ordered_exposure_rows": valid_exposure_n,
            "excluded_missing_exposure_rows": int(len(frame) - valid_exposure_n),
            "filtering_performed": valid_exposure_n != len(frame),
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

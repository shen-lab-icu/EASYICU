"""Contracts for deterministic ordered-group trend analyses.

The Planner declares the scientific coordinates while a host executor builds
the output tables.  This module supplies their dependency-neutral trust
boundary and contains no benchmark-, database-, or clinical-variable aliases.

Contract ``1.1`` (the host owner's current shape) makes the continuous outcome
optional and adds the adjusted per-level trend refit of the primary model as
one more trend-table row, checked against its recorded fit.  Contract ``1.0``
(binary and continuous outcomes, two crude tests) stays readable unchanged.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..methods.ordered_trends import (
    OrderedTrendResult,
    cochran_armitage_trend,
    jonckheere_terpstra_trend,
    normal_tail_probability,
    wilson_interval,
)
from .model_terms import level_spelling
from .ordered_stratified_spec import TREND_EFFECT_MEASURE_BY_KIND
from ..schema import AnalysisStep, ValidationFinding

CONTROLLED_METHOD = "ordinal_stratified_descriptive_analysis"
SCIENTIFIC_ACTION_ID = "association.ordinal_trend"
PARENT_PRODUCT = "table:adjusted_association_estimates"
CONTRACT_KEY = "ordered_stratified_contract"
CONTRACT_SCHEMA_VERSION = "1.1"
LEGACY_CONTRACT_SCHEMA_VERSION = "1.0"
ADJUSTED_TREND_TEST_ID = "adjusted_ordinal_linear_trend"
ORDERED_STRATIFIED_INTERPRETATION_CLASS = "ordered_stratified_secondary"
SECONDARY_REPORTING_SCHEMA_VERSION = "easyicu.ordered_stratified_reporting/2"
#: The envelope the host compiles the adjusted-trend claim from.
TREND_REPORTING_KEY = "reportable_trend_results"
TREND_REPORTING_SCHEMA_VERSION = "easyicu.ordered_trend_reporting/1"

_FIGURE_OUTPUT_KINDS = frozenset({"figure", "plot", "chart", "fig", "heatmap"})
_LEGACY_RENDERING_PRODUCTS = frozenset(
    {"figure", "plot", "chart", "heatmap", "visual", "visualization"}
)

_STRATIFIED_BINARY_COLUMNS = {
    "level_value",
    "level_order",
    "level_n",
    "level_percentage",
    "binary_outcome",
    "binary_n",
    "binary_missing_n",
    "binary_event_n",
    "binary_risk",
    "binary_percentage",
    "binary_ci_low",
    "binary_ci_high",
    "binary_ci_method",
    "binary_ci_alpha",
}
_STRATIFIED_CONTINUOUS_COLUMNS = {
    "continuous_outcome",
    "continuous_n",
    "continuous_missing_n",
    "continuous_median",
    "continuous_q25",
    "continuous_q75",
}
_STRATIFIED_REQUIRED_COLUMNS = _STRATIFIED_BINARY_COLUMNS | _STRATIFIED_CONTINUOUS_COLUMNS

_TREND_REQUIRED_COLUMNS = {
    "outcome",
    "outcome_type",
    "test_id",
    "test_name",
    "alternative",
    "n",
    "levels_with_data",
    "statistic",
    "statistic_name",
    "expected_statistic",
    "variance",
    "z_statistic",
    "chi_square",
    "effect_size",
    "effect_size_name",
    "p_value",
    "adjusted_p",
    "p_value_reporting",
    "log_p_value",
    "negative_log10_p",
    "p_value_bounded",
    "tie_correction",
    "continuity_correction",
    "implementation",
    "score_scheme",
    "family_id",
    "family_size",
    "prespecified",
    "multiplicity_policy",
    "status",
}


def is_ordered_stratified_analysis_step(step: AnalysisStep) -> bool:
    """True only for the non-rendering controlled analysis step."""

    if str(step.method or "").strip().lower() != CONTROLLED_METHOD:
        return False

    def _is_rendering_product(value: object) -> bool:
        token = str(value or "").strip().lower()
        kind, separator, product = token.partition(":")
        if separator:
            # A typed kind is authoritative.  A table/artifact name containing
            # ``figure`` remains an analysis product, not a rendering child.
            return kind.strip() in _FIGURE_OUTPUT_KINDS and bool(product.strip())
        return token in _LEGACY_RENDERING_PRODUCTS

    outputs = [value for value in step.expected_outputs if str(value or "").strip()]
    if outputs and all(_is_rendering_product(value) for value in outputs):
        return False
    return True


def has_fixed_ordered_stratified_input_roster(step: AnalysisStep) -> bool:
    """Whether the host contract owns this step's exact scientific inputs."""

    return (
        is_ordered_stratified_analysis_step(step)
        and step.scientific_action_id == SCIENTIFIC_ACTION_ID
    )


def _finding(kind: str, message: str, **detail: Any) -> ValidationFinding:
    return ValidationFinding(
        validator="ordered_stratified_contract",
        severity="error",
        message=message,
        detail={"kind": kind, **detail},
    )


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_positive_int(value: Any) -> bool:
    return _is_finite_number(value) and float(value).is_integer() and float(value) > 0


def _declares_continuous(contract: Mapping[str, Any]) -> bool:
    """A 1.0 contract always has a continuous outcome; 1.1 only when named."""

    if contract.get("schema_version") == LEGACY_CONTRACT_SCHEMA_VERSION:
        return True
    return str(contract.get("continuous_outcome_column") or "").strip() != ""


def _adjusted_trend_issues(record: Any) -> list[str]:
    """Fields of a recorded per-level trend fit that are absent or incoherent."""

    if not isinstance(record, Mapping):
        return ["adjusted_trend"]
    issues: list[str] = []
    if record.get("test_id") != ADJUSTED_TREND_TEST_ID:
        issues.append("test_id")
    kind = record.get("estimator_kind")
    if kind not in TREND_EFFECT_MEASURE_BY_KIND:
        issues.append("estimator_kind")
    elif record.get("effect_scale") != TREND_EFFECT_MEASURE_BY_KIND[kind]:
        issues.append("effect_scale")
    if record.get("increment") != "one_declared_level":
        issues.append("increment")
    numbers = {
        key: float(record[key]) if _is_finite_number(record.get(key)) else None
        for key in ("estimate", "ci_low", "ci_high", "standard_error", "p_value")
    }
    issues.extend(key for key, value in numbers.items() if value is None)
    if not issues:
        if not 0.0 < numbers["ci_low"] <= numbers["estimate"] <= numbers["ci_high"]:
            issues.append("effect_interval_order")
        if numbers["standard_error"] <= 0.0:
            issues.append("standard_error")
        if not 0.0 < numbers["p_value"] <= 1.0:
            issues.append("p_value")
    n = record.get("n")
    events = record.get("n_events")
    if not _is_positive_int(n):
        issues.append("n")
    if not (
        _is_finite_number(events)
        and float(events).is_integer()
        and 0 <= float(events) <= (float(n) if _is_finite_number(n) else -1)
    ):
        issues.append("n_events")
    covariates = record.get("covariates")
    if not isinstance(covariates, list) or not all(isinstance(item, str) for item in covariates):
        issues.append("covariates")
    if not str(record.get("model_id") or "").strip():
        issues.append("model_id")
    # The refit reproduces the primary model's analysed population exactly.
    if not str(record.get("primary_model_id") or "").strip():
        issues.append("primary_model_id")
    if record.get("primary_n") != n or not _is_positive_int(record.get("primary_n")):
        issues.append("primary_n")
    if record.get("primary_n_events") != events:
        issues.append("primary_n_events")
    return issues


def _trend_reporting_issues(
    step_summary: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[str]:
    """Reported trend fields that differ from the recorded refit.

    The claim envelope and the Writer's secondary envelope are copies of
    ``adjusted_trend``; a value in either that is not the recorded one would
    reach a sentence the replay never checked.
    """

    record = contract.get("adjusted_trend")
    if not isinstance(record, Mapping):
        return []  # reported by ``_adjusted_trend_issues``
    levels = contract.get("ordered_levels")
    expected_claim = {
        "schema_version": TREND_REPORTING_SCHEMA_VERSION,
        "exposure": contract.get("ordered_exposure_column"),
        "ordered_levels": (
            [level_spelling(value) for value in levels] if isinstance(levels, list) else None
        ),
        "outcome": contract.get("binary_outcome_column"),
        "analysis_set": record.get("analysis_set"),
        "adjustment_covariates": record.get("covariates"),
        "effect_scale": record.get("effect_scale"),
        "estimate": record.get("estimate"),
        "lower": record.get("ci_low"),
        "upper": record.get("ci_high"),
        "p_value": record.get("p_value"),
        "n": record.get("n"),
        "events": record.get("n_events"),
    }
    expected_secondary = {
        "interpretation_ceiling": "secondary_adjusted_association_not_causal",
        "effect_scale": record.get("effect_scale"),
        "increment": record.get("increment"),
        "estimate": record.get("estimate"),
        "ci_low": record.get("ci_low"),
        "ci_high": record.get("ci_high"),
        "p_value_reporting": record.get("p_value_reporting"),
        "n": record.get("n"),
        "n_events": record.get("n_events"),
        "covariates": record.get("covariates"),
    }
    issues: list[str] = []
    for name, observed, expected in (
        (TREND_REPORTING_KEY, step_summary.get(TREND_REPORTING_KEY), expected_claim),
        (
            "reportable_secondary_results.adjusted_trend",
            (step_summary.get("reportable_secondary_results") or {}).get("adjusted_trend")
            if isinstance(step_summary.get("reportable_secondary_results"), Mapping)
            else None,
            expected_secondary,
        ),
    ):
        if not isinstance(observed, Mapping):
            issues.append(name)
            continue
        issues.extend(
            f"{name}.{key}" for key, value in expected.items() if observed.get(key) != value
        )
    # The envelope around the adjusted ratio is an association, and its crude
    # tests say so: neither may read as adjusted or causal.
    secondary = step_summary.get("reportable_secondary_results")
    if isinstance(secondary, Mapping):
        if secondary.get("interpretation_ceiling") != "secondary_association_not_causal":
            issues.append("reportable_secondary_results.interpretation_ceiling")
        crude_parts = ["binary_trend"] + (
            ["continuous_trend"] if _declares_continuous(contract) else []
        )
        for part in crude_parts:
            block = secondary.get(part)
            if not isinstance(block, Mapping) or block.get("adjustment") != "crude":
                issues.append(f"reportable_secondary_results.{part}.adjustment")
    return issues


def _safe_basename(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or Path(text).name != text or Path(text).suffix.lower() != ".csv":
        return None
    return text


def ordered_stratified_structure_findings(
    *, step: AnalysisStep, step_summary: Mapping[str, Any]
) -> list[ValidationFinding]:
    """Validate the closed declaration without recomputing any results."""

    if not is_ordered_stratified_analysis_step(step):
        return []
    findings: list[ValidationFinding] = []
    contract = step_summary.get(CONTRACT_KEY)
    if not isinstance(contract, Mapping):
        return [
            _finding(
                "missing_contract",
                f"Controlled method {CONTROLLED_METHOD!r} requires a top-level "
                f"{CONTRACT_KEY!r} object so its variables, order, denominators, "
                "tests, and output tables can be independently replayed.",
                step_id=step.step_id,
            )
        ]

    def require_equal(key: str, expected: Any) -> None:
        if contract.get(key) != expected:
            findings.append(
                _finding(
                    "invalid_contract_field",
                    f"{CONTRACT_KEY}.{key} must equal {expected!r}; saw "
                    f"{contract.get(key)!r}.",
                    step_id=step.step_id,
                    field=key,
                    expected=expected,
                    observed=contract.get(key),
                )
            )

    if contract.get("schema_version") not in {
        CONTRACT_SCHEMA_VERSION,
        LEGACY_CONTRACT_SCHEMA_VERSION,
    }:
        require_equal("schema_version", CONTRACT_SCHEMA_VERSION)
    legacy = contract.get("schema_version") == LEGACY_CONTRACT_SCHEMA_VERSION
    continuous = _declares_continuous(contract)
    require_equal("ci_method", "wilson_score")
    require_equal("ci_alpha", 0.05)
    if continuous:
        require_equal("continuous_summary", "median_iqr")
        require_equal("quantile_method", "linear")
    require_equal("multiplicity_policy", "holm_familywise")
    require_equal("multiplicity_family_size", 2 if continuous else 1)
    if not legacy:
        issues = _adjusted_trend_issues(contract.get("adjusted_trend"))
        if issues:
            findings.append(
                _finding(
                    "invalid_adjusted_trend",
                    f"{CONTRACT_KEY}.adjusted_trend must record the per-level "
                    "refit of the primary model (one estimator-matched ratio "
                    "with a finite interval containing it, its standard error, "
                    "p-value, model rows and events); invalid fields: "
                    + ", ".join(issues)
                    + ".",
                    step_id=step.step_id,
                    invalid_fields=issues,
                )
            )
        reporting = _trend_reporting_issues(step_summary, contract)
        if reporting:
            findings.append(
                _finding(
                    "trend_reporting_mismatch",
                    "The reported adjusted trend must repeat the recorded "
                    f"{CONTRACT_KEY}.adjusted_trend values; differing fields: "
                    + ", ".join(reporting)
                    + ".",
                    step_id=step.step_id,
                    invalid_fields=reporting,
                )
            )

    plan_inputs = {str(value) for value in step.inputs}
    column_keys = (
        "ordered_exposure_column",
        "binary_outcome_column",
        *(("continuous_outcome_column",) if continuous else ()),
    )
    declared_columns: list[str] = []
    for key in column_keys:
        value = str(contract.get(key) or "").strip()
        if not value or value not in plan_inputs:
            findings.append(
                _finding(
                    "invalid_planned_column",
                    f"{CONTRACT_KEY}.{key} must name one exact current-step input; "
                    f"saw {value!r} with inputs {sorted(plan_inputs)!r}.",
                    step_id=step.step_id,
                    field=key,
                    observed=value or None,
                    plan_inputs=sorted(plan_inputs),
                )
            )
        declared_columns.append(value)
    if len(set(declared_columns)) != len(declared_columns):
        findings.append(
            _finding(
                "overlapping_roles",
                "The ordered exposure, binary outcome, and any continuous outcome "
                "must be distinct planned columns.",
                step_id=step.step_id,
                declared_columns=declared_columns,
            )
        )

    levels = contract.get("ordered_levels")
    scores = contract.get("cochran_armitage_scores")
    valid_levels = (
        isinstance(levels, list)
        and len(levels) >= 2
        and all(
            value is not None
            and isinstance(value, (str, int, float, bool))
            and (not isinstance(value, float) or math.isfinite(value))
            for value in levels
        )
    )
    if valid_levels:
        try:
            valid_levels = len({json.dumps(v, sort_keys=True) for v in levels}) == len(
                levels
            )
        except (TypeError, ValueError):
            valid_levels = False
    if not valid_levels:
        findings.append(
            _finding(
                "invalid_ordered_levels",
                f"{CONTRACT_KEY}.ordered_levels must be an explicit list of at "
                "least two unique JSON scalar levels in scientific order.",
                step_id=step.step_id,
                ordered_levels=levels,
            )
        )
    valid_scores = (
        isinstance(scores, list)
        and isinstance(levels, list)
        and len(scores) == len(levels)
        and all(_is_finite_number(value) for value in scores)
        and all(float(b) > float(a) for a, b in zip(scores, scores[1:]))
    )
    if not valid_scores:
        findings.append(
            _finding(
                "invalid_ca_scores",
                f"{CONTRACT_KEY}.cochran_armitage_scores must contain one finite, "
                "strictly increasing numeric score per ordered level. These scores "
                "are an explicit spacing assumption, not an order-only claim.",
                step_id=step.step_id,
                ordered_levels=levels,
                scores=scores,
            )
        )
    if contract.get("score_scheme") not in {
        "consecutive_ordinal_ranks",
        "prespecified_numeric_scores",
    }:
        findings.append(
            _finding(
                "invalid_score_scheme",
                f"{CONTRACT_KEY}.score_scheme must disclose either consecutive "
                "ordinal ranks or prespecified numeric scores.",
                step_id=step.step_id,
                observed=contract.get("score_scheme"),
            )
        )

    locked_n = contract.get("locked_cohort_n")
    valid_n = contract.get("valid_ordered_exposure_n")
    if not _is_positive_int(locked_n):
        findings.append(
            _finding(
                "invalid_locked_n",
                f"{CONTRACT_KEY}.locked_cohort_n must be a positive integer.",
                step_id=step.step_id,
                observed=locked_n,
            )
        )
    if not (
        _is_positive_int(valid_n)
        and _is_positive_int(locked_n)
        and int(valid_n) <= int(locked_n)
    ):
        findings.append(
            _finding(
                "invalid_valid_ordered_n",
                f"{CONTRACT_KEY}.valid_ordered_exposure_n must be a positive "
                "integer no larger than locked_cohort_n.",
                step_id=step.step_id,
                observed=valid_n,
                locked_cohort_n=locked_n,
            )
        )

    for key in ("stratified_table", "trend_table"):
        if _safe_basename(contract.get(key)) is None:
            findings.append(
                _finding(
                    "invalid_output_filename",
                    f"{CONTRACT_KEY}.{key} must be a CSV basename inside "
                    "STEP_OUT_DIR, not a path or non-CSV artefact.",
                    step_id=step.step_id,
                    field=key,
                    observed=contract.get(key),
                )
            )

    tests = contract.get("tests")
    expected_tests: dict[str, Any] = {
        "binary": {
            "test_id": "cochran_armitage",
            "alternative": "two-sided",
        },
    }
    if continuous:
        expected_tests["continuous"] = {
            "test_id": "jonckheere_terpstra",
            "alternative": "two-sided",
        }
    if tests != expected_tests:
        findings.append(
            _finding(
                "invalid_test_declaration",
                f"{CONTRACT_KEY}.tests must declare the exact controlled binary "
                "and any continuous trend test with two-sided alternatives.",
                step_id=step.step_id,
                expected=expected_tests,
                observed=tests,
            )
        )

    if contract.get("execution_owner") != "ordered_stratified_executor_v1":
        outputs = [str(value or "").lower() for value in step.expected_outputs]
        if not any(
            value.startswith("table:") and "stratified" in value
            for value in outputs
        ):
            findings.append(
                _finding(
                    "missing_planned_stratified_output",
                    "The controlled method requires a planned table:*stratified* output.",
                    step_id=step.step_id,
                    expected_outputs=list(step.expected_outputs),
                )
            )
        if not any(value == "test:ordinal_trend" for value in outputs):
            findings.append(
                _finding(
                    "missing_planned_trend_output",
                    "The controlled method requires expected output test:ordinal_trend.",
                    step_id=step.step_id,
                    expected_outputs=list(step.expected_outputs),
                )
            )
    return findings


def ordered_stratified_script_findings(
    *, step: AnalysisStep, script_text: str
) -> list[ValidationFinding]:
    """Require the controlled primitives in agent-authored source code."""

    if not is_ordered_stratified_analysis_step(step):
        return []
    try:
        tree = ast.parse(script_text)
    except SyntaxError:
        return []

    called_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            called_names.add(target.id)
        elif isinstance(target, ast.Attribute):
            called_names.add(target.attr)

    if "run_ordered_stratified_from_env" in called_names:
        # This exact entrypoint is the host adapter. Its body calls the reviewed
        # primitives; legacy agent-authored implementations remain subject to
        # the direct-call audit below.
        return []

    findings: list[ValidationFinding] = []
    # A binary-only step (the ordered exposure and its 0/1 outcome) has no
    # continuous outcome for a Jonckheere-Terpstra test to read.
    continuous = len([value for value in step.inputs if ":" not in str(value)]) >= 3
    required = {
        "wilson_interval",
        "cochran_armitage_trend",
        *(("jonckheere_terpstra_trend",) if continuous else ()),
    }
    missing = sorted(required - called_names)
    if missing:
        findings.append(
            _finding(
                "missing_validated_primitive_call",
                "The controlled ordered-stratified analysis must call the "
                "validated Wilson and Cochran-Armitage primitives"
                + (", and Jonckheere-Terpstra for its continuous outcome" if continuous else "")
                + f"; missing calls: {', '.join(missing)}.",
                step_id=step.step_id,
                missing_calls=missing,
            )
        )
    if "spearmanr" in called_names:
        findings.append(
            _finding(
                "spearman_substituted_for_jt",
                "Spearman correlation is not a Jonckheere-Terpstra equivalent. "
                "Use individual observations with jonckheere_terpstra_trend for "
                "the planned ordered-shift test.",
                step_id=step.step_id,
            )
        )
    return findings


def _labels_equal(left: Any, right: Any) -> bool:
    if _is_finite_number(left) and _is_finite_number(right):
        return float(left) == float(right)
    return str(left) == str(right)


def _level_mask(series: pd.Series, level: Any) -> pd.Series:
    if _is_finite_number(level):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric == float(level)
    return series.notna() & (series.astype(str) == str(level))


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _close(left: Any, right: Any, *, atol: float = 1e-9) -> bool:
    if left is None and right is None:
        return True
    if pd.isna(left) and right is None:
        return True
    if left is None and pd.isna(right):
        return True
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(left_float) or not math.isfinite(right_float):
        return False
    return bool(np.isclose(left_float, right_float, rtol=1e-7, atol=atol))


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm family-wise adjusted values in original order."""

    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p, kind="mergesort")
    adjusted_sorted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (len(p) - rank) * float(p[original_index]))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def _require_columns(
    table: pd.DataFrame,
    required: Iterable[str],
    *,
    table_name: str,
    step_id: str,
) -> list[ValidationFinding]:
    missing = sorted(set(required) - {str(column) for column in table.columns})
    if not missing:
        return []
    return [
        _finding(
            "missing_table_columns",
            f"Controlled table {table_name!r} is missing required columns: "
            + ", ".join(missing)
            + ".",
            step_id=step_id,
            table=table_name,
            missing_columns=missing,
        )
    ]


def _comparison_finding(
    *,
    kind: str,
    table: str,
    field: str,
    expected: Any,
    observed: Any,
    row: Any,
    step_id: str,
) -> ValidationFinding:
    return _finding(
        kind,
        f"{table} row {row!r} field {field!r} disagrees with locked-cohort "
        f"replay: observed {observed!r}, expected {expected!r}.",
        step_id=step_id,
        table=table,
        row=row,
        field=field,
        observed=observed,
        expected=expected,
    )


def _trend_expected_payload(result: OrderedTrendResult) -> dict[str, Any]:
    return {
        "n": result.n,
        "statistic": result.statistic,
        "statistic_name": result.statistic_type,
        "expected_statistic": result.expected_statistic,
        "variance": result.variance,
        "z_statistic": result.z_statistic,
        "chi_square": result.chi_square,
        "effect_size": result.effect_size,
        "effect_size_name": result.effect_size_name,
        "p_value": result.p_value,
        "p_value_reporting": result.p_value_reporting,
        "log_p_value": result.log_p_value,
        "negative_log10_p": result.negative_log10_p,
        "p_value_bounded": result.p_value_bounded,
        "tie_correction": result.tie_correction,
        "continuity_correction": result.continuity_correction,
        "implementation": result.implementation,
        "score_scheme": result.score_scheme,
    }


def ordered_stratified_numeric_findings(
    *,
    cohort_path: Path,
    step: AnalysisStep,
    out_dir: Path,
    step_summary: Mapping[str, Any],
) -> list[ValidationFinding]:
    """Replay every controlled descriptive and trend statistic from the cohort."""

    if not is_ordered_stratified_analysis_step(step):
        return []
    structural = ordered_stratified_structure_findings(
        step=step, step_summary=step_summary
    )
    if structural:
        # Structural errors belong to the ordinary step-contract channel.  The
        # numeric replay stays silent until that declaration is valid, avoiding
        # duplicate findings in the early repair and final manifest channels.
        return []
    contract = step_summary[CONTRACT_KEY]
    assert isinstance(contract, Mapping)

    findings: list[ValidationFinding] = []
    try:
        df = pd.read_parquet(cohort_path)
    except Exception as exc:
        return [
            _finding(
                "cohort_replay_failed",
                f"Could not read the locked cohort for ordered-stratified replay: {exc}",
                step_id=step.step_id,
            )
        ]

    exposure = str(contract["ordered_exposure_column"])
    binary_outcome = str(contract["binary_outcome_column"])
    legacy = contract.get("schema_version") == LEGACY_CONTRACT_SCHEMA_VERSION
    has_continuous = _declares_continuous(contract)
    continuous_outcome = (
        str(contract["continuous_outcome_column"]) if has_continuous else None
    )
    required_cohort_columns = {exposure, binary_outcome}
    if continuous_outcome is not None:
        required_cohort_columns.add(continuous_outcome)
    missing_cohort = sorted(required_cohort_columns - set(df.columns))
    if missing_cohort:
        return [
            _finding(
                "missing_cohort_columns",
                "Locked cohort lacks declared ordered-stratified columns: "
                + ", ".join(missing_cohort)
                + ".",
                step_id=step.step_id,
                missing_columns=missing_cohort,
            )
        ]

    levels = list(contract["ordered_levels"])
    scores = [float(value) for value in contract["cochran_armitage_scores"]]
    level_masks = [_level_mask(df[exposure], level) for level in levels]
    membership_count = sum(mask.astype(int) for mask in level_masks)
    overlapping_n = int((membership_count > 1).sum())
    if overlapping_n:
        return [
            _finding(
                "overlapping_level_membership",
                f"Declared ordered levels overlap for {overlapping_n} cohort rows.",
                step_id=step.step_id,
                overlapping_n=overlapping_n,
            )
        ]
    valid_exposure_mask = membership_count == 1
    valid_exposure_n = int(valid_exposure_mask.sum())
    if valid_exposure_n <= 0:
        return [
            _finding(
                "empty_ordered_analysis_set",
                "No locked-cohort rows belong to the explicitly declared ordered "
                "levels, so the controlled analysis is not estimable.",
                step_id=step.step_id,
                ordered_levels=levels,
            )
        ]
    if int(contract["locked_cohort_n"]) != len(df):
        findings.append(
            _finding(
                "locked_denominator_mismatch",
                f"Declared locked_cohort_n={contract['locked_cohort_n']} but the "
                f"locked cohort contains {len(df)} rows.",
                step_id=step.step_id,
                observed=contract["locked_cohort_n"],
                expected=len(df),
            )
        )
    if int(contract["valid_ordered_exposure_n"]) != valid_exposure_n:
        findings.append(
            _finding(
                "ordered_denominator_mismatch",
                f"Declared valid_ordered_exposure_n="
                f"{contract['valid_ordered_exposure_n']} but cohort replay found "
                f"{valid_exposure_n} rows in the declared levels.",
                step_id=step.step_id,
                observed=contract["valid_ordered_exposure_n"],
                expected=valid_exposure_n,
            )
        )

    binary_numeric = pd.to_numeric(df[binary_outcome], errors="coerce")
    binary_nonmissing = df[binary_outcome].notna()
    binary_valid = binary_numeric.isin([0.0, 1.0])
    invalid_binary_n = int((binary_nonmissing & ~binary_valid).sum())
    if invalid_binary_n:
        findings.append(
            _finding(
                "invalid_binary_domain",
                f"Declared binary outcome contains {invalid_binary_n} non-missing "
                "values outside the exact 0/1 domain.",
                step_id=step.step_id,
                column=binary_outcome,
                invalid_n=invalid_binary_n,
            )
        )

    continuous_numeric = (
        pd.to_numeric(df[continuous_outcome], errors="coerce")
        if continuous_outcome is not None
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    continuous_finite = pd.Series(
        np.isfinite(continuous_numeric.to_numpy(dtype=float)), index=df.index
    )
    invalid_continuous_n = (
        int((df[continuous_outcome].notna() & ~continuous_finite).sum())
        if continuous_outcome is not None
        else 0
    )
    if invalid_continuous_n:
        findings.append(
            _finding(
                "invalid_continuous_values",
                f"Declared continuous outcome contains {invalid_continuous_n} "
                "non-missing non-numeric or non-finite values.",
                step_id=step.step_id,
                column=continuous_outcome,
                invalid_n=invalid_continuous_n,
            )
        )

    stratified_name = str(contract["stratified_table"])
    trend_name = str(contract["trend_table"])
    stratified_path = out_dir / stratified_name
    trend_path = out_dir / trend_name
    missing_files = [
        path.name for path in (stratified_path, trend_path) if not path.is_file()
    ]
    if missing_files:
        findings.append(
            _finding(
                "missing_output_table",
                "Controlled ordered-stratified output table(s) are missing: "
                + ", ".join(missing_files)
                + ".",
                step_id=step.step_id,
                missing_files=missing_files,
            )
        )
        return findings
    try:
        stratified = pd.read_csv(stratified_path)
        trend = pd.read_csv(trend_path)
    except Exception as exc:
        findings.append(
            _finding(
                "unreadable_output_table",
                f"Could not read controlled ordered-stratified tables: {exc}",
                step_id=step.step_id,
            )
        )
        return findings
    schema_findings = _require_columns(
        stratified,
        _STRATIFIED_REQUIRED_COLUMNS if has_continuous else _STRATIFIED_BINARY_COLUMNS,
        table_name=stratified_name,
        step_id=step.step_id,
    ) + _require_columns(
        trend,
        _TREND_REQUIRED_COLUMNS,
        table_name=trend_name,
        step_id=step.step_id,
    )
    findings.extend(schema_findings)
    if schema_findings:
        return findings

    if len(stratified) != len(levels):
        findings.append(
            _finding(
                "wrong_level_row_count",
                f"{stratified_name} must contain exactly one row per declared "
                f"ordered level ({len(levels)}); saw {len(stratified)}.",
                step_id=step.step_id,
                expected_n=len(levels),
                observed_n=len(stratified),
            )
        )
        return findings

    event_counts: list[int] = []
    binary_totals: list[int] = []
    expected_level_rows: list[dict[str, Any]] = []
    for level_order, (level, level_mask) in enumerate(zip(levels, level_masks)):
        level_n = int(level_mask.sum())
        binary_mask = level_mask & binary_valid
        binary_n = int(binary_mask.sum())
        continuous_mask = level_mask & continuous_finite
        continuous_n = int(continuous_mask.sum())
        if binary_n <= 0 or (has_continuous and continuous_n <= 0):
            findings.append(
                _finding(
                    "empty_outcome_level",
                    f"Declared ordered level {level!r} has no valid "
                    f"{'binary' if binary_n <= 0 else 'continuous'} outcome "
                    "observations; the planned trend contract is not estimable.",
                    step_id=step.step_id,
                    level=level,
                    binary_n=binary_n,
                    continuous_n=continuous_n,
                )
            )
            continue
        event_n = int(binary_numeric.loc[binary_mask].sum())
        binary_totals.append(binary_n)
        event_counts.append(event_n)
        ci = wilson_interval(event_n, binary_n, alpha=float(contract["ci_alpha"]))

        continuous_values = continuous_numeric.loc[continuous_mask]
        expected_row = {
                "level_value": level,
                "level_order": level_order,
                "level_n": level_n,
                "level_percentage": 100.0 * level_n / valid_exposure_n,
                "binary_outcome": binary_outcome,
                "binary_n": binary_n,
                "binary_missing_n": level_n - binary_n,
                "binary_event_n": event_n,
                "binary_risk": event_n / binary_n,
                "binary_percentage": 100.0 * event_n / binary_n,
                "binary_ci_low": ci.ci_low,
                "binary_ci_high": ci.ci_high,
                "binary_ci_method": "wilson_score",
                "binary_ci_alpha": float(contract["ci_alpha"]),
        }
        if has_continuous:
            expected_row.update(
                {
                    "continuous_outcome": continuous_outcome,
                    "continuous_n": continuous_n,
                    "continuous_missing_n": level_n - continuous_n,
                    "continuous_median": float(continuous_values.median()),
                    "continuous_q25": float(
                        continuous_values.quantile(0.25, interpolation="linear")
                    ),
                    "continuous_q75": float(
                        continuous_values.quantile(0.75, interpolation="linear")
                    ),
                }
            )
        expected_level_rows.append(expected_row)

    if findings:
        return findings

    for expected_row in expected_level_rows:
        level = expected_row["level_value"]
        matches = stratified[
            stratified["level_value"].map(lambda value: _labels_equal(value, level))
        ]
        if len(matches) != 1:
            findings.append(
                _finding(
                    "missing_or_duplicate_level_row",
                    f"{stratified_name} must contain one row for declared level "
                    f"{level!r}; saw {len(matches)}.",
                    step_id=step.step_id,
                    level=level,
                    row_count=len(matches),
                )
            )
            continue
        row = matches.iloc[0]
        for field, expected_value in expected_row.items():
            observed_value = row[field]
            if field in {
                "level_value",
                "binary_outcome",
                "continuous_outcome",
                "binary_ci_method",
            }:
                equal = (
                    _labels_equal(observed_value, expected_value)
                    if field == "level_value"
                    else str(observed_value) == str(expected_value)
                )
            else:
                equal = _close(observed_value, expected_value)
            if not equal:
                findings.append(
                    _comparison_finding(
                        kind="stratified_value_mismatch",
                        table=stratified_name,
                        field=field,
                        expected=expected_value,
                        observed=observed_value,
                        row=level,
                        step_id=step.step_id,
                    )
                )

    jt_mask = valid_exposure_mask & continuous_finite
    try:
        ca_result = cochran_armitage_trend(
            event_counts,
            binary_totals,
            scores=scores,
            group_order=levels,
            alternative="two-sided",
        )
        jt_result = (
            jonckheere_terpstra_trend(
                continuous_numeric.loc[jt_mask].tolist(),
                df.loc[jt_mask, exposure].tolist(),
                group_order=levels,
                alternative="two-sided",
            )
            if has_continuous
            else None
        )
    except (TypeError, ValueError) as exc:
        findings.append(
            _finding(
                "trend_not_estimable",
                "Locked-cohort replay could not estimate the controlled trend "
                f"tests: {exc}",
                step_id=step.step_id,
            )
        )
        return findings
    expected_trends: dict[str, dict[str, Any]] = {
        "cochran_armitage": {
            "outcome": binary_outcome,
            "outcome_type": "binary",
            "test_id": "cochran_armitage",
            "test_name": ca_result.test_name,
            "alternative": "two-sided",
            "levels_with_data": len(levels),
            **_trend_expected_payload(ca_result),
        },
    }
    if jt_result is not None:
        expected_trends["jonckheere_terpstra"] = {
            "outcome": continuous_outcome,
            "outcome_type": "continuous",
            "test_id": "jonckheere_terpstra",
            "test_name": jt_result.test_name,
            "alternative": "two-sided",
            "levels_with_data": len(levels),
            **_trend_expected_payload(jt_result),
        }
    family = list(expected_trends)
    adjusted = _holm_adjust([expected_trends[key]["p_value"] for key in family])
    for key, value in zip(family, adjusted):
        expected_trends[key]["adjusted_p"] = value
        expected_trends[key].update(
            {
                "family_id": "ordered_trend_outcomes",
                "family_size": len(family),
                "prespecified": True,
                "multiplicity_policy": "holm_familywise",
                "status": "ok",
            }
        )
    if not legacy:
        expected_trends[ADJUSTED_TREND_TEST_ID] = _adjusted_trend_expected_row(
            contract["adjusted_trend"],
            binary_outcome=binary_outcome,
            levels_with_data=len(levels),
        )

    if len(trend) != len(expected_trends):
        findings.append(
            _finding(
                "wrong_trend_row_count",
                f"{trend_name} must contain exactly the {len(expected_trends)} planned "
                f"trend rows; saw {len(trend)} rows.",
                step_id=step.step_id,
                observed_n=len(trend),
                expected_n=len(expected_trends),
            )
        )
        return findings
    for test_id, expected_row in expected_trends.items():
        matches = trend[trend["test_id"].astype(str) == test_id]
        if len(matches) != 1:
            findings.append(
                _finding(
                    "missing_or_duplicate_trend_row",
                    f"{trend_name} must contain one {test_id!r} row; "
                    f"saw {len(matches)}.",
                    step_id=step.step_id,
                    outcome=expected_row["outcome"],
                    test_id=test_id,
                    row_count=len(matches),
                )
            )
            continue
        row = matches.iloc[0]
        outcome = expected_row["outcome"]
        method_text = " ".join(
            str(row.get(field, ""))
            for field in ("test_id", "test_name", "implementation")
        ).lower()
        if "spearman" in method_text:
            findings.append(
                _finding(
                    "spearman_substituted_for_jt",
                    f"{trend_name} labels or implements a Spearman calculation "
                    "inside the controlled ordered-trend contract. Spearman is not "
                    "a Jonckheere-Terpstra equivalent.",
                    step_id=step.step_id,
                    outcome=outcome,
                    method_text=method_text,
                )
            )
        raw_p = row["p_value"]
        if not _is_finite_number(raw_p) or not 0.0 < float(raw_p) <= 1.0:
            findings.append(
                _finding(
                    "invalid_p_value",
                    f"{trend_name} outcome {outcome!r} must serialize a finite "
                    "p-value in (0, 1]; p=0 is not a reportable probability. "
                    "Use the bounded numeric value plus log_p_value metadata.",
                    step_id=step.step_id,
                    outcome=outcome,
                    observed=raw_p,
                )
            )
        for field, expected_value in expected_row.items():
            if field not in row.index:
                findings.append(
                    _comparison_finding(
                        kind="trend_value_mismatch",
                        table=trend_name,
                        field=field,
                        expected=expected_value,
                        observed=None,
                        row=test_id,
                        step_id=step.step_id,
                    )
                )
                continue
            observed_value = row[field]
            if isinstance(expected_value, bool):
                equal = _bool_value(observed_value) is expected_value
            elif isinstance(expected_value, (int, float)) and not isinstance(
                expected_value, bool
            ):
                equal = _close(observed_value, expected_value)
            elif expected_value is None:
                equal = pd.isna(observed_value)
            else:
                equal = str(observed_value) == str(expected_value)
            if not equal:
                findings.append(
                    _comparison_finding(
                        kind="trend_value_mismatch",
                        table=trend_name,
                        field=field,
                        expected=expected_value,
                        observed=observed_value,
                        row=test_id,
                        step_id=step.step_id,
                    )
                )
    return findings


def _adjusted_trend_expected_row(
    record: Mapping[str, Any], *, binary_outcome: str, levels_with_data: int
) -> dict[str, Any]:
    """The trend-table row the recorded per-level refit must appear as.

    The fit itself is the adjusted-association owner's; this replays its Wald
    test from the recorded estimate and standard error, so the table, the
    record and the p-value cannot disagree.
    """

    estimate = float(record["estimate"])
    standard_error = float(record["standard_error"])
    z_statistic = math.log(estimate) / standard_error
    tail = normal_tail_probability(z_statistic, "two-sided")
    return {
        "outcome": binary_outcome,
        "outcome_type": "binary",
        "test_id": ADJUSTED_TREND_TEST_ID,
        "alternative": "two-sided",
        "n": int(record["n"]),
        "levels_with_data": levels_with_data,
        "statistic": z_statistic,
        "statistic_name": "wald_z",
        "z_statistic": z_statistic,
        "chi_square": z_statistic**2,
        "effect_size": estimate,
        "effect_size_name": f"adjusted_{record['effect_scale']}_per_level",
        "p_value": tail["p_value"],
        "p_value_reporting": tail["p_value_reporting"],
        "adjusted_p": None,
        "family_id": "adjusted_ordered_trend",
        "family_size": 1,
        "prespecified": True,
        "multiplicity_policy": "single_model_based_test",
        "status": "ok",
        "effect_scale": record["effect_scale"],
        "estimate": estimate,
        "ci_low": float(record["ci_low"]),
        "ci_high": float(record["ci_high"]),
        "standard_error": standard_error,
        "n_events": int(record["n_events"]),
        "model_id": record["model_id"],
        "estimator_kind": record["estimator_kind"],
    }


__all__ = [
    "ADJUSTED_TREND_TEST_ID",
    "CONTRACT_KEY",
    "CONTRACT_SCHEMA_VERSION",
    "CONTROLLED_METHOD",
    "LEGACY_CONTRACT_SCHEMA_VERSION",
    "ORDERED_STRATIFIED_INTERPRETATION_CLASS",
    "PARENT_PRODUCT",
    "SCIENTIFIC_ACTION_ID",
    "SECONDARY_REPORTING_SCHEMA_VERSION",
    "TREND_REPORTING_KEY",
    "TREND_REPORTING_SCHEMA_VERSION",
    "is_ordered_stratified_analysis_step",
    "ordered_stratified_numeric_findings",
    "ordered_stratified_script_findings",
    "ordered_stratified_structure_findings",
]

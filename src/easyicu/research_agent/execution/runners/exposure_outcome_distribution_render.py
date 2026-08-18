"""Deterministic renderer for the closed exposure-outcome distribution product.

This renderer consumes **one** table and nothing else. That is the point of it:
the product it reads carries its own denominators, missing counts, event counts
and typed uncertainty policy, *and* the design that produced them, so there is no second
lookup into a cohort summary to make the percentages meaningful. A renderer
that needed two tables could not have its input contract closed before its
parent ran, which is what left the figure steps unresolvable in a preflight.

It draws what the parent already measured and decides nothing: no cohort, no
exposure, no outcome, no category, no denominator, no interval. What it does do
is **re-derive** every published quantity from the counts beside it, using the
uncertainty policy the table itself declares, and refuse to draw when
one disagrees. Recomputing with the producer's own kernel cannot catch a bug in
that kernel -- the producer verifies itself for that -- but it does catch a
table that was edited, truncated or rebuilt between the two steps, which is the
failure this boundary exists to stop.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import textwrap
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ...contracts.figure_plan import (
    EXPOSURE_OUTCOME_DISTRIBUTION_COUNTS_ONLY_FIGURE_PANELS,
    EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_PANELS,
    EXPOSURE_OUTCOME_DISTRIBUTION_INPUT,
)
from ...contracts.ownership_verdict import OwnershipVerdict
from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep
from ...numeric_scalars import coerce_optional_finite_float as _finite
from .exposure_outcome_distribution_executor import (
    COUNTS_ONLY_COVARIANCE,
    EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS,
    EXPOSURE_OUTCOME_DISTRIBUTION_CONTRAST_COLUMNS,
    EXPOSURE_OUTCOME_DISTRIBUTION_DESIGN_COLUMNS,
    EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT,
    STRUCTURAL_TOTAL_COVARIANCE,
    percentage,
    wilson_interval,
)
from .figure_input_capability import TypedInputCapability
from .planner_display_labels import planner_binary_level_labels
from .typed_input_binding import BoundTypedInput, load_typed_input

__all__ = [
    "EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_CAPABILITY",
    "EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT",
    "exposure_outcome_distribution_figure_declaration_verdict",
    "exposure_outcome_distribution_figure_code",
    "exposure_outcome_distribution_figure_owns_step",
    "run_exposure_outcome_distribution_figure",
]

EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT = EXPOSURE_OUTCOME_DISTRIBUTION_INPUT
if EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT != EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT:
    raise RuntimeError("distribution figure input drifted from its producer output")

#: Same rule as the missingness renderer: the figure product id is a
#: Planner-owned label that becomes a filename, never a capability claim.
_FIGURE_PRODUCT_ID = re.compile(r"[a-z][a-z0-9_]{0,127}")

EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_CAPABILITY = TypedInputCapability(
    required=frozenset({EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT}),
)

_OVERALL_ROLE = "overall"
_LEVEL_ROLE = "exposure_level"
_ANALYSIS_KIND = "exposure_outcome_distribution_figure"


def _is_safe_figure_product_id(value: Any) -> bool:
    return bool(_FIGURE_PRODUCT_ID.fullmatch(str(value or "")))


def _method_head(value: Any) -> str:
    return str(value or "").strip().lower().split(" with ", 1)[0]


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if kind != "figure" or not separator or not _is_safe_figure_product_id(product):
        return None
    return product


def exposure_outcome_distribution_figure_declaration_verdict(
    step: AnalysisStep,
) -> OwnershipVerdict:
    """Distinguish a different figure from this figure with the wrong row mode."""

    products = [_figure_product(value) for value in step.expected_outputs]
    if not (
        EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_CAPABILITY.admits(step.inputs)
        and step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and len(products) == 1
        and products[0] is not None
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
        and step.exposure_outcome_distribution_spec is None
    ):
        return OwnershipVerdict.wrong_shape(
            _ANALYSIS_KIND,
            reason=(
                "the step is not one auxiliary visualization of the exact "
                "exposure/outcome distribution table"
            ),
        )

    if not EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_CAPABILITY.admits_step(step):
        return OwnershipVerdict.incomplete_declaration(
            _ANALYSIS_KIND,
            missing=(
                "input_consumption_contracts["
                f"{EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT}].mode=all_rows",
            ),
            reason=(
                "the deterministic renderer re-derives and draws the two exposure "
                "levels plus the overall denominator from all rows; single_row or "
                "role-subset consumption cannot realize that declared figure"
            ),
        )

    return OwnershipVerdict.claim(
        _ANALYSIS_KIND,
        reason="the exact typed distribution and its all_rows contract are declared",
    )


def exposure_outcome_distribution_figure_owns_step(step: AnalysisStep) -> bool:
    """Own a rendering-only step whose single typed input is this product."""

    return exposure_outcome_distribution_figure_declaration_verdict(step).claimed


def exposure_outcome_distribution_figure_code(
    step: AnalysisStep,
    *,
    display_labels: Mapping[str, str] | None = None,
) -> str:
    if not exposure_outcome_distribution_figure_owns_step(step):
        raise ValueError(
            "The step is not owned by the exposure-outcome distribution renderer"
        )
    product = _figure_product(step.expected_outputs[0])
    resolved = planner_binary_level_labels(display_labels)
    labels = (resolved[1], resolved[2]) if resolved is not None else None
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.exposure_outcome_distribution_render import (
            run_exposure_outcome_distribution_figure,
        )

        run_exposure_outcome_distribution_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
            level_labels={labels!r},
        )
        """
    ).strip()


def _load_binding(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
) -> BoundTypedInput:
    """Read exactly the one bound table through the shared binding owner.

    There is no separate loader here on purpose. Every check this renderer
    needs -- one manifest for this step, one input and no other, a capsule that
    agrees with its own identity record, a contained path, a digest verified
    before and after the read, and the exact product schema -- is the same
    question every other typed consumer asks, and a second implementation of it
    would only guarantee that the two drift.
    """

    return load_typed_input(
        input_key=EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT,
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        expected_columns=EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS,
        exclusive=True,
        require_consumption_contract=True,
        minimum_row_count=3,
    )


def _close(left: Any, right: Any) -> bool:
    """Whether two published quantities agree, both possibly absent."""

    first, second = _finite(left), _finite(right)
    if first is None or second is None:
        return first is None and second is None
    return abs(first - second) <= 1e-6


def _absent(value: Any) -> bool:
    """Whether a CSV cell represents a deliberately undeclared field."""

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _constant_column(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].astype("object")
    distinct = {repr(value) for value in values}
    if len(distinct) != 1:
        raise ValueError(
            f"distribution table rows disagree on {column!r}; one repeated "
            "contrast cannot carry several results"
        )
    return values.iloc[0]


def _declared_design(frame: pd.DataFrame) -> dict[str, Any]:
    """The one design every row must agree on, or refuse.

    The design columns are constant by construction, so a table whose rows
    disagree about which outcome value was the event, or at what confidence its
    intervals were built, is not one table -- and picking either answer would
    be this renderer deciding something it does not own.
    """

    design: dict[str, Any] = {}
    for column in EXPOSURE_OUTCOME_DISTRIBUTION_DESIGN_COLUMNS:
        values = frame[column].astype("object")
        distinct = {repr(value) for value in values}
        if len(distinct) != 1:
            raise ValueError(
                f"distribution table rows disagree on {column!r}; the design "
                "that produced the numbers must be one declaration"
            )
        design[column] = values.iloc[0]
    return design


def _declared_risk_difference(
    frame: pd.DataFrame,
    *,
    levels: pd.DataFrame,
    design: Mapping[str, Any],
    confidence_level: float | None,
) -> dict[str, Any] | None:
    """Validate the optional descriptive contrast without refitting a model.

    The renderer has only the digest-verified result table, not the patient
    grouping values.  It can therefore re-derive the point estimate, analysis
    count and Wald arithmetic, and verify that covariance metadata is closed;
    the producer remains the owner that fits HC1/cluster-robust covariance.
    """

    result = {
        column: _constant_column(frame, column)
        for column in EXPOSURE_OUTCOME_DISTRIBUTION_CONTRAST_COLUMNS
    }
    declaration_fields = (
        "risk_difference_reference_index",
        "risk_difference_comparison_index",
        "risk_difference_effect_measure",
        "risk_difference_interval_method",
    )
    declaration = {key: design[key] for key in declaration_fields}
    declared = any(not _absent(value) for value in declaration.values())
    if not declared:
        if any(not _absent(value) for value in result.values()):
            raise ValueError(
                "risk-difference numbers are present without a declared contrast"
            )
        dependence_fields = (
            "dependence_variance_estimator",
            "dependence_cluster_unit",
            "dependence_group_source",
            "dependence_group_derivation",
            "dependence_delimiter",
        )
        if any(not _absent(design[key]) for key in dependence_fields):
            raise ValueError(
                "dependence metadata is present without a declared risk difference"
            )
        return None
    if any(_absent(value) for value in declaration.values()) or any(
        _absent(value)
        for key, value in result.items()
        if key != "risk_difference_cluster_count"
    ):
        raise ValueError("risk-difference declaration or result is incomplete")

    indices: list[int] = []
    for key in (
        "risk_difference_reference_index",
        "risk_difference_comparison_index",
    ):
        parsed = _finite(design[key])
        if parsed is None or not parsed.is_integer():
            raise ValueError(f"{key} is not an integer level pointer")
        index = int(parsed)
        if not 0 <= index < len(levels):
            raise ValueError(f"{key} is outside the declared exposure levels")
        indices.append(index)
    reference_index, comparison_index = indices
    if reference_index == comparison_index:
        raise ValueError("risk-difference reference and comparison are identical")
    if str(design["risk_difference_effect_measure"]) != "risk_difference":
        raise ValueError("the contrast does not declare risk_difference")
    if str(design["risk_difference_interval_method"]) != "linear_probability_wald":
        raise ValueError("the risk-difference interval method is unsupported")
    if confidence_level is None:
        raise ValueError("a risk-difference contrast requires confidence authority")

    reference = levels.iloc[reference_index]
    comparison = levels.iloc[comparison_index]
    expected_n = int(reference["outcome_denominator"]) + int(
        comparison["outcome_denominator"]
    )
    n = _finite(result["risk_difference_n"])
    if n is None or not n.is_integer() or int(n) != expected_n:
        raise ValueError(
            "risk-difference analysis count is not its two group denominators"
        )
    estimate = _finite(result["risk_difference_pct"])
    standard_error = _finite(result["risk_difference_standard_error_pct"])
    low = _finite(result["risk_difference_ci_low_pct"])
    high = _finite(result["risk_difference_ci_high_pct"])
    if any(value is None for value in (estimate, standard_error, low, high)):
        raise ValueError("risk-difference estimate or uncertainty is invalid")
    assert estimate is not None
    assert standard_error is not None
    assert low is not None
    assert high is not None
    if standard_error <= 0:
        raise ValueError("risk-difference estimate or uncertainty is invalid")
    expected_estimate = round(
        float(comparison["outcome_rate_pct"]) - float(reference["outcome_rate_pct"]),
        6,
    )
    if not _close(estimate, expected_estimate):
        raise ValueError(
            "risk difference is not comparison absolute risk minus reference"
        )
    critical = statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence_level) / 2.0)
    expected_low = round(expected_estimate - critical * float(standard_error), 6)
    expected_high = round(expected_estimate + critical * float(standard_error), 6)
    if abs(float(low) - expected_low) > 2e-6 or abs(float(high) - expected_high) > 2e-6:
        raise ValueError(
            "risk-difference interval is not its declared Wald-normal arithmetic"
        )

    covariance = str(result["risk_difference_covariance"])
    cluster_count = result["risk_difference_cluster_count"]
    dependence = str(design["dependence_variance_estimator"])
    if covariance == "hc1":
        if not _absent(design["dependence_variance_estimator"]) or not _absent(
            cluster_count
        ):
            raise ValueError("HC1 contrast carries contradictory dependence metadata")
    elif covariance == "cluster_robust":
        count = _finite(cluster_count)
        stratum_cluster_counts = [
            _finite(reference["outcome_interval_cluster_count"]),
            _finite(comparison["outcome_interval_cluster_count"]),
        ]
        if (
            dependence != "cluster_robust"
            or str(design["dependence_cluster_unit"]) != "patient"
            or _absent(design["dependence_group_source"])
            or str(design["dependence_group_derivation"])
            not in {"identity", "prefix_before_delimiter"}
            or count is None
            or not count.is_integer()
            or count < 2
            or count > int(n)
            or any(value is None for value in stratum_cluster_counts)
            or count
            < max(
                value for value in stratum_cluster_counts if value is not None
            )
        ):
            raise ValueError(
                "cluster-robust contrast lacks a closed patient grouping design"
            )
        if str(
            design["dependence_group_derivation"]
        ) == "prefix_before_delimiter" and _absent(design["dependence_delimiter"]):
            raise ValueError(
                "prefix-derived patient grouping lacks its declared delimiter"
            )
    else:
        raise ValueError("risk-difference covariance is unsupported")

    return {
        **result,
        "reference_index": reference_index,
        "comparison_index": comparison_index,
        "reference_level": reference["exposure_level"],
        "comparison_level": comparison["exposure_level"],
        "reference_events": int(reference["outcome_events"]),
        "reference_n": int(reference["outcome_denominator"]),
        "reference_risk_pct": float(reference["outcome_rate_pct"]),
        "comparison_events": int(comparison["outcome_events"]),
        "comparison_n": int(comparison["outcome_denominator"]),
        "comparison_risk_pct": float(comparison["outcome_rate_pct"]),
        "confidence_level": confidence_level,
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
    }


def _validate_marginal_interval(
    row: pd.Series,
    *,
    numerator: int,
    denominator: int,
    estimate_key: str,
    low_key: str,
    high_key: str,
    standard_error_key: str,
    covariance_key: str,
    cluster_count_key: str,
    interval_method: str,
    confidence_level: float | None,
    label: str,
    structural_total: bool = False,
) -> None:
    expected_estimate = percentage(numerator, denominator)
    if expected_estimate is None:
        raise ValueError(f"a {label} has no finite denominator")
    if not _close(row[estimate_key], expected_estimate):
        if label == "outcome absolute risk":
            raise ValueError("an outcome rate is not its own events over denominator")
        raise ValueError("an exposure percentage is not its own counts")
    if interval_method == "none_counts_only":
        if (
            not _absent(row[standard_error_key])
            or not _absent(row[low_key])
            or not _absent(row[high_key])
            or str(row[covariance_key]) != COUNTS_ONLY_COVARIANCE
            or not _absent(row[cluster_count_key])
        ):
            raise ValueError(f"a {label} counts-only result carries uncertainty")
        return
    if structural_total:
        if (
            denominator <= 0
            or numerator != denominator
            or not _absent(row[standard_error_key])
            or not _absent(row[low_key])
            or not _absent(row[high_key])
            or str(row[covariance_key]) != STRUCTURAL_TOTAL_COVARIANCE
            or not _absent(row[cluster_count_key])
        ):
            raise ValueError(
                f"a {label} structural total carries inferential uncertainty"
            )
        return
    if interval_method == "wilson":
        if confidence_level is None:
            raise ValueError(f"a {label} Wilson interval lacks confidence authority")
        expected_low, expected_high = wilson_interval(
            numerator,
            denominator,
            confidence_level=confidence_level,
        )
        proportion = numerator / denominator
        expected_standard_error = round(
            100.0 * math.sqrt(proportion * (1.0 - proportion) / denominator),
            6,
        )
        if not _close(row[low_key], expected_low) or not _close(
            row[high_key], expected_high
        ):
            raise ValueError(
                f"a {label} interval is not the declared method (Wilson) at "
                "the declared confidence"
            )
        if (
            not _close(row[standard_error_key], expected_standard_error)
            or str(row[covariance_key]) != "binomial_independent"
            or not _absent(row[cluster_count_key])
        ):
            raise ValueError(
                f"a {label} Wilson interval carries contradictory independent "
                "uncertainty"
            )
        return
    standard_error = _finite(row[standard_error_key])
    cluster_count = _finite(row[cluster_count_key])
    if (
        interval_method != "patient_cluster_robust_wald"
        or standard_error is None
        or standard_error <= 0.0
        or str(row[covariance_key]) != "cluster_robust"
        or cluster_count is None
        or not cluster_count.is_integer()
        or cluster_count < 2
        or cluster_count > denominator
    ):
        raise ValueError(
            f"a {label} interval lacks its patient-cluster covariance receipt"
        )
    if confidence_level is None:
        raise ValueError(f"a {label} clustered interval lacks confidence authority")
    assert standard_error is not None
    critical = statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence_level) / 2.0)
    expected_low = round(
        max(0.0, float(expected_estimate) - critical * standard_error), 6
    )
    expected_high = round(
        min(100.0, float(expected_estimate) + critical * standard_error), 6
    )
    if (
        abs(float(row[low_key]) - expected_low) > 2e-6
        or abs(float(row[high_key]) - expected_high) > 2e-6
    ):
        raise ValueError(
            f"a {label} interval is not its patient-cluster Wald arithmetic"
        )


def _validate(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], dict[str, Any] | None]:
    """Re-derive every published quantity before drawing it.

    Not a plausibility check: each percentage is recomputed from the counts on
    its own row, each interval is rebuilt by the method and confidence level
    the table declares, and the strata are required to sum to the totals. A
    rate that merely lands inside its interval is not evidence that it is the
    right rate.
    """

    design = _declared_design(frame)
    interval_method = str(design["interval_method"])
    if interval_method not in {
        "wilson",
        "patient_cluster_robust_wald",
        "none_counts_only",
    }:
        raise ValueError(
            f"distribution table declares interval_method={interval_method!r}, "
            "which this renderer cannot re-derive"
        )
    counts_only = interval_method == "none_counts_only"
    if counts_only:
        if (
            str(design["independent_interval_method"]) != "none_counts_only"
            or not _absent(design["repeated_unit_interval_method"])
        ):
            raise ValueError(
                "counts-only distribution table carries an interval projection"
            )
    elif (
        str(design["independent_interval_method"]) != "wilson"
        or str(design["repeated_unit_interval_method"])
        != "patient_cluster_robust_wald"
    ):
        raise ValueError(
            "distribution table does not carry the closed marginal-interval projection"
        )
    has_dependence = not _absent(design["dependence_variance_estimator"])
    expected_effective_method = (
        str(design["repeated_unit_interval_method"])
        if has_dependence
        else str(design["independent_interval_method"])
    )
    if interval_method != expected_effective_method:
        raise ValueError(
            "effective marginal interval method contradicts the typed dependence design"
        )
    confidence_level = _finite(design["confidence_level"])
    if counts_only and confidence_level is not None:
        raise ValueError("counts-only distribution table carries confidence authority")
    if not counts_only and (
        confidence_level is None or not (0.5 < confidence_level < 1.0)
    ):
        raise ValueError("distribution table declares an unusable confidence level")
    denominator_policy = str(design["denominator_policy"])
    if denominator_policy not in {"all_declared_rows", "observed_outcome_rows"}:
        raise ValueError("distribution table declares an unknown denominator policy")

    # Selecting the two known roles would silently drop a third: a row nobody
    # recognises would then be excluded from every sum below and still be drawn.
    unknown_roles = set(frame["row_role"].astype(str)) - {_LEVEL_ROLE, _OVERALL_ROLE}
    if unknown_roles:
        raise ValueError(
            f"distribution table carries unknown row roles: {sorted(unknown_roles)}"
        )
    levels = frame[frame["row_role"] == _LEVEL_ROLE]
    overall = frame[frame["row_role"] == _OVERALL_ROLE]
    if len(overall) != 1:
        raise ValueError("distribution table needs exactly one overall row")
    if len(levels) < 2:
        raise ValueError("distribution table needs at least two exposure levels")
    if levels["exposure_level"].astype(str).duplicated().any():
        raise ValueError("an exposure level appears more than once")
    try:
        declared_levels = json.loads(str(design["exposure_levels_declared"]))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "distribution table declares unreadable exposure levels"
        ) from exc
    if not isinstance(declared_levels, list) or len(declared_levels) != len(levels):
        raise ValueError(
            "distribution table reports a different number of exposure levels "
            "from the number its own declaration closes over"
        )
    indices = [_finite(value) for value in levels["exposure_level_index"]]
    if (
        any(value is None or not value.is_integer() for value in indices)
        or {int(value) for value in indices if value is not None}
        != set(range(len(declared_levels)))
    ):
        raise ValueError(
            "distribution table level rows do not carry the declared typed-level indices"
        )
    levels = levels.assign(
        __level_index__=[int(value) for value in indices if value is not None]
    ).sort_values("__level_index__")
    # The index is the typed authority. CSV cannot preserve the distinction
    # between every scalar representation (for example 1 and 1.0), so figures
    # display the exact JSON declaration rather than reinterpreting the cell.
    levels = levels.copy()
    levels["exposure_level"] = pd.Series(
        declared_levels,
        index=levels.index,
        dtype="object",
    )
    levels = levels.drop(columns=["__level_index__"])
    total = overall.iloc[0]
    if int(levels["n_rows"].sum()) != int(total["n_rows"]):
        raise ValueError("exposure levels do not partition the reported cohort")
    for column in ("outcome_events", "outcome_observed_n", "outcome_missing_n"):
        if int(levels[column].sum()) != int(total[column]):
            raise ValueError(f"level {column} does not sum to the overall {column}")

    for _, row in frame.iterrows():
        if int(row["outcome_observed_n"]) + int(row["outcome_missing_n"]) != int(
            row["n_rows"]
        ):
            raise ValueError("observed plus missing does not equal the row count")
        if int(row["outcome_events"]) > int(row["outcome_denominator"]):
            raise ValueError("more events than the denominator they are taken over")
        if int(row["exposure_denominator"]) != int(total["n_rows"]):
            raise ValueError(
                "an exposure denominator is not the cohort the table reports"
            )
        expected_denominator = (
            int(row["n_rows"])
            if denominator_policy == "all_declared_rows"
            else int(row["outcome_observed_n"])
        )
        if int(row["outcome_denominator"]) != expected_denominator:
            raise ValueError(
                "an outcome denominator does not follow the declared "
                f"denominator_policy={denominator_policy!r}"
            )
        _validate_marginal_interval(
            row,
            numerator=int(row["n_rows"]),
            denominator=int(row["exposure_denominator"]),
            estimate_key="exposure_pct",
            low_key="exposure_ci_low_pct",
            high_key="exposure_ci_high_pct",
            standard_error_key="exposure_standard_error_pct",
            covariance_key="exposure_interval_covariance",
            cluster_count_key="exposure_interval_cluster_count",
            interval_method=interval_method,
            confidence_level=confidence_level,
            label="exposure prevalence",
            structural_total=str(row["row_role"]) == _OVERALL_ROLE,
        )
        _validate_marginal_interval(
            row,
            numerator=int(row["outcome_events"]),
            denominator=int(row["outcome_denominator"]),
            estimate_key="outcome_rate_pct",
            low_key="ci_low_pct",
            high_key="ci_high_pct",
            standard_error_key="outcome_standard_error_pct",
            covariance_key="outcome_interval_covariance",
            cluster_count_key="outcome_interval_cluster_count",
            interval_method=interval_method,
            confidence_level=confidence_level,
            label="outcome absolute risk",
        )
        # Deliberately NOT checked here: that the rate and interval are finite,
        # that ci_low <= ci_high, and that both lie in 0-100. Each was written,
        # probed, and removed as unreachable -- the exact re-derivations above
        # already pin all three quantities to values recomputed from the counts,
        # so a non-finite or out-of-range endpoint fails the equality check
        # several lines earlier with a message that names the real disagreement.
        # A range check downstream of an equality check cannot fire; adding one
        # back would only look like more safety. If the equality checks are ever
        # relaxed, these become live again and must return with them.
        rate = _finite(row["outcome_rate_pct"])
        low = _finite(row["ci_low_pct"])
        high = _finite(row["ci_high_pct"])
        if rate is not None and low is not None and high is not None:
            if not (low - 1e-6 <= rate <= high + 1e-6):
                raise ValueError("a reported rate falls outside its own interval")
    contrast = _declared_risk_difference(
        frame,
        levels=levels,
        design=design,
        confidence_level=confidence_level,
    )
    return levels, total, design, contrast


def _labels(levels: pd.DataFrame, level_labels: tuple[str, str] | None) -> list[str]:
    """Label rows from the Planner's display labels when they are binary.

    Falls back to the level value itself: an unlabelled category is still an
    honest category, whereas inventing a clinical name would not be.
    """

    values = list(levels["exposure_level"])
    if level_labels is not None and len(values) == 2:
        return [str(level_labels[0]), str(level_labels[1])]
    return [str(value) for value in values]


def run_exposure_outcome_distribution_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
    level_labels: tuple[str, str] | None = None,
) -> Mapping[str, Any]:
    """Render the two-panel distribution figure from its one bound table."""

    if not _is_safe_figure_product_id(figure_product):
        raise ValueError("unsafe or malformed figure product id")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bound = _load_binding(
        run_dir=Path(run_dir), resolved_inputs=resolved_inputs, step_id=step_id
    )
    frame, binding, source_name = bound.frame, bound.binding, bound.path.name
    levels, total, design, contrast = _validate(frame)
    counts_only = str(design["interval_method"]) == "none_counts_only"
    panel_templates = (
        EXPOSURE_OUTCOME_DISTRIBUTION_COUNTS_ONLY_FIGURE_PANELS
        if counts_only
        else EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_PANELS
    )

    full_source = out_dir / f"{figure_product}_input_source_data.csv"
    prevalence_source = out_dir / f"{figure_product}_prevalence_source_data.csv"
    outcome_source = out_dir / f"{figure_product}_outcome_source_data.csv"
    contrast_source = out_dir / f"{figure_product}_risk_difference_source_data.csv"
    frame.to_csv(full_source, index=False)
    levels[
        [
            "exposure_level",
            "exposure_level_index",
            "n_rows",
            "exposure_denominator",
            "exposure_pct",
            "exposure_ci_low_pct",
            "exposure_ci_high_pct",
            "exposure_standard_error_pct",
            "exposure_interval_covariance",
            "exposure_interval_cluster_count",
        ]
    ].to_csv(prevalence_source, index=False)
    levels[
        [
            "exposure_level",
            "exposure_level_index",
            "outcome_events",
            "outcome_denominator",
            "outcome_observed_n",
            "outcome_missing_n",
            "outcome_rate_pct",
            "ci_low_pct",
            "ci_high_pct",
            "outcome_standard_error_pct",
            "outcome_interval_covariance",
            "outcome_interval_cluster_count",
        ]
    ].to_csv(outcome_source, index=False)
    if contrast is not None:
        # Keep the figure source as an exact, row-addressable projection of the
        # parent table.  ``contrast`` also carries reader conveniences derived
        # across the two exposure rows (their labels, risks, and denominators),
        # so serialising that mapping creates a new synthetic row that no parent
        # row can authenticate.  The parent already stores the prespecified RD
        # and its uncertainty on its unique ``overall`` row; publish only those
        # same-name fields plus the row identity used by the source-data gate.
        contrast_columns = [
            "risk_difference_n",
            "risk_difference_pct",
            "risk_difference_standard_error_pct",
            "risk_difference_ci_low_pct",
            "risk_difference_ci_high_pct",
            "risk_difference_covariance",
            "risk_difference_cluster_count",
            "risk_difference_reference_index",
            "risk_difference_comparison_index",
            "risk_difference_effect_measure",
            "risk_difference_interval_method",
        ]
        parent_contrast = frame.loc[[total.name], contrast_columns].copy()
        parent_contrast.insert(0, "source_row_index", [int(total.name)])
        parent_contrast.to_csv(contrast_source, index=False)

    import matplotlib.pyplot as plt

    palette = apply_publication_style()
    labels = _labels(levels, level_labels)
    positions = list(range(len(levels)))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.4))

    prevalence = levels["exposure_pct"].astype(float)
    prevalence_low = (
        levels["exposure_ci_low_pct"].astype(float) if not counts_only else prevalence
    )
    prevalence_high = (
        levels["exposure_ci_high_pct"].astype(float) if not counts_only else prevalence
    )
    ax_a.barh(
        positions,
        prevalence,
        xerr=(
            [prevalence - prevalence_low, prevalence_high - prevalence]
            if not counts_only
            else None
        ),
        color=palette["blue"],
        error_kw={"ecolor": palette["neutral"], "capsize": 2.0, "elinewidth": 1.0},
        height=0.55,
    )
    ax_a.set_yticks(positions)
    ax_a.set_yticklabels(labels)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Share of the analysed cohort (%)")
    ax_a.set_title("Exposure distribution", loc="left", pad=4)
    ax_a.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for position, pct, n_rows, denominator in zip(
        positions,
        levels["exposure_pct"],
        levels["n_rows"],
        levels["exposure_denominator"],
    ):
        ax_a.text(
            float(pct) + 1.0,
            position,
            f"{float(pct):.1f}%  {int(n_rows):,}/{int(denominator):,}",
            va="center",
            fontsize=6.1,
        )
    add_panel_label(ax_a, "A", x=-0.14, y=1.04)

    rate = levels["outcome_rate_pct"].astype(float)
    low = levels["ci_low_pct"].astype(float) if not counts_only else rate
    high = levels["ci_high_pct"].astype(float) if not counts_only else rate
    if counts_only:
        ax_b.plot(rate, positions, "o", color=palette["blue"], markersize=4.2)
    else:
        ax_b.errorbar(
            rate,
            positions,
            xerr=[rate - low, high - rate],
            fmt="o",
            color=palette["blue"],
            ecolor=palette["neutral"],
            elinewidth=1.0,
            capsize=2.0,
            markersize=4.2,
        )
    ax_b.set_yticks(positions)
    ax_b.set_yticklabels(labels)
    ax_b.invert_yaxis()
    lower = min(0.0, float(low.min()) * 1.15)
    upper = max(5.0, float(high.max()) * 1.35)
    ax_b.set_xlim(lower, upper)
    ax_b.set_xlabel("Outcome rate (%)")
    ax_b.set_title("Outcome rate by exposure", loc="left", pad=4)
    ax_b.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for position, estimate, events, denominator, missing in zip(
        positions,
        rate,
        levels["outcome_events"],
        levels["outcome_denominator"],
        levels["outcome_missing_n"],
    ):
        suffix = f"  ({int(missing):,} unobserved)" if int(missing) else ""
        ax_b.text(
            min(
                float(estimate) + (upper - lower) * 0.025,
                upper - (upper - lower) * 0.02,
            ),
            position,
            f"{float(estimate):.1f}%  {int(events):,}/{int(denominator):,}{suffix}",
            va="center",
            ha="left" if estimate < lower + (upper - lower) * 0.86 else "right",
            fontsize=6.1,
        )
    add_panel_label(ax_b, "B", x=-0.14, y=1.04)
    if contrast is not None:
        coverage = 100.0 * float(contrast["confidence_level"])
        contrast_annotation = (
            f"Risk difference (comparison − reference): "
            f"{float(contrast['risk_difference_pct']):.1f} pp "
            f"({coverage:.0f}% CI "
            f"{float(contrast['risk_difference_ci_low_pct']):.1f} to "
            f"{float(contrast['risk_difference_ci_high_pct']):.1f}); "
            f"{contrast['risk_difference_covariance']}"
        )
        ax_b.text(
            0.0,
            -0.23,
            textwrap.fill(
                contrast_annotation,
                width=54,
                break_long_words=False,
                break_on_hyphens=False,
            ),
            transform=ax_b.transAxes,
            ha="left",
            va="top",
            fontsize=6.0,
            linespacing=1.25,
            color=palette["neutral"],
        )
    fig.subplots_adjust(
        left=0.16,
        right=0.98,
        bottom=0.29 if contrast is not None else 0.20,
        top=0.84,
        wspace=0.48,
    )

    source_data = [full_source.name, prevalence_source.name, outcome_source.name]
    if contrast is not None:
        source_data.append(contrast_source.name)
    outcome_evidence = [outcome_source.name]
    if contrast is not None:
        outcome_evidence.append(contrast_source.name)
    contrast_note = ""
    if contrast is not None:
        contrast_note = (
            " The prespecified unadjusted risk difference is comparison minus "
            "reference with a Wald-normal interval using "
            f"{contrast['risk_difference_covariance']} covariance. It is a "
            "descriptive contrast and does not authorize association or causal "
            "interpretation."
        )

    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The declared exposure levels and their outcome rates are rendered "
            "from one digest-verified, self-contained parent table."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=86.0,
        panels=[
            {
                "panel_id": panel_templates[0].panel_id,
                "title": "Exposure distribution",
                "role": panel_templates[0].article_role,
                "claim": (
                    "The parent's declared exposure levels partition the analysed "
                    "denominator, which each row carries with it."
                ),
                "evidence_ids": [prevalence_source.name],
                "metadata": {
                    "article_role": (
                        panel_templates[0].article_role
                    ),
                    "chart_type": (
                        panel_templates[0].chart_type
                    ),
                    "source_products": list(
                        panel_templates[0].source_products
                    ),
                    "source_data": [prevalence_source.name],
                },
            },
            {
                "panel_id": panel_templates[1].panel_id,
                "title": "Outcome rate by exposure",
                "role": panel_templates[1].article_role,
                "claim": (
                    "Events, the denominator they are taken over, and the "
                    + ("observed proportion" if counts_only else "interval")
                    + " are shown for every declared level."
                ),
                "evidence_ids": outcome_evidence,
                "metadata": {
                    "article_role": (
                        panel_templates[1].article_role
                    ),
                    "chart_type": (
                        panel_templates[1].chart_type
                    ),
                    "source_products": list(
                        panel_templates[1].source_products
                    ),
                    "source_data": outcome_evidence,
                },
            },
        ],
        source_data=source_data,
        statistics_note=(
            (
                "Counts, denominators, and observed percentages are reproduced "
                "from the bound parent table; no uncertainty is computed. "
                if counts_only
                else "Percentages and intervals are reproduced from the bound parent table. "
            )
            + "Outcome rates are taken over "
            f"{design['denominator_policy']} with missing outcomes handled as "
            f"{design['missing_outcome_policy']}. "
            + (
                ""
                if counts_only
                else f"Intervals are {design['interval_method']} at {float(design['confidence_level']):.3g} coverage. "
            )
            + "The renderer "
            "treats the overall exposure share as a structural 100% total, "
            "not an estimated proportion, so it intentionally carries no "
            "inferential interval. The renderer "
            "re-derives each published quantity from the counts beside it and "
            "introduces no cohort, exposure, outcome, denominator, or "
            "missing-data decision of its own." + contrast_note
        ),
    )
    # The contract is written by the exporter, not here: it decides the export
    # formats from the contract itself, so serialising a second copy alongside
    # would be a second source of truth for what was exported.
    outputs = save_publication_figure(
        fig,
        out_dir / figure_product,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
    )
    plt.close(fig)
    contract_path = outputs["contract"]
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]

    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "analysis_family": "descriptive",
        "rendering_only": True,
        "deterministic_standard_analysis": "exposure_outcome_distribution_figure",
        "interpretation_class": "exposure_outcome_distribution_figure",
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        "source_input": EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT,
        "source_table": source_name,
        "source_sha256": binding.get("sha256"),
        "source_evidence_id": binding.get("evidence_id"),
        "source_rows_consumed": int(len(frame)),
        "cohort_n": int(total["n_rows"]),
        # Echo only non-result design coordinates from the bound table.  The RD
        # numbers remain in the digest-bound parent and its source-data
        # projection; copying them into this rendering-only step summary makes
        # the renderer look like a second effect-estimation owner.
        "declared_design": {
            key: (
                None
                if _absent(value)
                else float(value)
                if key == "confidence_level"
                else int(float(value))
                if key
                in {
                    "outcome_positive_index",
                    "risk_difference_reference_index",
                    "risk_difference_comparison_index",
                }
                else str(value)
            )
            for key, value in design.items()
            if not str(key).startswith("risk_difference_")
        },
        "figure_path": f"{figure_product}.png",
        "figure_contract": contract_path.name,
        "contract_files": [contract_path.name],
        "figure_files": figure_files,
        "source_data_files": source_data,
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
        "adjusted_effect": None,
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary

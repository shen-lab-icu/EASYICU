"""Deterministic executor for one closed exposure-by-outcome distribution.

Every scientific choice is read from the Planner's
``exposure_outcome_distribution_spec``: which column is the exposure, which
levels of it are reported, which column is the outcome, which observed value
counts as the event, whose rows form each denominator, and how the interval is
built. This module decides none of them. In particular it never infers the
exposure from column names, from the order of ``inputs``, or from the intent
text -- a study's exposure and outcome are its design, and an engine that
picks them has taken the investigator's decision.

The product is deliberately **self-contained**: each row carries its own
denominator, its missing count, its event count and its rate with an interval,
*and* the design that produced them -- the exposure and outcome columns, the
closed level sets, the event value, the denominator, missing-data and matching
policies, the interval method and its confidence level.  When requested, it
also carries one prespecified unadjusted absolute-risk difference and its
uncertainty.  Repeated rows use only the host-bound patient grouping contract
for cluster-robust covariance; neither a column name nor prose can create that
authority. A renderer can
therefore draw and re-derive the whole figure from this one table, with no
second lookup into a cohort summary and no out-of-band knowledge of the spec.
That is what makes a figure step's input contract closable before its parent
has run.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...authority.declared_levels import execution_distribution_spec
from ...contracts.dependence import (
    PatientGroupResolutionError,
    resolve_patient_groups,
)
from ...contracts.claim_ceiling import (
    BOUND_TYPED_COHORT_ANALYSIS_SET,
    EXPOSURE_OBSERVED_ANALYSIS_SET,
)
from ...contracts.ownership_verdict import OwnershipVerdict
from ...schema import AnalysisStep, ExposureOutcomeDistributionSpec, _typed_level_key
from .plausibility_receipt import host_plausibility_receipt_injected
from .typed_input_binding import (
    load_typed_cohort,
    run_dir_from_env,
    sole_typed_cohort_input,
)

__all__ = [
    "EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS",
    "EXPOSURE_OUTCOME_DISTRIBUTION_CONTRAST_COLUMNS",
    "EXPOSURE_OUTCOME_DISTRIBUTION_DESIGN_COLUMNS",
    "EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT",
    "STRUCTURAL_TOTAL_COVARIANCE",
    "exposure_outcome_distribution_declaration_verdict",
    "exposure_outcome_distribution_executor_code",
    "exposure_outcome_distribution_executor_owns_step",
    "percentage",
    "run_exposure_outcome_distribution_from_env",
    "wilson_interval",
]

EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT = "table:exposure_outcome_distribution"

#: The design columns. Every row repeats them, which is the price of a product
#: that a downstream consumer can check without being told anything else.
EXPOSURE_OUTCOME_DISTRIBUTION_DESIGN_COLUMNS = (
    "exposure_column",
    "exposure_levels_declared",
    "outcome_column",
    "outcome_levels_declared",
    # A *pointer* into outcome_levels_declared, not the value itself. A CSV
    # cell cannot carry a typed scalar: ``1`` and ``"1"`` are written
    # identically and both read back as a number, which would silently undo the
    # one distinction the level policy exists to preserve. The levels array
    # survives the round trip intact, so the event is identified by position
    # within it.
    "outcome_positive_index",
    "level_match_policy",
    "denominator_policy",
    "missing_exposure_policy",
    "missing_outcome_policy",
    "independent_interval_method",
    "repeated_unit_interval_method",
    "interval_method",
    "confidence_level",
    "risk_difference_reference_index",
    "risk_difference_comparison_index",
    "risk_difference_effect_measure",
    "risk_difference_interval_method",
    "dependence_variance_estimator",
    "dependence_cluster_unit",
    "dependence_group_source",
    "dependence_group_derivation",
    "dependence_delimiter",
)

#: A single prespecified contrast is repeated on every row.  Repetition keeps
#: the product self-contained without introducing a synthetic "level" row
#: that would break the partition represented by the level rows.
EXPOSURE_OUTCOME_DISTRIBUTION_CONTRAST_COLUMNS = (
    "risk_difference_n",
    "risk_difference_pct",
    "risk_difference_standard_error_pct",
    "risk_difference_ci_low_pct",
    "risk_difference_ci_high_pct",
    "risk_difference_covariance",
    "risk_difference_cluster_count",
)

#: The closed product schema. A renderer binds on this, never on the table's
#: name: two studies may call the product different things and still be the
#: same shape, and the same name may be given to a different shape.
EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS = (
    (
        "row_role",
        "exposure_level_index",
        "exposure_level",
        "n_rows",
        "exposure_denominator",
        "exposure_pct",
        "exposure_ci_low_pct",
        "exposure_ci_high_pct",
        "exposure_standard_error_pct",
        "exposure_interval_covariance",
        "exposure_interval_cluster_count",
        "outcome_observed_n",
        "outcome_missing_n",
        "outcome_events",
        "outcome_denominator",
        "outcome_rate_pct",
        "ci_low_pct",
        "ci_high_pct",
        "outcome_standard_error_pct",
        "outcome_interval_covariance",
        "outcome_interval_cluster_count",
        # How many rows the exposure could not place. Repeated on every row, the
        # way Table 1 carries ``group_missing_excluded_n``: a denominator that
        # silently shrank is a denominator a reader cannot check.
        "missing_exposure_excluded_n",
    )
    + EXPOSURE_OUTCOME_DISTRIBUTION_CONTRAST_COLUMNS
    + EXPOSURE_OUTCOME_DISTRIBUTION_DESIGN_COLUMNS
)

_OVERALL_ROLE = "overall"
_LEVEL_ROLE = "exposure_level"
STRUCTURAL_TOTAL_COVARIANCE = "structural_identity_no_interval"


def _typed_cohort_input(step: AnalysisStep) -> str:
    """Return the single typed cohort input, or ``""`` when there is not one.

    An empty answer is disqualifying rather than a fallback. A step whose
    cohort arrives by bare ``COHORT_PARQUET`` has no digest, no product
    contract and no named producer for the rows it counted, so the table it
    would emit could not be bound to the plan that asked for it -- and calling
    that "deterministically owned" would be a claim the run cannot support.

    Only that last policy is this owner's own: *which* keys name the closed
    cohort product is one published vocabulary, so it is read from there rather
    than spelled out again here.
    """

    return sole_typed_cohort_input(step) or ""


def exposure_outcome_distribution_executor_owns_step(step: AnalysisStep) -> bool:
    """Own only a step whose distribution design is completely declared."""

    spec = step.exposure_outcome_distribution_spec
    if spec is None:
        return False
    role = str(step.planned_analysis_role or "").strip().casefold()
    primary_is_descriptive = bool(
        role == "primary"
        and str(step.method or "").strip().casefold() == "descriptive"
        and step.descriptive_claim is not None
        and step.descriptive_claim.claim_ceiling == "descriptive_only"
    )
    return bool(
        str(step.method or "").strip().casefold() in {"descriptive", "distribution"}
        and (role == "auxiliary" or primary_is_descriptive)
        and list(step.expected_outputs or []) == [EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT]
        and _typed_cohort_input(step)
        and not step.model_requirements
        and step.family_primary_result_requirement is None
        and step.scientific_capability is None
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
    )


#: The analysis kind this owner reports, in selection and in its verdict.
EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND = "exposure_outcome_distribution"


def exposure_outcome_distribution_declaration_verdict(
    step: AnalysisStep,
) -> OwnershipVerdict:
    """Report what a step must declare for this owner to compute it.

    Measured over every recorded run: 33 steps promise this owner's product, 29
    declare the spec, 28 are claimed, and 29 of 33 pass (88 %).  The SAME
    science planned under the Planner's own label --
    ``table:absolute_risk_context``, 28 steps -- declares the spec 0 times, is
    claimed 0 times, and was never asked.  Those 28 pass 82 % of the time, so
    the Coder writes a table; a DIFFERENT table every run.  25 of the 26
    recorded files have distinct headers, and every figure over them dies
    (14 recorded, 0 ok).  Declining silently is what let an 82 %-passing step
    emit an artifact with no contract, and the whole cost land on its consumer.

    Both gaps are asked together because the spec alone cannot close it: this
    executor writes its own filename and registers its own key, so a step
    promising a different product would still go unclaimed with a perfect spec,
    and a replan spent on half the answer is a replan wasted.  Neither is a
    scientific choice -- the step keeps its exposure, outcome and cohort.

    Guarded to steps this owner could really compute if they declared.  A step
    that fits a model, promises two products, draws a figure, or already
    carries another owner's typed spec is someone else's contract, and asking
    it to declare this one would demand work that leaves it exactly as unowned.

    NARROWED 2026-08-01 to steps that already promise this owner's product.
    The first version asked every single-table descriptive auxiliary step, on
    the reasoning that a spec alone could not close the gap for a step
    promising a different name.  That reasoning was right about the mechanism
    and wrong about the boundary: measured over every recorded run, it asked 48
    distinct step shapes and only 1 promised this product -- 27 promised
    ``table:cohort_summary``, 18 ``table:absolute_risk_context``, and one each
    ``table:stage_stratified_outcome`` and ``table:ordinal_trend_audit``.
    Demanding those rename their output is demanding they promise a different
    table, which is a scientific choice this owner does not get to make.
    canary33 proved the cost in a real run: ``04_absolute_risk_context``
    executed fine one run earlier and was refused here, taking a figure with
    it.  Whether the same science should be planned under this product's name
    is a real question, but it belongs to the Planner directive, not to a
    refusal raised at the step.
    """

    outputs = [str(value or "").strip() for value in step.expected_outputs or []]
    if (
        outputs != [EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT]
        # A precondition of this owner running at all: it reads one typed
        # cohort. Asking a step without one to declare the spec would demand
        # work that leaves it exactly as unowned.
        or not _typed_cohort_input(step)
        # A step already carrying another owner's typed spec is that owner's
        # contract. Only the two that can actually coexist with this product
        # are listed: ``table_one_spec`` requires ``table:table_one`` as an
        # expected output and ``robustness_replay_spec`` requires its products
        # to be declared outputs, so the schema already makes both impossible
        # here and guarding them was dead code. ``model_requirements`` is
        # likewise unguarded -- the schema forces it onto
        # ``method='adjusted_association_models'``.
        or getattr(step, "measurement_audit_spec", None) is not None
        or step.trajectory_stability_spec is not None
    ):
        # No ``method`` or ``planned_analysis_role`` clause. Both survived
        # mutation once the promised product was pinned, and the corpus says
        # why: of the 32 recorded shapes promising exactly this product, 32 are
        # ``auxiliary``, so the role clause never discriminated, and the method
        # allowlist turned away exactly one -- a step whose method string spelt
        # the same descriptive intent in more words, promising this product
        # with no spec, which is precisely the step this gap exists for. The
        # promised product IS the claim; a two-string method allowlist beside
        # it is the allowlist disease this owner already pays for elsewhere.
        return OwnershipVerdict.wrong_shape(
            EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND,
            reason=(
                f"the step does not promise {EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT!r} "
                "over a typed cohort free of another owner's spec, so this owner "
                "could not compute it however it were declared"
            ),
        )
    missing: List[str] = []
    if step.exposure_outcome_distribution_spec is None:
        missing.append("exposure_outcome_distribution_spec")
    if not missing:
        return OwnershipVerdict.wrong_shape(
            EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND,
            reason=(
                "the step declares this owner's spec and product yet is still "
                "unclaimed, so the gap is not in this declaration"
            ),
        )
    return OwnershipVerdict.incomplete_declaration(
        EXPOSURE_OUTCOME_DISTRIBUTION_ANALYSIS_KIND,
        missing=tuple(missing),
        reason=(
            "the host computes prevalence and outcome by exposure level against "
            "a fixed contract, and this step promises that science without the "
            "declaration it needs: declare exposure_outcome_distribution_spec "
            f"and promise {EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT!r}, the product "
            "this owner emits. Without both, the analysis is written by the "
            "Coder and its table has a different shape every run, so no figure "
            "over it can be drawn"
        ),
    )


def exposure_outcome_distribution_executor_code(
    step: AnalysisStep,
    *,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    """Return the sandbox entrypoint for the exact declared design.

    A flag-only plausibility obligation is mechanical host policy, not a
    reason to hand this otherwise closed product to the stochastic Coder.  The
    host-owned receipt is therefore appended to this owner's exact adapter and
    covered by the same code digest as the distribution product itself.
    """

    if not exposure_outcome_distribution_executor_owns_step(step):
        raise ValueError(
            "The step is not owned by the exposure-outcome distribution executor"
        )
    # The design as it must EXECUTE.  The Planner declares levels in the
    # host's own opaque placeholders whenever it was told a column's
    # cardinality and not its values; two recorded runs died here and were
    # rescued only by a replan that guessed ``[0, 1]``.
    spec = execution_distribution_spec(step)
    assert spec is not None  # narrowed by owns_step
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    require_consumption_contract = bool(
        [
            contract
            for contract in step.input_consumption_contracts
            if contract.input_key == _typed_cohort_input(step)
        ]
    )
    code = textwrap.dedent(
        f"""
        from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
            run_exposure_outcome_distribution_from_env,
        )

        run_exposure_outcome_distribution_from_env(
            spec_payload={spec.model_dump(mode="json")!r},
            typed_cohort_input={_typed_cohort_input(step)!r},
            analysis_role={step.planned_analysis_role!r},
            require_consumption_contract={require_consumption_contract!r},
        )
        """
    ).strip()
    return host_plausibility_receipt_injected(
        code,
        scope=plausibility_scope,
        already_satisfied=False,
    )


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentage(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(100.0 * numerator / denominator, 6)


def wilson_interval(
    events: int, denominator: int, *, confidence_level: float
) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval at the confidence level the spec declares.

    Both the method and its coverage are the Planner's choice, not this
    module's; the ``z`` multiplier is derived from ``confidence_level`` rather
    than written down, so there is no coverage baked into the code that a study
    never asked for. Both travel in the product, so a reader never has to guess
    what a percentage's interval means.
    """

    if denominator <= 0:
        return (None, None)
    if not (0.0 < confidence_level < 1.0):
        raise RuntimeError("confidence_level must lie strictly between 0 and 1")
    z = statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence_level) / 2.0)
    proportion = events / denominator
    centre = proportion + z * z / (2 * denominator)
    spread = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4 * denominator)) / denominator
    )
    factor = 1.0 + z * z / denominator
    low = max(0.0, (centre - spread) / factor)
    high = min(1.0, (centre + spread) / factor)
    return (round(100.0 * low, 6), round(100.0 * high, 6))


def _is_boolean(item: Any) -> bool:
    """True for a real boolean, including the numpy one pandas hands back."""

    return isinstance(item, bool) or type(item).__name__ == "bool_"


def _number(item: Any) -> Optional[float]:
    """The finite numeric value of a genuinely numeric scalar, else ``None``.

    This is the single place that decides a boolean is not a number, and it is
    load-bearing rather than defensive: ``isinstance(True, int)`` is true in
    Python, so without it a column of ``True``/``False`` answers a declared
    level of ``1``/``0`` and the study silently reports a different variable
    from the one it declared. A second copy of this guard elsewhere would make
    both removable without a test noticing, so there is only one.
    """

    if _is_boolean(item) or item is None:
        return None
    if isinstance(item, (int, float)) or type(item).__name__.startswith(
        ("int", "float", "uint")
    ):
        try:
            value = float(item)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
    return None


def _parsed_number(text: str) -> Optional[float]:
    try:
        value = float(text.strip())
    except (TypeError, ValueError, AttributeError):
        return None
    return value if math.isfinite(value) else None


def _matches_scalar(item: Any, level: Any, *, policy: str) -> bool:
    """Whether one observed value is one declared level, under ``policy``.

    No policy lets a boolean and a number match each other in either
    direction. ``numeric_string_equivalent`` only widens the *text spelling* of
    a number, which is a storage difference; a boolean is a different variable.
    """

    if _is_boolean(level):
        return _is_boolean(item) and bool(item) == bool(level)
    # No guard against a boolean *item* is needed here: a non-boolean level is
    # either numeric, where ``_number`` refuses booleans, or a string, which a
    # boolean is not. Repeating the rule would only let one copy be deleted
    # while tests stayed green on the other.
    numeric_equivalence = policy == "numeric_string_equivalent"
    if isinstance(level, (int, float)):
        number = _number(item)
        if number is not None:
            return number == float(level)
        if numeric_equivalence and isinstance(item, str):
            parsed = _parsed_number(item)
            return parsed is not None and parsed == float(level)
        return False
    if isinstance(item, str):
        if item == str(level):
            return True
        if numeric_equivalence:
            parsed_item = _parsed_number(item)
            parsed_level = _parsed_number(str(level))
            return (
                parsed_item is not None
                and parsed_level is not None
                and parsed_item == parsed_level
            )
        return False
    if numeric_equivalence:
        number = _number(item)
        parsed_level = _parsed_number(str(level))
        return (
            number is not None and parsed_level is not None and number == parsed_level
        )
    return False


def _matches(values: pd.Series, level: Any, *, policy: str) -> pd.Series:
    """Mask of the rows whose observed value is the declared ``level``."""

    return values.astype("object").map(
        lambda item: _matches_scalar(item, level, policy=policy)
    )


def _closed_level_masks(
    values: pd.Series,
    levels: List[Any],
    *,
    policy: str,
    column: str,
    role: str,
    observed: pd.Series,
) -> Dict[Tuple[str, str], pd.Series]:
    """Partition observed values across a closed level set, or refuse.

    Two failures are possible and both are fatal. A value matching *two*
    declared levels means the declaration is ambiguous, so every count built
    from it would be arbitrary. A non-missing value matching *none* of them
    means the study met data it never described -- and for an outcome that is
    the dangerous one, because an undeclared value is observed, is not the
    event, and would therefore be counted as a non-event, deflating the rate
    with nothing downstream able to see it.

    The refusal reports how many rows and how many distinct values were
    undeclared, and not the values themselves: the count is what makes the
    failure actionable, while the values are cohort data and a mis-declared
    column could put real measurements into a log.
    """

    masks: Dict[Tuple[str, str], pd.Series] = {}
    for level in levels:
        masks[_typed_level_key(level)] = _matches(values, level, policy=policy)

    counted = None
    for mask in masks.values():
        counted = mask.astype(int) if counted is None else counted + mask.astype(int)
    if counted is None:  # unreachable: the spec requires >= 2 levels
        raise RuntimeError(f"exposure_outcome_distribution declared no {role} levels")
    ambiguous = int((counted > 1).sum())
    if ambiguous:
        raise RuntimeError(
            f"{ambiguous} rows of {column!r} match more than one declared {role} "
            "level; the declared levels must be mutually exclusive"
        )
    undeclared = observed & (counted == 0)
    undeclared_rows = int(undeclared.sum())
    if undeclared_rows:
        distinct = int(values[undeclared].astype("object").nunique(dropna=False))
        raise RuntimeError(
            f"{undeclared_rows} observed rows of {column!r} carry a value that "
            f"is not one of the declared {role} levels ({distinct} distinct "
            "undeclared values). The closed level set is what stops an "
            "undeclared value from being silently counted as a non-event, so "
            "the step fails rather than reporting a deflated rate"
        )
    return masks


def _design_fields(spec: ExposureOutcomeDistributionSpec) -> Dict[str, Any]:
    """The declaration, in the form every product row carries."""

    contrast = spec.risk_difference_contrast
    dependence = spec.dependence
    exposure_keys = [_typed_level_key(level) for level in spec.exposure_levels]
    return {
        "exposure_column": spec.exposure,
        "exposure_levels_declared": json.dumps(
            spec.exposure_levels, ensure_ascii=False
        ),
        "outcome_column": spec.outcome,
        "outcome_levels_declared": json.dumps(spec.outcome_levels, ensure_ascii=False),
        "outcome_positive_index": [
            _typed_level_key(level) for level in spec.outcome_levels
        ].index(_typed_level_key(spec.outcome_positive_value)),
        "level_match_policy": spec.level_match_policy,
        "denominator_policy": spec.denominator_policy,
        "missing_exposure_policy": spec.missing_exposure_policy,
        "missing_outcome_policy": spec.missing_outcome_policy,
        "independent_interval_method": spec.interval_method,
        "repeated_unit_interval_method": spec.repeated_unit_interval_method,
        # This is the EFFECTIVE method used in the published table.  The
        # Planner's Wilson declaration is appropriate for independent rows;
        # once the host binds repeated-patient authority, that authority
        # deterministically upgrades every marginal interval to the same
        # patient-cluster sandwich design as the contrast.
        "interval_method": (
            spec.repeated_unit_interval_method
            if dependence is not None
            else spec.interval_method
        ),
        "confidence_level": spec.confidence_level,
        "risk_difference_reference_index": (
            exposure_keys.index(_typed_level_key(contrast.reference_exposure_level))
            if contrast is not None
            else None
        ),
        "risk_difference_comparison_index": (
            exposure_keys.index(_typed_level_key(contrast.comparison_exposure_level))
            if contrast is not None
            else None
        ),
        "risk_difference_effect_measure": (
            contrast.effect_measure if contrast is not None else None
        ),
        "risk_difference_interval_method": (
            contrast.interval_method if contrast is not None else None
        ),
        "dependence_variance_estimator": (
            dependence.variance_estimator if dependence is not None else None
        ),
        "dependence_cluster_unit": (
            dependence.cluster_unit if dependence is not None else None
        ),
        "dependence_group_source": (
            dependence.group_source if dependence is not None else None
        ),
        "dependence_group_derivation": (
            dependence.group_derivation if dependence is not None else None
        ),
        "dependence_delimiter": (
            dependence.delimiter if dependence is not None else None
        ),
    }


def _dependence_groups(
    frame: pd.DataFrame,
    *,
    spec: ExposureOutcomeDistributionSpec,
    analysis_mask: pd.Series,
) -> tuple[np.ndarray | None, int | None]:
    """Resolve only the exact host-bound patient grouping declaration."""

    dependence = spec.dependence
    if dependence is None:
        return None, None
    source = dependence.group_source
    if source not in frame.columns:
        raise RuntimeError(
            "The host-bound dependence group_source is absent from the bound "
            "cohort; cluster-robust inference cannot fall back to independent rows"
        )
    raw = frame.loc[analysis_mask, source].astype("object")
    missing = int(raw.isna().sum())
    if missing:
        raise RuntimeError(
            f"{missing} analysed rows have no host-bound patient grouping value; "
            "cluster-robust inference fails closed rather than treating each "
            "missing value as an independent patient"
        )

    try:
        resolved = resolve_patient_groups(
            raw.tolist(),
            requirement=dependence,
        )
    except PatientGroupResolutionError as exc:
        raise RuntimeError(str(exc)) from exc
    return np.asarray(resolved.groups, dtype=object), resolved.cluster_count


def _proportion_interval(
    frame: pd.DataFrame,
    *,
    spec: ExposureOutcomeDistributionSpec,
    denominator_mask: pd.Series,
    numerator_mask: pd.Series,
    structural_total: bool = False,
) -> Dict[str, Any]:
    """One marginal proportion under the plan's effective dependence design."""

    denominator = int(denominator_mask.sum())
    numerator = int((denominator_mask & numerator_mask).sum())
    if denominator <= 0:
        raise RuntimeError(
            "marginal proportion has no analysed rows in its declared denominator"
        )
    estimate_pct = percentage(numerator, denominator)
    if structural_total:
        if numerator != denominator:
            raise RuntimeError(
                "structural total must be the whole analysed denominator"
            )
        return {
            "estimate_pct": estimate_pct,
            "standard_error_pct": None,
            "ci_low_pct": None,
            "ci_high_pct": None,
            "covariance": STRUCTURAL_TOTAL_COVARIANCE,
            "cluster_count": None,
        }
    if spec.dependence is None:
        low, high = wilson_interval(
            numerator,
            denominator,
            confidence_level=spec.confidence_level,
        )
        proportion = numerator / denominator
        return {
            "estimate_pct": estimate_pct,
            "standard_error_pct": round(
                100.0 * math.sqrt(proportion * (1.0 - proportion) / denominator),
                6,
            ),
            "ci_low_pct": low,
            "ci_high_pct": high,
            "covariance": "binomial_independent",
            "cluster_count": None,
        }
    if denominator <= 2:
        raise RuntimeError(
            "patient-cluster marginal intervals require more than two analysed rows"
        )
    if numerator in {0, denominator}:
        raise RuntimeError(
            "patient-cluster Wald uncertainty is degenerate at an observed "
            "proportion of zero or one; refusing a zero-width confidence interval"
        )
    y = numerator_mask.loc[denominator_mask].astype(float).to_numpy()
    design = np.ones((len(y), 1), dtype=float)
    groups, cluster_count = _dependence_groups(
        frame,
        spec=spec,
        analysis_mask=denominator_mask,
    )
    assert groups is not None and cluster_count is not None
    fitted = sm.OLS(y, design, missing="raise").fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
            "use_correction": True,
            "df_correction": True,
        },
        use_t=False,
    )
    estimate = float(fitted.params[0])
    standard_error = float(fitted.bse[0])
    count_derived = numerator / denominator
    if (
        not math.isfinite(estimate)
        or not math.isfinite(standard_error)
        or standard_error <= 0.0
    ):
        raise RuntimeError(
            "patient-cluster marginal interval produced non-finite uncertainty"
        )
    if abs(estimate - count_derived) > 1e-10:
        raise RuntimeError(
            "patient-cluster marginal point estimate is not its published count "
            "over denominator"
        )
    critical = statistics.NormalDist().inv_cdf(
        1.0 - (1.0 - spec.confidence_level) / 2.0
    )
    return {
        "estimate_pct": estimate_pct,
        "standard_error_pct": round(100.0 * standard_error, 6),
        # The cluster sandwich supplies uncertainty; bounding its Wald
        # projection keeps a proportion interval on the probability scale.
        # Risk-difference intervals below remain unbounded, as they should.
        "ci_low_pct": round(
            max(0.0, 100.0 * (estimate - critical * standard_error)), 6
        ),
        "ci_high_pct": round(
            min(100.0, 100.0 * (estimate + critical * standard_error)), 6
        ),
        "covariance": "cluster_robust",
        "cluster_count": cluster_count,
    }


def _risk_difference_result(
    frame: pd.DataFrame,
    *,
    spec: ExposureOutcomeDistributionSpec,
    exposure_masks: Dict[Tuple[str, str], pd.Series],
    event: pd.Series,
    observed_outcome: pd.Series,
) -> Dict[str, Any]:
    """Fit the prespecified unadjusted identity-link contrast, if declared."""

    contrast = spec.risk_difference_contrast
    empty = {column: None for column in EXPOSURE_OUTCOME_DISTRIBUTION_CONTRAST_COLUMNS}
    if contrast is None:
        return empty

    reference = exposure_masks[_typed_level_key(contrast.reference_exposure_level)]
    comparison = exposure_masks[_typed_level_key(contrast.comparison_exposure_level)]
    analysis_mask = reference | comparison
    if spec.denominator_policy == "observed_outcome_rows":
        analysis_mask &= observed_outcome

    reference_mask = analysis_mask & reference
    comparison_mask = analysis_mask & comparison
    reference_n = int(reference_mask.sum())
    comparison_n = int(comparison_mask.sum())
    if reference_n <= 0 or comparison_n <= 0:
        raise RuntimeError(
            "risk_difference_contrast requires at least one analysed row in "
            "both the declared reference and comparison levels"
        )

    x = comparison.loc[analysis_mask].astype(float).to_numpy()
    y = event.loc[analysis_mask].astype(float).to_numpy()
    if len(y) <= 2:
        raise RuntimeError(
            "risk_difference_contrast needs more than two analysed rows for "
            "a finite uncertainty estimate"
        )
    design = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, design, missing="raise")
    groups, cluster_count = _dependence_groups(
        frame,
        spec=spec,
        analysis_mask=analysis_mask,
    )
    if groups is None:
        fitted = model.fit(cov_type="HC1", use_t=False)
        covariance = "hc1"
    else:
        fitted = model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups": groups,
                "use_correction": True,
                "df_correction": True,
            },
            use_t=False,
        )
        covariance = "cluster_robust"

    estimate = float(fitted.params[1])
    standard_error = float(fitted.bse[1])
    if (
        not math.isfinite(estimate)
        or not math.isfinite(standard_error)
        or standard_error <= 0.0
    ):
        raise RuntimeError(
            "risk_difference_contrast produced non-finite or zero-width uncertainty"
        )
    reference_risk = int(event.loc[reference_mask].sum()) / reference_n
    comparison_risk = int(event.loc[comparison_mask].sum()) / comparison_n
    count_derived = comparison_risk - reference_risk
    if abs(estimate - count_derived) > 1e-10:
        raise RuntimeError(
            "identity-link risk-difference coefficient is not the difference "
            "of the two published absolute risks"
        )
    critical = statistics.NormalDist().inv_cdf(
        1.0 - (1.0 - spec.confidence_level) / 2.0
    )
    return {
        "risk_difference_n": int(len(y)),
        "risk_difference_pct": round(100.0 * count_derived, 6),
        "risk_difference_standard_error_pct": round(100.0 * standard_error, 6),
        "risk_difference_ci_low_pct": round(
            100.0 * (estimate - critical * standard_error), 6
        ),
        "risk_difference_ci_high_pct": round(
            100.0 * (estimate + critical * standard_error), 6
        ),
        "risk_difference_covariance": covariance,
        "risk_difference_cluster_count": cluster_count,
    }


def _distribution_rows(
    frame: pd.DataFrame,
    *,
    spec: ExposureOutcomeDistributionSpec,
) -> List[Dict[str, Any]]:
    exposure_values = frame[spec.exposure]
    outcome_values = frame[spec.outcome]
    observed_outcome = outcome_values.notna()
    observed_exposure = exposure_values.notna()

    # A missing exposure is refused on its own terms rather than swept into the
    # undeclared-level bucket below. Both stop the step, but they send a reader
    # to different places: one means the data holds a category the study never
    # described, the other means the study has rows it cannot group at all.
    # Reporting the second as the first sends someone hunting for a stray code
    # that does not exist.
    missing_exposure_rows = int((~observed_exposure).sum())
    if missing_exposure_rows and spec.missing_exposure_policy == "fail_closed":
        raise RuntimeError(
            f"{missing_exposure_rows} rows have no observed value for "
            f"{spec.exposure!r}; the spec declares missing_exposure_policy="
            "'fail_closed', so they are neither dropped nor pooled into a "
            "declared exposure level"
        )
    if missing_exposure_rows:
        # Complete-case on the exposure. The rows leave the frame entirely --
        # every denominator below is then taken over the same rows, which is
        # the property a reader checks -- and the count they left behind
        # travels in the product so the shrink is visible rather than inferred
        # from a total that does not add up.
        #
        # canary13 is why this option exists at all: 8 of 1000 stays had no
        # AKI stage, `fail_closed` was the ONLY value the field could take, and
        # the step died with no result the Planner had any way to avoid.
        frame = frame.loc[observed_exposure]
        exposure_values = frame[spec.exposure]
        outcome_values = frame[spec.outcome]
        observed_outcome = outcome_values.notna()
        observed_exposure = exposure_values.notna()
    exposure_masks = _closed_level_masks(
        exposure_values,
        spec.exposure_levels,
        policy=spec.level_match_policy,
        column=spec.exposure,
        role="exposure",
        observed=observed_exposure,
    )
    outcome_masks = _closed_level_masks(
        outcome_values,
        spec.outcome_levels,
        policy=spec.level_match_policy,
        column=spec.outcome,
        role="outcome",
        observed=observed_outcome,
    )
    # The event mask is *the* declared positive level's mask, not a second
    # match: the spec guarantees the positive value is one of the closed
    # levels, so reusing it makes the events and the level counts one fact.
    event = outcome_masks[_typed_level_key(spec.outcome_positive_value)]

    missing_outcome_rows = int((~observed_outcome).sum())
    if missing_outcome_rows and spec.missing_outcome_policy == "fail_closed":
        raise RuntimeError(
            f"{missing_outcome_rows} rows have no observed value for "
            f"{spec.outcome!r}; the spec declares missing_outcome_policy="
            "'fail_closed', so they are neither dropped nor counted as "
            "non-events without an explicit decision"
        )

    total_rows = int(len(frame))
    exposure_denominator = total_rows
    all_rows = pd.Series(True, index=frame.index, dtype=bool)
    # Carried beside the declaration on every row, not merged into it: the
    # policy is what the plan asked for, this is what the data cost.
    design = {
        **_design_fields(spec),
        "missing_exposure_excluded_n": missing_exposure_rows,
    }
    contrast_result = _risk_difference_result(
        frame,
        spec=spec,
        exposure_masks=exposure_masks,
        event=event,
        observed_outcome=observed_outcome,
    )

    rows: List[Dict[str, Any]] = []
    for level_index, level in enumerate(spec.exposure_levels):
        mask = exposure_masks[_typed_level_key(level)]
        n_rows = int(mask.sum())
        exposure_interval = _proportion_interval(
            frame,
            spec=spec,
            denominator_mask=all_rows,
            numerator_mask=mask,
        )
        observed_n = int((mask & observed_outcome).sum())
        missing_n = n_rows - observed_n
        events = int((mask & event).sum())
        outcome_denominator = (
            n_rows if spec.denominator_policy == "all_declared_rows" else observed_n
        )
        outcome_denominator_mask = (
            mask
            if spec.denominator_policy == "all_declared_rows"
            else mask & observed_outcome
        )
        outcome_interval = _proportion_interval(
            frame,
            spec=spec,
            denominator_mask=outcome_denominator_mask,
            numerator_mask=event,
        )
        rows.append(
            {
                "row_role": _LEVEL_ROLE,
                "exposure_level_index": level_index,
                "exposure_level": level,
                "n_rows": n_rows,
                "exposure_denominator": exposure_denominator,
                "exposure_pct": exposure_interval["estimate_pct"],
                "exposure_ci_low_pct": exposure_interval["ci_low_pct"],
                "exposure_ci_high_pct": exposure_interval["ci_high_pct"],
                "exposure_standard_error_pct": exposure_interval["standard_error_pct"],
                "exposure_interval_covariance": exposure_interval["covariance"],
                "exposure_interval_cluster_count": exposure_interval["cluster_count"],
                "outcome_observed_n": observed_n,
                "outcome_missing_n": missing_n,
                "outcome_events": events,
                "outcome_denominator": outcome_denominator,
                "outcome_rate_pct": outcome_interval["estimate_pct"],
                "ci_low_pct": outcome_interval["ci_low_pct"],
                "ci_high_pct": outcome_interval["ci_high_pct"],
                "outcome_standard_error_pct": outcome_interval["standard_error_pct"],
                "outcome_interval_covariance": outcome_interval["covariance"],
                "outcome_interval_cluster_count": outcome_interval["cluster_count"],
                **contrast_result,
                **design,
            }
        )

    overall_observed = int(observed_outcome.sum())
    overall_events = int(event.sum())
    overall_denominator = (
        total_rows
        if spec.denominator_policy == "all_declared_rows"
        else overall_observed
    )
    overall_outcome_denominator_mask = (
        all_rows if spec.denominator_policy == "all_declared_rows" else observed_outcome
    )
    outcome_interval = _proportion_interval(
        frame,
        spec=spec,
        denominator_mask=overall_outcome_denominator_mask,
        numerator_mask=event,
    )
    exposure_interval = _proportion_interval(
        frame,
        spec=spec,
        denominator_mask=all_rows,
        numerator_mask=all_rows,
        structural_total=True,
    )
    rows.append(
        {
            "row_role": _OVERALL_ROLE,
            "exposure_level_index": None,
            "exposure_level": None,
            "n_rows": total_rows,
            "exposure_denominator": exposure_denominator,
            "exposure_pct": exposure_interval["estimate_pct"],
            "exposure_ci_low_pct": exposure_interval["ci_low_pct"],
            "exposure_ci_high_pct": exposure_interval["ci_high_pct"],
            "exposure_standard_error_pct": exposure_interval["standard_error_pct"],
            "exposure_interval_covariance": exposure_interval["covariance"],
            "exposure_interval_cluster_count": exposure_interval["cluster_count"],
            "outcome_observed_n": overall_observed,
            "outcome_missing_n": total_rows - overall_observed,
            "outcome_events": overall_events,
            "outcome_denominator": overall_denominator,
            "outcome_rate_pct": outcome_interval["estimate_pct"],
            "ci_low_pct": outcome_interval["ci_low_pct"],
            "ci_high_pct": outcome_interval["ci_high_pct"],
            "outcome_standard_error_pct": outcome_interval["standard_error_pct"],
            "outcome_interval_covariance": outcome_interval["covariance"],
            "outcome_interval_cluster_count": outcome_interval["cluster_count"],
            **contrast_result,
            **design,
        }
    )
    return rows


def _verify_product(
    rows: List[Dict[str, Any]], *, spec: ExposureOutcomeDistributionSpec
) -> None:
    """Refuse to publish a table that does not add up.

    The executor computed these numbers, so this is not defensive noise: it is
    the difference between a bug that surfaces here and one that surfaces as a
    wrong figure in a manuscript. Every published quantity is re-derived from
    the counts beside it, so a wrong percentage cannot pass merely by being
    plausible.
    """

    level_rows = [row for row in rows if row["row_role"] == _LEVEL_ROLE]
    overall = [row for row in rows if row["row_role"] == _OVERALL_ROLE]
    if len(overall) != 1:
        raise RuntimeError(
            "exposure_outcome_distribution needs exactly one overall row"
        )
    total = overall[0]
    contrast = spec.risk_difference_contrast
    critical = statistics.NormalDist().inv_cdf(
        1.0 - (1.0 - spec.confidence_level) / 2.0
    )

    def verify_interval(
        row: Dict[str, Any],
        *,
        numerator: int,
        denominator: int,
        estimate_key: str,
        low_key: str,
        high_key: str,
        standard_error_key: str,
        covariance_key: str,
        cluster_count_key: str,
        label: str,
        structural_total: bool = False,
    ) -> None:
        estimate = percentage(numerator, denominator)
        if row[estimate_key] != estimate:
            raise RuntimeError(f"{label} percentage is not its own counts")
        if structural_total:
            if (
                denominator <= 0
                or numerator != denominator
                or row[standard_error_key] is not None
                or row[low_key] is not None
                or row[high_key] is not None
                or row[covariance_key] != STRUCTURAL_TOTAL_COVARIANCE
                or row[cluster_count_key] is not None
            ):
                raise RuntimeError(
                    f"{label} structural total carries inferential uncertainty"
                )
            return
        if spec.dependence is None:
            proportion = numerator / denominator
            expected_standard_error = round(
                100.0 * math.sqrt(proportion * (1.0 - proportion) / denominator),
                6,
            )
            if (
                row[low_key],
                row[high_key],
            ) != wilson_interval(
                numerator,
                denominator,
                confidence_level=spec.confidence_level,
            ):
                raise RuntimeError(
                    f"{label} interval is not Wilson at the declared confidence"
                )
            if (
                row[standard_error_key] != expected_standard_error
                or row[covariance_key] != "binomial_independent"
                or row[cluster_count_key] is not None
            ):
                raise RuntimeError(
                    f"{label} Wilson interval carries contradictory independent "
                    "uncertainty"
                )
            return
        standard_error = _finite(row[standard_error_key])
        if standard_error is None or standard_error <= 0.0:
            raise RuntimeError(f"{label} patient-cluster standard error is invalid")
        expected_low = round(max(0.0, float(estimate) - critical * standard_error), 6)
        expected_high = round(
            min(100.0, float(estimate) + critical * standard_error), 6
        )
        if (
            abs(float(row[low_key]) - expected_low) > 2e-6
            or abs(float(row[high_key]) - expected_high) > 2e-6
        ):
            raise RuntimeError(
                f"{label} interval is not its patient-cluster Wald arithmetic"
            )
        cluster_count = row[cluster_count_key]
        if (
            row[covariance_key] != "cluster_robust"
            or cluster_count is None
            or int(cluster_count) < 2
        ):
            raise RuntimeError(
                f"{label} interval lacks its patient-cluster covariance receipt"
            )

    if sum(row["n_rows"] for row in level_rows) != total["n_rows"]:
        raise RuntimeError("declared exposure levels do not partition the cohort")
    if sum(row["outcome_events"] for row in level_rows) != total["outcome_events"]:
        raise RuntimeError("level events do not sum to the overall events")
    if (
        sum(row["outcome_observed_n"] for row in level_rows)
        != total["outcome_observed_n"]
    ):
        raise RuntimeError("level observed counts do not sum to the overall observed")
    if (
        sum(row["outcome_missing_n"] for row in level_rows)
        != total["outcome_missing_n"]
    ):
        raise RuntimeError("level missing counts do not sum to the overall missing")
    for row in rows:
        if row["outcome_events"] > row["outcome_denominator"]:
            raise RuntimeError("more events than the denominator they are taken over")
        if row["outcome_observed_n"] + row["outcome_missing_n"] != row["n_rows"]:
            raise RuntimeError("observed plus missing does not equal the row count")
        expected_denominator = (
            row["n_rows"]
            if spec.denominator_policy == "all_declared_rows"
            else row["outcome_observed_n"]
        )
        if row["outcome_denominator"] != expected_denominator:
            raise RuntimeError(
                "the outcome denominator does not follow the declared "
                f"denominator_policy={spec.denominator_policy!r}"
            )
        verify_interval(
            row,
            numerator=row["n_rows"],
            denominator=row["exposure_denominator"],
            estimate_key="exposure_pct",
            low_key="exposure_ci_low_pct",
            high_key="exposure_ci_high_pct",
            standard_error_key="exposure_standard_error_pct",
            covariance_key="exposure_interval_covariance",
            cluster_count_key="exposure_interval_cluster_count",
            label="exposure prevalence",
            structural_total=row["row_role"] == _OVERALL_ROLE,
        )
        verify_interval(
            row,
            numerator=row["outcome_events"],
            denominator=row["outcome_denominator"],
            estimate_key="outcome_rate_pct",
            low_key="ci_low_pct",
            high_key="ci_high_pct",
            standard_error_key="outcome_standard_error_pct",
            covariance_key="outcome_interval_covariance",
            cluster_count_key="outcome_interval_cluster_count",
            label="outcome absolute risk",
        )
        rate = _finite(row["outcome_rate_pct"])
        low = _finite(row["ci_low_pct"])
        high = _finite(row["ci_high_pct"])
        if rate is not None and low is not None and high is not None:
            if not (low - 1e-6 <= rate <= high + 1e-6):
                raise RuntimeError("the reported rate falls outside its own interval")
    for column in EXPOSURE_OUTCOME_DISTRIBUTION_CONTRAST_COLUMNS:
        if len({repr(row[column]) for row in rows}) != 1:
            raise RuntimeError(
                f"risk-difference result column {column!r} differs across rows"
            )
    if contrast is None:
        if any(
            row[column] is not None
            for row in rows
            for column in EXPOSURE_OUTCOME_DISTRIBUTION_CONTRAST_COLUMNS
        ):
            raise RuntimeError(
                "risk-difference results were emitted without a declared contrast"
            )
        return

    exposure_keys = [_typed_level_key(level) for level in spec.exposure_levels]
    reference_index = exposure_keys.index(
        _typed_level_key(contrast.reference_exposure_level)
    )
    comparison_index = exposure_keys.index(
        _typed_level_key(contrast.comparison_exposure_level)
    )
    reference_row = level_rows[reference_index]
    comparison_row = level_rows[comparison_index]
    expected_difference = round(
        float(comparison_row["outcome_rate_pct"])
        - float(reference_row["outcome_rate_pct"]),
        6,
    )
    if rows[0]["risk_difference_pct"] != expected_difference:
        raise RuntimeError(
            "risk difference is not comparison absolute risk minus reference "
            "absolute risk"
        )
    expected_n = int(reference_row["outcome_denominator"]) + int(
        comparison_row["outcome_denominator"]
    )
    if int(rows[0]["risk_difference_n"]) != expected_n:
        raise RuntimeError(
            "risk-difference analysis count is not the sum of the declared "
            "reference and comparison outcome denominators"
        )
    standard_error = _finite(rows[0]["risk_difference_standard_error_pct"])
    if standard_error is None or standard_error <= 0.0:
        raise RuntimeError("risk difference has invalid standard error")
    expected_low = round(expected_difference - critical * standard_error, 6)
    expected_high = round(expected_difference + critical * standard_error, 6)
    if (
        abs(float(rows[0]["risk_difference_ci_low_pct"]) - expected_low) > 2e-6
        or abs(float(rows[0]["risk_difference_ci_high_pct"]) - expected_high) > 2e-6
    ):
        raise RuntimeError(
            "risk-difference interval is not its point estimate plus/minus the "
            "declared Wald-normal uncertainty"
        )
    expected_covariance = "cluster_robust" if spec.dependence is not None else "hc1"
    if rows[0]["risk_difference_covariance"] != expected_covariance:
        raise RuntimeError(
            "risk-difference covariance does not match the bound dependence contract"
        )
    cluster_count = rows[0]["risk_difference_cluster_count"]
    if spec.dependence is None and cluster_count is not None:
        raise RuntimeError(
            "independent-row risk-difference inference cannot report patient clusters"
        )
    if spec.dependence is not None and (
        cluster_count is None or int(cluster_count) < 2
    ):
        raise RuntimeError(
            "cluster-robust risk-difference inference must report at least two "
            "patient clusters"
        )


def run_exposure_outcome_distribution_from_env(
    *,
    spec_payload: Dict[str, Any],
    typed_cohort_input: str,
    analysis_role: str = "auxiliary",
    require_consumption_contract: bool = False,
) -> Dict[str, Any]:
    """Execute the declared distribution from the standard runner environment."""

    spec = ExposureOutcomeDistributionSpec.model_validate(spec_payload)
    analysis_role = str(analysis_role or "").strip().casefold()
    if analysis_role not in {"primary", "secondary", "sensitivity", "auxiliary"}:
        raise RuntimeError(
            "exposure_outcome_distribution requires a closed analysis_role"
        )
    if not str(typed_cohort_input or "").strip():
        raise RuntimeError(
            "exposure_outcome_distribution requires an exact typed cohort "
            "binding; it does not read an unbound cohort from the environment, "
            "because a table counted from unverified bytes cannot be bound to "
            "the plan that asked for it"
        )
    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, cohort_path = load_typed_cohort(
        input_key=typed_cohort_input,
        run_dir=run_dir_from_env(),
        resolved_inputs_path=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]).resolve(),
        require_consumption_contract=require_consumption_contract,
    )

    missing = [
        column
        for column in (spec.exposure, spec.outcome)
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            "Declared exposure/outcome columns are absent from the bound cohort: "
            + ", ".join(missing)
        )

    rows = _distribution_rows(frame, spec=spec)
    _verify_product(rows, spec=spec)

    table = pd.DataFrame(rows, columns=list(EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS))
    table_path = out_dir / "exposure_outcome_distribution.csv"
    table.to_csv(table_path, index=False)

    level_rows = [row for row in rows if row["row_role"] == _LEVEL_ROLE]
    effective_interval_method = str(rows[0]["interval_method"])
    exposure_prevalence = []
    outcome_absolute_risks = []
    for index, (level, row) in enumerate(zip(spec.exposure_levels, level_rows)):
        exposure_prevalence.append(
            {
                "level_index": index,
                "level": level,
                "n": int(row["n_rows"]),
                "denominator": int(row["exposure_denominator"]),
                "estimate_pct": row["exposure_pct"],
                "standard_error_pct": row["exposure_standard_error_pct"],
                "ci_low_pct": row["exposure_ci_low_pct"],
                "ci_high_pct": row["exposure_ci_high_pct"],
                "confidence_level": spec.confidence_level,
                "interval_method": effective_interval_method,
                "covariance": row["exposure_interval_covariance"],
                "cluster_count": row["exposure_interval_cluster_count"],
            }
        )
        outcome_absolute_risks.append(
            {
                "level_index": index,
                "level": level,
                "events": int(row["outcome_events"]),
                "denominator": int(row["outcome_denominator"]),
                "estimate_pct": row["outcome_rate_pct"],
                "standard_error_pct": row["outcome_standard_error_pct"],
                "ci_low_pct": row["ci_low_pct"],
                "ci_high_pct": row["ci_high_pct"],
                "confidence_level": spec.confidence_level,
                "interval_method": effective_interval_method,
                "covariance": row["outcome_interval_covariance"],
                "cluster_count": row["outcome_interval_cluster_count"],
            }
        )
    contrast = spec.risk_difference_contrast
    excluded_exposure_n = int(rows[0]["missing_exposure_excluded_n"])
    analysed_n = int(
        next(row for row in rows if row["row_role"] == _OVERALL_ROLE)["n_rows"]
    )
    analysis_set = (
        EXPOSURE_OBSERVED_ANALYSIS_SET
        if excluded_exposure_n
        else BOUND_TYPED_COHORT_ANALYSIS_SET
    )
    risk_difference = None
    if contrast is not None:
        reference_index = int(rows[0]["risk_difference_reference_index"])
        comparison_index = int(rows[0]["risk_difference_comparison_index"])
        risk_difference = {
            "reference_level_index": reference_index,
            "reference_level": spec.exposure_levels[reference_index],
            "comparison_level_index": comparison_index,
            "comparison_level": spec.exposure_levels[comparison_index],
            "direction": "comparison_minus_reference",
            "n": int(rows[0]["risk_difference_n"]),
            "estimate_pct": rows[0]["risk_difference_pct"],
            "standard_error_pct": rows[0]["risk_difference_standard_error_pct"],
            "ci_low_pct": rows[0]["risk_difference_ci_low_pct"],
            "ci_high_pct": rows[0]["risk_difference_ci_high_pct"],
            "confidence_level": spec.confidence_level,
            "interval_method": rows[0]["risk_difference_interval_method"],
            "covariance": rows[0]["risk_difference_covariance"],
            "cluster_count": rows[0]["risk_difference_cluster_count"],
            "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        }
    descriptive_estimates = {
        "schema_version": "easyicu.exposure_outcome_descriptive_estimates/1",
        "analysis_role": analysis_role,
        "analysis_set": analysis_set,
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        # One typed authority governs every covariance-bearing estimate in this
        # envelope.  Counts can legitimately differ after exposure stratification,
        # but consumers must never infer a different grouping rule per estimate.
        "dependence": (
            spec.dependence.model_dump(mode="json")
            if spec.dependence is not None
            else None
        ),
        "exposure_prevalence": exposure_prevalence,
        "outcome_absolute_risks": outcome_absolute_risks,
        "risk_difference": risk_difference,
    }

    summary = {
        "status": "ok",
        "analysis_family": "descriptive",
        "interpretation_class": "exposure_outcome_distribution",
        "interpretation_ceiling": "descriptive_unadjusted_not_causal",
        "analysis_role": analysis_role,
        "analysis_set": analysis_set,
        "cohort_n": analysed_n,
        "exposure": spec.exposure,
        "outcome": spec.outcome,
        "denominator_policy": spec.denominator_policy,
        "missing_outcome_policy": spec.missing_outcome_policy,
        "level_match_policy": spec.level_match_policy,
        "interval_method": effective_interval_method,
        "independent_interval_method": spec.interval_method,
        "repeated_unit_interval_method": spec.repeated_unit_interval_method,
        "confidence_level": spec.confidence_level,
        "effective_interval_method": effective_interval_method,
        "typed_cohort_input": typed_cohort_input,
        "source_cohort": cohort_path.name,
        "source_row_count_reconciliation": {
            "source_rows": int(len(frame)),
            "analyzed_rows": analysed_n,
            "excluded_missing_exposure_rows": excluded_exposure_n,
            "filtering_performed": bool(excluded_exposure_n),
        },
        "adjusted_effect": None,
        "descriptive_estimates": descriptive_estimates,
        "descriptive_contrast": (
            {
                "effect_measure": "risk_difference",
                "estimate_pct": rows[0]["risk_difference_pct"],
                "ci_low_pct": rows[0]["risk_difference_ci_low_pct"],
                "ci_high_pct": rows[0]["risk_difference_ci_high_pct"],
                "confidence_level": spec.confidence_level,
                "interval_method": rows[0]["risk_difference_interval_method"],
                "covariance": rows[0]["risk_difference_covariance"],
                "cluster_count": rows[0]["risk_difference_cluster_count"],
                "interpretation_ceiling": "descriptive_unadjusted_not_causal",
            }
            if spec.risk_difference_contrast is not None
            else None
        ),
        "output_files": {EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT: table_path.name},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary

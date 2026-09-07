"""Exact claim boundary for the host-owned primary association executor.

This module exists because the same question -- *can the sealed EasyICU
executor actually fit the model this plan declares?* -- was being answered
independently in three places that could disagree, and did.

Measured on ``ba11f52`` with a plan declaring
``method='adjusted_association_models'``,
``table:adjusted_association_estimates`` and
``method_family='statsmodels_glm_binomial'`` (a canonical token in
``model_tokens._ASSOCIATION_METHOD_ALIASES``, so a Planner may legitimately
emit it):

* the capability registry answered ``association_adjusted_v1`` /
  ``primary_analysis='deterministic'``;
* ``validate_required_primary_result`` accepted the plan;
* the owner declined ``wrong_shape`` -- it implements Logit and OLS only;
* ``owner_declaration`` reports only ``incomplete_declaration``, so a
  wrong-shape decline is (correctly, for its own purpose) silent.

The run therefore advertised a deterministic host capability and executed the
LLM coder.  That is not a documentation error: ``run_status.json`` records the
capability, and readiness reads it, so a stochastic primary estimate inherits a
label asserting a sealed one.

The fix is not a fourth check.  It is one statement of what the owner
implements, imported by the owner, by plan validation and by the capability
resolver, so the three answers cannot drift.  ``contracts`` is the right home
for the same reason :mod:`.survival_execution` lives here: it is pure data plus
a pure predicate, with no dependency on the execution layer, so the planning
and reporting layers can read it without inverting the dependency direction.

Deliberately NOT here: the input-arity and product-bundling clauses of
:func:`..execution.runners.adjusted_association_executor
.adjusted_association_executor_verdict`.  Those need execution-layer binding
helpers, and no layer above execution has to reason about them -- they cannot
turn a supported estimator into an unsupported one, only decline a step for a
reason the coder path handles correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence

from .cohort_product_keys import sole_typed_cohort_input
from .capability_ids import LANDMARK_CATEGORICAL_ANALYSIS_KIND
from .model_tokens import (
    ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
    ADJUSTED_ASSOCIATION_OUTPUT,
    ASSOCIATION_LOGIT_ESTIMATOR,
    ASSOCIATION_OLS_ESTIMATOR,
    PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,
    canonical_association_method,
    normalise_model_contract_token,
)
from .ownership_verdict import OwnershipVerdict


#: The one mapping from a declared outcome type to the exact estimator token the
#: sealed executor implements.  Both directions matter: a continuous family such
#: as quantile regression fitted as OLS, or a GLM-binomial family fitted as
#: Logit, would answer a different scientific question under the declared
#: method's name -- which is precisely what this package refuses to do
#: silently.
ASSOCIATION_HOST_ESTIMATORS: Mapping[str, str] = {
    "binary": ASSOCIATION_LOGIT_ESTIMATOR,
    "continuous": ASSOCIATION_OLS_ESTIMATOR,
}


#: The registered agent-coded association capability reused by a sensitivity
#: child only after the host closes the narrower contract below.  The id alone
#: never grants sensitivity authority; role, parent, output and variant roster
#: must all match.
ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID = "association_freeform_v1"
ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT = "table:adjusted_association_estimates"


@dataclass(frozen=True, slots=True)
class AssociationBinarySensitivityContract:
    """Closed effect boundary for one prespecified binary sensitivity grid."""

    parent_product: str
    output_product: str
    sensitivity_ids: tuple[str, ...]
    effect_measure: str = "odds_ratio"


@dataclass(frozen=True, slots=True)
class AssociationBinarySensitivityPlanVerdict:
    """Attributable answer for a plan claiming the sensitivity capability."""

    claimed: bool
    reason_code: str
    reason: str
    contract: Optional[AssociationBinarySensitivityContract] = None


@dataclass(frozen=True, slots=True)
class AssociationBinarySensitivityResultIssue:
    """One execution-result violation without importing schema types."""

    reason_code: str
    message: str
    detail: Mapping[str, Any]


def association_binary_sensitivity_contract(
    step: Any,
) -> Optional[AssociationBinarySensitivityContract]:
    """Return the local closed contract, or ``None`` for any incomplete shape.

    This deliberately reads only typed plan coordinates.  Method prose,
    analysis-family inference, and effect-looking output names cannot grant
    authority.  Plan-wide parent ownership is checked separately by
    :func:`association_binary_sensitivity_plan_verdict`.
    """

    if (
        str(getattr(step, "scientific_capability", "") or "").strip()
        != ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID
        or getattr(step, "planned_analysis_role", None) != "sensitivity"
        or not str(getattr(step, "method", "") or "").strip()
    ):
        return None
    outputs = tuple(
        str(value or "").strip()
        for value in (getattr(step, "expected_outputs", None) or ())
    )
    if (
        len(outputs) != 1
        or not outputs[0].startswith("table:")
        or len(outputs[0].partition(":")[2]) == 0
    ):
        return None
    inputs = tuple(
        str(value or "").strip() for value in (getattr(step, "inputs", None) or ())
    )
    if inputs.count(ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT) != 1:
        return None
    sensitivity_ids = tuple(
        str(value or "").strip()
        for value in (getattr(step, "sensitivity_spec_ids", None) or ())
    )
    if (
        not sensitivity_ids
        or any(not value for value in sensitivity_ids)
        or len(sensitivity_ids) != len(set(sensitivity_ids))
    ):
        return None
    return AssociationBinarySensitivityContract(
        parent_product=ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT,
        output_product=outputs[0],
        sensitivity_ids=sensitivity_ids,
    )


@dataclass(frozen=True, slots=True)
class AssociationEstimatorSupport:
    """Whether the host owner implements one declared estimator contract.

    ``runtime_kind`` is the ``fit_estimator`` kind the executor would dispatch
    to, and is empty exactly when ``supported`` is false.  Keeping them in one
    frozen value stops a caller from re-deriving one of them from the other.
    """

    supported: bool
    runtime_kind: str
    declared_outcome_type: str
    canonical_method_family: str
    expected_estimator: str
    reason: str


def association_estimator_support(requirement: Any) -> AssociationEstimatorSupport:
    """State whether the sealed association owner implements this contract."""

    outcome_type = str(getattr(requirement, "outcome_type", "") or "").strip()
    declared = canonical_association_method(getattr(requirement, "method_family", None))
    expected = ASSOCIATION_HOST_ESTIMATORS.get(outcome_type, "")
    if not expected:
        return AssociationEstimatorSupport(
            supported=False,
            runtime_kind="",
            declared_outcome_type=outcome_type,
            canonical_method_family=declared,
            expected_estimator="",
            reason=(
                f"outcome_type {outcome_type!r} has no host-implemented "
                "association estimator"
            ),
        )
    if declared != expected:
        return AssociationEstimatorSupport(
            supported=False,
            runtime_kind="",
            declared_outcome_type=outcome_type,
            canonical_method_family=declared,
            expected_estimator=expected,
            reason=(
                f"method family {declared!r} for outcome type {outcome_type!r} is "
                f"not an estimator this owner implements (it implements "
                f"{expected!r})"
            ),
        )
    return AssociationEstimatorSupport(
        supported=True,
        runtime_kind="logistic" if outcome_type == "binary" else "linear",
        declared_outcome_type=outcome_type,
        canonical_method_family=declared,
        expected_estimator=expected,
        reason="the declared estimator is the exact host-implemented one",
    )


def association_estimator_kind(requirement: Any) -> str:
    """The ``fit_estimator`` kind for a supported contract, else ``""``."""

    return association_estimator_support(requirement).runtime_kind


def _association_execution_verdict(
    step: Any,
    *,
    allowed_methods: Sequence[str],
) -> OwnershipVerdict:
    """Own only a single, completely declared adjusted-association model.

    Every clause is a thing the host would otherwise have to decide:

    * exactly one model requirement, because ``bind_primary_output`` binds a
      one-row table; a two-model step is a different product shape, not a
      bigger version of this one;
    * a declared adjustment set, because reconstructing one from ``step.inputs``
      is inference (see the contract's own tests);
    * an estimator this module actually implements -- a quantile-regression
      family fitted as OLS would answer a different question under the declared
      method's name;
    * one typed cohort input at most, so the frame that was analysed is the
      digest-bound one.

    Measured over 553 recorded steps, 54 of the 59 declines here were a field
    the Planner simply never filled in -- and the ``bool`` this used to return
    sent every one of them to the coder without telling anyone.  See
    :mod:`.ownership_verdict`.

    Two clauses are deliberately **not** reported as incomplete declarations,
    because more declaring is not what would fix them:

    * a step bundling this product with others is task #105's question of
      whether an owner's claim may depend on Planner bundling at all, and
      calling it "missing" would misname an over-declaration;
    * more than one typed input, or an unimplemented estimator family, are
      contracts this owner does not have.
    """

    method = normalise_model_contract_token(
        str(getattr(step, "method", "") or "").lower().split(" with ", 1)[0]
    )
    normalized_allowed_methods = tuple(
        normalise_model_contract_token(value) for value in allowed_methods
    )
    if method not in normalized_allowed_methods:
        return OwnershipVerdict.wrong_shape(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            reason=(
                f"step method {method!r} is not one of the host-owned "
                f"adjusted-association methods {normalized_allowed_methods!r}"
            ),
        )
    declared_outputs = [
        str(value or "").strip()
        for value in getattr(step, "expected_outputs", None) or []
    ]
    if declared_outputs != [ADJUSTED_ASSOCIATION_OUTPUT]:
        return OwnershipVerdict.wrong_shape(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            reason=(
                f"step declares {len(declared_outputs)} expected output(s), not "
                f"exactly [{ADJUSTED_ASSOCIATION_OUTPUT}]"
            ),
        )
    requirements = list(getattr(step, "model_requirements", None) or [])
    if not requirements:
        return OwnershipVerdict.incomplete_declaration(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            missing=("model_requirements",),
            reason=(
                "the step declares the primary adjusted-association product but "
                "no model requirement, so the outcome, outcome type, method "
                "family and exposure are undeclared"
            ),
        )
    if len(requirements) != 1:
        return OwnershipVerdict.wrong_shape(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            reason=(
                f"step declares {len(requirements)} model requirements; a "
                "multi-model step is a different product shape"
            ),
        )
    requirement = requirements[0]
    if requirement.covariates is None:
        return OwnershipVerdict.incomplete_declaration(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            missing=("model_requirements[0].covariates",),
            reason=(
                "the model requirement declares no adjustment set, and "
                "reconstructing one from step.inputs would be inference"
            ),
        )
    if requirement.model_terms is None:
        return OwnershipVerdict.incomplete_declaration(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            missing=("model_requirements[0].model_terms",),
            reason=(
                "the model requirement names variables but does not declare "
                "their coding, levels, references and transforms"
            ),
        )
    support = association_estimator_support(requirement)
    if not support.supported:
        return OwnershipVerdict.wrong_shape(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            reason=support.reason,
        )
    if sole_typed_cohort_input(step) == "":
        return OwnershipVerdict.wrong_shape(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            reason=(
                "the step declares more than one typed input, or one this "
                "executor family does not support"
            ),
        )
    for spec_name in (
        "table_one_spec",
        "trajectory_stability_spec",
        "exposure_outcome_distribution_spec",
    ):
        if getattr(step, spec_name, None) is not None:
            return OwnershipVerdict.wrong_shape(
                ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
                reason=(
                    f"the step also declares {spec_name}, which another owner claims"
                ),
            )
    return OwnershipVerdict.claim(
        ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
        reason="a single, completely declared adjusted-association model",
    )


def association_execution_verdict(step: Any) -> OwnershipVerdict:
    """Own the generic adjusted-association method and no signed variant."""

    return _association_execution_verdict(
        step,
        allowed_methods=(PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,),
    )


def landmark_categorical_association_execution_verdict(
    step: Any,
) -> OwnershipVerdict:
    """Own the signed fixed-landmark wrapper over the same exact model contract."""

    return _association_execution_verdict(
        step,
        allowed_methods=(LANDMARK_CATEGORICAL_ANALYSIS_KIND,),
    )


def sole_primary_model_requirement(step: Any) -> Optional[Any]:
    """The single model requirement the host product binds, or ``None``.

    ``bind_primary_output`` binds a one-row table, so "zero" and "many" are
    both "not this owner's product shape".  Returning ``None`` for both is the
    same collapse the executor already makes; callers that need to tell them
    apart read ``model_requirements`` directly.
    """

    requirements = list(getattr(step, "model_requirements", None) or [])
    return requirements[0] if len(requirements) == 1 else None


def association_binary_sensitivity_plan_verdict(
    step: Any,
    *,
    plan_steps: Sequence[Any],
) -> AssociationBinarySensitivityPlanVerdict:
    """Verify the capability is inherited from one exact binary parent.

    The sensitivity step remains agent-coded.  What the host certifies is the
    narrower proposition needed by the effect gate: its input is the unique
    host-owned binary adjusted-association product, its requested variants are
    a closed id roster, and it owns exactly one typed result table.
    """

    declared = str(getattr(step, "scientific_capability", "") or "").strip()
    if declared != ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID:
        return AssociationBinarySensitivityPlanVerdict(
            claimed=False,
            reason_code="association_binary_sensitivity_not_declared",
            reason="the step does not declare the binary sensitivity capability",
        )
    contract = association_binary_sensitivity_contract(step)
    if contract is None:
        return AssociationBinarySensitivityPlanVerdict(
            claimed=False,
            reason_code="association_binary_sensitivity_shape_invalid",
            reason=(
                "the capability requires role='sensitivity', one table output, "
                "one adjusted-association product input, a non-empty unique "
                "sensitivity id roster, and a declared method"
            ),
        )
    steps = tuple(plan_steps)
    producers = [
        candidate
        for candidate in steps
        if contract.parent_product
        in {
            str(value or "").strip()
            for value in (getattr(candidate, "expected_outputs", None) or ())
        }
    ]
    if len(producers) != 1:
        return AssociationBinarySensitivityPlanVerdict(
            claimed=False,
            reason_code="association_binary_sensitivity_parent_ambiguous",
            reason=(
                "the inherited adjusted-association product must have exactly "
                f"one plan owner; found {len(producers)}"
            ),
        )
    parent = producers[0]
    try:
        parent_index = next(
            index for index, candidate in enumerate(steps) if candidate is parent
        )
        child_index = next(
            index for index, candidate in enumerate(steps) if candidate is step
        )
    except StopIteration:
        return AssociationBinarySensitivityPlanVerdict(
            claimed=False,
            reason_code="association_binary_sensitivity_step_unbound",
            reason="the sensitivity step or its parent is absent from the plan",
        )
    if parent_index >= child_index:
        return AssociationBinarySensitivityPlanVerdict(
            claimed=False,
            reason_code="association_binary_sensitivity_parent_not_preceding",
            reason="the adjusted-association parent must precede its sensitivity child",
        )
    # A current-case runtime may replace the generic primary method with its
    # signed landmark adapter while retaining the same closed model requirement
    # and output product.  Sensitivity children inherit that exact product, so
    # validate both host-owned parent spellings here without widening the
    # generic adjusted-association executor's own claim predicate.
    parent_verdict = _association_execution_verdict(
        parent,
        allowed_methods=(
            PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,
            LANDMARK_CATEGORICAL_ANALYSIS_KIND,
        ),
    )
    requirement = sole_primary_model_requirement(parent)
    if (
        getattr(parent, "planned_analysis_role", None) != "primary"
        or not parent_verdict.claimed
        or requirement is None
        or str(getattr(requirement, "outcome_type", "") or "").strip() != "binary"
    ):
        return AssociationBinarySensitivityPlanVerdict(
            claimed=False,
            reason_code="association_binary_sensitivity_parent_invalid",
            reason=(
                "the parent must be the preceding host-owned primary adjusted "
                "association with one binary-outcome model requirement; "
                f"owner_reason={parent_verdict.reason}"
            ),
        )
    return AssociationBinarySensitivityPlanVerdict(
        claimed=True,
        reason_code="association_binary_sensitivity_contract_closed",
        reason=(
            "one prespecified sensitivity grid inherits one host-owned binary "
            "adjusted-association product"
        ),
        contract=contract,
    )


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nonnegative_integer(value: Any) -> Optional[int]:
    number = _finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def association_binary_sensitivity_result_issues(
    step: Any,
    step_summary: Mapping[str, Any],
) -> tuple[AssociationBinarySensitivityResultIssue, ...]:
    """Validate the machine-readable result promised by the closed contract."""

    contract = association_binary_sensitivity_contract(step)
    if contract is None:
        return ()
    rows = step_summary.get("analysis_rows")
    if not isinstance(rows, list):
        return (
            AssociationBinarySensitivityResultIssue(
                reason_code="association_binary_sensitivity_rows_missing",
                message=(
                    "The binary sensitivity capability requires an analysis_rows "
                    "list in step_summary."
                ),
                detail={"expected_analysis_ids": list(contract.sensitivity_ids)},
            ),
        )
    row_ids = [
        str(row.get("analysis_id") or "").strip() if isinstance(row, Mapping) else ""
        for row in rows
    ]
    issues: list[AssociationBinarySensitivityResultIssue] = []
    if (
        len(row_ids) != len(contract.sensitivity_ids)
        or len(row_ids) != len(set(row_ids))
        or set(row_ids) != set(contract.sensitivity_ids)
    ):
        issues.append(
            AssociationBinarySensitivityResultIssue(
                reason_code="association_binary_sensitivity_ids_mismatch",
                message=(
                    "Sensitivity results must contain each planner-owned "
                    "sensitivity id exactly once and no undeclared ids."
                ),
                detail={
                    "expected_analysis_ids": list(contract.sensitivity_ids),
                    "observed_analysis_ids": row_ids,
                },
            )
        )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            issues.append(
                AssociationBinarySensitivityResultIssue(
                    reason_code="association_binary_sensitivity_row_invalid",
                    message="Every binary sensitivity result row must be an object.",
                    detail={"row_index": index},
                )
            )
            continue
        analysis_id = str(row.get("analysis_id") or "").strip()
        n_stays = _nonnegative_integer(row.get("n_stays"))
        n_deaths = _nonnegative_integer(row.get("n_deaths"))
        odds_ratio = _finite_number(row.get("odds_ratio"))
        ci_low = _finite_number(row.get("ci_low"))
        ci_high = _finite_number(row.get("ci_high"))
        invalid_fields: list[str] = []
        if n_stays is None or n_stays < 1:
            invalid_fields.append("n_stays")
        if n_deaths is None or (n_stays is not None and n_deaths > n_stays):
            invalid_fields.append("n_deaths")
        if odds_ratio is None or odds_ratio <= 0:
            invalid_fields.append("odds_ratio")
        if ci_low is None or ci_low <= 0:
            invalid_fields.append("ci_low")
        if ci_high is None or ci_high <= 0:
            invalid_fields.append("ci_high")
        if (
            odds_ratio is not None
            and ci_low is not None
            and ci_high is not None
            and not (0 < ci_low <= odds_ratio <= ci_high)
        ):
            invalid_fields.append("effect_interval_order")
        if invalid_fields:
            issues.append(
                AssociationBinarySensitivityResultIssue(
                    reason_code="association_binary_sensitivity_row_invalid",
                    message=(
                        "Each binary sensitivity row requires coherent finite "
                        "counts and a positive odds ratio within its interval."
                    ),
                    detail={
                        "row_index": index,
                        "analysis_id": analysis_id or None,
                        "invalid_fields": invalid_fields,
                    },
                )
            )
    return tuple(issues)


__all__ = [
    "ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID",
    "ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT",
    "ASSOCIATION_HOST_ESTIMATORS",
    "AssociationBinarySensitivityContract",
    "AssociationBinarySensitivityPlanVerdict",
    "AssociationBinarySensitivityResultIssue",
    "AssociationEstimatorSupport",
    "association_binary_sensitivity_contract",
    "association_binary_sensitivity_plan_verdict",
    "association_binary_sensitivity_result_issues",
    "association_estimator_kind",
    "association_estimator_support",
    "association_execution_verdict",
    "landmark_categorical_association_execution_verdict",
    "sole_primary_model_requirement",
]

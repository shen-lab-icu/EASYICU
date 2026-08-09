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
from typing import Any, Mapping, Optional

from .cohort_product_keys import sole_typed_cohort_input
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


def association_execution_verdict(step: Any) -> OwnershipVerdict:
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
    if method != PLANNED_MODEL_REQUIREMENTS_STEP_METHOD:
        return OwnershipVerdict.wrong_shape(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            reason=(
                f"step method {method!r} is not "
                f"{PLANNED_MODEL_REQUIREMENTS_STEP_METHOD!r}"
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


def sole_primary_model_requirement(step: Any) -> Optional[Any]:
    """The single model requirement the host product binds, or ``None``.

    ``bind_primary_output`` binds a one-row table, so "zero" and "many" are
    both "not this owner's product shape".  Returning ``None`` for both is the
    same collapse the executor already makes; callers that need to tell them
    apart read ``model_requirements`` directly.
    """

    requirements = list(getattr(step, "model_requirements", None) or [])
    return requirements[0] if len(requirements) == 1 else None


__all__ = [
    "ASSOCIATION_HOST_ESTIMATORS",
    "AssociationEstimatorSupport",
    "association_estimator_kind",
    "association_estimator_support",
    "association_execution_verdict",
    "sole_primary_model_requirement",
]

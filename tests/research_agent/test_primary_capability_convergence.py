"""One scientific-capability verdict, read by every layer that acts on it.

The defect these lock: on ``ba11f52`` a plan declaring the host's
adjusted-association method and product with ``method_family=
'statsmodels_glm_binomial'`` -- a canonical token a Planner may legitimately
emit -- was labelled ``association_adjusted_v1`` / ``deterministic`` by the
capability registry, accepted by ``validate_required_primary_result``, declined
``wrong_shape`` by the sealed owner, and executed by the LLM coder. The
deterministic label reached ``run_status.json`` and readiness while a
stochastic actor produced the estimate.

The second defect: ``association_freeform_v1`` was registered, routed to by
``get_capability_for_plan`` and covered by its own registry test, but
``validate_required_primary_result`` required the exact single-model contract
of *every* ``association_study``, so no plan carrying that capability could
survive Planner parse.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.association_execution import (
    association_estimator_support,
    association_execution_verdict,
)
from easyicu.research_agent.contracts.model_terms import ModelTermSpec
from easyicu.research_agent.planning.capability_registry import (
    resolve_primary_capability,
)
from easyicu.research_agent.planning.primary_result_contract import (
    validate_required_primary_result,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    PlannedModelRequirement,
    ResearchContext,
)


QUESTION = "Is sepsis associated with in-hospital mortality?"


def _terms() -> list[ModelTermSpec]:
    return [
        ModelTermSpec(
            name="sep3", role="exposure", coding="continuous", transform="identity"
        ),
        ModelTermSpec(
            name="age", role="covariate", coding="continuous", transform="identity"
        ),
    ]


def _requirement(method_family: str) -> PlannedModelRequirement:
    return PlannedModelRequirement(
        requirement_id="m1",
        analysis_role="primary",
        analysis_set="complete_case",
        required_for_step_success=True,
        exposure_source="sep3",
        outcome="death",
        outcome_type="binary",
        method_family=method_family,
        covariates=["age"],
        model_terms=_terms(),
    )


def _context() -> ResearchContext:
    return ResearchContext(
        research_question=QUESTION,
        cohort=CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=8, n_stays=8
        ),
        primary_exposure="sep3",
        target_outcome="death",
        variables=[
            ConceptDescriptor(name="sep3", description="sepsis", dtype="int64"),
            ConceptDescriptor(name="death", description="death", dtype="int64"),
            ConceptDescriptor(name="age", description="age", dtype="float64"),
        ],
    )


def _exact_plan(method_family: str) -> AnalysisPlan:
    step = AnalysisStep(
        step_id="06_primary",
        intent="Fit the primary adjusted association model.",
        method="adjusted_association_models",
        inputs=["table:analysis_cohort"],
        expected_outputs=["table:adjusted_association_estimates"],
        planned_analysis_role="primary",
        model_requirements=[_requirement(method_family)],
    )
    return AnalysisPlan(
        research_question=QUESTION, analysis_type="association_study", steps=[step]
    )


def _freeform_plan(
    *,
    outputs: tuple[str, ...] = ("table:interaction_model_estimates",),
) -> AnalysisPlan:
    # No ``model_requirements``: ``AnalysisStep`` refuses them on any step that
    # is not the exact host method/product pair, which is precisely why the
    # free-form capability is agent-coded.
    step = AnalysisStep(
        step_id="06_primary",
        intent="Fit an exposure-by-age interaction model.",
        method="association_interaction_model",
        inputs=["table:analysis_cohort"],
        expected_outputs=list(outputs),
        planned_analysis_role="primary",
        scientific_capability="association_freeform_v1",
    )
    return AnalysisPlan(
        research_question=QUESTION, analysis_type="association_study", steps=[step]
    )


# --- the deterministic contract still resolves and still validates -----------


def test_supported_estimator_resolves_to_the_deterministic_host_owner() -> None:
    verdict = resolve_primary_capability(
        analysis_type="association_study", plan=_exact_plan("statsmodels_logit_mle")
    )
    assert verdict.capability_id == "association_adjusted_v1"
    assert verdict.execution_owner == "host_deterministic"
    assert verdict.owner_claimed is True
    assert verdict.coherent
    validate_required_primary_result(
        plan=_exact_plan("statsmodels_logit_mle"), context=_context()
    )


# --- the label/runtime mismatch is now reported, not executed ----------------


def test_host_product_with_unimplemented_estimator_is_not_labelled_deterministic() -> (
    None
):
    plan = _exact_plan("statsmodels_glm_binomial")
    verdict = resolve_primary_capability(analysis_type="association_study", plan=plan)

    # The owner is the one that would actually run: it declines.
    assert association_execution_verdict(plan.steps[0]).claimed is False
    # So no layer may call this a deterministic host capability.
    assert verdict.execution_owner == "unresolved"
    assert verdict.failure_reason == "primary_capability_owner_mismatch"
    assert verdict.scientific_validation == "unsupported"
    assert not verdict.coherent


def test_planner_refuses_a_plan_whose_owner_cannot_run_it() -> None:
    with pytest.raises(ValueError, match="sealed executor cannot run it"):
        validate_required_primary_result(
            plan=_exact_plan("statsmodels_glm_binomial"), context=_context()
        )


def test_unimplemented_estimator_is_not_silently_rerouted_to_freeform() -> None:
    """Re-routing would let any plan escape the deterministic owner.

    Naming an estimator the host does not implement must not be a way to get
    the looser agent-coded contract for the host's own product key.
    """

    verdict = resolve_primary_capability(
        analysis_type="association_study", plan=_exact_plan("statsmodels_glm_binomial")
    )
    assert verdict.capability_id != "association_freeform_v1"


def test_capability_assessment_refuses_the_mismatched_label() -> None:
    from easyicu.research_agent.planning.capability_registry import (
        assess_scientific_capability,
    )

    assessment = assess_scientific_capability(
        analysis_type="association_study",
        context=_context(),
        plan=_exact_plan("statsmodels_glm_binomial"),
    )
    assert assessment.claim_ceiling == "unsupported"
    assert assessment.issue_code == "primary_capability_owner_mismatch"
    assert not assessment.claim_ceiling_allows_reportable


# --- the registered free-form capability is now reachable --------------------


def test_freeform_association_plan_survives_planner_validation() -> None:
    plan = _freeform_plan()
    verdict = resolve_primary_capability(analysis_type="association_study", plan=plan)
    assert verdict.capability_id == "association_freeform_v1"
    assert verdict.execution_owner == "agent_coded"
    assert verdict.coherent
    validate_required_primary_result(plan=plan, context=_context())


def test_freeform_step_may_not_claim_the_sealed_owners_product_key() -> None:
    plan = _freeform_plan(outputs=("table:adjusted_association_estimates",))
    verdict = resolve_primary_capability(analysis_type="association_study", plan=plan)
    assert verdict.failure_reason == "freeform_step_claims_host_product"
    with pytest.raises(ValueError, match="sealed executor"):
        validate_required_primary_result(plan=plan, context=_context())


def test_freeform_is_declared_not_inferred_from_a_non_matching_method() -> None:
    """An under-declared association plan may not fall into the looser contract.

    A feasibility audit and an interaction model are structurally identical to
    a validator: one primary step, a plausible method, a typed result product.
    Only an explicit declaration separates them, so the absence of one selects
    the strict family default.
    """

    plan = _freeform_plan()
    undeclared = plan.steps[0].model_copy(update={"scientific_capability": None})
    plan = plan.model_copy(update={"steps": [undeclared]})

    verdict = resolve_primary_capability(analysis_type="association_study", plan=plan)
    assert verdict.capability_id == "association_adjusted_v1"
    with pytest.raises(ValueError, match="adjusted_association_models"):
        validate_required_primary_result(plan=plan, context=_context())


# --- the rule has one definition ---------------------------------------------


@pytest.mark.parametrize(
    ("outcome_type", "outcome", "method_family", "supported"),
    [
        ("binary", "death", "statsmodels_logit_mle", True),
        ("binary", "death", "logistic_regression", True),
        ("binary", "death", "statsmodels_glm_binomial", False),
        ("continuous", "los_icu", "statsmodels_ols", True),
        ("continuous", "los_icu", "statsmodels_quantreg", False),
    ],
)
def test_estimator_support_is_the_single_statement_the_owner_uses(
    outcome_type: str, outcome: str, method_family: str, supported: bool
) -> None:
    requirement = PlannedModelRequirement(
        requirement_id="m1",
        analysis_role="primary",
        analysis_set="complete_case",
        required_for_step_success=True,
        exposure_source="sep3",
        outcome=outcome,
        outcome_type=outcome_type,
        method_family=method_family,
        covariates=["age"],
        model_terms=_terms(),
    )
    support = association_estimator_support(requirement)
    assert support.supported is supported
    # The owner's own decline reason is this statement, not a second copy.
    step = _exact_plan("statsmodels_logit_mle").steps[0]
    step = step.model_copy(update={"model_requirements": [requirement]})
    verdict = association_execution_verdict(step)
    assert verdict.claimed is supported
    if not supported:
        assert verdict.reason == support.reason

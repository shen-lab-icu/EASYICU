"""The typed headline requirement covers every currently result-bearing family.

Association studies keep their model-roster contract. Causal and survival
studies instead use a family-specific primary-result packet plus a reconciliation
gate that binds an exact registered CSV row to that packet. The narrow roster
validator below deliberately remains association-only: a hazard ratio or causal
estimand is not a logistic-model contract.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.audits.validators import PrimaryModelContractValidator
from easyicu.research_agent.planning.primary_result_contract import (
    validate_required_primary_result,
)
from easyicu.research_agent.schema import (
    PLANNED_MODEL_REQUIREMENTS_OUTPUT,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
    PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)

#: Families whose plans this contract currently refuses when the typed headline
#: obligation is missing. Extending the contract means adding to this list in the
#: same commit that extends the code.
FAMILIES_WITH_A_TYPED_HEADLINE_OBLIGATION = (
    "association_study",
    "causal_inference",
    "survival",
)

#: Result-bearing families it is currently silent for. This is intentionally
#: empty; introducing another family requires a corresponding typed contract.
FAMILIES_WITHOUT_ONE = ()


def _context(question: str) -> ResearchContext:
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="c", database="miiv", n_patients=9, n_stays=9
        ),
        variables=[
            ConceptDescriptor(
                name="vaso",
                description="exposure",
                dtype="int64",
                role=VariableRole.INTERVENTION,
            ),
            ConceptDescriptor(
                name="death",
                description="outcome",
                dtype="int64",
                role=VariableRole.OUTCOME,
            ),
        ],
        primary_exposure="vaso",
        target_outcome="death",
    )


def _plan_with_no_typed_obligation(family: str) -> AnalysisPlan:
    """One primary step, a plausible method and product, zero requirements.

    This is the exact shape of all 9 recorded h2 plans and all 13 h1 plans.
    """

    step = AnalysisStep(
        step_id="04_primary",
        intent="Estimate the primary effect.",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:primary_causal_contrast"],
        method="causal_effect_estimation_iptw",
        planned_analysis_role="primary",
    )
    return AnalysisPlan(
        research_question=(
            "Does vasopressor exposure change 28-day mortality? Estimate the "
            "causal effect."
        ),
        analysis_type=family,
        steps=[step],
    )


@pytest.mark.parametrize("family", FAMILIES_WITH_A_TYPED_HEADLINE_OBLIGATION)
def test_a_covered_family_is_refused_without_the_typed_obligation(family: str) -> None:
    plan = _plan_with_no_typed_obligation(family)
    with pytest.raises(
        ValueError,
        match="model_requirements|family_primary_result|Declare which registered",
    ):
        validate_required_primary_result(
            plan=plan, context=_context(plan.research_question)
        )


def test_the_two_lists_are_disjoint_and_name_real_families() -> None:
    from easyicu.research_agent.planning.analysis_types import (
        canonical_analysis_family,
    )

    covered = set(FAMILIES_WITH_A_TYPED_HEADLINE_OBLIGATION)
    uncovered = set(FAMILIES_WITHOUT_ONE)
    assert not covered & uncovered
    for family in covered | uncovered:
        assert canonical_analysis_family(family) == family, family


def test_the_adjusted_association_model_reconciliation_remains_narrow() -> None:
    """The third link, and the reason the first two cannot be extended alone.

    The association validator stays scoped to its own model roster. Causal and
    survival primary results use the separate family-primary reconciliation
    gate, rather than pretending their estimands are logistic-model contracts.
    """

    assert PrimaryModelContractValidator._CLOSED_EFFECT_METHODS == {
        PLANNED_MODEL_REQUIREMENTS_STEP_METHOD
    }
    assert PrimaryModelContractValidator._CLOSED_EFFECT_PRODUCTS == {
        (
            PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
            PLANNED_MODEL_REQUIREMENTS_OUTPUT,
        )
    }
    # And the constants are the association ones, so the sets are not merely
    # small but specifically that family's.
    assert PLANNED_MODEL_REQUIREMENTS_STEP_METHOD == "adjusted_association_models"
    assert PLANNED_MODEL_REQUIREMENTS_OUTPUT == "adjusted_association_estimates"


def test_the_typed_roster_states_its_own_scope() -> None:
    """The docstring is the declaration of scope; keep it honest as scope grows.

    A schema that silently represented a survival or causal estimand while
    saying it does not would be worse than one that refuses: a reader checking
    whether the roster can express a hazard ratio would be told no.
    """

    from easyicu.research_agent.schema import PlannedModelRequirement

    doc = PlannedModelRequirement.__doc__ or ""
    assert "does not represent survival" in doc

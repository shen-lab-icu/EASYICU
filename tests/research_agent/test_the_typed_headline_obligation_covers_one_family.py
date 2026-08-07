"""Characterization: which families the typed headline contract actually covers.

Written BEFORE extending it, to record what is true now and to make the scope a
single editable list rather than something a reader has to re-derive.

MEASURED over the recorded plans of every result-bearing task, counting primary
steps that carry a required primary ``model_requirements`` entry:

    association_study   e1 112/115   e3 38/38   m1 13/13   e2 17/17   = 180/183
    survival            h1   0/13
    causal_inference    h2    0/9

The four tasks that have ever produced a verified manuscript are exactly the four
whose family carries this obligation. 183 against 22, no overlap. That is
correlational -- those families differ in other ways too -- but the obligation is
the difference the code can act on, and it is absent from both result-bearing
families that have never produced a manuscript.

The chain has three links and all three are keyed to one family:

* ``validate_required_primary_result`` returns early unless the family is
  ``association_study``;
* the typed roster ``PlannedModelRequirement`` says in its own docstring that
  "this v1 schema does not represent survival, prediction, mixed-effects, or
  clustering contracts";
* the execution-side reconciliation activates on ``_CLOSED_EFFECT_METHODS`` and
  ``_CLOSED_EFFECT_PRODUCTS``, each a single-element set built from one constant,
  with the comment "only for the adjusted-association families this validator
  implements".

So a causal step can name its output ``primary_causal_contrast``, emit
arm-specific weighted risks with no between-arm contrast, and pass every
deterministic gate -- which is what three recorded auditor findings say happened.
Only the LLM auditor noticed.

These tests do not endorse that scope. They pin it, so extending it is a
deliberate edit to the lists below and not a silent widening.
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
FAMILIES_WITH_A_TYPED_HEADLINE_OBLIGATION = ("association_study",)

#: Result-bearing families it is currently silent for. Each entry is a family
#: whose primary step may declare nothing and still be accepted.
FAMILIES_WITHOUT_ONE = ("causal_inference", "survival")


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
    with pytest.raises(ValueError, match="model_requirements"):
        validate_required_primary_result(
            plan=plan, context=_context(plan.research_question)
        )


@pytest.mark.parametrize("family", FAMILIES_WITHOUT_ONE)
def test_an_uncovered_family_is_accepted_with_nothing_declared(family: str) -> None:
    """The gap, recorded rather than asserted to be correct.

    When this test starts failing because the contract was extended, that is the
    intended outcome: move the family from FAMILIES_WITHOUT_ONE to the covered
    list above in the same commit.
    """

    plan = _plan_with_no_typed_obligation(family)
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


def test_the_execution_side_reconciliation_is_keyed_to_one_method_and_product() -> None:
    """The third link, and the reason the first two cannot be extended alone.

    Even a causal plan that declared a typed obligation would not have it
    reconciled against the emitted model contract: this validator activates only
    on the single association method and the single association product.
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

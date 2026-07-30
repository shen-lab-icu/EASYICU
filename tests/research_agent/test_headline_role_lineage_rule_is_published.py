"""A role credited only on the primary lineage must say so where it is read.

MEASURED 2026-07-30 on the recorded ``h1_ventilation_survival`` failure in
``batch_20260730_luna_miiv_dev_88d3983_canonical9_full02``: five Planner
attempts, three distinct causes, nothing executed.  Two of the five were::

    PlannerArticleContractError: Planner plan is missing required article
    contract role(s): data_quality, temporal_absolute_risk.
      Required typed step examples: temporal_absolute_risk -> 'table:survival_curve'

``temporal_absolute_risk`` is one of the survival family's
``planner_owned_result_roles``, so ``roles_covered_by_plan`` credits it only
from steps inside ``_declared_primary_lineage_step_ids``.  Meanwhile
``AnalysisPlan`` permits at most one ``planned_analysis_role='primary'`` step
and refuses a primary step whose products are all rendering.

Put together, the plan a clinician would actually write -- the Cox model as the
single primary step, the Kaplan-Meier curve as its own secondary display step
reading the analysis cohort -- cannot cover the role.  Only two shapes can:
bundling the curve into the primary step, or having the display step consume a
typed product that only a lineage step produces.

Neither was written down anywhere the Planner reads.  The rendered contract
block contained none of the words ``primary``, ``lineage``, ``headline`` or
``planner_owned``, and its own rule -- "every required role must be owned by an
explicit analysis step; put its typed_example in expected_outputs" -- is false
for exactly these roles.  The remediation hint repeated the same incomplete
instruction, which is why the retries kept re-adding the same off-lineage step.

This is not one family's quirk.  Seven of fifteen families declare a required
headline-owned role; among the canonical nine that is the ventilation-survival,
mortality-prediction, vasopressor-causal and trajectory-clustering tasks.

The repair is publication, not a new gate.  ``roles_covered_by_plan`` is left
alone on purpose: its docstring says a sensitivity step may not be routed into
the headline lineage, and loosening it would let a sensitivity curve satisfy the
headline requirement.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.planning.analysis_types import _FAMILY_ALIASES
from easyicu.research_agent.reporting.article_contract import (
    build_article_analysis_contract,
    render_article_analysis_contract_for_prompt,
    roles_covered_by_plan,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)

_LINEAGE_RULE = "credited only when the declaring step is on the primary lineage"


def _survival_context() -> ResearchContext:
    """The recorded task, reduced to what decides the family."""

    return ResearchContext(
        research_question=(
            "Among adults receiving invasive mechanical ventilation within the "
            "first 24 ICU hours, is early ventilation associated with 28-day "
            "survival?"
        ),
        cohort=CohortDescriptor(
            cohort_name="analysis_cohort",
            database="miiv",
            n_patients=94458,
            n_stays=94458,
            id_columns=["stay_id"],
            outcome_columns=["mort_28d"],
        ),
        variables=[
            ConceptDescriptor(
                name="mort_28d",
                role=VariableRole.OUTCOME,
                source_concept="mort_28d",
                dtype="int",
            ),
            ConceptDescriptor(
                name="mech_vent",
                role=VariableRole.INTERVENTION,
                source_concept="mech_vent",
                dtype="int",
            ),
        ],
        target_outcome="mort_28d",
    )


def _step(step_id: str, inputs, outputs, method: str, role: str) -> AnalysisStep:
    return AnalysisStep(
        step_id=step_id,
        intent="Estimate the effect and show absolute risk over time.",
        inputs=list(inputs),
        expected_outputs=list(outputs),
        method=method,
        planned_analysis_role=role,
    )


def _plan(steps) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Is early ventilation associated with 28-day survival?",
        analysis_type="survival",
        cohort=None,
        steps=list(steps),
    )


_COX = _step(
    "06_cox",
    ["artifact:analysis_cohort"],
    ["table:survival_effect_estimates"],
    "cox_proportional_hazards",
    "primary",
)


# ---------------------------------------------------------------------------
# The premise: the natural plan really cannot satisfy the contract
# ---------------------------------------------------------------------------


def test_the_role_is_headline_owned_in_this_family() -> None:
    """If this ever stops being true the rest of the file is vacuous."""

    contract = build_article_analysis_contract(
        _survival_context(), analysis_type="survival"
    )

    assert "temporal_absolute_risk" in contract.required_roles
    assert "temporal_absolute_risk" in contract.planner_owned_result_roles


def test_the_plan_a_clinician_would_write_misses_the_role() -> None:
    """Cox primary, curve secondary, curve reads the cohort. Measured: missed."""

    contract = build_article_analysis_contract(
        _survival_context(), analysis_type="survival"
    )
    plan = _plan(
        [
            _COX,
            _step(
                "07_km",
                ["artifact:analysis_cohort"],
                ["figure:survival_curve"],
                "kaplan_meier_survival",
                "secondary",
            ),
        ]
    )

    covered = roles_covered_by_plan(plan, contract)

    assert "survival_effect" in covered
    assert "temporal_absolute_risk" not in covered


def test_a_second_primary_step_is_refused_by_the_plan_schema() -> None:
    """The first workaround a reader reaches for is not available."""

    with pytest.raises(ValueError, match="at most one step"):
        _plan(
            [
                _COX,
                _step(
                    "07_km",
                    ["artifact:analysis_cohort"],
                    ["table:survival_curve"],
                    "kaplan_meier_survival",
                    "primary",
                ),
            ]
        )


def test_a_primary_step_of_only_displays_is_refused_too() -> None:
    """And so is the second workaround."""

    with pytest.raises(ValueError, match="non-rendering scientific result"):
        _plan(
            [
                _step(
                    "07_km",
                    ["artifact:analysis_cohort"],
                    ["figure:survival_curve"],
                    "kaplan_meier_survival",
                    "primary",
                )
            ]
        )


@pytest.mark.parametrize(
    "label,steps_factory",
    [
        (
            "bundled into the single primary step",
            lambda: [
                _step(
                    "06_cox",
                    ["artifact:analysis_cohort"],
                    ["table:survival_effect_estimates", "figure:survival_curve"],
                    "cox_proportional_hazards",
                    "primary",
                )
            ],
        ),
        (
            "a display step consuming the primary step's typed output",
            lambda: [
                _COX,
                _step(
                    "07_km",
                    ["table:survival_effect_estimates"],
                    ["figure:survival_curve"],
                    "kaplan_meier_survival",
                    "secondary",
                ),
            ],
        ),
    ],
)
def test_the_two_shapes_the_rule_names_do_satisfy_it(label, steps_factory) -> None:
    """The published rule has to describe something that actually works."""

    contract = build_article_analysis_contract(
        _survival_context(), analysis_type="survival"
    )

    covered = roles_covered_by_plan(_plan(steps_factory()), contract)

    assert "temporal_absolute_risk" in covered, label


# ---------------------------------------------------------------------------
# The rule is now where the Planner reads it
# ---------------------------------------------------------------------------


def test_the_rendered_contract_states_the_lineage_rule() -> None:
    rendered = render_article_analysis_contract_for_prompt(
        build_article_analysis_contract(_survival_context())
    )

    assert "headline_owned=true" in rendered
    assert _LINEAGE_RULE in rendered
    # The two shapes the measurement proved, named rather than implied.
    assert "at most one primary step" in rendered
    assert "consumes the primary step's typed output" in rendered
    assert "reading only the cohort does not join the lineage" in rendered


def test_only_the_headline_owned_modules_carry_the_marker() -> None:
    """Marking everything would publish nothing."""

    contract = build_article_analysis_contract(
        _survival_context(), analysis_type="survival"
    )
    rendered = render_article_analysis_contract_for_prompt(contract)
    owned = set(contract.planner_owned_result_roles)

    for requirement in contract.requirements:
        if not requirement.required:
            continue
        line = next(
            text
            for text in rendered.splitlines()
            if text.strip().startswith(f"- {requirement.module_id} (")
        )
        assert ("headline_owned=true" in line) is (requirement.role in owned), (
            f"{requirement.module_id} (role={requirement.role}) is marked "
            "headline_owned inconsistently with the contract"
        )


def test_every_family_that_needs_the_rule_publishes_it_and_no_other_does() -> None:
    """Case-neutral scope, asserted over every family rather than assumed.

    Measured 2026-07-30: seven of fifteen families declare a required
    headline-owned role.
    """

    context = _survival_context()
    with_rule = set()
    for family in sorted(set(_FAMILY_ALIASES.values())):
        contract = build_article_analysis_contract(context, analysis_type=family)
        rendered = render_article_analysis_contract_for_prompt(contract)
        needs = bool(
            set(contract.planner_owned_result_roles) & set(contract.required_roles)
        )
        assert (_LINEAGE_RULE in rendered) is needs, (
            f"family {family!r} publishes the lineage rule={_LINEAGE_RULE in rendered} "
            f"but needs it={needs}"
        )
        if needs:
            with_rule.add(family)

    assert with_rule == {
        "association_study",
        "causal_inference",
        "dynamic_prediction",
        "prediction_model",
        "survival",
        "trajectory_clustering",
        "validation",
    }


# ---------------------------------------------------------------------------
# And in the rejection the Planner actually receives
# ---------------------------------------------------------------------------


def test_the_rejection_tells_the_planner_why_its_display_step_did_not_count() -> None:
    """Reachability: the real parser, the real plan shape that died.

    Without this the Planner is told to declare a product it already declared,
    which is what turned one defect into five paid attempts.
    """

    from easyicu.research_agent.agents.core import (
        PlannerAgent,
        PlannerArticleContractError,
    )

    context = _survival_context()
    plan_json = json.dumps(
        {
            "research_question": context.research_question,
            "analysis_type": "survival",
            "cohort": None,
            "robustness_specs": [],
            "rationale": "Cox as the headline result, a survival curve beside it.",
            "steps": [
                {
                    "step_id": "06_cox",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate the adjusted hazard ratio.",
                    "inputs": ["artifact:analysis_cohort"],
                    "expected_outputs": ["table:survival_effect_estimates"],
                    "method": "cox_proportional_hazards",
                    "icu_rule_refs": [],
                },
                {
                    "step_id": "07_km",
                    "planned_analysis_role": "secondary",
                    "intent": "Show absolute risk over follow-up by exposure.",
                    "inputs": ["artifact:analysis_cohort"],
                    "expected_outputs": ["figure:survival_curve"],
                    "method": "kaplan_meier_survival",
                    "icu_rule_refs": [],
                },
            ],
        }
    )

    with pytest.raises(PlannerArticleContractError) as excinfo:
        PlannerAgent(llm=object())._parse(
            plan_json,
            context,
            enforce_article_contract=True,
            article_contract_context=context,
        )

    message = str(excinfo.value)
    assert "temporal_absolute_risk" in message
    # The flag sits on the role->example line, where the Planner reads what to
    # declare, and again in the sentence that says what to do about it.
    assert "temporal_absolute_risk (headline_owned) ->" in message
    assert "credited only on the primary lineage" in message
    assert "does not join the lineage" in message


def test_a_plan_missing_only_ordinary_roles_gets_no_lineage_note() -> None:
    """The note must attach to the roles it explains, not to every rejection."""

    from easyicu.research_agent.agents.core import (
        PlannerAgent,
        PlannerArticleContractError,
    )

    context = _survival_context()
    plan_json = json.dumps(
        {
            "research_question": context.research_question,
            "analysis_type": "survival",
            "cohort": None,
            "robustness_specs": [],
            "rationale": "Cover both headline roles, skip the ordinary ones.",
            "steps": [
                {
                    "step_id": "06_cox",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate the hazard ratio and show absolute risk.",
                    "inputs": ["artifact:analysis_cohort"],
                    "expected_outputs": [
                        "table:adjusted_survival_contrast",
                        "figure:survival_curve",
                    ],
                    "method": "cox_proportional_hazards",
                    "icu_rule_refs": [],
                }
            ],
        }
    )

    with pytest.raises(PlannerArticleContractError) as excinfo:
        PlannerAgent(llm=object())._parse(
            plan_json,
            context,
            enforce_article_contract=True,
            article_contract_context=context,
        )

    message = str(excinfo.value)
    assert "temporal_absolute_risk" not in message
    assert "credited only on the primary lineage" not in message
    # A flag on every line is a flag on nothing: the roles still missing here
    # are the ordinary ones, and none of them is credited by lineage.
    assert "(headline_owned)" not in message

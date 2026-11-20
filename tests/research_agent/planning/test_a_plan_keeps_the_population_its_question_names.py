"""A plan keeps the population its question names (leg 3 and B25, S1713).

Leg 3: a frozen question asked about septic patients.  The Planner read every
input row as the population and called "septic patients" a definition, which
the host recorded as a warning only: the run would have answered another
population.  A definition that nothing in an all-row plan reads or selects is
now not covered, so the plan goes back.

B25: the exposure grouping phase split stays by the first day's maximum
lactate at 2 mmol/L, and the Planner's population restated the same words as
an inclusion criterion over the grouping's own variable; it could not apply,
and the card asked the researcher to accept the Planner's error.  A population
criterion that restates a staged grouping goes back to the Planner.

Synthetic contexts and plans only.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
)
from easyicu.research_agent.planning.progressive_compiler import _stated_population_spec
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveCohortIntent,
    ProgressivePlanCompileError,
)
from easyicu.research_agent.planning.question_requirements import (
    QuestionRequirement,
    concept_relatives,
    judge_question_requirements,
    question_requirement_findings,
)
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole

from .family_spec_fixtures import (
    _prediction_context,
    _prediction_payload,
    _request,
    _run,
)

_FEATURES = ["age", "sex", "hr_max", "lactate_max", "map_min"]


def _septic_context():
    base = _prediction_context()
    sepsis = ConceptDescriptor(
        name="sep3_sofa1",
        description="Sepsis-3 syndrome indicator",
        role=VariableRole.OTHER,
        dtype="int64",
        source_concept="sep3_sofa1",
    )
    return base.model_copy(
        update={
            "research_question": base.research_question + " in septic patients",
            "variables": [*base.variables, sepsis],
        }
    )


def _plan(context, **cohort: Any):
    import json

    request = _request(context, cohort_mode=None)
    _llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=_FEATURES))] * 3,
        required_primary_cohort_selection_mode=None,
    )
    plan = result.output
    if not cohort:
        return plan
    return plan.model_copy(
        update={
            "cohort": CohortDefinition(
                **{"name": "study", "selection_mode": "all_input_rows", **cohort}
            )
        }
    )


def _septic(concept: str = "sep3_sofa1") -> QuestionRequirement:
    return QuestionRequirement.model_validate(
        {
            "id": "r2",
            "kind": "definition",
            "quote": "septic patients",
            "concepts": [concept],
            "coverage": "definition_only",
            "gap": None,
            "note": "Defines the requested sepsis target population.",
        }
    )


def _judge(plan, context, concept: str = "sep3_sofa1"):
    return judge_question_requirements(
        [_septic(concept)],
        plan=plan,
        family_template=True,
        relatives=concept_relatives(context),
        check_gap=lambda gap: ("unverified", ""),
    )


def test_a_population_word_over_every_input_row_sends_the_plan_back() -> None:
    context = _septic_context()

    judged = _judge(_plan(context, inclusion=()), context)
    findings = question_requirement_findings(judged)

    assert [item.disposition for item in judged] == ["not_covered"]
    assert judged[0].host_judged is True
    assert [item.detail["reason"] for item in findings] == [
        "question_requirement_not_covered"
    ]
    assert "'septic patients'" in findings[0].message
    assert "no population condition selects on it" in findings[0].message


def _documented_los() -> ConceptPredicate:
    return ConceptPredicate.from_dict(
        {
            "concept_id": "los_icu",
            "time_window": {
                "anchor": "icu_admission",
                "start_offset_hours": 0,
                "end_offset_hours": 24,
            },
            "aggregation": "first",
            "op": ">",
            "value": 1,
        }
    )


def test_a_population_word_the_population_states_stays_a_definition() -> None:
    context = _septic_context()

    # The source already applied it; the all-row plan keeps the predicate as
    # documentation (B22's shape).
    documented = _judge(
        _plan(context, inclusion=(_documented_los(),)), context, concept="los_icu"
    )
    # A criterion the plan states but cannot apply is on the card as not
    # applied: the word is visible, not dropped.
    shown = _judge(
        _plan(context, unapplied_population_criteria=("septic patients",)), context
    )
    # A plan that selects rows by its own predicates is judged as before.
    selecting = _judge(_plan(context), context)

    assert [item.disposition for item in documented] == ["definition_only"]
    assert [item.disposition for item in shown] == ["definition_only"]
    assert [item.disposition for item in selecting] == ["definition_only"]


def _grouped_context():
    base = _prediction_context()
    grouping = {
        "variable": "lact_group_x1",
        "concept": "lact",
        "scale": "ordinal",
        "levels": [
            {"code": 1, "group": "g1", "label": "max lactate <= 2 mmol/L"},
            {"code": 2, "group": "g2", "label": "max lactate > 2 mmol/L"},
        ],
        "parameters": {
            "concept": "lact",
            "window": {"start_hours": 0.0, "end_hours": 24.0},
            "scale": "ordinal",
            "groups": [
                {
                    "id": "g1",
                    "rule": {"summary": "max", "op": "<=", "value": 2.0, "unit": "mmol/L"},
                },
                {"id": "g2", "rule": "otherwise"},
            ],
            "unmeasured": "exclude",
            "source_columns": {"max": "lact_max"},
        },
        "rule_wording_version": 2,
    }
    question = (
        "Is in-hospital mortality higher in adults (18 years or older) whose "
        "maximum lactate on the first ICU day is above 2 mmol/L?"
    )
    cohort = base.cohort.model_copy(
        update={
            "provenance": {
                **dict(base.cohort.provenance or {}),
                "exposure_group_labels": [grouping],
            }
        }
    )
    return base.model_copy(update={"research_question": question, "cohort": cohort})


_ADULTS = {
    "id": "c1",
    "quote": "adults (18 years or older)",
    "source": "question",
    "role": "include",
    "kind": "age_years",
    "min_years": 18.0,
    "max_years": None,
}
_ABOVE_TWO = "maximum lactate on the first ICU day is above 2 mmol/L"


def _intent(*criteria: dict[str, Any]) -> ProgressiveCohortIntent:
    return ProgressiveCohortIntent.model_validate(
        {
            "name": "study",
            "selection_mode": "all_input_rows",
            "population_spec": {"criteria": [_ADULTS, *criteria]},
        }
    )


@pytest.mark.parametrize(
    "criterion",
    [
        pytest.param(
            {
                "id": "c2",
                "quote": _ABOVE_TWO,
                "source": "question",
                "role": "include",
                "kind": "condition_present",
                "concepts_all_of": ["lact_group_x1"],
                "window": {"start_hours": 0.0, "end_hours": 24.0},
                "missing_policy": "exclude",
            },
            id="over-the-grouping-variable",
        ),
        pytest.param(
            {
                "id": "c2",
                "quote": _ABOVE_TWO,
                "source": "question",
                "role": "include",
                "kind": "measurement",
                "concept": "lact",
                "summary": "max",
                "window": {"start_hours": 0.0, "end_hours": 24.0},
                "op": ">",
                "value": 2.0,
                "unit": "mmol/L",
            },
            id="at-the-grouping-threshold",
        ),
    ],
)
def test_a_population_criterion_restating_the_grouping_goes_back(
    criterion: dict[str, Any],
) -> None:
    with pytest.raises(ProgressivePlanCompileError) as refused:
        _stated_population_spec(_intent(criterion), context=_grouped_context())

    assert refused.value.reason_code == "progressive_population_restates_exposure_grouping"
    assert "lact_group_x1" in str(refused.value)
    assert "Omit it" in str(refused.value)


def test_a_population_criterion_beside_the_grouping_stays() -> None:
    other_threshold = {
        "id": "c2",
        "quote": "maximum lactate on the first ICU day",
        "source": "question",
        "role": "include",
        "kind": "measurement",
        "concept": "lact",
        "summary": "max",
        "window": {"start_hours": 0.0, "end_hours": 24.0},
        "op": ">",
        "value": 0.5,
        "unit": "mmol/L",
    }

    spec = _stated_population_spec(_intent(other_threshold), context=_grouped_context())
    ungrouped = _stated_population_spec(_intent(), context=_prediction_context().model_copy(
        update={"research_question": _grouped_context().research_question}
    ))

    assert spec is not None and [item.id for item in spec.criteria] == ["c1", "c2"]
    assert ungrouped is not None and [item.id for item in ungrouped.criteria] == ["c1"]

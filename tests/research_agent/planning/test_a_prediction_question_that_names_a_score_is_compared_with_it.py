"""A prediction question that names an existing score is compared with it.

A question such as "how well do first-day vitals predict death, compared with
the APACHE IVa predicted mortality?" asks for a benchmark.  The prediction
template drafted no comparing step, so the requirement could only ever be a
capability gap.  The request now offers each roster column the concept
dictionary states as a probability or an oriented score; a benchmark
requirement the plan answers names one, and the template compares the model
with it on the same validation stays.  A benchmark is never also a predictor.
Synthetic contexts only.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.family_spec_planner import family_spec_user_prompt
from easyicu.research_agent.contracts.prediction_execution import (
    PREDICTION_BENCHMARK_PRODUCT,
    prediction_bootstrap_method,
    static_prediction_bootstrap_spec,
    static_prediction_owns_step,
)
from easyicu.research_agent.planning.family_spec.contract import (
    FamilySpecError,
    spec_from_mapping,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.question_requirements import (
    QuestionRequirement,
    concept_relatives,
    judge_question_requirements,
)
from easyicu.research_agent.schema import ConceptDescriptor, ResearchContext, VariableRole

from .family_spec_fixtures import (
    _prediction_context,
    _prediction_payload,
    _request,
    _run,
)

FEATURES = ["age", "sex", "hr_max", "lactate_max", "map_min"]
APACHE = "apache_iv_pred_hosp_mort"
QUOTE = "compared with the APACHE IVa predicted mortality"


def _context() -> ResearchContext:
    base = _prediction_context()
    return base.model_copy(
        update={
            "cohort": base.cohort.model_copy(update={"database": "eicu"}),
            "research_question": base.research_question.rstrip("?") + f", {QUOTE}?",
            "variables": [
                *base.variables,
                ConceptDescriptor(
                    name=APACHE, role=VariableRole.OTHER, dtype="float64",
                    source_concept=APACHE, analysis_window="icu_admission[0,24]h",
                ),
            ],
        }
    )


def _requirement(*concepts: str) -> dict:
    return {
        "id": "r1", "kind": "benchmark", "quote": QUOTE, "concepts": list(concepts),
        "coverage": "plan", "gap": None, "note": None,
    }


def _plan(context: ResearchContext, requirement: dict, features=FEATURES):
    request = _request(context, cohort_mode=None)
    payload = _prediction_payload(request, features=features)
    payload["question_requirements"] = [requirement]
    _llm, result = _run(
        context, [json.dumps(payload)], required_primary_cohort_selection_mode=None
    )
    return result.output


def _judged(context: ResearchContext, plan, requirement: dict):
    [judged] = judge_question_requirements(
        [QuestionRequirement.model_validate(requirement)],
        plan=plan,
        family_template=True,
        relatives=concept_relatives(context),
        check_gap=lambda gap: ("unverifiable", ""),
    )
    return judged


def test_the_request_offers_the_scores_the_dictionary_orients() -> None:
    context = _context()
    request = _request(context, cohort_mode=None)

    [candidate] = request.benchmark_candidates
    assert (candidate.name, candidate.concept, candidate.kind) == (APACHE, APACHE, "probability")
    prompt = family_spec_user_prompt(request, variable_descriptions={})
    assert "Benchmark comparators" in prompt and f'"name": "{APACHE}"' in prompt
    # A study without such a column offers none and keeps its request digest.
    plain = _request(_prediction_context(), cohort_mode=None)
    assert plain.benchmark_candidates == []
    assert "benchmark_candidates" not in plain.model_dump(mode="json", exclude_none=True)
    assert "Benchmark comparators" not in family_spec_user_prompt(plain, variable_descriptions={})


def test_a_named_benchmark_is_compared_on_the_same_validation_stays() -> None:
    context = _context()
    requirement = _requirement(APACHE)
    plan = _plan(context, requirement)
    steps = {step.step_id: step for step in plan.steps}

    comparison = steps["benchmark_comparison"]
    assert comparison.planned_analysis_role == "secondary"
    assert comparison.inputs == [APACHE, "artifact:analysis_cohort", "table:prediction_scores"]
    assert comparison.expected_outputs == [PREDICTION_BENCHMARK_PRODUCT]
    assert static_prediction_owns_step(comparison)
    assert PREDICTION_BENCHMARK_PRODUCT in steps["report"].inputs
    # The benchmark is not a predictor of the model it is compared with.
    assert APACHE not in steps["primary_performance"].inputs
    judged = _judged(context, plan, requirement)
    assert (judged.disposition, judged.owner_step_ids) == ("covered", ("benchmark_comparison",))


def test_a_benchmark_the_host_cannot_compare_stays_a_capability_gap() -> None:
    context = _context()
    # A vital sign is no existing score: the template draws no comparison.
    requirement = _requirement("hr_max")
    plan = _plan(context, requirement, features=["age", "sex", "lactate_max", "map_min"])

    assert "benchmark_comparison" not in {step.step_id for step in plan.steps}
    judged = _judged(context, plan, requirement)
    assert judged.disposition == "capability_gap"
    assert judged.reason_code == "question_benchmark_step_unavailable_in_family_template"


def test_a_plan_without_a_benchmark_is_unchanged() -> None:
    context = _context()
    request = _request(context, cohort_mode=None)
    _llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=FEATURES))],
        required_primary_cohort_selection_mode=None,
    )

    assert [step.step_id for step in result.output.steps] == [
        "cohort_accounting", "baseline_context", "measurement_audit", "primary_performance",
        "calibration_metrics", "internal_validation", "clinical_utility", "visualization",
        "report",
    ]


def test_template_declares_bootstrap_budget_seed_and_consumed_performance() -> None:
    plan = _plan(_context(), _requirement(APACHE))
    steps = {s.step_id: s for s in plan.steps}
    spec = static_prediction_bootstrap_spec(steps["primary_performance"])
    assert spec is not None
    assert spec.replicates == 200
    assert spec.seed == 1729
    assert steps["primary_performance"].method == prediction_bootstrap_method(spec)
    assert steps["internal_validation"].inputs == ["table:prediction_scores", "table:model_performance"]
    assert static_prediction_owns_step(steps["internal_validation"])


def test_source_unavailable_benchmark_is_declared_as_omitted_in_plan() -> None:
    from easyicu.databases.profiles import iter_database_profiles
    from easyicu.research_agent.concept_availability import explain_concept_availability

    context = _context()
    database = next(p.key for p in iter_database_profiles(public_only=True)
                    if not explain_concept_availability(concept=APACHE, database=p.key).available)
    context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"database": database})})
    request = _request(context, cohort_mode=None)
    assert request.benchmark_candidates == []
    assert request.benchmark_omissions[0]["reason"] == "benchmark_comparator_source_unavailable"
    plan = _plan(context, _requirement(APACHE))
    assert "benchmark_comparison" not in {s.step_id for s in plan.steps}
    report_intent = next(s.intent for s in plan.steps if s.step_id == "report")
    assert "benchmark_comparator_source_unavailable" in report_intent
    assert APACHE in report_intent
    assert database in report_intent
    assert request.benchmark_omissions[0]["availability_reason"] in report_intent


def test_a_benchmark_is_never_also_a_predictor() -> None:
    context = _context()
    request = _request(context, cohort_mode=None)
    payload = _prediction_payload(request, features=["age", "sex", "hr_max", APACHE])
    payload["question_requirements"] = [_requirement(APACHE)]

    with pytest.raises(FamilySpecError) as refused:
        validate_family_plan_spec(spec_from_mapping(payload), request)

    assert refused.value.reason_code == "family_spec_benchmark_used_as_predictor"
    assert refused.value.path == "feature_variables[3]"

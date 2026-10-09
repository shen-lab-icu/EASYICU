"""What a question explicitly asks of its plan is stated, held to its words, and judged.

A question can ask for more than its design: here, to compare the model it
builds with an existing severity score.  The family template has no step that
compares two models on the same rows, so a plan built from it used to answer
half the question and offer itself for approval.  The Planner now states each
such requirement with the question's words; the host holds the quote to the
question and the concepts to the run, refuses a spec that leaves a concept the
question names unaccounted for, and after compiling judges each requirement:

* a benchmark is answered only by a step that compares models on the same rows
  and reads the benchmark's column, never by reading that column as a
  predictor, and a family template's benchmark is a capability gap whatever
  the Planner claims;
* an unanswered requirement, or one the plan cannot answer, refuses approval,
  so an unattended run pauses on it;
* a gap the Planner declares is checked against the study: one the study
  contradicts goes back to the Planner, one the host cannot check stops the
  plan and is recorded as unverified;
* an estimand or another analysis is unanswered when no analysis step reads
  its concepts, and only attested when they do (reading a concept is not
  doing the analysis); one naming no concept is attested too;
* a concept stated only to define another element is shown, with what it
  defines, as a claim;
* the family route is a typed fact of the run, not a bookkeeping key;
* on the outline route, which states no requirements, every named concept
  other than a sealed coordinate is a visible warning naming the steps that
  read it, a benchmark read as a predictor included;
* a malformed record of the named concepts stops planning with its reason;
* the plan a review request offers, which the pipeline shaped after planning,
  is judged again from the planning record alone, in either direction, with
  the same bytes each time; judged again on the compiled plan, every row of
  the planning record comes back; a planning record that cannot be read, or
  is gone, refuses approval, and a defect in judging is not disguised as one.

Synthetic contexts and generic wording only.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.family_spec_planner import (
    FAMILY_SPEC_GUIDE,
    family_spec_response_shape,
    family_spec_structured_output_request,
    family_spec_user_prompt,
)
from easyicu.research_agent.orchestration.progressive_planning import (
    question_requirement_outcome,
    registered_stop,
)
from easyicu.research_agent.orchestration.workflow import human_review_requests_for_plan
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.planning.approval_stops import PLAN_APPROVAL_STOPS
from easyicu.research_agent.planning.capability_gap import check_capability_gap
from easyicu.research_agent.planning.family_spec.contract import (
    FamilySpecError,
    spec_from_mapping,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning import (
    question_requirements as question_requirements_owner,
)
from easyicu.research_agent.planning.question_requirements import (
    QUESTION_REQUIREMENT_STOP_CODES,
    QUESTION_REQUIREMENTS_FILENAME,
    QUESTION_REQUIREMENTS_GUIDE,
    QUESTION_REQUIREMENTS_REVIEW_FILENAME,
    UNREADABLE_REASON,
    NamedQuestionConcept,
    QuestionRequirement,
    analysis_plan_sha256,
    concept_relatives,
    judge_question_requirements,
    judge_recorded_requirements,
    outline_route_unstated,
    question_requirement_coverage,
    question_requirement_findings,
    question_requirements_on_plan_under_review,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.research_agent.planning.requirement_coverage import (
    PlanRequirementCoverageRecord,
)
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ResearchContext,
    ValidationFinding,
    VariableRole,
)

from .family_spec_fixtures import (
    _prediction_context,
    _prediction_payload,
    _request,
    _run,
)

FEATURES = ["age", "sex", "hr_max", "lactate_max", "map_min"]
_QUESTION = (
    "Among adult ICU stays, build a model from first-24-hour vitals and labs that "
    "predicts in-hospital mortality, and compare its discrimination and calibration "
    "with the admission severity score's predicted mortality."
)
_COMPARISON = (
    "compare its discrimination and calibration with the admission severity score's "
    "predicted mortality"
)
_BENCHMARK = {
    "id": "r1",
    "kind": "benchmark",
    "quote": _COMPARISON,
    "concepts": ["score_pred_mort"],
    "coverage": "plan",
    "gap": None,
    "note": None,
}
#: A gap the host has no typed evidence about.
_UNCHECKABLE = {
    "id": "r2",
    "kind": "estimand",
    "quote": "discrimination and calibration",
    "concepts": [],
    "coverage": "capability_gap",
    "gap": {
        "requirement": "estimand_unsupported",
        "concept": None,
        "element": "analysis",
        "detail": "This plan cannot estimate the named measure on held-out rows.",
    },
    "note": None,
}


def _context(*, named: bool = True) -> ResearchContext:
    """A prediction study whose question names an existing score's predicted risk."""

    base = _prediction_context()
    score = ConceptDescriptor(
        name="severity_score",
        description="admission severity score, first ICU day",
        role=VariableRole.OTHER,
        dtype="float64",
        source_concept="severity_score",
    )
    predicted = ConceptDescriptor(
        name="score_pred_mort",
        description="in-hospital mortality predicted by the admission severity score",
        role=VariableRole.OTHER,
        dtype="float64",
        source_concept="score_pred_mort",
        # The concept dictionary relates the predicted risk to its score.
        derived_from_concepts=["severity_score"],
    )
    constraints = json.loads(base.user_preferences.data_constraints or "{}")
    if named:
        constraints["question_named_concepts"] = [
            {"concepts": ["severity_score"], "evidence": "admission severity score"}
        ]
    return base.model_copy(
        update={
            "research_question": _QUESTION,
            "variables": [*base.variables, score, predicted],
            "user_preferences": base.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            ),
        }
    )


def _payload(request, requirements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_prediction_payload(request, features=FEATURES),
        "question_requirements": requirements,
    }


def _planned(context: ResearchContext, requirements: list[dict[str, Any]]):
    request = _request(context, cohort_mode=None)
    _llm, result = _run(
        context,
        [json.dumps(_payload(request, requirements))],
        required_primary_cohort_selection_mode=None,
    )
    return result


def _host_check(context: ResearchContext):
    return lambda gap: check_capability_gap(gap, context=context)


def _requirement(**changes: Any) -> QuestionRequirement:
    return QuestionRequirement.model_validate({**_BENCHMARK, **changes})


def _with_step(plan, step_id: str, **changes: Any):
    primary = next(
        item for item in plan.steps if item.planned_analysis_role == "primary"
    )
    added = primary.model_copy(update={"step_id": step_id, **changes})
    return plan.model_copy(update={"steps": [*plan.steps, added]})


# -- the request and the spec --------------------------------------------------


def test_the_request_names_each_concept_as_every_column_it_can_denote() -> None:
    request = _request(_context(), cohort_mode=None)

    (named,) = request.question_named_concepts
    # The dictionary relates the predicted risk to the score the question names.
    assert set(named.concepts) == {"severity_score", "score_pred_mort"}
    assert named.evidence == "admission severity score"


def test_a_question_that_names_nothing_keeps_the_request_identity() -> None:
    request = _request(_context(named=False), cohort_mode=None)

    assert request.question_named_concepts == []
    assert "question_named_concepts" not in request.model_dump(mode="json")


def test_the_planner_is_asked_for_the_requirements_and_given_the_names() -> None:
    request = _request(_context(), cohort_mode=None)
    schema = json.loads(family_spec_structured_output_request(request).schema_json)

    assert QUESTION_REQUIREMENTS_GUIDE in FAMILY_SPEC_GUIDE
    assert "question_requirements" in schema["required"]
    concept_enum = schema["properties"]["question_requirements"]["items"]["properties"][
        "concepts"
    ]["items"]["enum"]
    assert {"severity_score", "score_pred_mort"} <= set(concept_enum)
    assert '"question_requirements"' in family_spec_response_shape(request)
    prompt = family_spec_user_prompt(request, variable_descriptions={})
    assert "admission severity score" in prompt


@pytest.mark.parametrize(
    ("change", "code"),
    [
        pytest.param(
            {"quote": "compare it with a published model"},
            "question_requirement_quote_not_in_question",
            id="quote-not-in-the-question",
        ),
        pytest.param(
            {"concepts": ["not_a_column"]},
            "question_requirement_concept_unknown",
            id="concept-not-offered",
        ),
    ],
)
def test_a_requirement_is_held_to_the_question_and_the_run(
    change: dict[str, Any], code: str
) -> None:
    request = _request(_context(), cohort_mode=None)
    spec = spec_from_mapping(_payload(request, [{**_BENCHMARK, **change}]))

    with pytest.raises(FamilySpecError) as refused:
        validate_family_plan_spec(spec, request)

    assert refused.value.reason_code == code


def test_a_named_concept_no_requirement_reads_refuses_the_spec() -> None:
    request = _request(_context(), cohort_mode=None)
    spec = spec_from_mapping(_payload(request, []))

    with pytest.raises(FamilySpecError) as refused:
        validate_family_plan_spec(spec, request)

    assert refused.value.reason_code == "question_named_concept_unaccounted"
    assert "admission severity score" in str(refused.value)


@pytest.mark.parametrize(
    "requirement",
    [
        pytest.param(_BENCHMARK, id="the-related-column"),
        pytest.param(
            {
                **_BENCHMARK,
                "kind": "definition",
                "concepts": ["severity_score"],
                "coverage": "definition_only",
                "note": "Names the score whose predicted risk is compared.",
            },
            id="only-a-definition",
        ),
    ],
)
def test_a_requirement_reading_the_name_or_its_relative_accounts_for_it(
    requirement: dict[str, Any],
) -> None:
    request = _request(_context(), cohort_mode=None)

    validate_family_plan_spec(
        spec_from_mapping(_payload(request, [requirement])), request
    )


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"coverage": "capability_gap"}, id="gap-claimed-without-one"),
        pytest.param(
            {
                "gap": {
                    "requirement": "design_element_unsupported",
                    "concept": None,
                    "element": "comparison",
                    "detail": "No step compares two models.",
                }
            },
            id="gap-without-claiming-one",
        ),
        pytest.param(
            {
                "coverage": "capability_gap",
                "gap": {"requirement": "design_element_unsupported"},
            },
            id="gap-without-its-element-or-why",
        ),
        pytest.param({"kind": "definition"}, id="definition-answered-by-the-plan"),
        pytest.param(
            {"kind": "definition", "coverage": "definition_only"},
            id="definition-without-what-it-defines",
        ),
        pytest.param({"concepts": []}, id="benchmark-answered-without-its-concept"),
        pytest.param(
            {"concepts": ["score_pred_mort", "score_pred_mort"]}, id="repeated-concept"
        ),
    ],
)
def test_a_requirement_states_its_coverage_consistently(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        _requirement(**changes)


# -- the compiled plan ----------------------------------------------------------


def test_a_benchmark_the_family_template_cannot_compare_refuses_approval(
    tmp_path: Path,
) -> None:
    context = _context()
    result = _planned(context, [_BENCHMARK])

    findings = question_requirement_outcome(
        context=context, plan=result.output, facts=result.facts, run_dir=tmp_path
    )

    (stop,) = [item for item in findings if item.severity == "error"]
    assert stop.detail["reason"] == "question_requirement_capability_gap"
    assert stop.detail["approval_allowed"] is False
    (row,) = stop.detail["requirements"]
    # The Planner claimed the plan answers it; the host decided otherwise.
    assert (row["claimed_coverage"], row["disposition"], row["host_judged"]) == (
        "plan",
        "capability_gap",
        True,
    )
    # The host's gap, verified by the method sets, not a claim.
    assert row["gap"] == {
        "requirement": "design_element_unsupported",
        "concept": None,
        "element": "comparison",
        "detail": row["gap"]["detail"],
    }
    assert (row["gap_verification"], row["verified_by_host"]) == ("verified", True)
    assert "prediction.delong_ci" in row["gap_fact"]
    # The stop names the requirement; the gap's own sentence ends the line.
    assert row["gap"]["detail"].endswith(".")
    assert stop.message.endswith(
        f"r1 {_COMPARISON!r} (benchmark): {row['gap']['detail']}"
    )
    record = json.loads((tmp_path / "question_requirements.json").read_text("utf-8"))
    assert record["route"] == "family_template"
    assert record["compiled_plan_sha256"] == analysis_plan_sha256(result.output)
    # What the judgment read of the study travels with it.
    assert record["denoted_columns"] == {
        "score_pred_mort": ["score_pred_mort", "severity_score"]
    }
    assert record["sealed"] == [
        value for value in (context.primary_exposure, context.target_outcome) if value
    ]
    assert [
        (item["kind"], item["question_kind"], item["status"])
        for item in record["coverage"]["records"]
    ] == [("question", "benchmark", "unsupported")]


@pytest.mark.parametrize(
    "require_plan_review", [True, False], ids=["reviewed", "unattended"]
)
def test_a_run_pauses_on_a_requirement_it_cannot_answer(
    tmp_path: Path, require_plan_review: bool
) -> None:
    context = _context()
    result = _planned(context, [_BENCHMARK])
    findings = question_requirement_outcome(
        context=context, plan=result.output, facts=result.facts, run_dir=tmp_path
    )

    requests = human_review_requests_for_plan(
        findings=findings, plan=result.output, require_plan_review=require_plan_review
    )

    assert [
        (item.kind, item.payload["reason"], item.payload["approval_allowed"])
        for item in requests
    ] == [("scientific_stop", "question_requirement_capability_gap", False)]


def test_a_declared_gap_the_study_contradicts_goes_back_to_the_planner() -> None:
    context = _context()
    request = _request(context, cohort_mode=None)
    contradicted = {
        **_UNCHECKABLE,
        "gap": {
            "requirement": "levels_from_thresholds_unavailable",
            "concept": "not_a_column",
            "element": "exposure",
            "detail": "This plan cannot group the named measure by thresholds.",
        },
    }

    llm, result = _run(
        context,
        [
            json.dumps(_payload(request, [_BENCHMARK, contradicted])),
            json.dumps(_payload(request, [_BENCHMARK])),
        ],
        required_primary_cohort_selection_mode=None,
    )

    assert len(llm.calls) == 2
    retry = " ".join(message.content for message in llm.calls[1][0])
    assert "progressive_capability_gap_claim_unverified" in retry
    assert "is not a variable of this study" in retry
    assert [item.id for item in result.facts.question_requirements] == ["r1"]


def test_a_gap_the_host_cannot_check_stops_the_plan_as_unverified(
    tmp_path: Path,
) -> None:
    context = _context()
    result = _planned(context, [_BENCHMARK, _UNCHECKABLE])

    findings = question_requirement_outcome(
        context=context, plan=result.output, facts=result.facts, run_dir=tmp_path
    )

    (stop,) = [item for item in findings if item.severity == "error"]
    rows = {row["id"]: row for row in stop.detail["requirements"]}
    assert (rows["r2"]["disposition"], rows["r2"]["host_judged"]) == (
        "capability_gap",
        False,
    )
    assert (rows["r2"]["gap_verification"], rows["r2"]["verified_by_host"]) == (
        "unverifiable",
        False,
    )
    # Each requirement's line ends with one full stop, its gap's own.
    assert stop.message.endswith(
        f"r2 {_UNCHECKABLE['quote']!r} (estimand): {_UNCHECKABLE['gap']['detail']}"
    )
    assert ".. r2" not in stop.message


@pytest.mark.parametrize(
    ("family_template", "disposition"),
    [(False, "not_covered"), (True, "capability_gap")],
)
def test_a_benchmark_read_as_a_predictor_answers_nothing(
    family_template: bool, disposition: str
) -> None:
    context = _context()
    plan = _planned(context, [_BENCHMARK]).output
    # The primary model reads the benchmark's column as one more predictor.
    plan = plan.model_copy(
        update={
            "steps": [
                item.model_copy(update={"inputs": [*item.inputs, "score_pred_mort"]})
                if item.planned_analysis_role == "primary"
                else item
                for item in plan.steps
            ]
        }
    )

    (judged,) = judge_question_requirements(
        [_requirement()],
        plan=plan,
        family_template=family_template,
        relatives=concept_relatives(context),
        check_gap=_host_check(context),
    )

    assert judged.disposition == disposition
    assert judged.owner_step_ids == ()


@pytest.mark.parametrize("concept", ["score_pred_mort", "severity_score"])
def test_a_step_comparing_models_on_the_same_rows_answers_the_benchmark(
    concept: str,
) -> None:
    context = _context()
    plan = _with_step(
        _planned(context, [_BENCHMARK]).output,
        "benchmark_discrimination",
        scientific_action_id="prediction.delong_ci",
        planned_analysis_role="secondary",
        inputs=["table:prediction_scores", "score_pred_mort"],
    )

    (judged,) = judge_question_requirements(
        [_requirement(concepts=[concept])],
        plan=plan,
        family_template=True,
        relatives=concept_relatives(context),
        check_gap=_host_check(context),
    )

    assert (judged.disposition, judged.owner_step_ids) == (
        "covered",
        ("benchmark_discrimination",),
    )
    coverage = question_requirement_coverage([judged], context=context, plan=plan)
    assert coverage.complete is True


def test_an_analysis_the_plan_does_not_read_is_not_covered() -> None:
    context = _context()
    # A Table 1 that reads the concept answers no analysis of it.
    plan = _with_step(
        _planned(context, [_BENCHMARK]).output,
        "descriptive_table",
        planned_analysis_role="auxiliary",
        scientific_action_id=None,
        inputs=["artifact:analysis_cohort", "severity_score"],
    )
    requirement = _requirement(
        id="r2",
        kind="analysis",
        quote="from first-24-hour vitals and labs",
        concepts=["severity_score"],
    )

    (judged,) = judge_question_requirements(
        [requirement],
        plan=plan,
        family_template=True,
        relatives=concept_relatives(context),
        check_gap=_host_check(context),
    )
    findings = question_requirement_findings([judged])

    assert judged.disposition == "not_covered"
    assert [(item.severity, item.detail["reason"]) for item in findings] == [
        ("error", "question_requirement_not_covered")
    ]


def test_a_claim_the_host_cannot_verify_is_shown_as_attested() -> None:
    context = _context()
    plan = _planned(context, [_BENCHMARK]).output
    estimand = _requirement(
        id="r2",
        kind="estimand",
        quote="discrimination and calibration",
        concepts=[],
        note="The primary step reports discrimination and calibration.",
    )
    definition = _requirement(
        id="r3",
        kind="definition",
        quote="admission severity score",
        concepts=["severity_score"],
        coverage="definition_only",
        note="Names the score whose predicted risk is compared.",
    )

    judged = judge_question_requirements(
        [estimand, definition],
        plan=plan,
        family_template=True,
        relatives=concept_relatives(context),
        check_gap=_host_check(context),
    )
    findings = question_requirement_findings(judged)

    assert [item.disposition for item in judged] == ["attested", "definition_only"]
    assert [
        (item.severity, item.detail["reason_code"], item.detail["verified_by_host"])
        for item in findings
    ] == [
        ("warning", "question_requirement_attested", False),
        ("warning", "question_requirement_definition_stated", False),
    ]
    assert "Names the score whose predicted risk is compared." in findings[1].message
    assert human_review_requests_for_plan(findings=findings, plan=plan) == ()


def test_an_analysis_whose_concepts_the_plan_reads_is_attested_not_verified() -> None:
    context = _context()
    plan = _planned(context, [_BENCHMARK]).output
    requirement = _requirement(
        id="r2",
        kind="analysis",
        quote="from first-24-hour vitals and labs",
        concepts=["hr_max", "lactate_max"],
    )

    (judged,) = judge_question_requirements(
        [requirement],
        plan=plan,
        family_template=True,
        relatives=concept_relatives(context),
        check_gap=_host_check(context),
    )

    # Reading the concepts is not doing the analysis the question names.
    primary = next(
        item.step_id for item in plan.steps if item.planned_analysis_role == "primary"
    )
    assert (judged.disposition, judged.verified_by_host) == ("attested", False)
    assert (judged.owner_step_ids, primary in judged.reading_step_ids) == ((), True)
    assert (
        question_requirement_coverage([judged], context=context, plan=plan).records
        == ()
    )
    (shown,) = question_requirement_findings([judged])
    assert (shown.severity, shown.detail["reason_code"]) == (
        "warning",
        "question_requirement_attested",
    )
    assert primary in shown.message


@pytest.mark.parametrize(
    "read_as_predictor", [False, True], ids=["unread", "a-predictor"]
)
def test_the_outline_route_names_every_concept_it_does_not_state(
    read_as_predictor: bool,
) -> None:
    context = _context()
    plan = _planned(context, [_BENCHMARK]).output
    primary = next(
        item.step_id for item in plan.steps if item.planned_analysis_role == "primary"
    )
    if read_as_predictor:
        plan = plan.model_copy(
            update={
                "steps": [
                    item.model_copy(
                        update={"inputs": [*item.inputs, "score_pred_mort"]}
                    )
                    if item.step_id == primary
                    else item
                    for item in plan.steps
                ]
            }
        )
    named = [
        NamedQuestionConcept(
            concepts=["severity_score", "score_pred_mort"],
            evidence="admission severity score",
        )
    ]

    unstated = outline_route_unstated(named, plan=plan, sealed=("", "death"))
    findings = question_requirement_findings((), unstated=unstated)

    # Read or not, what the question asks of it is left to review.
    (item,) = unstated
    assert item.reading_step_ids == ((primary,) if read_as_predictor else ())
    assert [(row.severity, row.detail["reason_code"]) for row in findings] == [
        ("warning", "question_named_concept_unstated")
    ]
    assert "admission severity score" in findings[0].message
    assert human_review_requests_for_plan(findings=findings, plan=plan) == ()
    # A sealed coordinate the name denotes accounts for it.
    assert outline_route_unstated(named, plan=plan, sealed=("score_pred_mort",)) == ()


def test_the_family_route_is_a_typed_fact_of_the_run(tmp_path: Path) -> None:
    context = _context()
    result = _planned(context, [_BENCHMARK])

    assert result.facts.family_template is True
    # Read as the outline route's, the same plan is judged without the
    # template's gap: the host's decision rests on the typed fact alone.
    outline = dataclasses.replace(result.facts, family_template=False)
    findings = question_requirement_outcome(
        context=context, plan=result.output, facts=outline, run_dir=tmp_path
    )

    record = json.loads((tmp_path / "question_requirements.json").read_text("utf-8"))
    assert record["route"] == "outline"
    assert [(item.severity, item.detail.get("reason")) for item in findings][0] == (
        "error",
        "question_requirement_not_covered",
    )


@pytest.mark.parametrize(
    "entries",
    ["severity_score", [{"concepts": [], "evidence": "admission severity score"}]],
    ids=["not-a-list", "an-empty-name"],
)
def test_a_malformed_record_of_named_concepts_stops_planning_with_its_reason(
    tmp_path: Path, entries: Any
) -> None:
    context = _context(named=False)
    plan = _planned(context, [])
    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    constraints["question_named_concepts"] = entries
    malformed = context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            )
        }
    )

    with pytest.raises(FamilySpecError) as refused:
        _request(malformed, cohort_mode=None)
    with pytest.raises(ProgressivePlanCompileError) as stopped:
        question_requirement_outcome(
            context=malformed, plan=plan.output, facts=plan.facts, run_dir=tmp_path
        )

    assert refused.value.reason_code == "question_named_concepts_malformed"
    assert stopped.value.reason_code == "progressive_question_named_concepts_malformed"


def test_a_question_coverage_row_states_what_the_question_asks_for() -> None:
    with pytest.raises(ValidationError):
        PlanRequirementCoverageRecord(
            requirement_id="question:r1:score_pred_mort",
            kind="question",
            concept_identity="score_pred_mort",
            source_ref="research_context.json.research_question",
            status="missing",
            reason_code="question_requirement_not_covered",
        )
    with pytest.raises(ValidationError):
        PlanRequirementCoverageRecord(
            requirement_id="outcome:death",
            kind="outcome",
            question_kind="benchmark",
            concept_identity="death",
            source_ref="research_context.json.cohort.requested_outcome_columns",
            status="covered",
            owner_step_ids=("primary",),
            reason_code="typed_plan_owner_present",
        )


def test_an_approval_stop_reaches_every_reader_only_once_registered() -> None:
    def stop(reason: str) -> ValidationFinding:
        return ValidationFinding(
            validator="question_requirements",
            severity="error",
            message="This plan cannot be approved.",
            detail={"reason": reason, "approval_allowed": False},
        )

    for reason in PLAN_APPROVAL_STOPS:
        assert registered_stop(stop(reason)).detail["reason"] == reason
    with pytest.raises(ProgressivePlanCompileError) as refused:
        registered_stop(stop("an_unregistered_stop"))
    assert refused.value.reason_code == "progressive_approval_stop_unregistered"


# -- the plan offered for review ---------------------------------------------------


def _benchmark_step(plan, *, reads: str = "score_pred_mort"):
    return _with_step(
        plan,
        "benchmark_discrimination",
        scientific_action_id="prediction.delong_ci",
        planned_analysis_role="secondary",
        inputs=["table:prediction_scores", reads],
    )


def _compares_with_the_score(plan):
    # The step reads the score itself, a relative of its predicted risk.
    return _benchmark_step(plan, reads="severity_score")


def _without_benchmark_step(plan):
    return plan.model_copy(
        update={
            "steps": [
                item
                for item in plan.steps
                if item.step_id != "benchmark_discrimination"
            ]
        }
    )


def _planning(tmp_path: Path, requirements: list[dict[str, Any]], *, compiled=None):
    """The plan phase: the requirements judged on the compiled plan and recorded."""

    context = _context()
    result = _planned(context, requirements)
    plan = compiled(result.output) if compiled else result.output
    findings = question_requirement_outcome(
        context=context, plan=plan, facts=result.facts, run_dir=tmp_path
    )
    return plan, findings


def _question_stops(findings) -> list[str]:
    return [
        item.detail["reason"]
        for item in findings
        if item.validator == "question_requirements" and item.severity == "error"
    ]


@pytest.mark.parametrize(
    ("compiled", "shaped", "planned_stops", "reviewed_stops", "disposition"),
    [
        # The pipeline drops the comparing step after planning (the step cap).
        pytest.param(
            _benchmark_step,
            _without_benchmark_step,
            [],
            ["question_requirement_capability_gap"],
            "capability_gap",
            id="answered-then-dropped",
        ),
        # The pipeline adds a comparing step after planning.
        pytest.param(
            None,
            _compares_with_the_score,
            ["question_requirement_capability_gap"],
            [],
            "covered",
            id="unanswered-then-answered",
        ),
    ],
)
def test_the_plan_offered_for_review_is_judged_again_in_either_direction(
    tmp_path: Path,
    compiled,
    shaped,
    planned_stops: list[str],
    reviewed_stops: list[str],
    disposition: str,
) -> None:
    plan, findings = _planning(tmp_path, [_BENCHMARK], compiled=compiled)
    assert _question_stops(findings) == planned_stops
    planning_record = (tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_bytes()
    under_review = shaped(plan)

    reviewed = question_requirements_on_plan_under_review(
        findings, plan=under_review, run_dir=tmp_path
    )

    assert _question_stops(reviewed) == reviewed_stops
    record = json.loads((tmp_path / QUESTION_REQUIREMENTS_REVIEW_FILENAME).read_text())
    # Bound to the digest a review request's authority binds.
    assert record["plan_sha256"] == analysis_plan_sha256(under_review)
    assert (
        record["plan_sha256"]
        == PlanReviewAuthority.create(plan=under_review).plan_sha256
    )
    assert [row["disposition"] for row in record["judged"]] == [disposition]
    # The planning record stays as written.
    assert (tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_bytes() == planning_record


@pytest.mark.parametrize(
    "require_plan_review", [True, False], ids=["reviewed", "unattended"]
)
def test_the_requests_offer_the_judgment_of_the_plan_they_bind(
    tmp_path: Path, require_plan_review: bool
) -> None:
    plan, findings = _planning(tmp_path, [_BENCHMARK], compiled=_benchmark_step)
    under_review = _without_benchmark_step(plan)

    requests = human_review_requests_for_plan(
        findings=findings,
        plan=under_review,
        evidence=SimpleNamespace(root=tmp_path, records=lambda: ()),
        require_plan_review=require_plan_review,
    )

    (stop,) = requests
    assert (stop.payload["reason"], stop.payload["approval_allowed"]) == (
        "question_requirement_capability_gap",
        False,
    )
    record = json.loads((tmp_path / QUESTION_REQUIREMENTS_REVIEW_FILENAME).read_text())
    assert record["plan_sha256"] == stop.payload["plan_review_authority"]["plan_sha256"]


def test_judging_the_plan_under_review_again_writes_the_same_bytes(
    tmp_path: Path,
) -> None:
    plan, findings = _planning(tmp_path, [_BENCHMARK, _UNCHECKABLE])
    under_review = _benchmark_step(plan)

    first = question_requirements_on_plan_under_review(
        findings, plan=under_review, run_dir=tmp_path
    )
    written = (tmp_path / QUESTION_REQUIREMENTS_REVIEW_FILENAME).read_bytes()
    # On resume the requests are derived again, from the plan phase's findings
    # or from these.
    for given in (findings, first):
        again = question_requirements_on_plan_under_review(
            given, plan=under_review, run_dir=tmp_path
        )
        assert (
            tmp_path / QUESTION_REQUIREMENTS_REVIEW_FILENAME
        ).read_bytes() == written
        assert [item.model_dump() for item in again] == [
            item.model_dump() for item in first
        ]
    # A declared gap keeps the verdict it was given: a check of the study.
    (stop,) = [item for item in first if item.severity == "error"]
    ((row_id, verification),) = [
        (row["id"], row["gap_verification"]) for row in stop.detail["requirements"]
    ]
    assert (row_id, verification) == ("r2", "unverifiable")


def test_the_plan_under_review_is_judged_from_the_record_alone(
    tmp_path: Path,
) -> None:
    plan, findings = _planning(tmp_path, [_BENCHMARK])
    path = tmp_path / QUESTION_REQUIREMENTS_FILENAME
    record = json.loads(path.read_text(encoding="utf-8"))
    # The record, not the study, says which columns the concept denotes.
    record["denoted_columns"] = {"score_pred_mort": ["score_pred_mort"]}
    path.write_text(json.dumps(record), encoding="utf-8")

    reviewed = question_requirements_on_plan_under_review(
        findings,
        plan=_benchmark_step(plan, reads="severity_score"),
        run_dir=tmp_path,
    )

    assert _question_stops(reviewed) == ["question_requirement_capability_gap"]


def _break_record(record: dict[str, Any], how: str) -> Any:
    if how == "another-schema":
        return {**record, "schema_version": "other/1"}
    if how == "no-columns-for-a-concept":
        return {**record, "denoted_columns": {}}
    if how == "a-study-digest-that-is-not-one":
        return {**record, "coverage": {**record["coverage"], "context_sha256": "x"}}
    if how == "a-gap-without-its-check":
        return {
            **record,
            "judged": [{**row, "gap_verification": None} for row in record["judged"]],
        }
    return ["not", "a", "record"]


@pytest.mark.parametrize(
    "how",
    [
        "not-json",
        "not-an-object",
        "another-schema",
        "no-columns-for-a-concept",
        "a-gap-without-its-check",
        "a-study-digest-that-is-not-one",
    ],
)
def test_a_planning_record_that_cannot_be_read_refuses_approval(
    tmp_path: Path, how: str
) -> None:
    plan, findings = _planning(tmp_path, [_BENCHMARK, _UNCHECKABLE])
    path = tmp_path / QUESTION_REQUIREMENTS_FILENAME
    if how == "not-json":
        path.write_text("{", encoding="utf-8")
    else:
        record = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(_break_record(record, how)), encoding="utf-8")

    reviewed = question_requirements_on_plan_under_review(
        findings, plan=_benchmark_step(plan), run_dir=tmp_path
    )

    assert _question_stops(reviewed) == [UNREADABLE_REASON]
    (stop,) = [item for item in reviewed if item.validator == "question_requirements"]
    assert stop.detail["approval_allowed"] is False
    assert UNREADABLE_REASON in QUESTION_REQUIREMENT_STOP_CODES
    assert UNREADABLE_REASON in PLAN_APPROVAL_STOPS
    assert not (tmp_path / QUESTION_REQUIREMENTS_REVIEW_FILENAME).exists()


def test_without_a_planning_record_the_findings_stand(tmp_path: Path) -> None:
    findings = [
        ValidationFinding(
            validator="another_owner",
            severity="warning",
            message="Another owner's note.",
        )
    ]

    assert (
        question_requirements_on_plan_under_review(
            findings, plan=_planned(_context(), [_BENCHMARK]).output, run_dir=tmp_path
        )
        == findings
    )
    assert not (tmp_path / QUESTION_REQUIREMENTS_REVIEW_FILENAME).exists()


@pytest.mark.parametrize("family_template", [True, False], ids=["family", "outline"])
def test_judging_the_compiled_plan_again_reproduces_the_planning_record(
    tmp_path: Path, family_template: bool
) -> None:
    # The record carries everything the judgment read of the study: judged
    # again on the very plan it judged, every row comes back the same.  The
    # question names its outcome too, a sealed coordinate.
    context = _context()
    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    constraints["question_named_concepts"].append(
        {"concepts": [context.target_outcome], "evidence": "in-hospital mortality"}
    )
    context = context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            )
        }
    )
    result = _planned(context, [_BENCHMARK])
    plan = _benchmark_step(result.output)
    requirements = (
        _requirement(),
        QuestionRequirement.model_validate(_UNCHECKABLE),
        _requirement(id="r3", kind="analysis", concepts=["readmit_flag"]),
        _requirement(id="r4", kind="estimand", concepts=["lactate_max"]),
        _requirement(
            id="r5",
            kind="definition",
            coverage="definition_only",
            concepts=["age"],
            note="It defines the adult population.",
        ),
        _requirement(id="r6", kind="subgroup", concepts=["sex"]),
    )
    facts = dataclasses.replace(
        result.facts,
        question_requirements=requirements,
        family_template=family_template,
    )
    findings = question_requirement_outcome(
        context=context, plan=plan, facts=facts, run_dir=tmp_path
    )
    planning = json.loads((tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_text())

    judgment = judge_recorded_requirements(planning, plan=plan)

    for key in ("judged", "unstated", "coverage"):
        assert judgment.record[key] == planning[key]
    assert [item.model_dump() for item in judgment.findings()] == [
        item.model_dump() for item in findings
    ]
    assert {row["disposition"] for row in planning["judged"]} == {
        "covered",
        "capability_gap",
        "not_covered",
        "attested",
        "definition_only",
    }
    assert [row["concepts"] for row in planning["unstated"]] == (
        [] if family_template else [["score_pred_mort", "severity_score"]]
    )


def test_a_judgment_whose_planning_record_is_gone_refuses_approval(
    tmp_path: Path,
) -> None:
    plan, findings = _planning(tmp_path, [_BENCHMARK])
    (tmp_path / QUESTION_REQUIREMENTS_FILENAME).unlink()

    reviewed = question_requirements_on_plan_under_review(
        findings, plan=_benchmark_step(plan), run_dir=tmp_path
    )

    assert _question_stops(reviewed) == [UNREADABLE_REASON]


def test_a_defect_in_judging_is_not_reported_as_an_unreadable_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, findings = _planning(tmp_path, [_BENCHMARK])

    def defective(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("a defect in judging")

    monkeypatch.setattr(
        question_requirements_owner, "judge_question_requirements", defective
    )

    # A fresh plan would meet the same defect: it is raised, not disguised.
    with pytest.raises(ValueError, match="a defect in judging"):
        question_requirements_on_plan_under_review(
            findings, plan=plan, run_dir=tmp_path
        )

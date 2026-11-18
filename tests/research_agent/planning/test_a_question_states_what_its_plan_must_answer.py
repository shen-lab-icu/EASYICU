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
    _question_requirement_schema_feedback,
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
    AnalysisPlan,
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


@pytest.mark.parametrize(("original", "quote"), [
    ("比较 ABC 指标（１２分），再报告结果。", "比较abc指标(12分),再报告结果."),
    ("Compare “Cafe\u0301 Index”\n  （１２ points）…", 'compare "Café Index" (12 points)...'),
    ("Compare mean x\u0304", "compare mean x\u0304"),
])
def test_normalized_quote_keeps_exact_question_text_and_offsets(tmp_path, original, quote):
    question = f"  {_QUESTION}\n{original} Confirm the result.  "
    context = _context(named=False).model_copy(update={"research_question": question})
    claim = {**_BENCHMARK, "kind": "analysis", "concepts": [], "quote": quote}
    result = _planned(context, [claim])

    question_requirement_outcome(
        context=context, plan=result.output, facts=result.facts, run_dir=tmp_path,
    )

    record = json.loads((tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_text())
    (row,) = record["judged"]
    assert record["stated"][0]["quote"] == quote
    assert row["quote"] == original
    start = question.index(original)
    assert row["quote_span"] == {"start": start, "end": start + len(original)}
    replayed = judge_recorded_requirements(record, plan=result.output)
    assert replayed.record["judged"] == [row]

    legacy_row = {key: value for key, value in row.items() if key != "quote_span"}
    legacy_row["quote"] = quote
    legacy = judge_recorded_requirements({**record, "judged": [legacy_row]}, plan=result.output)
    assert legacy.record["judged"] == [legacy_row]


@pytest.mark.parametrize(("question", "quote"), [
    ("In chronic kidney disease, report HR", "HR"),
    ("For sepsis patients, report OR", "OR"),
    ("Assess dosage and age", "age"),
    ("Compare 124 against 241 and report 24", "24"),
])
def test_judged_quote_skips_matches_inside_ascii_runs(tmp_path, question, quote):
    context = _context(named=False).model_copy(update={"research_question": question})
    claim = {**_BENCHMARK, "kind": "analysis", "concepts": [], "quote": quote}
    result = _planned(context, [claim])
    question_requirement_outcome(
        context=context, plan=result.output, facts=result.facts, run_dir=tmp_path,
    )

    record = json.loads((tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_text())
    (row,) = record["judged"]
    start = question.rindex(quote)
    assert row["quote_span"] == {"start": start, "end": start + len(quote)}
    assert row["quote"] == quote


@pytest.mark.parametrize(("question", "quote"), [
    ("比较 ABC 指标（１２分）与观察时长。", "比较 ABC 得分（１２分）"),
    ("比较 ABC 指标（１２分）与观察时长。", "观察时长与 ABC 指标"),
    ("Compare North Bay and South Bay (１２ points).", "North Bay and South Bay (13 points)"),
    ("Compare North Bay and South Bay (１２ points).", "NorthBay and South Bay"),
    ("Compare North Bay and South Bay (１２ points).", "North Bay ... South Bay"),
    ("Compare the scores…", "the scores."),
    ("Compare North Bay and South Bay.", '"North Bay"'),
    ("Compare North Bay and South Bay.", "   "),
    ("For sepsis patients, compare mortality", "OR"),
    ("Compare 240 and 124 hours", "24"),
    ("Compare mean x\u0304 and SD", "Compare mean x"),
    ("Compare p\u0302 and SD", "\u0302 and SD"),
])
def test_quote_must_be_one_contiguous_source_span(question, quote):
    context = _context(named=False).model_copy(update={"research_question": question})
    request = _request(context, cohort_mode=None)
    claim = {**_BENCHMARK, "kind": "analysis", "concepts": [], "quote": quote}
    spec = spec_from_mapping(_payload(request, [claim]))

    with pytest.raises(FamilySpecError) as refused:
        validate_family_plan_spec(spec, request)
    assert refused.value.reason_code == "question_requirement_quote_not_in_question"
    assert refused.value.path == "question_requirements[0].quote"


def test_family_retry_names_allowed_keys_and_all_rejected_quote_paths():
    context = _context(named=False)
    request = _request(context, cohort_mode=None)
    first = _BENCHMARK
    second = {
        **_BENCHMARK, "id": "r2", "kind": "analysis", "concepts": [],
        "quote": "first-24-hour vitals and labs",
    }
    extra_gap = {**_UNCHECKABLE, "gap": {**_UNCHECKABLE["gap"], "extra": True}}
    wrong_first = "compare to another score"
    wrong_second = "labs and vitals across two days"
    llm, result = _run(
        context,
        [
            json.dumps(_payload(request, [
                {**first, "quote_span": {"start": 1, "end": 2}}, extra_gap,
            ])),
            json.dumps(_payload(request, [{**first, "quote": wrong_first}, {**second, "quote": wrong_second}])),
            json.dumps(_payload(request, [first, second])),
        ],
        required_primary_cohort_selection_mode=None,
    )

    assert len(llm.calls) == 3
    schema = json.loads(family_spec_structured_output_request(request).schema_json)
    item_schema = schema["properties"]["question_requirements"]["items"]
    assert item_schema["additionalProperties"] is False
    assert "quote_span" not in item_schema["properties"]
    extra_feedback = llm.calls[1][0][-1].content
    allowed_keys = ", ".join(sorted(item_schema["properties"]))
    assert (
        f"question_requirements.0.quote_span: Remove the extra key. Allowed keys: {allowed_keys}."
        in extra_feedback
    )
    gap_schema = item_schema["properties"]["gap"]["anyOf"][0]
    gap_keys = ", ".join(sorted(gap_schema["properties"]))
    assert (
        f"question_requirements.1.gap.extra: Remove the extra key. Allowed keys: {gap_keys}."
        in extra_feedback
    )
    quote_feedback = llm.calls[2][0][-1].content
    assert "question_requirement_quote_not_in_question" in quote_feedback
    assert "Copy one contiguous span from the research question" in quote_feedback
    assert f"question_requirements[0].quote: {wrong_first!r}" in quote_feedback
    assert f"question_requirements[1].quote: {wrong_second!r}" in quote_feedback
    assert [item.quote for item in result.facts.question_requirements] == [
        first["quote"], second["quote"],
    ]


def test_family_retry_preserves_extra_key_feedback_for_unknown_nested_objects():
    request = _request(_context(named=False), cohort_mode=None)
    error = ValidationError.from_exception_data("FamilyPlanSpec", [{
        "type": "extra_forbidden",
        "loc": ("question_requirements", 0, "gap", "nested", "extra"),
        "input": True,
    }])

    assert _question_requirement_schema_feedback(error, request) == error.errors(include_url=False)


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


def test_only_a_verified_gap_can_stop_the_plan_as_a_capability_gap(
    tmp_path: Path,
) -> None:
    context = _context()
    result = _planned(context, [_BENCHMARK, _UNCHECKABLE])

    findings = question_requirement_outcome(
        context=context, plan=result.output, facts=result.facts, run_dir=tmp_path
    )

    stops = {item.detail["reason"]: item for item in findings if item.severity == "error"}
    (verified,) = stops["question_requirement_capability_gap"].detail["requirements"]
    assert verified["gap_verification"] == "verified"
    uncovered = stops["question_requirement_not_covered"]
    (row,) = uncovered.detail["requirements"]
    assert row["quote"] == _UNCHECKABLE["quote"]
    assert row["disposition"] == "not_covered"
    assert row["host_judged"] is row["verified_by_host"] is True
    assert row["gap_verification"] == "unverifiable"
    assert uncovered.message.endswith(row["gap_fact"] + ".")


@pytest.mark.parametrize("quote, kinds", [
    ("web article preview and downloadable PDF", ("article_preview", "manuscript_pdf")),
    ("网页文章预览、可下载的 PDF", ("article_preview", "manuscript_pdf")),
    ("result tables, figures and reference list", ("result_tables", "figures", "references")),
    ("结果表、图表、正文、引用", ("result_tables", "figures", "article_preview", "references")),
    ("Include the result tables", ("result_tables",)),
    ("请提供结果表", ("result_tables",)),
])
@pytest.mark.parametrize("kind", ["analysis", "delivery"])
def test_host_delivery_checks_the_plan_owner_and_records_why(tmp_path, quote, kinds, kind) -> None:
    context = _context(named=False).model_copy(update={"research_question": _QUESTION + " " + quote})
    claim = {**_UNCHECKABLE, "kind": kind, "quote": quote}
    result = _planned(context, [claim])
    findings = question_requirement_outcome(
        context=context, plan=result.output, run_dir=tmp_path, facts=result.facts,
    )
    record = json.loads((tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_text())
    assert record["stated"][0] == claim
    (row,) = record["judged"]
    assert row["disposition"] == "covered"
    assert row["claimed_coverage"] == "capability_gap"
    assert row["host_judged"] is row["verified_by_host"] is True
    assert row["gap"] == claim["gap"]
    assert record["coverage"]["complete"] is True
    assert row["reason_code"] == "question_requirement_covered_by_host_delivery"
    assert row["delivery"]["source"] == "host_verified"
    assert row["delivery"]["contract"] == "easyicu.run_delivery/1"
    assert set(row["delivery"]["deliverable_kinds"]) == set(kinds)
    steps = {step.step_id: step for step in result.output.steps}
    owners = row["delivery"]["owner_step_ids_by_kind"]
    assert set(owners) == set(kinds)
    for coverage in record["coverage"]["records"]:
        delivery = coverage["delivery_kind"]
        assert coverage["concept_identity"] == delivery
        assert coverage["owner_step_ids"] == owners[delivery]
        assert owners[delivery] and set(owners[delivery]) <= steps.keys()
        if delivery in {"article_preview", "manuscript_pdf", "references"}:
            assert [steps[owner].host_delivery_kind for owner in owners[delivery]] == [delivery]
    assert _question_stops(findings) == []
    reviewed = judge_recorded_requirements(record, plan=result.output)
    assert reviewed.record["judged"] == record["judged"]
    assert _question_stops(reviewed.findings()) == []


@pytest.mark.parametrize("question,kinds", [
    ("Describe shift recovery. Please include a browser preview, a PDF and a bibliography.",
     ("article_preview", "manuscript_pdf", "references")),
    ("请描述值班期间的康复情况。请提供在线预览、PDF和参考文献列表。",
     ("article_preview", "manuscript_pdf", "references")),
    ("Describe the recovery rates by shift.", ()),
    ("请描述不同班次的康复率。", ()),
    ("Describe shift recovery. Do not provide a PDF or bibliography.", ()),
    ("请描述康复率。不需要PDF或参考文献。", ()),
])
def test_the_planner_adds_host_steps_only_when_the_question_requests_them(question, kinds):
    context = _context(named=False).model_copy(update={"research_question": question})
    result = _planned(context, [])
    owners = [step for step in result.output.steps if getattr(step, "host_delivery_kind", None)]
    assert [step.host_delivery_kind for step in owners] == list(kinds)
    assert len({step.step_id for step in owners}) == len(kinds)


@pytest.mark.parametrize("quote", [
    "fit an unavailable joint model and provide a PDF",
    "download patient-level data as PDF",
    "downloadable DOCX",
    "upload the article to an external web server",
    "未支持的纵向模型、图表和参考文献",
])
@pytest.mark.parametrize("kind", ["analysis", "delivery"])
def test_delivery_words_cannot_hide_a_method_or_unavailable_delivery_gap(quote, kind) -> None:
    context = _context()
    plan = _planned(context, [_BENCHMARK]).output
    item = QuestionRequirement.model_validate({**_UNCHECKABLE, "kind": kind, "quote": quote})
    (judged,) = judge_question_requirements(
        [item], plan=plan, family_template=True,
        relatives=concept_relatives(context), check_gap=_host_check(context),
    )
    assert judged.disposition == "not_covered"
    assert judged.verification == "unverifiable"
    assert _question_stops(question_requirement_findings([judged])) == [
        "question_requirement_not_covered",
    ]


def test_typed_delivery_uses_the_host_contract_independently_of_wording(monkeypatch) -> None:
    quote = "give me a typeset copy of the article"
    context = _context(named=False).model_copy(update={"research_question": _QUESTION + " " + quote})
    claim = {**_UNCHECKABLE, "kind": "delivery", "quote": quote,
             "deliverable_kinds": ["manuscript_pdf"]}
    def no_scientific_gap_check(*args, **kwargs):
        raise AssertionError("the host delivery contract owns this gap claim")
    monkeypatch.setattr(
        "easyicu.research_agent.agents.family_spec_planner.check_capability_gap", no_scientific_gap_check,
    )
    result = _planned(context, [claim])
    assert result.facts.question_requirements[0].deliverable_kinds == ["manuscript_pdf"]
    schema = question_requirements_owner.question_requirements_schema([])
    assert "deliverable_kinds" in schema["items"]["required"]
    assert "manuscript_pdf" in schema["items"]["properties"]["deliverable_kinds"]["items"]["enum"]
    (judged,) = judge_question_requirements(
        result.facts.question_requirements, plan=result.output, family_template=True,
        relatives=concept_relatives(context), check_gap=_host_check(context),
    )
    assert judged.disposition == "covered"
    (owner_id,) = judged.delivery["owner_step_ids_by_kind"]["manuscript_pdf"]
    assert next(step for step in result.output.steps if step.step_id == owner_id).host_delivery_kind == "manuscript_pdf"
    with pytest.raises(ValidationError, match="only a delivery requirement"):
        QuestionRequirement.model_validate({**claim, "kind": "analysis", "quote": "fit an unsupported joint model"})
    with pytest.raises(ValidationError, match="not scientific concepts"):
        QuestionRequirement.model_validate({**claim, "concepts": ["age"]})


def test_delivery_coverage_cannot_replace_a_scientific_step_owner() -> None:
    with pytest.raises(ValidationError, match="owner step or host delivery"):
        PlanRequirementCoverageRecord(
            requirement_id="outcome:death", kind="outcome", concept_identity="death",
            source_ref="research_context.json.target_outcome", status="covered",
            reason_code="typed_plan_owner_present",
        )
    with pytest.raises(ValidationError, match="host delivery covers only its typed output"):
        PlanRequirementCoverageRecord(
            requirement_id="outcome:death", kind="outcome", concept_identity="death",
            source_ref="research_context.json.target_outcome", status="covered",
            delivery_kind="manuscript_pdf", reason_code="question_requirement_covered_by_host_delivery",
        )


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
    if how == "no-judged-rows":
        return {key: value for key, value in record.items() if key != "judged"}
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
        "no-judged-rows",
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
        "not_covered",
        "attested",
        "definition_only",
        *({"capability_gap"} if family_template else ()),
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


# Methods statements need their own provenance, rather than an analysis input.
def _clause_plan():
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    return AnalysisPlan(
        research_question="Assess marker A and marker B against the outcome.",
        analysis_type="association_study",
        steps=[AnalysisStep(
            step_id="main", intent="Estimate the marker A association.",
            planned_analysis_role="primary", inputs=["marker_a", "outcome"],
            method="adjusted_association_models",
            expected_outputs=["table:adjusted_association_estimates"],
        )],
    )


def _methods_requirement(identifier, quote, *, note=None, concepts=()):
    return QuestionRequirement(
        id=identifier, kind="analysis", quote=quote, concepts=list(concepts),
        coverage="plan", note=note,
    )


def _clause_judgment(requirements, **kwargs):
    return judge_question_requirements(
        requirements, plan=_clause_plan(), family_template=True,
        relatives=lambda name: frozenset({name}),
        check_gap=lambda gap: ("unverifiable", "No synthetic capability claim."),
        **kwargs,
    )


def test_m1_compound_methods_are_covered_by_separate_recorded_statements():
    requirements = [
        _methods_requirement("r1", "the exposure window", note="Use hours 0–12."),
        _methods_requirement("r2", "units", note="Use the recorded mmol/L scale."),
        _methods_requirement(
            "r3", "repeat-stay handling", note="Retain each stay and cluster by patient."
        ),
        _methods_requirement(
            "r4", "Specify the exposure window, units, and repeat-stay handling",
            concepts=["stay_key"],
        ),
    ]
    row = _clause_judgment(requirements)[-1].row()

    assert row["disposition"] == "covered"
    assert [clause["quote"] for clause in row["clauses"]] == [
        "the exposure window", "units", "repeat-stay handling",
    ]
    assert all(clause["evidence"]["source_ref"] for clause in row["clauses"])


def test_m1_one_unanswered_clause_still_blocks_and_is_reported_on_its_own():
    requirements = [
        _methods_requirement("r1", "the exposure window", note="Use hours 0–12."),
        _methods_requirement("r2", "units", note="Use mmol/L."),
        _methods_requirement(
            "r3", "Specify the exposure window, units, and external validation",
            concepts=["unavailable_external_cohort"],
        ),
    ]
    judged = _clause_judgment(requirements)
    assert judged[-1].disposition == "not_covered"
    stop, = [f for f in question_requirement_findings(judged) if f.severity == "error"]
    assert [row["quote"] for row in stop.detail["requirements"]] == ["external validation"]
    assert "the exposure window, units" not in stop.message


@pytest.mark.parametrize("provenance", [None, "synthetic_host_policy:cluster_by_patient/1"])
def test_m1_a_default_needs_recorded_provenance(provenance):
    plan = _clause_plan()
    default = {
        "facet": "repeated_stays", "concepts": [], "source": "host_default",
        "statement": "Retain stays and cluster by patient.",
        "plan_sha256": analysis_plan_sha256(plan),
        **({"source_ref": provenance} if provenance else {}),
    }
    requirements = [_methods_requirement(
        "r1", "Specify repeat-stay handling", concepts=["stay_key"],
    )]
    row = _clause_judgment(requirements, host_facts=[default])[0].row()
    assert row["disposition"] == ("covered" if provenance else "not_covered")


def _joint_exposure_requirements():
    return [
        QuestionRequirement(
            id="r1", kind="definition", quote="marker B", concepts=["marker_b"],
            coverage="definition_only", note="Defines a score; marker A stays primary.",
        ),
        _methods_requirement(
            "r2", "Assess marker A and marker B against the outcome",
            concepts=["marker_a", "marker_b", "outcome"],
        ),
    ]


def test_m1_a_definition_does_not_replace_a_second_exposure_analysis():
    row = _clause_judgment(
        _joint_exposure_requirements(), sealed=("marker_a", "outcome"),
    )[-1].row()
    assert row["disposition"] == "not_covered"
    assert row["exposure_choice"]["additional_exposures"] == ["marker_b"]
    assert row["exposure_choice"]["options"] == [
        "analyse_both", "primary_and_secondary", "only_primary",
    ]


def test_m1_a_single_exposure_question_keeps_its_existing_judgment():
    item = _methods_requirement(
        "r1", "Assess marker A against the outcome", concepts=["marker_a", "outcome"],
    )
    judged, = _clause_judgment([item])
    assert judged.disposition == "attested"
    assert "exposure_choice" not in judged.row()


def test_m1_a_default_bound_to_another_plan_does_not_cover_a_clause():
    item = _methods_requirement("r1", "Specify repeat-stay handling", concepts=["stay_key"])
    row, = _clause_judgment([item], host_facts=[{
        "facet": "repeated_stays", "concepts": [], "source": "host_default",
        "source_ref": "synthetic_policy:clustering/1", "statement": "Cluster by patient.",
        "plan_sha256": "0" * 64,
    }])
    assert row.disposition == "not_covered"


def test_m1_a_default_does_not_override_an_explicit_first_stay_request():
    item = _methods_requirement(
        "r1", "Specify repeat-stay handling: keep only the first stay", concepts=["stay_key"],
    )
    row, = _clause_judgment([item], host_facts=[{
        "facet": "repeated_stays", "concepts": [], "source": "host_default",
        "source_ref": "synthetic_policy:clustering/1", "statement": "Retain each stay and cluster.",
        "plan_sha256": analysis_plan_sha256(_clause_plan()),
    }])
    assert row.disposition == "not_covered"


@pytest.mark.parametrize("typed_analysis", [False, True])
def test_m1_an_additional_exposure_needs_a_typed_analysis_not_an_input(typed_analysis):
    from easyicu.research_agent.contracts.model_terms import PlannedModelRequirement
    from easyicu.research_agent.contracts.model_tokens import ASSOCIATION_LOGIT_ESTIMATOR
    from easyicu.research_agent.schema import AnalysisStep

    def model(exposure, role):
        return PlannedModelRequirement(
            requirement_id=f"{exposure}_association", outcome="outcome", outcome_type="binary",
            method_family=ASSOCIATION_LOGIT_ESTIMATOR, exposure_source=exposure,
            analysis_role=role, analysis_set="complete_case",
        )

    plan = _clause_plan()
    primary = plan.steps[0].model_copy(update={"model_requirements": [model("marker_a", "primary")]})
    companion = AnalysisStep(
        step_id="companion", intent="Prespecified companion-marker analysis.",
        planned_analysis_role="secondary", method="adjusted_association_models",
        inputs=["marker_b", "outcome"], expected_outputs=["table:adjusted_association_estimates"],
        model_requirements=[model("marker_b", "secondary")] if typed_analysis else [],
    )
    row = judge_question_requirements(
        _joint_exposure_requirements(), plan=plan.model_copy(update={"steps": [primary, companion]}),
        family_template=False, relatives=lambda name: frozenset({name}),
        check_gap=lambda gap: ("unverifiable", "No capability claim."),
        sealed=("marker_a", "outcome"),
    )[-1]
    assert row.disposition == ("covered" if typed_analysis else "not_covered")
    if typed_analysis:
        assert row.owner_step_ids == ("main", "companion")


def test_m1_a_compound_statement_cannot_prove_a_shorter_statement_circularly():
    requirements = [
        _methods_requirement("r1", "Specify the exposure window, units", concepts=["marker_a"]),
        _methods_requirement(
            "r2", "Specify the exposure window, units, and repeat-stay handling",
            concepts=["marker_a"], note="No concrete window or unit has been chosen.",
        ),
    ]
    assert all(row.disposition == "not_covered" for row in _clause_judgment(requirements))


@pytest.mark.parametrize("source_ref", [None, "synthetic_export_manifest/patient_grouping"])
def test_m1_the_host_records_a_repeat_stay_default_only_with_its_source(source_ref):
    context = SimpleNamespace(
        variables=[], primary_exposure="marker_a",
        user_preferences=SimpleNamespace(sensitivity_specs=[], data_constraints=json.dumps({
            "analysis_design": {"analysis_unit": "icu_stay", "cluster_unit": "patient",
                                "variance_estimator": "cluster_robust"},
            "verified_patient_grouping": {"mapping_sha256": "a" * 64,
                                         **({"authority_ref": source_ref} if source_ref else {})},
        })),
    )
    facts = question_requirements_owner.question_requirement_host_facts(
        context=context, plan=_clause_plan(),
        requirements=[_methods_requirement("r1", "Specify repeat-stay handling")],
    )
    assert bool(facts) == bool(source_ref)


def test_m1_an_explicit_clause_cannot_borrow_a_stated_exposure_for_an_unaddressed_analysis():
    item = _methods_requirement(
        "r1", "Assess marker A and perform external validation",
        concepts=["marker_a"],
    ).model_copy(update={"clauses": ["Assess marker A", "perform external validation"]})
    row, = _clause_judgment([item])
    assert row.disposition == "not_covered"
    assert [part["quote"] for part in row.clauses if part["disposition"] == "not_covered"] == item.clauses


@pytest.mark.parametrize("quote,facet,statement", [
    ("Specify the exposure window of 48 hours", "window", {"landmark_hours": 24.0}),
    ("Specify units in mmol/L", "unit", "mg/dL"),
])
def test_m1_a_recorded_default_must_match_explicit_requested_values(quote, facet, statement):
    item = _methods_requirement("r1", quote, concepts=["marker_a"])
    row, = _clause_judgment([item], host_facts=[{
        "facet": facet, "concepts": ["marker_a"], "source": "host_default",
        "source_ref": "synthetic_policy:measurement/1", "statement": statement,
        "plan_sha256": analysis_plan_sha256(_clause_plan()),
    }])
    assert row.disposition == "not_covered"


def test_m1_a_component_grading_clause_cites_its_own_rule_before_the_aggregate(monkeypatch):
    from easyicu import clinical_contracts

    def contract(name, concepts, depends=()):
        return SimpleNamespace(
            contract_id=name, version="1", concepts=concepts,
            source_id="synthetic_clinical_spec", source_table="grading rules",
            definition=f"Registered {name} grading rule.",
            reference_implementation=f"synthetic.{name}", validation_status="synthetic_test",
            depends_on_contracts=depends,
        )

    monkeypatch.setattr(clinical_contracts, "load_clinical_contracts", lambda: {
        "aggregate": contract("aggregate", ["aggregate_marker"], ["component"]),
        "component": contract("component", ["component_marker"]),
    })
    context = SimpleNamespace(
        primary_exposure="marker_a", user_preferences=None,
        variables=[SimpleNamespace(name=name, source_concept=name, unit=None, analysis_window=None)
                   for name in ("aggregate_marker", "component_marker")],
    )
    definition = QuestionRequirement(
        id="r1", kind="definition", quote="marker grading", coverage="definition_only",
        concepts=["aggregate_marker", "component_marker"], note="Use the registered component rule.",
    )
    compound = _methods_requirement("r2", "Specify marker grading, units")
    facts = question_requirements_owner.question_requirement_host_facts(
        context=context, plan=_clause_plan(), requirements=[definition, compound],
    )
    row = _clause_judgment([definition, compound], host_facts=facts)[-1]
    assert row.clauses[0]["evidence"]["source_ref"] == "clinical_contract:component@1"
    assert row.clauses[1]["disposition"] == "not_covered"


@pytest.mark.parametrize("stated", [False, True])
def test_m1_a_clause_can_be_stated_in_the_plan_but_a_negation_does_not_answer_it(stated):
    plan = _clause_plan()
    intent = ("Repeat-stay handling retains all stays and clusters by patient." if stated
              else "No repeat-stay handling is planned.")
    plan = plan.model_copy(update={"steps": [plan.steps[0].model_copy(update={"intent": intent})]})
    item = _methods_requirement("r1", "Specify repeat-stay handling")
    row, = judge_question_requirements(
        [item], plan=plan, family_template=True, relatives=lambda name: frozenset({name}),
        check_gap=lambda gap: ("unverifiable", "No capability claim."),
    )
    assert row.disposition == ("covered" if stated else "not_covered")
    if stated:
        assert row.clauses[0]["evidence"]["source_ref"] == "analysis_plan.json.steps[main].intent"


def test_m1_a_clinical_definition_already_in_the_context_keeps_its_provenance():
    definition = SimpleNamespace(
        contract_id="component", version="1", source_id="synthetic_spec", definition="Registered grading rule.",
        validation_status="synthetic_test",
    )
    context = SimpleNamespace(
        variables=[SimpleNamespace(name="component_marker", unit=None, analysis_window=None,
                                   clinical_definition=definition)],
        primary_exposure="marker_a", user_preferences=None,
    )
    facts = question_requirements_owner.question_requirement_host_facts(
        context=context, plan=_clause_plan(),
        requirements=[_methods_requirement("r1", "Specify grading rules", concepts=["component_marker"])],
    )
    assert facts[0]["source_ref"] == "research_context.json.variables[component_marker].clinical_definition"


def test_m1_a_planner_reference_to_a_host_default_is_not_the_defaults_provenance():
    requirements = [
        _methods_requirement("r1", "repeat-stay handling", note="Use the host default."),
        _methods_requirement("r2", "Specify repeat-stay handling"),
    ]
    assert _clause_judgment(requirements)[-1].disposition == "not_covered"


def test_m1_explicit_clauses_cannot_omit_an_unanswered_part_of_the_requirement():
    with pytest.raises(ValidationError, match="complete requirement quote"):
        QuestionRequirement(
            id="r1", kind="analysis", coverage="plan",
            quote="Specify the exposure window, units, and external validation",
            clauses=["the exposure window", "units"],
        )


def test_m1_grouping_a_methods_list_into_one_clause_does_not_hide_a_missing_member():
    quote = "Specify the exposure window, units"
    item = QuestionRequirement(id="r2", kind="analysis", coverage="plan", quote=quote, clauses=[quote])
    stated = _methods_requirement("r1", "the exposure window", note="Use hours 0–12.")
    row = _clause_judgment([stated, item])[-1]
    assert row.disposition == "not_covered"
    assert [clause["quote"] for clause in row.clauses if clause["disposition"] == "not_covered"] == ["units"]


@pytest.mark.parametrize("question, quote, kinds", [
    ("Compare recovery with and without chronic lung disease, and please export a PDF.",
     "please export a PDF", ("manuscript_pdf",)),
    ("比较不同班次的康复情况，请提供在线预览、PDF和参考文献清单。",
     "请提供在线预览、PDF和参考文献清单", ("article_preview", "manuscript_pdf", "references")),
    ("Describe afternoon recovery, and do not provide a PDF.", None, ()),
    ("比较不同班次的康复情况，不要提供PDF。", None, ()),
])
def test_a_delivery_clause_agrees_with_its_judged_plan_owner(question, quote, kinds):
    from easyicu.research_agent.planning.requirement_typing.delivery import bind_requested_host_delivery

    context = _context(named=False).model_copy(update={"research_question": question})
    claims = [] if quote is None else [{"id": "r1", "kind": "delivery", "quote": quote, "coverage": "plan"}]
    result = _planned(context, claims)
    # The final planning boundary also rebinds from the original user question.
    plan = bind_requested_host_delivery(result.output, question=question)
    owners = {step.host_delivery_kind: step.step_id for step in plan.steps if step.host_delivery_kind}
    assert tuple(owners) == kinds
    judged = judge_question_requirements(
        result.facts.question_requirements, plan=plan, family_template=True,
        relatives=concept_relatives(context), check_gap=_host_check(context), question=question,
    )
    if quote is None:
        assert judged == ()
    else:
        (row,) = judged
        assert row.disposition == "covered"
        assert row.delivery["owner_step_ids_by_kind"] == {kind: [owner] for kind, owner in owners.items()}


@pytest.mark.parametrize("question, quote, requested, ambiguous", [
    ("Compare shift recovery, please export a PDF, and do not include references.",
     "please export a PDF", ("manuscript_pdf",), ()),
    ("比较不同班次的康复情况，请导出PDF，不要提供在线预览。",
     "请导出PDF", ("manuscript_pdf",), ()),
    ("Please export a PDF comparing patients with and without sepsis.",
     "Please export a PDF", ("manuscript_pdf",), ()),
    ("Do not provide an online preview and export a PDF.",
     "export a PDF", (), ("manuscript_pdf",)),
    ("不要提供在线预览并导出PDF。",
     "导出PDF", (), ("manuscript_pdf",)),
    ("Include the night-ward estimates without changing the study and supply a bibliography.",
     "supply a bibliography", ("references",), ()),
    ("请提供在线预览来比较有无发热的住院记录。",
     "请提供在线预览", ("article_preview",), ()),
    ("不要PDF。请提供参考文献清单。",
     "请提供参考文献清单", ("references",), ()),
])
def test_delivery_selection_is_shared_by_planning_judging_and_approval(tmp_path, question, quote, requested, ambiguous):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.contracts.product_files import DeliveryRequest, PlanDeliverables, delivery_requests
    from easyicu.research_agent.orchestration.workflow import human_review_requests_for_plan
    from easyicu.research_agent.planning.requirement_typing.delivery import (
        bind_confirmed_host_delivery, bind_requested_host_delivery,
    )

    context = _context(named=False).model_copy(update={"research_question": question})
    claim = {"id": "r1", "kind": "delivery", "quote": quote, "coverage": "plan",
             "deliverable_kinds": list(requested or ambiguous)}
    result = _planned(context, [claim])
    requests = delivery_requests(question)
    assert {kind for kind, decision in requests.items() if decision is DeliveryRequest.REQUESTED} == set(requested)
    assert {kind for kind, decision in requests.items() if decision is DeliveryRequest.CHECK} == set(ambiguous)
    plan = bind_requested_host_delivery(result.output, question=question)
    owners = {step.host_delivery_kind: step.step_id for step in plan.steps if step.host_delivery_kind}
    assert tuple(owners) == (() if ambiguous else requested)
    assert plan.deliverables.confirmed is not bool(ambiguous)
    findings = question_requirement_outcome(context=context, plan=plan, facts=result.facts, run_dir=tmp_path)
    (review,) = human_review_requests_for_plan(
        findings=findings, plan=plan, require_plan_review=True, evidence=EvidenceStore(tmp_path), question=question,
    )
    record = json.loads((tmp_path / QUESTION_REQUIREMENTS_FILENAME).read_text())
    (row,) = record["judged"]
    if ambiguous:
        assert review.payload["reason"] == row["reason_code"] == "host_delivery_request_ambiguous"
        assert review.payload["approval_allowed"] is False
        assert review.payload["deliverable_kinds"] == list(ambiguous)
        assert row["disposition"] == "needs_clarification"
        assert not record["coverage"]["complete"]
        selected = bind_confirmed_host_delivery(plan, PlanDeliverables(
            requested=ambiguous, confirmed=True, source="user_edit",
        ))
        (resolved,) = judge_question_requirements(
            result.facts.question_requirements, plan=selected, family_template=True,
            relatives=concept_relatives(context), check_gap=_host_check(context), question=question,
        )
        assert resolved.disposition == "covered"
        assert resolved.delivery["owner_step_ids_by_kind"] == {
            step.host_delivery_kind: [step.step_id] for step in selected.steps if step.host_delivery_kind
        }
        (approvable,) = human_review_requests_for_plan(
            findings=findings, plan=selected, require_plan_review=True,
            evidence=EvidenceStore(tmp_path), question=question,
        )
        assert approvable.payload["reason"] == "operator_plan_approval_required"
        assert approvable.authority_sha256 != review.authority_sha256
    else:
        assert review.payload["reason"] == "operator_plan_approval_required"
        assert row["disposition"] == "covered"
        assert row["delivery"]["owner_step_ids_by_kind"] == {kind: [owner] for kind, owner in owners.items()}


@pytest.mark.parametrize("question, fragment", [
    ("Compare afternoon recovery, and do not provide a PDF.", "provide a PDF"),
    ("请描述晨班记录，不要PDF。", "PDF"),
])
def test_a_planner_quote_cannot_recast_an_excluded_export_as_a_request(question, fragment):
    from easyicu.research_agent.contracts.product_files import DeliveryRequest, delivery_requests

    from easyicu.research_agent.planning.requirement_typing.delivery import bind_requested_host_delivery

    item = QuestionRequirement(id="r1", kind="delivery", quote=fragment, coverage="plan",
                               deliverable_kinds=["manuscript_pdf"])
    plan = bind_requested_host_delivery(AnalysisPlan(research_question=question, steps=[]))
    (row,) = judge_question_requirements(
        [item], plan=plan, family_template=True, relatives=lambda _: frozenset(),
        check_gap=lambda _: pytest.fail("no gap is claimed"), question=question,
    )
    assert delivery_requests(question)["manuscript_pdf"] is DeliveryRequest.NOT_REQUESTED
    assert row.disposition == "not_requested"
    assert row.delivery is None


@pytest.mark.parametrize("question, requested, excluded", [
    ("Exclude patients without baseline lactate, and export a PDF.", "manuscript_pdf", None),
    ("Please export a PDF comparing patients without sepsis to controls.", "manuscript_pdf", None),
    ("请提供在线预览来比较不需要机械通气的患者。", "article_preview", None),
    ("Please export a PDF, without references.", "manuscript_pdf", "references"),
    ("Please export a PDF, but do not include references.", "manuscript_pdf", "references"),
    ("Summarize overnight recovery without a PDF.", None, "manuscript_pdf"),
    ("请导出PDF，但是不要提供参考文献。", "manuscript_pdf", "references"),
])
def test_clinical_negation_and_export_exclusions_prefill_independently(question, requested, excluded):
    from easyicu.research_agent.contracts.product_files import DeliveryRequest, PlanDeliverables, delivery_requests
    from easyicu.research_agent.planning.requirement_typing.delivery import (
        bind_confirmed_host_delivery, bind_requested_host_delivery,
    )

    states = delivery_requests(question)
    if requested:
        assert states[requested] is DeliveryRequest.REQUESTED
    if excluded:
        assert states[excluded] is DeliveryRequest.NOT_REQUESTED
    plan = bind_requested_host_delivery(AnalysisPlan(research_question=question, steps=[]))
    assert plan.deliverables.confirmed
    assert [step.host_delivery_kind for step in plan.steps] == ([requested] if requested else [])
    changed = ("manuscript_pdf",) if requested is None else ()
    edited = bind_confirmed_host_delivery(plan, PlanDeliverables(
        requested=changed, confirmed=True, source="user_edit",
    ))
    assert edited.deliverables.confirmed
    assert tuple(step.host_delivery_kind for step in edited.steps) == changed


def test_an_edited_selection_stays_bound_when_the_detector_changes(monkeypatch):
    from easyicu.research_agent.authority.plan_scope import plan_scientific_scope_signature
    from easyicu.research_agent.contracts.product_files import PlanDeliverables, host_delivery_owners
    from easyicu.research_agent.orchestration.workflow import human_review_requests_for_plan
    from easyicu.research_agent.planning.requirement_typing import delivery

    question = "Compare recovery by ward. Export a PDF or provide an online preview."
    draft = AnalysisPlan(research_question=question, steps=[])
    prefilled = delivery.bind_requested_host_delivery(draft)
    assert prefilled.deliverables.check == ("article_preview", "manuscript_pdf")
    edited = delivery.bind_confirmed_host_delivery(draft, PlanDeliverables(
        requested=("article_preview",), confirmed=True, source="user_edit",
    ))
    assert plan_scientific_scope_signature(edited) == plan_scientific_scope_signature(prefilled)
    (request,) = human_review_requests_for_plan(findings=[], plan=edited, require_plan_review=True)
    approved = AnalysisPlan.model_validate(request.payload["plan_review_authority"]["plan_payload"])

    def detector_must_not_run(_):
        pytest.fail("an approved run cannot reinterpret a delivery request")

    monkeypatch.setattr(delivery, "delivery_requests", detector_must_not_run)
    item = QuestionRequirement(id="r1", kind="delivery", quote="Export a PDF", coverage="plan")
    (row,) = judge_question_requirements(
        [item], plan=approved, family_template=True, relatives=lambda _: frozenset(),
        check_gap=lambda _: pytest.fail("that is a deliverable, not science"), question=question,
    )
    assert row.disposition == "not_requested"
    assert host_delivery_owners(approved.model_dump(mode="json")) == {
        "article_preview": approved.steps[0].step_id,
    }
    (reopened,) = human_review_requests_for_plan(findings=[], plan=approved, require_plan_review=True)
    assert reopened.authority_sha256 == request.authority_sha256

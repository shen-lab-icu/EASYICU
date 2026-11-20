"""An exposure choice the researcher answered is not asked again (B29).

UX leg 3 (rc_l3): the question named "乳酸水平或峰值乳酸", the plan review
asked whether to analyse both or only the primary, the researcher chose
"仅分析主要暴露", and the revised plan came back with the same choice: the
Planner again claimed ``needs_choice`` and the judge asked again, forever.
The host now records the option the researcher sent as a typed decision,
carries it into the revised plan's planning input, and judges the plan on
it.  Synthetic contexts and plans only.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.agents import progressive_planner
from easyicu.research_agent.orchestration.progressive_planning import (
    question_requirement_outcome,
)
from easyicu.research_agent.planning.question_requirements import (
    EXPOSURE_DECISION_REASON,
    EXPOSURE_DECISIONS_KEY,
    ExposureArrangementDecision,
    judge_recorded_requirements,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.pi_copilot.plan_revision_intent import exposure_arrangement_decision
from easyicu.webserver.plan_change_request import PlanChangeRequest

from .family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _context,
    _request,
    _spec_payload,
)

_QUOTE = "Assess the association of injury stage and a companion marker against in-hospital death"
_PRIMARY, _OTHER = "injury_stage", "severity_score_24h"


def _decision(option: str = "only_primary", *, additional=(_OTHER,)) -> dict:
    return ExposureArrangementDecision(
        option=option, primary=_PRIMARY, additional=additional, source_run_id="run_src",
    ).model_dump(mode="json")


def _decided_context(*decisions: dict):
    context = _context()
    context = context.model_copy(update={"research_question": context.research_question + ". " + _QUOTE})
    if not decisions:
        return context
    constraints = json.loads(context.user_preferences.data_constraints)
    constraints[EXPOSURE_DECISIONS_KEY] = list(decisions)
    return context.model_copy(update={
        "user_preferences": context.user_preferences.model_copy(
            update={"data_constraints": json.dumps(constraints)}
        ),
    })


def _answer(context, *, needs_choice: bool) -> str:
    payload = _spec_payload(_request(context))
    payload["question_requirements"] = [
        {"id": "r1", "kind": "definition", "quote": "a companion marker",
         "concepts": [_OTHER], "coverage": "definition_only", "gap": None,
         "note": "Only the primary exposure is analysed."},
        {"id": "r2", "kind": "analysis", "quote": _QUOTE,
         "concepts": [_PRIMARY, _OTHER, "death"],
         "coverage": "needs_choice" if needs_choice else "plan", "gap": None,
         "note": "The question names two exposures.",
         **({"analysis_exposures": [_PRIMARY, _OTHER]} if needs_choice else {})},
    ]
    return json.dumps(payload)


def _plan(context, answers):
    llm = ScriptedMockLLMClient(answers)
    result = progressive_planner.ProgressivePlannerAgent(llm).run_attempt(
        context,
        planner_strategy=progressive_planner.FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode="predicate_filtered",
    )
    return llm, result


def _judged(context, tmp_path, *, needs_choice: bool = True):
    _llm, result = _plan(context, [_answer(context, needs_choice=needs_choice)] * 3)
    findings = question_requirement_outcome(
        context=context, plan=result.output, facts=result.facts, run_dir=tmp_path,
    )
    record = json.loads((tmp_path / "question_requirements.json").read_text(encoding="utf-8"))
    (row,) = [item for item in record["judged"] if item["id"] == "r2"]
    return row, findings, record, result.output


def test_without_an_answer_the_review_asks_which_exposures_to_analyse(tmp_path) -> None:
    row, _findings, record, _plan_ = _judged(_decided_context(), tmp_path)

    assert row["disposition"] == "not_covered"
    assert row["reason_code"] == "question_named_exposure_choice_required"
    assert row["exposure_choice"]["primary_exposure"] == _PRIMARY
    assert "exposure_decisions" not in record


def test_an_answered_choice_is_judged_on_the_answer_and_not_offered_again(tmp_path) -> None:
    # The Planner claims needs_choice again, as on UX leg 3; the answer holds.
    row, findings, record, plan = _judged(_decided_context(_decision()), tmp_path)

    assert row["disposition"] == "covered"
    assert row["reason_code"] == EXPOSURE_DECISION_REASON
    assert "exposure_choice" not in row
    assert row["owner_step_ids"]
    assert not [item for item in findings if item.detail.get("reason") == "question_requirement_not_covered"]
    assert record["exposure_decisions"] == [_decision()]
    # A plan under review is judged from the record alone, on the same answer.
    again = judge_recorded_requirements(record, plan=plan)
    assert {entry.requirement.id: entry.reason_code for entry in again.judged}["r2"] == EXPOSURE_DECISION_REASON


@pytest.mark.parametrize(
    "decision",
    [
        pytest.param(_decision(additional=("age",)), id="other-exposures"),
        pytest.param(
            ExposureArrangementDecision(
                option="only_primary", primary=_OTHER, additional=("age",), source_run_id="run_src",
            ).model_dump(mode="json"),
            id="other-primary",
        ),
    ],
)
def test_an_answer_for_other_exposures_answers_nothing(tmp_path, decision) -> None:
    row, _findings, _record, _plan_ = _judged(_decided_context(decision), tmp_path)

    assert row["reason_code"] == "question_named_exposure_choice_required"
    assert row["exposure_choice"]["additional_exposures"] == [_OTHER]


def test_analysing_both_still_needs_an_owner_for_each_and_does_not_ask_again(tmp_path) -> None:
    row, findings, _record, _plan_ = _judged(_decided_context(_decision("analyse_both")), tmp_path)

    assert row["disposition"] == "not_covered"
    assert row["reason_code"] == "question_requirement_not_covered"
    assert "exposure_choice" not in row
    assert _OTHER in row["gap_fact"] and "analyse_both" in row["gap_fact"]
    assert [item.detail.get("reason") for item in findings].count("question_requirement_not_covered") == 1


def test_the_planner_keeps_the_other_mention_a_definition_once_the_researcher_chose(tmp_path) -> None:
    # Without the answer, a definition-demoted second exposure is sent back.
    plain = _decided_context()
    llm, _result = _plan(plain, [_answer(plain, needs_choice=False), _answer(plain, needs_choice=True)])
    decided = _decided_context(_decision())
    decided_llm, decided_result = _plan(decided, [_answer(decided, needs_choice=False)])

    assert len(llm.calls) == 2
    assert len(decided_llm.calls) == 1
    assert "question_exposure_choice_fallback" not in decided_result.facts.prompt_metrics


def _projection(run_id: str = "run_src"):
    options = {
        key: {"en": f"EN {key}", "zh": f"ZH {key}"}
        for key in ("primary_and_secondary", "analyse_both", "only_primary")
    }
    return SimpleNamespace(workflow=SimpleNamespace(plan_review_summary={
        "run_id": run_id,
        "exposure_arrangement": {
            "primary": _PRIMARY, "additional": [_OTHER], "options": options,
            "default_option": "primary_and_secondary",
        },
    }))


def test_the_host_records_the_option_the_researcher_sent_on_the_card() -> None:
    assert exposure_arrangement_decision("ZH only_primary", projection=_projection(), source_run_id="run_src") == _decision()
    assert exposure_arrangement_decision("only the primary please", projection=_projection(), source_run_id="run_src") is None
    assert exposure_arrangement_decision("ZH only_primary", projection=_projection("run_new"), source_run_id="run_src") is None


def test_the_change_request_carries_the_answer_into_the_revised_plans_input() -> None:
    request = PlanChangeRequest(source_run_id="run_src", user_message="ZH only_primary", exposure_decision=_decision())
    legacy = PlanChangeRequest(source_run_id="run_src", user_message="ZH only_primary")

    preferences = agent_pipeline_runs._research_user_preferences(
        {"id": "study_x", "question": _QUOTE}, exposure_decisions=[_decision()],
    )

    # A request without an answer keeps its recorded bytes.
    assert "exposure_decision" not in legacy.model_dump(mode="json")
    assert request.model_dump(mode="json")["exposure_decision"] == _decision()
    assert json.loads(preferences["data_constraints"])[EXPOSURE_DECISIONS_KEY] == [_decision()]
    with pytest.raises(ValueError):
        PlanChangeRequest(
            source_run_id="run_other", user_message="ZH only_primary", exposure_decision=_decision(),
        )


def test_a_revision_carries_the_answer_its_source_plan_was_judged_on(monkeypatch) -> None:
    source_record = SimpleNamespace(artifact_payloads={
        "question_requirements.json": {"exposure_decisions": [_decision("primary_and_secondary")]},
    })
    monkeypatch.setattr(
        agent_pipeline_runs.agent_runs, "list_run_history",
        lambda **_kwargs: {"runs": [{"run_id": "run_src", "project_dir": "/nowhere/run_src"}]},
    )
    monkeypatch.setattr(agent_pipeline_runs.agent_runs, "read_run_record", lambda _path: source_record)
    request = PlanChangeRequest(source_run_id="run_src", user_message="ZH only_primary", exposure_decision=_decision())

    from_request = agent_pipeline_runs._carried_exposure_decisions(
        study={"id": "study_x"}, project_root="/nowhere", change_request=request, source_run_id="",
    )
    from_record = agent_pipeline_runs._carried_exposure_decisions(
        study={"id": "study_x"}, project_root="/nowhere", change_request=None, source_run_id="run_src",
    )
    nothing = agent_pipeline_runs._carried_exposure_decisions(
        study={"id": "study_x"}, project_root="/nowhere", change_request=None, source_run_id="",
    )

    # This request's answer is newer than the one its source carried.
    assert from_request == [_decision()]
    assert from_record == [_decision("primary_and_secondary")]
    assert nothing == []

"""A recorded condition the question restricts to is never silently dropped (S1743).

The requirement layer answers for 「脓毒症」 only when the run offers a sepsis
column (``bind_named_question_concepts`` drops a name that denotes none), so on
an export without one a plan could study every row under a sepsis label.  The
plan card now names a recorded condition the plan selects on nowhere, with the
S1737 line.  A researcher's recorded all-rows decision against such a question
is a conflict between two of the researcher's statements: the card names both.
Synthetic plans and studies only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.planning.question_requirements import (
    NamedQuestionConcept,
    bind_named_question_concepts,
)
from easyicu.webserver.pi_copilot import workflow as workflow_module
from easyicu.webserver.pi_copilot.plan_decisions import (
    question_population_conflict,
    unselected_question_conditions,
)
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"
_SEPSIS = "在脓毒症 ICU 患者中，ICU 入科时首次乳酸与院内死亡有什么关联？"
_ALL_ROWS = {"name": "source_cohort", "selection_mode": "all_input_rows", "inclusion": [], "exclusion": []}


def _predicate(concept: str) -> dict[str, Any]:
    return {"concept_id": concept, "anchor": "icu_admission", "start_offset_hours": 0, "end_offset_hours": 24,
            "aggregation": "any", "op": "==", "value": True}


def test_an_export_without_the_sepsis_column_leaves_no_requirement() -> None:
    """Why the card owns this case: the requirement list has nothing to answer for."""
    named = NamedQuestionConcept.model_validate({"concepts": ["sep3", "sep3_sofa1", "sep3_sofa2"], "evidence": "脓毒症"})
    assert bind_named_question_concepts([named], roster=["lact", "death", "age"],
                                        relatives=lambda column: frozenset({column})) == ()


def test_a_plan_that_selects_on_no_sepsis_concept_names_it() -> None:
    study = {"question": _SEPSIS}
    assert unselected_question_conditions(study, {"cohort": _ALL_ROWS}) == ["脓毒症"]
    assert unselected_question_conditions(study, {"cohort": {**_ALL_ROWS, "selection_mode": "predicate_filtered",
                                                             "inclusion": [_predicate("age")]}}) == ["脓毒症"]


@pytest.mark.parametrize("concept", ["sep3", "sep3_sofa1", "sep3_sofa2"])
def test_a_plan_that_selects_on_sepsis_is_not_named(concept: str) -> None:
    plan = {"cohort": {**_ALL_ROWS, "selection_mode": "predicate_filtered", "inclusion": [_predicate(concept)]}}
    assert unselected_question_conditions({"question": _SEPSIS}, plan) == []


def test_a_criterion_already_shown_as_not_applied_is_not_named_twice() -> None:
    plan = {"cohort": {**_ALL_ROWS, "unapplied_population_criteria": ["在脓毒症 ICU 患者中"]}}
    assert unselected_question_conditions({"question": _SEPSIS}, plan) == []


def test_no_plan_or_no_condition_names_nothing() -> None:
    assert unselected_question_conditions({"question": _SEPSIS}, {}) == []
    assert unselected_question_conditions({"question": _SEPSIS}, None) == []
    assert unselected_question_conditions({"question": "成人 ICU 患者中，乳酸与院内死亡"}, {"cohort": _ALL_ROWS}) == []


def _summary(study: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    # The snapshot is compiled on the study without its eligibility record (a
    # signed record is the store's); the card fields read the study as given.
    snapshot = build_research_workflow_snapshot(
        study={key: value for key, value in study.items() if key != "cohort_eligibility_authority"},
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-condition", "run_type": "full", "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked", "run_status": "human_review_pending",
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "scientific_configuration_sha256": "a" * 64,
            "artifact_names": ["agent_plan.json", "scientific_plan_review.json"],
        },
        plan_review_authority={
            "run_id": "run-condition", "resumable_here": True, "scientific_configuration_sha256": "a" * 64,
            "scientific_plan_review": {"status": "approvable", "findings": []},
        },
    )
    enriched = workflow_module._enrich_plan_review(  # noqa: SLF001
        snapshot, study=study, review={"artifact_payloads": {"agent_plan.json": plan}}
    )
    return dict(enriched.plan_review_summary or {})


def _study(**extra: Any) -> dict[str, Any]:
    return {"id": "study-condition", "revision": 2, "question": _SEPSIS, "active_job_id": None,
            "analysis_design": {"analysis_family": "association_study", "analysis_unit": "icu_stay",
                                "variance_estimator": "model_based"}, **extra}


def test_the_plan_under_review_names_an_unselected_sepsis_population() -> None:
    assert _summary(_study(), {"cohort": _ALL_ROWS})["question_population_unlisted"] == ["脓毒症"]
    selected = {"cohort": {**_ALL_ROWS, "selection_mode": "predicate_filtered", "inclusion": [_predicate("sep3_sofa1")]}}
    assert "question_population_unlisted" not in _summary(_study(), selected)


@pytest.mark.parametrize("state", ["confirmed", "declined"])
def test_a_recorded_all_rows_decision_against_the_question_is_a_named_conflict(state: str) -> None:
    authority = {"selection_state": state, "selection_mode": "all_input_rows", "study_context_revision": 1}
    assert question_population_conflict(_study(cohort_eligibility_authority=authority)) == ["脓毒症"]
    assert _summary(_study(cohort_eligibility_authority=authority), {"cohort": _ALL_ROWS})[
        "question_population_conflict"] == ["脓毒症"]


def test_no_conflict_without_a_recorded_all_rows_decision() -> None:
    assert question_population_conflict(_study()) == []
    filtered = {"selection_state": "confirmed", "selection_mode": "predicate_filtered", "study_context_revision": 1}
    assert question_population_conflict(_study(cohort_eligibility_authority=filtered)) == []
    declined = {"selection_state": "declined", "selection_mode": "all_input_rows", "study_context_revision": 1}
    assert question_population_conflict({**_study(cohort_eligibility_authority=declined),
                                         "question": "成人 ICU 患者中，乳酸与院内死亡"}) == []


def _notes_html(summary: dict[str, Any], lang: str) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = r"""
global.window = {};
require(process.argv[1]);
let api = null;
window.EasyICU = { guidedPi: { declare: (_name, value) => { api = value; } } };
require(process.argv[2]);
const lang = process.argv[4];
process.stdout.write(api.notesHtml(JSON.parse(process.argv[3]), {
  tr: (en, zh) => (lang === 'zh' ? zh : en), esc: window.EU_HTML.esc,
}));
"""
    result = run_node(node, script, str((STATIC / "js" / "html-escape.js").resolve()),
                      str((STATIC / "js" / "screens-guided-pi-question-requirements.js").resolve()),
                      json.dumps(summary), lang, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def test_the_card_names_both_statements() -> None:
    zh = _notes_html({"question_population_conflict": ["脓毒症"]}, "zh")
    en = _notes_html({"question_population_conflict": ["sepsis"]}, "en")
    assert "研究设置中记录了不限定人群（纳入全部患者），而题面限定为「脓毒症」患者，两者不一致" in zh
    assert "请在对话中说明以哪一个为准" in zh
    assert "record that the study takes every patient, but the question restricts it to “sepsis” patients" in en
    assert _notes_html({}, "zh") == ""

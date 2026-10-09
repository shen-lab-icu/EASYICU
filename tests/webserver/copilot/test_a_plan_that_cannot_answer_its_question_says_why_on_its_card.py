"""A plan that leaves an analysis its question asks for unanswered says why.

The planning owner refuses approval of a reviewed plan when no step answers an
analysis the question asks for, when the plan cannot answer it, or when the
record of what the question asks for could not be read or did not validate as
the plan was submitted, so the plan was never checked against its question.
Each card says the plan cannot be approved and nothing has run, and offers to
generate the plan again; the agent route chooses which plan, so no card
promises one (from these states it is a metadata-only candidate, as the last
test checks).  The gap card says a new plan may stop again unless the question
changes.  The first two link the judgment of the plan under review.  Every
stop in ``approval_stops.PLAN_APPROVAL_STOPS`` has a card and an aside line,
so a stop an owner adds cannot leave the plan card silent.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.planning.approval_stops import PLAN_APPROVAL_STOPS
from easyicu.research_agent.planning.question_requirements import (
    QUESTION_REQUIREMENT_APPROVAL_STOPS,
    QUESTION_REQUIREMENT_STOP_CODES,
    UNREADABLE_REASON,
)
from easyicu.webserver import agent_runs
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"
NOT_COVERED = QUESTION_REQUIREMENT_APPROVAL_STOPS["not_covered"]
CAPABILITY_GAP = QUESTION_REQUIREMENT_APPROVAL_STOPS["capability_gap"]
UNREADABLE = UNREADABLE_REASON
QUESTION_STOPS = (NOT_COVERED, CAPABILITY_GAP, UNREADABLE)


def _card(code: str, *, run_id: str = "run_question_1") -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = r"""
let declared = null;
global.window = {
  EU_LANG: 'zh',
  EasyICU: { guidedPi: { declare: (_name, api) => { declared = api; }, optional: () => null } },
};
require(process.argv[1]);
const code = process.argv[2];
const runId = process.argv[3];
const owner = declared.create({
  tr: (en, zh) => zh || en,
  esc: value => String(value),
  iconHtml: () => '',
  resourceButton: resource => `<a data-resource="${resource.artifact}"></a>`,
  sessionIsStale: () => false,
  busy: () => false,
  session: () => ({ binding: { run_id: runId } }),
  workflow: () => ({ next_action_code: code, plan_review_summary: { run_id: runId } }),
});
const confirmation = owner.workflowConfirmation();
const html = owner.workflowConfirmationHtml();
process.stdout.write(JSON.stringify({
  code: confirmation && confirmation.code,
  title: confirmation && confirmation.title,
  note: confirmation && confirmation.note,
  approve: confirmation && confirmation.approve,
  nonApprovable: Boolean(confirmation && confirmation.nonApprovable),
  grants: confirmation && confirmation.grants,
  execute: html.includes('data-gpi-confirm-action'),
  resources: (html.match(/data-resource="([^"]+)"/g) || []).map(row => row.slice(15, -1)),
  offeredResources: ((confirmation && confirmation.reviewResources) || []).length,
}));
"""
    owner = STATIC / "js" / "screens-guided-pi-confirmation.js"
    result = run_node(node, script, str(owner.resolve()), code, run_id, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def _aside_entry(code: str) -> str:
    aside = (STATIC / "js" / "screens-guided-pi-aside.js").read_text(encoding="utf-8")
    entries = [
        line for line in aside.splitlines() if line.strip().startswith(f"{code}: tr(")
    ]
    assert len(entries) == 1, code
    return entries[0]


def test_an_analysis_no_step_answers_offers_a_fresh_candidate_plan() -> None:
    card = _card(NOT_COVERED)

    assert card["code"] == NOT_COVERED
    assert card["title"] == "题面要求的一项分析，这份计划没有回答"
    assert "这份计划不能批准，分析尚未开始" in card["note"]
    assert (card["nonApprovable"], card["execute"], card["approve"]) == (
        False,
        True,
        "重新生成计划",
    )
    assert card["grants"] == ["provider_run", "literature"]
    assert card["resources"] == [
        "question_requirements_review.json",
        "agent_plan.json",
        "scientific_plan_review.json",
    ]


def test_an_analysis_the_plan_cannot_do_says_a_new_plan_may_stop_again() -> None:
    card = _card(CAPABILITY_GAP)

    assert card["code"] == CAPABILITY_GAP
    # The Planner may have declared the gap, which the host could not always
    # check, so the card says planning found it.
    assert card["title"] == "规划发现：题面要求的一项分析，这份计划做不到"
    assert "可以在对话中修改或去掉这项要求，或重新生成计划" in card["note"]
    assert "题面不改的话，新计划可能还会停在这里" in card["note"]
    assert (card["nonApprovable"], card["execute"], card["approve"]) == (
        False,
        True,
        "重新生成计划",
    )
    assert card["grants"] == ["provider_run", "literature"]
    assert card["resources"] == [
        "question_requirements_review.json",
        "agent_plan.json",
        "scientific_plan_review.json",
    ]


def test_a_plan_never_checked_against_its_question_offers_a_fresh_plan() -> None:
    # The owner's stops are exactly the three this file reads.
    assert set(QUESTION_REQUIREMENT_STOP_CODES) == set(QUESTION_STOPS)
    assert set(QUESTION_STOPS) <= set(PLAN_APPROVAL_STOPS)
    card = _card(UNREADABLE)

    assert card["code"] == UNREADABLE
    assert card["title"] == "这份计划没有按题面要求核对"
    assert "这份计划不能批准，分析尚未开始" in card["note"]
    assert "题面要求的核对记录读取或校验失败" in card["note"]
    assert (card["nonApprovable"], card["execute"], card["approve"]) == (
        False,
        True,
        "重新生成计划",
    )
    assert card["grants"] == ["provider_run", "literature"]
    # The record could not be read, so the card links only the plan and its review.
    assert card["resources"] == ["agent_plan.json", "scientific_plan_review.json"]


def test_without_a_reviewed_plan_the_card_names_no_resource() -> None:
    for code in QUESTION_STOPS:
        card = _card(code, run_id="")
        assert (card["resources"], card["offeredResources"]) == ([], 0)


def test_each_stop_starts_a_fresh_plan_the_host_keeps_metadata_only() -> None:
    actions = (STATIC / "js" / "screens-guided-pi-plan-actions.js").read_text(
        encoding="utf-8"
    )
    fresh = re.search(r"const FRESH_PLAN_CODES = new Set\(\[(.*?)\]\);", actions, re.S)

    assert fresh
    for code in QUESTION_STOPS:
        assert f"'{code}'" in fresh.group(1)
    assert set(QUESTION_STOPS) <= agent_routes._CANDIDATE_PLAN_WORKFLOW_CODES


@pytest.mark.parametrize(
    ("code", "zh"),
    [
        (NOT_COVERED, "这份计划没有任何步骤回答，此计划不能批准；请重新生成计划"),
        (
            CAPABILITY_GAP,
            "这份计划做不到，此计划不能批准；请修改这项要求或重新生成计划",
        ),
        (
            UNREADABLE,
            "这份计划未按题面核对，不能批准；请重新生成计划",
        ),
    ],
)
def test_the_workflow_aside_names_each_stop(code: str, zh: str) -> None:
    entry = _aside_entry(code)

    assert zh in entry
    assert "cannot be approved" in entry


@pytest.mark.parametrize("code", PLAN_APPROVAL_STOPS)
def test_every_plan_approval_stop_has_a_card_and_an_aside_line(code: str) -> None:
    card = _card(code)

    assert card["code"] == code
    assert card["title"]
    assert "不能批准" in card["note"]
    # Which plan "Generate fresh plan" makes is the route's choice for the
    # study, so no card promises one.
    assert "只读元数据" not in card["note"]
    assert "cannot be approved" in _aside_entry(code)
    # Every file the card links is one the run serves.
    assert card["resources"]
    assert set(card["resources"]) <= set(agent_runs._RUN_ARTIFACT_NAMES)


@pytest.mark.parametrize("code", PLAN_APPROVAL_STOPS)
def test_a_fresh_plan_from_a_stop_is_a_candidate_plan_only(
    code: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The card's "Generate fresh plan" sends no budget; the route recovers
    # candidate-plan authority from the workflow's next action, so the new plan
    # reads metadata only instead of patient rows before its review.  The route
    # may read the study, so the study store is this test's own.
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )
    monkeypatch.setattr(
        agent_routes,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(next_action_code=code)
        ),
    )

    assert agent_routes._candidate_plan_only_authorized(
        {"study_context_id": "study-1", "planner_start_mode": "fresh"}
    )

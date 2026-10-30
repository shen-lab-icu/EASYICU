"""A plan whose cohort leaves out a stated inclusion says why, and what to do.

The planning owner refuses approval with one code per remedy
(``population_compile.POPULATION_APPROVAL_STOPS``).  When nothing applies the
criterion as written, the plan card offers a fresh candidate plan, which the
host keeps metadata-only from that state.  When an extraction of the study's
own population would apply it, the card offers no plan action: the
conversation asks for the extraction, and binding it supersedes the plan.
The workflow aside names both.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from easyicu.research_agent.planning.population_compile import POPULATION_APPROVAL_STOPS
from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"
NOT_APPLIED = POPULATION_APPROVAL_STOPS["not_applied"]
REQUIRES_EXTRACTION = POPULATION_APPROVAL_STOPS["requires_extraction"]


def _card(code: str, *, run_id: str = "run_population_1", preparation=None, acceptance=None, language="zh") -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = r"""
let declared = null;
global.window = {
  EU_LANG: process.argv[6],
  EasyICU: { guidedPi: { declare: (_name, api) => { declared = api; }, optional: () => null } },
};
require(process.argv[1]);
const code = process.argv[2];
const runId = process.argv[3];
const preparation = JSON.parse(process.argv[4]);
const acceptance = JSON.parse(process.argv[5]);
const owner = declared.create({
  tr: (en, zh) => process.argv[6] === 'en' ? en : zh || en,
  esc: value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
  iconHtml: () => '',
  resourceButton: resource => `<a data-resource="${resource.artifact}"></a>`,
  sessionIsStale: () => false,
  busy: () => false,
  session: () => ({ binding: { run_id: runId } }),
  workflow: () => ({ next_action_code: code, plan_review_summary: { run_id: runId, population_preparation: preparation, population_acceptance: acceptance } }),
});
const confirmation = owner.workflowConfirmation();
const html = owner.workflowConfirmationHtml();
process.stdout.write(JSON.stringify({
  code: confirmation && confirmation.code,
  title: confirmation && confirmation.title,
  nonApprovable: Boolean(confirmation && confirmation.nonApprovable),
  grants: confirmation && confirmation.grants,
  execute: html.includes('data-gpi-confirm-action'),
  edit: html.includes('data-gpi-confirm-edit'),
  resources: (html.match(/data-resource="([^"]+)"/g) || []).map(row => row.slice(15, -1)),
  offeredResources: ((confirmation && confirmation.reviewResources) || []).length,
  html,
}));
"""
    owner = STATIC / "js" / "screens-guided-pi-confirmation.js"
    result = run_node(node, script, str(owner.resolve()), code, run_id, json.dumps(preparation), json.dumps(acceptance), language, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def test_a_criterion_nothing_applies_offers_a_fresh_candidate_plan() -> None:
    card = _card(NOT_APPLIED)

    assert card["code"] == NOT_APPLIED
    assert card["title"] == "研究写明的纳入条件按原文无法施加"
    assert (card["nonApprovable"], card["execute"], card["edit"]) == (False, True, True)
    assert card["grants"] == ["provider_run", "literature"]
    assert card["resources"] == ["agent_plan.json", "scientific_plan_review.json"]


def test_a_criterion_an_extraction_applies_offers_no_plan_action() -> None:
    card = _card(REQUIRES_EXTRACTION)

    assert card["code"] == REQUIRES_EXTRACTION
    assert card["title"] == "研究写明的纳入条件需要按本研究人群重新提取数据"
    assert (card["nonApprovable"], card["execute"], card["grants"]) == (True, False, [])
    assert card["resources"] == ["agent_plan.json", "scientific_plan_review.json"]


def test_a_bound_supported_proposal_shows_its_exact_change_and_explicit_action():
    card = _card(REQUIRES_EXTRACTION, preparation={
        "available": True,
        "criteria": [{"quote": "首次 ICU 入住 <script>alert(1)</script>", "proposed_by_system": True, "disposition": "requires_extraction"}],
        "changes": [{"field": "exclude_readmissions", "before": False, "after": True}],
    })
    assert card["execute"] is True
    assert card["grants"] == []
    assert "系统建议" in card["html"] and "否 → 是" in card["html"]
    assert "应用这些人群条件" in card["html"]
    assert "<script>" not in card["html"] and "&lt;script&gt;" in card["html"]


def test_an_unsupported_projection_explains_each_criterion_without_an_action():
    card = _card(REQUIRES_EXTRACTION, preparation={
        "available": False, "status": "blocked",
        "criteria": [{"quote": "年龄至少 18.5 岁", "proposed_by_system": False, "disposition": "requires_extraction", "reason": "population_preparation_bound_unrepresentable"}],
    })
    assert card["execute"] is False
    assert "研究原文" in card["html"] and "18.5" in card["html"]
    assert "无法保留这组精确界限" in card["html"]


def test_patient_first_and_hospital_first_mismatch_is_explained_without_approval():
    preparation = {
        "available": False, "status": "blocked",
        "criteria": [{"quote": "每位患者首次 ICU 入住", "disposition": "requires_extraction",
                      "reason": "population_preparation_patient_hospitalization_scope_mismatch"}],
    }
    card = _card(REQUIRES_EXTRACTION, preparation=preparation)
    assert card["execute"] is False
    assert "每次住院首次 ICU 不是同一人群" in card["html"]
    assert "核实患者入住顺序" in card["html"]


@pytest.mark.parametrize("kind,code", [("preparation", REQUIRES_EXTRACTION), ("acceptance", NOT_APPLIED)])
@pytest.mark.parametrize("language,system,interpretation,setting", [
    ("en", "System proposal", "Words from the question, system interpretation", "Study setting"),
    ("zh", "系统建议", "题目用语，系统解读", "研究设置"),
])
def test_population_cards_show_system_interpretations_and_exact_settings(
    kind, code, language, system, interpretation, setting,
):
    proposal = {
        "available": False,
        "criteria": [{
            "quote": "first 24 hours", "source": "outline", "proposed_by_system": True,
            "stated_by_study": False, "quote_found_in_question": True,
            "attribution": "typed_study_setting", "typed_study_setting": {"cohort.min_icu_los_hours": 24},
            "disposition": "requires_extraction" if kind == "preparation" else "not_applied",
        }],
    }
    card = _card(code, language=language, **{kind: proposal})
    assert system in card["html"]
    assert interpretation in card["html"]
    assert setting in card["html"]
    assert "cohort.min_icu_los_hours = 24" in card["html"]
    assert "研究原文" not in card["html"] and "Stated by the study" not in card["html"]
    assert card["execute"] is False


def test_without_a_reviewed_plan_the_card_names_no_resource() -> None:
    for code in (NOT_APPLIED, REQUIRES_EXTRACTION):
        card = _card(code, run_id="")
        assert (card["resources"], card["offeredResources"]) == ([], 0)


def test_only_the_criterion_nothing_applies_starts_a_fresh_plan() -> None:
    actions = (STATIC / "js" / "screens-guided-pi-plan-actions.js").read_text(encoding="utf-8")
    fresh = re.search(r"const FRESH_PLAN_CODES = new Set\(\[(.*?)\]\);", actions, re.S)
    replay = re.search(r"const FRESH_REPLAY_CODES = new Set\(\[(.*?)\]\);", actions, re.S)

    assert fresh and replay
    assert f"'{NOT_APPLIED}'" in fresh.group(1)
    assert f"'{REQUIRES_EXTRACTION}'" not in fresh.group(1)
    assert REQUIRES_EXTRACTION not in replay.group(1)
    # The host keeps a plan generated from either state metadata-only.
    assert set(POPULATION_APPROVAL_STOPS.values()) <= agent_routes._CANDIDATE_PLAN_WORKFLOW_CODES


@pytest.mark.parametrize(
    ("code", "zh"),
    [
        (REQUIRES_EXTRACTION, "按本研究人群重新提取数据后再生成计划，此计划不能批准"),
        (NOT_APPLIED, "按原文无法施加，此计划不能批准；请修改该条件或重新生成计划"),
    ],
)
def test_the_workflow_aside_names_each_stop(code: str, zh: str) -> None:
    aside = (STATIC / "js" / "screens-guided-pi-aside.js").read_text(encoding="utf-8")

    (entry,) = [line for line in aside.splitlines() if line.strip().startswith(f"{code}: tr(")]
    assert zh in entry
    assert "cannot be approved" in entry


def test_unapplied_criteria_show_scope_and_one_explicit_omission_choice():
    card = _card(NOT_APPLIED, acceptance={
        "available": True, "complete": False,
        "applied_criteria": ["年龄至少 18 岁"],
        "criteria": [
            {"id": "c1", "quote": "特定会诊条件", "reason": "population_kind_not_typed", "disposition": "not_applied", "accepted": True},
            {"id": "c2", "quote": "某种既往治疗 <script>", "reason": "population_concept_unavailable", "disposition": "not_applied", "proposed_by_system": True},
        ],
    })
    assert card["grants"] == []
    assert card["execute"] is True
    assert "计划实际筛选的人群" in card["html"] and "年龄至少 18 岁" in card["html"]
    assert "已接受此条局限" in card["html"] and "所需临床概念不可用" in card["html"]
    assert "当前待确认" in card["html"] and "接受当前这条局限" in card["html"]
    assert "仍须单独审阅并批准完整计划" in card["html"]
    assert "<script>" not in card["html"]


def test_an_invalid_omission_record_has_no_acceptance_action():
    card = _card(NOT_APPLIED, acceptance={"available": False, "record_invalid": True, "criteria": []})
    assert card["execute"] is False
    assert card["grants"] == []
    assert "已保存的选择无法核验" in card["html"]

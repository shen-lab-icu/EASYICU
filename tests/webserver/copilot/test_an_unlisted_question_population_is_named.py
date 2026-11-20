"""A population the question restricts to, with no condition EasyICU records, is named (S1737).

Sepsis, AKI and mechanical ventilation are conditions EasyICU records: they are
named concepts the plan must answer for.  Any other restriction (「在心衰患者中」,
"among patients with heart failure", 「ARDS 患者的」) has no concept to select on,
so when the reviewed population lists no criterion quoting it, the plan card
says 「问题限定了人群“…”，方案未包含对应的人群条件，请确认」 with a request to add it.
Never silent, never an approval stop.  Synthetic studies and cards only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver.pi_copilot import workflow as workflow_module
from easyicu.webserver.pi_copilot.plan_decisions import unlisted_question_population
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.routes import agent as agent_routes
from easyicu.webserver.study_intent import stated_population_restrictions
from tests.support.node import run_node

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"
_HEART_FAILURE = "在心衰患者中，ICU 入科后 24 小时最高乳酸与院内死亡有什么关联？"


@pytest.mark.parametrize(
    ("question", "restrictions"),
    [
        pytest.param(_HEART_FAILURE, (("心衰", False),), id="zh-zai-zhong"),
        pytest.param("ARDS 患者的乳酸与院内死亡有什么关联？", (("ARDS", False),), id="zh-de"),
        pytest.param(
            "Among patients with heart failure, is peak lactate associated with in-hospital death?",
            (("heart failure", False),),
            id="en-with",
        ),
        pytest.param("In trauma patients, is peak lactate associated with death?", (("trauma", False),), id="en-x"),
        pytest.param(
            "在脓毒症 ICU 患者中，ICU 入科后 0–24 小时的乳酸水平或峰值乳酸与院内死亡有什么关联？",
            (("脓毒症 ICU", True),),
            id="supported-sepsis",
        ),
        pytest.param("在 18 岁及以上的成人 ICU 患者中，乳酸与院内死亡", (), id="age-and-setting-only"),
        pytest.param("在首次入住 ICU 的患者中，乳酸与死亡", (), id="first-stay-only"),
        pytest.param("不研究心衰患者；在成人 ICU 患者中，乳酸与死亡", (), id="negated"),
    ],
)
def test_the_question_states_its_population_restrictions(question: str, restrictions) -> None:
    assert stated_population_restrictions(question) == restrictions


def _spec(*quotes: str) -> dict[str, Any]:
    return {"criteria": [{"criterion": {"kind": "condition_present", "quote": quote}} for quote in quotes]}


def test_an_unsupported_population_the_plan_does_not_list_is_named() -> None:
    study = {"question": _HEART_FAILURE}

    assert unlisted_question_population(study, None) == ["心衰"]
    assert unlisted_question_population(study, _spec("年龄 ≥ 18 岁")) == ["心衰"]
    assert unlisted_question_population(study, _spec("在心衰患者中")) == []
    # A recorded condition is the requirement layer's, not this line's.
    assert unlisted_question_population({"question": "在脓毒症 ICU 患者中，乳酸与院内死亡"}, None) == []


def _summary(question: str) -> dict[str, Any]:
    study = {
        "id": "study-unlisted-population",
        "revision": 2,
        "question": question,
        "active_job_id": None,
        "analysis_design": {"analysis_family": "association_study", "analysis_unit": "icu_stay", "variance_estimator": "model_based"},
    }
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-unlisted-population",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "scientific_configuration_sha256": "a" * 64,
            "artifact_names": ["agent_plan.json", "scientific_plan_review.json"],
        },
        plan_review_authority={
            "run_id": "run-unlisted-population",
            "resumable_here": True,
            "scientific_configuration_sha256": "a" * 64,
            "scientific_plan_review": {"status": "approvable", "findings": []},
        },
    )
    enriched = workflow_module._enrich_plan_review(  # noqa: SLF001
        snapshot, study=study, review={"artifact_payloads": {"agent_plan.json": {}}}
    )
    return dict(enriched.plan_review_summary or {})


def test_the_plan_under_review_names_the_unlisted_population() -> None:
    assert _summary(_HEART_FAILURE)["question_population_unlisted"] == ["心衰"]
    assert "question_population_unlisted" not in _summary("在成人 ICU 患者中，乳酸与院内死亡")


def _card(summary: dict[str, Any], lang: str = "zh", code: str = "operator_plan_approval_required") -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = r"""
const lang = process.argv[4];
global.window = { EU_LANG: lang };
const path = require('node:path');
const ownerDir = path.dirname(process.argv[1]);
require(path.join(ownerDir, 'screens-guided-pi-modules.js'));
require(path.join(ownerDir, 'screens-guided-pi-confirmation-data.js'));
require(process.argv[1]);
const code = process.argv[2];
const summary = JSON.parse(process.argv[3]);
const owner = window.EasyICU.guidedPi.require('confirmation').create({
  tr: (en, zh) => (lang === 'zh' ? zh || en : en),
  esc: value => String(value),
  iconHtml: () => '',
  resourceButton: resource => `<a data-resource="${resource.artifact}"></a>`,
  sessionIsStale: () => false,
  busy: () => false,
  session: () => ({ binding: { run_id: 'run-unlisted-population' } }),
  workflow: () => ({ next_action_code: code, plan_review_summary: { run_id: 'run-unlisted-population', ...summary } }),
});
process.stdout.write(owner.workflowConfirmationHtml());
"""
    owner = STATIC / "js" / "screens-guided-pi-confirmation.js"
    result = run_node(node, script, str(owner.resolve()), code, json.dumps(summary), lang, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def test_the_card_names_it_with_a_way_to_add_it() -> None:
    zh = _card({"question_population_unlisted": ["心衰"]})
    en = _card({"question_population_unlisted": ["heart failure"]}, lang="en")
    with_first_stay = _card({"question_population_unlisted": ["心衰"], "first_stay_setting_unlisted": "first_icu_stay"})

    assert "问题限定了人群“心衰”，方案未包含对应的人群条件，请确认。" in zh
    assert 'data-gpi-composer-prefill="请把问题中的人群限定“心衰”作为方案的人群条件加入。"' in zh
    assert "restricts the population to “heart failure”, but the plan lists no matching population condition" in en
    assert "问题限定了人群“心衰”" in with_first_stay and "研究设置（系统默认）" in with_first_stay
    assert "问题限定了人群" not in _card({})

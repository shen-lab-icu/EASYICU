"""A first-stay cohort setting the reviewed population does not list is named.

The study's typed cohort setting can keep each patient's first ICU stay (or the
first stay within each hospitalization) while the Planner's population lists
no first-stay criterion.  The plan card then names the setting, attributed to
the study setting, with a request to change it that the researcher sends; it
names nothing when the population already lists the criterion or the study
keeps every stay (S1639).  Synthetic studies and cards only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver.pi_copilot import workflow as workflow_module
from easyicu.webserver.pi_copilot.plan_decisions import unlisted_first_stay_setting
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"


def _spec(*kinds: str) -> dict[str, Any]:
    return {
        "schema_version": "easyicu.reviewed_population_spec_projection/1",
        "criteria": [
            {"criterion": {"kind": kind, "quote": kind}, "disposition": "applied_by_plan"}
            for kind in kinds
        ],
    }


# --- the owner ----------------------------------------------------------------------


@pytest.mark.parametrize("spec", [None, {}, _spec("age_minimum"), {"criteria": "not-a-list"}])
def test_a_patient_first_stay_setting_the_population_does_not_list_is_named(spec: Any) -> None:
    study = {"cohort": {"preset": "all_icu", "exclude_readmissions": True}}

    assert unlisted_first_stay_setting(study, spec) == "first_icu_stay"


def test_a_per_hospitalization_setting_is_named_as_such() -> None:
    study = {"cohort": {"preset": "all_icu", "first_icu_stay_per_hospitalization": True}}

    assert unlisted_first_stay_setting(study, _spec("age_minimum")) == "first_icu_stay_per_hospitalization"


@pytest.mark.parametrize("kind", ["first_icu_stay", "first_icu_stay_per_hospitalization"])
def test_a_setting_the_population_already_lists_is_not_repeated(kind: str) -> None:
    study = {"cohort": {"preset": "all_icu", "exclude_readmissions": True}}

    assert unlisted_first_stay_setting(study, _spec("age_minimum", kind)) is None


@pytest.mark.parametrize("cohort", [{"preset": "all_icu", "exclude_readmissions": False}, None, "garbage"])
def test_a_study_that_keeps_every_stay_names_nothing(cohort: Any) -> None:
    assert unlisted_first_stay_setting({"cohort": cohort}, None) is None


# --- the workflow states it for the plan under review --------------------------------------


def _summary(cohort: dict[str, Any], database: str | None = None) -> dict[str, Any]:
    study = {
        **({"data_source": {"database": database, "path": "/unused"}} if database else {}),
        "id": "study-unlisted-first-stay",
        "revision": 2,
        "question": "Is the first-day maximum creatinine associated with hospital death?",
        "active_job_id": None,
        "cohort": cohort,
        "analysis_design": {"analysis_family": "association_study", "analysis_unit": "icu_stay", "variance_estimator": "model_based"},
    }
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-unlisted-first-stay",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "scientific_configuration_sha256": "a" * 64,
            "artifact_names": ["agent_plan.json", "scientific_plan_review.json"],
        },
        plan_review_authority={
            "run_id": "run-unlisted-first-stay",
            "resumable_here": True,
            "scientific_configuration_sha256": "a" * 64,
            "scientific_plan_review": {"status": "approvable", "findings": []},
        },
    )
    enriched = workflow_module._enrich_plan_review(  # noqa: SLF001
        snapshot, study=study, review={"artifact_payloads": {"agent_plan.json": {}}}
    )
    return dict(enriched.plan_review_summary or {})


def test_the_plan_under_review_names_the_unlisted_setting() -> None:
    named = _summary({"preset": "all_icu", "exclude_readmissions": True})
    assert named["first_stay_setting_unlisted"] == "first_icu_stay"

    every_stay = _summary({"preset": "all_icu", "exclude_readmissions": False})
    assert "first_stay_setting_unlisted" not in every_stay


@pytest.mark.parametrize(
    ("database", "record_scope"),
    [("miiv", True), ("mimic_demo", True), ("aumc", False), (None, False)],
)
def test_a_mimic_first_stay_is_named_within_the_databases_record(database, record_scope) -> None:
    # B20: MIMIC selects the first stay within its own record (host recipe).
    named = _summary({"preset": "all_icu", "exclude_readmissions": True}, database)

    assert named["first_stay_within_database_record"] is record_scope


# --- the cards ------------------------------------------------------------------------


def _card(code: str, summary: dict[str, Any], lang: str = "zh") -> str:
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
  session: () => ({ binding: { run_id: 'run-unlisted-first-stay' } }),
  workflow: () => ({ next_action_code: code, plan_review_summary: { run_id: 'run-unlisted-first-stay', ...summary } }),
});
process.stdout.write(owner.workflowConfirmationHtml());
"""
    owner = STATIC / "js" / "screens-guided-pi-confirmation.js"
    result = run_node(
        node, script, str(owner.resolve()), code, json.dumps(summary), lang, check=False
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


LINE = "仅保留每位患者的首次 ICU 入住 · 研究设置（系统默认）"


@pytest.mark.parametrize(
    "code", ["operator_plan_approval_required", "plan_execution_upgrade_required"]
)
def test_the_plan_cards_name_the_unlisted_setting_with_a_way_to_change_it(code: str) -> None:
    html = _card(code, {"first_stay_setting_unlisted": "first_icu_stay"})

    assert LINE in html
    assert "在对话中修改" in html
    assert 'data-gpi-composer-prefill="不只保留每位患者的首次 ICU 入住，改为保留所有 ICU 入住。"' in html
    assert LINE not in _card(code, {})


def test_the_english_card_names_it_too() -> None:
    html = _card(
        "plan_execution_upgrade_required", {"first_stay_setting_unlisted": "first_icu_stay"}, lang="en"
    )

    assert "Keep each patient’s first ICU stay only · Study setting (system default)" in html
    assert "Change in the conversation" in html


def test_a_per_hospitalization_setting_is_not_called_per_patient() -> None:
    html = _card(
        "operator_plan_approval_required",
        {"first_stay_setting_unlisted": "first_icu_stay_per_hospitalization"},
    )

    assert "仅保留每次住院的首次 ICU 入住 · 研究设置（系统默认）" in html
    assert LINE not in html


def test_a_card_that_does_not_move_the_plan_names_no_setting() -> None:
    html = _card("failed_pipeline_requires_fresh_plan", {"first_stay_setting_unlisted": "first_icu_stay"})

    assert LINE not in html


def test_a_mimic_card_names_the_first_stay_within_the_databases_record() -> None:
    summary = {"first_stay_setting_unlisted": "first_icu_stay", "first_stay_within_database_record": True}

    zh = _card("plan_execution_upgrade_required", summary)
    en = _card("plan_execution_upgrade_required", summary, lang="en")

    assert "仅保留每位患者在数据库记录中的首次 ICU 入住（MIMIC 无院外/更早入住史） · 研究设置（系统默认）" in zh
    assert LINE not in zh
    assert "first ICU stay within this database’s record (MIMIC has no outside or earlier stay history)" in en

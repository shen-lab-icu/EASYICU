"""A design the launch gate cannot accept yet does not cost the rest of the setup.

Copilot saves the setup a researcher states in one update. At setup the launch
gate judges the analysis design, and a trajectory family arrives before its
design, which the plan decision declares from the reviewed plan. The gate
refused the family and the update was refused as a whole, so the population
the researcher stated (age at least 18, an ICU stay of at least 48 hours) was
never saved, and planning went ahead without it. The update now withholds the
design alone and saves the rest: the Planner's context states the population
as applied, and the reply the host finalizes says which design was not saved,
why, and who supplies it. Synthetic, case-neutral studies only.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.export_selection import (
    export_applied_selection,
)
from easyicu.research_agent.research_context.outbound import (
    outbound_safe_context_payload,
)
from easyicu.webserver.agent_pipeline_runs import (
    _exclusion_criteria,
    _inclusion_criteria,
    _research_user_preferences,
)
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)
from tests.support.node import run_node

APP_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver" / "pi_copilot" / "node_app"
)
CURRENT = {
    "id": "study-withheld-design",
    "revision": 1,
    "active_job_id": None,
    "question": "Which trajectories of marker-y occur in ICU stays with condition-x?",
}
POPULATION = {
    "label": "Adult ICU stays of at least 48 hours with condition-x",
    "review": "adults; ICU stay of at least 48 hours; condition-x",
    "age_min": 18,
    "min_icu_los_hours": 48,
}
SETUP = {
    "question": (
        "Among adult ICU stays of at least 48 hours with condition-x, which "
        "trajectories of marker-y occur, and how do they relate to outcome-z?"
    ),
    "purpose": "Describe trajectory classes of marker-y and their association with outcome-z.",
    "cohort": POPULATION,
    "analysis_design": {
        "analysis_family": "trajectory_clustering",
        "analysis_unit": "icu_stay",
        "variance_estimator": "model_based",
    },
}
MESSAGE = (
    "成年（≥18 岁）、ICU 住院至少 48 小时、有 condition-x 的入住，"
    "做 marker-y 的轨迹聚类，描述轨迹类别与 outcome-z 的关联。"
)
WORKFLOWS = {
    "needs_source": {
        "next_action_code": "study_setup_incomplete",
        "missing_setup_fields": ["data_source"],
        "study_setup_receipt": {"configuration": {"data_source": {}}},
    },
    "ready": {
        "next_action_code": "provider_ready_to_generate_plan",
        "missing_setup_fields": [],
        "study_setup_receipt": {"configuration": {"data_source": {"database": "synthetic"}}},
    },
}


def _update(monkeypatch, params: dict[str, Any], workflow: str = "needs_source"):
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: dict(CURRENT))
    monkeypatch.setattr(
        tool_module, "_workflow_snapshot", lambda *_args, **_kwargs: dict(WORKFLOWS[workflow])
    )
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw)) or {**CURRENT, **raw, "revision": 2},
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(
            session_id="pi-withheld-design",
            binding=AuthorityBinding(
                study_context_id=CURRENT["id"], study_revision=CURRENT["revision"]
            ),
        ),
        user_message=MESSAGE,
        allowed_actions={"configure"},
    )
    result = tool_module.execute_tool("easyicu_update_study_context", params, context)
    return result, writes, context.grant.consume_once("configure")


def test_a_trajectory_family_before_its_design_keeps_the_stated_population(monkeypatch):
    result, writes, grant = _update(monkeypatch, SETUP)

    assert (result["status"], result["code"]) == ("ok", "study_context_updated")
    assert len(writes) == 1 and "analysis_design" not in writes[0]
    assert {key: writes[0]["cohort"][key] for key in ("age_min", "min_icu_los_hours")} == {
        "age_min": 18,
        "min_icu_los_hours": 48,
    }
    assert writes[0]["question"] == SETUP["question"]
    withheld = result["details"]["unsaved_design"]
    assert (withheld["fields"], withheld["code"]) == (
        ["analysis_design"],
        "web_trajectory_design_required",
    )
    assert "NOT saved this turn: analysis_design (web_trajectory_design_required)" in result["summary"]
    assert grant == "consumed"


@pytest.mark.parametrize(
    "rest",
    [{}, {"question": CURRENT["question"]}, {"confirmations": {"trajectory_design_reviewed": True}}],
    ids=["design_alone", "with_an_unchanged_slot", "with_a_confirmation_only"],
)
def test_a_refused_design_with_nothing_else_to_change_saves_nothing(monkeypatch, rest):
    """Nothing changes, so the grant stays for a corrected design this turn."""

    params = {**rest, "analysis_design": SETUP["analysis_design"]}
    result, writes, grant = _update(monkeypatch, params)

    assert (result["status"], result["code"]) == ("blocked", "web_trajectory_design_required")
    assert result["owner"] == "easyicu.webserver.agent_pipeline_runs.analysis_design"
    assert (writes, grant) == ([], "granted")


def test_the_saved_population_reaches_the_planner_and_the_population_audit(monkeypatch):
    """The two bounds arrive as applied contracts, not only in the saved setup."""

    _, writes, _ = _update(monkeypatch, SETUP)
    study = {**CURRENT, **writes[0]}
    preferences = _research_user_preferences(study, source_selection_basis="export_contract")
    context = build_research_context(
        research_question=SETUP["question"],
        cohort=pd.DataFrame({"stay_id": [1, 2], "age": [34.0, 61.0]}),
        cohort_name="synthetic",
        database="synthetic",
        inclusion_criteria=_inclusion_criteria(study),
        exclusion_criteria=_exclusion_criteria(study),
        user_preferences=preferences,
    )

    applied = ["age range: 18 to *", "minimum ICU length of stay: 48 hours"]
    payload = outbound_safe_context_payload(context)
    assert payload["cohort"]["inclusion_contract"] == applied
    stated = json.loads(payload["study_preferences"]["data_constraints"])["cohort"]
    assert (stated["age_min"], stated["min_icu_los_hours"]) == (18, 48)
    # The reader planning and the population audit share.
    selection = export_applied_selection(context)
    assert selection.recorded and list(selection.known_applied.inclusion) == applied


def _finalized_reply(result: dict[str, Any], params: dict[str, Any], language: str) -> str:
    node = shutil.which("node")
    if not node or not (APP_DIR / "node_modules").is_dir():
        pytest.skip("Pinned Pi Node runtime is unavailable")
    module = APP_DIR / "src" / "post-tool-finalization.mjs"
    script = f"""
      import {{ hostPostToolFinalization }} from {json.dumps(module.as_uri())};
      const model = {{ api: 'openai-completions', provider: 'test', id: 'test' }};
      const user = {{ role: 'user', content: [{{ type: 'text', text: {json.dumps(MESSAGE)} }}] }};
      const assistant = {{ role: 'assistant', content: [{{
        type: 'toolCall', id: 'call-update', name: 'easyicu_update_study_context',
        arguments: {json.dumps(params)},
      }}] }};
      const result = {{ role: 'toolResult', toolCallId: 'call-update',
        toolName: 'easyicu_update_study_context', isError: false, content: [],
        details: {json.dumps(result)} }};
      const stream = hostPostToolFinalization(model, {{ messages: [user, assistant, result] }}, {json.dumps(language)});
      if (!stream) throw new Error('expected a host-finalized reply');
      const message = await stream.result();
      console.log(JSON.stringify(message.content[0].text));
    """
    completed = run_node(node, script, module=True, cwd=APP_DIR, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


@pytest.mark.parametrize("workflow", ["needs_source", "ready"])
def test_the_reply_says_which_design_was_not_saved_why_and_who_supplies_it(monkeypatch, workflow):
    result, _, _ = _update(monkeypatch, SETUP, workflow)

    zh = _finalized_reply(result, SETUP, "zh")
    assert "分析设计这次没有保存：轨迹聚类需要先有经审阅的轨迹设计" in zh
    assert "候选研究计划会提出轨迹设计，你在审阅中批准后" in zh
    en = _finalized_reply(result, SETUP, "en")
    assert "The analysis design was not saved: trajectory clustering needs a reviewed trajectory design" in en
    assert "The candidate research plan proposes one" in en
    if workflow == "needs_source":
        # It is told before the next step, which stays the reply's last part.
        assert zh.index("分析设计这次没有保存") < zh.index("**下一步：**")


def test_a_reply_without_a_withheld_design_says_nothing_of_one(monkeypatch):
    params = {key: value for key, value in SETUP.items() if key != "analysis_design"}
    result, _, _ = _update(monkeypatch, params, "ready")

    assert "unsaved_design" not in result["details"]
    assert "没有保存" not in _finalized_reply(result, params, "zh")

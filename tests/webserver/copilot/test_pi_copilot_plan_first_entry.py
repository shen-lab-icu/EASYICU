"""Question/source entry must yield to the workflow, not a setup questionnaire."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.pi_copilot import tools as study_tools
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding, PiSessionRecord, ToolExecutionContext,
)


APP = Path(__file__).parents[3] / "src/easyicu/webserver/pi_copilot/node_app"


def _node(script):
    node = shutil.which("node")
    if not node or not (APP / "node_modules").is_dir():
        pytest.skip("Pinned Pi Node runtime is unavailable")
    result = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        cwd=APP, capture_output=True, text=True, timeout=30, check=True,
    )
    return json.loads(result.stdout)


def _workflow(source=True):
    study = {"id": "entry-test", "question": "我想研究乳酸与院内死亡的关系"}
    if source:
        study["data_source"] = {"database": "miiv", "path": "/test/miiv"}
    return build_research_workflow_snapshot(
        study=study, active_export_present=False, active_job=None, latest_run=None,
    ).model_dump(mode="json")


def _finalize(workflow, args=None, code="study_context_updated", status="ok"):
    args = args or {"question": "我想研究乳酸与院内死亡的关系"}
    context = {"messages": [
        {"role": "assistant", "content": [{
            "type": "toolCall", "id": "c1", "name": "easyicu_update_study_context",
            "arguments": args,
        }]},
        {"role": "toolResult", "toolCallId": "c1",
         "toolName": "easyicu_update_study_context", "isError": status != "ok",
         "details": {"status": status, "code": code, "details": {"workflow": workflow}}},
    ]}
    return _node(f"""
        import {{hostPostToolFinalization}} from {json.dumps((APP / 'src/post-tool-finalization.mjs').as_uri())};
        const stream = hostPostToolFinalization(
          {{api: 'test', provider: 'test', id: 'test'}}, {json.dumps(context)}, 'zh');
        console.log(JSON.stringify(stream ? await stream.result() : null));
    """)


@pytest.mark.parametrize("args", [
    {"question": "我想研究乳酸与院内死亡的关系"},
    {"question": "我想研究乳酸与院内死亡的关系", "primary_exposure": "乳酸",
     "outcome": "院内死亡", "analysis_goal": "评估二者关系", "time_window": {"hours": 24}},
    {"data_source": {"database": "miiv", "path": "/test/miiv"}},
])
def test_plan_ready_receipt_stops_questionnaire_even_when_proposed_slots_were_omitted(args):
    workflow = _workflow()
    assert workflow["next_action_code"] == "provider_ready_to_generate_plan"
    assert "outcome" in workflow["missing_setup_fields"]
    result = _finalize(workflow, args)
    assert result is not None
    text = result["content"][0]["text"]
    assert "生成候选研究计划" in text
    assert "尚未开始数据提取或分析" in text
    assert "确认主要" not in text
    assert "下一步：" not in text  # The host supplies one actionable plan card.
    assert result["usage"]["totalTokens"] == 0


def test_no_source_still_requires_source_selection_not_a_plan():
    result = _finalize(_workflow(source=False))
    assert "请先选择数据库" in result["content"][0]["text"]


@pytest.mark.parametrize("code", [
    "plan_scientific_changes_required", "operator_plan_approval_required",
    "plan_configuration_superseded", "study_setup_incomplete",
])
def test_other_workflow_states_are_not_overridden_with_plan_readiness(code):
    workflow = {**_workflow(), "next_action_code": code}
    assert _finalize(workflow, {"outcome": "院内死亡"}) is None


def test_rejected_update_does_not_claim_readiness():
    assert _finalize(_workflow(), status="blocked", code="study_outcome_confirmation_required") is None


def test_source_transition_has_no_prepare_before_plan_fallback():
    main = (APP / "src/main.mjs").read_text()
    transition = main.split(': turnIntent === "advance_after_data_source_confirmation"', 1)[1].split(': turnIntent ===', 1)[0]
    assert "data-preparation confirmation, not a study plan" not in transition
    assert "after the data package and local preflight are ready" not in transition
    assert "Do not call a tool" in transition


def test_workflow_progress_orders_plan_before_extraction():
    ids = [stage["id"] for stage in _workflow()["stages"]]
    assert ids.index("setup") < ids.index("plan") < ids.index("extraction") < ids.index("analysis")


def test_opening_update_preserves_question_without_silently_confirming_design(monkeypatch):
    question = "我想研究 ICU 患者乳酸与院内死亡的关系。"
    current = {
        "id": "entry-owner-test", "revision": 1, "question": "",
        "data_source": {"database": "miiv", "path": "/test/miiv"},
    }
    writes = []
    monkeypatch.setattr(study_tools, "_bound_context", lambda _binding: dict(current))
    monkeypatch.setattr(
        study_tools.study_contexts, "upsert_context",
        lambda row, **_kw: writes.append(dict(row)) or {**current, **row, "revision": 2},
    )
    monkeypatch.setattr(
        study_tools, "_workflow_snapshot",
        lambda _ctx, *, study_override=None: build_research_workflow_snapshot(
            study=study_override, active_export_present=False,
            active_job=None, latest_run=None,
        ).model_dump(mode="json"),
    )
    args = {"question": question, "outcome": "院内死亡",
            "primary_exposure": "乳酸", "analysis_goal": "评估二者关系"}
    result = study_tools.execute_tool(
        "easyicu_update_study_context", args,
        ToolExecutionContext(
            session=PiSessionRecord(
                session_id="pi-entry-owner-test",
                binding=AuthorityBinding(study_context_id=current["id"], study_revision=1),
            ),
            user_message=question, allowed_actions={"configure"},
        ),
    )
    assert result["code"] == "study_context_updated"
    assert "ask the user to select each one explicitly" not in result["summary"]
    assert writes[0]["question"] == question
    for slot in ("outcome", "primary_exposure", "analysis_goal"):
        assert not writes[0].get(slot)
        assert slot in result["details"]["omitted_unconfirmed_fields"]
    workflow = result["details"]["workflow"]
    assert workflow["next_action_code"] == "provider_ready_to_generate_plan"
    assert _finalize(workflow, args) is not None

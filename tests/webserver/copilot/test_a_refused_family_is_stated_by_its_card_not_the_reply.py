"""A study family v1 does not offer on its source is stated by its card, not the reply.

pilot1 round 2b (10-22), R1: after the researcher chose eICU for a Cox
question, the plan step was blocked with ``study_family_database_out_of_scope``
and the host card stated the refusal and what the study could do instead.
The model's reply to the data-source transition restated it in its own words
and leaked the gap's codes: 「`survival` 分析族目前仅支持 `miiv` 数据库」,
then 「等待主界面提供针对该能力限制的后续选项」; on a plan the source made ready
it wrote 「host 将显示下一步操作」.  After the data-source click a host card
follows either way, so the host now finalizes that reply itself, saying only
which data the study uses.  Another host transition -- the plan-generation
click, which must call the run tool -- a source confirmation followed by no
card, and an ordinary researcher turn are still answered by the model.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import pytest

from tests.support.node import run_node

APP_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver" / "pi_copilot" / "node_app"
)
MODULE = APP_DIR / "src" / "post-tool-finalization.mjs"
QUESTION = "在 eICU 中，用 Cox 回归分析入 ICU 首日最高乳酸与 28 天生存的关系。"
GAP = {
    "code": "study_family_database_out_of_scope",
    "family": "survival",
    "database": "eicu",
    "supported_databases": ["miiv"],
}
SOURCE = {"source_id": "src_synthetic", "label": "eICU-CRD", "database": "eicu"}


def _workflow(
    *, gap: bool = True, source: Optional[dict] = None, code: str = "provider_ready_to_generate_plan",
) -> dict[str, Any]:
    return {
        "next_action_code": "study_family_database_out_of_scope" if gap else code,
        "study_setup_receipt": {"configuration": {"data_source": source or SOURCE}},
        "study_family_gap": GAP if gap else None,
    }


def _owner_receipts(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "status": "ok",
        "code": "easyicu_research_workflow_projected",
        "details": {"workflow": workflow},
    }]


def _finalized(messages_js: str, language: str = "zh") -> Optional[str]:
    node = shutil.which("node")
    if not node or not (APP_DIR / "node_modules").is_dir():
        pytest.skip("Pinned Pi Node runtime is unavailable")
    script = f"""
      import {{ hostPostToolFinalization, OWNER_CONTEXT_MARKER }} from {json.dumps(MODULE.as_uri())};
      const model = {{ api: 'openai-completions', provider: 'test', id: 'test' }};
      const messages = {messages_js};
      const stream = hostPostToolFinalization(model, {{ messages }}, {json.dumps(language)});
      if (!stream) {{ console.log('null'); }}
      else {{
        const message = await stream.result();
        console.log(JSON.stringify(message.content[0].text));
      }}
    """
    completed = run_node(node, script, module=True, cwd=APP_DIR, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


_SOURCE_CONFIRMED = (
    "\\n\\n[EASYICU_HOST_TRANSITION_V1]\\n" + "[EASYICU_DATA_SOURCE_CONFIRMED_V1]"
    + "\\nThe host has confirmed the study data source."
)
_PLAN_GENERATION = (
    "\\n\\n[EASYICU_HOST_TRANSITION_V1]\\nThe user activated the host-owned "
    "provider_ready_to_generate_plan control. Call easyicu_run exactly once."
)


def _transition(workflow: dict[str, Any], *, tail: str = _SOURCE_CONFIRMED) -> str:
    # The prompts main.mjs builds for a host transition.
    head = "EasyICU 已确认所选数据来源，请在当前对话中继续。"
    receipts = json.dumps(_owner_receipts(workflow), ensure_ascii=False)
    return f"""[
      {{ role: 'user', content: [{{ type: 'text', text: {json.dumps(QUESTION)} }}] }},
      {{ role: 'assistant', content: [{{ type: 'text', text: '研究问题已保存。' }}] }},
      {{ role: 'user', content: [{{ type: 'text', text:
        {json.dumps(head)} + OWNER_CONTEXT_MARKER + {json.dumps(receipts)} + "{tail}" }}] }},
    ]"""


def _update(workflow: dict[str, Any]) -> str:
    details = json.dumps({
        "status": "ok", "code": "study_context_updated", "details": {"workflow": workflow},
    }, ensure_ascii=False)
    return f"""[
      {{ role: 'user', content: [{{ type: 'text', text: {json.dumps(QUESTION)} }}] }},
      {{ role: 'assistant', content: [{{ type: 'toolCall', id: 'call-update',
        name: 'easyicu_update_study_context', arguments: {{ question: {json.dumps(QUESTION)} }} }}] }},
      {{ role: 'toolResult', toolCallId: 'call-update', toolName: 'easyicu_update_study_context',
        isError: false, content: [], details: {details} }},
    ]"""


@pytest.mark.parametrize("gap", [True, False], ids=["refusal-card", "plan-card"])
def test_the_source_transition_says_only_which_data_the_study_uses(gap: bool) -> None:
    assert _finalized(_transition(_workflow(gap=gap))) == "已确认使用 eICU-CRD 数据。"
    assert _finalized(_transition(_workflow(gap=gap)), "en") == "The study now uses the eICU-CRD data."
    unlabeled = _workflow(gap=gap, source={"database": "eicu"})
    assert _finalized(_transition(unlabeled)) == "已确认数据来源。"


def test_a_question_saved_onto_a_bound_source_says_only_that_it_is_saved() -> None:
    assert _finalized(_update(_workflow())) == "研究问题已保存。"
    assert _finalized(_update(_workflow()), "en") == "The research question is saved."


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            lambda: _transition(_workflow(gap=False), tail=_PLAN_GENERATION), id="plan-generation",
        ),
        pytest.param(
            lambda: _transition(_workflow(gap=False, code="study_setup_incomplete")), id="no-card",
        ),
        pytest.param(lambda: _transition(_workflow(), tail=""), id="researcher-turn"),
    ],
)
def test_other_turns_are_answered_by_the_model(messages) -> None:
    assert _finalized(messages()) is None

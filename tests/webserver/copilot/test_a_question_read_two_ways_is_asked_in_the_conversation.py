"""A question the conversation and the host read as two families is asked, then recorded.

When the conversation's stated family and the host's word reading differ
(``question_family_check``), the plan step is blocked with
``study_family_reading_disagreement`` and the workflow carries the two
offered readings.  The host writes the reply that asks -- after the
data-source click or after the question is saved -- with the two readings as
choices; a click on one is recorded by the host through the setup owner,
with no model in between, and the reply then says the choice is recorded.
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
QUESTION = "eICU 中，入 ICU 首日最高乳酸与生存出院有关吗？"
ASSOCIATION = "按关联研究做（暴露与结局是否相关、相关多强）"
SURVIVAL = "按生存分析做（时间到事件）"


def _workflow(code: str = "study_family_reading_disagreement") -> dict[str, Any]:
    gap = {
        "code": "study_family_reading_disagreement",
        "model_family": "association_study",
        "host_family": "survival",
        "database": "eicu",
        "choices": [
            {"family": "association_study", "zh": ASSOCIATION, "en": "Treat it as an association study"},
            {"family": "survival", "zh": SURVIVAL, "en": "Treat it as a survival analysis (time to event)"},
        ],
    }
    return {
        "next_action_code": code,
        "study_setup_receipt": {"configuration": {
            "data_source": {"label": "eICU-CRD", "database": "eicu"},
            "question_family_check": {
                "model_family": "association_study", "words": "与生存出院有关吗",
                "host_family": "survival", "status": "host_disagrees",
            },
        }},
        "study_family_gap": gap if code == "study_family_reading_disagreement" else None,
    }


def _receipts(workflow: dict[str, Any]) -> str:
    return json.dumps([{
        "status": "ok", "code": "easyicu_research_workflow_projected",
        "details": {"workflow": workflow},
    }], ensure_ascii=False)


def _run(messages_js: str, language: str = "zh") -> Optional[dict[str, Any]]:
    node = shutil.which("node")
    if not node or not (APP_DIR / "node_modules").is_dir():
        pytest.skip("Pinned Pi Node runtime is unavailable")
    script = f"""
      import {{ hostPostToolFinalization, OWNER_CONTEXT_MARKER, DATA_SOURCE_CONFIRMED_MARKER }}
        from {json.dumps(MODULE.as_uri())};
      const model = {{ api: 'openai-completions', provider: 'test', id: 'test' }};
      const messages = {messages_js};
      const stream = hostPostToolFinalization(model, {{ messages }}, {json.dumps(language)});
      if (!stream) {{ console.log('null'); }}
      else {{
        const message = await stream.result();
        console.log(JSON.stringify(message.content[0]));
      }}
    """
    completed = run_node(node, script, module=True, cwd=APP_DIR, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


def _user(text_js: str) -> str:
    return f"{{ role: 'user', content: [{{ type: 'text', text: {text_js} }}] }}"


def _with_owner_context(text: str, workflow: dict[str, Any], tail_js: str = "''") -> str:
    return f"{json.dumps(text, ensure_ascii=False)} + OWNER_CONTEXT_MARKER + {json.dumps(_receipts(workflow), ensure_ascii=False)} + {tail_js}"


def _update_result(workflow: dict[str, Any], extra: Optional[dict] = None) -> str:
    details = json.dumps({
        "status": "ok", "code": "study_context_updated",
        "details": {"workflow": workflow, **(extra or {})},
    }, ensure_ascii=False)
    return f"""
      {{ role: 'assistant', content: [{{ type: 'toolCall', id: 'call-update',
        name: 'easyicu_update_study_context', arguments: {{}} }}] }},
      {{ role: 'toolResult', toolCallId: 'call-update', toolName: 'easyicu_update_study_context',
        isError: false, content: [], details: {details} }}"""


def test_the_source_transition_asks_which_reading_the_question_means() -> None:
    transition = _with_owner_context(
        "EasyICU 已确认所选数据来源。", _workflow(),
        "'\\n\\n[EASYICU_HOST_TRANSITION_V1]\\n' + DATA_SOURCE_CONFIRMED_MARKER",
    )
    reply = _run(f"[{_user(json.dumps(QUESTION, ensure_ascii=False))}, {_user(transition)}]")

    assert reply["type"] == "text"
    assert reply["text"] == (
        "已确认使用 eICU-CRD 数据。这个问题可以按两种研究来做，请选一种：\n\n"
        f"**下一步：**\n- {ASSOCIATION}\n- {SURVIVAL}"
    )


def test_a_saved_question_read_two_ways_asks_the_same() -> None:
    reply = _run(f"[{_user(json.dumps(QUESTION, ensure_ascii=False))}, {_update_result(_workflow())}]")

    assert reply["text"].startswith("研究问题已保存。这个问题可以按两种研究来做，请选一种：")
    assert reply["text"].endswith(f"- {ASSOCIATION}\n- {SURVIVAL}")


@pytest.mark.parametrize(("sent", "family"), [(ASSOCIATION, "association_study"), (SURVIVAL + "。", "survival")])
def test_a_clicked_reading_is_recorded_by_the_host(sent: str, family: str) -> None:
    turn = _with_owner_context(sent, _workflow())
    call = _run(f"[{_user(json.dumps(QUESTION, ensure_ascii=False))}, {_user(turn)}]")

    assert call["type"] == "toolCall" and call["name"] == "easyicu_update_study_context"
    assert call["arguments"] == {"question_family": {"family": family, "words": "与生存出院有关吗"}}


def test_other_words_on_an_open_question_are_the_models() -> None:
    turn = _with_owner_context("我想比较出院时是否存活", _workflow())

    assert _run(f"[{_user(json.dumps(QUESTION, ensure_ascii=False))}, {_user(turn)}]") is None


def test_the_reply_after_a_choice_says_it_is_recorded() -> None:
    recorded = {"question_family_check": {"status": "researcher_chose"}}
    reply = _run(
        f"[{_user(json.dumps(ASSOCIATION, ensure_ascii=False))}, "
        f"{_update_result(_workflow('provider_ready_to_generate_plan'), recorded)}]"
    )

    assert reply["text"].startswith("已按你的选择记录。研究问题和数据源已就绪")

"""The host's first-turn data-source reply, as a real turn delivers it.

The initial question is finalized by the host (post-tool-finalization.mjs)
without a second provider call; these contracts cover the named official demo
and the preloaded data-source catalog.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[3] / "src" / "easyicu" / "webserver" / "pi_copilot" / "node_app"


def _finalize_initial_question_with_preloaded_catalog(question: str) -> str:
    """Run the initial-question finalization the way a real turn delivers it.

    The data-source catalog is not a tool result in a real first turn: main.mjs
    preloads it and appends it to the researcher's text as owner context.
    """

    node = shutil.which("node")
    if not node or not (APP_DIR / "node_modules").is_dir():
        pytest.skip("Pinned Pi Node runtime is unavailable")
    module = APP_DIR / "src" / "post-tool-finalization.mjs"
    catalog = {
        "status": "ok", "code": "easyicu_data_sources_listed", "summary": "listed", "owner": "test",
        "details": {
            "supported_databases": [
                {"database": "miiv", "display_label": "MIMIC-IV v3.1", "reference_release": "3.1"},
                {"database": "eicu", "display_label": "eICU v2.0", "reference_release": "2.0"},
                {"database": "aumc", "display_label": "AmsterdamUMCdb v1.0.2", "reference_release": "1.0.2"},
                {"database": "hirid", "display_label": "HiRID v1.1.1", "reference_release": "1.1.1"},
                {"database": "mimic", "display_label": "MIMIC-III v1.4", "reference_release": "1.4"},
                {"database": "sic", "display_label": "SICdb v1.0.6", "reference_release": "1.0.6"},
            ],
            "official_demos": [
                {"source_id": "mimic_iv_demo_v2_2", "label": "MIMIC-IV Clinical Database Demo", "database": "miiv", "version": "2.2"},
                {"source_id": "eicu_demo_v2_0_1", "label": "eICU Collaborative Research Database Demo", "database": "eicu", "version": "2.0.1"},
            ],
        },
    }
    script = f"""
      import {{ hostPostToolFinalization, OWNER_CONTEXT_MARKER }} from {json.dumps(module.as_uri())};
      const model = {{ api: 'openai-completions', provider: 'test', id: 'test' }};
      const prompt = {json.dumps(question)}
        + '\\n\\n[EASYICU_INTERNAL_RESPONSE_LANGUAGE_V1]\\nRespond in Simplified Chinese.'
        + OWNER_CONTEXT_MARKER + JSON.stringify([{json.dumps(catalog)}]);
      const user = {{ role: 'user', content: [{{ type: 'text', text: prompt }}] }};
      const assistant = {{ role: 'assistant', content: [{{
        type: 'toolCall', id: 'call-update', name: 'easyicu_update_study_context',
        arguments: {{ question: {json.dumps(question)} }},
      }}] }};
      const result = {{ role: 'toolResult', toolCallId: 'call-update', toolName: 'easyicu_update_study_context',
        isError: false, content: [], details: {{ status: 'ok', code: 'study_context_updated', details: {{ workflow: {{
          next_action_code: 'study_setup_incomplete', missing_setup_fields: ['data_source', 'outcome'],
          study_setup_receipt: {{ configuration: {{ data_source: {{}} }} }},
        }} }} }} }};
      const stream = hostPostToolFinalization(model, {{ messages: [user, assistant, result] }}, 'zh');
      if (!stream) throw new Error('expected data-source finalization');
      const message = await stream.result();
      console.log(JSON.stringify(message.content[0].text));
    """
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        cwd=APP_DIR, text=True, capture_output=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


def test_initial_question_naming_an_official_demo_points_to_its_card() -> None:
    """「在 eICU demo 数据里…」 is not answered with 「请先选择数据库」.

    The browser's data-source card offers the named demo with one click, so
    the host's first-turn text names that demo and points to the card instead
    of listing databases (whose chips would be a second prompt for one action).
    Naming the demo still binds nothing.
    """

    text = _finalize_initial_question_with_preloaded_catalog(
        "在 eICU demo 数据里，评估入 ICU 第一个 24 小时内的最高乳酸与 ICU 死亡率之间的关系"
    )
    assert "你的问题指定了 eICU Collaborative Research Database Demo v2.0.1（仅官方 Demo 数据）" in text
    assert "**下一步：**在下方数据源卡片点击「用于本次会话」" in text
    assert "确认数据源不等于批准分析" in text
    assert "请先选择数据库" not in text
    assert "\n- " not in text


def test_initial_question_reads_the_preloaded_catalog_for_database_choices() -> None:
    """A question that names no demo lists every supported database.

    The catalog arrives as preloaded owner context in a real turn; reading it
    turns the two generic fallback choices into the six exact databases.
    """

    text = _finalize_initial_question_with_preloaded_catalog("我想研究乳酸与院内死亡的关系")
    assert "请先选择数据库" in text
    assert text.count("\n- 使用 ") == 6
    assert "- 使用 eICU v2.0" in text
    assert "查看并选择 EasyICU 支持的数据库" not in text
    # A bare database name is not a demo; two named demos are ambiguous.
    for question in ("在 eICU 数据里评估乳酸", "对比 mimic-iv demo 和 eicu demo 的乳酸分布"):
        assert "请先选择数据库" in _finalize_initial_question_with_preloaded_catalog(question)


def test_owner_context_marker_is_shared_with_the_prompt_builder() -> None:
    entrypoint = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")
    finalization = (APP_DIR / "src" / "post-tool-finalization.mjs").read_text(encoding="utf-8")
    marker = '"\\n\\n[EASYICU_CURRENT_TURN_OWNER_CONTEXT_V1]\\n"'
    assert f"export const OWNER_CONTEXT_MARKER = {marker};" in finalization
    assert f"const OWNER_CONTEXT_MARKER = {marker};" in entrypoint
    assert "`\\n\\n[EASYICU_CURRENT_TURN_OWNER_CONTEXT_V1]\\n${boundedText(JSON.stringify(receipts), 24000)}`" in entrypoint
    # The researcher's text precedes every host section.
    assert "return `${message}${HOST_LANGUAGE_MARKER}${requirement}${currentContext}${sourceContext}${transition}`;" in entrypoint

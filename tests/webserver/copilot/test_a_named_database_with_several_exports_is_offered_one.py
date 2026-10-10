"""A named database with several registered exports is offered one of them.

The source listing recommends the most complete EasyICU export of the database
the researcher names.  Equally complete exports are told apart by the active
one, then by the latest generation time; when nothing tells them apart, or one
has more stays and another more modules, the listing numbers them for the
researcher to choose.  The host's first reply then lists them, and says the
database is unregistered only when no export of it is registered.

Synthetic registries and catalogs only; no path reaches the reply.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext
from easyicu.webserver.pi_copilot.source_recommendation import recommend_registered_export
from easyicu.webserver.pi_copilot.turn_authority import (
    explicitly_confirms_easyicu_registered_source,
)
from tests.support.node import run_node

APP_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "easyicu"
    / "webserver"
    / "pi_copilot"
    / "node_app"
)
MODULES = ["demographics", "outcome", "vitals"]


def _export(source_id: str, *, generated: str, stays: int = 94458, modules: int = 3) -> dict[str, Any]:
    return {
        "id": source_id,
        "path": f"/private/{source_id}",
        "label": f"MIMIC-IV {source_id} run",
        "database": "miiv",
        "generated": generated,
        "ok": True,
        "modules": MODULES[:modules],
        "summary": {"stays": stays, "modules": modules},
    }


def _listing(
    monkeypatch: pytest.MonkeyPatch, exports: list[dict[str, Any]], *, active: str = ""
) -> dict[str, Any]:
    monkeypatch.setattr(
        tool_module.sources,
        "load_registry",
        lambda: {"active_path": f"/private/{active}" if active else "", "sources": exports},
    )
    result = tool_module.execute_tool(
        "easyicu_list_data_sources",
        {"database": "miiv"},
        ToolExecutionContext(
            session=PiSessionRecord(session_id="pi-several-exports"),
            user_message="用 MIMIC-IV 研究乳酸与院内死亡",
        ),
    )
    assert result["code"] == "easyicu_data_sources_listed"
    return result["details"]


# --- the listing ---------------------------------------------------------------


def test_equally_complete_exports_recommend_the_active_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    details = _listing(
        monkeypatch,
        [
            _export("src_older", generated="2026-10-01T08:00:00"),
            _export("src_newer", generated="2026-10-08T08:00:00"),
        ],
        active="src_older",
    )

    assert details["recommended_source"]["source_id"] == "src_older"
    assert details["recommended_source"]["selection_reason"] == "active_local_dataset"
    assert details["recommended_source"]["auto_select_for_exact_database_request"] is True
    assert details["registered_source_choices"] == []


def test_equally_complete_exports_without_an_active_one_recommend_the_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    details = _listing(
        monkeypatch,
        [
            _export("src_newer", generated="2026-10-08T08:00:00"),
            _export("src_older", generated="2026-10-01T08:00:00"),
        ],
    )

    assert details["recommended_source"]["source_id"] == "src_newer"
    assert details["recommended_source"]["selection_reason"] == "latest_local_dataset"
    assert details["registered_source_choices"] == []


def test_exports_nothing_tells_apart_are_numbered_for_the_researcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    details = _listing(
        monkeypatch,
        [
            _export("src_first", generated="2026-10-08T08:00:00"),
            _export("src_second", generated="2026-10-08T08:00:00"),
        ],
    )

    assert details["recommended_source"] is None
    assert details["registered_source_choices"] == [
        {
            "choice": 1,
            "source_id": "src_first",
            "label": "MIMIC-IV v3.1",
            "generated_date": "2026-10-08",
            "stays": 94458,
            "module_count": 3,
        },
        {
            "choice": 2,
            "source_id": "src_second",
            "label": "MIMIC-IV v3.1",
            "generated_date": "2026-10-08",
            "stays": 94458,
            "module_count": 3,
        },
    ]
    assert "registered_export" in details["source_modes"]
    assert "/private" not in json.dumps(details["registered_source_choices"])


def test_an_export_without_a_generation_time_is_not_the_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    details = _listing(
        monkeypatch,
        [
            _export("src_dated", generated="2026-10-08T08:00:00"),
            _export("src_undated", generated=""),
        ],
    )

    assert details["recommended_source"] is None
    assert [row["generated_date"] for row in details["registered_source_choices"]] == [
        "2026-10-08",
        None,
    ]


def test_exports_that_trade_stays_against_modules_are_left_to_the_researcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    details = _listing(
        monkeypatch,
        [
            _export("src_more_stays", generated="2026-10-08T08:00:00", stays=94458, modules=2),
            _export("src_more_modules", generated="2026-10-01T08:00:00", stays=90000, modules=3),
            _export("src_less_of_both", generated="2026-10-09T08:00:00", stays=80000, modules=2),
        ],
        active="src_less_of_both",
    )

    assert details["recommended_source"] is None
    assert [row["source_id"] for row in details["registered_source_choices"]] == [
        "src_more_stays",
        "src_more_modules",
    ]


def test_the_most_complete_export_is_still_recommended_alone() -> None:
    def row(source_id: str, stays: int) -> dict[str, Any]:
        return {
            "source_id": source_id,
            "source_scope": "registered_export",
            "module_count": 3,
            "aggregate": {"stays": stays},
            "active": source_id == "src_small",
        }

    recommendation = recommend_registered_export([row("src_small", 100), row("src_full", 94458)])

    assert recommendation.source["source_id"] == "src_full"
    assert recommendation.reason == "most_complete_local_dataset"
    assert recommendation.choices == ()


def test_a_database_without_a_registered_export_has_nothing_to_choose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    details = _listing(monkeypatch, [])

    assert details["recommended_source"] is None
    assert details["registered_source_choices"] == []


# --- the host's first reply ----------------------------------------------------------

MIMIC_IV = {"database": "miiv", "label": "MIMIC-IV", "reference_release": "3.1"}
CHOICES = [
    {"choice": 1, "source_id": "src_first", "label": "MIMIC-IV v3.1",
     "generated_date": "2026-10-08", "stays": 94458, "module_count": 19},
    {"choice": 2, "source_id": "src_second", "label": "MIMIC-IV v3.1",
     "generated_date": "2026-10-08", "stays": 94458, "module_count": 19},
]


def _first_reply(question: str, language: str, **catalog_details: Any) -> str:
    """The initial-question reply, from the preloaded catalog a real turn appends."""

    node = shutil.which("node")
    if not node or not (APP_DIR / "node_modules").is_dir():
        pytest.skip("Pinned Pi Node runtime is unavailable")
    module = APP_DIR / "src" / "post-tool-finalization.mjs"
    catalog = {
        "status": "ok", "code": "easyicu_data_sources_listed", "summary": "listed", "owner": "test",
        "details": {"supported_databases": [], "official_demos": [], **catalog_details},
    }
    instruction = "Respond in Simplified Chinese." if language == "zh" else "Respond in English."
    script = f"""
      import {{ hostPostToolFinalization, OWNER_CONTEXT_MARKER }} from {json.dumps(module.as_uri())};
      const model = {{ api: 'openai-completions', provider: 'test', id: 'test' }};
      const prompt = {json.dumps(question)}
        + '\\n\\n[EASYICU_INTERNAL_RESPONSE_LANGUAGE_V1]\\n' + {json.dumps(instruction)}
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
      const stream = hostPostToolFinalization(model, {{ messages: [user, assistant, result] }}, {json.dumps(language)});
      if (!stream) throw new Error('expected data-source finalization');
      const message = await stream.result();
      console.log(JSON.stringify(message.content[0].text));
    """
    completed = run_node(node, script, module=True, cwd=APP_DIR, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


def test_a_named_database_with_exports_to_choose_from_lists_them() -> None:
    text = _first_reply(
        "用 MIMIC-IV 研究乳酸与院内死亡", "zh",
        selected_database=MIMIC_IV, recommended_source=None, registered_source_choices=CHOICES,
    )

    assert "EasyICU 中已登记 2 份 MIMIC-IV v3.1 数据导出，无法自动判断用哪一份" in text
    first = "- 使用 EasyICU 中已准备好的 MIMIC-IV v3.1 数据导出（第 1 份，生成于 2026-10-08，94,458 个 ICU 入住记录，19 个数据模块）"
    assert first in text
    assert "（第 2 份，生成于 2026-10-08" in text
    assert "还没有登记" not in text and "数据目录" not in text
    assert explicitly_confirms_easyicu_registered_source(first.removeprefix("- "))


def test_the_english_reply_lists_them_too() -> None:
    text = _first_reply(
        "Use MIMIC-IV to study lactate and in-hospital death", "en",
        selected_database=MIMIC_IV, recommended_source=None, registered_source_choices=CHOICES,
    )

    assert "EasyICU has 2 registered MIMIC-IV v3.1 data exports it cannot choose between" in text
    first = "- Use the prepared MIMIC-IV v3.1 EasyICU data export (export 1, generated 2026-10-08, 94,458 ICU stays, 19 data modules)"
    assert first in text
    assert "no registered copy" not in text
    assert explicitly_confirms_easyicu_registered_source(first.removeprefix("- "))


def test_only_a_database_without_exports_is_called_unregistered() -> None:
    text = _first_reply(
        "用 MIMIC-IV 研究乳酸与院内死亡", "zh",
        selected_database=MIMIC_IV, recommended_source=None, registered_source_choices=[],
    )

    assert "但 EasyICU 里还没有登记这份数据" in text
    assert "第 1 份" not in text


def test_the_conversation_rule_lists_exports_it_cannot_choose_between() -> None:
    rule = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")

    assert (
        "When recommended_source is null and registered_source_choices is not empty, "
        "EasyICU already has those exports"
    ) in rule
    assert "never say the database is unregistered or ask for a local directory" in rule

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
from easyicu.webserver.pi_copilot.source_recommendation import (
    recommend_registered_export,
    registered_sources_for_database,
)
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
        ToolExecutionContext(data_handling="zero_retention_declared",
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


def _named_database_reply(question: str, language: str, **catalog_details: Any) -> str:
    """Exercise the pure Node narration owner without the optional Pi stream SDK."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = (APP_DIR / "src" / "post-tool-finalization.mjs").read_text(encoding="utf-8")
    owner = source[source.index("function boundedLabel("):source.index("\nfunction dataSourceSelectionText(")]
    script = f"""
      {owner}
      console.log(JSON.stringify(namedDatabaseText(
        {json.dumps(language)}, {json.dumps(catalog_details)}, {json.dumps(question)}
      )));
    """
    return json.loads(run_node(node, script, check=True).stdout)


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

    assert "尚未添加 MIMIC-IV v3.1 数据。" in text
    assert "第 1 份" not in text


def test_the_conversation_rule_lists_exports_it_cannot_choose_between() -> None:
    rule = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")

    assert (
        "When recommended_source is null and registered_source_choices is not empty, "
        "EasyICU already has those exports"
    ) in rule
    assert "never say the database is unregistered or ask for a local directory" in rule


HIRID = {"database": "hirid", "label": "HiRID", "reference_release": "1.1.1"}


@pytest.mark.parametrize("language", ["zh", "en"])
def test_a_hosted_server_without_the_database_offers_no_local_folder(language: str) -> None:
    """Missing data is a source setup issue, not a database-wide refusal."""

    text = _named_database_reply(
        "在 HiRID 数据库中描述体温记录的分布。", language,
        selected_database=HIRID, recommended_source=None, registered_source_choices=[],
        local_folder_selection_available=False,
    )

    assert text == (
        "研究问题已保存。尚未添加 HiRID v1.1.1 数据。"
        if language == "zh"
        else "The research question is saved. HiRID v1.1.1 data have not been added."
    )
    local = _named_database_reply(
        "在 HiRID 数据库中描述体温记录的分布。", "zh",
        selected_database=HIRID, recommended_source=None, registered_source_choices=[],
    )
    assert "尚未添加 HiRID v1.1.1 数据。" in local
    assert "选择本机的 HiRID v1.1.1 数据目录" in local


def _synthetic_catalog(database: str) -> dict[str, Any]:
    return {"sources": [{
        "id": "src_synthetic", "database": database, "ok": True, "path": "/synthetic/never-open",
        "label": "Synthetic", "modules": ["demographics", "outcome"], "summary": {"stays": 12},
    }]}


@pytest.mark.parametrize(
    ("database", "alias"), [("hirid", "HiRID"), ("aumc", "AmsterdamUMCdb"), ("sic", "SICdb")],
)
def test_a_named_database_recommends_its_registered_export(
    database: str, alias: str,
) -> None:
    """Study feasibility belongs to the capability gate after source choice."""

    choices, recommendation = registered_sources_for_database(_synthetic_catalog(database), alias)

    assert len(choices) == 1
    assert choices[0]["availability"] == "available_in_easyicu"
    assert recommendation.source == choices[0]
    assert not recommendation.choices
    assert recommendation.reason == "most_complete_local_dataset"
    assert "/synthetic/" not in json.dumps(choices)


@pytest.mark.parametrize(
    ("database", "alias"), [("eicu", "eicu"), ("miiv", "MIMIC-IV"),
                            ("mimic", "MIMIC-III"), ("eicu_demo", "eicu_demo")],
)
def test_a_supported_database_still_recommends_its_registered_export(database: str, alias: str) -> None:
    _, recommendation = registered_sources_for_database(_synthetic_catalog(database), alias)

    assert recommendation.source["source_id"] == "src_synthetic"
    assert recommendation.source["database"] == database


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_a_named_question_recommends_and_binds_a_registered_synthetic_source(
    tmp_path, monkeypatch, database,
) -> None:
    from easyicu.databases.profiles import get_database_profile
    from easyicu.synthetic_sources import generate_source
    from easyicu.webserver import sources, study_contexts
    from easyicu.webserver.pi_copilot.contracts import PiSessionDataSourceAuthorization
    from easyicu.webserver.pi_copilot.registered_source_offer import (
        bind_offered_registered_source, registered_source_offer,
    )

    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(sources, "_autodiscovery_bases", lambda: [])
    monkeypatch.setattr(sources.settings_store, "load_settings", lambda: {})
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "contexts.json")
    path = tmp_path / "synthetic"
    generate_source(path, database, stays=4)
    registry = sources.register_source(str(path), active=False)
    registry_before = sources._CONFIG_PATH.read_bytes()
    row = registry["sources"][0]
    assert row["source_kind"] == "synthetic"
    question = f"Describe the temperature records in {get_database_profile(database).display_name}."
    record = PiSessionRecord(session_id="pi-synthetic-choice", data_source_authorization=(
        PiSessionDataSourceAuthorization(status="pending")))
    context = ToolExecutionContext(session=record, user_message=question)
    policy = context.data_handling
    result = tool_module.execute_tool("easyicu_list_data_sources", {}, context)
    assert result["code"] == "easyicu_data_sources_listed"
    recommendation = result["details"]["recommended_source"]
    assert recommendation["source_id"] == row["id"]
    assert recommendation["database"] == database
    assert recommendation["auto_select_for_exact_database_request"] is True
    assert str(path) not in json.dumps(result)
    offer = registered_source_offer(record, database)
    assert offer["recommended"]["source_id"] == row["id"]
    study = study_contexts.upsert_context({"id": "study-synthetic-choice", "question": question})
    bound = bind_offered_registered_source(
        record, study, database=database, source_id=row["id"], stale={},
    )
    assert bound["data_source"]["database"] == database
    assert bound["data_source"]["path"] == str(path)
    assert context.data_handling == policy
    assert sources._CONFIG_PATH.read_bytes() == registry_before


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
def test_recommending_a_credentialed_source_does_not_grant_provider_clearance(
    tmp_path, monkeypatch, database,
) -> None:
    from easyicu.research_agent.providers.client_trust import PROVIDER_NOT_CLEARED
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError

    registry = _synthetic_catalog(database)
    registry["sources"][0]["path"] = str(tmp_path / "unverified")
    monkeypatch.setattr(tool_module.sources, "load_registry", lambda: registry)
    _, recommendation = registered_sources_for_database(registry, database)
    assert recommendation.source["source_id"] == "src_synthetic"
    context = ToolExecutionContext(session=PiSessionRecord(session_id="pi-uncleared"))
    policy = context.data_handling
    with pytest.raises(PiCopilotError) as error:
        tool_module.execute_tool("easyicu_list_data_sources", {"database": database}, context)
    assert error.value.status_code == 403
    assert error.value.code == PROVIDER_NOT_CLEARED
    assert context.data_handling == policy
    assert not registry.get("active_path")


@pytest.mark.parametrize("database", ["hirid", "aumc", "sic"])
@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("choice_count", [1, 2])
def test_named_database_narration_preserves_export_facts(database, language, choice_count):
    from easyicu.databases.profiles import get_database_profile

    profile = get_database_profile(database)
    selected = {"database": database, "label": profile.display_name,
                "reference_release": profile.reference_release}
    row = {"source_id": "src_synthetic", "database": database,
           "availability": "available_in_easyicu", "aggregate": {"stays": 12}, "module_count": 2}
    choices = [{"choice": index, "source_id": f"src_synthetic_{index}", "stays": 12,
                "module_count": 2, "generated_date": "2026-10-08"} for index in range(1, 3)]
    text = _named_database_reply(
        f"Describe temperature records in {profile.display_name}.", language,
        selected_database=selected, recommended_source=row if choice_count == 1 else None,
        registered_source_choices=choices if choice_count == 2 else [],
    )
    label = profile.display_name + (f" v{profile.reference_release}" if profile.reference_release else "")
    assert label in text
    assert ("12 个 ICU 入住记录，2 个数据模块" if language == "zh" else "12 ICU stays, 2 data modules") in text
    assert ("确认数据源不等于批准分析" if language == "zh" else "Confirming a source does not approve analysis") in text
    assert "暂不支持" not in text and "does not support" not in text
    if choice_count == 1:
        assert ("（推荐）" if language == "zh" else "(recommended)") in text
    else:
        assert ("第 2 份，生成于 2026-10-08" if language == "zh" else "export 2, generated 2026-10-08") in text

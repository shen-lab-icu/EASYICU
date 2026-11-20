"""A study planned from an official Demo's raw data extracts its own population.

B19: on the raw-source plan card (B18) the researcher asks, in the conversation,
for this study's extraction. The study reads a verified official Demo's raw
folder and may state no modules. Before, its extraction stopped for missing
modules, and preparing the Demo again only re-registered the generic
all-module export while the reply called it this study's package.

Now the Demo cache owner's verification resolves the modules (the Demo's full
module scope, as its official all-module export), preparing the same Demo
again redirects to the study's extraction without spending the turn's grant,
and a generic Demo export says it is not any study's package. Any other raw
folder still needs its modules stated. Synthetic Demo cache only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver import demo_source_storage as storage
from easyicu.webserver import study_contexts
from easyicu.webserver.demo_source_contracts import SOURCES
from easyicu.webserver.pi_copilot import extraction_handoff
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import PiCopilotError, ToolExecutionContext
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.routes import jobs as jobs_route
from tests.support.official_demo_choice import choice as choice
from tests.support.official_demo_raw import official_raw as official_raw

# The raw-source plan card's drafted request (6b93f2c69): it grants extract.
_SENT = "我确认：按本研究现有的人群条件进行数据提取，生成本研究的数据包。"


@pytest.fixture
def raw_demo(choice, monkeypatch):
    """The study after the raw-source choice, stating no modules."""
    choice.service.authorize_data_source(choice.record.session_id, **choice.body)
    study = {**study_contexts.get_context(choice.study["id"]), "modules": []}
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: study)
    extractions: list[dict] = []
    monkeypatch.setattr(
        jobs_route,
        "jobs_extract",
        lambda body: (
            extractions.append(body)
            or {
                "job_id": "extract-1",
                "kind": "extract",
                "status": "running",
                "study_context_id": study["id"],
                "study_context_revision": study["revision"] + 1,
            }
        ),
    )
    preparations: list[str] = []
    monkeypatch.setattr(
        tool_module.jobs.MANAGER,
        "submit",
        lambda kind, runner: (
            preparations.append(kind)
            or SimpleNamespace(id="prepare-1", kind=kind, status="running")
        ),
    )
    monkeypatch.setattr(
        tool_module.demo_sources, "make_prepare_runner", lambda source_id: None
    )
    demo = next(
        spec for spec in SOURCES if spec.database == study["data_source"]["database"]
    )
    return SimpleNamespace(
        study=study,
        demo=demo,
        choice=choice,
        extractions=extractions,
        preparations=preparations,
    )


def _turn(raw_demo) -> ToolExecutionContext:
    return ToolExecutionContext(
        data_handling="zero_retention_declared",
        session=raw_demo.choice.record,
        user_message=_SENT,
        allowed_actions={"extract"},
    )


def _missing(study) -> list[str]:
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=False,
        active_job=None,
        latest_run=None,
        export_cohort_refusal=None,
    )
    return list(snapshot.missing_setup_fields)


def test_the_study_extracts_its_population_with_the_demos_full_module_scope(
    raw_demo,
) -> None:
    result = tool_module.execute_tool("easyicu_start_extraction", {}, _turn(raw_demo))

    assert result["status"] == "ok"
    assert result["code"] == "easyicu_extraction_submitted"
    [body] = raw_demo.extractions
    assert body["path"] == raw_demo.study["data_source"]["path"]
    assert body["registered_export_path"] is None
    assert body["modules"] is None  # every module, as the official all-module export
    assert body["cohort"] == raw_demo.study["cohort"]
    assert body["study_context_id"] == raw_demo.study["id"]
    assert str(raw_demo.choice.paths.root) not in json.dumps(result)


def test_the_workflow_does_not_ask_for_modules_the_demo_resolves(raw_demo) -> None:
    assert "modules" not in _missing(raw_demo.study)


def test_preparing_the_same_demo_again_redirects_without_spending_the_grant(
    raw_demo,
) -> None:
    turn = _turn(raw_demo)

    redirected = tool_module.execute_tool(
        "easyicu_prepare_demo_source", {"source_id": raw_demo.demo.id}, turn
    )

    assert redirected["status"] == "blocked"
    assert redirected["code"] == "pi_demo_source_raw_bound_extract_study"
    assert redirected["details"]["next_tool"] == "easyicu_start_extraction"
    assert "generic all-module Demo export" in redirected["summary"]
    assert raw_demo.preparations == []

    # The same turn's one-use grant still starts this study's extraction; the
    # submission changes authority, so the turn then stops until rebind.
    started = tool_module.execute_tool("easyicu_start_extraction", {}, turn)
    assert started["code"] == "easyicu_extraction_submitted"
    with pytest.raises(PiCopilotError):
        tool_module.execute_tool("easyicu_start_extraction", {}, turn)
    assert len(raw_demo.extractions) == 1


def test_only_the_studys_extraction_advances_an_authorized_turn(raw_demo) -> None:
    turn = _turn(raw_demo)
    revision = study_contexts.get_context(raw_demo.study["id"])["revision"]

    outcomes = {
        name: tool_module.execute_tool(name, params, turn)
        for name, params in (
            ("easyicu_prepare_demo_source", {"source_id": raw_demo.demo.id}),
            ("easyicu_start_extraction", {}),
        )
    }

    advanced = [name for name, result in outcomes.items() if result["status"] == "ok"]
    assert advanced == ["easyicu_start_extraction"]
    assert raw_demo.preparations == []
    assert [body["study_context_id"] for body in raw_demo.extractions] == [
        raw_demo.study["id"]
    ]
    assert study_contexts.get_context(raw_demo.study["id"])["revision"] == revision


def test_a_raw_folder_the_demo_owner_does_not_verify_still_needs_its_modules(
    raw_demo, tmp_path
) -> None:
    # Same Demo database, but not the Demo cache owner's verified raw folder.
    # (Copilot's data clearance already refuses such a source before any tool
    # runs; the extraction owner and the workflow must not resolve it either.)
    copied = Path(tmp_path) / "copied_demo_raw"
    copied.mkdir()
    unverified = {
        **raw_demo.study,
        "data_source": {**raw_demo.study["data_source"], "path": str(copied)},
    }
    assert storage.verified_public_demo_raw(copied) is None

    assert extraction_handoff.official_demo_raw_source(unverified) is False
    assert extraction_handoff.official_demo_raw_source(raw_demo.study) is True
    assert "modules" in _missing(unverified)
    bodies: list[dict] = []
    extraction_handoff.submit_study_extraction(
        study=unverified,
        registered_source=None,
        submit=lambda body: bodies.append(body) or {},
    )
    assert bodies[0]["modules"] == []


def test_a_raw_folder_of_another_database_still_needs_its_modules(
    raw_demo, monkeypatch, tmp_path
) -> None:
    other = {
        **raw_demo.study,
        "data_source": {"database": "miiv", "path": str(Path(tmp_path) / "raw_miiv")},
    }
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: other)

    result = tool_module.execute_tool("easyicu_start_extraction", {}, _turn(raw_demo))

    assert result["code"] == "extraction_setup_incomplete"
    assert raw_demo.extractions == []
    assert "modules" in _missing(other)


def test_a_generic_demo_export_says_it_is_no_studys_package(
    raw_demo, monkeypatch
) -> None:
    exported = {**raw_demo.choice.study, "modules": []}  # bound to the Demo's export
    monkeypatch.setattr(tool_module, "_bound_context", lambda binding: exported)

    result = tool_module.execute_tool(
        "easyicu_prepare_demo_source", {"source_id": raw_demo.demo.id}, _turn(raw_demo)
    )

    assert result["code"] == "easyicu_demo_source_preparation_submitted"
    assert result["details"]["study_scoped"] is False
    assert result["details"]["export_scope"] == "all_modules"
    assert (
        "not a data package extracted for any study's population" in result["summary"]
    )
    assert raw_demo.preparations == ["demo-source-prepare"]

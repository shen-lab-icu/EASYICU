"""Demo and full-database source identities never substitute for one another.

All registries are synthetic metadata; no export or patient rows are read.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver import sources
from easyicu.webserver.pi_copilot import tools
from easyicu.webserver.pi_copilot.contracts import (
    PiCopilotError,
    PiSessionDataSourceAuthorization,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.source_recommendation import (
    database_reader_label,
    official_demos_for_database,
    registered_sources_for_database,
)
from tests.support.node import run_node

offer_owner = importlib.import_module("easyicu.webserver.pi_copilot.registered_source_offer")


def export(database: str, *, stays: int = 12, label: str = "internal-export") -> dict:
    return {
        "id": f"src_{database}", "database": database, "ok": True,
        "path": f"/synthetic/{database}", "label": label,
        "modules": ["demographics", "outcome"], "summary": {"stays": stays},
    }


@pytest.mark.parametrize("database", ["eicu", "mimic"])
def test_a_more_complete_demo_is_never_a_full_database_choice(database):
    full = export(database)
    demo = export(f"{database}_demo", stays=24)
    choices, recommendation = registered_sources_for_database({"sources": [demo, full]}, database)
    assert [row["source_id"] for row in choices] == [full["id"]]
    assert recommendation.source["source_id"] == full["id"]


@pytest.mark.parametrize("database", ["eicu", "mimic"])
def test_a_demo_only_registry_has_no_full_database_export(database):
    choices, recommendation = registered_sources_for_database(
        {"sources": [export(f"{database}_demo")]}, database,
    )
    assert choices == []
    assert recommendation.source is None
    assert recommendation.choices == ()


@pytest.mark.parametrize(("database", "label"), [
    ("eicu_demo", "eICU Demo v2.0.1"),
    ("mimic_demo", "MIMIC-III Demo v1.4"),
])
def test_an_exact_demo_request_uses_its_own_reader_label_and_count(database, label):
    choices, recommendation = registered_sources_for_database(
        {"sources": [export(database)]}, database,
    )
    assert len(choices) == 1
    assert choices[0]["database"] == database
    assert choices[0]["label"] == label
    assert choices[0]["aggregate"]["stays"] == 12
    assert recommendation.source == choices[0]
    assert database_reader_label(database) == label


def test_a_demo_only_registry_offers_no_eicu_export_but_names_the_demo(monkeypatch):
    registry = {"sources": [export("eicu_demo")]}
    record = PiSessionRecord(
        session_id="pi-demo-identity",
        data_source_authorization=PiSessionDataSourceAuthorization(status="pending"),
    )
    monkeypatch.setattr(sources, "load_registry", lambda: registry)
    offer = offer_owner.registered_source_offer(record, "eicu", registry=registry)
    assert offer["recommended"] is None
    assert offer["choices"] == []
    assert {key: value for key, value in offer["official_demo"].items() if key != "registered_offer"} == {
        "id": "eicu_demo_v2_0_1", "database": "eicu_demo",
        "label": "eICU Demo v2.0.1", "stays": 12,
    }
    demo_offer = offer["official_demo"]["registered_offer"]
    assert demo_offer["recommended"]["source_id"] == "src_eicu_demo"
    listing = tools.execute_tool(
        "easyicu_list_data_sources", {"database": "eicu"},
        ToolExecutionContext(session=record, user_message="Use eICU for this study"),
    )["details"]
    assert listing["sources"] == []
    assert listing["recommended_source"] is None
    assert listing["official_demos"][0]["database"] == "eicu_demo"


def test_a_registered_official_demo_can_satisfy_an_exact_demo_request():
    demo = export("eicu_demo", label="eICU Collaborative Research Database Demo v2.0.1")
    assert sources.is_official_demo_source(demo)
    _, recommendation = registered_sources_for_database({"sources": [demo]}, "eicu_demo")
    assert recommendation.source["source_id"] == demo["id"]
    record = PiSessionRecord(
        session_id="pi-demo-identity",
        data_source_authorization=PiSessionDataSourceAuthorization(status="pending"),
    )
    offer = offer_owner.registered_source_offer(record, "eicu_demo", registry={"sources": [demo]})
    assert offer["recommended"]["source_id"] == demo["id"]
    assert offer["recommended"]["label"] == "eICU Demo v2.0.1"


def test_an_unprepared_official_demo_card_still_states_the_catalog_stay_count():
    demo = official_demos_for_database({"sources": []}, "eicu")[0]
    assert demo["database"] == "eicu_demo"
    assert demo["stays"] == ">2500"


def test_binding_checks_exact_identity_even_if_a_demo_was_offered(monkeypatch):
    demo = export("eicu_demo")
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [demo]})
    monkeypatch.setattr(offer_owner, "registered_source_offer", lambda *args, **kwargs: {
        "database": "eicu", "recommended": {"source_id": demo["id"]}, "choices": [],
    })
    monkeypatch.setattr(offer_owner.study_context_update, "update_study_context", lambda *args, **kwargs: {
        "status": "ok",
    })
    with pytest.raises(PiCopilotError) as caught:
        offer_owner.bind_offered_registered_source(
            PiSessionRecord(session_id="pi-demo-identity"), {}, database="eicu",
            source_id=demo["id"], stale={},
        )
    assert caught.value.code == "pi_data_source_database_mismatch"


def test_binding_an_exact_demo_preserves_its_identity_and_stay_count(tmp_path, monkeypatch):
    from easyicu.webserver import study_contexts
    from easyicu.webserver.pi_copilot import study_context_update
    from easyicu.webserver.pi_copilot.service import PiCopilotService

    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "contexts.json")
    study = study_contexts.upsert_context({"id": "study-demo", "question": "Describe ICU stays."})
    demo = export("eicu_demo", label="eICU")
    updated = {}

    def project(_context, *, study_override):
        updated.update(study_override)
        return {}

    result = study_context_update.update_study_context(
        ToolExecutionContext(session=PiSessionRecord(session_id="pi-demo-identity"), allowed_actions={"configure"}),
        {"bind_source_id": demo["id"]}, load_context=lambda _binding: study,
        project_workflow=project, source_registry={"sources": [demo]},
    )
    assert result["status"] == "ok"
    assert updated["data_source"]["database"] == "eicu_demo"
    assert updated["data_source"]["label"] == "eICU Demo v2.0.1"
    assert updated["data_source"]["stays"] == 12
    saved = study_contexts.get_context(study["id"])
    assert saved["data_source"] == updated["data_source"]
    reference = PiCopilotService._session_source_reference(saved)
    assert reference.database == "eicu_demo"
    assert reference.label == "eICU Demo"
    assert reference.reference_release == "2.0.1"
    assert reference.stays == 12
    changed = {**saved, "data_source": {**saved["data_source"], "stays": 11}}
    assert PiCopilotService._session_source_reference(changed).identity_sha256 != reference.identity_sha256
    from easyicu.webserver.pi_copilot.projections import project_study_context, project_study_setup_receipt

    assert project_study_context(saved)["data_source"]["stays"] == 12
    assert project_study_setup_receipt(saved).configuration["data_source"]["stays"] == 12


@pytest.mark.parametrize(("database", "raw_label", "reader_label"), [
    ("eicu_demo", "eICU", "eICU Demo v2.0.1"),
    ("mimic_demo", "MIMIC-III", "MIMIC-III Demo v1.4"),
    ("miiv", "MIMIC-IV Clinical Database Demo v2.2", "MIMIC-IV Clinical Database Demo v2.2"),
])
def test_a_demo_run_manifest_never_names_the_full_database_as_its_source(
    tmp_path, database, raw_label, reader_label,
):
    from easyicu.webserver import agent_pipeline_runs

    study = {
        "id": "study-demo", "question": "Describe the recorded ICU stays.",
        "data_source": {"database": database, "label": raw_label, "stays": 12},
    }
    wrapper = tmp_path / "run-demo"
    agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper, study=study, provider={}, acquisition=SimpleNamespace(), run_dir=None,
    )
    source = json.loads((wrapper / "run_context.json").read_text())["source"]
    assert source["database"] == database
    assert source["reader_label"] == reader_label
    assert source["label"] == reader_label
    assert source["stays"] == 12
    manifest = json.loads((wrapper / "source_run_manifest.json").read_text())
    assert manifest["source"] == source
    assert source["label"] not in {"eICU", "MIMIC-IV", "MIMIC-III"}


@pytest.mark.parametrize("hosted", [False, True])
def test_a_full_database_card_offers_the_demo_without_a_hosted_prepare_action(hosted):
    static = Path(__file__).parents[3] / "src/easyicu/webserver/static/js"
    harness = """
      global.window = global;
      const owners = {};
      window.EasyICU = {guidedPi: {declare:(key,value)=>owners[key]=value,
        optional:()=>({active:()=>HOSTED,supportContact:()=>''})}};
    """.replace("HOSTED", json.dumps(hosted))
    harness += (static / "html-escape.js").read_text()
    harness += (static / "screens-guided-pi-data-consent.js").read_text()
    harness += """
      const source = {database:'eicu_demo',label:'eICU Demo',reference_release:'2.0.1',stays:12};
      const session = {data_source_authorization:{status:'pending'}};
      const html = owners.dataConsent.render(session, {esc:EU_HTML.esc,tr:en=>en,icon:()=>'',
        namedDatabase:{key:'eicu',label:'eICU',offer:{state:'ready',recommended:null,choices:[],
          official_demo:{id:'eicu_demo_v2_0_1',database:'eicu_demo',label:'eICU Demo v2.0.1',stays:12}}}});
      const selected = owners.dataConsent.sourceLabel({source});
      const projectHtml = owners.dataConsent.render({data_source_authorization:{status:'pending',
        reason:'project_source_confirmation_required',source}}, {esc:EU_HTML.esc,tr:en=>en,icon:()=>''});
      console.log(JSON.stringify({html,selected,projectHtml}));
    """
    rendered = json.loads(run_node("node", harness, check=True).stdout)
    assert "eICU Demo v2.0.1" in rendered["html"]
    assert "12 ICU stays" in rendered["html"]
    if hosted:
        assert "data-gpi-named-demo" not in rendered["html"]
        assert "administrator" in rendered["html"]
    else:
        assert 'data-gpi-named-demo="eicu_demo_v2_0_1"' in rendered["html"]
    assert 'data-gpi-data-source-action="use_registered_source"' not in rendered["html"]
    assert "eICU Demo v2.0.1" in rendered["selected"]
    assert "12 ICU stays" in rendered["selected"]
    assert "eICU Demo v2.0.1" in rendered["projectHtml"]
    assert "12 ICU stays" in rendered["projectHtml"]


@pytest.mark.parametrize(("message", "database"), [
    ("Use eICU demo for this study", "eicu_demo"),
    ("Use MIMIC-III demo for this study", "mimic_demo"),
])
def test_an_explicit_demo_name_keeps_the_demo_identity_in_the_listing(monkeypatch, message, database):
    demo = export(database)
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [demo]})
    listing = tools.execute_tool(
        "easyicu_list_data_sources", {"database": database},
        ToolExecutionContext(session=PiSessionRecord(session_id="pi-demo-identity"), user_message=message),
    )["details"]
    assert listing["selected_database"]["database"] == database
    assert listing["sources"][0]["database"] == database
    assert listing["recommended_source"]["database"] == database


def test_a_cross_database_receipt_keeps_full_and_demo_sources_distinct():
    from easyicu.webserver.crossdb_review import _crossdb_selection_receipt

    receipt = _crossdb_selection_receipt([export("eicu"), export("eicu_demo")])
    assert [row["database"] for row in receipt["sources"]] == ["eicu", "eicu_demo"]
    assert receipt["source_count"] == 2


def test_a_cross_database_demo_summary_names_the_demo_and_its_count():
    from easyicu.webserver.crossdb_review import _safe_registered_source

    source = _safe_registered_source(export("eicu_demo", label="eICU"))
    assert source["database"] == "eicu_demo"
    assert source["label"] == "eICU Demo v2.0.1"
    assert source["stays"] == 12


def test_the_planning_context_records_the_demo_database_and_its_stay_count():
    import pandas as pd

    from easyicu.research_agent.research_context.builder import build_research_context
    from easyicu.research_agent.research_context.outbound import outbound_safe_context_payload

    context = build_research_context(
        research_question="Describe the recorded ICU stays.",
        cohort=pd.DataFrame({"patientunitstayid": [1, 2], "signal": [4.0, 6.0]}),
        cohort_name="synthetic_demo", database="eicu_demo", id_columns=["patientunitstayid"],
    )
    assert context.cohort.database == "eicu_demo"
    payload = outbound_safe_context_payload(context)
    assert payload["cohort"]["database"] == "eicu_demo"
    assert payload["cohort"]["n_stays"] == 2


@pytest.mark.parametrize("stays", [-1, True, "12", 1.5])
def test_a_source_stay_count_must_be_a_nonnegative_integer(stays):
    from easyicu.webserver.study_contexts import StudyContextError, _data_source

    with pytest.raises(StudyContextError) as caught:
        _data_source({"database": "eicu_demo", "stays": stays})
    assert caught.value.detail["field"] == "data_source.stays"


def test_a_legacy_official_demo_confirmation_keeps_its_demo_release():
    from easyicu.webserver.pi_copilot.service import PiCopilotService

    reference = PiCopilotService._session_source_reference({
        "revision": 1, "data_source": {
            "database": "miiv", "label": "MIMIC-IV Clinical Database Demo v2.2", "stays": 12,
        },
    })
    assert reference.label == "MIMIC-IV Clinical Database Demo v2.2"
    assert reference.reference_release is None
    assert reference.stays == 12

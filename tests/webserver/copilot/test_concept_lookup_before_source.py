"""Database metadata can resolve stated readings before selecting an export."""

import json

import pytest

from easyicu.research_agent.acquisition import catalog
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.pi_copilot import tools
from easyicu.webserver.pi_copilot.contracts import (
    PiCopilotError,
    PiSessionDataSourceAuthorization,
    PiSessionRecord,
    ToolExecutionContext,
)

from .stated_variable_fixtures import _update


def _unexpected_source_access(*args, **kwargs):
    pytest.fail("A database capability lookup must not inspect sources or exports")


@pytest.fixture
def metadata_lookup(monkeypatch):
    study = {}
    calls = []
    monkeypatch.setattr(tools, "_bound_context", lambda _binding: study)
    monkeypatch.setattr(tools.sources, "load_registry", _unexpected_source_access)
    monkeypatch.setattr(catalog, "build_available_catalog", _unexpected_source_access)

    def build_database(database):
        calls.append(database)
        return catalog.AvailableCatalog(
            source=f"easyicu-database-capability:{database}",
            concepts=[
                catalog.CatalogConcept(
                    concept_id=f"{database}_trial_score", description="Trial scale",
                    column_role="value",
                ),
                catalog.CatalogConcept(concept_id="other_measure", description="Other measure"),
            ],
        )

    monkeypatch.setattr(catalog, "build_database_capability_catalog", build_database)

    def lookup(*, message="", status="pending", params=None):
        context = ToolExecutionContext(
            session=PiSessionRecord(
                session_id="metadata-concept-session",
                data_source_authorization=PiSessionDataSourceAuthorization(status=status),
            ),
            user_message=message,
        )
        return tools.execute_tool("easyicu_list_source_concepts", params or {}, context)

    return study, calls, lookup


@pytest.mark.parametrize("database,name", [("eicu", "eICU"), ("miiv", "MIMIC-IV")])
@pytest.mark.parametrize("origin", ["message", "question", "setup"])
@pytest.mark.parametrize("status", ["pending", "selection_in_progress"])
def test_named_database_lookup_is_metadata_only(metadata_lookup, database, name, origin, status):
    study, calls, lookup = metadata_lookup
    question = f"In {name}, how is the trial marker related to the outcome?"
    message = question if origin == "message" else "Find the trial scale concept."
    if origin == "question":
        study["question"] = question
    elif origin == "setup":
        study["data_source"] = {"database": name}

    result = lookup(message=message, status=status, params={"query": "trial", "limit": 1})

    assert calls == [database]
    assert result["status"] == "ok"
    assert result["code"] == "easyicu_database_concepts_listed"
    details = result["details"]
    assert details["catalog_scope"] == "database_capability"
    assert details["database"] == database and details["source_bound"] is False
    assert "source_id" not in details
    assert [row["concept_id"] for row in details["concepts"]] == [f"{database}_trial_score"]
    assert details["concepts"][0]["role"] == "value"
    assert details["returned_count"] == 1
    assert "not yet a bound source" in result["summary"]
    assert "path" not in json.dumps(result).lower()


@pytest.mark.parametrize("message", [
    "How is the trial marker related to the outcome?",
    "In UnlistedDB, how is the trial marker related to the outcome?",
])
def test_unnamed_or_unsupported_database_does_not_probe_a_source(metadata_lookup, message):
    _, calls, lookup = metadata_lookup
    with pytest.raises(PiCopilotError) as caught:
        lookup(message=message)
    assert caught.value.code == "pi_source_id_required"
    assert calls == []


@pytest.mark.parametrize("database", ["UnlistedDB", "eicu_demo"])
def test_an_unsupported_setup_database_is_refused(metadata_lookup, database):
    study, calls, lookup = metadata_lookup
    study["data_source"] = {"database": database}
    with pytest.raises(PiCopilotError) as caught:
        lookup()
    assert caught.value.code == "pi_database_not_supported"
    assert calls == []


@pytest.mark.parametrize("params,expected", [
    ({"query": "trial"}, ["eicu_trial_score"]),
    ({"query": "absent"}, []),
    ({"limit": 1}, ["eicu_trial_score"]),
    ({}, ["eicu_trial_score", "other_measure"]),
])
def test_database_metadata_keeps_query_and_limit_bounds(metadata_lookup, params, expected):
    _, _, lookup = metadata_lookup
    result = lookup(message="Study a trial marker in eICU.", params=params)
    assert [row["concept_id"] for row in result["details"]["concepts"]] == expected
    assert result["details"]["returned_count"] == len(expected)


def test_database_metadata_cannot_claim_export_modules(metadata_lookup):
    _, calls, lookup = metadata_lookup
    result = lookup(message="Study a trial marker in eICU.", params={"modules": ["scores"]})
    assert result["code"] == "pi_source_concept_modules_invalid"
    assert "Omit modules" in result["summary"]
    assert calls == []


@pytest.mark.parametrize("field,words", [("stated_adjustments", "控制试验量表后"),
                                        ("stated_subgroups", "按试验量表分层")])
def test_looked_up_reading_survives_save_and_source_binding(monkeypatch, metadata_lookup, field, words):
    study, _, lookup = metadata_lookup
    question = f"在 eICU 中，{words}，指标与结局的关系如何？"
    study.update(id="metadata-reading-study", revision=1, question=question)
    monkeypatch.setattr("easyicu.webserver.study_intent.catalog_named_concepts", lambda _text: ())
    result = lookup(params={"query": "trial"})
    concept = result["details"]["concepts"][0]["concept_id"]
    saved, writes = _update(
        monkeypatch, current=study, question=question, field=field, concepts=(concept,),
        statements=[{"variable": "试验量表", "words": words, "concept": concept}],
    )
    assert saved["status"] == "ok"
    record, = writes[0][field]
    assert record["concept"] == concept and record["source_status"] == "unbound"

    compiled = agent_pipeline_runs._research_user_preferences({
        **writes[0], "data_source": {"path": "/synthetic/source", "database": "eicu"},
    })
    checked, = json.loads(compiled["data_constraints"])[field]
    assert checked == {**record, "source_status": "available"}


@pytest.mark.parametrize("status", ["pending", "selection_in_progress"])
@pytest.mark.parametrize("source_id", [None, "synthetic-source"])
def test_a_bound_source_keeps_its_confirmation_gate(metadata_lookup, status, source_id):
    study, calls, lookup = metadata_lookup
    study.update(question="Study the trial marker in eICU.",
                 data_source={"database": "eicu", "path": "/synthetic/source"})
    result = lookup(status=status, params={"source_id": source_id} if source_id else {})
    assert result["code"] == "pi_session_data_source_confirmation_required"
    assert calls == []


def test_an_explicit_source_keeps_its_confirmation_gate(metadata_lookup):
    _, calls, lookup = metadata_lookup
    result = lookup(message="Study the trial marker in eICU.",
                    params={"source_id": "synthetic-source"})
    assert result["code"] == "pi_session_data_source_confirmation_required"
    assert calls == []


def test_a_confirmed_bound_source_returns_its_export_catalog(monkeypatch):
    source = {"id": "synthetic-source", "path": "/synthetic/source", "ok": True,
              "database": "miiv", "modules": ["scores"]}
    monkeypatch.setattr(tools, "_bound_context", lambda _binding: {
        "question": "How is a trial marker related to the outcome in eICU?",
        "data_source": source,
    })
    monkeypatch.setattr(tools.sources, "load_registry", lambda: {"sources": [source]})
    monkeypatch.setattr(catalog, "build_database_capability_catalog", _unexpected_source_access)
    calls = []

    def build_export(path):
        calls.append(str(path))
        return catalog.AvailableCatalog(source=str(path), concepts=[
            catalog.CatalogConcept(concept_id="export_score", file_name="scores.parquet"),
        ])

    monkeypatch.setattr(catalog, "build_available_catalog", build_export)
    context = ToolExecutionContext(session=PiSessionRecord(
        session_id="bound-concept-session",
        data_source_authorization=PiSessionDataSourceAuthorization(status="confirmed"),
    ))
    result = tools.execute_tool("easyicu_list_source_concepts", {
        "source_id": "synthetic-source", "modules": ["scores"], "query": "export_score",
    }, context)

    assert calls == ["/synthetic/source"]
    assert result["code"] == "easyicu_source_concepts_listed"
    assert result["details"] == {
        "source_id": "synthetic-source",
        "concepts": [{"concept_id": "export_score", "module": "scores", "role": "value",
                      "description": "", "selection_mode": "ordinary", "selection_note": "",
                      "canonical_alternative": ""}],
        "canonical_alternatives": [], "returned_count": 1, "truncated": False,
    }

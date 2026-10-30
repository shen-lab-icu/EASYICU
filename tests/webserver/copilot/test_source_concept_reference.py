import json

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.webserver.pi_copilot.contracts import PiSessionRecord
from easyicu.webserver.pi_copilot.source_concept_reference import source_concept_reference
from easyicu.webserver.pi_copilot.tools import ToolExecutionContext, _list_concept_catalog


def test_reference_is_database_specific_and_does_not_claim_actual_reads():
    sic = source_concept_reference("lact", "sicdb")
    assert sic["database"] == "sic"
    assert sic["declared_units"] == ["mmol/L"]
    assert sic["declared_mappings"][0]["table"] == "laboratory"
    assert set(sic["declared_mappings"][0]["ids"]) == {454, 465, 657}
    assert sic["declared_mappings"][0]["source_table_defaults"] == {
        "id_var": "CaseID", "index_var": "Offset", "val_var": "LaboratoryValue"
    }
    assert len(sic["source_config_sha256"]) == 64
    assert sic["actual_source_reads_verified"] is False
    assert sic["record_units_verified"] is False
    assert source_concept_reference("lact", "eicu")["declared_mappings"][0]["table"] == "lab"
    assert source_concept_reference("lact", "unrecognized") is None
    assert source_concept_reference("not_a_concept", "sic") is None


def test_catalog_tool_exposes_reference_without_source_paths(monkeypatch):
    from easyicu.research_agent.acquisition import catalog

    monkeypatch.setattr(catalog, "build_available_catalog", lambda _: AvailableCatalog(
        source="/private/source", concepts=[CatalogConcept(
            concept_id="lact", description="lactate", file_name="blood_gas.parquet"
        )]
    ))
    context = ToolExecutionContext(data_handling="zero_retention_declared",
                                   session=PiSessionRecord(session_id="pi-concepts"))
    result = _list_concept_catalog(context, {"query": "lact"}, source={
        "id": "source", "path": "/private/source", "database": "sic", "modules": []
    })
    item = result["details"]["concepts"][0]
    assert item["source_dictionary_reference"]["declared_mappings"][0]["sub_var"] == "LaboratoryID"
    assert "dictionary references only" in result["details"]["source_dictionary_reference_note"]
    assert "/private/source" not in json.dumps(result)


def test_catalog_tool_omits_identifier_names_without_weakening_privacy_gate(monkeypatch):
    from easyicu.research_agent.acquisition import catalog
    from easyicu.webserver.pi_copilot.projections import ensure_safe_projection

    monkeypatch.setattr(catalog, "build_available_catalog", lambda _: AvailableCatalog(
        source="/private/source", concepts=[CatalogConcept(
            concept_id="death", description="hospital mortality", file_name="outcome.parquet"
        )]
    ))
    context = ToolExecutionContext(data_handling="zero_retention_declared",
                                   session=PiSessionRecord(session_id="pi-concepts"))
    result = _list_concept_catalog(context, {"query": "death"}, source={
        "id": "source", "path": "/private/source", "database": "miiv", "modules": []
    })
    mappings = result["details"]["concepts"][0]["source_dictionary_reference"]["declared_mappings"]
    assert all("id_var" not in mapping.get("source_table_defaults", {}) for mapping in mappings)
    assert ensure_safe_projection(result) == result


def test_unavailable_dictionary_does_not_invent_a_mapping(monkeypatch):
    from easyicu.webserver.pi_copilot import source_concept_reference as owner

    def missing():
        raise FileNotFoundError("missing dictionary")

    monkeypatch.setattr(owner, "_dictionary", missing)
    assert owner.source_concept_reference("lact", "sic") is None

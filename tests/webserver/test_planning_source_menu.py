"""Physical columns cannot overrule a canonical negative source contract."""
import json

import pytest

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.webserver import agent_pipeline_runs as owner
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError


@pytest.fixture
def source_menu(monkeypatch):
    from easyicu.research_agent.acquisition import catalog

    base = AvailableCatalog(source="canonical", concepts=[CatalogConcept("death")])
    source = AvailableCatalog(source="legacy", concepts=[
        CatalogConcept("icu_readmission", n_rows=0),
        CatalogConcept("local_measurement", description="Source-owned measurement"),
        CatalogConcept("death", description="Source-owned death definition"),
    ])
    monkeypatch.setattr(catalog, "build_database_capability_catalog", lambda _: base)
    monkeypatch.setattr(catalog, "build_available_catalog", lambda _: source)
    return source


def test_source_merge_preserves_negative_contract_and_local_metadata(source_menu):
    merged = owner._metadata_only_planning_catalog(database="miiv", export_path="/metadata")
    assert merged.ids() == ["death", "local_measurement"]
    assert merged.concepts[0].description == "Source-owned death definition"
    assert source_menu.ids() == ["icu_readmission", "local_measurement", "death"]


def test_prepared_cohort_menu_contains_only_physical_source_columns(
    source_menu, tmp_path, monkeypatch,
):
    from easyicu.research_agent.acquisition import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "build_database_capability_catalog",
        lambda _: AvailableCatalog(
            source="canonical",
            concepts=[CatalogConcept("death"), CatalogConcept("crea")],
        ),
    )
    (tmp_path / "easyicu_export_manifest.json").write_text(
        json.dumps({"entry_mode": "study_local_prepared_cohort"}),
        encoding="utf-8",
    )

    catalog = owner._metadata_only_planning_catalog(
        database="miiv", export_path=tmp_path,
    )

    assert catalog.ids() == ["local_measurement", "death"]
    assert source_menu.ids() == ["icu_readmission", "local_measurement", "death"]


def test_prepared_cohort_without_verifiable_catalog_fails_closed(
    tmp_path, monkeypatch,
):
    from easyicu.research_agent.acquisition import catalog as catalog_module

    (tmp_path / "easyicu_export_manifest.json").write_text(
        json.dumps({"entry_mode": "study_local_prepared_cohort"}),
        encoding="utf-8",
    )
    def invalid_catalog(_):
        raise ValueError("invalid package")

    monkeypatch.setattr(catalog_module, "build_available_catalog", invalid_catalog)

    with pytest.raises(ResearchPipelineRunError) as caught:
        owner._metadata_only_planning_catalog(
            database="miiv", export_path=tmp_path,
        )

    assert caught.value.code == "research_pipeline_prepared_source_catalog_unavailable"


@pytest.mark.parametrize("requirement", [
    {"required_concepts": ("icu_readmission",)},
    {"target_outcome": "icu_readmission"},
    {"question": "Compare ICU readmission after treatment."},
])
def test_explicit_required_unsupported_input_blocks_before_model_or_output(
    source_menu, tmp_path, requirement,
):
    llm = ScriptedMockLLMClient([json.dumps({
        "selected_concepts": ["death"], "rationale": "An alternative outcome.",
        "inclusion_exclusion": [],
    })])
    options = {"question": "Compare hospital death.", **requirement}
    with pytest.raises(ResearchPipelineRunError) as caught:
        owner._metadata_only_planning_acquisition(
            database="miiv", export_path="/metadata", llm=llm,
            output_dir=tmp_path / "new-catalog", **options,
        )
    assert caught.value.code == "research_pipeline_required_concept_structurally_unavailable"
    assert caught.value.details["required_concepts"] == ["icu_readmission"]
    assert llm.calls == []
    assert not (tmp_path / "new-catalog").exists()


def test_optional_unsupported_column_is_not_reintroduced(source_menu, tmp_path):
    import pyarrow.parquet as pq

    llm = ScriptedMockLLMClient([json.dumps({
        "selected_concepts": ["death", "icu_readmission"],
        "rationale": "Mortality with optional covariate.", "inclusion_exclusion": [],
    })])
    result = owner._metadata_only_planning_acquisition(
        database="miiv", export_path="/metadata", question="Describe death, not readmission.",
        llm=llm, output_dir=tmp_path / "catalog",
    )
    assert not result.blocked
    assert pq.read_schema(result.universe_path).names == ["stay_id", "death"]
    assert pq.read_metadata(result.universe_path).num_rows == 0
    receipt = json.loads(result.provenance_path.read_text())
    assert receipt["unavailable_model_concepts"] == ["icu_readmission"]



def test_listed_but_unresolvable_optional_pick_is_dropped_not_fatal(tmp_path, monkeypatch):
    """A registry concept the source cannot provide leaves the menu quietly.

    The merged planning menu lists registry concepts next to the export's
    typed columns.  Once typed owners exist, an untyped registry-only entry
    can never resolve, so the model may pick something coverage will call
    missing.  Measured 2026-09-22 on the official eICU demo: one optional
    pick (``icu_unit_type``) blocked the whole run.  An optional pick is
    dropped and recorded; only an empty selection fails closed, with a code.
    """
    import pyarrow.parquet as pq

    from easyicu.research_agent.acquisition import catalog as catalog_module

    typed_death = CatalogConcept(
        "death", file_name="outcome.parquet", typed_metadata=True, column_role="event_status",
    )
    registry_only = CatalogConcept("icu_unit_type", description="ICU unit type", category="demographics")
    monkeypatch.setattr(
        catalog_module, "build_database_capability_catalog",
        lambda _: AvailableCatalog(source="canonical", concepts=[registry_only, CatalogConcept("death")]),
    )
    monkeypatch.setattr(
        catalog_module, "build_available_catalog",
        lambda _: AvailableCatalog(source="export", concepts=[typed_death]),
    )
    llm = ScriptedMockLLMClient([json.dumps({
        "selected_concepts": ["death", "icu_unit_type"],
        "rationale": "Mortality with an optional unit-type covariate.", "inclusion_exclusion": [],
    })])
    result = owner._metadata_only_planning_acquisition(
        database="eicu", export_path="/metadata", question="Describe death by unit.",
        llm=llm, output_dir=tmp_path / "catalog",
    )
    assert not result.blocked
    assert result.blocked_reason_code == ""
    assert pq.read_schema(result.universe_path).names == ["patientunitstayid", "death"]
    receipt = json.loads(result.provenance_path.read_text())
    assert receipt["unavailable_model_concepts"] == ["icu_unit_type"]

    only_unavailable = ScriptedMockLLMClient([json.dumps({
        "selected_concepts": ["icu_unit_type"],
        "rationale": "Unit type only.", "inclusion_exclusion": [],
    })])
    blocked = owner._metadata_only_planning_acquisition(
        database="eicu", export_path="/metadata", question="Describe unit type.",
        llm=only_unavailable, output_dir=tmp_path / "blocked",
    )
    assert blocked.blocked
    assert blocked.blocked_reason_code == "no_available_concepts"
    assert not (tmp_path / "blocked").exists()

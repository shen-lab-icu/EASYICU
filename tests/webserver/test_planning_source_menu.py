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

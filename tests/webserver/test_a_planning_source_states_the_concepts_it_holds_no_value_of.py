"""Planning neither requires nor offers a concept its source holds no value of.

A native export states which concepts it lists without a single value
(``intake.export_package.concepts_without_values``).  The metadata-only
planning catalog stops a study that requires one before the Planner is called,
as it stops one whose source cannot hold a required concept; it takes an
optional pick of one off the menu and records it; and it states the source's
concepts without values beside the catalog, where the planning context reads
them.  Synthetic manifests and catalogs only.
"""

import json

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
from easyicu.research_agent.contracts.concept_values import (
    CONCEPTS_WITHOUT_VALUES_KEY,
    names_without_values,
)
from easyicu.research_agent.intake.export_package import NATIVE_MANIFEST
from easyicu.research_agent.intake.materialized_metadata import MaterializedMetadataError
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.webserver import agent_pipeline_runs as owner
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

_WITHOUT_VALUES = ["apache_iv", "circ_failure"]


@pytest.fixture
def export(tmp_path, monkeypatch):
    from easyicu.research_agent.acquisition import catalog

    names = ("death", "age", "circ_failure", "apache_iv")
    monkeypatch.setattr(
        catalog,
        "build_database_capability_catalog",
        lambda _: AvailableCatalog(
            source="canonical", concepts=[CatalogConcept(name) for name in names]
        ),
    )
    monkeypatch.setattr(
        catalog,
        "build_available_catalog",
        lambda _: AvailableCatalog(
            source="export", concepts=[CatalogConcept("death"), CatalogConcept("age")]
        ),
    )
    root = tmp_path / "export"
    root.mkdir()
    statuses = {
        "outcome": {"death": "available"},
        "demographics": {"age": "available"},
        "circulatory": {"circ_failure": "produced_all_null"},
        "other_scores": {"apache_iv": "structurally_unavailable_placeholder"},
    }
    (root / NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "database": "miiv",
                "files": [
                    {
                        "module": module,
                        "concept_status": {
                            concept: {"availability": availability}
                            for concept, availability in concepts.items()
                        },
                    }
                    for module, concepts in statuses.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize(
    "requirement",
    [
        {"required_concepts": ("circ_failure",)},
        {"target_outcome": "circ_failure"},
    ],
)
def test_a_required_concept_without_values_stops_before_the_planner(
    export, tmp_path, requirement
):
    llm = ScriptedMockLLMClient([json.dumps({
        "selected_concepts": ["death"], "rationale": "Mortality.", "inclusion_exclusion": [],
    })])
    options = {"question": "Does circulatory failure relate to death?", **requirement}
    with pytest.raises(ResearchPipelineRunError) as caught:
        owner._metadata_only_planning_acquisition(
            database="miiv", export_path=export, llm=llm,
            output_dir=tmp_path / "catalog", **options,
        )

    assert caught.value.code == "research_pipeline_required_concept_without_values"
    assert caught.value.details["required_concepts"] == ["circ_failure"]
    assert llm.calls == []
    assert not (tmp_path / "catalog").exists()


def test_an_optional_pick_without_values_leaves_the_menu_and_is_stated(export, tmp_path):
    import pyarrow.parquet as pq

    llm = ScriptedMockLLMClient([json.dumps({
        "selected_concepts": ["death", "circ_failure"],
        "rationale": "Mortality with an optional covariate.", "inclusion_exclusion": [],
    })])
    result = owner._metadata_only_planning_acquisition(
        database="miiv", export_path=export, question="Describe death.",
        llm=llm, output_dir=tmp_path / "catalog",
    )

    assert not result.blocked
    # The model was never offered it.
    [(messages, _options)] = llm.calls
    assert "circ_failure" not in "\n".join(message.content for message in messages)
    assert pq.read_schema(result.universe_path).names == ["stay_id", "death"]
    receipt = json.loads(result.provenance_path.read_text())
    assert receipt["unavailable_model_concepts"] == ["circ_failure"]
    assert receipt[CONCEPTS_WITHOUT_VALUES_KEY] == _WITHOUT_VALUES
    # The planning context states them, so a criterion over one is refused
    # by its owner (``planning.population_compile``).
    context = build_research_context(
        research_question="Describe death.",
        cohort=pd.read_parquet(result.universe_path),
        cohort_name="planning",
        database="miiv",
    )
    assert context.cohort.provenance[CONCEPTS_WITHOUT_VALUES_KEY] == _WITHOUT_VALUES
    assert set(_WITHOUT_VALUES) <= names_without_values(context)


def test_a_catalog_that_misstates_its_concepts_without_values_is_refused():
    frame = pd.DataFrame({"stay_id": pd.Series(dtype="int64")})
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog",
        "patient_rows_read": False,
        CONCEPTS_WITHOUT_VALUES_KEY: "circ_failure",
    }
    with pytest.raises(MaterializedMetadataError, match="without values"):
        build_research_context(
            research_question="Describe death.",
            cohort=frame,
            cohort_name="planning",
            database="miiv",
        )

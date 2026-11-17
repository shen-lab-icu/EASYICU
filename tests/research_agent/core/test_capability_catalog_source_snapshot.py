"""A metadata menu shares source models without changing capability answers."""

import pytest

from easyicu import resources
from easyicu.research_agent import concept_availability as availability
from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog


@pytest.fixture
def source_reads(monkeypatch):
    counts = {"dictionary": 0, "registry": 0}
    dictionary = resources.load_dictionary
    registry = resources.load_data_sources

    def read_dictionary(**kwargs):
        counts["dictionary"] += 1
        return dictionary(**kwargs)

    def read_registry():
        counts["registry"] += 1
        return registry()

    monkeypatch.setattr(resources, "load_dictionary", read_dictionary)
    monkeypatch.setattr(resources, "load_data_sources", read_registry)
    availability._explain_concept_availability_cached.cache_clear()
    yield counts
    availability._explain_concept_availability_cached.cache_clear()


def test_a_snapshot_preserves_direct_derived_and_blocked_capabilities(source_reads):
    concepts = ["crea", "sofa2_cns", "susp_inf"]
    expected = [
        availability.explain_concept_availability(concept=c, database="hirid")
        for c in concepts
    ]
    assert source_reads["dictionary"] > 1
    availability._explain_concept_availability_cached.cache_clear()
    source_reads.update(dictionary=0, registry=0)
    with availability.availability_source_snapshot("hirid"):
        actual = [
            availability.explain_concept_availability(concept=c, database="hirid")
            for c in concepts
        ]
    assert actual == expected
    assert source_reads == {"dictionary": 1, "registry": 1}


def test_a_cold_database_menu_loads_metadata_once_per_source_scope(source_reads):
    catalog = build_database_capability_catalog("miiv")
    assert catalog.concepts
    # One generic concept roster and one database-specific availability
    # dictionary, independent of the number of concepts and dependencies.
    assert source_reads == {"dictionary": 2, "registry": 1}


def test_a_failed_projection_releases_its_source_snapshot(source_reads):
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with availability.availability_source_snapshot("miiv"):
            raise RuntimeError("synthetic failure")
    availability.explain_concept_availability(concept="crea", database="miiv")
    assert source_reads == {"dictionary": 2, "registry": 2}

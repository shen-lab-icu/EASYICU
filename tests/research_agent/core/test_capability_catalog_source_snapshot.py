"""A metadata menu shares source models without changing capability answers."""

import pytest

from easyicu import resources
from easyicu.research_agent import concept_availability as availability
from easyicu.research_agent.acquisition import catalog as catalog_owner
from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog

# Every cache on the menu's source-read path: a test file that ran earlier in
# the same worker must not change the counts read here (full Linux batch at
# d810aadd1, -n 6 --dist loadfile: one dictionary read instead of two).
_SOURCE_READ_CACHES = (
    availability._explain_concept_availability_cached,
    availability.concept_records_one_value_per_stay,
    availability.stay_level_concept_category,
    catalog_owner._methodology_dictionary,
)


def _clear_source_read_caches():
    for cached in _SOURCE_READ_CACHES:
        cached.cache_clear()


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
    _clear_source_read_caches()
    yield counts
    _clear_source_read_caches()


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

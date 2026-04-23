from __future__ import annotations

from easyicu.resources import load_data_sources, load_dictionary


def test_concept_dictionary_contains_core_cross_database_concepts() -> None:
    dictionary = load_dictionary(include_sofa2=True)

    assert len(list(dictionary.keys())) >= 140
    for concept_name in ["hr", "sofa2", "sep3_sofa2", "rrt"]:
        assert concept_name in dictionary


def test_data_sources_cover_supported_public_icu_databases() -> None:
    data_sources = load_data_sources()
    supported = {source.name for source in data_sources}

    assert {"mimic", "miiv", "eicu", "aumc", "hirid", "sic"}.issubset(supported)

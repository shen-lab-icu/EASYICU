from __future__ import annotations

import pytest

from easyicu.resources import load_data_sources, load_dictionary


def test_concept_dictionary_contains_core_cross_database_concepts() -> None:
    dictionary = load_dictionary(include_sofa2=True)

    assert len(list(dictionary.keys())) >= 140
    for concept_name in ["hr", "sofa2", "sep3_sofa2", "rrt"]:
        assert concept_name in dictionary


@pytest.mark.clinical_conformance
def test_sofa2_sepsis_phenotype_is_explicitly_experimental() -> None:
    dictionary = load_dictionary(include_sofa2=True)
    definition = dictionary["sep3_sofa2"]

    assert definition.clinical_status == "experimental"
    assert definition.canonical_definition is False
    assert definition.requires_explicit_opt_in is True
    assert definition.definition_source == "SOFA-2 Table 2 + Sepsis-3 sensitivity adaptation"


def test_data_sources_cover_supported_public_icu_databases() -> None:
    data_sources = load_data_sources()
    supported = {source.name for source in data_sources}

    assert {"mimic", "miiv", "eicu", "aumc", "hirid", "sic"}.issubset(supported)

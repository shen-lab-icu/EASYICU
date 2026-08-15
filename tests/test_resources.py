from __future__ import annotations

import json

import pytest

from easyicu.concept.schema import ConceptDictionary
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


def _write_overlay_pair(tmp_path):
    base = {
        "crea": {
            "unit": "mg/dL",
            "min": 0.1,
            "depends_on": ["base_dependency"],
            "sources": {
                "miiv": [
                    {"table": "labevents", "sub_var": "itemid", "ids": [50912]}
                ]
            },
        }
    }
    overlay = {
        "crea": {
            "sources": {
                "custom": [
                    {"table": "labs", "sub_var": "code", "ids": ["CREA"]}
                ]
            }
        }
    }
    base_path = tmp_path / "base.json"
    overlay_path = tmp_path / "overlay.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    overlay_path.write_text(json.dumps(overlay), encoding="utf-8")
    return base_path, overlay_path


def _assert_source_patch_preserves_definition(dictionary) -> None:
    definition = dictionary["crea"]
    assert set(definition.sources) == {"miiv", "custom"}
    assert definition.units == ["mg/dL"]
    assert definition.minimum == 0.1
    assert definition.depends_on == ["base_dependency"]


def test_resource_overlay_uses_per_concept_patch_semantics(tmp_path) -> None:
    _write_overlay_pair(tmp_path)

    dictionary = load_dictionary(
        "base", directories=[tmp_path], extras=["overlay"]
    )

    _assert_source_patch_preserves_definition(dictionary)


def test_multiple_json_uses_same_per_concept_patch_semantics(tmp_path) -> None:
    base_path, overlay_path = _write_overlay_pair(tmp_path)

    dictionary = ConceptDictionary.from_multiple_json([base_path, overlay_path])

    _assert_source_patch_preserves_definition(dictionary)

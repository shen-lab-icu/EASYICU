from __future__ import annotations

import json

import pandas as pd
import pytest

from easyicu.concept.export_metadata import ExportMetadataError, build_export_file_metadata_binding
from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.concept.schema import ConceptDictionary
from easyicu.resources import load_data_sources, load_dictionary


def test_concept_dictionary_contains_core_cross_database_concepts() -> None:
    dictionary = load_dictionary(include_sofa2=True)

    assert len(list(dictionary.keys())) >= 140
    for concept_name in ["hr", "sofa2", "sep3_sofa2", "rrt"]:
        assert concept_name in dictionary


@pytest.mark.parametrize("name", ["concept-dict", "concept-dict.json"])
@pytest.mark.parametrize("include_sofa2", [False, True])
def test_the_readmitted_stay_attribute_declares_its_logical_domain(name, include_sofa2) -> None:
    dictionary = load_dictionary(name, include_sofa2=include_sofa2)
    definition = dictionary.get("icu_stay_is_readmission")
    assert definition is not None
    assert definition.class_name == "lgl_cncpt"
    assert definition.sources == {}  # The outcomes loader owns the stay sequence.
    frame = pd.DataFrame({"stay_id": [1, 2, 3], "icu_stay_is_readmission": [0, 1, None]})
    binding = build_export_file_metadata_binding(
        relative_path="demographics.parquet", module="demographics", frame=frame,
        concept_ids=("icu_stay_is_readmission",), database="miiv",
        database_class_prefixes=(), dictionary=dictionary,
    )
    assert binding.columns["icu_stay_is_readmission"].metadata.role is ConceptColumnRole.EVENT_STATUS
    frame.loc[0, "icu_stay_is_readmission"] = 65.0
    with pytest.raises(ExportMetadataError, match="column_metadata_value_domain_invalid"):
        build_export_file_metadata_binding(
            relative_path="demographics.parquet", module="demographics", frame=frame,
            concept_ids=("icu_stay_is_readmission",), database="miiv",
            database_class_prefixes=(), dictionary=dictionary,
        )


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
    assert set(dictionary.keys()) == {"crea"}


def test_an_external_default_dictionary_keeps_its_explicit_membership(tmp_path) -> None:
    (tmp_path / "concept-dict.json").write_text(json.dumps({"synthetic_marker": {"sources": {}}}))

    assert set(load_dictionary(directories=[tmp_path]).keys()) == {"synthetic_marker"}


def test_a_caller_overlay_can_refine_the_computed_stay_attribute(tmp_path) -> None:
    (tmp_path / "override.json").write_text(json.dumps({
        "icu_stay_is_readmission": {"description": "Explicit synthetic stay attribute"},
    }))

    definition = load_dictionary(directories=[tmp_path], extras=["override"])["icu_stay_is_readmission"]

    assert definition.class_name == "lgl_cncpt"
    assert definition.description == "Explicit synthetic stay attribute"
    assert definition.sources == {}


def test_multiple_json_uses_same_per_concept_patch_semantics(tmp_path) -> None:
    base_path, overlay_path = _write_overlay_pair(tmp_path)

    dictionary = ConceptDictionary.from_multiple_json([base_path, overlay_path])

    _assert_source_patch_preserves_definition(dictionary)

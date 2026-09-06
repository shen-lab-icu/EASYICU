"""Catalog-only outputs retain producer metadata before patient extraction."""

import pandas as pd
import pytest

from easyicu import get_concept_info
from easyicu.concept.catalog import CONCEPT_DICTIONARY
from easyicu.concept.export_metadata import build_export_file_metadata_binding
from easyicu.resources import load_dictionary
from easyicu.research_agent.authority.declared_levels import closed_planning_levels_for
from easyicu.research_agent.planning.progressive_host_materialization import _table_summary
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import ConceptDescriptor


@pytest.mark.parametrize("concept", ["icu_readmission", "mort_28d", "mort_90d"])
def test_catalog_only_flags_have_the_same_type_before_and_after_export(concept):
    info = get_concept_info(concept)
    assert info["name"] == concept
    assert info["description"]
    assert info["category"] == "outcome"
    assert info["class_name"] == "lgl_cncpt"
    assert info["sources"] == []  # Catalog metadata is not raw-source authority.

    frame = pd.DataFrame({
        "stay_id": pd.Series(dtype="int64"),
        concept: pd.Series(dtype="float64"),
    })
    context = build_research_context(
        research_question="Describe the available population.",
        cohort=frame, cohort_name="row_free_catalog", database="miiv",
    )
    variable = context.variable(concept)
    assert variable.source_concept == concept
    assert variable.observed_domain is None
    assert variable.missingness.n_total == 0
    assert _table_summary(variable) == "count_percent"
    assert closed_planning_levels_for(name=concept, variables={concept: variable}) == [0, 1]

    binding = build_export_file_metadata_binding(
        relative_path="outcome.parquet", module="outcome", database="miiv",
        frame=pd.DataFrame({"stay_id": [1, 2], concept: [True, False]}),
        concept_ids=(concept,), database_class_prefixes=(), dictionary=load_dictionary(),
    )
    assert binding.columns[concept].metadata.description == info["description"]
    assert binding.columns[concept].metadata.role.value == "event_status"


def test_catalog_numeric_output_does_not_acquire_a_boolean_domain():
    info = get_concept_info("icu_free_days_28")
    assert info["class_name"] != "lgl_cncpt"
    assert info["unit"] == CONCEPT_DICTIONARY["icu_free_days_28"][2]
    variable = ConceptDescriptor(
        name="icu_free_days_28", source_concept="icu_free_days_28", dtype="float64",
    )
    assert closed_planning_levels_for(
        name=variable.name, variables={variable.name: variable},
    ) == []


def test_unknown_concept_is_not_promoted_to_catalog_metadata():
    with pytest.raises(ValueError, match="未知概念"):
        get_concept_info("unknown_derived_flag")


def test_dictionary_definition_keeps_precedence_over_catalog_label():
    assert get_concept_info("hr")["description"] == "heart rate"
    assert get_concept_info("hr")["units"] == ["bpm", "/min"]


def test_executable_output_alias_retains_definition_review_boundaries():
    public = get_concept_info("sep3_sofa1")
    source = get_concept_info("sep3")
    assert public["name"] == "sep3_sofa1"
    assert public["sources"] == []
    for field in (
        "clinical_status", "canonical_definition", "definition_source",
        "definition_version", "clinical_contract_id",
    ):
        assert public[field] == source[field]
    assert public["clinical_contract_id"] == "sepsis3_2016"

    context = build_research_context(
        research_question="Describe Sepsis-3 status.",
        cohort=pd.DataFrame({"stay_id": [1, 2], "sep3_sofa1_max": [0, 1]}),
        cohort_name="synthetic", database="miiv",
    )
    reference = context.variable("sep3_sofa1_max").clinical_definition
    assert reference.contract_id == "sepsis3_2016"
    assert "independent_clinical_review_pending" in reference.validation_status
    assert reference.database_conformance["miiv"] == "mapping_only"


def test_provenance_only_composite_is_not_a_definition_alias():
    assert get_concept_info("icu_readmission")["clinical_contract_id"] is None
    assert get_concept_info("sep3_sofa2")["clinical_contract_id"] != "sepsis3_2016"

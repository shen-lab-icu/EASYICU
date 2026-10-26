"""Cross-database concept availability uses EasyICU concept metadata."""

from __future__ import annotations

import pytest

from easyicu.concept.availability_signal import ConceptAvailabilityRecord
from easyicu.research_agent.concept_availability import (
    concept_database_availability_from_load_record,
)


@pytest.mark.parametrize("database,concept,missing", [
    (database, concept, "samp") for database in ("hirid", "sic")
    for concept in ("susp_inf", "sep3", "sep3_sofa2")
] + [("sic", "sofa2_cns", "gcs")])
def test_required_recipe_input_blocks_availability_and_names_the_input(database, concept, missing):
    from easyicu.databases.research_capabilities import research_capability_profile
    from easyicu.databases.research_requirements import StudyRequirements, assess_study
    from easyicu.research_agent.concept_availability import explain_concept_availability

    cell = explain_concept_availability(concept=concept, database=database)
    assert cell.status == "blocked"
    assert not cell.available
    assert cell.structural_unavailable
    assert cell.reason == "required_dependency_blocked"
    assert missing in cell.missing_dependencies
    profile = research_capability_profile(database)
    assert profile.concept(concept).derivation_status == "blocked"
    assert missing in profile.concept(concept).missing_dependencies
    decision = assess_study(profile, StudyRequirements(family="association_study", concepts=(concept,)))
    assert decision.status == "refusal"
    assert decision.missing[0].code == "required_concept_unmapped"


def test_required_inputs_follow_the_recipe_for_an_invented_source(monkeypatch):
    from easyicu.concept.schema import ConceptDefinition, ConceptDictionary, ConceptSource
    from easyicu.research_agent.concept_availability import explain_concept_availability
    from easyicu import resources

    dictionary = ConceptDictionary({
        "invented_score": ConceptDefinition(name="invented_score", sources={},
            callback="sofa2_cns", sub_concepts=["gcs", "optional_measurement"]),
        "gcs": ConceptDefinition(name="gcs", sources={}),
        "optional_measurement": ConceptDefinition(name="optional_measurement", sources={
            "invented-source": [ConceptSource(table="measurements", value_var="value")]}),
    })
    monkeypatch.setattr(resources, "load_dictionary", lambda **kwargs: dictionary)
    cell = explain_concept_availability(concept="invented_score", database="invented_source")
    assert cell.status == "blocked"
    assert cell.missing_dependencies == ["gcs"]
    assert cell.available_dependencies == ["optional_measurement"]


@pytest.mark.parametrize("database,concept", [("eicu", "susp_inf"), ("eicu_demo", "susp_inf"),
                                              ("hirid", "sofa2_cns"), ("miiv", "sofa2")])
def test_optional_missing_inputs_do_not_block_an_evaluable_recipe(database, concept):
    from easyicu.research_agent.concept_availability import explain_concept_availability

    assert explain_concept_availability(concept=concept, database=database).status != "blocked"


def test_cross_database_concept_availability_resolves_recursive_sofa2(ra):
    availability = ra.cross_database_concept_availability(
        concepts=["sofa2", "creatinine"],
        databases=["miiv", "eicu", "hirid"],
    )

    assert set(availability) == {"sofa2", "creatinine"}
    assert availability["creatinine"]["miiv"]["concept"] == "crea"
    assert availability["creatinine"]["miiv"]["status"] == "full"
    assert availability["sofa2"]["miiv"]["status"] in {"full", "degraded"}
    assert availability["sofa2"]["miiv"]["available"] is True


def test_hypothesis_cross_database_feasibility_summarizes_statuses(ra):
    summary = ra.hypothesis_cross_database_feasibility(
        concepts=["kdigo_aki", "death"],
        databases=["miiv", "sicdb"],
    )

    assert "kdigo_aki" in summary["concept_dependencies"]
    assert set(summary["cross_database_feasibility"]) == {"miiv", "sic"}
    assert summary["cross_database_feasibility"]["miiv"] in {
        "full",
        "degraded",
        "blocked",
    }
    assert isinstance(summary["degraded_reason"], dict)


def test_common_clinical_aliases_resolve_to_packaged_concepts(ra):
    summary = ra.hypothesis_cross_database_feasibility(
        concepts=["lactate", "hospital_mortality"],
        databases=["miiv"],
    )

    assert summary["concept_dependencies"] == ["lact", "death"]
    assert summary["cross_database_feasibility"]["miiv"] != "blocked"


def test_runtime_load_availability_maps_to_research_agent_status_terms():
    record = ConceptAvailabilityRecord(
        concept="norepi_rate",
        database="mimic",
        reason="source_unavailable",
        n_rows=0,
        sources_defined=("inputevents",),
        missing_tables=("inputevents",),
    )

    cell = concept_database_availability_from_load_record(
        record,
        requested_concept="norepinephrine",
    )

    assert cell.concept == "norepi_rate"
    assert cell.requested_concept == "norepinephrine"
    assert cell.status == "blocked"
    assert cell.available is False
    assert cell.reason == "source_unavailable"
    assert cell.runtime_reason == "source_unavailable"
    assert cell.structural_unavailable is True
    assert cell.source_missing_tables == ["inputevents"]

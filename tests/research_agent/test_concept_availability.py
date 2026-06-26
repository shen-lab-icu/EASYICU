"""Cross-database concept availability uses EasyICU concept metadata."""

from __future__ import annotations

from easyicu.concept.availability_signal import ConceptAvailabilityRecord
from easyicu.research_agent.concept_availability import (
    concept_database_availability_from_load_record,
)


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

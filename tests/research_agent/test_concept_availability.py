"""Cross-database concept availability uses EasyICU concept metadata."""

from __future__ import annotations


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

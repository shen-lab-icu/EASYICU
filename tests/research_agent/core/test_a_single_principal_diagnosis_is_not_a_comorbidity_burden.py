"""A single principal diagnosis is not a comorbidity burden.

The Charlson and Elixhauser indices score the conditions a patient carries
into an admission, coded over all of the admission's diagnoses.  SICdb records
one ICD-10 principal diagnosis per case (``cases.ICD10Main``) and no other
diagnosis source, so an index computed there scores the reason for admission:
most cases score 0, and the score a case does get is not its comorbidity
burden.  The comorbidity loader nevertheless read that one column, and the
planning menu offered both indices on SICdb, where a study could use Charlson
as an exposure, a Table 1 row or an adjustment and report it as comorbidity.

The loader declaration now names SICdb as a database without a comorbidity
source, with its own reason.  The loader, the availability owner and the
planning menu read the one declaration.  Packaged metadata only; no table is
opened.
"""

from __future__ import annotations

import pytest

from easyicu.concept_output_sources import (
    COMPOSITE_LOADER_SUPPORT,
    CompositeLoaderSupport,
)
from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog
from easyicu.research_agent.concept_availability import explain_concept_availability
from easyicu.scores import comorbidity

_INDICES = ("charlson", "elixhauser")


def test_sicdb_has_no_comorbidity_source_for_its_own_reason() -> None:
    support = COMPOSITE_LOADER_SUPPORT["comorbidity_loader"]

    assert "sic" in support.no_source_databases
    assert support.reason_unavailable("sic") == "single_principal_diagnosis_only"
    # The databases without any diagnosis source keep theirs.
    assert support.reason_unavailable("hirid") == "no_icd_diagnosis_source"
    assert support.reason_unavailable("aumc") == "no_icd_diagnosis_source"


@pytest.mark.parametrize("database", ["sic", "sic_demo"])
@pytest.mark.parametrize("concept", _INDICES)
def test_the_availability_owner_reports_the_index_unavailable_on_sicdb(
    concept: str, database: str
) -> None:
    cell = explain_concept_availability(concept=concept, database=database)

    assert cell.status == "blocked"
    assert cell.available is False
    assert cell.structural_unavailable is True
    assert cell.reason == "single_principal_diagnosis_only"


@pytest.mark.parametrize("database", ["miiv", "eicu"])
def test_a_database_with_its_diagnoses_still_has_the_indices(database: str) -> None:
    for concept in _INDICES:
        cell = explain_concept_availability(concept=concept, database=database)
        assert cell.status != "blocked", (concept, database, cell.reason)
    offered = {
        item.concept_id for item in build_database_capability_catalog(database).concepts
    }
    assert set(_INDICES) <= offered


def test_the_planning_menu_does_not_offer_the_indices_on_sicdb() -> None:
    offered = {
        item.concept_id for item in build_database_capability_catalog("sic").concepts
    }

    assert not set(_INDICES) & offered


@pytest.mark.parametrize("database", ["sic", "sic_demo", "SIC"])
def test_the_loader_opens_no_table_on_sicdb(monkeypatch, database: str) -> None:
    def no_tables(*args, **kwargs):
        pytest.fail("a database without a comorbidity source must not open a table")

    monkeypatch.setattr(comorbidity, "_build_datasource", no_tables)

    assert comorbidity.load_comorbidity(database, system="charlson").empty
    assert comorbidity.load_comorbidity(database, system="elixhauser").empty


def test_a_reason_cannot_name_a_database_that_has_a_source() -> None:
    with pytest.raises(ValueError, match="names a database with a source"):
        CompositeLoaderSupport(
            no_source_databases=frozenset({"hirid"}),
            no_source_reason="no_icd_diagnosis_source",
            no_source_reasons=(("miiv", "single_principal_diagnosis_only"),),
        )

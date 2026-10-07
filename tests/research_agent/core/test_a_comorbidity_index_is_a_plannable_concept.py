"""A comorbidity index is a concept a study can plan with.

The comorbidity loader derives the Charlson and Elixhauser indices from ICD
diagnoses wherever a database has a diagnosis source, and the availability
owner reports them there (CA).  They were still absent from the research
agent's concept catalog, so the planning menu never offered them: a planning
run could not carry the most common adjustment for baseline comorbidity into
a later extraction.  They join the catalog under their clinical names.

The catalog lists code-derived outputs, so its membership is not the
dictionaries'.  The literature owner read it as such, and the Elixhauser
export display label ("... (van Walraven) Score") would have become a PubMed
search phrase.  Membership is now the dictionaries' own; for any code-derived
output the display label yields to the catalog's clinical name, not to the
output's identifier.  Synthetic contexts and packaged metadata only.
"""

from __future__ import annotations

import pytest

from easyicu.concept_output_sources import COMPOSITE_LOADER_SUPPORT
from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog
from easyicu.research_agent.concept_catalog import concept_dictionary_ids, load_concept_catalog
from easyicu.research_agent.literature import _protocol_search_term
from easyicu.research_agent.literature_concepts import is_export_display_label
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext

_INDICES = ("charlson", "elixhauser")


@pytest.mark.parametrize("database", ["miiv", "eicu", "sic", "aumc", "hirid"])
def test_the_planning_menu_offers_the_indices_where_their_loader_runs(database) -> None:
    offered = {item.concept_id for item in build_database_capability_catalog(database).concepts}
    no_source = COMPOSITE_LOADER_SUPPORT["comorbidity_loader"].no_source_databases

    if database in no_source:
        assert not set(_INDICES) & offered
    else:
        assert set(_INDICES) <= offered


def test_the_catalog_names_the_indices_clinically() -> None:
    catalog = load_concept_catalog()

    charlson = [alias.lower() for alias in catalog.concept_aliases["charlson"]]
    elixhauser = [alias.lower() for alias in catalog.concept_aliases["elixhauser"]]

    assert charlson[0] == "charlson comorbidity index"
    assert elixhauser[0] == "elixhauser comorbidity index"
    assert "van walraven score" in elixhauser
    # No bare acronym that also names chronic critical illness.
    assert "cci" not in charlson
    # An index is an adjustment or an exposure, never a 0/1 outcome.
    assert not set(_INDICES) & set(catalog.outcome_determinability)


def test_dictionary_membership_is_the_dictionaries_own() -> None:
    dictionary = concept_dictionary_ids()
    catalog = set(load_concept_catalog().available_concepts)

    assert {"death", "age", "lact"} <= dictionary
    assert {"charlson", "elixhauser", "mort_28d", "sep3_sofa1"} <= catalog - dictionary


def _context(name: str, description: str) -> ResearchContext:
    return ResearchContext(
        research_question="Is the exposure associated with in-hospital mortality in ICU stays?",
        cohort=CohortDescriptor(cohort_name="ICU", database="synthetic", n_stays=0),
        variables=[
            ConceptDescriptor(
                name=name, dtype="float64", source_concept=name, description=description,
            ),
            ConceptDescriptor(
                name="death", dtype="int64", source_concept="death",
                description="in hospital mortality",
            ),
        ],
        primary_exposure=name,
        target_outcome="death",
    )


@pytest.mark.parametrize(
    ("name", "label", "phrase"),
    [
        ("charlson", "Charlson Comorbidity Index", "Charlson comorbidity index"),
        ("elixhauser", "Elixhauser (van Walraven) Score", "Elixhauser comorbidity index"),
        ("mort_28d", "28-day Mortality", "28-day mortality"),
        ("uo_rt_6hr", "Urine Output Rate (6h rolling window)", "urine output rate 6 hours"),
        ("sep3_sofa1", "Sepsis-3 (SOFA-1 based)", "Sepsis-3"),
    ],
)
def test_a_code_derived_display_label_yields_to_its_clinical_name(name, label, phrase) -> None:
    assert is_export_display_label(label, [name])
    assert _protocol_search_term(_context(name, label), name) == phrase


def test_a_dictionary_concept_keeps_its_dictionary_text() -> None:
    context = _context("charlson", "Charlson Comorbidity Index")

    assert not is_export_display_label("in hospital mortality", ["death"])
    assert _protocol_search_term(context, "death") == "in hospital mortality"

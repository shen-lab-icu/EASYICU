"""Ventilation exports are searched as the clinical exposure, not their storage form."""

from __future__ import annotations

import pytest

from easyicu.research_agent.literature import _protocol_search_term
from easyicu.research_agent.literature_concepts import literature_concept_identity
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext


def _context(question: str, description: str) -> ResearchContext:
    binary = {"is_binary": True, "n_unique": 2, "levels": [0, 1]}
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0, id_columns=["stay_id"]),
        primary_exposure="vent_ind", target_outcome="mort_28d",
        variables=[
            ConceptDescriptor(name="stay_id", role="other", dtype="int64"),
            ConceptDescriptor(name="vent_ind", role="intervention", dtype="bool", observed_domain=binary,
                              description=description),
            ConceptDescriptor(name="mort_28d", role="outcome", dtype="bool", observed_domain=binary),
        ],
    )


@pytest.mark.parametrize("question", [
    "在 ICU 第一天接受机械通气的患者 28 天生存是否不同？",
    "Is first-day ventilation associated with 28-day survival?",
])
def test_the_ventilation_exposure_is_searched_as_mechanical_ventilation(question):
    term = _protocol_search_term(_context(question, "ventilation durations"), "vent_ind")
    assert term == "mechanical ventilation"


@pytest.mark.parametrize("concept", ["vent_ind", "mech_vent"])
def test_ventilation_concepts_carry_a_reviewed_mesh_label(concept):
    identity = literature_concept_identity(concept)
    assert identity.canonical_phrase == "mechanical ventilation"
    assert identity.mesh_terms == ("Respiration, Artificial",)


def test_other_ventilation_exports_keep_their_own_identity():
    assert literature_concept_identity("vent_mode").canonical_phrase == "mechanical ventilation mode"

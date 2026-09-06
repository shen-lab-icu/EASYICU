"""A score's version-specific definition must survive column renaming."""

from easyicu.research_agent.literature import _curated_for
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)


def _citations(name, source=None):
    context = ResearchContext(
        research_question="Assess the declared organ score in adult ICU stays.",
        cohort=CohortDescriptor(cohort_name="test", database="miiv", n_stays=0),
        variables=[ConceptDescriptor(name=name, dtype="float64", source_concept=source)],
    )
    return {citation.key: citation for citation in _curated_for(context)}


def test_sofa2_gets_its_2025_definition_not_only_the_original_score() -> None:
    citations = _citations("sofa2_liv_max")
    source = citations["ranzani_sofa2_2025"]

    assert source.pmid == "41159833"
    assert source.doi == "10.1001/jama.2025.20516"
    assert source.year == "2025"
    assert "SOFA)-2" in source.title
    assert "vincent_sofa_1996" in citations


def test_renamed_sofa2_column_retains_its_owner_definition() -> None:
    assert "ranzani_sofa2_2025" in _citations("liver_score", "sofa2_liv")


def test_original_sofa_does_not_silently_become_sofa2() -> None:
    citations = _citations("sofa_liv_max")
    assert "vincent_sofa_1996" in citations
    assert "ranzani_sofa2_2025" not in citations

from easyicu.research_agent.literature_concepts import (
    concept_id,
    literature_concept_phrase,
)
from easyicu.webserver.ideas import direct_evidence_search


def test_materialized_concepts_share_one_literature_phrase_owner() -> None:
    assert concept_id("lact_first") == "lact"
    assert literature_concept_phrase("lact_first") == "lactate"
    assert direct_evidence_search.concept_phrase("lact_first") == "lactate"


def test_unknown_concepts_use_neutral_owner_supplied_fallbacks() -> None:
    assert literature_concept_phrase("creatinine_first", fallback="Creatinine") == (
        "Creatinine"
    )
    assert literature_concept_phrase("novel_marker") == "novel marker"


def test_non_e1_scope_uses_the_same_typed_query_compiler() -> None:
    clause = direct_evidence_search.build_scope_clause(
        {
            "exposure_concept": "lact_first",
            "outcome_concept": "aki",
        }
    )

    assert '"lactate"[Title/Abstract]' in clause
    assert '"acute kidney injury"[Title/Abstract]' in clause
    assert '"AKI"[Title/Abstract]' in clause
    assert "Sepsis" not in clause

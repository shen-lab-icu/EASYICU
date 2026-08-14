from easyicu.research_agent.literature_concepts import (
    concept_id,
    literature_concept_identity,
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


def test_shared_dictionary_projects_non_benchmark_icu_concepts_to_literature() -> None:
    map_identity = literature_concept_identity("map_first")
    norepinephrine_identity = literature_concept_identity("norepi_rate_first")

    assert map_identity is not None
    assert map_identity.concept_id == "map"
    assert map_identity.canonical_phrase == "mean arterial pressure"
    assert map_identity.retrieval_alternatives == (("mean arterial pressure",),)
    assert norepinephrine_identity is not None
    assert norepinephrine_identity.canonical_phrase == "norepinephrine rate"


def test_non_benchmark_query_uses_dictionary_projection_without_sepsis_terms() -> None:
    clause = direct_evidence_search.build_scope_clause(
        {
            "exposure_concept": "map_first",
            "outcome_concept": "rrt",
        }
    )

    assert '"mean arterial pressure"[Title/Abstract]' in clause
    assert '"renal replacement therapy in use"[Title/Abstract]' in clause
    assert "Sepsis" not in clause


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

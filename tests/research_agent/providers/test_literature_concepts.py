import pytest

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


def test_strict_kdigo_physical_column_uses_clinical_retrieval_identity() -> None:
    from easyicu.research_agent.literature import (
        build_pubmed_protocol_queries_for_context,
    )
    from easyicu.research_agent.schema import (
        CohortDescriptor, ConceptDescriptor, ResearchContext,
    )

    context = ResearchContext(
        research_question="KDIGO AKI stage and in-hospital mortality in ICU stays",
        cohort=CohortDescriptor(cohort_name="ICU", database="miiv", n_stays=100),
        variables=[
            ConceptDescriptor(name="aki_stage_strict", dtype="int64"),
            ConceptDescriptor(name="death", dtype="int64"),
        ],
        primary_exposure="aki_stage_strict", target_outcome="death",
    )
    identity = literature_concept_identity("aki_stage_strict")
    assert identity is not None
    assert identity.canonical_phrase == "KDIGO acute kidney injury"
    queries = build_pubmed_protocol_queries_for_context(context)
    assert all('"aki stage strict"' not in query for query in queries)
    assert all('"KDIGO"' in query or '"AKI"' in query for query in queries)


def test_protocol_definition_is_not_used_as_fluid_balance_search_phrase() -> None:
    identity = literature_concept_identity("fluid_balance_cumulative")

    assert identity is not None
    assert identity.canonical_phrase == "cumulative fluid balance"
    assert literature_concept_phrase("fluid_balance_cumulative") == (
        "cumulative fluid balance"
    )
    assert "pre-admission" not in identity.canonical_phrase
    assert literature_concept_phrase("Mechanical ventilation liberation outcomes") == (
        "ventilator liberation"
    )


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


def _derived_column_context(*, name, source_concept, description, question):
    from easyicu.research_agent.schema import (
        CohortDescriptor, ConceptDescriptor, ResearchContext,
    )

    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="ICU", database="eicu_demo", n_stays=100),
        variables=[
            ConceptDescriptor(
                name=name, dtype="float64",
                source_concept=source_concept, description=description,
            ),
            ConceptDescriptor(
                name="death", dtype="int64",
                source_concept="death", description="in hospital mortality",
            ),
        ],
        primary_exposure=name, target_outcome="death",
    )


_DERIVED_COLUMNS = [
    # A host-derived column named after its concept, bound to a component
    # source concept that has no clinical name of its own.
    dict(
        name="aki_stage_strict", source_concept="aki_stage_creat_reference",
        description="Reference AKI Stage (Creatinine)",
        question="KDIGO AKI stage and in-hospital mortality in ICU stays",
        clinical_atom='"AKI"[Title/Abstract]', clinical_term="acute kidney injury",
        implementation_text="Reference AKI Stage",
        title="Acute kidney injury and in-hospital mortality in critically ill adults",
    ),
    dict(
        name="lact", source_concept="lact_arterial_reference",
        description="Reference Lactate (Arterial)",
        question="Lactate and in-hospital mortality in ICU stays",
        clinical_atom='"lactate"[Title/Abstract]', clinical_term="lactate",
        implementation_text="Reference Lactate",
        title="Lactate and in-hospital mortality in critically ill adults",
    ),
]


@pytest.mark.parametrize("case", _DERIVED_COLUMNS, ids=lambda case: case["name"])
def test_a_column_retrieves_through_the_clinical_name_of_its_own_concept(case) -> None:
    from easyicu.research_agent.literature import (
        CitationRecord,
        _screening_decision_for_record,
        _variable_focus_terms,
        build_pubmed_protocol_queries_for_context,
    )

    context = _derived_column_context(
        name=case["name"], source_concept=case["source_concept"],
        description=case["description"], question=case["question"],
    )

    queries = build_pubmed_protocol_queries_for_context(context)
    focus = _variable_focus_terms(context, case["name"])
    decision = _screening_decision_for_record(
        context=context,
        record=CitationRecord(
            key="comparator", year="2021", title=case["title"],
            relevance=(
                "Study-design excerpt: This retrospective cohort study of adult "
                "intensive care unit patients evaluated the association between "
                f"{case['clinical_term']} and in-hospital mortality."
            ),
            publication_types=["Observational Study"],
        ),
        source="pubmed", query=None,
    )

    assert queries
    assert all(case["implementation_text"] not in query for query in queries)
    assert case["clinical_atom"] in queries[0]
    assert case["clinical_term"] in focus
    assert decision.exposure_match
    assert decision.disposition == "include"


def test_an_export_display_label_is_never_a_search_phrase() -> None:
    from easyicu.research_agent.literature import (
        _protocol_search_term,
        build_pubmed_protocol_queries_for_context,
    )
    from easyicu.research_agent.literature_concepts import is_export_display_label

    # The Elixhauser index is a code-derived public output: it has no
    # dictionary entry, so its description is the export display label.
    context = _derived_column_context(
        name="elixhauser", source_concept="elixhauser",
        description="Elixhauser (van Walraven) Score",
        question="Comorbidity burden and in-hospital mortality in ICU stays",
    )

    queries = build_pubmed_protocol_queries_for_context(context)

    assert is_export_display_label("Elixhauser (van Walraven) Score", ["elixhauser"])
    assert _protocol_search_term(context, "elixhauser") == "elixhauser"
    assert '"elixhauser"[Title/Abstract]' in queries[0]
    assert all("van Walraven" not in query for query in queries)
    # A dictionary concept's description is dictionary text and still used.
    assert not is_export_display_label("in hospital mortality", ["death"])
    assert '"hospital mortality"[Title/Abstract]' in queries[0]


@pytest.mark.parametrize("stratified", [True, False], ids=["strata", "single_query"])
def test_the_retained_excerpt_keeps_the_sentence_naming_the_exposure_clinically(
    stratified: bool,
) -> None:
    import json

    from easyicu.research_agent.literature import PubMedLiteratureClient

    context = _derived_column_context(**{
        key: _DERIVED_COLUMNS[0][key]
        for key in ("name", "source_concept", "description", "question")
    })
    design = " ".join((
        "Adult patients were eligible.",
        "The cohort comprised admissions to 12 units.",
        "Exclusion criteria were applied at admission.",
        "Follow-up continued to hospital discharge.",
        "Inclusion required a stay of at least one day.",
        "Patients were grouped at the index time.",
    ))
    abstract = (
        design
        + " Acute kidney injury was more frequent among non-survivors."
        + " In-hospital mortality was 12%."
    )
    responses = {
        "esearch.fcgi": json.dumps({"esearchresult": {"idlist": ["111"]}}).encode(),
        "esummary.fcgi": json.dumps({"result": {"uids": ["111"], "111": {
            "uid": "111", "title": "Kidney injury in intensive care",
            "pubdate": "2020", "source": "Crit Care",
            "authors": [{"name": "Doe J"}],
        }}}).encode(),
        "efetch.fcgi": (
            "<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>111</PMID>"
            f"<Article><Abstract><AbstractText>{abstract}</AbstractText></Abstract>"
            "</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
        ).encode(),
    }
    client = PubMedLiteratureClient(timeout=1.0)
    client._http_get = lambda path, params: responses.get(path)  # type: ignore[attr-defined]

    if stratified:
        [record] = client.search_context_strata(context, retmax=5).records
    else:
        [record] = client.search_for_context(context, retmax=5)

    assert "Acute kidney injury was more frequent" in record.relevance
    assert "In-hospital mortality was 12%" in record.relevance

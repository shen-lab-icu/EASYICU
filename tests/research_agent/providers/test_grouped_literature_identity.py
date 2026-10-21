"""Clinical retrieval identities for synthetic grouped measurements, offline."""

from dataclasses import replace

import pytest

from easyicu.research_agent import literature
from easyicu.research_agent.contracts.exposure_group_rules import (
    EXPOSURE_GROUP_LABELS_KEY,
    grouping_level_description,
    grouping_level_labels,
    read_grouping_rules,
)
from easyicu.research_agent.literature_concepts import literature_concept_identity
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)


def _context(concept="map", *, grouped=True):
    name = "pressure_bands_z9" if grouped else concept
    parameters = {
        "concept": concept,
        "window": {"start_hours": 0, "end_hours": 12},
        "scale": "nominal",
        "groups": [
            {"id": "g1", "rule": {"summary": "mean", "op": "<", "value": 65}},
            {"id": "g2", "rule": "otherwise"},
        ],
        "unmeasured": "exclude",
        "source_columns": {"mean": f"{concept}_mean"},
    }
    rules = read_grouping_rules(parameters)
    labels = {"g1": "lower band", "g2": "upper band"}
    identity = literature_concept_identity(concept)
    return ResearchContext(
        research_question="Compare hospital mortality across measured pressure bands.",
        cohort=CohortDescriptor(
            cohort_name="synthetic", database="miiv", n_stays=80,
            provenance={EXPOSURE_GROUP_LABELS_KEY: [{
                "variable": name, "concept": concept, "scale": rules.scale,
                "levels": list(grouping_level_labels(rules, labels)),
                "parameters": parameters, "rule_wording_version": 2,
            }]} if grouped else {},
        ),
        variables=[
            ConceptDescriptor(
                name=name, source_concept=name, dtype="int64",
                description=(grouping_level_description(rules, labels)
                             if grouped else identity.canonical_phrase),
                unit_normalization=rules.transform_id if grouped else None,
                derived_from_concepts=[concept] if grouped else [],
            ),
            ConceptDescriptor(
                name="death", source_concept="death", dtype="int64",
                description="in-hospital mortality",
            ),
        ],
        primary_exposure=name, target_outcome="death",
    )


@pytest.mark.parametrize("concept", ["map", "creat", "sep3_sofa2"])
def test_grouping_uses_its_declared_concept_for_queries_and_screening(concept):
    context = _context(concept)
    identity = literature_concept_identity(concept)
    ungrouped = _context(concept, grouped=False)

    assert literature._protocol_search_term(context, context.primary_exposure) == (
        identity.canonical_phrase
    )
    assert literature.build_pubmed_protocol_queries_for_context(context) == (
        literature.build_pubmed_protocol_queries_for_context(ungrouped)
    )
    assert literature._variable_literature_identity(context, context.primary_exposure) == identity
    assert literature._variable_focus_terms(context, context.primary_exposure) == (
        literature._variable_focus_terms(ungrouped, ungrouped.primary_exposure)
    )


@pytest.mark.parametrize("complete", [True, False])
def test_grouped_screen_retains_all_composite_identity_requirements(complete):
    context = _context("sep3_sofa2")
    exposure = "Sepsis-3 based on SOFA-2" if complete else "SOFA-2"
    record = literature.CitationRecord(
        key="synthetic", year="2025",
        title=f"{exposure} and hospital mortality in adult ICU patients",
        relevance=("Study-design excerpt: This retrospective cohort study "
                   f"evaluated the association between {exposure} and hospital "
                   "mortality in adult intensive care patients."),
        publication_types=["Observational Study"],
    )
    decision = literature._screening_decision_for_record(
        context=context, record=record, source="pubmed", query=None,
    )
    assert decision.exposure_match is complete
    assert (decision.disposition == "include") is complete


def test_ungrouped_description_keeps_the_studys_endpoint_specificity():
    context = _context(grouped=False)
    assert literature._protocol_search_term(context, "map") == "mean arterial pressure"
    queries = literature.build_pubmed_protocol_queries_for_context(context)
    assert len(queries) == 3
    assert queries[0].startswith('"mean arterial pressure"[Title/Abstract] AND ')
    assert '"in-hospital mortality"[Title/Abstract]' in queries[0]


_UNSAFE_TERMS = [
    ("measure_band_x7", None, "internal_column_name"),
    ("measure17", None, "internal_column_name"),
    ("axis", "measure band x7", "internal_column_name"),
    ("axis", "Level codes: 1=lower; 2=upper", "group_level_codes"),
    ("axis", "eICU-CRD", "database_name"),
    ("MIMIC-IV", None, "database_name"),
    ("axis", "Synthetic Registry", "database_name"),
]


def _unsafe_context(name, description):
    context = _context(grouped=False)
    context.cohort.database = "Synthetic Registry"
    context = context.model_copy(update={"primary_exposure": name})
    context.variables[0] = ConceptDescriptor(
        name=name, source_concept=name, dtype="int64", description=description,
    )
    return context


@pytest.mark.parametrize(("name", "description", "reason"), _UNSAFE_TERMS)
def test_invalid_exposure_terms_are_refused_before_query_building(name, description, reason):
    context = _unsafe_context(name, description)
    with pytest.raises(literature.LiteratureSearchTermError) as stopped:
        literature.build_pubmed_protocol_queries_for_context(context)
    assert stopped.value.refusal.reason_code == reason
    assert stopped.value.refusal.variable == name


def test_an_unresolved_export_label_is_refused(monkeypatch):
    context = _unsafe_context("elixhauser", "Elixhauser (van Walraven) Score")
    monkeypatch.setattr(literature, "literature_concept_identity", lambda _key: None)
    with pytest.raises(literature.LiteratureSearchTermError) as stopped:
        literature.build_pubmed_protocol_queries_for_context(context)
    assert stopped.value.refusal.reason_code == "export_display_label"


def test_a_catalog_alternative_cannot_bypass_the_term_guard(monkeypatch):
    context = _context(grouped=False)
    original = literature.literature_concept_identity
    unsafe = replace(original("map"), retrieval_alternatives=(("measure_band_x7",),))
    monkeypatch.setattr(
        literature, "literature_concept_identity",
        lambda key: unsafe if key == "map" else original(key),
    )
    with pytest.raises(literature.LiteratureSearchTermError) as stopped:
        literature.build_pubmed_protocol_queries_for_context(context)
    assert stopped.value.refusal.reason_code == "internal_column_name"


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "concept", "scale", "levels", "transform"])
def test_unbound_or_inconsistent_grouping_never_falls_back_to_a_column(corruption):
    context = _context()
    records = context.cohort.provenance[EXPOSURE_GROUP_LABELS_KEY]
    if corruption == "missing":
        records.clear()
    elif corruption == "duplicate":
        records.append(dict(records[0]))
    elif corruption == "levels":
        records[0]["levels"][0]["group"] = "g2"
        records[0]["levels"][1]["group"] = "g1"
    elif corruption == "transform":
        context.variables[0] = context.variables[0].model_copy(
            update={"unit_normalization": "exposure_group_ordinal"},
        )
    else:
        records[0][corruption] = "creat" if corruption == "concept" else "ordinal"
    with pytest.raises(literature.LiteratureSearchTermError) as stopped:
        literature.build_pubmed_protocol_queries_for_context(context)
    assert stopped.value.refusal.reason_code == "grouping_record_invalid"


def test_a_grouped_concept_without_a_literature_identity_is_refused(monkeypatch):
    context = _context()
    monkeypatch.setattr(literature, "literature_concept_identity", lambda _key: None)
    with pytest.raises(literature.LiteratureSearchTermError) as stopped:
        literature.build_pubmed_protocol_queries_for_context(context)
    assert stopped.value.refusal.reason_code == "concept_identity_unavailable"


class _NoSearch:
    def search_context_strata(self, *_args, **_kwargs):
        pytest.fail("A refused term must never reach a retrieval source")

    search_for_context = search_context_strata


@pytest.mark.parametrize(("name", "description", "reason"), _UNSAFE_TERMS)
def test_provenance_records_refusal_without_claiming_an_empty_search(name, description, reason):
    context = _unsafe_context(name, description)
    bundle = literature.LiteratureAgent(
        enable_pubmed=True, pubmed_client=_NoSearch(),
        enable_tavily=True, tavily_client=_NoSearch(),
    ).run(context)
    provenance = bundle.search_provenance
    assert provenance.sources_enabled == ["pubmed", "tavily"]
    assert provenance.search_conducted is False
    assert provenance.search_queries == {}
    assert provenance.sources_returning == []
    assert provenance.searched_at is None
    assert provenance.term_refusals[0].reason_code == reason
    assert provenance.model_dump(mode="json")["term_refusals"][0]["variable"] == name
    assert reason in provenance.sources_failing["pubmed"][0]
    assert bundle.prisma is None
    assert bundle.screening_decisions == []


def test_a_database_token_cannot_become_the_no_exposure_topic():
    context = _context(grouped=False)
    context = context.model_copy(update={
        "primary_exposure": None,
        "research_question": "eICU-CRD cohort patterns and hospital mortality",
    })
    bundle = literature.LiteratureAgent(enable_pubmed=True, pubmed_client=_NoSearch()).run(context)
    assert not bundle.search_provenance.search_conducted
    assert bundle.search_provenance.term_refusals[0].reason_code == "database_name"

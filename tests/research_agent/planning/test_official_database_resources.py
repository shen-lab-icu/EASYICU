"""Official resource identifiers do not confer content or version authority."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from easyicu.research_agent.bibliographic_metadata import CitationRecord, LiteratureScreeningDecision
from easyicu.research_agent.planning.literature_contract import ReviewPolicy
from easyicu.research_agent.planning.literature_design_authority import assess_record
from easyicu.research_agent.planning.official_database_resources import match_official_source_identity


PUB_TITLE = "The Salzburg Intensive Care database (SICdb): an openly available critical care dataset"
DATA_TITLE = "Salzburg Intensive Care database (SICdb), a freely accessible intensive care database"
PUB_DOI = "10.1007/s00134-023-07046-3"
DATA_DOI = "10.13026/dm9f-dm56"


def ref(title=PUB_TITLE, pmid="37052626", doi=PUB_DOI):
    return SimpleNamespace(title=title, pmid=pmid, doi=doi)


def test_official_publication_identity_is_not_a_dataset_version_or_content_review():
    result = match_official_source_identity("sic", ref())
    assert result.kind == "source_publication" and result.dataset_version is None
    assert result.content_status == "not_verified" and result.registered_dataset_version_status == "not_assessed"


def test_version_specific_doi_is_not_local_dataset_version_proof():
    result = match_official_source_identity("sic", ref(title=DATA_TITLE, pmid=None, doi=DATA_DOI))
    assert result.kind == "dataset_release" and result.dataset_version == "1.0.6"
    assert result.registered_dataset_version_status == "not_assessed"


@pytest.mark.parametrize("item", [
    ref(pmid="41551844"),
    ref(doi="10.1007/s00134-023-07046-4"),
    ref(title="Clinical mortality study using SICdb"),
    ref(title=DATA_TITLE, pmid=None, doi="10.13026/another-release"),
])
def test_title_and_mismatched_identifiers_cannot_establish_source_identity(item):
    assert match_official_source_identity("sic", item) is None


def test_original_publication_without_abstract_is_identity_only_not_a_card():
    r = CitationRecord(key="official_sicdb", title=PUB_TITLE, pmid="37052626", doi=PUB_DOI, year="2023")
    d = LiteratureScreeningDecision(citation_key=r.key, source="pubmed", disposition="exclude",
                                    evidence_role="related_context", rationale="Abstract unavailable.")
    record, decision, card = assess_record(r, d, policy=ReviewPolicy(), question="ICU lactate and length of stay",
                                            year=datetime.now(timezone.utc).year, database="sic")
    assert record.review.source_resource_identity.kind == "source_publication"
    assert record.review.source_resource_identity.content_status == "not_verified"
    assert "data_source" not in record.review.roles and card is None
    assert decision.disposition == "exclude"


def test_known_identity_cannot_be_assigned_to_another_registered_database():
    assert match_official_source_identity("miiv", ref()) is None


def test_registered_source_version_remains_unassessed_and_forged_identity_fails():
    from easyicu.research_agent.planning.literature_source_integrity import inspect_plan_literature_sources

    r = CitationRecord(key="official_sicdb", title=PUB_TITLE, pmid="37052626", doi=PUB_DOI, year="2023",
                       relevance="Study-design excerpt: SICdb comprises adult intensive care patients.")
    d = LiteratureScreeningDecision(citation_key=r.key, source="pubmed", disposition="exclude",
                                    evidence_role="related_context", rationale="Database description candidate.")
    record, decision, card = assess_record(r, d, policy=ReviewPolicy(), question="ICU lactate and length of stay",
                                            year=datetime.now(timezone.utc).year, database="sic")
    assert card is not None
    bundle = SimpleNamespace(citations=[record], screening_decisions=[decision], design_evidence_cards=[card])
    plan = SimpleNamespace(steps=[SimpleNamespace(literature_citation_keys=[r.key])], design_selection=None)
    assert "registered_dataset_version_not_assessed" in {i.reason for i in inspect_plan_literature_sources(plan, bundle)}
    forged = record.review.source_resource_identity.model_copy(update={"dataset_version": "1.0.6"})
    bundle.citations = [record.model_copy(update={"review": record.review.model_copy(update={"source_resource_identity": forged})})]
    issues = inspect_plan_literature_sources(plan, bundle)
    assert any(i.reason == "source_resource_identity_mismatch" and i.severity == "error" for i in issues)


MIMIC_PUBLICATION = "MIMIC-IV, a freely accessible electronic health record dataset"
MIMIC_PUBLICATION_DOI = "10.1038/s41597-022-01899-x"


def test_mimic_iv_source_identity_distinguishes_paper_and_dataset_releases():
    article = ref(title=MIMIC_PUBLICATION, pmid="36596836", doi=MIMIC_PUBLICATION_DOI)
    publication = match_official_source_identity("miiv", article)
    assert publication.database == "miiv" and publication.kind == "source_publication"
    assert publication.dataset_version is None
    assert publication.content_status == "not_verified"
    assert publication.registered_dataset_version_status == "not_assessed"
    for doi, expected in (("10.13026/kpb9-mt58", "3.1"), ("10.13026/6mm1-ek67", "2.2")):
        release = match_official_source_identity("MIMIC-IV", ref(title="MIMIC-IV", pmid=None, doi=doi))
        assert release.kind == "dataset_release" and release.dataset_version == expected
        assert release.registered_dataset_version_status == "not_assessed"


def test_release_version_conflicts_and_unknown_or_cross_database_do_not_match():
    wrong_version = SimpleNamespace(title="MIMIC-IV", pmid=None, doi="10.13026/kpb9-mt58", dataset_version="2.2")
    assert match_official_source_identity("miiv", wrong_version) is None
    assert match_official_source_identity("miiv", ref(title="MIMIC-IV", pmid=None, doi="10.13026/unregistered")) is None
    assert match_official_source_identity("unknown_database", ref(title="MIMIC-IV", pmid=None, doi="10.13026/kpb9-mt58")) is None
    assert match_official_source_identity("sic", ref(title="MIMIC-IV", pmid=None, doi="10.13026/kpb9-mt58")) is None


def test_historical_registry_keeps_previous_sic_identity_without_promoting_new_entries():
    old = match_official_source_identity("sic", ref(), registry_version="easyicu.official_source_identity/1")
    current = match_official_source_identity("sic", ref())
    assert old.registry_version == "easyicu.official_source_identity/1"
    assert current.registry_version == "easyicu.official_source_identity/2"
    assert old.doi == current.doi and old.dataset_version == current.dataset_version
    assert match_official_source_identity(
        "miiv", ref(title="MIMIC-IV", pmid=None, doi="10.13026/kpb9-mt58"),
        registry_version="easyicu.official_source_identity/1",
    ) is None
    assert match_official_source_identity("sic", ref(), registry_version="easyicu.official_source_identity/999") is None


def test_mimic_resource_binding_cannot_authorize_a_sic_study():
    from easyicu.research_agent.planning.literature_source_integrity import inspect_plan_literature_sources

    r = CitationRecord(key="mimic_resource", title=MIMIC_PUBLICATION, pmid="36596836", doi=MIMIC_PUBLICATION_DOI,
                       year="2023", relevance="Study-design excerpt: MIMIC-IV contains critical care data.")
    d = LiteratureScreeningDecision(citation_key=r.key, source="pubmed", disposition="exclude",
                                    evidence_role="related_context", rationale="Resource description candidate.")
    record, decision, card = assess_record(r, d, policy=ReviewPolicy(), question="ICU lactate and length of stay",
                                            year=datetime.now(timezone.utc).year, database="miiv")
    assert card is not None and record.review.source_resource_identity.database == "miiv"
    bundle = SimpleNamespace(citations=[record], screening_decisions=[decision], design_evidence_cards=[card])
    plan = SimpleNamespace(steps=[SimpleNamespace(literature_citation_keys=[r.key])], design_selection=None)
    assert any(i.reason == "source_resource_database_mismatch" and i.severity == "error"
               for i in inspect_plan_literature_sources(plan, bundle, database="sic"))
    assert not any(i.severity == "error" for i in inspect_plan_literature_sources(plan, bundle, database="miiv"))


def test_unregistered_resource_cannot_gain_source_role_from_title_alone():
    r = CitationRecord(key="unregistered_hirid", title="HiRID: an openly available critical care dataset",
                       doi="10.13026/not-a-registered-release", year="2024",
                       relevance="Study-design excerpt: HiRID contains adult intensive care patients.")
    d = LiteratureScreeningDecision(citation_key=r.key, source="pubmed", disposition="exclude",
                                    evidence_role="related_context", rationale="Unverified source candidate.",
                                    population_match=True)
    record, decision, card = assess_record(r, d, policy=ReviewPolicy(), question="ICU lactate and length of stay",
                                            year=datetime.now(timezone.utc).year, database="hirid")
    assert record.review.source_resource_identity is None
    assert "data_source" not in record.review.roles
    assert decision.disposition == "exclude" and card is None


def test_cross_database_resource_binding_blocks_the_actual_scientific_review():
    from easyicu.research_agent.literature import LiteratureBundle
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    from tests.research_agent.planning.scientific_review_fixtures import _context
    from tests.research_agent.planning.test_literature_source_integrity import _real_plan

    source = CitationRecord(key="source_1", title=MIMIC_PUBLICATION, pmid="36596836", doi=MIMIC_PUBLICATION_DOI,
                            year="2023", relevance="Study-design excerpt: MIMIC-IV contains critical care data.")
    decision = LiteratureScreeningDecision(citation_key=source.key, source="pubmed", disposition="exclude",
                                           evidence_role="related_context", rationale="Resource candidate.")
    record, screened, card = assess_record(source, decision, policy=ReviewPolicy(),
                                            question="ICU lactate and length of stay", year=2026, database="miiv")
    literature = LiteratureBundle(research_question="ICU lactate and length of stay", citations=[record],
                                  screening_decisions=[screened], design_evidence_cards=[card])
    context = _context().model_copy(update={"cohort": _context().cohort.model_copy(update={"database": "sic"})})
    review = build_plan_scientific_review(context=context, plan=_real_plan(), literature=literature)
    assert not review.approval_allowed
    assert any(f.code == "LITERATURE_SOURCE_INTEGRITY_INVALID" and "source_resource_database_mismatch" in f.message
               for f in review.findings)

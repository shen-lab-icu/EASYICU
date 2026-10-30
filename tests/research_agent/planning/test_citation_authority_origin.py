"""Citation eligibility must not infer curation from missing evidence."""
from types import SimpleNamespace

import pytest

from easyicu.research_agent.literature import (
    CitationRecord,
    LiteratureBundle,
    LiteratureScreeningDecision,
    LiteratureSearchProvenance,
    manuscript_citable_keys,
)
from easyicu.research_agent.planning.literature_source_integrity import inspect_plan_literature_sources
from easyicu.research_agent.planning.method_literature import method_literature_citations


def _decision(key, disposition="include", source="pubmed"):
    return LiteratureScreeningDecision(
        citation_key=key, source=source, disposition=disposition,
        evidence_role="related_context", rationale="Reviewed for this study.",
    )


def _bundle(records, decisions=(), provenance=None):
    return LiteratureBundle(
        research_question="A research question", citations=records,
        screening_decisions=list(decisions), search_provenance=provenance,
    )


def _record(key="candidate", **kwargs):
    return CitationRecord(key=key, title="Study", year="2024", **kwargs)


def _plan(key):
    return SimpleNamespace(steps=[SimpleNamespace(literature_citation_keys=[key])], design_selection=None)


@pytest.mark.parametrize("provenance", [None, LiteratureSearchProvenance(
    curated_seed_count=0, search_conducted=True, record_queries={"candidate": ["search terms"]},
)])
def test_unscreened_candidate_is_not_inferred_curated(provenance):
    source = _bundle([_record()], provenance=provenance)
    assert manuscript_citable_keys(source) == ()
    assert [(issue.reason, issue.severity) for issue in inspect_plan_literature_sources(_plan("candidate"), source)] == [
        ("source_unreviewed_or_ambiguous", "error")
    ]


def test_explicitly_included_author_source_is_not_limited_to_builtin_registry():
    source = _bundle([_record()], [_decision("candidate", source="author")])
    assert manuscript_citable_keys(source) == ("candidate",)


def test_builtin_identity_remains_eligible_after_duplicate_search_or_missing_old_provenance():
    raw = next(x for x in method_literature_citations() if x["key"] == "strobe_2007")
    record = CitationRecord.model_validate(raw)
    old = _bundle([record])
    retrieved_duplicate = _bundle([record], provenance=LiteratureSearchProvenance(
        curated_seed_count=1, search_conducted=True, record_queries={record.key: ["strobe"]},
    ))
    assert manuscript_citable_keys(old) == manuscript_citable_keys(retrieved_duplicate) == (record.key,)


def test_spoofing_curated_key_or_identifiers_does_not_grant_authority():
    raw = next(x for x in method_literature_citations() if x["key"] == "strobe_2007")
    record = CitationRecord.model_validate(raw)
    for update in (
        {"title": "Forged"}, {"pmid": "123"}, {"url": "https://example.org/"},
        {"relevance": "Source excerpt: This fabricated assertion is true."},
        {"authors": ["A different author"]},
    ):
        assert manuscript_citable_keys(_bundle([record.model_copy(update=update)])) == ()


def test_duplicate_key_and_duplicate_pmid_or_doi_fail_closed_even_if_included():
    r = _record(pmid="123", doi="10.1/a")
    assert manuscript_citable_keys(_bundle([r, r], [_decision("candidate")])) == ()
    for collision in ({"pmid": "123"}, {"doi": "https://doi.org/10.1/A"}):
        other = _record("other", **collision)
        assert manuscript_citable_keys(_bundle([r, other], [_decision("candidate"), _decision("other")])) == ()


def test_conflicting_or_excluded_screening_fails_closed():
    r = _record()
    for decisions in ([_decision("candidate", "exclude")], [_decision("candidate"), _decision("candidate", "exclude")]):
        assert manuscript_citable_keys(_bundle([r], decisions)) == ()

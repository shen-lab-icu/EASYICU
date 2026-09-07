from copy import deepcopy

import pytest

from easyicu.research_agent.bibliographic_metadata import complete_missing_authors
from easyicu.research_agent.planning.method_literature import method_literature_citations


def _record():
    return {
        "key": "not_an_author_hint", "authors": [], "pmid": "17938396",
        "doi": "10.7326/0003-4819-147-8-200710160-00010", "year": "2007",
        "venue": "Annals of Internal Medicine",
        "title": "The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies.",
        "bibliographic_notices": ["Erratum: Ann Intern Med. 2008;148(2):168."],
    }


def test_metadata_completion_is_source_ordered_nonmutating_and_metadata_only():
    original = _record()
    before = deepcopy(original)
    completed, receipt = complete_missing_authors(original)
    assert completed["authors"][0] == "Erik von Elm"
    assert completed["authors"][-1] == "STROBE Initiative"
    assert len(completed["authors"]) == 7
    assert original == before and completed["bibliographic_notices"] == before["bibliographic_notices"]
    assert receipt["fields"] == ["authors"]
    assert receipt["scope"] == "bibliographic_metadata_only"
    assert receipt["source_url"].endswith("/17938396/")
    assert len(receipt["record_sha256"]) == len(receipt["snapshot_sha256"]) == 64


@pytest.mark.parametrize("field", ["pmid", "doi", "title", "year", "venue"])
def test_wrong_or_missing_identity_never_borrows_authors(field):
    for replacement in ("", "another article"):
        record = _record()
        record[field] = replacement
        completed, receipt = complete_missing_authors(record)
        assert completed == record and receipt is None


def test_existing_authors_and_other_metadata_are_not_silently_rewritten():
    record = _record()
    record["authors"] = ["Source-provided author requiring separate review"]
    assert complete_missing_authors(record) == (record, None)


def test_curated_method_citations_carry_authors_including_collective_names():
    records = {row["key"]: row for row in method_literature_citations()}
    assert records["suissa_immortal_time_2008"]["authors"] == ["Samy Suissa"]
    assert records["record_2015"]["authors"][-1] == "RECORD Working Committee"
    assert records["strobe_2007"]["bibliographic_notices"] == _record()["bibliographic_notices"]


def test_frozen_reader_completes_metadata_with_a_separate_receipt(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.literature import LiteratureBundle
    from easyicu.research_agent.reporting.manuscript_reader import build_manuscript_reader

    literature = LiteratureBundle.model_validate({"research_question": "Question", "citations": [_record()]})
    original = literature.model_dump_json()
    payload = build_manuscript_reader(manuscript="Context [@not_an_author_hint].", evidence=EvidenceStore(tmp_path), literature=literature)
    row = payload["references"][0]
    assert row["authors"][0] == "Erik von Elm"
    assert row["metadata_source"]["fields"] == ["authors"]
    assert payload["publication_authorized"] is False
    assert literature.model_dump_json() == original

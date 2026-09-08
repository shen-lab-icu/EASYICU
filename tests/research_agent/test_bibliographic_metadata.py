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


def test_mimic_dataset_authors_require_full_identity_and_preserve_corrections():
    record = {"pmid": "36596836", "doi": "10.1038/s41597-022-01899-x",
              "title": "MIMIC-IV, a freely accessible electronic health record dataset.",
              "venue": "Scientific Data", "year": "2023", "authors": [],
              "bibliographic_notices": ["Author correction: 10.1038/s41597-023-02136-9."]}
    original = deepcopy(record)
    completed, receipt = complete_missing_authors(record)
    assert len(completed["authors"]) == 13
    assert completed["authors"][0] == "Alistair E W Johnson"
    assert completed["authors"][-1] == "Roger G Mark"
    assert completed["bibliographic_notices"] == record["bibliographic_notices"]
    assert receipt["source_url"] == "https://pubmed.ncbi.nlm.nih.gov/36596836/"
    assert original == record
    for key in ("pmid", "doi", "title", "venue", "year"):
        conflict = {**record, key: "different"}
        assert complete_missing_authors(conflict) == (conflict, None)


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


def test_entire_curated_method_pack_has_source_verified_author_coverage():
    for record in method_literature_citations():
        assert record.get("authors"), record["key"]
        frozen = {**record, "authors": []}
        completed, receipt = complete_missing_authors(frozen)
        assert completed["authors"] == record["authors"]
        assert receipt["scope"] == "bibliographic_metadata_only"
        for field in ("pmid", "doi", "title", "year", "venue"):
            changed = {**frozen, field: "conflicting identity"}
            assert complete_missing_authors(changed) == (changed, None)


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


def test_cached_web_reader_refresh_is_metadata_only_and_does_not_mutate_history():
    from easyicu.webserver.agent_runs import _public_review_payloads

    reader = {
        "schema_version": "easyicu.manuscript-provenance/1",
        "references": [{"number": 1, **_record()}],
        "manuscript_sha256": "frozen", "claims": [{"value": 42}],
        "publication_authorized": False,
    }
    payloads = {"manuscript_provenance.json": reader, "manuscript_draft.json": {"reader": reader}}
    before = deepcopy(payloads)
    result = _public_review_payloads(payloads)
    assert payloads == before
    for projected in (result["manuscript_provenance.json"], result["manuscript_draft.json"]["reader"]):
        assert projected["references"][0]["authors"][0] == "Erik von Elm"
        assert projected["references"][0]["number"] == 1
        assert {k: v for k, v in projected.items() if k != "references"} == {
            k: v for k, v in reader.items() if k != "references"
        }


@pytest.mark.parametrize("references", [None, "malformed", [{"doi": "unverified", "authors": []}]])
def test_cached_reader_does_not_invent_authors_for_unverifiable_input(references):
    from easyicu.research_agent.reporting.manuscript_reader import refresh_reader_bibliography

    payload = {"schema_version": "easyicu.manuscript-provenance/1", "references": references}
    assert refresh_reader_bibliography(payload) == payload
    assert refresh_reader_bibliography(None) is None

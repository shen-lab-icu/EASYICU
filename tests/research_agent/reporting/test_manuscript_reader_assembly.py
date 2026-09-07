"""A reader assembles only the manuscript's registered, citable sources."""

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.literature import LiteratureBundle
from easyicu.research_agent.reporting.manuscript_provenance import ManuscriptProvenanceError
from easyicu.research_agent.reporting.manuscript_reader import build_manuscript_reader


def _literature():
    return LiteratureBundle.model_validate({
        "research_question": "Describe the cohort",
        "citations": [
            {"key": "later", "title": "Second source", "year": "2020", "authors": ["B Author"]},
            {"key": "first", "title": "Original & corrected title", "year": "2019",
             "authors": ["A Author"], "doi": "10.1234/example",
             "bibliographic_notices": ["Correction to the author list."]},
            {"key": "unused", "title": "Not cited", "year": "2021"},
        ],
    })


def test_reference_order_comes_from_prose_and_preserves_source_metadata(tmp_path):
    evidence = EvidenceStore(tmp_path)
    text = "# Draft\n\n## Introduction\n\nContext [@first; @later]. Again [@first]."
    result = build_manuscript_reader(manuscript=text, evidence=evidence, literature=_literature())
    assert [row["key"] for row in result["references"]] == ["first", "later"]
    assert result["references"][0]["number"] == 1
    assert result["references"][0]["authors"] == ["A Author"]
    assert result["references"][0]["bibliographic_notices"] == ["Correction to the author list."]
    assert result["references"][1]["authors"] == ["B Author"]
    assert result["publication_authorized"] is False
    assert result["claim_count"] == 0


@pytest.mark.parametrize("mutation", ["unknown", "excluded", "duplicate"])
def test_reader_cannot_present_missing_or_ambiguous_citations_as_verified(tmp_path, mutation):
    literature = _literature()
    if mutation == "unknown":
        literature = literature.model_copy(update={"citations": literature.citations[1:]})
    elif mutation == "excluded":
        from easyicu.research_agent.literature import LiteratureScreeningDecision
        literature = literature.model_copy(update={"screening_decisions": [
            LiteratureScreeningDecision(citation_key="later", source="pubmed", disposition="exclude", evidence_role="related_context", rationale="Wrong population")
        ]})
    else:
        literature = literature.model_copy(update={"citations": [*literature.citations, literature.citations[0]]})
    with pytest.raises(ManuscriptProvenanceError):
        build_manuscript_reader(manuscript="Context [@later].", evidence=EvidenceStore(tmp_path), literature=literature)


def test_reader_uses_existing_table_owner_without_changing_source(tmp_path):
    from tests.research_agent.reporting.test_manuscript_tables import _source

    plan, record, _ = _source(tmp_path)
    # The existing exact-plan table contract already covers the table's numbers.
    from types import SimpleNamespace
    evidence = SimpleNamespace(root=tmp_path, numeric_claims=lambda: [], records=lambda: [record])
    before = (tmp_path / record.relative_path).read_bytes()
    result = build_manuscript_reader(
        manuscript="# Draft\n\n## Results\n\nSee Table 1.", evidence=evidence,
        plan=plan, evidence_records=[record],
    )
    assert len(result["tables"]) == 1
    assert result["tables"][0]["label"] == "Table 1"
    assert result["tables"][0]["rows"][1][3] == "2.00 [1.50, 2.50]"
    assert record.sha256 in " ".join(result["tables"][0]["notes"])
    assert (tmp_path / record.relative_path).read_bytes() == before

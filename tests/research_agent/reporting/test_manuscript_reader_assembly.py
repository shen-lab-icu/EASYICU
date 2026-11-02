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
            {"key": "later", "pmid": "36596836", "doi": "10.1038/s41597-022-01899-x",
             "title": "MIMIC-IV, a freely accessible electronic health record dataset.",
             "venue": "Scientific Data", "year": "2023", "authors": [
                 "Alistair E W Johnson", "Lucas Bulgarelli", "Lu Shen", "Alvin Gayles", "Ayad Shammout",
                 "Steven Horng", "Tom J Pollard", "Sicheng Hao", "Benjamin Moody", "Brian Gow",
                 "Li-Wei H Lehman", "Leo A Celi", "Roger G Mark"]},
            {"key": "first", "pmid": "17938396", "doi": "10.7326/0003-4819-147-8-200710160-00010",
             "title": "The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies.",
             "venue": "Annals of Internal Medicine", "year": "2007", "authors": [
                 "Erik von Elm", "Douglas G Altman", "Matthias Egger", "Stuart J Pocock",
                 "Peter C Gøtzsche", "Jan P Vandenbroucke", "STROBE Initiative"],
             "bibliographic_notices": ["Erratum: Ann Intern Med. 2008;148(2):168."]},
            {"key": "unused", "title": "Synthetic unselected record", "year": "2021"},
        ],
        "screening_decisions": [
            {"citation_key": key, "source": "synthetic_test", "disposition": "include",
             "evidence_role": "related_context", "rationale": "Identified fixture record included"}
            for key in ("first", "later")
        ],
    })


def test_reference_order_comes_from_prose_and_preserves_source_metadata(tmp_path):
    evidence = EvidenceStore(tmp_path)
    text = "# Draft\n\n## Introduction\n\nContext [@first; @later]. Again [@first]."
    result = build_manuscript_reader(manuscript=text, evidence=evidence, literature=_literature())
    assert [row["key"] for row in result["references"]] == ["first", "later"]
    assert result["references"][0]["number"] == 1
    assert result["references"][0]["authors"] == _literature().citations[1].authors
    assert result["references"][0]["bibliographic_notices"] == ["Erratum: Ann Intern Med. 2008;148(2):168."]
    assert result["references"][1]["authors"] == _literature().citations[0].authors
    assert result["publication_authorized"] is False
    assert result["claim_count"] == 0


@pytest.mark.parametrize("mutation", ["unknown", "excluded", "duplicate", "unreviewed", "conflicting"])
def test_reader_cannot_present_missing_or_ambiguous_citations_as_verified(tmp_path, mutation):
    literature = _literature()
    if mutation == "unknown":
        literature = literature.model_copy(update={"citations": literature.citations[1:]})
    elif mutation == "excluded":
        from easyicu.research_agent.literature import LiteratureScreeningDecision
        literature = literature.model_copy(update={"screening_decisions": [
            LiteratureScreeningDecision(citation_key="later", source="pubmed", disposition="exclude", evidence_role="related_context", rationale="Wrong population")
        ]})
    elif mutation == "unreviewed":
        literature = literature.model_copy(update={"screening_decisions": []})
    elif mutation == "conflicting":
        from easyicu.research_agent.literature import LiteratureScreeningDecision
        literature = literature.model_copy(update={"screening_decisions": [*literature.screening_decisions,
            LiteratureScreeningDecision(citation_key="later", source="synthetic_test", disposition="exclude",
                evidence_role="related_context", rationale="Conflicting fixture decision")],
        })
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
    assert result["tables"][0]["rows"][1][2] == "2.00 [1.50, 2.50]"
    assert record.sha256 in " ".join(result["tables"][0]["notes"])
    assert (tmp_path / record.relative_path).read_bytes() == before


def test_invalid_figure_contract_blocks_reader_through_provenance_error(tmp_path):
    from types import SimpleNamespace
    from .test_manuscript_figures import _bundle

    records = _bundle(tmp_path, caption="Invalid\ncaption")
    evidence = SimpleNamespace(root=tmp_path, numeric_claims=lambda: [], records=lambda: records)
    with pytest.raises(ManuscriptProvenanceError, match="invalid"):
        build_manuscript_reader(manuscript="# Draft", evidence=evidence)

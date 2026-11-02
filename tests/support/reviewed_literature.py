"""Source-identical method references for synthetic planning contracts."""

from easyicu.research_agent.literature import CitationRecord, LiteratureBundle
from easyicu.research_agent.planning.method_literature import method_literature_citations


def curated_method_bundle(question: str) -> LiteratureBundle:
    """Provide actual built-in method records, without inventing source evidence cards."""
    keys = {"strobe_2007", "record_2015"}
    records = [CitationRecord.model_validate(value) for value in method_literature_citations()
               if value["key"] in keys]
    assert {record.key for record in records} == keys
    return LiteratureBundle(research_question=question, citations=records)

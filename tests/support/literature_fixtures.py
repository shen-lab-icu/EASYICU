"""Invented source records whose retained excerpts can be verified in tests."""
from datetime import datetime, timezone
from hashlib import sha256

from easyicu.research_agent.bibliographic_metadata import CitationRecord, LiteratureScreeningDecision
from easyicu.research_agent.literature import LiteratureBundle
from easyicu.research_agent.planning.literature_design_authority import (
    LiteratureDesignEvidence, LiteratureDesignEvidenceCard,
)


def synthetic_reviewed_literature(question: str, keys=("strobe_2007", "record_2015")):
    """Return invented records, exact source excerpts, cards and screening decisions."""
    records, cards, decisions = [], [], []
    for key in keys:
        excerpt = f"Synthetic test source {key} describes an observational study."
        records.append(CitationRecord(key=key, title=f"Synthetic fixture {key}", year="2020",
                                      relevance="Study-design excerpt: " + excerpt))
        cards.append(LiteratureDesignEvidenceCard(
            citation_key=key, evidence_role="method", access_mode="abstract",
            abstract_locator=f"https://example.invalid/synthetic/{key}",
            abstract_sha256=sha256(excerpt.encode()).hexdigest(), supplement_status="unknown",
            reviewed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            evidence=[LiteratureDesignEvidence(dimension="study_population", review_role="analysis_method",
                                               source_backed_summary=excerpt, locator="Abstract")],
        ))
        decisions.append(LiteratureScreeningDecision(citation_key=key, source="synthetic", disposition="include",
            evidence_role="method", rationale="Synthetic test fixture, selected for integrity checks."))
    return LiteratureBundle(research_question=question, citations=records, design_evidence_cards=cards,
                            screening_decisions=decisions)

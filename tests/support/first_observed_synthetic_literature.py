"""Invented, offline-only comparator for first-observed review tests.

All records and source text here are synthetic. Never cite this fixture in a
real study or treat its seven design statements as external evidence.
"""
from datetime import datetime, timezone
from hashlib import sha256
from easyicu.research_agent.literature import CitationRecord, LiteratureBundle, LiteratureScreeningDecision
from easyicu.research_agent.planning.literature_design_authority import (
    LITERATURE_DESIGN_DIMENSIONS, LiteratureDesignEvidence, LiteratureDesignEvidenceCard,
)

KEY = 'synthetic_comparator_2025'
FACTS = {
    'study_population': 'Synthetic study describes adult patients in a single observed hospitalization.',
    'time_zero_and_windows': 'Synthetic study uses ICU admission as the initial time origin.',
    'variable_operationalization': 'Synthetic study records sex and hospital outcome using declared categories.',
    'missingness_and_censoring': 'Synthetic study describes unknown outcomes separately from observed outcomes.',
    'primary_model_and_sensitivities': 'Synthetic study reports grouped proportions with uncertainty intervals.',
    'table_and_figure_completeness': 'Synthetic study displays group denominators and estimates in a table.',
    'conclusion_boundaries': 'Synthetic study limits interpretation to its observed sample.',
}
TEXT = '\n'.join(FACTS.values()).encode('utf-8')


def synthetic_first_observed_literature(question, output_dir):
    """Write invented full text and return its typed comparator bundle."""
    path = output_dir / 'synthetic_comparator.txt'
    path.write_bytes(TEXT)
    card = LiteratureDesignEvidenceCard(
        citation_key=KEY, evidence_role='direct_comparator', access_mode='user_supplied_fulltext',
        full_text_locator=str(path), full_text_sha256=sha256(TEXT).hexdigest(),
        supplement_status='not_published', reviewed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        evidence=[LiteratureDesignEvidence(dimension=d, source_backed_summary=FACTS[d],
                                           locator=f'synthetic:{d}') for d in LITERATURE_DESIGN_DIMENSIONS],
    )
    return LiteratureBundle(
        research_question=question,
        citations=[CitationRecord(key=KEY, title='Synthetic ICU cohort fixture', year='2025',
                                  venue='Synthetic test journal')],
        screening_decisions=[LiteratureScreeningDecision(
            citation_key=KEY, source='user_supplied', disposition='include', evidence_role='direct_comparator',
            rationale='Invented comparator for testing the review contract only.',
            population_match=True, outcome_match=True, design_excerpt_available=True)],
        design_evidence_cards=[card],
    )

"""Broad multilingual retrieval does not grant endpoint equivalence."""
import json

import pytest

from easyicu.research_agent.literature import (
    CitationRecord,
    _candidate_outcome_background_clause,
    _screening_decision_for_record,
    build_review_query_families,
)
from tests.research_agent.providers.test_candidate_outcome_literature import _context


def _candidate(label: str):
    context = _context()
    data = json.loads(context.user_preferences.data_constraints)
    data['candidate_outcome_for_literature']['label'] = label
    context.user_preferences.data_constraints = json.dumps(data, ensure_ascii=False)
    return context


@pytest.mark.parametrize('label', [
    '从首次转入 ICU 或中间监护至病例关闭的时长',
    '病例关闭时长',
    'duration of hospital stay',
])
def test_candidate_stay_duration_gets_broad_method_search_only(label):
    context = _candidate(label)
    clause = _candidate_outcome_background_clause(context)
    assert '"length of stay"[Title/Abstract]' in clause
    families = build_review_query_families(context)
    assert any(clause in query for query in families['analysis_method'])
    assert not any(clause in query for query in families['topic_gap'])
    assert context.target_outcome is None and context.endpoint is None


def test_retrieval_does_not_promote_different_stay_to_direct_evidence():
    context = _candidate('从首次转入 ICU 或中间监护至病例关闭的时长')
    record = CitationRecord(
        key='different_stay', year='2024',
        title='Lactate and ICU length of stay in critically ill adults',
        relevance='Source excerpt: Adult ICU patients had lactate measured and ICU length of stay assessed.',
        publication_types=['Observational Study'],
    )
    query = next(q for q in build_review_query_families(context)['analysis_method'] if '"length of stay"' in q)
    decision = _screening_decision_for_record(context=context, record=record, source='pubmed', query=query)
    assert decision.outcome_match is False
    assert decision.disposition == 'exclude'
    assert decision.evidence_role != 'direct_comparator'


@pytest.mark.parametrize('label', ['', '乳酸', '病例时长待定', '不是住院时长', 'ICU 住院时长？', '是否为住院时长', 'duration of stay or mortality'])
def test_ambiguous_or_unrelated_labels_do_not_create_proxy_query(label):
    assert _candidate_outcome_background_clause(_candidate(label)) == ''


def test_executable_endpoint_does_not_get_unbound_proxy_query():
    context = _candidate('病例关闭时长')
    context = context.model_copy(update={'target_outcome': 'los_icu'})
    assert _candidate_outcome_background_clause(context) == ''

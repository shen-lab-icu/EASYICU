import pytest
from easyicu.webserver.pi_copilot import tools
from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext


@pytest.mark.parametrize(('source_type', 'excerpt', 'parameter_excerpt', 'expected'), [
    ('url', '', '', ''),
    ('url', '', 'model invented', ''),
    ('url', 'A bounded abstract', 'model invented', 'A bounded abstract'),
    ('pdf', 'PDF excerpt', 'model invented', 'PDF excerpt'),
    ('manual', '', '', 'Research question'),
])
def test_mining_does_not_invent_an_article_excerpt(monkeypatch, source_type, excerpt, parameter_excerpt, expected):
    captured = {}
    def mine(body):
        captured.update(body)
        return {'run_id': 'test', 'idea_ledger': [], 'pre_experiment': {}}
    monkeypatch.setattr(tools.idea_mining, 'mine_ideas', mine)
    context = ToolExecutionContext(
        data_handling='zero_retention_declared',
        session=PiSessionRecord(session_id='pi-source-test'), allowed_actions={'idea'},
        idea_source={'source_type': source_type, 'title': 'Article', 'excerpt': excerpt,
                     'url': 'https://pubmed.ncbi.nlm.nih.gov/26903338/'},
    )
    tools.execute_tool('easyicu_mine_ideas', {'topic': 'Research question', 'excerpt': parameter_excerpt}, context)
    assert captured['excerpt'] == expected
    if source_type == 'url':
        source = tools.idea_mining._source_record(captured)
        assert (source['evidence_quote'] or '') == expected


def test_url_fields_do_not_fallback_to_model_metadata_or_question(monkeypatch):
    captured = {}
    def mine(body):
        captured.update(body)
        return {'run_id': 'test', 'idea_ledger': [], 'pre_experiment': {}}
    monkeypatch.setattr(tools.idea_mining, 'mine_ideas', mine)
    context = ToolExecutionContext(
        data_handling='zero_retention_declared',
        session=PiSessionRecord(session_id='pi-source-identity'), allowed_actions={'idea'},
        idea_source={'source_type': 'url', 'url': 'https://pubmed.ncbi.nlm.nih.gov/26903338/'},
    )
    tools.execute_tool('easyicu_mine_ideas', {
        'topic': 'Research question', 'title': 'Invented title', 'journal': 'Invented journal',
        'year': 2024, 'doi': '10.1234/invented', 'pmid': '1',
    }, context)
    assert not captured['title']
    for key in ('journal', 'year', 'doi', 'pmid'):
        assert not captured[key]
    source = tools.idea_mining._source_record(captured)
    assert source['title'] == 'Untitled source'

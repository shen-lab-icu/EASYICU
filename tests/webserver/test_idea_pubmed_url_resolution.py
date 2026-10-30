from unittest.mock import Mock
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import settings
from easyicu.webserver.app import app
from easyicu.webserver.ideas import mining


@pytest.fixture
def resolver(monkeypatch):
    monkeypatch.setattr(settings, 'load_settings', lambda: {'connector_pubmed_enabled': True})
    # Reproduce a local DNS environment that cannot be used for arbitrary URLs.
    monkeypatch.setattr(mining, '_resolve_public_http_target', Mock(side_effect=mining.UnsafeURL('non-public DNS')))
    records = Mock(return_value=[{
        'pmid': '26903338', 'title': 'Sepsis definitions', 'journal': 'JAMA',
        'year': 2016, 'doi': '10.1001/jama.2016.0287',
        'abstract_excerpt': 'A bounded abstract.',
    }])
    monkeypatch.setattr(mining, '_pubmed_article_records', records)
    return TestClient(app), records


def test_pubmed_url_uses_fixed_official_metadata_endpoint(resolver):
    client, records = resolver
    response = client.post('/api/ideas/resolve-source', json={
        'source_type': 'url', 'url': 'https://pubmed.ncbi.nlm.nih.gov/26903338/',
        'allow_network': True,
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload['source_adapter']['status'] == 'metadata_fetched'
    assert payload['source_adapter']['metadata_source'] == 'pubmed'
    assert payload['suggested_payload']['title'] == 'Sepsis definitions'
    assert payload['suggested_payload']['pmid'] == '26903338'
    assert payload['suggested_payload']['doi'] == '10.1001/jama.2016.0287'
    assert payload['privacy']['external_llm_calls'] == 0
    records.assert_called_once_with(['26903338'])


@pytest.mark.parametrize('url', [
    'http://pubmed.ncbi.nlm.nih.gov/26903338/',
    'https://pubmed.ncbi.nlm.nih.gov:8443/26903338/',
    'https://user@pubmed.ncbi.nlm.nih.gov/26903338/',
    'https://pubmed.ncbi.nlm.nih.gov.example.org/26903338/',
    'https://pubmed.ncbi.nlm.nih.gov/26903338/other/',
    'https://pubmed.ncbi.nlm.nih.gov/%32%36%39%30%33%33%33%38/',
])
def test_noncanonical_url_does_not_use_fixed_pubmed_adapter(resolver, url):
    client, records = resolver
    response = client.post('/api/ideas/resolve-source', json={
        'source_type': 'url', 'url': url, 'allow_network': True,
    })
    assert response.json()['source_adapter']['status'] == 'unsafe_url'
    records.assert_not_called()


def test_network_opt_in_and_connector_still_required(resolver, monkeypatch):
    client, records = resolver
    body = {'source_type': 'url', 'url': 'https://pubmed.ncbi.nlm.nih.gov/26903338/'}
    assert client.post('/api/ideas/resolve-source', json=body).json()['source_adapter']['status'] == 'blocked_network_opt_in_required'
    monkeypatch.setattr(settings, 'load_settings', lambda: {'connector_pubmed_enabled': False})
    assert client.post('/api/ideas/resolve-source', json={**body, 'allow_network': True}).json()['source_adapter']['status'] == 'blocked_network_opt_in_required'
    records.assert_not_called()


def test_pubmed_result_must_match_requested_id(resolver):
    client, records = resolver
    records.return_value = [{'pmid': '123', 'title': 'Another article'}]
    response = client.post('/api/ideas/resolve-source', json={
        'source_type': 'url', 'url': 'https://pubmed.ncbi.nlm.nih.gov/26903338/',
        'allow_network': True,
    })
    assert response.json()['source_adapter']['status'] != 'metadata_fetched'
    assert response.json()['suggested_payload']['title'] is None


def test_conflicting_user_supplied_pmid_blocks_before_network(resolver):
    client, records = resolver
    response = client.post('/api/ideas/resolve-source', json={
        'source_type': 'url', 'url': 'https://pubmed.ncbi.nlm.nih.gov/26903338/',
        'pmid': '123', 'allow_network': True,
    })
    assert response.json()['source_adapter']['status'] == 'metadata_identity_mismatch'
    assert response.json()['source_adapter']['network_calls'] == 0
    records.assert_not_called()


@pytest.mark.parametrize(('doi', 'returned_doi', 'expected'), [
    ('10.1001/jama.2016.0287', '10.1001/JAMA.2016.0287', 'metadata_fetched'),
    ('10.1001/jama.2016.0288', '10.1001/jama.2016.0287', 'metadata_identity_mismatch'),
    ('10.1001/jama.2016.0287', None, 'metadata_identity_unverified'),
    ('10.1001/jama.2016.0287', 'not-a-doi', 'metadata_identity_unverified'),
    ('bad doi', '10.1001/jama.2016.0287', 'metadata_identity_unverified'),
    ('10.1234/α+$@X', '10.1234/α+$@x', 'metadata_fetched'),
    ('', None, 'metadata_fetched'),
])
def test_pubmed_doi_identity_is_not_silently_mixed(resolver, doi, returned_doi, expected):
    client, records = resolver
    records.return_value[0]['doi'] = returned_doi
    response = client.post('/api/ideas/resolve-source', json={
        'source_type': 'url', 'url': 'https://pubmed.ncbi.nlm.nih.gov/26903338/',
        'pmid': '26903338', 'doi': doi, 'allow_network': True,
    })
    payload = response.json()
    assert payload['source_adapter']['status'] == expected
    if expected != 'metadata_fetched':
        assert payload['suggested_payload']['title'] is None
    if doi == 'bad doi':
        records.assert_not_called()
    else:
        records.assert_called_once_with(['26903338'])


def test_question_is_not_mistaken_for_the_article_excerpt(resolver):
    client, _ = resolver
    topic = 'Find open questions from this article https://pubmed.ncbi.nlm.nih.gov/26903338/'
    payload = client.post('/api/ideas/resolve-source', json={
        'source_type': 'url', 'url': 'https://pubmed.ncbi.nlm.nih.gov/26903338/',
        'topic': topic, 'title': 'My topic title', 'allow_network': True,
    }).json()
    suggested, source = payload['suggested_payload'], payload['resolved_source']
    assert suggested['topic'] == topic
    assert suggested['title'] == source['title'] == 'Sepsis definitions'
    assert suggested['user_supplied_title'] == 'My topic title'
    assert suggested['excerpt'] == source['evidence_quote'] == 'A bounded abstract.'
    assert suggested['excerpt_origin'] == source['source_text_basis'] == 'pubmed_abstract_excerpt'
    assert source['source_text_sha256'] == sha256(b'A bounded abstract.').hexdigest()
    identity = mining._source_record({
        key: source[key] for key in ('source_type', 'title', 'journal', 'year', 'doi', 'pmid', 'url')
    })
    assert source['source_id'] == identity['source_id']
    assert source['citation_key'] == suggested['citation_key'] == identity['citation_key']
    assert payload['privacy']['full_text_stored'] is False


def test_missing_abstract_stays_unknown_not_question(resolver):
    client, records = resolver
    records.return_value[0]['abstract_excerpt'] = ''
    payload = client.post('/api/ideas/resolve-source', json={
        'source_type': 'url', 'url': 'https://pubmed.ncbi.nlm.nih.gov/26903338/',
        'topic': 'Find gaps in this study', 'allow_network': True,
    }).json()
    assert payload['source_adapter']['status'] == 'metadata_fetched'
    assert not payload['suggested_payload']['excerpt']
    assert not payload['resolved_source']['evidence_quote']
    assert payload['resolved_source']['source_text_sha256'] is None


def test_explicit_user_quote_remains_unverified(resolver):
    client, _ = resolver
    payload = client.post('/api/ideas/resolve-source', json={
        'source_type': 'url', 'url': 'https://pubmed.ncbi.nlm.nih.gov/26903338/',
        'topic': 'Find gaps', 'source_quote': 'My supplied quote', 'allow_network': True,
    }).json()
    assert payload['suggested_payload']['excerpt'] == 'My supplied quote'
    assert payload['suggested_payload']['excerpt_origin'] == 'user_supplied_unverified'
    assert payload['resolved_source']['source_provenance_status'] == 'user_supplied_unverified'
    assert payload['resolved_source']['source_text_sha256'] == sha256(b'My supplied quote').hexdigest()


def test_pubmed_metadata_does_not_silently_inherit_user_journal_or_year(resolver):
    client, records = resolver
    body = {
        'source_type': 'url', 'url': 'https://pubmed.ncbi.nlm.nih.gov/26903338/',
        'journal': 'User journal', 'year': 2024, 'allow_network': True,
    }
    payload = client.post('/api/ideas/resolve-source', json=body).json()
    assert payload['suggested_payload']['journal'] == 'JAMA'
    assert payload['suggested_payload']['year'] == 2016
    assert payload['resolved_source']['user_supplied_journal'] == 'User journal'
    assert payload['resolved_source']['user_supplied_year'] == 2024
    records.return_value[0]['journal'] = None
    records.return_value[0]['year'] = None
    payload = client.post('/api/ideas/resolve-source', json=body).json()
    assert payload['suggested_payload']['journal'] is None
    assert payload['suggested_payload']['year'] is None
    assert payload['resolved_source']['user_supplied_journal'] == 'User journal'

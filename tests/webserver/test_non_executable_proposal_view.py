"""Host scopes proposal display to the matching project, session, run and config."""
from types import SimpleNamespace
import hashlib
import json
import pytest
from fastapi.testclient import TestClient
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.literature import LiteratureBundle
from easyicu.webserver import non_executable_proposal_view as view


class Project:
    def resolve(self, project):
        return 'study1' if project == 'project1' else None


class Replay:
    def snapshot(self, **kwargs):
        return {'child_jobs': [{'run_id': 'run1'}]}
    def question_wording_context(self, **kwargs):
        return {'user_turns': [{'text': 'original <script>', 'message_origin': 'human'}]}


class Service:
    project_store = Project()
    replay_store = Replay()
    def list_sessions(self, **kwargs):
        return {'sessions': [{'session_id': 'session1'}]}


def setup(monkeypatch, tmp_path):
    EvidenceStore(tmp_path)
    study = {'id': 'study1', 'question': 'question', 'outcome': 'outcome'}
    row = {'run_id': 'run1', 'study_id': 'study1', 'project_dir': str(tmp_path),
           'scientific_configuration_sha256': 'bound'}
    monkeypatch.setattr(view.study_contexts, 'get_context', lambda study_id: study)
    monkeypatch.setattr(view.study_contexts, 'scientific_configuration_sha256', lambda current: 'bound')
    monkeypatch.setattr(view, 'research_pipeline_project_root', lambda study_id: tmp_path)
    monkeypatch.setattr(view, 'list_bound_run_history', lambda **kw: [row])
    monkeypatch.setattr(view.run_step_trace, 'pipeline_run_dir', lambda wrapper, run: tmp_path)
    monkeypatch.setattr(view.run_step_trace, 'failed_literature_preview_binding', lambda *a, **kw: None)
    return study, row


def test_host_scopes_and_refuses_stale_or_cross_session(monkeypatch, tmp_path):
    study, row = setup(monkeypatch, tmp_path)
    kw = dict(service=Service(), project_id='project1', session_id='session1', run_id='run1')
    result = view.project_proposal(**kw)
    assert result['proposal']['original_question'] == 'original <script>'
    assert result['proposal']['data_access_authorized'] is False
    assert result['proposal']['decisions'][0]['unbound_evidence'] == ['source', 'literature']
    assert result['literature_candidates'] is None
    with pytest.raises(ValueError, match='session_not_bound'):
        view.project_proposal(**{**kw, 'session_id': 'other'})
    with pytest.raises(ValueError, match='run_not_in_session'):
        view.project_proposal(**{**kw, 'run_id': 'other'})
    row['scientific_configuration_sha256'] = 'old'
    with pytest.raises(ValueError, match='configuration_mismatch'):
        view.project_proposal(**kw)


def test_html_escapes_and_has_no_approval_action(monkeypatch):
    from easyicu.webserver.routes import non_executable_proposal as route
    monkeypatch.setattr(route, 'get_pi_copilot_service', lambda: object())
    monkeypatch.setattr(route, 'project_proposal', lambda **kw: {'proposal': {
        'original_question': '<script>', 'decisions': [{'decision_id':'q', 'text':'<b>outcome</b>'}]},
        'literature_candidates': None})
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(route.router)
    response = TestClient(app).get('/api/pi/proposal/project/session/run')
    assert response.status_code == 200
    assert '&lt;script&gt;' in response.text and '<script>' not in response.text
    assert '未授权提取患者数据' in response.text
    assert '提出候选绑定' in response.text
    assert '批准研究方案' not in response.text
    assert 'href="/?pi_project=project&amp;pi_session=session#guided"' in response.text


def test_return_link_percent_encodes_untrusted_identifiers(monkeypatch):
    from easyicu.webserver.routes import non_executable_proposal as route
    monkeypatch.setattr(route, 'get_pi_copilot_service', lambda: object())
    monkeypatch.setattr(route, 'project_proposal', lambda **kw: {'proposal': {
        'original_question': '', 'decisions': []}, 'literature_candidates': None})
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(route.router)
    text = TestClient(app).get('/api/pi/proposal/a%22%20onclick%3D%22x/session%3F%26bad/run').text
    assert 'href="/?pi_project=a%22+onclick%3D%22x&amp;pi_session=session%3F%26bad#guided"' in text
    assert ' onclick="' not in text


def test_literature_projection_keeps_actual_screening_roles_and_outcome_gaps():
    from easyicu.research_agent.literature import LiteratureBundle

    bundle = LiteratureBundle.model_validate({
        'research_question': 'Research question',
        'citations': [
            {'key': 'method1', 'title': 'Method study', 'year': '2019',
             'pmid': '12345', 'doi': '10.1234/example'},
            {'key': 'excluded1', 'title': 'Excluded study', 'year': '2020'},
        ],
        'screening_decisions': [
            {'citation_key': 'method1', 'source': 'PubMed', 'disposition': 'include',
             'evidence_role': 'method', 'rationale': 'Method only', 'outcome_match': False},
            {'citation_key': 'excluded1', 'source': 'PubMed', 'disposition': 'exclude',
             'evidence_role': 'related_context', 'rationale': 'Different outcome'},
        ],
    })
    projected = view._literature_projection(bundle, evidence_id='preplan_literature_bundle', sha256='a' * 64)
    assert projected['screened'] == 2
    assert projected['included'] == 1
    assert projected['excluded'] == 1
    assert projected['included_direct_comparators'] == 0
    assert projected['decision_applicability_verified'] is False
    assert [row['key'] for row in projected['records']] == ['method1']
    assert projected['records'][0]['evidence_role'] == 'method'
    assert projected['records'][0]['outcome_match'] is False


def test_literature_html_escapes_external_data_and_does_not_infer_bindings():
    from easyicu.webserver.routes.non_executable_proposal import _literature_html

    candidates = {
        'retained_citation_keys': ['safe'], 'screened': 1, 'included': 1,
        'excluded': 0, 'included_direct_comparators': 0, 'records_total': 1,
        'records': [{'key': 'safe', 'title': '<script>bad</script>', 'year': '2020',
                     'screening': 'include', 'evidence_role': 'method', 'outcome_match': False,
                     'card_access_mode': 'abstract', 'screening_rationale': '<b>reason</b>',
                     'pmid': '12345', 'doi': '10.1234/example" onclick="bad'}],
    }
    html = _literature_html(candidates)
    assert '&lt;script&gt;bad&lt;/script&gt;' in html and '<script>' not in html
    assert '&lt;b&gt;reason&lt;/b&gt;' in html
    assert '筛选记录：已纳入' in html and '角色：方法背景' in html
    assert '结局匹配：否' in html and '证据卡范围：摘要' in html
    assert '纳入的直接对照候选 0 条' in html
    assert '文献清单本身不代表支持具体决定' in html
    assert 'https://pubmed.ncbi.nlm.nih.gov/12345/' in html
    assert ' onclick="bad' not in html and 'href="https://doi.org/' not in html


def test_registered_explicit_binding_projects_only_the_actual_verbatim_excerpt(monkeypatch, tmp_path):
    study, _ = setup(monkeypatch, tmp_path)
    store = EvidenceStore(tmp_path)
    abstract = 'A registered abstract describes the tested method for analysis.'
    bundle = LiteratureBundle.model_validate({
        'research_question': 'question',
        'citations': [{'key': 'method1', 'title': 'Method study', 'year': '2020', 'pmid': '12345', 'doi': '10.1234/example',
                       'relevance': 'Source excerpt: ' + abstract}],
        'screening_decisions': [{'citation_key': 'method1', 'source': 'pubmed', 'disposition': 'include',
            'evidence_role': 'method', 'rationale': 'Method reference'}],
        'design_evidence_cards': [{'citation_key': 'method1', 'evidence_role': 'method', 'access_mode': 'abstract',
            'abstract_locator': 'https://pubmed.ncbi.nlm.nih.gov/12345/',
            'abstract_sha256': hashlib.sha256(abstract.encode()).hexdigest(), 'supplement_status': 'unknown',
            'reviewed_at': '2026-01-01T00:00:00Z', 'evidence': [{'dimension': 'primary_model_and_sensitivities',
                'source_backed_summary': 'This describes a tested method.', 'locator': 'abstract'}]}],
    })
    def register(key, content):
        store.register_text(kind='log', description='fixture', text=content, filename=key + '.json', evidence_id=key)
    register('preplan_literature_bundle', bundle.model_dump_json())
    register('abstract1', json.dumps({'citation_key': 'method1', 'pmid': '12345', 'doi': '10.1234/example', 'abstract': abstract}))
    kw = dict(service=Service(), project_id='project1', session_id='session1', run_id='run1')
    proposed = view.CandidateLiteratureReference(decision_id='outcome', citation_key='method1',
        source_evidence_id='abstract1', verbatim_excerpt='describes the tested method')
    temporary = view.project_proposal(**kw, proposed=proposed)
    assert next(x for x in temporary['proposal']['decisions'] if x['decision_id'] == 'outcome')['bindings']
    from easyicu.webserver.routes import non_executable_proposal as route
    from fastapi import FastAPI
    monkeypatch.setattr(route, 'get_pi_copilot_service', lambda: Service())
    app = FastAPI()
    app.include_router(route.router)
    xml = (b'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID>'
           b'<Article><ArticleTitle>Method study</ArticleTitle><Abstract><AbstractText>'
           + abstract.encode() + b'</AbstractText></Abstract></Article></MedlineCitation>'
           b'<PubmedData><ArticleIdList><ArticleId IdType="doi">10.1234/example</ArticleId></ArticleIdList></PubmedData>'
           b'</PubmedArticle></PubmedArticleSet>')
    monkeypatch.setattr(view.PubMedLiteratureClient, '_http_get', lambda self, path, params: xml)
    client = TestClient(app)
    assert client.post('/api/pi/proposal/project1/session1/run1/pubmed-abstract',
        data={'citation_key': 'method1'}).status_code == 403
    collected = client.post('/api/pi/proposal/project1/session1/run1/pubmed-abstract',
        headers={'Origin': 'http://testserver'}, data={'citation_key': 'method1'})
    assert collected.status_code == 200 and '已登记候选原摘要' in collected.text
    separate = view._candidate_evidence_root('study1', 'session1', 'run1')
    registry = view.ReadOnlyEvidenceRegistry(separate)
    new_id = next(row.evidence_id for row in registry.records if row.evidence_id.startswith('candidate_abstract_'))
    assert store.get(new_id) is None
    from_collection = view.project_proposal(**kw, proposed=proposed.model_copy(update={'source_evidence_id': new_id}))
    assert next(x for x in from_collection['proposal']['decisions'] if x['decision_id'] == 'outcome')['bindings']
    save_data = proposed.model_copy(update={'source_evidence_id': new_id, 'decision_id': 'method_background'}).model_dump()
    saved = client.post('/api/pi/proposal/project1/session1/run1/candidate/save',
        headers={'Origin': 'http://testserver'}, data=save_data)
    assert saved.status_code == 200 and '候选绑定已保存' in saved.text
    persisted = view.project_proposal(**kw)
    method = next(x for x in persisted['proposal']['decisions'] if x['decision_id'] == 'method_background')
    assert method['bindings'][0]['applicability_verified'] is False
    assert method['bindings'][0]['literature_role'] == 'method'
    assert store.get('candidate_literature_bindings') is None
    monkeypatch.setattr(view.PubMedLiteratureClient, '_http_get', lambda self, path, params: xml.replace(b'12345', b'99999'))
    mismatch = client.post('/api/pi/proposal/project1/session1/run1/pubmed-abstract',
        headers={'Origin': 'http://testserver'}, data={'citation_key': 'method1'})
    assert mismatch.status_code == 409
    wrong_doi = xml.replace(b'10.1234/example', b'10.1234/wrong').replace(
        b'</PubmedArticle>', b'<ReferenceList><ArticleId IdType="doi">10.1234/example</ArticleId></ReferenceList></PubmedArticle>')
    monkeypatch.setattr(view.PubMedLiteratureClient, '_http_get', lambda self, path, params: wrong_doi)
    mismatch = client.post('/api/pi/proposal/project1/session1/run1/pubmed-abstract',
        headers={'Origin': 'http://testserver'}, data={'citation_key': 'method1'})
    assert mismatch.status_code == 409
    response = TestClient(app).post('/api/pi/proposal/project1/session1/run1/candidate', data=proposed.model_dump())
    assert response.status_code == 200
    assert '当前是临时候选预览，未保存、未审核' in response.text
    assert 'describes the tested method' in response.text
    assert '适用性尚未核验' in response.text
    invalid = TestClient(app).post('/api/pi/proposal/project1/session1/run1/candidate',
        data={**proposed.model_dump(), 'verbatim_excerpt': 'invented excerpt here'})
    assert invalid.status_code == 409
    unchanged = view.project_proposal(**kw)
    assert not next(x for x in unchanged['proposal']['decisions'] if x['decision_id'] == 'outcome')['bindings']
    with pytest.raises(ValueError, match='original_excerpt_unverified'):
        view.project_proposal(**kw, proposed=proposed.model_copy(update={'verbatim_excerpt': 'invented excerpt here'}))
    manifest = {'schema_version': 'easyicu.candidate-literature-bindings/1',
        'configuration_sha256': canonical_sha256(study), 'references': [{'decision_id': 'outcome',
        'decision_text_sha256': hashlib.sha256('结局：outcome'.encode()).hexdigest(), 'citation_key': 'method1',
        'source_evidence_id': 'abstract1', 'verbatim_excerpt': 'describes the tested method'}]}
    register('candidate_literature_bindings', json.dumps(manifest))
    result = view.project_proposal(**kw)
    outcome = next(x for x in result['proposal']['decisions'] if x['decision_id'] == 'outcome')
    assert outcome['bindings'][0]['verbatim_excerpt'] == 'describes the tested method'
    assert outcome['bindings'][0]['literature_role'] == 'method'
    assert outcome['bindings'][0]['applicability_verified'] is False
    assert outcome['unbound_evidence'] == ['source']
    question = next(x for x in result['proposal']['decisions'] if x['decision_id'] == 'question')
    assert question['unbound_evidence'] == ['source', 'literature']
    from easyicu.webserver.routes.non_executable_proposal import _binding_status
    rendered = _binding_status(outcome)
    assert '方法背景' in rendered and '适用性尚未核验' in rendered

    original = store.get('abstract1')
    (tmp_path / original.relative_path).write_text('tampered')
    with pytest.raises(ValueError, match='integrity_failed'):
        view.project_proposal(**kw)


def test_known_fields_are_readable_and_unknown_or_missing_are_safe(monkeypatch):
    from easyicu.webserver.routes import non_executable_proposal as route
    monkeypatch.setattr(route, 'get_pi_copilot_service', lambda: object())
    monkeypatch.setattr(route, 'project_proposal', lambda **kw: {'proposal': {
        'original_question': '初始问题', 'decisions': [
            {'decision_id': 'question', 'text': '研究问题：当前问题'},
            {'decision_id': 'primary_exposure', 'text': '暴露：乳酸'},
            {'decision_id': 'outcome', 'text': '结局：病例关闭时长'},
            {'decision_id': '<script>', 'text': '<b>内容</b>'},
        ]}, 'literature_candidates': {'retained_citation_keys': ['one'] * 12}})
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(route.router)
    text = TestClient(app).get('/api/pi/proposal/project/session/run').text
    assert '<h2>研究问题</h2><p>当前问题</p>' in text
    assert '<h2>暴露</h2><p>乳酸</p>' in text
    assert '<h2>结局</h2><p>病例关闭时长</p>' in text
    assert '<h2>研究人群</h2><p>尚未填写</p>' in text
    assert '<h2>question</h2>' not in text and '<h2>primary_exposure</h2>' not in text
    assert '&lt;script&gt;' in text and '<script>' not in text
    assert '12 条可引用文献标识；文献清单本身不代表支持具体决定' in text


@pytest.mark.parametrize('cohort, expected', [
    ({'label': '成人', 'review': '定义待审'}, '研究人群：成人；定义待审'),
    ('成人', '研究人群：成人'),
    ({'label': '', 'review': ''}, None),
    (None, None),
])
def test_cohort_projection_preserves_only_saved_user_visible_values(monkeypatch, tmp_path, cohort, expected):
    study, _ = setup(monkeypatch, tmp_path)
    study['cohort'] = cohort
    result = view.project_proposal(service=Service(), project_id='project1', session_id='session1', run_id='run1')
    rows = [row for row in result['proposal']['decisions'] if row['decision_id'] == 'cohort']
    assert ([row['text'] for row in rows] == [expected]) if expected else not rows


def test_cohort_text_is_escaped_and_not_claimed_verified(monkeypatch):
    from easyicu.webserver.routes import non_executable_proposal as route
    monkeypatch.setattr(route, 'get_pi_copilot_service', lambda: object())
    monkeypatch.setattr(route, 'project_proposal', lambda **kw: {'proposal': {
        'original_question': '问题', 'decisions': [
            {'decision_id': 'cohort', 'text': '研究人群：成人<script>；定义待审'},
        ]}, 'literature_candidates': None})
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(route.router)
    text = TestClient(app).get('/api/pi/proposal/project/session/run').text
    assert '成人&lt;script&gt;；定义待审' in text and '<script>' not in text
    assert '人群定义尚未核验' in text


@pytest.mark.parametrize('changed', ['question', 'session', 'run', 'project', 'configuration'])
def test_concurrent_authority_change_fails_closed(monkeypatch, tmp_path, changed):
    setup(monkeypatch, tmp_path)
    service = Service()
    if changed == 'question':
        texts = iter(['original', 'changed'])
        monkeypatch.setattr(service.replay_store, 'question_wording_context',
                            lambda **kw: {'user_turns': [{'text': next(texts)}]})
    elif changed == 'session':
        listings = iter([{'sessions': [{'session_id': 'session1'}]}, {'sessions': []}])
        monkeypatch.setattr(service, 'list_sessions', lambda **kw: next(listings))
    elif changed == 'run':
        snapshots = iter([{'child_jobs': [{'run_id': 'run1'}]}, {'child_jobs': []}])
        monkeypatch.setattr(service.replay_store, 'snapshot', lambda **kw: next(snapshots))
    elif changed == 'project':
        projects = iter(['study1', 'study2'])
        monkeypatch.setattr(service.project_store, 'resolve', lambda *a: next(projects))
    else:
        digests = iter(['bound', 'changed'])
        monkeypatch.setattr(view.study_contexts, 'scientific_configuration_sha256', lambda _: next(digests))
    with pytest.raises(ValueError, match='changed'):
        view.project_proposal(service=service, project_id='project1', session_id='session1', run_id='run1')

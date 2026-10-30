"""Host scopes proposal display to the matching project, session, run and config."""
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from easyicu.research_agent.authority.evidence_store import EvidenceStore
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
    assert '<form' not in response.text
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
    assert '12 条可引用文献标识；尚未与具体决定绑定' in text


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

"""Synthetic project/session owner for non-executable proposal tests."""
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
           'scientific_configuration_sha256': 'b' * 64}
    monkeypatch.setattr(view.study_contexts, 'get_context', lambda study_id: study)
    monkeypatch.setattr(view.study_contexts, 'scientific_configuration_sha256', lambda current: 'b' * 64)
    monkeypatch.setattr(view, 'research_pipeline_project_root', lambda study_id: tmp_path)
    monkeypatch.setattr(view, 'list_bound_run_history', lambda **kw: [row])
    monkeypatch.setattr(view.run_step_trace, 'pipeline_run_dir', lambda wrapper, run: tmp_path)
    monkeypatch.setattr(view.run_step_trace, 'failed_literature_preview_binding', lambda *a, **kw: None)
    return study, row

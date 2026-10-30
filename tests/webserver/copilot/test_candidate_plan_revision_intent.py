"""Explicit plan amendments enter the existing review-bound replan owner."""
from types import SimpleNamespace

import pytest

from easyicu.webserver.pi_copilot import tools
from easyicu.webserver.pi_copilot.contracts import AuthorityBinding
from easyicu.webserver.pi_copilot.plan_revision_intent import requested_candidate_run

pytest_plugins = ['tests.webserver.copilot.pi_copilot_contract_fixtures']

RUN = 'run_20261030T054023_e9ff8e'
MESSAGE = ('请修订完整研究计划：原问题要求每位患者首次 ICU 入住。'
           '仅纳入能可靠确定首次 ICU 入住的患者，报告排除数及局限，'
           '增加各性别组院内死亡比例的 95% 置信区间。只形成计划供审阅。')


def projection():
    # Shape matches ProjectWorkflowProjection from the actual eICU Demo run.
    return SimpleNamespace(
        workflow=SimpleNamespace(plan_review_summary={'run_id': RUN, 'status': 'analysis_only'}),
        latest_run={'present': True, 'run_id': RUN, 'gate_reason_code': 'human_plan_review_required'},
        active_job={'present': False},
        runs=[{'run_id': RUN, 'run_status': 'human_review_pending', 'authoritative': True,
               'artifact_names': ['agent_plan.json', 'scientific_plan_review.json']}],
    )


def test_explicit_revision_targets_the_current_reviewable_demo_plan():
    assert requested_candidate_run(MESSAGE, message_origin='typed', projection=projection()) == RUN


@pytest.mark.parametrize('message,origin', [
    ('', 'typed'),
    ('请解释如何修改研究计划？', 'typed'),
    ('不要修改研究计划。', 'typed'),
    ('请修改研究计划。', 'model_option'),
    ('请修改研究计划。', 'host_prompt'),
    ('请修改研究计划。', None),
    ('请给这份计划一些建议。', 'typed'),
    ('请修改研究计划。', 'typed'),
    ('请修改研究计划供审阅，不执行分析。', 'typed'),
    ('请根据以下审阅意见修订整份研究计划，保留原始问题与数据来源，并说明修改依据。不要开始分析：', 'typed'),
    ('Please revise the complete research plan using the following review comments, preserve the original question and data source, and explain the changes. Do not start analysis:', 'typed'),
    ('请整理研究报告。', 'typed'),
])
def test_non_authorized_or_non_revision_messages_do_not_route(message, origin):
    assert requested_candidate_run(message, message_origin=origin, projection=projection()) == ''


def test_review_control_draft_with_user_amendment_routes():
    draft = '请根据以下审阅意见修订整份研究计划，保留原始问题与数据来源，并说明修改依据。不要开始分析：'
    assert requested_candidate_run(draft + '仅纳入能可靠识别首次入住的患者，并报告排除数。', message_origin='typed', projection=projection()) == RUN


@pytest.mark.parametrize('mutation', ['active', 'changed_run', 'historical', 'not_pending', 'no_plan', 'missing_review'])
def test_ambiguous_or_changed_candidate_does_not_route(mutation):
    p = projection()
    if mutation == 'active':
        p.active_job = {'present': True}
    if mutation == 'changed_run':
        p.latest_run = {'present': True, 'run_id': 'another'}
    if mutation == 'historical':
        p.runs[0]['authoritative'] = False
    if mutation == 'not_pending':
        p.runs[0]['run_status'] = 'completed'
    if mutation == 'no_plan':
        p.runs[0]['artifact_names'] = []
    if mutation == 'missing_review':
        p.workflow.plan_review_summary = None
    assert requested_candidate_run(MESSAGE, message_origin='typed', projection=p) == ''


def test_replan_cannot_switch_to_a_newer_run_after_message_admission(monkeypatch):
    context = SimpleNamespace(session=SimpleNamespace(binding=AuthorityBinding()), expected_replan_source_run_id=RUN)
    monkeypatch.setattr(tools, '_bound_context', lambda _: {'id': 'study'})
    monkeypatch.setattr(tools, '_select_run', lambda _: {'run_id': 'newer'})
    result = tools._request_replan(context, {'strategy': 'fresh'})
    assert result['status'] == 'blocked'
    assert result['code'] == 'plan_revision_source_changed'


def test_revision_cannot_be_replaced_by_checkpoint_resume(monkeypatch):
    context = SimpleNamespace(session=SimpleNamespace(binding=AuthorityBinding()), expected_replan_source_run_id=RUN)
    monkeypatch.setattr(tools, '_bound_context', lambda _: {'id': 'study'})
    monkeypatch.setattr(tools, '_select_run', lambda _: {'run_id': RUN})
    result = tools._request_replan(context, {'strategy': 'resume_checkpoint'})
    assert result['status'] == 'blocked'
    assert result['code'] == 'plan_revision_requires_fresh_candidate'


def test_same_run_no_longer_pending_cannot_be_revised(monkeypatch):
    context = SimpleNamespace(session=SimpleNamespace(binding=AuthorityBinding()), expected_replan_source_run_id=RUN)
    monkeypatch.setattr(tools, '_bound_context', lambda _: {'id': 'study'})
    monkeypatch.setattr(tools, '_select_run', lambda _: {
        'run_id': RUN, 'study_id': 'study', 'run_status': 'completed',
        'artifact_names': ['agent_plan.json'],
    })
    result = tools._request_replan(context, {'strategy': 'fresh'})
    assert result['status'] == 'blocked'
    assert result['code'] == 'plan_revision_source_changed'


@pytest.mark.parametrize('change', ['review_approved', 'active_job', 'new_run'])
def test_revision_rechecks_candidate_at_provider_authorization(monkeypatch, change):
    from easyicu.webserver.pi_copilot.contracts import PiSessionRecord, ToolExecutionContext

    session = PiSessionRecord(
        session_id='pi-stale-revision', external_llm_opt_in=True,
        binding=AuthorityBinding(study_context_id='study'),
    )
    context = ToolExecutionContext(
        session=session, user_message=MESSAGE, allowed_actions={'provider_run'},
        data_handling='zero_retention_declared', expected_replan_source_run_id=RUN,
    )
    current = projection()
    if change == 'review_approved':
        current.runs[0]['run_status'] = 'completed'
    elif change == 'active_job':
        current.active_job = {'present': True}
    else:
        current.latest_run = {'present': True, 'run_id': 'newer'}
    monkeypatch.setattr(tools, '_bound_context', lambda _: {'id': 'study'})
    monkeypatch.setattr(tools, '_account_environment_for_research_provider', lambda _: (None, None))
    monkeypatch.setattr(tools, '_workflow_snapshot', lambda *_args, **_kwargs: {})
    monkeypatch.setattr(tools, 'build_project_workflow_projection', lambda **_: current)

    def submit(_request, *, authorize, **_kwargs):
        authorize()
        raise AssertionError('stale revision must not consume authorization or start')

    monkeypatch.setattr(tools.research_run_submission, 'submit_research_run', submit)
    result = tools._run(context, {'run_type': 'full'}, run_intent='candidate_plan')
    assert result['status'] == 'blocked'
    assert result['code'] == 'plan_revision_source_changed'
    assert 'provider_run' in context.grant.available_actions


def test_service_routes_typed_revision_and_requires_a_submission_receipt(tmp_path, study_state, monkeypatch):
    import time
    from easyicu.webserver.pi_copilot import service as service_module
    from easyicu.webserver.pi_copilot.service import PiCopilotService
    from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway

    class Gateway(FakeGateway):
        emitted = False
        def request(self, method, params, **kwargs):
            state = super().request(method, params, **kwargs)
            if method == 'session.prompt' and self.emitted:
                kwargs['event_sink']({'type': 'tool_end', 'tool_name': 'easyicu_request_replan',
                                      'code': 'easyicu_full_run_submitted', 'job_id': 'job-test', 'status': 'ok',
                                      'is_error': self.emitted == 'error'})
            return state

    gateway = Gateway()
    service = PiCopilotService(store_path=tmp_path / 'sessions.json', gateway=gateway)
    session_id = service.create_session(project_id='project-revision', external_llm_opt_in=True)['session']['session_id']
    monkeypatch.setattr(service_module, 'build_project_workflow_projection', lambda **_: projection())
    monkeypatch.setattr(service, '_watch_child_job_for_replay', lambda **_: None)

    def run_once():
        submitted = service.send_message(session_id, project_id='project-revision', message=MESSAGE, message_origin='typed')
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = service_module.jobs.MANAGER.get(submitted['job_id'])
            if job and job.status != 'running':
                break
            time.sleep(0.01)
        return job

    failed = run_once()
    assert failed.status == 'failed'
    assert 'pi_plan_revision_not_submitted' in str(failed.error)
    call = [call for call in gateway.calls if call[0] == 'session.prompt'][-1]
    assert call[1]['intent'] == 'request_candidate_plan_revision'
    assert call[2].user_message == MESSAGE
    assert call[2].expected_replan_source_run_id == RUN
    assert call[2].allowed_actions == frozenset({'provider_run'})
    gateway.emitted = 'error'
    assert run_once().status == 'failed'
    gateway.emitted = 'ok'
    assert run_once().status == 'done'

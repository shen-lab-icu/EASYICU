"""Report-only chat submissions retain host authority and the shared run owner."""
from types import SimpleNamespace

import pytest

from easyicu.webserver import research_run_submission as submission
from easyicu.webserver.pi_copilot import tools
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding, PiCopilotError, PiSessionRecord, ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.turn_authority import infer_explicit_turn_actions


@pytest.mark.parametrize('message', [
    '请修订当前报告：统一图表与正文的术语，修正摘要中的重复短语，更新当前版本 PDF。请复用已有分析结果。',
    '请修复这份论文。', '更新报告并重新生成 PDF，不要重跑分析。',
    'Please repair the current report using existing results.',
    'Regenerate the PDF.',
])
def test_report_request_grants_only_report_revision(message):
    assert infer_explicit_turn_actions(message) == frozenset({'report_revision'})


@pytest.mark.parametrize('message', [
    '不要修订当前报告。', '请解释怎么更新报告。', '只查看当前 PDF。',
    'How should I revise the report?', "Don't regenerate the PDF.",
    '请审阅当前报告，暂不修改稿件。',
])
def test_readonly_and_denied_report_requests_do_not_grant(message):
    assert not infer_explicit_turn_actions(message)


@pytest.fixture
def setup(monkeypatch):
    captured = []
    monkeypatch.setattr(tools, '_run_rows', lambda ctx: [{'run_id': 'run-source'}])
    monkeypatch.setattr(tools, '_bound_context', lambda binding: {'id': 'study-report'})
    monkeypatch.setattr(tools, '_workflow_snapshot', lambda *a, **k: {'next_action_code': 'provider_ready_to_generate_plan'})
    monkeypatch.setattr(tools, '_account_environment_for_research_provider', lambda c: (None, None))
    def submit(request, *, authorize, **kwargs):
        authorize()
        captured.append(request)
        return SimpleNamespace(resume_source_job_id=None, model_dump=lambda **kw: {
            'job_id': 'report-job', 'kind': 'agent-run', 'status': 'running',
            'study_context_id': 'study-report', 'study_context_revision': 4,
            'engine': 'research_agent_pipeline',
        })
    monkeypatch.setattr(submission, 'submit_research_run', submit)
    return captured


def context(actions=('report_revision',), opt_in=True):
    return ToolExecutionContext(session=PiSessionRecord(
        session_id='pi-report', external_llm_opt_in=opt_in,
        binding=AuthorityBinding(study_context_id='study-report', study_revision=3),
    ), allowed_actions=actions)


def test_report_tool_uses_exact_owned_source_and_existing_submission(setup):
    ctx = context(('report_revision', 'literature', 'provider_run'))
    result = tools.execute_tool('easyicu_repair_report', {}, ctx)
    assert result['code'] == 'easyicu_report_repair_submitted'
    assert result['details']['report_only'] is True
    assert result['details']['run_id_status'] == 'existing_analysis'
    assert result['details']['source_run_id'] == 'run-source'
    req = setup[0]
    assert req.report_only and req.execution_resume_source_run_id == 'run-source'
    assert req.intent == 'reviewed_analysis'
    assert req.planner_start_mode == 'auto'
    assert not req.literature_search_authorized
    assert req.plan_change_request is None
    assert req.plan_revision_source_run_id == ''
    assert 'provider_run' in ctx.allowed_actions
    with pytest.raises(PiCopilotError):
        tools.execute_tool('easyicu_repair_report', {}, ctx)
    assert len(setup) == 1


@pytest.mark.parametrize('actions', [(), ('provider_run',), ('run',)])
def test_other_grants_cannot_authorize_report_repair(setup, actions):
    result = tools.execute_tool('easyicu_repair_report', {}, context(actions))
    assert result['code'] == 'pi_action_authorization_required'
    assert setup == []


def test_report_tool_rejects_foreign_or_missing_source(setup):
    result = tools.execute_tool('easyicu_repair_report', {'run_id': 'foreign-run'}, context())
    assert result['code'] == 'report_repair_run_required'
    assert setup == []


def test_report_tool_preserves_opt_in_gate(setup):
    result = tools.execute_tool('easyicu_repair_report', {}, context(opt_in=False))
    assert result['code'] == 'external_llm_opt_in_required'
    assert setup == []


def test_report_tool_does_not_admit_scientific_or_provider_parameters(setup):
    with pytest.raises(PiCopilotError, match='unknown arguments'):
        tools.execute_tool('easyicu_repair_report', {'report_only': False}, context())
    assert setup == []

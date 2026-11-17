"""The page shows a decision the host is starting as state, not as a button.

A job-starting click carries the decision it answers (``host_action``); the
host refuses a repeat while the decision starts, a decision another job
answers, and a stale one (``webserver.host_action_jobs``).  The page reads
those answers as the study's state: it shows "正在准备…" without a button,
reloads its projection, and follows the start until the job exists.
"""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read,
)


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    return node


def test_a_starting_decision_is_a_card_with_nothing_to_act_on() -> None:
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value ?? '') }} }};
      eval({_read('js/screens-guided-pi-confirmation-data.js')!r});
      eval({_read("js/screens-guided-pi-confirmation.js")!r});
      const confirmation = window.EU_GUIDED_PI_CONFIRMATION.create({{
        tr: (en, zh) => zh, esc: value => String(value), iconHtml: () => '',
        resourceButton: () => '', sessionIsStale: () => false, busy: () => false,
        workflow: () => ({{
          next_action_code: 'starting',
          starting: {{ action_code: 'prepare_analysis_data', started_at: 1 }},
        }}),
        session: () => ({{ archived_child_jobs: [] }}),
      }});
      process.stdout.write(JSON.stringify({{
        spec: confirmation.workflowConfirmation(),
        html: confirmation.workflowConfirmationHtml(),
      }}));
    """
    rendered = json.loads(run_node(_node(), script, check=True).stdout)

    assert rendered["spec"]["nonApprovable"] is True
    assert rendered["spec"]["grants"] == []
    assert "正在准备…" in rendered["html"]
    for control in (
        "data-gpi-confirm-action",
        "data-gpi-confirm-edit",
        "data-gpi-confirm-reject",
    ):
        assert control not in rendered["html"]


_ACTIONS = """
  global.window = {};
  eval(__ACTIONS__);
  const calls = [];
  const timers = [];
  global.setTimeout = (callback, ms) => { timers.push([callback, ms]); return timers.length; };
  let workflow = {
    next_action_code: 'plan_execution_upgrade_required',
    host_decisions: {plan_transition: {
      family: 'plan_transition', next_action_code: 'plan_execution_upgrade_required',
      scientific_configuration_sha256: 'a'.repeat(64), source_run_id: 'run_candidate',
    }},
  };
  const host = {
    tr: (en, zh) => zh, errorText: error => 'copy:' + error.code,
    regeneration: {}, nextActions: {}, replay: {}, projectId: () => 'project-1',
    turnGrants: () => [], busy: () => false, sessionIsStale: () => false,
    session: () => ({
      session_id: 'session-1',
      binding: {run_id: 'run_candidate', study_context_id: 'study-1', study_revision: 3},
      research_provider: {provider: 'openai', credential_source: 'pi_verified'},
    }),
    workflow: () => workflow,
    api: () => ({
      loadStudyContext: async () => ({context: {question: 'q', data_source: {path: '/prepared'}}}),
      startAgentRun: async body => {
        calls.push(['plan', body]);
        throw Object.assign(new Error('refused'), {code: REFUSAL});
      },
    }),
    loadWorkflow: async () => {
      calls.push(['workflow']);
      workflow = {...workflow, next_action_code: AFTER};
    },
    render: (...args) => calls.push(['render', ...args]),
    recordHostAction: async (...args) => calls.push(['host-action', ...args]),
    watchChildJob: (...args) => calls.push(['child', ...args]),
    setBusy: () => {}, setError: value => calls.push(['error', value]),
    appendMessage: value => calls.push(['message', value.text]),
  };
  const actions = window.EU_GUIDED_PI_PLAN_ACTIONS.create(host);
  const settle = () => new Promise(resolve => setImmediate(resolve));
"""


def _plan_actions(body: str, *, refusal: str = "", after: str = "starting") -> dict:
    script = (
        _ACTIONS.replace(
            "__ACTIONS__", json.dumps(_read("js/screens-guided-pi-plan-actions.js"))
        )
        .replace("REFUSAL", json.dumps(refusal))
        .replace("AFTER", json.dumps(after))
        + body
    )
    return json.loads(run_node(_node(), script, check=True).stdout)


def test_refused_prepare_reclicks_only_on_another_explicit_click() -> None:
    result = _plan_actions("""
      (async () => {
        const confirmation = {code: 'plan_execution_upgrade_required', grants: ['provider_run']};
        await actions.confirmWorkflow(confirmation);
        for (let poll = 0; poll < 12; poll += 1) {
          host.render();
          await actions.continueSystemOwnedPlanProgression({passive: true});
        }
        const beforeReclick = calls.filter(row => row[0] === 'plan').length;
        await actions.confirmWorkflow(confirmation);
        process.stdout.write(JSON.stringify({beforeReclick, calls}));
      })().catch(error => { console.error(error); process.exit(1); });
    """, refusal="research_pipeline_execution_runtime_unavailable")
    assert result["beforeReclick"] == 1
    posts = [row[1] for row in result["calls"] if row[0] == "plan"]
    assert len(posts) == 2
    assert posts[0] == posts[1]
    assert posts[0]["host_action"]["action_code"] == "prepare_analysis_data"
    assert posts[0]["plan_revision_source_run_id"] == "run_candidate"
    assert posts[0]["credential_source"] == "pi_verified"
    assert "report_only" not in posts[0]


def test_run_readiness_is_at_the_card_and_the_action_boundary() -> None:
    source = _read("js/screens-guided-pi-run-readiness.js")
    body = """
      eval(__READINESS__);
      eval(__DATA__);
      eval(__CONFIRMATION__);
      (async () => {
        const state = {
          session: {...host.session(), research_provider: {
            provider: 'codex', model: 'synthetic-model', credential_source: 'codex_user_auth',
          }}, workflow, runtime: {status: 'ready'}, codexAuth: null,
        };
        const readiness = window.EasyICU.guidedPi.require('runReadiness').create({state, tr: host.tr});
        host.session = () => state.session;
        host.runReadiness = readiness.forAction;
        const confirmation = window.EU_GUIDED_PI_CONFIRMATION.create({
          ...host, esc: value => String(value), iconHtml: () => '', resourceButton: () => '',
        });
        const blocked = confirmation.workflowConfirmationHtml();
        await actions.confirmWorkflow(confirmation.workflowConfirmation());
        const noPost = !calls.some(row => row[0] === 'plan');
        state.codexAuth = {authentication_verified: true};
        const ready = confirmation.workflowConfirmationHtml();
        state.workflow.plan_review_summary = {run_id: 'another_plan'};
        const mismatch = confirmation.workflowConfirmationHtml();
        const mismatchState = readiness.forAction('plan_execution_upgrade_required');
        state.workflow.plan_review_summary = {run_id: 'run_candidate'};
        state.session.research_provider = {provider: 'openai', model: 'synthetic', credential_source: 'pi_verified'};
        state.runtime = {status: 'unavailable'};
        const providerMissing = readiness.forAction('plan_execution_upgrade_required');
        state.runtime = {status: 'ready', api_transport: 'openai-completions'};
        const providerReady = readiness.forAction('plan_execution_upgrade_required');
        const handled = readiness.noteRefusal({code: 'pi_provider_research_inference_unverified'});
        const providerRefused = readiness.forAction('plan_execution_upgrade_required');
        window.EasyICU.guidedPi.declare('hosted', {active: () => true});
        const hostedBlocked = confirmation.workflowConfirmationHtml();
        process.stdout.write(JSON.stringify({blocked, ready, mismatch, mismatchState, providerMissing, providerReady, handled, providerRefused, hostedBlocked, noPost, calls}));
      })().catch(error => { console.error(error); process.exit(1); });
    """.replace("__READINESS__", json.dumps(source)).replace(
        "__DATA__", json.dumps(_read("js/screens-guided-pi-confirmation-data.js"))
    ).replace("__CONFIRMATION__", json.dumps(_read("js/screens-guided-pi-confirmation.js")))
    result = _plan_actions(body)
    assert result["noPost"] is True
    assert "此浏览器登录账户" in result["blocked"]
    assert "data-gpi-config" in result["blocked"]
    assert "data-gpi-confirm-action" not in result["blocked"]
    assert "data-gpi-confirm-action" in result["ready"]
    assert "data-gpi-confirm-action disabled" in result["mismatch"]
    assert result["mismatchState"]["connect"] is False
    assert result["providerMissing"]["connect"] is True
    assert result["providerReady"] is None
    assert result["handled"] is True
    assert result["providerRefused"]["connect"] is True
    assert "管理员处理" in result["hostedBlocked"]
    assert "data-gpi-config" not in result["hostedBlocked"]
    assert "data-gpi-confirm-action disabled" in result["hostedBlocked"]
    assert not any(row[0] == "error" and row[1] for row in result["calls"])


def test_run_readiness_sibling_is_loaded_before_the_screen() -> None:
    index = _read("index.html")
    readiness = "js/screens-guided-pi-run-readiness.js?v=20261117-run1"
    assert index.index(readiness) < index.index('src="js/screens-guided-pi.js?')
    screen = _read("js/screens-guided-pi.js")
    assert "runReadiness: RUN_READINESS.forAction" in screen
    assert "noteRunRefusal: RUN_READINESS.noteRefusal" in screen


def test_a_repeat_the_host_says_is_starting_shows_no_banner_and_follows_it() -> None:
    result = _plan_actions(
        """
      (async () => {
        await actions.confirmWorkflow({
          code: 'plan_execution_upgrade_required', message: 'internal', grants: ['provider_run'],
        });
        await settle();
        process.stdout.write(JSON.stringify({calls, scheduled: timers.length}));
      })().catch(error => { console.error(error); process.exit(1); });
        """,
        refusal="host_action_in_progress",
    )

    calls = result["calls"]
    [plan] = [call for call in calls if call[0] == "plan"]
    assert plan[1]["host_action"]["action_code"] == "prepare_analysis_data"
    assert not any(call[0] == "error" and call[1] for call in calls)
    assert ["workflow"] in calls
    assert result["scheduled"] == 1
    assert not any(call[0] in {"host-action", "child"} for call in calls)


@pytest.mark.parametrize("refusal", ["study_job_running", "host_action_decision_stale"])
def test_a_refused_decision_is_reported_and_the_page_shows_the_study_again(
    refusal: str,
) -> None:
    result = _plan_actions(
        """
      (async () => {
        await actions.confirmWorkflow({
          code: 'plan_execution_upgrade_required', message: 'internal', grants: ['provider_run'],
        });
        await settle();
        process.stdout.write(JSON.stringify({calls, scheduled: timers.length}));
      })().catch(error => { console.error(error); process.exit(1); });
        """,
        refusal=refusal,
        after="research_planning_running",
    )

    assert ["error", f"copy:{refusal}"] in result["calls"]
    assert ["workflow"] in result["calls"]
    assert result["scheduled"] == 0


def test_a_reopened_page_follows_a_starting_decision_without_starting_a_plan() -> None:
    result = _plan_actions(
        """
      workflow = {...workflow, next_action_code: 'starting'};
      (async () => {
        const passive = await actions.continueSystemOwnedPlanProgression({passive: true});
        const again = await actions.continueSystemOwnedPlanProgression({passive: true});
        const scheduled = timers.length;
        await timers[0][0]();
        process.stdout.write(JSON.stringify({passive, again, scheduled, calls, rescheduled: timers.length}));
      })().catch(error => { console.error(error); process.exit(1); });
        """,
        after="starting",
    )

    assert result["passive"] is False and result["again"] is False
    assert result["scheduled"] == 1  # one follow-up at a time
    assert ["workflow"] in result["calls"]
    assert result["rescheduled"] == 2  # still starting: follow again
    assert not any(call[0] == "plan" for call in result["calls"])
    # Nothing moved on: no repaint closes what the researcher opened.
    assert not any(call[0] == "render" for call in result["calls"])


def test_a_start_that_moves_on_is_repainted_once_keeping_the_view() -> None:
    result = _plan_actions(
        """
      workflow = {...workflow, next_action_code: 'starting'};
      (async () => {
        await actions.continueSystemOwnedPlanProgression({passive: true});
        await timers[0][0]();
        process.stdout.write(JSON.stringify({calls, rescheduled: timers.length}));
      })().catch(error => { console.error(error); process.exit(1); });
        """,
        after="research_planning_running",
    )

    renders = [call for call in result["calls"] if call[0] == "render"]
    assert renders == [["render", True]]  # the log keeps its scroll position
    assert result["rescheduled"] == 1  # the job exists: the follow ends


def test_host_decision_refusals_have_their_own_copy() -> None:
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value ?? '') }} }};
      eval({_read("js/screens-guided-pi-error-text.js")!r});
      const text = window.EU_GUIDED_PI_ERROR_TEXT.create({{
        tr: (en, zh) => zh, staticPreview: () => false,
      }}).errorText;
      process.stdout.write(JSON.stringify({{
        starting: text({{ code: 'host_action_in_progress' }}),
        running: text({{ code: 'study_job_running' }}),
        failed: text({{ code: 'host_action_decision_stale', details: {{ job_status: 'failed' }} }}),
        done: text({{ code: 'host_action_decision_stale', details: {{ job_status: 'done' }} }}),
        changed: text({{ code: 'host_action_decision_stale', details: {{}} }}),
        mismatch: text({{ code: 'host_action_request_mismatch' }}),
      }}));
    """
    copy = json.loads(run_node(_node(), script, check=True).stdout)

    assert "正在准备这一步" in copy["starting"]
    assert "另一个任务正在运行或准备中" in copy["running"]
    assert "启动过，没有完成" in copy["failed"]
    assert "已经完成了这一步" in copy["done"]
    assert "研究状态已经变化" in copy["changed"]
    assert "刷新项目" in copy["mismatch"]
    assert len(set(copy.values())) == len(copy)

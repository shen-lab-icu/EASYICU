"""The retry action must name the run the host offered the retry for.

E2's approved landmark execution failed closed on one auxiliary step, so the
host offered `failed_pipeline_execution_retry_available` -- "Retry analysis from
the failed step", reusing the completed steps and never re-running the Planner.
Clicking that only offered action could not work: the offer is computed from the
authoritative latest run, while the browser sent `session.binding.run_id`, which
still named the *reviewed candidate plan* run two dozen study revisions earlier.
Measured on the real product: the binding coordinate fails inside the job with
`research_pipeline_execution_retry_source_not_failed_execution`, and the failed
run's own coordinate is accepted and resumes from the digest-verified cohort.

The binding cannot simply be advanced: it is also the coordinate the review
submission approves. So the action owner carries the authority coordinate
instead, and this contract keeps it there.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness,  # noqa: F401  (autouse fixture)
    _read,
)


SCRIPT = r"""
  global.window = {};
  eval(__CONFIRMATION_SOURCE__);
  eval(__ACTIONS_SOURCE__);
  eval(__REPLAY_SOURCE__);
  const calls = [];
  let busy = false;
  const workflow = {
    next_action_code: 'failed_pipeline_execution_retry_available',
    plan_review_summary: null,
    latest_attempt_failure: null,
  };
  const host = {
    tr: (en, zh) => zh, esc: value => String(value), iconHtml: () => '',
    resourceButton: resource => `<button>${resource.label}</button>`,
    errorText: error => String(error), regeneration: {}, nextActions: {},
    replay: window.EU_GUIDED_PI_REPLAY,
    session: () => ({
      session_id: 'session', archived_child_jobs: [],
      binding: {run_id: 'reviewed-candidate-plan', study_context_id: 'study',
        study_revision: 22},
      research_provider: {provider: 'openai', credential_source: 'pi_verified'},
    }),
    workflow: () => workflow, busy: () => busy,
    latestRun: () => CASE === 'no-projection'
      ? {present: false}
      : {present: true, run_id: 'failed-approved-execution'},
    sessionIsStale: () => false,
    api: () => ({
      loadStudyContext: async id => ({context: {
        id, question: 'Unchanged question', data_source: {path: '/prepared/miiv'},
      }}),
      startAgentRun: async body => {calls.push(body); return {job_id: 'retry-job'};},
    }),
    projectId: () => 'project', turnGrants: () => [],
    setBusy: value => {busy = value;}, setError: value => calls.push(['error', value]),
    render: () => {}, recordHostAction: async (...args) => calls.push(['record', ...args]),
    watchChildJob: (...args) => calls.push(['watch', ...args]),
    appendMessage: message => calls.push(['message', message.text]),
    sendText: async (...args) => calls.push(['send', ...args]),
  };
  const confirmation = window.EU_GUIDED_PI_CONFIRMATION.create(host);
  const actions = window.EU_GUIDED_PI_PLAN_ACTIONS.create(host);
  (async () => {
    const spec = confirmation.workflowConfirmation();
    await actions.confirmWorkflow(spec);
    process.stdout.write(JSON.stringify({spec, calls}));
  })().catch(error => {console.error(error); process.exit(1);});
"""


def exercise(case: str = "projected") -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    script = SCRIPT.replace(
        "__CONFIRMATION_SOURCE__",
        json.dumps(_read("js/screens-guided-pi-confirmation.js")),
    )
    script = script.replace(
        "__ACTIONS_SOURCE__",
        json.dumps(_read("js/screens-guided-pi-plan-actions.js")),
    )
    script = script.replace(
        "__REPLAY_SOURCE__",
        json.dumps(_read("js/screens-guided-pi-replay.js")),
    )
    script = f"const CASE = {json.dumps(case)};\n" + script
    result = subprocess.run([node, "--eval", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def _launch(calls: list) -> dict:
    bodies = [call for call in calls if not isinstance(call, list)]
    assert len(bodies) == 1, calls
    return bodies[0]


def test_retry_of_a_failed_execution_resumes_the_failed_run_not_the_reviewed_plan() -> None:
    result = exercise()
    assert result["spec"]["code"] == "failed_pipeline_execution_retry_available"

    body = _launch(result["calls"])
    assert body["execution_resume_source_run_id"] == "failed-approved-execution"
    assert body["execution_resume_source_run_id"] != "reviewed-candidate-plan"
    # The governed fast path: reuse the approved plan, never re-plan, never
    # re-authorize a literature search behind the operator's back.
    assert body["planner_start_mode"] == "auto"
    assert body.get("report_only") is None
    assert body["external_llm_opt_in"] is True


def test_retry_falls_back_to_the_binding_only_when_no_run_is_projected() -> None:
    """An unloaded projection must not invent a coordinate; the server checks it."""

    body = _launch(exercise("no-projection")["calls"])
    assert body["execution_resume_source_run_id"] == "reviewed-candidate-plan"

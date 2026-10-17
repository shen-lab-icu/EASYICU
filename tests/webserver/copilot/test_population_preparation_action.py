"""An explicit population click changes settings once, without starting jobs."""

import json
import shutil
from pathlib import Path

import pytest

from tests.support.node import run_node


@pytest.mark.parametrize("mode", ["apply", "passive", "stale", "switch"])
@pytest.mark.parametrize("kind", ["preparation", "acceptance"])
def test_population_action_preserves_its_scope(mode, kind):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable")
    owner = (
        Path(__file__).parents[3]
        / "src/easyicu/webserver/static/js/screens-guided-pi-plan-actions.js"
    )
    program = r"""
let owner;
global.window = {EasyICU: {guidedPi: {declare: (name, api) => owner = api}}};
require(process.argv[1]);
const mode = process.argv[2];
const kind = process.argv[3];
const calls = [];
let busy = false;
let project = 'p';
let session = {session_id: 's', binding: {run_id: 'r', study_revision: 1}};
const workflow = {next_action_code: kind === 'acceptance' ? 'population_inclusion_not_applied' : 'population_inclusion_requires_extraction', plan_review_summary: {
  ['population_' + kind]: {available: true, run_id: 'r', study_revision: mode === 'stale' ? 2 : 1, proposal_sha256: 'a'.repeat(64), criteria: [{id: 'c1', accepted: false}]},
}};
const host = {
  tr: (en, zh) => zh, session: () => session, projectId: () => project,
  busy: () => busy, sessionIsStale: () => false, workflow: () => workflow,
  setBusy: value => {busy = value;}, setError: value => {if (value) calls.push(['error', value]);}, render: () => {},
  errorText: error => String(error),
  api: () => ({[kind === 'acceptance' ? 'acceptPiCopilotPopulationInclusion' : 'applyPiCopilotPopulationPreparation']: async (id, body) => {
    calls.push(['apply', id, body]);
    if (mode === 'switch') {project = 'other'; session = {session_id: 'other'};}
    return {next_action: 'request_extraction'};
  }}),
  refreshSession: async () => {calls.push(['refresh']); session.binding.study_revision = 2;},
  loadWorkflow: async () => {calls.push(['workflow']);},
  sendText: () => {throw new Error('must not send synthetic user consent');},
};
(async () => {
  const actions = owner.create(host);
  await actions.continueSystemOwnedPlanProgression({passive: true});
  if (mode !== 'passive') {
    await actions.confirmWorkflow({code: workflow.next_action_code});
    await actions.confirmWorkflow({code: workflow.next_action_code});
  }
  process.stdout.write(JSON.stringify(calls));
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    result = run_node(node, program, str(owner.resolve()), mode, kind, check=True)
    calls = json.loads(result.stdout)
    if mode in {"passive", "stale"}:
        assert calls == []
    elif mode == "switch":
        assert [row[0] for row in calls] == ["apply"]
    else:
        assert [row[0] for row in calls] == ["apply", "refresh", "workflow"]
        assert calls[0][2] == {
            **({"criterion_id": "c1"} if kind == "acceptance" else {}),
            "project_id": "p",
            "run_id": "r",
            "expected_revision": 1,
            "proposal_sha256": "a" * 64,
        }

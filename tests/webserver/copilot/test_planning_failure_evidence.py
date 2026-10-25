"""Early planning failures do not masquerade as completed candidate drafts."""

import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    STATIC,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
)


@pytest.mark.parametrize("history", [False, True])
def test_contract_failure_keeps_its_typed_reason_and_only_existing_materials(history):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable")
    script = r"""
global.window = {};
require(process.argv[3]);
require(process.argv[1]);
require(process.argv[4]);
const previous = process.argv[2] === 'true' ? [{
  kind: 'agent-run', status: 'done', run_id: 'run_previous', artifact_refs: [
    {artifact: 'agent_plan.json', run_id: 'run_previous'},
    {artifact: 'literature_evidence.json', run_id: 'run_previous'},
  ],
}] : [];
const seen = [];
const host = {
  tr: (en) => en,
  workflow: () => ({next_action_code: 'failed_pipeline_requires_fresh_plan', runs: [{
    authoritative: true, run_status: 'failed',
    gate_reason_code: 'research_pipeline_plan_contract_exhausted',
    gate_detail_code: 'exposure_grouping_unanswered',
    gate_detail_cause_code: 'exposure_grouping_planner',
  }]}),
  session: () => ({archived_child_jobs: [...previous, {
    kind: 'agent-run', status: 'failed',
    error_code: 'research_pipeline_plan_contract_exhausted',
  }]}),
  runFailureText: (code, detail) => {seen.push({code, detail}); return 'typed-owner-reason';},
};
const card = window.EU_GUIDED_PI_CONFIRMATION.create(host).workflowConfirmation();
const runStopHtml = window.EasyICU.guidedPi.require('runStop').create(host)
  .render({savedRuns: host.workflow().runs});
process.stdout.write(JSON.stringify({card, seen, runStopHtml}));
"""
    result = run_node(
        node, script, str(STATIC / "js/screens-guided-pi-confirmation.js"),
        str(history).lower(), str(STATIC / "js/html-escape.js"),
        str(STATIC / "js/screens-guided-pi-run-stop.js"), check=True,
    )
    result = json.loads(result.stdout)
    card = result["card"]
    assert card["reason"] == ""
    assert result["runStopHtml"].count("typed-owner-reason") == 1
    assert result["seen"][-1]["detail"]["code"] == "exposure_grouping_unanswered"
    assert card["grants"] == ["provider_run", "literature"]
    assert {row["run_id"] for row in card["reviewResources"]} == (
        {"run_previous"} if history else set()
    )
    # Neither a retry count nor a literature result exists in this receipt.
    assert "Four" not in card["note"] and "0 direct matches" not in card["note"]

"""Whole-plan revision draft stays separate from a setup questionnaire."""
import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import STATIC, _read


def test_plan_change_draft_requests_whole_plan_revision_without_setup_questionnaire() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    script = r"""
global.window = {};
const path = require('node:path');
const ownerDir = path.dirname(process.argv[1]);
require(path.join(ownerDir, 'screens-guided-pi-modules.js'));
require(path.join(ownerDir, 'screens-guided-pi-confirmation-data.js'));
require(process.argv[1]);
let workflow = {next_action_code: 'plan_execution_upgrade_required'};
const host = {
  tr: (en, zh) => zh,
  workflow: () => workflow,
};
const owner = window.EasyICU.guidedPi.require('confirmation').create(host);
const candidate = owner.planChangeDraft();
workflow = {next_action_code: 'operator_plan_approval_required'};
const executable = owner.planChangeDraft();
workflow = {
  next_action_code: 'plan_scientific_changes_required',
  plan_review_summary: {authorization_questions: [{question: 'Which external review is authorized?'}]},
};
const requiredDecision = owner.planChangeDraft();
process.stdout.write(JSON.stringify({candidate, executable, requiredDecision}));
"""
    completed = run_node(
        node,
        script,
        str(STATIC / "js" / "screens-guided-pi-confirmation.js"),
        check=True,
    )
    payload = json.loads(completed.stdout)
    for key in ("candidate", "executable"):
        assert "修订整份研究计划" in payload[key]
        assert "不要开始分析" in payload[key]
        assert "一次只问" not in payload[key]
    assert payload["requiredDecision"] == "Which external review is authorized?"
    shell = _read("js/screens-guided-pi.js")
    edit = shell.split("function editWorkflow()", 1)[1].split("function studySetupReviewPrompt", 1)[0]
    assert "state.draft = CONFIRMATION.planChangeDraft();" in edit

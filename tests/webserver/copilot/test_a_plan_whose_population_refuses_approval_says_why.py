"""A plan whose cohort leaves out a stated inclusion says why, and what to do.

The planning owner refuses approval with one code per remedy
(``population_compile.POPULATION_APPROVAL_STOPS``).  When nothing applies the
criterion as written, the plan card offers a fresh candidate plan, which the
host keeps metadata-only from that state.  When an extraction of the study's
own population would apply it, the card offers no plan action: the
conversation asks for the extraction, and binding it supersedes the plan.
The workflow aside names both.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from easyicu.research_agent.planning.population_compile import POPULATION_APPROVAL_STOPS
from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"
NOT_APPLIED = POPULATION_APPROVAL_STOPS["not_applied"]
REQUIRES_EXTRACTION = POPULATION_APPROVAL_STOPS["requires_extraction"]


def _card(code: str, *, run_id: str = "run_population_1") -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = r"""
let declared = null;
global.window = {
  EU_LANG: 'zh',
  EasyICU: { guidedPi: { declare: (_name, api) => { declared = api; }, optional: () => null } },
};
require(process.argv[1]);
const code = process.argv[2];
const runId = process.argv[3];
const owner = declared.create({
  tr: (en, zh) => zh || en,
  esc: value => String(value),
  iconHtml: () => '',
  resourceButton: resource => `<a data-resource="${resource.artifact}"></a>`,
  sessionIsStale: () => false,
  busy: () => false,
  session: () => ({ binding: { run_id: runId } }),
  workflow: () => ({ next_action_code: code, plan_review_summary: { run_id: runId } }),
});
const confirmation = owner.workflowConfirmation();
const html = owner.workflowConfirmationHtml();
process.stdout.write(JSON.stringify({
  code: confirmation && confirmation.code,
  title: confirmation && confirmation.title,
  nonApprovable: Boolean(confirmation && confirmation.nonApprovable),
  grants: confirmation && confirmation.grants,
  execute: html.includes('data-gpi-confirm-action'),
  edit: html.includes('data-gpi-confirm-edit'),
  resources: (html.match(/data-resource="([^"]+)"/g) || []).map(row => row.slice(15, -1)),
  offeredResources: ((confirmation && confirmation.reviewResources) || []).length,
}));
"""
    owner = STATIC / "js" / "screens-guided-pi-confirmation.js"
    result = run_node(node, script, str(owner.resolve()), code, run_id, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def test_a_criterion_nothing_applies_offers_a_fresh_candidate_plan() -> None:
    card = _card(NOT_APPLIED)

    assert card["code"] == NOT_APPLIED
    assert card["title"] == "研究写明的纳入条件按原文无法施加"
    assert (card["nonApprovable"], card["execute"], card["edit"]) == (False, True, True)
    assert card["grants"] == ["provider_run", "literature"]
    assert card["resources"] == ["agent_plan.json", "scientific_plan_review.json"]


def test_a_criterion_an_extraction_applies_offers_no_plan_action() -> None:
    card = _card(REQUIRES_EXTRACTION)

    assert card["code"] == REQUIRES_EXTRACTION
    assert card["title"] == "研究写明的纳入条件需要按本研究人群重新提取数据"
    assert (card["nonApprovable"], card["execute"], card["grants"]) == (True, False, [])
    assert card["resources"] == ["agent_plan.json", "scientific_plan_review.json"]


def test_without_a_reviewed_plan_the_card_names_no_resource() -> None:
    for code in (NOT_APPLIED, REQUIRES_EXTRACTION):
        card = _card(code, run_id="")
        assert (card["resources"], card["offeredResources"]) == ([], 0)


def test_only_the_criterion_nothing_applies_starts_a_fresh_plan() -> None:
    actions = (STATIC / "js" / "screens-guided-pi-plan-actions.js").read_text(encoding="utf-8")
    fresh = re.search(r"const FRESH_PLAN_CODES = new Set\(\[(.*?)\]\);", actions, re.S)
    replay = re.search(r"const FRESH_REPLAY_CODES = new Set\(\[(.*?)\]\);", actions, re.S)

    assert fresh and replay
    assert f"'{NOT_APPLIED}'" in fresh.group(1)
    assert f"'{REQUIRES_EXTRACTION}'" not in fresh.group(1)
    assert REQUIRES_EXTRACTION not in replay.group(1)
    # The host keeps a plan generated from either state metadata-only.
    assert set(POPULATION_APPROVAL_STOPS.values()) <= agent_routes._CANDIDATE_PLAN_WORKFLOW_CODES


@pytest.mark.parametrize(
    ("code", "zh"),
    [
        (REQUIRES_EXTRACTION, "按本研究人群重新提取数据后再生成计划，此计划不能批准"),
        (NOT_APPLIED, "按原文无法施加，此计划不能批准；请修改该条件或重新生成计划"),
    ],
)
def test_the_workflow_aside_names_each_stop(code: str, zh: str) -> None:
    aside = (STATIC / "js" / "screens-guided-pi-aside.js").read_text(encoding="utf-8")

    (entry,) = [line for line in aside.splitlines() if line.strip().startswith(f"{code}: tr(")]
    assert zh in entry
    assert "cannot be approved" in entry

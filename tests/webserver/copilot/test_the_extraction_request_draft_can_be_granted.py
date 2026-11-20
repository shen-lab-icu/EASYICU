"""The extraction request a card drafts is one the researcher's turn can grant.

B19: the raw-source plan card (B18) and the bound-export mismatch card offer to
ask for this study's extraction in the conversation.  Extraction is a one-use
privileged action granted only from the researcher's own sent words
(``turn_authority.infer_explicit_turn_actions``: a confirmation and an
extraction scope in the same turn).  The draft used to ask only for the
requirements, so no sent draft could ever start the extraction.  The card
still only drafts the text; the grant comes from the turn the researcher sends.
Synthetic workflows only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.turn_authority import infer_explicit_turn_actions
from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"

_CARDS = {
    "bound_export_cohort_mismatch": {
        "next_action_code": "bound_export_cohort_mismatch"
    },
    "raw_source_plan": {
        "next_action_code": "plan_execution_upgrade_required",
        "execution_input_available": False,
    },
}


def _draft(workflow: dict, lang: str) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = r"""
const lang = process.argv[3];
global.window = { EU_LANG: lang };
const path = require('node:path');
const ownerDir = path.dirname(process.argv[1]);
require(path.join(ownerDir, 'screens-guided-pi-modules.js'));
require(path.join(ownerDir, 'screens-guided-pi-confirmation-data.js'));
require(process.argv[1]);
const workflow = JSON.parse(process.argv[2]);
const owner = window.EasyICU.guidedPi.require('confirmation').create({
  tr: (en, zh) => (lang === 'zh' ? zh || en : en),
  esc: value => String(value),
  iconHtml: () => '',
  resourceButton: () => '',
  sessionIsStale: () => false,
  busy: () => false,
  session: () => ({}),
  workflow: () => workflow,
});
process.stdout.write(owner.planChangeDraft());
"""
    owner = STATIC / "js" / "screens-guided-pi-confirmation.js"
    result = run_node(
        node, script, str(owner.resolve()), json.dumps(workflow), lang, check=False
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("card", sorted(_CARDS))
def test_the_drafted_extraction_request_is_grantable_when_sent(
    card: str, lang: str
) -> None:
    draft = _draft(_CARDS[card], lang)

    assert draft.strip()
    assert infer_explicit_turn_actions(draft) == frozenset({"extract"})


@pytest.mark.parametrize(
    "text",
    [
        # The earlier drafts asked only for the requirements: never a grant.
        "我想按本研究现有的人群条件准备数据，请列出提取要求。",
        "I want to prepare data for this study’s current cohort. Please show the extraction requirements.",
        # Asking about it, denying it in the same clause, or confirming without an
        # extraction scope is not a grant either.
        "我确认这个研究问题。",
        "Should I confirm data extraction for this study?",
        "不要把这句话当成我确认进行数据提取。",
        "Do not read this as: I confirm data extraction for this study.",
    ],
)
def test_a_request_that_does_not_confirm_extraction_is_not_granted(text: str) -> None:
    assert "extract" not in infer_explicit_turn_actions(text)


@pytest.mark.parametrize(
    "code",
    [
        "plan_execution_upgrade_required",
        "operator_plan_approval_required",
        "planner_checkpoint_resume_available",
        "plan_scientific_changes_required",
    ],
)
@pytest.mark.parametrize("available", [True, False, None])
def test_the_card_and_the_host_trigger_read_the_same_raw_source_state(
    code: str, available
) -> None:
    # B19 routing: the host starts a raw Demo study's extraction only in the
    # state where this card drafts the request. One Python predicate owns the
    # state; the card's condition must give the same verdict on the same input.
    from types import SimpleNamespace

    from easyicu.webserver.pi_copilot.host_study_extraction import (
        raw_source_extraction_state,
    )

    workflow = {"next_action_code": code}
    if available is not None:
        workflow["execution_input_available"] = available
    raw_draft = _draft(_CARDS["raw_source_plan"], "en")

    card_drafts_extraction = _draft(workflow, "en") == raw_draft
    host_state = raw_source_extraction_state(
        SimpleNamespace(next_action_code=code, execution_input_available=available)
    )

    assert card_drafts_extraction is host_state

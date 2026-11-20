"""A plan card that cannot be approved always offers an enabled forward step.

S1570/S1571 guard: when the confirmation owner marks a card non-approvable,
at least one enabled control must move the study forward (approve or apply,
generate a fresh plan, a decision option, a projected stop choice, a review
recovery, or editing the plan or a condition in the conversation). A card
whose only controls are disabled or view-only is a dead end. Every workflow
code the confirmation owner names is rendered on its own and with the
population review payloads it can carry.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from easyicu.research_agent.planning.approval_stops import PLAN_APPROVAL_STOPS
from easyicu.research_agent.planning.population_compile import POPULATION_APPROVAL_STOPS
from easyicu.webserver import stop_next_actions
from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"
OWNER = STATIC / "js" / "screens-guided-pi-confirmation.js"
FORWARD = (
    "data-gpi-confirm-action", "data-gpi-confirm-edit", "data-gpi-stop-action",
    "data-gpi-plan-decision-option", "data-gpi-review-recovery", "data-project-new",
)
_PREPARATION_REASONS = (
    "population_preparation_first_stay_scope_conflict",
    "population_preparation_all_stays_conflict",
    "population_preparation_patient_hospitalization_scope_mismatch",
    "population_preparation_eicu_demo_first_observed_unavailable",
    "population_preparation_kind_unsupported",
    "population_inclusion_not_applied",
)


def _codes() -> list[str]:
    named = set(re.findall(r"code === '([a-z_0-9]+)'", OWNER.read_text(encoding="utf-8")))
    return sorted(named | set(PLAN_APPROVAL_STOPS) | set(POPULATION_APPROVAL_STOPS.values())
                  | set(stop_next_actions.STOP_POLICIES))


def _criterion(reason=None, *, accepted=False):
    return {"id": "c1", "quote": "synthetic stated condition", "proposed_by_system": False,
            "disposition": "requires_extraction" if reason != "population_inclusion_not_applied" else "not_applied",
            "reason": reason, "accepted": accepted}


def _variants() -> list[tuple[str, dict]]:
    variants = [("empty", {})]
    variants += [
        (f"preparation:{reason}", {"population_preparation": {
            "available": False, "status": "blocked", "criteria": [_criterion(reason)], "blocker_codes": [reason]}})
        for reason in _PREPARATION_REASONS
    ]
    variants.append(("preparation:ready", {"population_preparation": {
        "available": True, "status": "ready", "criteria": [_criterion()], "proposal_sha256": "f" * 64,
        "changes": [{"field": "exclude_readmissions", "before": False, "after": True}]}}))
    variants.append(("preparation:already_configured", {"population_preparation": {
        "available": False, "status": "already_configured", "criteria": [_criterion()], "changes": []}}))
    for name, acceptance in (
        ("pending", {"available": True, "complete": False, "record_invalid": False,
                     "criteria": [_criterion("population_inclusion_not_applied")]}),
        ("complete", {"available": False, "complete": True, "record_invalid": False,
                      "criteria": [_criterion("population_inclusion_not_applied", accepted=True)]}),
        ("invalid", {"available": False, "complete": False, "record_invalid": True, "criteria": []}),
    ):
        variants.append((f"acceptance:{name}", {"population_acceptance": acceptance}))
    # The review states a plan review card can hold.
    questions = [{"code": "OUTCOME_DEFINITION_UNRESOLVED", "question": "Which endpoint?"}]
    blockers = [{"code": "plan_revision_owner_resolution_required", "message": "An owner must resolve it."}]
    repairs = {"agent_plan_revision": [{"finding": "synthetic repair"}]}
    for name, review in (
        ("review:repairs", {"remediation_buckets": repairs, "authorization_questions": [], "automatic_revision_blockers": []}),
        ("review:repairs+questions", {"remediation_buckets": repairs, "authorization_questions": questions, "automatic_revision_blockers": []}),
        ("review:repairs+blockers", {"remediation_buckets": repairs, "authorization_questions": [], "automatic_revision_blockers": blockers}),
        ("review:blockers", {"remediation_buckets": {}, "authorization_questions": [], "automatic_revision_blockers": blockers}),
        ("review:questions", {"remediation_buckets": {}, "authorization_questions": questions, "automatic_revision_blockers": []}),
        ("review:nothing", {"remediation_buckets": {}, "authorization_questions": [], "automatic_revision_blockers": []}),
    ):
        variants.append((name, review))
    return variants


_SCRIPT = r"""
const path = require('node:path');
const ownerDir = path.dirname(process.argv[1]);
const cases = JSON.parse(require('node:fs').readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const language of ['zh', 'en']) {
  global.window = { EU_LANG: language };
  for (const key of Object.keys(require.cache)) delete require.cache[key];
  require(path.join(ownerDir, 'screens-guided-pi-modules.js'));
  // The modules the live page loads beside the confirmation owner, as the
  // stop catalog (tests/js/guided_pi_stop_buttons.test.js) loads them.
  for (const name of ['html-escape.js', 'product-labels.js', 'screens-guided-pi-confirmation-data.js',
    'screens-guided-pi-target-trial.js', 'screens-guided-pi-run-stop.js', 'screens-guided-pi-events.js',
    'screens-guided-pi-run-outcome.js', 'screens-guided-pi-plan-actions.js', 'screens-guided-pi-error-text.js',
    'screens-guided-pi-provider-control.js']) require(path.join(ownerDir, name));
  require(process.argv[1]);
  for (const item of cases) {
    const owner = window.EasyICU.guidedPi.require('confirmation').create({
      tr: (en, zh) => language === 'en' ? en : zh || en,
      esc: value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
      iconHtml: () => '', resourceButton: resource => `<a data-resource="${resource.artifact}"></a>`,
      sessionIsStale: () => false, busy: () => false,
      session: () => ({ binding: { run_id: 'run_guard_1' } }),
      workflow: () => ({ next_action_code: item.code, plan_review_summary: { run_id: 'run_guard_1', ...item.summary } }),
    });
    let confirmation; let html;
    try { confirmation = owner.workflowConfirmation(); html = owner.workflowConfirmationHtml(); }
    catch (error) { out.push({ ...item, language, error: String(error && error.stack || error) }); continue; }
    const buttons = [...html.matchAll(/<button[^>]*>/g)].map(match => match[0]);
    out.push({ code: item.code, variant: item.variant, language,
      nonApprovable: Boolean(confirmation && confirmation.nonApprovable), buttons });
  }
}
process.stdout.write(JSON.stringify(out));
"""


def _render(tmp_path: Path) -> list[dict]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    cases = [{"code": code, "variant": name, "summary": summary}
             for code in _codes() for name, summary in _variants()]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    result = run_node(node, _SCRIPT, str(OWNER.resolve()), str(cases_path), check=False, timeout=120)
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def test_the_guard_reads_the_population_stops_and_the_owners_codes() -> None:
    codes = _codes()
    assert set(POPULATION_APPROVAL_STOPS.values()) <= set(codes)
    assert "plan_scientific_changes_required" in codes and len(codes) > 30


#: A progress state, not a decision: the host is already starting a decision
#: the researcher took, and the card follows it.
TRANSIENT = frozenset({"starting"})
#: Dead states the guard found on 2026-11-20, reported to the supervisor
#: (S1571) for a ruling; the strict xfail below flips when they are fixed.
#: - plan_scientific_changes_required: a review holding a blocker no plan can
#:   close (automatic_revision_blockers) and no question, or neither repairs
#:   nor questions, renders no control.
#: - agent_plan_revision_nonconvergent: when its one retry is not offered
#:   (questions, blockers or no repairs), it renders no control.
PENDING_RULING = frozenset({"plan_scientific_changes_required", "agent_plan_revision_nonconvergent"})


def _dead(rows) -> dict[str, set[str]]:
    dead: dict[str, set[str]] = {}
    for row in rows:
        if row["nonApprovable"] and not any(
            attribute in button and " disabled" not in button
            for button in row["buttons"] for attribute in FORWARD
        ):
            dead.setdefault(row["code"], set()).add(f"{row['variant']}/{row['language']}")
    return dead


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> list[dict]:
    rows = _render(tmp_path_factory.mktemp("guard"))
    assert not [row for row in rows if "error" in row], [row for row in rows if "error" in row][:3]
    assert any(row["nonApprovable"] for row in rows), "the guard must see unapprovable cards"
    return rows


def test_every_unapprovable_card_offers_an_enabled_forward_step(rendered) -> None:
    dead = {code: states for code, states in _dead(rendered).items()
            if code not in TRANSIENT | PENDING_RULING}
    assert not dead, f"unapprovable cards without an enabled forward step: {dead}"


@pytest.mark.xfail(strict=True, reason="S1571: dead review states reported for a ruling")
def test_the_review_stops_pending_a_ruling_offer_a_forward_step(rendered) -> None:
    dead = {code: states for code, states in _dead(rendered).items() if code in PENDING_RULING}
    assert not dead, f"review stops without an enabled forward step: {dead}"


def test_a_resolved_review_stop_still_offers_its_step(rendered) -> None:
    # The pending stops are dead only in the states named above.
    live = {(row["code"], row["variant"]) for row in rendered if row["language"] == "zh"
            and any(attribute in button and " disabled" not in button
                    for button in row["buttons"] for attribute in FORWARD)}
    assert ("agent_plan_revision_nonconvergent", "review:repairs") in live
    assert ("plan_scientific_changes_required", "review:questions") in live


def test_a_card_whose_only_control_is_disabled_is_caught() -> None:
    # The guard's own predicate: a disabled forward control does not count.
    buttons = ['<button class="btn primary sm" type="button" data-gpi-confirm-action disabled>']
    assert not any(attribute in button and " disabled" not in button
                   for button in buttons for attribute in FORWARD)

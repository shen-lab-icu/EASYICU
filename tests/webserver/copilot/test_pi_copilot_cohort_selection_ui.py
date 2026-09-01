"""Frontend ownership contracts for host-confirmed primary cohorts."""

import json
from pathlib import Path
import shutil
import subprocess


STATIC = Path("src/easyicu/webserver/static")


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_cohort_selection_owner_is_wired_before_guided_pi_shell() -> None:
    index = _read("index.html")

    assert "css/guided-pi-cohort-eligibility.css" in index
    assert "js/screens-guided-pi-cohort-eligibility.js" in index
    assert "js/screens-guided-pi-cohort-eligibility.js?v=20260831-simple-choice1" in index
    assert index.index("js/screens-guided-pi-cohort-eligibility.js") < index.index(
        "js/screens-guided-pi.js"
    )


def test_cohort_selection_renders_exact_server_contracts() -> None:
    owner = _read("js/screens-guided-pi-cohort-eligibility.js")
    shell = _read("js/screens-guided-pi.js")
    api = _read("js/api.js")

    assert "option.primary_cohort_contract" in owner
    assert 'data-gpi-cohort-event="${esc(option.selection_event_id)}"' in owner
    assert "actionFromEvent" in owner
    assert "confirmPiCopilotCohortEligibility" in shell
    assert "confirmPiCopilotCohortEligibility" in api
    assert "/cohort-eligibility-selection" in api
    assert "message.includes" not in owner
    assert "message.match" not in owner


def test_repeated_stay_review_uses_typed_plan_decision_not_chat_text() -> None:
    node = shutil.which("node")
    if node is None:
        return
    cohort_owner = STATIC / "js" / "screens-guided-pi-cohort-eligibility.js"
    confirmation_owner = STATIC / "js" / "screens-guided-pi-confirmation.js"
    script = r"""
global.window = { EU_LANG: 'zh' };
require(process.argv[1]);
require(process.argv[2]);
const options = [
  { id: 'first_admission_only', expected_revision: 29,
    primary_cohort_contract_sha256: 'first-scope', selection_event_id: 'first-event' },
  { id: 'no_eligibility_filter', expected_revision: 29,
    primary_cohort_contract_sha256: 'all-scope', selection_event_id: 'all-event' },
];
const session = { cohort_eligibility_selection: {
  present: true, stated: false,
  primary_cohort_contract: { admission_eligibility: {
    minimum_age_years: 0, minimum_icu_duration_hours: 0,
  }}, options,
}};
const cohort = window.EU_GUIDED_PI_COHORT_ELIGIBILITY.create({
  tr: (en, zh) => zh || en, esc: value => String(value),
  session: () => session, busy: () => false, sessionIsStale: () => false,
});
const confirmation = window.EU_GUIDED_PI_CONFIRMATION.create({
  tr: (en, zh) => zh || en, esc: value => String(value), iconHtml: () => '',
  resourceButton: () => '', sessionIsStale: () => false, busy: () => false,
  session: () => ({ binding: { run_id: 'run-1' } }),
  workflow: () => ({
    next_action_code: 'plan_scientific_changes_required',
    plan_review_summary: { authorization_questions: [{
      code: 'REPEATED_STAY_IDENTITY_UNAVAILABLE',
    }]},
  }),
  cohortEligibilityDecisionHtml: copies => cohort.repeatedStayDecisionHtml(copies),
});
const html = confirmation.workflowConfirmationHtml();
process.stdout.write(JSON.stringify({
  clustered: html.includes('data-gpi-plan-decision-option="all_icu_stays_clustered"'),
  code: html.includes('data-gpi-plan-decision-code="REPEATED_STAY_IDENTITY_UNAVAILABLE"'),
  oldCohortSelector: html.includes('data-gpi-cohort-option='),
  chatFallback: html.includes('data-gpi-next-choice='),
}));
"""
    result = subprocess.run(
        [
            node,
            "-e",
            script,
            str(cohort_owner.resolve()),
            str(confirmation_owner.resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "clustered": True,
        "code": True,
        "oldCohortSelector": False,
        "chatFallback": False,
    }


def test_cohort_selection_only_replaces_a_plan_stage_blocker() -> None:
    shell = _read("js/screens-guided-pi.js")
    owner = _read("js/screens-guided-pi-cohort-eligibility.js")

    # The Planner proposes inclusion/exclusion and the analysis unit in the
    # candidate plan.  The selector appears only after the host reports the
    # exact missing authority, and it replaces rather than duplicates the
    # stale workflow card.
    assert "COHORT_ELIGIBILITY.render() || workflowConfirmationHtml()" in shell
    assert "value.blocker_code !== 'cohort_eligibility_confirmation_required'" in owner
    assert "planNeedsThisDecision" in owner


def test_cohort_selection_css_and_event_marker_stay_route_owned() -> None:
    css_owner = STATIC / "css" / "guided-pi-cohort-eligibility.css"
    js_owner = STATIC / "js" / "screens-guided-pi-cohort-eligibility.js"

    assert ".gpi-cohort-eligibility" in css_owner.read_text(encoding="utf-8")
    assert "data-gpi-cohort-option" in js_owner.read_text(encoding="utf-8")
    for path in (STATIC / "css").glob("*.css"):
        if path != css_owner:
            assert ".gpi-cohort-eligibility" not in path.read_text(encoding="utf-8")
    for path in (STATIC / "js").glob("*.js"):
        if path != js_owner:
            assert "data-gpi-cohort-option" not in path.read_text(encoding="utf-8")

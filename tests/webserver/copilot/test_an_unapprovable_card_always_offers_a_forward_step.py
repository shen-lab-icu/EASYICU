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
      iconHtml: () => '', resourceButton: resource => `<a data-resource="${resource.artifact}"></a>`, runFailureText: code => code ? `REASON:${code}.` : '',
      sessionIsStale: () => false, busy: () => false,
      session: () => ({ binding: { run_id: 'run_guard_1' }, ...(item.session || {}) }),
      workflow: () => ({ next_action_code: item.code, ...(item.workflow || {}), plan_review_summary: { run_id: 'run_guard_1', ...item.summary } }),
    });
    let confirmation; let html;
    try { confirmation = owner.workflowConfirmation(); html = owner.workflowConfirmationHtml(); }
    catch (error) { out.push({ ...item, language, error: String(error && error.stack || error) }); continue; }
    const buttons = [...html.matchAll(/<button[^>]*>/g)].map(match => match[0]);
    out.push({ code: item.code, variant: item.variant, language,
      nonApprovable: Boolean(confirmation && confirmation.nonApprovable), buttons, html });
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
    # B18: a candidate whose source has no extracted package (raw files).
    cases += [{"code": code, "variant": "source:no_package", "summary": {},
               "workflow": {"execution_input_available": False}} for code in _codes()]
    # S1574: the latest preparation read no data (metadata only, null hashes).
    cases += [{"code": code, "variant": f"prepared_nothing:{name}", "summary": {},
               "workflow": {"preparation_read_no_data": True, "execution_input_available": available}}
              for code in _codes() for name, available in (("no_package", False), ("package", True))]
    # S1577 (m35): the conversation's preparation of this candidate already
    # ended (here: failed in seconds before writing a run), yet the workflow
    # offers the same step again.
    # (variant, turn status, turn candidate, job run id, extra workflow, summary)
    wrote_run = {"latest_attempt_failure": {"run_id": "run_attempt_1", "candidate_run_id": "run_guard_1",
                                            "reason": "preparation_failed", "checkpoint_resume_available": False}}
    for label, status, candidate, run_id, extra, summary in (
        ("failed", "failed", "run_guard_1", None, {}, {}),
        ("done", "done", "run_guard_1", None, {}, {}),
        ("interrupted", "interrupted", "run_guard_1", None, {}, {}),
        ("running", "running", "run_guard_1", None, {}, {}),
        ("other", "failed", "run_other_1", None, {}, {}),
        ("wrote_run", "failed", "run_guard_1", "run_attempt_1", wrote_run, {}),
        ("no_candidate", "failed", "", None, {}, {"run_id": ""}),
    ):
        session = {"conversation_replay": {"turns": [{
            "kind": "host_action", "action_code": "prepare_analysis_data", "status": status, "child_job_id": "job_guard_1",
            "action_key": f"plan:{'a' * 16}:{candidate}:plan_execution_upgrade_required"}]},
            "archived_child_jobs": [{"job_id": "job_guard_1", "kind": "agent-run", "run_id": run_id,
                                     "status": {"running": "running", "done": "done"}.get(status, "failed"),
                                     "error_code": None if status in {"running", "done"} else "codex_auth_app_server_request_failed"}]}
        for name, available in (("no_package", False), ("package", True)):
            cases += [{"code": code, "variant": f"turn:{label}:{name}", "summary": summary,
                       "workflow": {"execution_input_available": available, **extra}, "session": session}
                      for code in _codes()]
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
    dead = {code: states for code, states in _dead(rendered).items() if code not in TRANSIENT}
    assert not dead, f"unapprovable cards without an enabled forward step: {dead}"


_RETURN = {"zh": "提出修改（回到对话）", "en": "Request changes (back to the conversation)"}
_WHY = {
    "plan_scientific_changes_required": {"zh": "无法通过自动修订计划解决", "en": "No automatic plan revision can resolve"},
    "agent_plan_revision_nonconvergent": {"zh": "自动修订已停止，这里无法再次发起", "en": "Automatic revision has stopped and cannot run again"},
}


@pytest.mark.parametrize("code, variant", [
    ("plan_scientific_changes_required", "review:blockers"),
    ("plan_scientific_changes_required", "review:repairs+blockers"),
    ("plan_scientific_changes_required", "review:nothing"),
    ("agent_plan_revision_nonconvergent", "review:blockers"),
    ("agent_plan_revision_nonconvergent", "review:questions"),
    ("agent_plan_revision_nonconvergent", "review:nothing"),
])
def test_a_review_no_revision_can_close_returns_to_the_conversation(rendered, code, variant) -> None:
    # S1572: these states had no control. They now say why in one sentence and
    # offer only the conversation, never a revision that would fail silently.
    for row in (r for r in rendered if r["code"] == code and r["variant"] == variant):
        language = row["language"]
        enabled = [button for button in row["buttons"] if " disabled" not in button]
        assert len(enabled) == 1 and "data-gpi-confirm-edit" in enabled[0], (language, enabled)
        assert not any("data-gpi-confirm-action" in button for button in enabled)
        assert _RETURN[language] in row["html"] and _WHY[code][language] in row["html"]


def test_states_that_offer_their_own_step_keep_it(rendered) -> None:
    rows = {(r["code"], r["variant"]): r for r in rendered if r["language"] == "zh"}
    retry = rows[("agent_plan_revision_nonconvergent", "review:repairs")]
    assert any("data-gpi-confirm-action" in b and " disabled" not in b for b in retry["buttons"])
    assert _RETURN["zh"] not in retry["html"]
    revise = rows[("plan_scientific_changes_required", "review:repairs")]
    assert not revise["nonApprovable"] and _RETURN["zh"] not in revise["html"]
    question = rows[("plan_scientific_changes_required", "review:questions")]
    assert "回答这个问题" in question["html"] and _RETURN["zh"] not in question["html"]


def test_a_card_whose_only_control_is_disabled_is_caught() -> None:
    # The guard's own predicate: a disabled forward control does not count.
    buttons = ['<button class="btn primary sm" type="button" data-gpi-confirm-action disabled>']
    assert not any(attribute in button and " disabled" not in button
                   for button in buttons for attribute in FORWARD)


_EXTRACT = {"zh": "在对话中请求提取本研究数据", "en": "Ask to extract this study’s data"}
_PREPARED = {"zh": ("请确认数据准备", "确认方案并准备数据", "重新准备数据"),
             "en": ("confirm data preparation", "Confirm plan and prepare data", "Prepare data again")}


def test_a_candidate_whose_source_has_no_package_never_offers_preparation(rendered) -> None:
    # S1573/B18: preparing from raw files yields no data and the run only
    # plans on metadata again. The card must not offer preparation as if it
    # would succeed; it says why and offers the one step that works.
    rows = [r for r in rendered if r["code"] == "plan_execution_upgrade_required"
            and r["variant"] == "source:no_package"]
    assert {r["language"] for r in rows} == {"zh", "en"}
    for row in rows:
        language = row["language"]
        enabled = [button for button in row["buttons"] if " disabled" not in button]
        assert row["nonApprovable"]
        assert len(enabled) == 1 and "data-gpi-confirm-edit" in enabled[0], (language, enabled)
        assert _EXTRACT[language] in row["html"]
        assert not any(phrase in row["html"] for phrase in _PREPARED[language]), language


def test_a_candidate_whose_source_has_a_package_still_offers_preparation(rendered) -> None:
    for row in (r for r in rendered if r["code"] == "plan_execution_upgrade_required" and r["variant"] == "empty"):
        assert any("data-gpi-confirm-action" in button and " disabled" not in button for button in row["buttons"])
        assert _EXTRACT[row["language"]] not in row["html"]


_FIRST_TIME = {"zh": "候选研究计划已生成，请确认数据准备", "en": "Candidate plan ready; confirm data preparation next"}
_NO_DATA = {"zh": "上次数据准备没有读到数据", "en": "The last data preparation read no data"}


@pytest.mark.parametrize("variant", ["prepared_nothing:no_package", "prepared_nothing:package"])
def test_a_preparation_that_read_no_data_never_returns_the_same_card(rendered, variant) -> None:
    # S1574 invariant: a preparation run that ends asking for preparation again
    # built no cohort. It is not a success, and the card must change: it names
    # the empty preparation and offers extraction (no package) or preparing
    # again from the package now present, never the first-time card unchanged.
    rows = [r for r in rendered if r["code"] == "plan_execution_upgrade_required" and r["variant"] == variant]
    assert {r["language"] for r in rows} == {"zh", "en"}
    for row in rows:
        language = row["language"]
        assert _FIRST_TIME[language] not in row["html"], language
        enabled = [button for button in row["buttons"] if " disabled" not in button]
        assert len(enabled) >= 1, language
        if variant.endswith("no_package"):
            assert _NO_DATA[language] in row["html"] and _EXTRACT[language] in row["html"]
            assert not any("data-gpi-confirm-action" in button for button in enabled)
        else:
            assert ("The last preparation read no data" if language == "en" else "上次数据准备没有读到数据") in row["html"]
            assert any("data-gpi-confirm-action" in button for button in enabled)


_ENDED = {"zh": "上次数据准备未完成", "en": "The last data preparation did not finish"}
_RAW_FIRST = {"zh": "候选研究计划已生成，需先提取本研究数据", "en": "Candidate plan ready; this study’s data must be extracted first"}
_PLAN_UNFINISHED = {"zh": "这次计划生成未完成", "en": "Plan generation did not finish"}
_REASON = "REASON:codex_auth_app_server_request_failed."


def _rows(rendered, variant):
    rows = [r for r in rendered if r["code"] == "plan_execution_upgrade_required" and r["variant"] == variant]
    assert {r["language"] for r in rows} == {"zh", "en"}, variant
    return rows


@pytest.mark.parametrize("label", ["failed", "done", "interrupted"])
def test_a_preparation_that_ended_without_a_run_never_returns_the_same_card(rendered, label) -> None:
    # S1577: m35 failed in two seconds (a transient model-service sign-in error)
    # before writing a run, and the card came back unchanged with no failure
    # line. Any ended, runless preparation of the shown candidate changes it.
    for name in ("no_package", "package"):
        for row in _rows(rendered, f"turn:{label}:{name}"):
            language, html = row["language"], row["html"]
            assert _FIRST_TIME[language] not in html and _RAW_FIRST[language] not in html, (name, language)
            # The job's failure is named; a preparation that finished has none.
            assert (_REASON in html) is (label != "done" and name == "package"), (label, name, language)
            enabled = [button for button in row["buttons"] if " disabled" not in button]
            if name == "package":
                assert _ENDED[language] in html and _PREPARED[language][2] in html
                assert sum("data-gpi-confirm-action" in button for button in enabled) == 1
            else:
                assert _NO_DATA[language] in html and _EXTRACT[language] in html
                assert len(enabled) == 1 and "data-gpi-confirm-edit" in enabled[0]


@pytest.mark.parametrize("label", ["running", "other", "no_candidate"])
def test_a_running_unrelated_or_unnamed_preparation_leaves_the_card(rendered, label) -> None:
    for name, first in (("package", _FIRST_TIME), ("no_package", _RAW_FIRST)):
        for row in _rows(rendered, f"turn:{label}:{name}"):
            if label == "no_candidate" and name == "package":
                continue  # no candidate run: the card has nothing to name
            assert first[row["language"]] in row["html"], (label, name)
            assert _ENDED[row["language"]] not in row["html"] and _NO_DATA[row["language"]] not in row["html"]


def test_a_preparation_that_wrote_a_run_keeps_that_runs_failure(rendered) -> None:
    # A preparation that wrote a run and then failed planning is reported by
    # that run's attempt ("plan generation did not finish"), not as runless.
    for row in _rows(rendered, "turn:wrote_run:package"):
        assert _PLAN_UNFINISHED[row["language"]] in row["html"] and _ENDED[row["language"]] not in row["html"]

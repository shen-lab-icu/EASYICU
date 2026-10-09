"""A causal study's target trial is approved on its card, line by line.

The host computes the card (``target_trial_card``) and names the study's step
as the workflow's next action.  The plan card shows it: what the compile is
doing, why it stopped, what holds approval, and, for an approvable record, its
protocol and each line to confirm.  The card approves only once every line is
ticked, and the request names the record the card shows, which the host then
approves.  A refused click reads as its line; an approved trial generates its
plan on the study's data, through the ordinary plan request.  The compile
job's rows follow the job, and every code the host's target trial owners
return has a line in both languages.  Synthetic records only.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver import study_contexts as context_store
from easyicu.webserver import target_trial_card as card_owner
from easyicu.webserver import target_trial_records
from tests.support.node import run_node
from tests.support.target_trial import STUDY_ID, kept_target_trial_record

STATIC = Path(card_owner.__file__).resolve().parent / "static"
JS = STATIC / "js"
_CAUSAL = {
    "analysis_family": "causal_inference",
    "analysis_unit": "icu_stay",
    "variance_estimator": "model_based",
}
_CARD_FILES = (
    "screens-guided-pi-target-trial-copy.js",
    "screens-guided-pi-target-trial.js",
    "screens-guided-pi-confirmation.js",
)
_ITEM_LABELS = ["入选标准", "治疗策略", "分配", "时间零点", "随访", "结局", "因果对比", "分析计划"]

_HARNESS = r"""
const path = require('path');
const [jsDir, lang, files, input] = process.argv.slice(1, 5);
global.window = { EU_LANG: lang };
const registry = {};
window.EasyICU = { guidedPi: {
  declare: (name, api) => { registry[name] = api; },
  optional: name => registry[name] || null,
  require: name => { if (!registry[name]) throw new Error('missing owner ' + name); return registry[name]; },
} };
require(path.join(jsDir, 'html-escape.js'));
for (const file of JSON.parse(files)) require(path.join(jsDir, file));
const tr = (en, zh) => (lang === 'zh' ? zh : en) || en;
const esc = window.EU_HTML.esc;
const INPUT = JSON.parse(input);
const out = value => process.stdout.write(JSON.stringify(value === undefined ? null : value));
const SESSION = { session_id: 'session-1', binding: { study_context_id: 'study-trial', study_revision: 7 } };
function confirmationFor(workflow, session) {
  return registry.confirmation.create({
    tr, esc, iconHtml: () => '',
    resourceButton: resource => `<a data-resource="${resource.artifact}"></a>`,
    sessionIsStale: () => false, busy: () => false,
    session: () => session || SESSION, workflow: () => workflow,
  });
}
(async () => {
__SCENARIO__
})().catch(error => { console.error(error && error.stack || String(error)); process.exit(1); });
"""


def _node(scenario: str, *, files=_CARD_FILES, payload: Any = None, lang: str = "zh") -> Any:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = run_node(
        node,
        _HARNESS.replace("__SCENARIO__", scenario),
        str(JS),
        lang,
        json.dumps(list(files)),
        json.dumps(payload if payload is not None else {}),
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """An isolated study store; ``["value"]`` is the study's latest compile job."""

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )
    latest: dict[str, Any] = {"value": None}
    monkeypatch.setattr(
        card_owner, "latest_target_trial_compile", lambda study_id: latest["value"]
    )
    return latest


# --- the wording ------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_every_code_the_host_returns_has_its_line(lang: str) -> None:
    lines = _node(
        """
        const copy = registry.targetTrialCopy;
        out(Object.fromEntries(copy.CODES.map(code => [code, copy.line(code, tr)])));
        """,
        files=("screens-guided-pi-target-trial-copy.js",),
        lang=lang,
    )

    # Both directions: a code without a line, or a line for no code, fails.
    assert sorted(lines) == sorted(set(card_owner.TARGET_TRIAL_UI_CODES))
    assert all(text.strip() for text in lines.values())
    if lang == "zh":
        assert all(re.search(r"[一-鿿]", text) for text in lines.values())


# --- the card the host computes ------------------------------------------------------


def _compile_job(status: str, kept: Any) -> dict[str, Any]:
    return {
        "job_id": f"job_{status}",
        "status": status,
        "reason_code": None,
        "compile_sha256": kept.compile_sha256 if status == "compiled" else None,
        "detail": None,
        "missing_concepts": [],
        "cause_code": None,
    }


def _approvable_card(isolated: dict[str, Any]) -> tuple[dict, Any, dict]:
    """The card the host computes for the record the latest compile wrote."""

    kept = kept_target_trial_record()
    created = context_store.upsert_context(
        {"id": STUDY_ID, "question": "q", "analysis_design": dict(_CAUSAL)}
    )
    target_trial_records.keep_target_trial_record(STUDY_ID, kept)
    study = context_store.bind_target_trial_design(
        STUDY_ID, kept.design(), expected_revision=created["revision"]
    )
    isolated["value"] = _compile_job("compiled", kept)
    card = card_owner.target_trial_card(study, isolated["value"])
    assert card is not None and card["state"] == "approvable"
    return card, kept, study


_TICK_SCENARIO = """
const workflow = { next_action_code: 'target_trial_review', target_trial_card: INPUT.card };
const session = { session_id: 'session-1', binding: { study_context_id: INPUT.study, study_revision: INPUT.revision } };
const owner = confirmationFor(workflow, session);
const trial = registry.targetTrial;
const before = owner.workflowConfirmationHtml();
const digest = INPUT.card.compile_sha256;
const lines = INPUT.card.confirmations.length;
for (let index = 0; index < lines - 1; index += 1) trial.noteTick(digest, index, true);
const short = trial.approvalRequest(workflow, session, 'project-1');
trial.noteTick(digest, lines - 1, true);
const after = owner.workflowConfirmationHtml();
out({
  title: owner.workflowConfirmation().title,
  boxes: (before.match(/data-gpi-trial-line=/g) || []).length,
  previous: before.includes('class="gpi-trial-stale"'),
  ceilings: (before.match(/证据上限：仅为分析/g) || []).length,
  disabledBefore: before.includes('data-gpi-confirm-action disabled'),
  disabledAfter: after.includes('data-gpi-confirm-action disabled'),
  checkedAfter: (after.match(/ checked>/g) || []).length,
  labels: [...before.matchAll(/<dt>([^<]*)<\\/dt>/g)].map(match => match[1]),
  short,
  request: trial.approvalRequest(workflow, session, 'project-1'),
});
"""


def test_the_card_the_host_computes_approves_once_every_line_is_ticked(
    isolated: dict[str, Any],
) -> None:
    card, kept, study = _approvable_card(isolated)

    shown = _node(
        _TICK_SCENARIO,
        payload={"card": card, "study": STUDY_ID, "revision": study["revision"]},
    )

    assert shown["title"] == "核对并批准目标试验"
    assert shown["boxes"] == kept.confirmation_lines == len(card["confirmations"])
    assert shown["previous"] is False
    # The record lists its evidence ceiling among its limitations: once.
    assert card["evidence_ceiling"] == "analysis_only"
    assert "evidence_ceiling" in {row["code"] for row in card["limitations"]}
    assert shown["ceilings"] == 1
    assert (shown["disabledBefore"], shown["disabledAfter"]) == (True, False)
    assert shown["checkedAfter"] == kept.confirmation_lines
    assert shown["labels"] == _ITEM_LABELS
    assert shown["short"] is None
    assert shown["request"] == {
        "project_id": "project-1",
        "study_context_id": STUDY_ID,
        "expected_revision": study["revision"],
        "compile_sha256": kept.compile_sha256,
        "n_lines_confirmed": kept.confirmation_lines,
    }
    # The request the card builds is the one the host approves.
    request = shown["request"]
    approved = card_owner.approve_target_trial(
        STUDY_ID,
        expected_revision=request["expected_revision"],
        compile_sha256=request["compile_sha256"],
        n_lines_confirmed=request["n_lines_confirmed"],
    )
    assert approved["repeated"] is False


def test_a_card_that_is_not_the_latest_statement_offers_nothing_to_approve(
    isolated: dict[str, Any],
) -> None:
    _card_now, kept, study = _approvable_card(isolated)
    # A newer statement is compiling: the host shows this record as the
    # previous version, which the card cannot approve.
    card = card_owner.target_trial_card(study, _compile_job("running", kept))
    assert card is not None and (card["state"], card["stale"]) == ("compiling", True)
    assert card["confirmations"]

    shown = _node(
        _TICK_SCENARIO,
        payload={"card": card, "study": STUDY_ID, "revision": study["revision"]},
    )

    assert shown["title"] == "正在编译目标试验"
    assert (shown["previous"], shown["boxes"], shown["request"]) == (True, 0, None)


# --- each state the card can be in ------------------------------------------------------


def _card(**fields: Any) -> dict[str, Any]:
    return {
        "schema_version": "easyicu.target_trial_card/1",
        "state": "stopped",
        "reason_code": None,
        "compile_sha256": None,
        "confirmation_lines": None,
        "protocol": [],
        "confirmations": [],
        "limitations": [],
        "evidence_ceiling": None,
        "approvable": False,
        "blocking": [],
        "approval": None,
        "latest_compile": None,
        "stale": True,
        **fields,
    }


def _latest(status: str, reason: str | None = None, **fields: Any) -> dict[str, Any]:
    return {
        "job_id": "job_a",
        "status": status,
        "reason_code": reason,
        "compile_sha256": None,
        "detail": None,
        "missing_concepts": [],
        "cause_code": None,
        **fields,
    }


_PROTOCOL = [{"item": "time_zero", "text": "6 h after ICU admission."}]
_STATES = {
    "statement_needed": (
        "target_trial_statement_needed",
        None,
        "说明本研究模拟的目标试验",
        ["请在对话里描述治疗、策略、时间零点"],
        {"approve": False, "edit": True},
    ),
    "compiling": (
        "target_trial_review",
        _card(state="compiling", latest_compile=_latest("running")),
        "正在编译目标试验",
        ["正在用研究数据编译目标试验"],
        {"approve": False, "edit": False},
    ),
    "data_unavailable": (
        "target_trial_review",
        _card(
            latest_compile=_latest(
                "stopped",
                "target_trial_data_unavailable",
                detail="The data package does not provide what the trial reads.",
                missing_concepts=["vaso_ind"],
            )
        ),
        "目标试验没有编成",
        ["研究的数据包缺少试验要读的概念", "数据包缺少的概念：", "<code>vaso_ind</code>"],
        {"approve": False, "edit": True},
    ),
    "failed": (
        "target_trial_review",
        _card(
            latest_compile=_latest(
                "failed", "target_trial_compile_failed", cause_code="KeyError"
            )
        ),
        "目标试验没有编成",
        ["编译意外失败", "原因码：", "<code>KeyError</code>"],
        {"approve": False, "edit": True},
    ),
    "interrupted": (
        "target_trial_review",
        _card(latest_compile=_latest("interrupted", "target_trial_compile_interrupted")),
        "目标试验没有编成",
        ["编译作业中断，没有留下结果"],
        {"approve": False, "edit": True},
    ),
    "blocked": (
        "target_trial_review",
        _card(
            state="blocked",
            stale=False,
            protocol=_PROTOCOL,
            blocking=[
                {
                    "source": "element",
                    "name": "treatment",
                    "reason": "tte_treatment_onset_not_materialized",
                    "detail": "<onset> is read after ICU admission.",
                }
            ],
        ),
        "目标试验还不能批准",
        [
            "<strong>treatment</strong>",
            "数据中没有从入 ICU 到宽限期结束的治疗开始时刻",
            "&lt;onset&gt; is read after ICU admission.",
        ],
        {"approve": False, "edit": True},
    ),
    "previous_version": (
        "target_trial_review",
        _card(state="compiling", protocol=_PROTOCOL, latest_compile=_latest("running")),
        "正在编译目标试验",
        ["下面是上一版试验；新的陈述编好之前不能批准。", "6 h after ICU admission."],
        {"approve": False, "edit": False},
    ),
    "approved": (
        "target_trial_plan_ready",
        _card(
            state="approved",
            stale=False,
            approvable=True,
            protocol=_PROTOCOL,
            approval={"approval_event_id": "approval:x", "n_lines_confirmed": 5},
        ),
        "目标试验已批准",
        ["生成计划会读取本研究的数据包", "已批准，确认了 5 行。", "按本研究数据生成计划"],
        {"approve": True, "edit": False},
    ),
    "record_missing": (
        "target_trial_review",
        _card(
            reason_code="target_trial_record_missing",
            stale=False,
            latest_compile=_latest("compiled", detail="The trial compiled."),
        ),
        "目标试验没有编成",
        ["宿主已找不到这一版试验的编译记录"],
        {"approve": False, "edit": True},
    ),
}


@pytest.mark.parametrize("state", sorted(_STATES))
def test_each_state_of_the_card_says_what_happens_next(state: str) -> None:
    code, card, title, lines, actions = _STATES[state]

    shown = _node(
        """
        const owner = confirmationFor({ next_action_code: INPUT.code, target_trial_card: INPUT.card });
        const html = owner.workflowConfirmationHtml();
        out({
          title: owner.workflowConfirmation().title,
          html,
          approve: html.includes('data-gpi-confirm-action'),
          edit: html.includes('data-gpi-confirm-edit'),
        });
        """,
        payload={"code": code, "card": card},
    )

    assert shown["title"] == title
    for line in lines:
        assert line in shown["html"]
    assert {"approve": shown["approve"], "edit": shown["edit"]} == actions
    # The record's own text never reaches the page unescaped.
    assert "<onset>" not in shown["html"]
    # A card the host's compile detail does not explain shows none of it.
    if state == "record_missing":
        assert "The trial compiled." not in shown["html"]


def test_a_change_request_starts_from_the_trial() -> None:
    drafts = _node(
        """
        out(['target_trial_statement_needed', 'target_trial_review', 'target_trial_plan_ready']
          .map(code => confirmationFor({ next_action_code: code, target_trial_card: null }).planChangeDraft()));
        """
    )

    assert drafts == ["我要模拟的目标试验：", "我想修改目标试验：", "我想修改目标试验："]


# --- the click ----------------------------------------------------------------------------


_ACTIONS_SCENARIO = """
const calls = [];
let busy = false;
const workflow = {
  next_action_code: INPUT.code, target_trial_card: INPUT.card,
  host_decisions: { plan_transition: { family: 'plan_transition', next_action_code: INPUT.code } },
};
const session = {
  session_id: 'session-1', archived_child_jobs: [],
  binding: { study_context_id: 'study-trial', study_revision: 7, run_id: '' },
  research_provider: { provider: 'openai', credential_source: 'pi_verified' },
};
const host = {
  tr, esc, iconHtml: () => '', resourceButton: () => '', errorText: error => 'raw:' + String(error && error.message),
  regeneration: {}, nextActions: {}, replay: {}, session: () => session, workflow: () => workflow,
  busy: () => busy, latestRun: () => null, sessionIsStale: () => false, projectId: () => 'project-1',
  turnGrants: () => [], setBusy: value => { busy = value; }, setError: value => { if (value) calls.push(['error', value]); },
  render: () => {}, refreshSession: async () => calls.push(['refresh']), loadWorkflow: async () => calls.push(['workflow']),
  watchChildJob: (...args) => calls.push(['watch', ...args]), appendMessage: message => calls.push(['message', message.text]),
  sendText: async (...args) => calls.push(['send', ...args]), recordHostAction: async () => null,
  api: () => ({
    approvePiCopilotTargetTrial: async (sessionId, body) => {
      calls.push(['approve', sessionId, body]);
      if (INPUT.refusal) { const error = new Error('refused'); error.code = INPUT.refusal; throw error; }
      return { ok: true };
    },
    loadStudyContext: async id => ({ context: { id, question: 'q', data_source: { path: '/prepared/miiv' } } }),
    startAgentRun: async body => { calls.push(['run', body]); return { job_id: 'plan-job' }; },
  }),
};
const actions = registry.planActions.create(host);
const confirm = () => actions.confirmWorkflow(confirmationFor(workflow, session).workflowConfirmation());
await confirm();
const beforeTicks = calls.length;
const ticks = INPUT.ticks === undefined ? (INPUT.card && INPUT.card.confirmations || []).length : INPUT.ticks;
for (let index = 0; index < ticks; index += 1) registry.targetTrial.noteTick(INPUT.card.compile_sha256, index, true);
if (INPUT.code === 'target_trial_review') await confirm();
out({ beforeTicks, calls });
"""
_ACTION_FILES = (*_CARD_FILES, "screens-guided-pi-plan-actions.js")
_DIGEST = "a" * 64


def _ticking_card() -> dict[str, Any]:
    return _card(
        state="approvable",
        stale=False,
        approvable=True,
        compile_sha256=_DIGEST,
        confirmation_lines=2,
        confirmations=[{"kind": "k", "element": "e", "text": "one"}, {"kind": "k", "element": "e", "text": "two"}],
        protocol=_PROTOCOL,
    )


def test_the_click_approves_the_record_on_the_card_once_every_line_is_ticked() -> None:
    shown = _node(
        _ACTIONS_SCENARIO,
        files=_ACTION_FILES,
        payload={"code": "target_trial_review", "card": _ticking_card()},
    )

    # Before every line is ticked the click sends nothing.
    assert shown["beforeTicks"] == 0
    assert shown["calls"] == [
        [
            "approve",
            "session-1",
            {
                "project_id": "project-1",
                "study_context_id": "study-trial",
                "expected_revision": 7,
                "compile_sha256": _DIGEST,
                "n_lines_confirmed": 2,
            },
        ],
        ["refresh"],
        ["workflow"],
    ]


def test_a_card_that_lists_other_lines_than_its_record_names_does_not_approve() -> None:
    # The record names two lines and the card lists three: ticking two of them
    # does not confirm the record.
    card = _ticking_card()
    card["confirmations"] = [*card["confirmations"], {"kind": "k", "element": "e", "text": "three"}]

    shown = _node(
        _ACTIONS_SCENARIO,
        files=_ACTION_FILES,
        payload={"code": "target_trial_review", "card": card, "ticks": 2},
    )

    assert shown["calls"] == []


@pytest.mark.parametrize(
    ("refusal", "line"),
    [
        ("target_trial_restatement_pending", "更新的试验陈述还在编译，这一版不能批准"),
        ("study_context_revision_conflict", "读取之后研究被修改了"),
        ("target_trial_design_invalid", "已确认的行数与记录不符"),
    ],
)
def test_a_refused_click_shows_the_card_the_host_has_now_and_why(
    refusal: str, line: str
) -> None:
    shown = _node(
        _ACTIONS_SCENARIO,
        files=_ACTION_FILES,
        payload={"code": "target_trial_review", "card": _ticking_card(), "refusal": refusal},
    )

    calls = shown["calls"]
    assert [call[0] for call in calls] == ["approve", "workflow", "error"]
    assert line in calls[-1][1]


def test_the_approved_trial_generates_its_plan_on_the_studys_data() -> None:
    shown = _node(
        _ACTIONS_SCENARIO,
        files=_ACTION_FILES,
        payload={"code": "target_trial_plan_ready", "card": _card(state="approved", stale=False)},
    )

    calls = {call[0]: call[1:] for call in shown["calls"]}
    assert calls["message"] == ["按本研究数据生成计划"]
    body = calls["run"][0]
    # The ordinary formal plan request: the route plans an approved trial on
    # the study's data, so the browser asks for nothing else.
    assert body["planner_start_mode"] == "fresh"
    assert body["plan_revision_source_run_id"] == ""
    assert body["host_action"]["action_code"] == "generate_plan"
    assert calls["watch"] == ["plan-job", "easyicu_full_run_submitted"]
    actions = (JS / "screens-guided-pi-plan-actions.js").read_text(encoding="utf-8")
    for name in ("FRESH_PLAN_CODES", "AUTOMATIC_PROVIDER_RUN_CODES"):
        listed = re.search(rf"const {name} = new Set\(\[(.*?)\]\);", actions, re.S)
        assert listed and "'target_trial_plan_ready'" not in listed.group(1)


@pytest.mark.parametrize(
    ("lang", "line"),
    [
        ("zh", "这项研究的目标试验已批准，计划要按研究数据生成，不能只读元数据生成候选计划。请直接生成计划。"),
        ("en", "This study’s target trial is approved, so its plan is generated on the study’s data, not as a metadata-only candidate. Generate the plan."),
    ],
)
def test_a_candidate_plan_the_launch_refuses_reads_as_its_cause(lang: str, line: str) -> None:
    # research_pipeline_run_preparation refuses a metadata-only candidate of a
    # study whose trial is approved; the submission returns its code.
    shown = _node(
        """
        const owner = registry.errorText.create({ tr, staticPreview: () => false });
        out(owner.errorText({ code: 'research_pipeline_target_trial_plans_on_data', message: 'The host text.' }));
        """,
        files=("screens-guided-pi-error-text.js",),
        lang=lang,
    )

    assert shown == line


# --- the conversation's rows ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("job", "title", "terminal", "blocked"),
    [
        (
            {"status": "done", "target_trial_compile": {"status": "compiled", "approvable": True}},
            "目标试验已编成",
            "请在试验卡片上逐行核对后批准",
            False,
        ),
        (
            {"status": "done", "target_trial_compile": {"status": "compiled", "approvable": False}},
            "目标试验已编成",
            "试验卡片列出了挡住批准的各项",
            True,
        ),
        (
            {"status": "done", "target_trial_compile": {"status": "stopped", "reason_code": "target_trial_database_out_of_scope"}},
            "目标试验没有编成",
            "本版本只在 MIMIC-IV 中模拟目标试验",
            True,
        ),
        (
            {"status": "failed", "error_code": "target_trial_compile_failed"},
            "目标试验编译失败",
            "编译意外失败",
            False,
        ),
    ],
    ids=["approvable", "blocked", "stopped", "failed"],
)
def test_the_compile_row_reads_its_own_result(
    job: dict, title: str, terminal: str, blocked: bool
) -> None:
    shown = _node(
        "out(registry.replay.childJobPresentation(INPUT.job, tr));",
        files=("screens-guided-pi-target-trial-copy.js", "screens-guided-pi-replay.js"),
        payload={"job": {"kind": "target-trial-compile", **job}},
    )

    assert shown["title"] == title
    assert terminal in shown["terminalLabel"]
    assert shown["blocked"] is blocked


def test_the_conversation_follows_the_compile_and_writes_the_approval() -> None:
    def read(name: str) -> str:
        return (JS / name).read_text(encoding="utf-8")

    code = "'easyicu_target_trial_compile_submitted'"
    # The tool result spends the turn and hands over its job, live and on reload.
    assert code in read("screens-guided-pi-live-stream.js").split("function create", 1)[0]
    assert code in read("screens-guided-pi-transcript.js").split("const IDEA_EXPLORATION_TOOLS", 1)[0]
    assert "kind === 'target-trial-compile' ? 'easyicu_target_trial_compile_submitted'" in read(
        "screens-guided-pi.js"
    )
    assert "target_trial_card: (payload && payload.target_trial_card) || null" in read(
        "screens-guided-pi.js"
    )
    childjob = read("screens-guided-pi-childjob.js")
    assert "value === 'easyicu_target_trial_compile_submitted' || value === 'target-trial-compile'" in childjob
    # The host writes the approval row; its text is the approval's own line.
    assert "done: trialLine('target_trial_approved'" in read("screens-guided-pi-transcript.js")
    activity = read("screens-guided-pi-activity.js")
    assert activity.count("easyicu_state_target_trial: tr(") == 2
    aside = read("screens-guided-pi-aside.js")
    for workflow_code in card_owner.TARGET_TRIAL_WORKFLOW_CODES:
        assert len(re.findall(rf"^\s*{workflow_code}: tr\(", aside, re.M)) == 1


# --- repeated stays in a causal study ------------------------------------------------------


_COHORT_SCENARIO = """
const owner = registry.cohortEligibility.create({
  tr, esc, busy: () => false, sessionIsStale: () => false, planConfigurationError: () => '',
  session: () => ({ cohort_eligibility_selection: {
    present: true, stated: false, blocker_code: 'cohort_eligibility_confirmation_required',
    primary_cohort_contract: { admission_eligibility: {
      minimum_age_years: 18, minimum_icu_duration_hours: 0, repeated_admission_policy: 'all_icu_admissions' } },
    options: [
      { id: 'adults_all_admissions', label: { en: 'All stays', zh: '全部入住' } },
      { id: 'adults_first_admission', label: { en: 'First stay', zh: '首次入住' } },
    ],
  } }),
  workflow: () => ({ next_action_code: 'cohort_eligibility_confirmation_required',
    study_setup_receipt: { configuration: { analysis_design: INPUT.design } } }),
});
const html = owner.render();
const match = html.match(/<p class="gpi-cohort-rationale"><strong>[^<]*<\\/strong><span>([^<]*)<\\/span>/);
out(match ? match[1] : null);
"""


@pytest.mark.parametrize(
    ("estimator", "lang", "line"),
    [
        ("bootstrap", "zh", "当前以 ICU stay 为分析单位，并已设置按患者重抽样的 bootstrap，因此推荐保留全部符合条件的 ICU 入住。"),
        ("bootstrap", "en", "repeated stays are already handled with a bootstrap that resamples patients"),
        ("cluster_robust", "zh", "当前以 ICU stay 为分析单位，并已设置患者层聚类稳健方差，因此推荐保留全部符合条件的 ICU 入住。"),
        ("model_based", "zh", "推荐项会保留当前研究已经配置好的队列定义。"),
    ],
)
def test_repeated_stays_a_bootstrap_resamples_by_patient_are_kept(
    estimator: str, lang: str, line: str
) -> None:
    # A causal study's variance is the bootstrap of its trial suite; with
    # repeat stays kept it resamples patients, which handles them.
    shown = _node(
        _COHORT_SCENARIO,
        files=("screens-guided-pi-cohort-eligibility.js",),
        payload={
            "design": {
                "analysis_unit": "icu_stay",
                "variance_estimator": estimator,
                "cluster_unit": "patient",
            }
        },
        lang=lang,
    )

    assert shown is not None and line in shown

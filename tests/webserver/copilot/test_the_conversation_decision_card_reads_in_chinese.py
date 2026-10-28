"""The conversation's decision card reads in Chinese for every review decision.

A plan review that needs the researcher's answer shows it in the conversation
as a card: a title, the question, and the review's evidence.  The card wrote
its own Chinese for five decisions and fell back to the host's English for
the rest, while the plan review artifact already read every decision from the
reader vocabulary.  The card now takes the question and the evidence from the
same vocabulary; English readers keep the host's text.  Method choices the
system owns never become a card, so they are not checked here.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from tests.support.node import run_node
from tests.support.review_decisions import decision_codes

STATIC = Path(__file__).parents[3] / "src" / "easyicu" / "webserver" / "static"
MODULES_SOURCE = (STATIC / "js/screens-guided-pi-modules.js").read_text(encoding="utf-8")
VOCAB_SOURCE = (STATIC / "js/screens-agent-reader-vocab.js").read_text(encoding="utf-8")
CONFIRMATION_SOURCE = (STATIC / "js/screens-guided-pi-confirmation.js").read_text(
    encoding="utf-8"
)
CJK = re.compile(r"[一-鿿]")


def _system_owned_codes() -> set[str]:
    match = re.search(
        r"const systemOwnedPlanFindingCodes = new Set\(\[(.*?)\]\)", CONFIRMATION_SOURCE, re.S
    )
    assert match, "the card's system-owned decisions are not declared"
    return set(re.findall(r"'([A-Z_]+)'", match.group(1)))


CARD_DECISIONS = sorted(decision_codes() - _system_owned_codes())


def _card(code: str, *, lang: str, decision_context: dict | None = None) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    question = {
        "code": code,
        "question": f"Host question for {code}?",
        "evidence": f"Host evidence for {code}.",
        "remediation": f"Host remediation for {code}.",
    }
    if decision_context is not None:
        question["decision_context"] = decision_context
    review = {
        "run_id": "run-decision-card",
        "authorization_questions": [question],
        "remediation_buckets": {
            "agent_plan_revision": [],
            "runtime_capability": [],
            "study_authority_change": [code],
            "external_evidence": [],
            "independent_review": [],
        },
    }
    workflow = {"next_action_code": "plan_scientific_changes_required", "plan_review_summary": review}
    script = f"""
      global.window = {{ EU_LANG: {json.dumps(lang)} }};
      eval({json.dumps(VOCAB_SOURCE)});
      eval({json.dumps(MODULES_SOURCE)});
      eval({json.dumps(CONFIRMATION_SOURCE)});
      const confirmation = window.EasyICU.guidedPi.require('confirmation').create({{
        tr: (en, zh) => {json.dumps(lang)} === 'zh' ? (zh || en) : en,
        esc: value => String(value),
        iconHtml: () => '',
        resourceButton: () => '',
        sessionIsStale: () => false,
        workflow: () => ({json.dumps(workflow)}),
        session: () => ({{ archived_child_jobs: [] }}),
        busy: () => false,
      }});
      process.stdout.write(confirmation.workflowConfirmationHtml());
    """
    return run_node(node, script, check=True).stdout


def _shown(html: str) -> tuple[str, str, str]:
    """The card's title, question and evidence, as the reader sees them."""

    body = re.search(
        r'class="gpi-confirmation-body">.*?<strong>(.*?)</strong>.*?<small>(.*?)</small>',
        html,
        re.S,
    )
    evidence = re.search(
        r'class="gpi-decision-evidence"><span>.*?</span><strong>.*?</strong><small>(.*?)</small>',
        html,
        re.S,
    )
    assert body and evidence, html[:400]
    return body.group(1), body.group(2), evidence.group(1)


def test_the_card_decisions_are_found() -> None:
    assert {
        "OUTCOME_DEFINITION_UNRESOLVED",
        "REQUESTED_DOSE_RESPONSE_NOT_ESTIMABLE",
        "PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH",
    } <= set(CARD_DECISIONS)
    assert "REPEATED_STAY_METHOD_NOT_DECLARED" not in CARD_DECISIONS


def test_an_undefined_requested_endpoint_explains_the_stop_without_replacement_choices():
    html = _card("OUTCOME_DEFINITION_UNRESOLVED", lang="zh", decision_context={
        "endpoint_definition_unavailable": True, "endpoint_options": [],
    })
    assert "主要结局的定义不完整" in html
    assert "所问结局的定义补全前无法分析，不会用其他结局代替。" in html
    assert "请选择本研究的主要结局" not in html


@pytest.mark.parametrize("code", CARD_DECISIONS)
def test_a_decision_card_reads_in_chinese(code: str) -> None:
    title, question, evidence = _shown(_card(code, lang="zh"))

    assert CJK.search(title)
    assert CJK.search(question) and "Host question" not in question
    assert CJK.search(evidence) and "Host evidence" not in evidence


def test_the_endpoint_choice_reads_in_chinese() -> None:
    code = "OUTCOME_DEFINITION_UNRESOLVED"
    html = _card(
        code,
        lang="zh",
        decision_context={
            "endpoint_options": [
                {"concept": "death_28d", "label_en": "28-day death", "label_zh": "28 天死亡"}
            ]
        },
    )

    title, question, evidence = _shown(html)

    assert title == "请选择本研究的主要结局"
    assert CJK.search(question) and "Host question" not in question
    assert CJK.search(evidence) and "Host evidence" not in evidence


def test_english_readers_keep_the_hosts_question_and_evidence() -> None:
    code = "REQUESTED_DOSE_RESPONSE_NOT_ESTIMABLE"

    _, question, evidence = _shown(_card(code, lang="en"))

    assert question == f"Host question for {code}?"
    assert evidence == f"Host evidence for {code}."


SURVIVAL_CARD = {
    "decision_code": "SURVIVAL_LANDMARK_OWNER_NOT_SEALED", "option_id": "fixed_landmark_survival",
    "scientific_configuration_sha256": "a" * 64, "exposure": "synthetic_exposure",
    "landmark_hours": 48.0, "endpoint": "death_60d", "horizon_days": 60.0,
    "followup_column": "followup_60d", "covariates": ["covariate_one", "covariate_two"],
    "rmst_alternative": {"time_column": "t", "event_column": "e", "group_column": "g",
                         "group_levels": ["0", "1"], "tau": 58.0, "time_unit": "days", "event_code": 1.0},
    "card_sha256": "b" * 64,
}


def test_the_survival_landmark_card_states_the_offered_design_and_binds_its_digest() -> None:
    html = _card("SURVIVAL_LANDMARK_OWNER_NOT_SEALED", lang="zh", decision_context=SURVIVAL_CARD)

    assert "请确定生存随访从何时开始" in html
    assert "第 48 小时 landmark · 60 天生存" in html
    assert "调整：covariate_one、covariate_two。" in html
    assert "敏感性分析：58 天限制性平均生存时间。" in html
    assert 'data-gpi-plan-decision-option="fixed_landmark_survival"' in html
    assert f'data-gpi-plan-decision-card="{"b" * 64}"' in html
    assert "采用第 48 小时 landmark" in html
    # The researcher can still choose another approach instead.
    assert "data-gpi-confirm-edit" in html


def test_a_survival_card_without_its_digest_offers_no_landmark_choice() -> None:
    card = {key: value for key, value in SURVIVAL_CARD.items() if key != "card_sha256"}
    html = _card("SURVIVAL_LANDMARK_OWNER_NOT_SEALED", lang="zh", decision_context=card)

    assert "fixed_landmark_survival" not in html
    assert "data-gpi-plan-decision-card" not in html


def test_the_survival_landmark_card_reads_in_english() -> None:
    html = _card("SURVIVAL_LANDMARK_OWNER_NOT_SEALED", lang="en", decision_context=SURVIVAL_CARD)

    assert "Choose when survival follow-up starts" in html
    assert "48-hour landmark · 60-day survival" in html
    assert "Use the 48-hour landmark" in html

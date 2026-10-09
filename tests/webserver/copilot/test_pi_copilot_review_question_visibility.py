"""The page offers the endpoint question the review routes to the user.

Method choices (timing design, repeated stays, sensitivity analyses) stay
system-owned and never become questions.  The endpoint is different: it belongs
to the estimand, and a review that routes it to the user blocks the revision
until it is answered.  The page used to treat it as a method choice, so the
card offered neither the question nor the revision and read as if nothing were
needed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.support.node import run_node


STATIC = Path(__file__).parents[3] / "src" / "easyicu" / "webserver" / "static"
MODULES_SOURCE = (STATIC / "js/screens-guided-pi-modules.js").read_text(encoding="utf-8")
CONFIRMATION_SOURCE = (STATIC / "js/screens-guided-pi-confirmation.js").read_text(encoding="utf-8")
RENDERER_SOURCE = (STATIC / "js/screens-agent-render.js").read_text(encoding="utf-8")
NO_ANSWER_NEEDED = "不需要再次回答科学设定问题"


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    return node


def _card(review: dict) -> str:
    workflow = {"next_action_code": "plan_scientific_changes_required", "plan_review_summary": review}
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({json.dumps(MODULES_SOURCE)});
      eval({json.dumps(CONFIRMATION_SOURCE)});
      const confirmation = window.EasyICU.guidedPi.require('confirmation').create({{
        tr: (en, zh) => zh || en,
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
    return run_node(_node(), script, check=True).stdout


def _review(questions: list[dict], **buckets: list[str]) -> dict:
    return {
        "run_id": "run-review-questions",
        "authorization_questions": questions,
        "remediation_buckets": {
            "agent_plan_revision": [], "runtime_capability": [], "study_authority_change": [],
            "external_evidence": [], "independent_review": [], **buckets,
        },
    }


def test_an_endpoint_question_names_itself_and_opens_the_answer():
    html = _card(_review(
        [{"code": "OUTCOME_DEFINITION_UNRESOLVED",
          "question": "Please confirm the intended clinical endpoint and time horizon in a new study version."}],
        agent_plan_revision=["A", "B", "C", "D"], runtime_capability=["E"],
        study_authority_change=["OUTCOME_DEFINITION_UNRESOLVED"], external_evidence=["F", "G"],
    ))

    assert "计划还需回答 1 个问题才能修订" in html
    assert "这项研究应使用哪个当前数据可支持的临床结局及时间范围？" in html
    assert "data-gpi-confirm-edit" in html
    assert NO_ANSWER_NEEDED not in html
    assert "data-gpi-confirm-action" not in html


@pytest.mark.parametrize(
    "code",
    [
        "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED",
        "REPEATED_STAY_IDENTITY_UNAVAILABLE",
        "REPEATED_STAY_METHOD_NOT_DECLARED",
        "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED",
        "REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY",
    ],
)
def test_a_method_choice_never_becomes_a_question(code):
    html = _card(_review(
        [{"code": code, "question": "Choose a method."}],
        agent_plan_revision=["FIGURE_ROLE_COVERAGE_INCOMPLETE"], study_authority_change=[code],
    ))

    assert "data-gpi-plan-decision-option" not in html
    assert "data-gpi-confirm-edit" not in html
    assert NO_ANSWER_NEEDED in html


def test_the_endpoint_question_offers_the_projected_endpoints():
    html = _card(_review(
        [{"code": "OUTCOME_DEFINITION_UNRESOLVED", "question": "Confirm the endpoint.",
          "decision_context": {"endpoint_options": [
              {"concept": "mort_90d", "label_en": "90-day Mortality", "label_zh": "90天死亡率"},
              {"concept": "death", "label_en": "In-hospital Mortality", "label_zh": "院内死亡"},
          ]}}],
        study_authority_change=["OUTCOME_DEFINITION_UNRESOLVED"],
    ))

    assert "请选择本研究的主要结局" in html
    assert 'data-gpi-plan-decision-option="mort_90d"' in html
    assert 'data-gpi-plan-decision-option="death"' in html
    assert "90天死亡率" in html and "院内死亡" in html
    assert "选择其他方案" in html


def test_without_a_question_the_revision_is_offered():
    html = _card(_review([], agent_plan_revision=["FIGURE_ROLE_COVERAGE_INCOMPLETE"]))

    assert "生成修订版候选计划" in html
    assert NO_ANSWER_NEEDED in html


def _review_view(findings: list[dict]) -> str:
    script = f"""
      global.window = {{
        EU_HTML: {{ esc: value => String(value ?? ''), escAttr: value => String(value ?? '') }},
        t: (en, zh) => zh, icon: () => '',
      }};
      eval({json.dumps(RENDERER_SOURCE)});
      process.stdout.write(window.AGENT_RENDER.artifactStructuredView(
        'scientific_plan_review.json', {json.dumps({"approval_allowed": False, "findings": findings})},
      ));
    """
    return run_node(_node(), script, check=True).stdout


def test_the_review_details_ask_for_an_endpoint_routed_to_the_user():
    html = _review_view([
        {"code": "OUTCOME_DEFINITION_UNRESOLVED", "remediation_route": "study_authority_change",
         "requires_user_authorization": True},
        {"code": "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED", "remediation_route": "study_authority_change",
         "requires_user_authorization": True},
    ])

    assert "计划还差 1 个决定" in html
    assert "确认主要结局与观察时间" in html
    assert "请在左侧对话中回答。" in html
    assert "Planner 将补全主要结局定义" not in html
    assert "Planner 将提出敏感性分析" in html


def test_the_review_details_keep_a_legacy_endpoint_finding_with_the_planner():
    html = _review_view([
        {"code": "OUTCOME_DEFINITION_UNRESOLVED", "requires_user_authorization": True},
    ])

    assert "Planner 将补全主要结局定义" in html
    assert "现在只做这一步" not in html

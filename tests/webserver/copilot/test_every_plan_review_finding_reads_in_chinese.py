"""Every plan-review finding reads in Chinese.

The plan review lists each finding by a title and a detail.  The reader
vocabulary, or the renderer's own copy, owns their Chinese wording; without it
a Chinese reader sees the host's English message.  The findings are read from
the review owner's source, so one added later without Chinese wording fails
here.
"""

from __future__ import annotations

import json
import re
import shutil

import pytest

from tests.support.node import run_node
from tests.support.review_decisions import finding_codes
from tests.webserver.copilot.pi_copilot_static_fixtures import _read

CJK = re.compile(r"[一-鿿]")

FINDINGS = sorted(finding_codes({"PlanScientificFinding"}))


def _render_zh(findings: list[dict]) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    payload = {"approval_allowed": False, "status": "changes_required", "findings": findings}
    script = f"""
      global.window = {{
        EU_HTML: {{
          esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
          escAttr: value => String(value ?? ''),
        }},
        t: (en, zh) => zh,
        icon: () => '',
      }};
      eval({json.dumps(_read('js/screens-agent-reader-vocab.js'))});
      eval({json.dumps(_read('js/screens-agent-render.js'))});
      process.stdout.write(window.AGENT_RENDER.artifactStructuredView('scientific_plan_review.json', {json.dumps(payload)}));
    """
    return run_node(node, script, check=True).stdout


def test_the_review_owner_findings_are_found() -> None:
    # Both branches of a code chosen by a condition.
    assert {
        "PRIMARY_MODEL_RETENTION_INSUFFICIENT",
        "PRIMARY_MODEL_RETENTION_REDUCED",
        "OUTCOME_DEFINITION_UNRESOLVED",
        "STUDY_TIME_ZERO_MISMATCH",
    } <= set(FINDINGS)


def test_every_plan_review_finding_has_a_chinese_title_and_detail() -> None:
    html = _render_zh(
        [
            {
                "code": code,
                "severity": "major",
                "remediation_route": "agent_plan_revision",
                "requires_user_authorization": False,
                "message": f"Host message for {code}.",
                "remediation": f"Host remediation for {code}.",
            }
            for code in FINDINGS
        ]
    )

    rows = re.findall(r"<li><strong>(.*?)</strong><span>(.*?)</span></li>", html, re.S)
    assert len(rows) == len(FINDINGS)
    english = [
        title
        for title, detail in rows
        if not CJK.search(title) or not CJK.search(detail)
    ]
    assert english == []


def test_a_requested_variable_without_data_has_the_reviewed_next_step() -> None:
    html = _render_zh([{
        "code": "REQUESTED_CONCEPT_NO_DATA", "severity": "blocker",
        "message": "The data contain no recorded values for fixture_response_max.",
    }])
    assert "所需变量没有数据" in html
    assert (
        "问题要求汇总的变量在当前数据中没有任何记录值，这部分结果无法给出。"
        "请换用有记录的变量或数据源后重新规划。"
    ) in html


def test_unimplemented_sensitivity_is_shown_in_the_runtime_lane() -> None:
    html = _render_zh([{
        "code": "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED", "severity": "major",
        "remediation_route": "runtime_capability", "requires_user_authorization": False,
    }])
    lanes = re.findall(r"<article>(.*?)</article>", html, re.S)
    assert "敏感性检验尚不能执行" not in lanes[0]
    assert "敏感性检验尚不能执行" in lanes[2]
    assert "单纯重新生成计划不能补足这项能力" in html
    assert "Planner 将提出敏感性分析" not in html


@pytest.mark.parametrize(
    "reason",
    [
        "repeated_stays_possible_without_verified_patient_grouping",
        "repeated_stays_possible_with_invalid_patient_grouping_authority",
    ],
)
def test_withheld_intervals_do_not_claim_patient_grouping_is_absent(reason) -> None:
    html = _render_zh([
        {
            "code": "DESCRIPTIVE_INTERVALS_WITHHELD_WITHOUT_PATIENT_GROUPING",
            "severity": "major",
            "remediation_route": "runtime_capability",
            "requires_user_authorization": False,
            "message": "The host withheld uncertainty intervals.",
            "descriptive_interval_limitation": {
                "declared_variance_estimator": "model_based",
                "reason": reason,
            },
        },
    ])

    (title, detail), = re.findall(
        r"<li><strong>(.*?)</strong><span>(.*?)</span></li>", html, re.S,
    )
    assert CJK.search(title) and CJK.search(detail)
    assert "患者分组依据未经核实或未通过校验" in detail
    assert "保留全部住院" in detail
    assert "计数与比例" in detail


def test_a_prepared_package_missing_source_grouping_states_the_distinct_cause() -> None:
    html = _render_zh([{
        "code": "DESCRIPTIVE_INTERVALS_WITHHELD_WITHOUT_PATIENT_GROUPING",
        "severity": "major",
        "descriptive_interval_limitation": {
            "declared_variance_estimator": "model_based",
            "reason": "verified_patient_grouping_not_carried_by_prepared_package",
        },
    }])
    assert "来源已有经核实的患者分组" in html
    assert "准备后的数据包未携带" in html
    assert "计数与比例" in html
    assert "患者分组依据未经核实或未通过校验" not in html


#: The prediction-validity round's findings.  Their rows land before the
#: round's review owner emits them; FINDINGS then covers them as well.
PREDICTION_FINDINGS = {
    "PREDICTION_PREDICTOR_TIMING_UNPROVEN": "预测变量的观测时间无法确认",
    "PREDICTION_RISK_SET_NOT_KEPT": "队列没有限定预测时点仍在 ICU 的住院",
    "PREDICTION_PATIENT_GROUPING_UNAVAILABLE": "数据来源无法按患者分组",
    "PREDICTION_PATIENT_GROUPING_NOT_CARRIED_BY_TRAJECTORY": "轨迹或 landmark 设计不带患者分组",
}


def test_the_prediction_validity_findings_read_in_chinese() -> None:
    html = _render_zh(
        [
            {
                "code": code,
                "severity": "major",
                "remediation_route": "agent_plan_revision",
                "requires_user_authorization": False,
                "message": f"Host message for {code}.",
                "remediation": f"Host remediation for {code}.",
            }
            for code in PREDICTION_FINDINGS
        ]
    )

    rows = re.findall(r"<li><strong>(.*?)</strong><span>(.*?)</span></li>", html, re.S)
    assert [title for title, _ in rows] == list(PREDICTION_FINDINGS.values())
    assert all(CJK.search(detail) and "Host message" not in detail for _, detail in rows)

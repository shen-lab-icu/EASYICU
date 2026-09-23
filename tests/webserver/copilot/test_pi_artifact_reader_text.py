"""Reader text of the shared artifact renderer follows the reader's language.

Plan-review findings, method-source uses, result-table names and columns are
host codes; the reader vocabulary owns their names, and a registered number is
only ever reformatted for display.
"""

import json
import shutil
import subprocess

import pytest

from tests.webserver.copilot.pi_copilot_static_fixtures import _read


def render(name, payload, *, lang="zh"):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    script = f"""
      global.window = {{
        EU_HTML: {{
          esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
          escAttr: value => String(value ?? ''),
        }},
        t: (en, zh) => {json.dumps(lang)} === 'zh' ? zh : en,
        icon: () => '',
      }};
      eval({json.dumps(_read('js/screens-agent-reader-vocab.js'))});
      eval({json.dumps(_read('js/screens-agent-render.js'))});
      process.stdout.write(window.AGENT_RENDER.artifactStructuredView({json.dumps(name)}, {json.dumps(payload)}));
    """
    return subprocess.run([node, "-e", script], check=True, capture_output=True, text=True).stdout


REVIEW = {
    "approval_allowed": True,
    "status": "analysis_only",
    "score": 92,
    "findings": [
        {"code": "REPEATED_STAY_IDENTITY_UNAVAILABLE", "severity": "major", "remediation_route": "runtime_capability",
         "message": "The stay-level source does not expose patient identity.", "remediation": "Retain all stays."},
        {"code": "NOVELTY_POSITIONING_REVIEW_REQUIRED", "severity": "major", "remediation_route": "independent_review",
         "message": "A screened comparison-source candidate exists.", "remediation": "Review the exact comparator."},
    ],
    "facts": {"literature_design_bindings": {"steps": [{"citations": [{
        "citation_key": "strobe_2007", "title": "The STROBE statement.", "year": "2007",
        "application": "Apply the host-curated method card(s) reporting_observational_study to the primary adjusted association.",
        "design_elements": ["dependence", "reporting"],
        "method_card_support": {"matched_card_ids": ["reporting_observational_study", "repeated_units_per_patient"]},
    }]}]}},
}


def test_plan_review_findings_and_method_uses_read_in_chinese():
    html = render("scientific_plan_review.json", REVIEW)

    assert "无法识别同一患者的多次入住" in html
    assert "创新性定位待独立审阅" in html
    assert "方法：观察性研究报告规范、同一患者的重复单位；支撑：相关性结构、报告规范" in html
    assert "Apply the host-curated method card" not in html
    assert "The stay-level source does not expose patient identity." not in html
    # What approval grants is stated; the raw score stays in the audit view.
    assert "仅分析级：批准后执行分析，不授予发表权限。" in html
    assert "92" not in html


def test_english_readers_keep_the_hosts_own_finding_message():
    html = render("scientific_plan_review.json", REVIEW, lang="en")

    assert "The stay-level source does not expose patient identity." in html
    assert "Method: Observational-study reporting, Repeated units per patient" in html


def test_result_tables_read_with_named_columns_and_display_precision():
    payload = {"tables": [
        {"name": "table_step_artifact_x__exposure_outcome_distribution.csv",
         "label": "Table exposure_outcome_distribution from step 03_distribution.",
         "headers": ["row_role", "exposure_level", "n_rows", "exposure_pct", "outcome_rate_pct"],
         "rows": [["exposure_level", "0.0", "62862", "66.550213", "8.246638"]]},
        {"name": "table_step_artifact_y__table_one.csv",
         "label": "Table table_one from step 02_table_one.",
         "headers": ["variable", "group", "denominator_n"], "rows": [["age", "Overall", "94458"]]},
        {"name": "cohort_flow_execute_repair__cohort_analysis_flow.csv",
         "label": "Exact sequential attrition ledger for the host-materialized analysis cohort.",
         "headers": ["n_before", "n_excluded", "n_remaining"], "rows": [["94458", "0", "94458"]]},
    ]}

    html = render("result_tables.json", payload)

    assert "结局比例 (%)" in html and "占比 (%)" in html and "暴露水平" in html
    assert "66.6" in html and "66.550213" not in html
    assert "62,862" in html
    assert "基线特征表（Table 1）" in html
    assert "队列流程（逐步排除）" in html
    assert "Exact sequential attrition ledger" not in html
    assert "row role" not in html and "outcome rate pct" not in html


def test_plan_reader_hides_row_identity_and_states_the_modelled_levels():
    payload = {
        "research_question": "KDIGO stage and in-hospital mortality",
        "display_labels": {"aki_stage_strict": "KDIGO AKI stage", "death": "In-hospital mortality"},
        "design_selection": {"candidates": [{
            "disposition": "selected", "analysis_type": "adjusted_association",
            "required_variables": ["patientunitstayid", "aki_stage_strict", "death"],
        }]},
        "steps": [
            {"step_id": "01", "method": "cohort_definition",
             "cohort_definition_spec": {"identity_column": "patientunitstayid"}},
            {"step_id": "04", "method": "adjusted_association", "model_requirements": [{
                "analysis_role": "primary", "exposure_source": "aki_stage_strict",
                "exposure_levels": ["0", "1", "2", "3"], "exposure_reference_level": "0",
                "primary_contrast_level": "3",
            }]},
        ],
    }

    html = render("agent_plan.json", payload)

    assert "patientunitstayid" not in html
    note = "主模型中的暴露水平：</strong>0、1、2、3（参照 0）；主要对比 3 vs 0。其他取值不进入主模型。"
    assert html.count(note) == 1

"""Reader text of the shared artifact renderer follows the reader's language.

Plan-review findings, method-source uses, result-table names and columns are
host codes; the reader vocabulary owns their names, and a registered number is
only ever reformatted for display.
"""

import json
import shutil

import pytest

from tests.support.node import run_node
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
      eval({json.dumps(_read('js/database-names.js'))});
      eval({json.dumps(_read('js/screens-agent-reader-vocab.js'))});
      eval({json.dumps(_read('js/screens-agent-run-file-reader.js'))});
      eval({json.dumps(_read('js/screens-agent-render.js'))});
      process.stdout.write(window.AGENT_RENDER.artifactStructuredView({json.dumps(name)}, {json.dumps(payload)}));
    """
    return run_node(node, script, check=True).stdout


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


def test_an_unrunnable_primary_summary_reads_in_both_languages():
    review = {**REVIEW, "approval_allowed": False, "status": "changes_required", "findings": [{
        "code": "PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED", "severity": "blocker",
        "remediation_route": "agent_plan_revision",
        "message": "The primary descriptive summary 'summary' cannot run.",
        "remediation": "Plan several measurements as the measurement_audit step.",
    }]}

    zh = render("scientific_plan_review.json", review)
    assert "主要描述性汇总无法执行" in zh and "测量审计" in zh
    assert "cannot run" not in zh
    assert "The primary descriptive summary 'summary' cannot run." in render(
        "scientific_plan_review.json", review, lang="en",
    )


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


def test_prediction_and_comparison_tables_read_with_named_columns():
    payload = {"tables": [
        {"name": "table_step_artifact_p__prediction_performance.csv",
         "label": "Table prediction_performance from step primary_performance.",
         "headers": ["validation_n", "validation_subject_n", "auroc", "auroc_ci_low", "auroc_ci_high",
                     "auroc_ci_method", "auroc_bootstrap_n", "auroc_bootstrap_skipped_n", "brier_score"],
         "rows": [["200", "190", "0.8123", "0.7712", "0.8478",
                   "patient_stratified_bootstrap_percentile_95pct", "2000", "3", "0.1034"]]},
        {"name": "table_step_artifact_b__benchmark_comparison.csv",
         "label": "Table benchmark_comparison from step benchmark_comparison.",
         "headers": ["comparator_column", "metric", "model_value", "comparator_value", "difference",
                     "difference_ci_low", "difference_ci_high", "interval_method", "comparison_n",
                     "information_window_relation"],
         "rows": [["apache_iv", "auroc", "0.81", "0.76", "0.05", "0.01", "0.09",
                   "delong_paired_normal_95pct", "180", "same"]]},
    ]}

    html = render("result_tables.json", payload)

    assert "预测性能" in html and "与已有评分或模型的比较" in html
    assert "验证集患者数" in html and "AUROC 区间方法" in html
    assert "AUROC bootstrap 重抽样次数" in html and "AUROC 跳过的重抽样次数" in html
    assert "差值（模型 − 比较对象）" in html and "差值 95% CI 下限" in html
    assert "参与比较记录数" in html and "信息窗口关系" in html
    for raw in ("validation subject n", "auroc bootstrap skipped n", "difference ci low",
                "information window relation", "comparison n"):
        assert raw not in html
    english = render("result_tables.json", payload, lang="en")
    assert "Validation patients" in english and "AUROC resamples skipped" in english
    assert "Difference (model − comparator)" in english and "Comparison with existing scores and models" in english



def test_coded_values_in_prediction_tables_read_by_name():
    headers = ["comparator_column", "comparator_kind", "metric", "interval_method", "calibration_status",
               "calibration_reason", "information_window_relation", "information_window_differs"]
    payload = {"tables": [
        {"name": "table_step_artifact_p__prediction_performance.csv",
         "label": "Table prediction_performance from step primary_performance.",
         "headers": ["auroc", "auroc_ci_method", "calibration_status"],
         "rows": [["0.8123", "patient_stratified_bootstrap_percentile_95pct", "not_estimable_perfect_separation"]]},
        {"name": "table_step_artifact_b__benchmark_comparison.csv",
         "label": "Table benchmark_comparison from step benchmark_comparison.",
         "headers": headers,
         "rows": [
             ["apache_iv", "score", "auroc", "delong_paired_normal_95pct", "calibration_not_compared",
              "score_scale", "comparator_ends_after_prediction_time", "True"],
             ["saps_ii_prob", "probability", "calibration_slope", "", "compared", "", "same", "False"],
             # A value outside its column's set reads as written.
             ["new_score", "ordinal", "brier_score", "new_method_95pct", "pending",
              "new_reason", "comparator_window_ahead", "maybe"],
         ]},
    ]}

    html = render("result_tables.json", payload)

    for name in ("按患者重抽样", "无法估计：结局被完全分离", "评分", "概率", "配对 DeLong", "未比较", "已比较",
                 "评分刻度不是概率", "比较对象窗口晚于预测时点结束", "比较对象窗口在预测时点结束",
                 "校准斜率", "Brier 分数", "<td>是</td>", "<td>否</td>"):
        assert name in html, name
    for code in ("patient_stratified_bootstrap_percentile_95pct", "not_estimable_perfect_separation",
                 "delong_paired_normal_95pct", "calibration_not_compared", "score_scale",
                 "comparator_ends_after_prediction_time", "<td>same</td>", "<td>True</td>"):
        assert code not in html, code
    for raw in ("<td>ordinal</td>", "<td>new_method_95pct</td>", "<td>pending</td>",
                "<td>new_reason</td>", "<td>comparator_window_ahead</td>", "<td>maybe</td>"):
        assert raw in html, raw
    english = render("result_tables.json", payload, lang="en")
    assert "resampled by patient" in english and "paired DeLong" in english
    assert "the score is not a probability" in english
    assert "comparator window ends after the prediction time" in english

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


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_saved_plan_does_not_invent_a_pre_execution_lifecycle(lang):
    html = render("agent_plan.json", {
        "research_question": "Synthetic study", "steps": [],
        "design_selection": {"candidates": [{"disposition": "selected", "reviewable_plan": ["Recorded setting"] * 6}]},
    }, lang=lang)
    assert "尚未开始患者数据分析" not in html
    assert "No patient-data analysis has started" not in html
    assert "候选研究计划" not in html and "Candidate research plan" not in html
    assert "尚未视为研究者确认" not in html and "not yet treated as researcher-confirmed" not in html
    assert ("研究计划" if lang == "zh" else "Research plan") in html


def test_cohort_summary_reads_recorded_concepts_outcome_and_window():
    html = render("cohort_summary.json", {
        "cohort": {"criteria": ["Synthetic adults"], "exclusion_criteria": [],
                   "target_outcome": "death", "time_window_hours": 24},
        "summary": {"selected_concepts": ["age", "sex", "lact", "death"],
                    "materialized_concept_count": 1, "coverage_sufficient": True},
    })
    for text in ("研究结局", "院内死亡", "时间窗", "24 小时", "年龄", "性别", "乳酸", "Synthetic adults"):
        assert text in html, text
    assert "4 字段" not in html and "3 字段" not in html
    assert "入住数" not in html and "患者数" not in html  # no denominator was recorded
    assert "未记录" not in html and "排除条件" not in html  # an unrecorded row is left out
    missing = render("cohort_summary.json", {"cohort": {}, "summary": {}})
    assert "没有记录队列定义" in missing
    assert "未记录" not in missing and "0 小时" not in missing


def test_core_estimates_and_bounded_table_omissions_are_named():
    html = render("result_tables.json", {"unpreviewed_table_count": 3, "tables": [{
        "name": "adjusted_association_estimates.csv",
        "label": "Table adjusted_association_estimates from step synthetic.",
        "headers": ["outcome_n", "outcome_event_n", "outcome_risk_pct"], "rows": [[24, 3, 12.5]],
        "preview_columns_truncated": True, "preview_truncated": True,
    }]})
    assert "没有核心结果表通过预览策略" not in html
    for text in ("调整后关联估计", "结局分母", "结局事件数", "结局风险 (%)", "部分行", "部分列", "3", "未展示"):
        assert text in html, text
    assert "有界预览" in html


def test_run_context_leads_with_the_question_and_keeps_identifiers_in_the_audit_block():
    html = render("run_context.json", {
        "question": "Synthetic question", "run_id": "run_x", "run_type": "full",
        "engine": "easyicu.research_agent.pipeline", "local_first": {"uploads": 0},
        "scientific_configuration_sha256": "a" * 64,
        "source": {"database": "miiv", "label": "synthetic export"},
        "summary": {"selected_concepts": ["age", "death"]},
    })
    for text in ("本次运行回答的问题", "Synthetic question", "MIMIC-IV", "年龄、院内死亡", "完整运行", "审计信息", "run_x"):
        assert text in html, text
    assert html.index("审计信息") > html.index("MIMIC-IV")
    for noise in ("表格视图", "原始 JSON", "字段</td>", "schema version", "1 字段", "a" * 64):
        assert noise not in html, noise


def test_quality_gate_reads_each_check_once_in_words():
    html = render("quality_gate.json", {"gate": {"status": "blocked", "reportable": False, "checks": [
        {"id": "execution_complete", "passed": True, "reason": None},
        {"id": "numeric_verified", "passed": False, "reason": "numeric_verified_not_satisfied"},
    ]}, "quality": []})
    for text in ("分析已执行完成", "分析步骤执行完成", "报告数值与结果一致", "已通过", "未满足"):
        assert text in html, text
    assert "numeric_verified_not_satisfied" not in html and "5 字段" not in html


def test_saved_plan_without_a_design_selection_is_not_judged_for_approval():
    html = render("agent_plan.json", {
        "research_question": "Synthetic saved plan", "analysis_type": "descriptive_epidemiology",
        "endpoint": None, "robustness_specs": [], "display_labels": {"death": "In-hospital death", "death=1": "Died"},
        "steps": [{"step_id": "s1", "intent": "Count stays.", "expected_outputs": ["table:cohort_flow"]}],
    })
    assert "为什么这一版还不能批准" not in html and "EasyICU 需要修订" not in html
    assert "要估计什么" not in html and "<p>—</p>" not in html
    assert "In-hospital death" in html and "尚未明确" not in html


def test_result_tables_leave_out_empty_and_index_columns_and_name_row_roles():
    html = render("result_tables.json", {"tables": [{
        "name": "table_x__exposure_outcome_distribution.csv",
        "headers": ["row_role", "exposure_level", "n_rows", "interval_method", "exposure_level_index", "exposure_ci_low_pct"],
        "rows": [["exposure_level", "0", "60461", "none_counts_only", "0", ""], ["overall", "", "94458", "none_counts_only", "", ""]],
    }]})
    for text in ("暴露组", "全部记录", "仅计数，无区间", "60,461"):
        assert text in html, text
    for hidden in ("exposure_level_index", "none_counts_only", ">exposure_level<"):
        assert hidden not in html, hidden

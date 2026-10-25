"""Synthetic plan facts, never model-written sentences, drive the short card."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.plan_projection import project_plan_conversation_preview
from tests.support.node import run_node


def example_plan(family="association_study"):
    return {
        "research_question": "A synthetic clinical question",
        "analysis_type": family,
        "cohort": {
            "inclusion": [{"concept_id": "age", "op": ">=", "value": 18,
                           "aggregation": "first", "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24}}],
            "exclusion": [],
        },
        "endpoint": {"name": "death", "kind": "binary"},
        "design_selection": {"candidates": [{
            "disposition": "selected", "analysis_type": family,
            "reviewable_plan": ["Do not show patient_stay_id or invented model prose."] * 6,
            "estimand": "The complete model remains available.",
        }]},
        "steps": [{
            "step_id": "synthetic_primary", "planned_analysis_role": "primary",
            "model_requirements": [{"analysis_role": "primary", "exposure_source": "lact",
                                    "outcome": "death", "covariates": ["age", "sex"],
                                    "analysis_set": "complete_case",
                                    "model_terms": [{"name": "lact", "role": "exposure", "coding": "continuous"}]}],
        }],
    }


def summary(plan, **kwargs):
    return project_plan_conversation_preview(plan, **kwargs)["clinical_summary"]


@pytest.mark.parametrize("family", ["association_study", "descriptive_epidemiology", "prediction_model"])
def test_five_ordered_lines_read_typed_fields_without_mutating_the_plan(family):
    plan = example_plan(family)
    if family == "descriptive_epidemiology":
        plan["steps"] = [{"table_one_spec": {"group_by": "sex", "group_levels": ["F", "M"]}}]
    elif family == "prediction_model":
        plan["steps"] = [{"planned_analysis_role": "primary",
                          "scientific_action_id": "prediction.discrimination_calibration",
                          "expected_outputs": ["table:prediction_scores", "table:model_performance"],
                          "inputs": ["age", "sex", "death", "patient_stay_id", "cohort:analysis_cohort", "support_only"]}]
    original = deepcopy(plan)
    result = summary(plan)
    rows = {row["key"]: row for row in result["lines"]}
    assert list(rows) == ["population", "comparison", "outcome", "adjustment", "data_checks"]
    assert rows["population"]["criteria"][0]["value"] == 18
    assert rows["population"]["criteria"][0]["op"] == ">="
    assert rows["outcome"]["variables"][0]["concept"] == "death"
    assert "patient_stay_id" not in json.dumps(result)
    assert "invented model prose" not in json.dumps(result)
    assert plan == original
    if family == "descriptive_epidemiology":
        assert rows["comparison"]["variables"] == [{"concept": "sex"}]
        assert rows["adjustment"]["mode"] == "descriptive"
    elif family == "prediction_model":
        assert rows["comparison"]["mode"] == "prediction"
        assert rows["adjustment"]["variables"] == [{"concept": "age"}, {"concept": "sex"}]
        assert "support_only" not in json.dumps(result)
        assert rows["data_checks"]["missing"] == "training_imputation"
        assert rows["adjustment"]["evaluation"] == "discrimination_calibration"
    else:
        assert rows["comparison"]["variables"] == [{"concept": "lact"}]
        assert rows["adjustment"]["variables"] == [{"concept": "age"}, {"concept": "sex"}]
        assert rows["data_checks"]["missing"] == "complete_case"


def test_timing_is_present_only_when_declared_and_never_borrowed_from_prose():
    plan = example_plan()
    assert summary(plan)["timing"] is None
    plan["endpoint"]["time_origin"] = "icu_admit"
    assert summary(plan)["timing"] == {"anchor": "icu_admit", "hours": 0, "risk_set": False}
    plan["endpoint"]["time_origin"] = "unsupported_column_id"
    assert summary(plan)["timing"] == {"unresolved": True}
    plan["steps"][0]["sensitivity_spec_ids"] = ["synthetic_landmark"]
    study = {"confirmations": {"plan_timing_landmark_24h": True}, "sensitivity_specs": [
        {"spec_id": "synthetic_landmark", "axis": "timing", "strategy": "landmark",
         "landmark_hours": 24, "require_alive_at_landmark": True},
    ]}
    assert summary(plan, study=study)["timing"] == {"anchor": "icu_admit", "hours": 24, "risk_set": True}
    plan["steps"][0]["sensitivity_spec_ids"] = []
    assert summary(plan, study=study)["timing"] == {"unresolved": True}


def test_uncertainty_stays_on_its_line_and_unknown_identity_never_becomes_a_name():
    plan = example_plan()
    plan["cohort"]["unapplied_population_criteria"] = ["private_internal_column"]
    plan["steps"][0]["model_requirements"][0].update(exposure_source="patient_stay_id", covariates=None)
    result = summary(plan)
    rows = {row["key"]: row for row in result["lines"]}
    assert rows["population"]["unresolved"] is True
    assert rows["comparison"]["unresolved"] is True
    assert rows["adjustment"]["unresolved"] is True
    assert "private_internal_column" not in json.dumps(result)
    assert "patient_stay_id" not in json.dumps(result)
    assert "patient_stay_id" not in json.dumps(project_plan_conversation_preview(plan))


def test_declared_group_thresholds_keep_the_studys_units_and_symbols():
    plan = example_plan()
    plan["steps"][0]["model_requirements"][0]["exposure_source"] = "lact_groups"
    groups = {"status": "shown", "groupings": [{
        "variable": "lact_groups", "concept": "lact", "status": "labelled",
        "reference": 0, "contrast": 1,
        "levels": [{"code": 0, "label": "<4 mmol/L", "reader_name": "<4 mmol/L"},
                   {"code": 1, "label": "≥4 mmol/L", "reader_name": "≥4 mmol/L"}],
    }]}
    row = summary(plan, exposure_groups=groups)["lines"][1]
    assert row["variables"] == [{"concept": "lact"}]
    assert row["groups"] == [{"en": "<4 mmol/L", "zh": "<4 mmol/L"}, {"en": "≥4 mmol/L", "zh": "≥4 mmol/L"}]


def test_secondary_models_cannot_supply_the_primary_summary():
    plan = example_plan()
    plan["steps"][0]["planned_analysis_role"] = "secondary"
    assert summary(plan)["lines"][1]["unresolved"] is True


def test_review_uncertainty_and_registered_names_are_kept_on_the_affected_line():
    plan = example_plan()
    plan["steps"][0]["model_requirements"][0]["exposure_source"] = "measured_marker"
    plan["display_labels"] = {"measured_marker": "最高乳酸"}
    row = summary(plan, review_summary={"finding_codes": ["POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"]})["lines"][1]
    assert row["variables"] == [{"concept": "measured_marker", "label": "最高乳酸"}]
    assert row["unresolved"] is True


def test_missing_categories_are_not_reported_as_complete_case():
    plan = example_plan()
    plan["steps"][0]["model_requirements"][0]["baseline_missing_handling"] = {
        "policy": "explicit_missing_category", "covariates": ["age"],
    }
    assert summary(plan)["lines"][4]["missing"] == "explicit_missing_category"


def test_a_declared_but_untyped_start_stays_visible_without_repeating_model_prose():
    plan = example_plan()
    plan["design_selection"]["candidates"][0]["time_zero"] = "Model prose with private_internal_column"
    assert summary(plan)["timing"] == {"unresolved": True}


def test_typed_summary_does_not_require_the_old_six_model_sentences():
    plan = example_plan()
    plan["design_selection"]["candidates"][0].pop("reviewable_plan")
    assert summary(plan)["lines"][1]["variables"] == [{"concept": "lact"}]


def test_invalid_prediction_shape_does_not_claim_an_implemented_missing_data_policy():
    plan = example_plan("prediction_model")
    plan["steps"] = [{"planned_analysis_role": "primary",
                      "scientific_action_id": "prediction.discrimination_calibration",
                      "inputs": ["age", "death", "cohort:analysis_cohort"],
                      "expected_outputs": ["table:unsupported"]}]
    assert summary(plan)["lines"][4]["missing"] == "unresolved"
    assert summary(plan)["lines"][3]["evaluation"] is None


def test_prediction_evaluation_names_only_the_metrics_claimed_by_the_typed_owner():
    plan = example_plan("prediction_model")
    plan["steps"] = [{"planned_analysis_role": "primary",
                      "scientific_action_id": "prediction.discrimination_calibration",
                      "inputs": ["age", "sex", "lact", "death", "cohort:analysis_cohort"],
                      "expected_outputs": ["table:prediction_scores", "table:model_performance"]}]
    assert summary(plan)["lines"][3]["evaluation"] == "discrimination_calibration"
    plan["steps"][0]["expected_outputs"] = ["table:unsupported"]
    assert summary(plan)["lines"][3]["evaluation"] is None
    plan["steps"][0]["scientific_action_id"] = "unsupported.synthetic_action"
    plan["steps"][0]["intent"] = "Evaluate discrimination and calibration"
    assert summary(plan)["lines"][3]["evaluation"] is None


def test_prediction_without_the_static_owner_still_names_its_declared_predictors():
    plan = example_plan("prediction_model")
    result = summary(plan)
    assert result["lines"][1]["variables"] == [{"concept": "age"}, {"concept": "sex"}]
    assert result["lines"][3]["evaluation"] is None
    assert result["lines"][3]["unresolved"] is False


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("family", ["association_study", "descriptive_epidemiology", "prediction_model"])
def test_python_projection_reaches_the_shared_bilingual_vocabulary(family, language):
    plan = example_plan(family)
    kwargs = {}
    if family == "descriptive_epidemiology":
        plan["cohort"] = {"selection_mode": "all_input_rows", "inclusion": [], "exclusion": []}
        plan["endpoint"] = {"name": "los_icu", "kind": "continuous"}
        plan["steps"] = [{"table_one_spec": {"group_by": "sex"}}]
    elif family == "prediction_model":
        plan["steps"] = [{"planned_analysis_role": "primary",
                          "scientific_action_id": "prediction.discrimination_calibration",
                          "inputs": ["age", "sex", "lact", "death", "cohort:analysis_cohort"],
                          "expected_outputs": ["table:prediction_scores", "table:model_performance"]}]
    else:
        plan["steps"][0]["model_requirements"][0]["exposure_source"] = "lact_groups"
        plan["steps"][0]["sensitivity_spec_ids"] = ["synthetic_landmark"]
        kwargs = {
            "study": {"confirmations": {"plan_timing_landmark_24h": True}, "sensitivity_specs": [
                {"spec_id": "synthetic_landmark", "strategy": "landmark", "landmark_hours": 24,
                 "require_alive_at_landmark": True},
            ]},
            "exposure_groups": {"groupings": [{"variable": "lact_groups", "concept": "lact",
                                              "reference": 0, "contrast": 1, "levels": [
                                                  {"code": 0, "label": "<4 mmol/L", "reader_name": "<4 mmol/L"},
                                                  {"code": 1, "label": "≥4 mmol/L", "reader_name": "≥4 mmol/L"},
                                              ]}]},
        }
    owner = Path(__file__).parents[3] / "src/easyicu/webserver/static/js/product-labels.js"
    script = f"""
global.window = {{}};
require({json.dumps(str(owner.with_name('data-catalog.js')))});
require({json.dumps(str(owner))});
const tr = (en, zh) => {json.dumps(language)} === 'zh' ? zh : en;
process.stdout.write(JSON.stringify(window.EU_PRODUCT_LABELS.clinicalPlan({json.dumps(summary(plan, **kwargs))}, tr)));
"""
    rendered = json.loads(run_node("node", script, check=True).stdout)
    assert len(rendered["lines"]) == 5
    labels = (["研究人群", "比较", "结局", "调整", "数据检查"] if language == "zh"
              else ["Population", "Comparison", "Outcome", "Adjustment", "Data checks"])
    values = (["年龄 ≥18 岁的 ICU 患者", "乳酸 <4 mmol/L 与 ≥4 mmol/L 两组", "院内死亡", "年龄、性别",
               "完整病例分析；运行前核对缺失与可用性"] if language == "zh"
              else ["ICU patients aged ≥18", "Lactate <4 vs ≥4 mmol/L", "In-hospital mortality", "Age, sex",
                    "Complete-case analysis; availability checked before the run"])
    if family == "descriptive_epidemiology":
        labels[1] = "分组" if language == "zh" else "Groups"
        values = (["所提供队列的全部患者", "按性别分层", "ICU 住院时长", "不做调整，仅分层描述",
                   "缺失处理待确认；运行前核对缺失与可用性"] if language == "zh"
                  else ["All patients in the supplied cohort", "By sex", "ICU length of stay",
                        "None; stratified description only",
                        "Missing-data handling to be confirmed; availability checked before the run"])
    elif family == "prediction_model":
        labels[1], labels[3] = ("预测变量", "评估") if language == "zh" else ("Predictors", "Evaluation")
        values[1] = "年龄、性别、乳酸" if language == "zh" else "Age, sex, lactate"
        values[3] = ("预测与实际结局对照（区分度与校准）" if language == "zh"
                     else "Predictions against observed outcomes (discrimination and calibration)")
        values[4] = ("仅用训练集信息填补缺失；运行前核对缺失与可用性" if language == "zh"
                     else "Imputation using training-set information only; availability checked before the run")
    assert [row["label"] for row in rendered["lines"]] == labels
    assert [row["text"] for row in rendered["lines"]] == values
    assert all(not row["text"].endswith(("。", ".")) for row in rendered["lines"])
    if family == "association_study":
        assert rendered["timing"] == ("从入 ICU 后第 24 小时开始随访，只纳入此时仍在院的患者。" if language == "zh"
                                      else "Follow-up starts 24 hours after ICU admission and includes only patients still in hospital then.")
    else:
        assert rendered["timing"] == ""
    assert "patient_stay_id" not in json.dumps(rendered)
    assert "invented model prose" not in json.dumps(rendered)
    if language == "zh":
        assert all(len(line["text"]) <= 48 for line in rendered["lines"])

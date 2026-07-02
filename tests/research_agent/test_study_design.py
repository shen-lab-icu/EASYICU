from __future__ import annotations

from pathlib import Path


def _context(ra, question: str, *, outcome: str = "death", exposure: str | None = None):
    return ra.schema.ResearchContext(
        research_question=question,
        cohort=ra.schema.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
            outcome_columns=[outcome],
        ),
        variables=[],
        target_outcome=outcome,
        primary_exposure=exposure,
    )


def test_study_design_brief_infers_distinct_question_families(ra):
    from easyicu.research_agent.study_design import infer_study_design_family

    cases = {
        "Is exposure X associated with in-hospital mortality after adjustment?": "association",
        "Build a mortality prediction model and report AUROC and calibration.": "prediction",
        "Estimate time-to-event survival using Cox regression.": "time_to_event",
        "Discover ICU phenotypes using clustering and trajectory features.": "phenotyping",
        "Emulate a target trial for treatment effect using propensity weighting.": "causal_emulation",
    }

    for question, expected in cases.items():
        assert infer_study_design_family(_context(ra, question)) == expected


def test_study_design_brief_infers_chinese_canonical_question_families(ra):
    from easyicu.research_agent.study_design import infer_study_design_family

    cases = {
        "六个 ICU 数据库中患者年龄、性别、LOS、器官支持和死亡率分布如何不同？": "descriptive",
        "用 ICU 前 24h vitals + labs 预测院内死亡，跨数据库性能如何？": "prediction",
        "估计生存时间到事件并报告风险比。": "time_to_event",
        "使用聚类发现 ICU 轨迹表型。": "phenotyping",
        "vasopressor exposure 与 mortality 的关联是否受 indication bias 混杂？": "causal_emulation",
    }

    for question, expected in cases.items():
        assert infer_study_design_family(_context(ra, question)) == expected


def test_e1_prevalence_plus_adjusted_association_routes_to_association(ra):
    from easyicu.research_agent.study_design import build_study_design_brief

    question = (
        "Among adult ICU patients in MIMIC-IV, what is the prevalence of "
        "Sepsis-3, and is Sepsis-3 status associated with in-hospital mortality "
        "after adjusting for baseline illness severity? Report the Sepsis-3 "
        "prevalence, the adjusted association with in-hospital mortality, and "
        "your cohort attrition."
    )

    brief = build_study_design_brief(
        _context(ra, question, outcome="death", exposure="sepsis3")
    )

    assert brief.analysis_family == "association"
    roles = {module.role for module in brief.display_modules if module.tier == "core"}
    assert {"cohort_accounting", "baseline_context", "data_quality", "primary_estimand", "robustness"} <= roles
    modules = {module.module_id: module for module in brief.display_modules}
    assert "prevalence_or_event_rate_question" in " ".join(brief.adaptive_triggers)
    assert modules["exposure_outcome_distribution"].role == "descriptive_result"


def test_prediction_brief_requires_prediction_specific_displays(ra):
    from easyicu.research_agent.study_design import (
        build_study_design_brief,
        render_study_design_brief_for_prompt,
    )

    brief = build_study_design_brief(
        _context(
            ra,
            "Build a risk prediction model for mortality with AUROC, calibration, "
            "and decision-curve thresholds.",
        )
    )

    assert brief.analysis_family == "prediction"
    assert any("calibration" in item.lower() for item in brief.main_text_displays)
    assert any("hyperparameter" in item.lower() for item in brief.supplementary_displays)
    assert "leakage" in brief.covariate_strategy.lower()
    modules = {module.module_id: module for module in brief.display_modules}
    assert modules["discrimination"].role == "model_performance"
    assert modules["calibration"].tier == "core"
    assert modules["clinical_utility"].tier == "conditional"
    assert "display_playbook" in render_study_design_brief_for_prompt(brief)


def test_association_plan_validator_flags_single_display_plan(ra, tmp_path: Path):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
    from easyicu.research_agent.study_design import (
        build_study_design_brief,
        validate_plan_against_study_design_brief,
    )

    context = _context(
        ra,
        "Estimate whether exposure X is associated with mortality after adjustment.",
        exposure="x",
    )
    brief = build_study_design_brief(context)
    plan = AnalysisPlan(
        research_question=context.research_question,
        steps=[
            AnalysisStep(
                step_id="01_model",
                intent="Fit one adjusted logistic regression model.",
                method="logistic regression",
                expected_outputs=["figure:forest_plot"],
            )
        ],
    )

    findings = validate_plan_against_study_design_brief(plan=plan, brief=brief)

    assert findings
    assert any(f.validator == "study_design_brief" for f in findings)
    missing = " ".join(str(f.detail) for f in findings if f.detail)
    assert "Table 1 baseline characteristics" in missing
    assert "sensitivity / robustness summary" in missing
    assert any("too narrow" in f.message for f in findings)


def test_association_plan_validator_accepts_article_level_display_suite(ra):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
    from easyicu.research_agent.study_design import (
        build_study_design_brief,
        validate_plan_against_study_design_brief,
    )

    context = _context(
        ra,
        "Estimate whether exposure X is associated with mortality after adjustment.",
        exposure="x",
    )
    brief = build_study_design_brief(context)
    plan = AnalysisPlan(
        research_question=context.research_question,
        rationale=(
            "Plan includes cohort attrition, Table 1, adjusted association, "
            "missingness/data quality, and robustness sensitivity analyses."
        ),
        steps=[
            AnalysisStep(
                step_id="01_cohort_flow",
                intent="Define cohort eligibility and attrition denominator.",
                expected_outputs=["table:cohort_flow"],
            ),
            AnalysisStep(
                step_id="02_table_one",
                intent="Render Table 1 baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            AnalysisStep(
                step_id="03_primary",
                intent="Fit adjusted association model and report odds ratio.",
                method="adjusted logistic regression",
                expected_outputs=["table:primary_adjusted_effect_estimate"],
            ),
            AnalysisStep(
                step_id="04_missingness",
                intent="Audit missingness and data quality.",
                expected_outputs=["table:missingness_data_quality"],
            ),
            AnalysisStep(
                step_id="05_sensitivity",
                intent="Run robustness variants over exposure, cohort, and missing data.",
                expected_outputs=["figure:sensitivity_robustness_summary"],
            ),
        ],
    )

    findings = validate_plan_against_study_design_brief(plan=plan, brief=brief)

    assert not [f for f in findings if "main-text displays" in f.message]


def test_prediction_plan_validator_accepts_multi_module_prediction_suite(ra):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
    from easyicu.research_agent.study_design import (
        build_study_design_brief,
        validate_plan_against_study_design_brief,
    )

    context = _context(
        ra,
        "Build a mortality prediction model with external validation, AUROC, calibration, "
        "and triage threshold utility.",
    )
    brief = build_study_design_brief(context)
    plan = AnalysisPlan(
        research_question=context.research_question,
        rationale=(
            "Plan covers modelling denominator, Table 1, external validation, "
            "discrimination, calibration, missingness/leakage audit, and net benefit."
        ),
        steps=[
            AnalysisStep(
                step_id="01_flow",
                intent="Define modelling cohort denominator and train-test validation split.",
                expected_outputs=["table:modelling_cohort_flow"],
            ),
            AnalysisStep(
                step_id="02_baseline",
                intent="Compare Table 1 baseline characteristics across development and validation sets.",
                expected_outputs=["table:baseline_table"],
            ),
            AnalysisStep(
                step_id="03_model",
                intent="Fit prediction model and report ROC AUROC and AUPRC discrimination metrics.",
                method="regularized logistic regression",
                expected_outputs=["figure:roc_curve", "table:auroc"],
            ),
            AnalysisStep(
                step_id="04_calibration",
                intent="Plot calibration curve and Brier score.",
                expected_outputs=["figure:calibration_plot"],
            ),
            AnalysisStep(
                step_id="05_quality",
                intent="Audit feature missingness, preprocessing, and leakage.",
                expected_outputs=["table:missingness_leakage_audit"],
            ),
            AnalysisStep(
                step_id="06_utility",
                intent="Assess clinical utility with decision curve and threshold net benefit.",
                expected_outputs=["figure:decision_curve"],
            ),
        ],
    )

    findings = validate_plan_against_study_design_brief(plan=plan, brief=brief)

    assert not [f for f in findings if "core article-display modules" in f.message]
    assert not [f for f in findings if "too narrow" in f.message]


def test_family_playbooks_are_distinct_and_not_effect_only(ra):
    from easyicu.research_agent.study_design import build_study_design_brief

    cases = {
        "association": "Is exposure X associated with mortality after adjustment?",
        "prediction": "Build a mortality prediction model and validate calibration.",
        "time_to_event": "Estimate survival time-to-event with Cox regression.",
        "phenotyping": "Discover trajectory phenotypes using clustering.",
        "causal_emulation": "Emulate a target trial for treatment effect using IPTW.",
        "descriptive": "Describe age, sex, LOS, organ support, and mortality distributions.",
    }
    role_sets = {}
    for family, question in cases.items():
        brief = build_study_design_brief(_context(ra, question))
        roles = {module.role for module in brief.display_modules if module.tier == "core"}
        role_sets[family] = roles
        assert len(roles) >= 4
        assert roles != {"primary_estimand"}

    assert "calibration" in role_sets["prediction"]
    assert "temporal_absolute_risk" in role_sets["time_to_event"]
    assert "phenotype_structure" in role_sets["phenotyping"]
    assert "causal_protocol" in role_sets["causal_emulation"]


def test_cross_database_context_adds_transportability_module(ra):
    from easyicu.research_agent.study_design import build_study_design_brief

    context = _context(
        ra,
        "Compare mortality distributions across six ICU databases.",
    ).model_copy(update={"cross_database_validation": ["eicu", "hirid"]})

    brief = build_study_design_brief(context)

    assert any("cross_database" in trigger for trigger in brief.adaptive_triggers)
    modules = {module.module_id: module for module in brief.display_modules}
    assert "cross_database_heterogeneity" in modules
    assert modules["cross_database_heterogeneity"].role == "transportability"

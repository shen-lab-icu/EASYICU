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
    from easyicu.research_agent.planning.study_design import infer_study_design_family

    cases = {
        "Is exposure X associated with in-hospital mortality after adjustment?": "association",
        "Build a mortality prediction model and report AUROC and calibration.": "prediction",
        "Estimate time-to-event survival using Cox regression.": "time_to_event",
        "Discover ICU phenotypes using clustering and trajectory features.": "phenotyping",
        "Emulate a target trial for treatment effect using propensity weighting.": "causal_emulation",
    }

    for question, expected in cases.items():
        assert infer_study_design_family(_context(ra, question)) == expected


def test_survival_question_phrased_as_association_routes_to_time_to_event(ra):
    """H1 regression: a time-to-event question that uses the word "association".

    The canonical H1 item asks to "estimate the ASSOCIATION between mechanical
    ventilation and 28-day mortality" while respecting exposure timing and
    censoring -- i.e. a survival study worded as an association. The keyword
    cascade used to stop at "association" (the literal word), so the figure
    renderer and the methodological-rigor auditor -- both keyed on this
    function -- routed to the association family and never fired the survival
    figure / method-match check, even though the plan contract (keyed on the
    richer analysis-type scorer) correctly built a survival step. The design
    family must now agree with the plan contract.
    """
    from easyicu.research_agent.planning.study_design import infer_study_design_family

    question = (
        "Among adult ICU patients in MIMIC-IV, estimate the association between "
        "first-24h mechanical ventilation status/duration and in-hospital "
        "(28-day) mortality. Respect exposure timing, define time zero, and "
        "handle censoring with a Cox proportional-hazards / Kaplan-Meier "
        "time-to-event analysis."
    )
    ctx = _context(ra, question, outcome="death", exposure="vent")
    assert infer_study_design_family(ctx) == "time_to_event"


def test_study_design_brief_infers_chinese_canonical_question_families(ra):
    from easyicu.research_agent.planning.study_design import infer_study_design_family

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
    from easyicu.research_agent.planning.study_design import build_study_design_brief

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
    assert {
        "cohort_accounting",
        "baseline_context",
        "data_quality",
        "primary_estimand",
        "robustness",
    } <= roles
    modules = {module.module_id: module for module in brief.display_modules}
    assert "prevalence_or_event_rate_question" in " ".join(brief.adaptive_triggers)
    assert modules["exposure_outcome_distribution"].role == "descriptive_result"


def test_prediction_brief_requires_prediction_specific_displays(ra):
    from easyicu.research_agent.planning.study_design import (
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
    assert any(
        "hyperparameter" in item.lower() for item in brief.supplementary_displays
    )
    assert "leakage" in brief.covariate_strategy.lower()
    modules = {module.module_id: module for module in brief.display_modules}
    assert modules["discrimination"].role == "model_performance"
    assert modules["calibration"].tier == "core"
    assert modules["clinical_utility"].tier == "conditional"
    assert "display_playbook" in render_study_design_brief_for_prompt(brief)


def test_association_plan_validator_flags_single_display_plan(ra, tmp_path: Path):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
    from easyicu.research_agent.planning.study_design import (
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


def test_display_coverage_requires_exact_structured_product_not_decoy(ra):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
    from easyicu.research_agent.planning.study_design import (
        _module_covered,
        _structured_plan_declarations,
        build_study_design_brief,
    )

    context = _context(
        ra,
        "Estimate an adjusted association and report the analytic cohort.",
        exposure="x",
    )
    brief = build_study_design_brief(context)
    cohort_module = next(
        module for module in brief.display_modules if module.module_id == "cohort_flow"
    )
    decoy = AnalysisPlan(
        research_question=context.research_question,
        steps=[
            AnalysisStep(
                step_id="01_decoy",
                intent="A different product whose prose mentions flow.",
                method="not_a_cohort_flow",
                expected_outputs=["table:cohort_flow_prediction_features"],
            )
        ],
    )
    truthful = decoy.model_copy(
        update={
            "steps": [
                decoy.steps[0].model_copy(
                    update={
                        "method": "cohort_definition",
                        "expected_outputs": ["table:cohort_flow"],
                    }
                )
            ]
        }
    )

    _decoy_methods, decoy_outputs = _structured_plan_declarations(decoy)
    _truthful_methods, truthful_outputs = _structured_plan_declarations(truthful)
    assert not _module_covered(cohort_module, decoy_outputs)
    assert _module_covered(cohort_module, truthful_outputs)

    for wrong_output in ("log:cohort_flow", "test:cohort_flow", "cohort_flow"):
        wrong_kind = truthful.model_copy(
            update={
                "steps": [
                    truthful.steps[0].model_copy(
                        update={"expected_outputs": [wrong_output]}
                    )
                ]
            }
        )
        _methods, outputs = _structured_plan_declarations(wrong_kind)
        assert not _module_covered(cohort_module, outputs), wrong_output


def test_data_quality_article_contract_stays_measurement_only(ra):
    from easyicu.research_agent.reporting.article_contract import (
        build_article_analysis_contract,
    )

    context = _context(
        ra,
        "Audit bilirubin and vasopressor measurement completeness.",
    )
    contract = build_article_analysis_contract(
        context,
        analysis_type="data_quality_audit",
    )

    assert contract.required_roles == ["data_quality"]
    assert {
        requirement.module_id for requirement in contract.requirements
    } == {"missingness_data_quality", "measurement_process_audit"}


def test_article_contract_flags_and_can_augment_narrow_association_plan(ra):
    from easyicu.research_agent.reporting.article_contract import (
        augment_plan_for_article_contract,
        build_article_analysis_contract,
        roles_covered_by_plan,
        validate_plan_against_article_contract,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    context = _context(
        ra,
        "Estimate exposure prevalence and whether exposure X is associated "
        "with mortality after adjustment.",
        exposure="x",
    )
    contract = build_article_analysis_contract(context)
    narrow = AnalysisPlan(
        research_question=context.research_question,
        steps=[
            AnalysisStep(
                step_id="01_forest",
                intent="Fit adjusted model and draw a forest plot.",
                method="logistic regression",
                planned_analysis_role="primary",
                expected_outputs=[
                    "table:adjusted_effect",
                    "figure:forest_plot",
                ],
            )
        ],
    )

    findings = validate_plan_against_article_contract(
        plan=narrow,
        contract=contract,
    )
    assert findings
    assert "baseline_context" in findings[0].detail["missing_roles"]
    assert "data_quality" in findings[0].detail["missing_roles"]
    assert "descriptive_result" in findings[0].detail["missing_roles"]

    augmented, augment_findings = augment_plan_for_article_contract(
        plan=narrow,
        contract=contract,
    )

    assert augment_findings
    assert len(augmented.steps) > len(narrow.steps)
    covered = roles_covered_by_plan(augmented, contract)
    assert set(contract.required_roles) <= covered


def test_article_contract_ignores_role_words_in_step_prose_and_wrong_kinds(ra):
    from easyicu.research_agent.reporting.article_contract import (
        build_article_analysis_contract,
        roles_covered_by_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    context = _context(
        ra,
        "Estimate an adjusted association and report the analytic cohort.",
        exposure="x",
    )
    contract = build_article_analysis_contract(context)
    decoy = AnalysisPlan(
        research_question=context.research_question,
        steps=[
            AnalysisStep(
                step_id="cohort_flow_and_table_one_decoy",
                intent=(
                    "This note mentions cohort flow, Table 1, missingness, "
                    "robustness, and a forest plot without producing them."
                ),
                method="narrative_note",
                expected_outputs=["log:cohort_flow", "test:forest_plot"],
            )
        ],
    )

    covered = roles_covered_by_plan(decoy, contract)

    assert "cohort_accounting" not in covered
    assert "baseline_context" not in covered
    assert "primary_estimand" not in covered


def test_model_specification_review_prose_does_not_claim_robustness_artifact(
    ra,
    tmp_path,
):
    import json

    from easyicu.research_agent.reporting.article_contract import (
        build_article_analysis_contract,
        roles_covered_by_artifacts,
    )

    context = _context(
        ra,
        "Estimate an adjusted association and report the analytic cohort.",
        exposure="x",
    )
    contract = build_article_analysis_contract(context)
    publication_dir = tmp_path / "publication_figures"
    publication_dir.mkdir()
    (publication_dir / "primary.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "primary",
                "core_claim": "The adjusted association is displayed.",
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Adjusted association",
                        "role": "primary_estimand",
                        "claim": "The registered primary estimate is shown.",
                        "review_risk": (
                            "Interpretability depends on the upstream model "
                            "specification and validator findings."
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    covered = roles_covered_by_artifacts(
        contract=contract,
        evidence_records=[],
        per_step_records=[],
        run_dir=tmp_path,
    )

    assert "robustness" not in covered


def test_planner_article_contract_retries_missing_robustness_instead_of_faking_it(
    ra,
):
    import json
    import pytest

    from easyicu.research_agent.agents.core import (
        PlannerAgent,
        PlannerArticleContractError,
    )
    from easyicu.research_agent.providers.mocks import ExternalCaptureMockLLMClient
    from easyicu.research_agent.reporting.article_contract import (
        build_article_analysis_contract,
        render_article_analysis_contract_for_prompt,
    )

    context = _context(
        ra,
        "Estimate an adjusted association and report the analytic cohort.",
        exposure="x",
    )
    payload = {
        "research_question": context.research_question,
        "analysis_type": "association_study",
        "steps": [
            {
                "step_id": "01_cohort",
                "planned_analysis_role": "auxiliary",
                "intent": "Report cohort accounting.",
                "expected_outputs": ["table:cohort_attrition"],
            },
            {
                "step_id": "02_baseline",
                "planned_analysis_role": "auxiliary",
                "intent": "Report baseline characteristics.",
                "expected_outputs": ["table:baseline_characteristics"],
            },
            {
                "step_id": "03_quality",
                "planned_analysis_role": "auxiliary",
                "intent": "Audit data quality.",
                "expected_outputs": ["table:missingness_profile"],
            },
            {
                "step_id": "04_descriptive",
                "planned_analysis_role": "auxiliary",
                "intent": "Report absolute outcome risk.",
                "expected_outputs": ["table:outcome_incidence"],
            },
            {
                "step_id": "05_primary",
                "planned_analysis_role": "primary",
                "intent": "Estimate the adjusted association.",
                "expected_outputs": ["table:adjusted_association_estimates"],
                "method": "adjusted_association_models",
                "model_requirements": [
                    {
                        "requirement_id": "primary_x_death",
                        "outcome": "death",
                        "outcome_type": "binary",
                        "method_family": "binary_logistic_regression",
                        "exposure_source": "x",
                        "analysis_role": "primary",
                        "analysis_set": "complete_case",
                        "required_for_step_success": True,
                    }
                ],
            },
        ],
    }
    planner = PlannerAgent(object())  # _parse is transport-free.

    with pytest.raises(
        PlannerArticleContractError,
        match="robustness, robustness_specs",
    ):
        planner._parse(
            json.dumps(payload),
            context,
            enforce_article_contract=True,
        )
    missing_robustness_payload = json.loads(json.dumps(payload))

    payload["robustness_specs"] = [
        {
            "spec_id": "alt_missing_median",
            "axis": "missing",
            "description": "Median-impute the declared baseline covariate.",
            "missing_override": {"strategy": "median_imputation"},
        }
    ]
    payload["steps"].append(
        {
            "step_id": "06_robustness",
            "planned_analysis_role": "auxiliary",
            "intent": "Replay the primary model under the locked sensitivity spec.",
            "inputs": ["table:adjusted_association_estimates"],
            "expected_outputs": [
                "table:robustness_matrix",
                "statistic:robustness_summary",
            ],
            "method": "robustness_sensitivity",
        }
    )

    parsed = planner._parse(
        json.dumps(payload),
        context,
        enforce_article_contract=True,
    )

    assert parsed.steps[-1].method == "robustness_sensitivity"
    assert [spec.spec_id for spec in parsed.robustness_specs] == ["alt_missing_median"]

    contract = build_article_analysis_contract(context)
    rendered_contract = render_article_analysis_contract_for_prompt(contract)
    capture = ExternalCaptureMockLLMClient(
        [
            json.dumps(missing_robustness_payload),
            json.dumps(payload),
        ]
    )
    recovered = PlannerAgent(capture).run(
        context,
        enforce_article_contract=True,
        planning_contract_context=rendered_contract,
    )

    assert recovered.steps[-1].method == "robustness_sensitivity"
    assert len(capture.calls) == 2
    initial_prompt = "\n".join(
        message.content for message in capture.calls[0][0]
    )
    retry_prompt = "\n".join(message.content for message in capture.calls[1][0])
    assert "HOST-DERIVED PRE-PLAN DESIGN PROFILE" in initial_prompt
    assert rendered_contract in initial_prompt
    assert (
        "required_article_roles: baseline_context, cohort_accounting, "
        "data_quality, descriptive_result, primary_estimand, robustness"
        in initial_prompt
    )
    assert "robustness_specs (array; non-empty when the binding contract" in retry_prompt
    assert "method='robustness_sensitivity'" in retry_prompt

    payload["steps"] = [
        step for step in payload["steps"] if step["step_id"] != "01_cohort"
    ]
    with pytest.raises(PlannerArticleContractError) as caught:
        planner._parse(
            json.dumps(payload),
            context,
            enforce_article_contract=True,
        )
    assert "cohort_accounting" in str(caught.value)
    assert "table:cohort_flow" in str(caught.value)


def test_nonprimary_output_prefixes_do_not_satisfy_primary_estimand_role(ra):
    from easyicu.research_agent.reporting.article_contract import (
        build_article_analysis_contract,
        roles_covered_by_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    context = _context(
        ra,
        "Estimate an adjusted association and report the analytic cohort.",
        exposure="x",
    )
    contract = build_article_analysis_contract(context)
    for prefix in ("secondary", "supporting", "alternative"):
        decoy = AnalysisPlan(
            research_question=context.research_question,
            steps=[
                AnalysisStep(
                    step_id=f"{prefix}_effect",
                    intent=f"Report a {prefix} effect only.",
                    method="descriptive",
                    expected_outputs=[f"table:{prefix}_adjusted_effect_table"],
                )
            ],
        )
        assert "primary_estimand" not in roles_covered_by_plan(decoy, contract), prefix


def test_sensitivity_role_cannot_cover_primary_estimand_plan_or_artifact(ra, tmp_path):
    from easyicu.research_agent.reporting.article_contract import (
        build_article_analysis_contract,
        roles_covered_by_artifacts,
        roles_covered_by_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    context = _context(
        ra,
        "Estimate an adjusted association and report the analytic cohort.",
        exposure="x",
    )
    contract = build_article_analysis_contract(context)
    step = AnalysisStep(
        step_id="06_sensitivity_association",
        intent="Fit a sensitivity association model.",
        method="adjusted_association",
        planned_analysis_role="sensitivity",
        expected_outputs=["table:association_estimates"],
    )
    plan = AnalysisPlan(research_question=context.research_question, steps=[step])
    record = {
        "step_id": step.step_id,
        "status": "ok",
        "planned_analysis_role": "sensitivity",
        "analysis_request": {"step": step.model_dump(mode="json")},
        "step_summary": {
            "output_files": {"table:association_estimates": "association_estimates.csv"}
        },
    }

    assert "primary_estimand" not in roles_covered_by_plan(plan, contract)
    assert "primary_estimand" not in roles_covered_by_artifacts(
        contract=contract,
        evidence_records=[],
        per_step_records=[record],
        run_dir=tmp_path,
    )

    primary_step = step.model_copy(update={"planned_analysis_role": "primary"})
    primary_record = {
        **record,
        "planned_analysis_role": "primary",
        "analysis_request": {"step": primary_step.model_dump(mode="json")},
    }
    assert "primary_estimand" in roles_covered_by_plan(
        AnalysisPlan(
            research_question=context.research_question,
            steps=[primary_step],
        ),
        contract,
    )
    assert "primary_estimand" in roles_covered_by_artifacts(
        contract=contract,
        evidence_records=[],
        per_step_records=[primary_record],
        run_dir=tmp_path,
    )

    figure_step = AnalysisStep(
        step_id="07_primary_figure",
        intent="Render the Planner-owned primary result.",
        method="publication_figure_generation",
        planned_analysis_role="auxiliary",
        inputs=["table:association_estimates"],
        expected_outputs=["figure:forest_plot"],
    )
    figure_record = {
        "step_id": figure_step.step_id,
        "status": "ok",
        "planned_analysis_role": "auxiliary",
        "analysis_request": {"step": figure_step.model_dump(mode="json")},
        "step_summary": {"output_files": {"figure:forest_plot": "forest_plot.svg"}},
    }
    silent_primary_record = {**primary_record, "step_summary": {}}
    assert "primary_estimand" not in roles_covered_by_artifacts(
        contract=contract,
        evidence_records=[],
        per_step_records=[silent_primary_record, figure_record],
        run_dir=tmp_path,
    )

    import hashlib
    import json

    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    evidence_path = evidence_dir / "primary_estimate__association_estimates.csv"
    evidence_path.write_text("term,estimate\nx,1.2\n", encoding="utf-8")
    evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    evidence_id = "primary_estimate"
    primary_with_evidence = {
        **silent_primary_record,
        "evidence_ids": [evidence_id],
    }
    resolved_dir = tmp_path / "resolved_inputs"
    resolved_dir.mkdir()
    resolved_path = resolved_dir / f"{figure_step.step_id}.json"
    identity_row = {
        "input_key": "table:association_estimates",
        "declared_kind": "table",
        "product": "association_estimates",
        "evidence_id": evidence_id,
        "sha256": evidence_sha,
        "produced_by_step": primary_step.step_id,
    }
    resolved_path.write_text(
        json.dumps(
            {
                "schema_version": "2.1",
                "step_id": figure_step.step_id,
                "planner_declared_inputs": ["table:association_estimates"],
                "inputs": {
                    "table:association_estimates": {
                        **identity_row,
                        "identity_row": identity_row,
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    figure_with_binding = {
        **figure_record,
        "resolved_inputs_path": str(resolved_path.relative_to(tmp_path)),
        "resolved_inputs_sha256": hashlib.sha256(
            resolved_path.read_bytes()
        ).hexdigest(),
        "resolved_input_evidence_ids": [evidence_id],
    }
    evidence_record = {
        "evidence_id": evidence_id,
        "kind": "table",
        "description": "Primary adjusted association estimate",
        "relative_path": str(evidence_path.relative_to(tmp_path)),
        "sha256": evidence_sha,
        "produced_by_step": primary_step.step_id,
    }
    assert "primary_estimand" in roles_covered_by_artifacts(
        contract=contract,
        evidence_records=[evidence_record],
        per_step_records=[primary_with_evidence, figure_with_binding],
        run_dir=tmp_path,
    )

    publication_dir = tmp_path / "publication_figures"
    publication_dir.mkdir()
    (publication_dir / "decoy.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "decoy",
                "title": "Sensitivity forest plot",
                "core_claim": "Primary adjusted effect estimate",
            }
        ),
        encoding="utf-8",
    )
    assert "primary_estimand" not in roles_covered_by_artifacts(
        contract=contract,
        evidence_records=[],
        per_step_records=[silent_primary_record],
        run_dir=tmp_path,
    )


def test_unmapped_family_negations_do_not_trigger_second_keyword_router(ra):
    from easyicu.research_agent.planning.analysis_types import infer_analysis_type
    from easyicu.research_agent.planning.study_design import infer_study_design_family

    examples = (
        (
            "Evaluate a dynamic treatment policy using reinforcement learning; "
            "avoid causal interpretation.",
            "reinforcement_learning",
        ),
        (
            "Fuse clinical notes and waveforms into a multimodal representation; "
            "do not perform survival analysis.",
            "multimodal",
        ),
    )
    for question, expected_analysis_type in examples:
        context = _context(ra, question, exposure="x")
        assert infer_analysis_type(context).key == expected_analysis_type
        assert infer_study_design_family(context) == "descriptive"


def test_noncausal_treatment_response_uses_descriptive_not_causal_contract(ra):
    from easyicu.research_agent.planning.analysis_types import infer_analysis_type
    from easyicu.research_agent.planning.study_design import build_study_design_brief
    from easyicu.research_agent.reporting.article_contract import (
        build_article_analysis_contract,
    )

    context = _context(
        ra,
        "Characterize treatment responders and heterogeneity descriptively "
        "without causal interpretation.",
        exposure="treatment",
    )
    assert infer_analysis_type(context).key == "treatment_response"
    brief = build_study_design_brief(context)
    contract = build_article_analysis_contract(context, brief=brief)
    assert brief.analysis_family == "descriptive"
    assert contract.source_analysis_type == "treatment_response"
    assert contract.planner_owned_result_roles == ["heterogeneity"]
    assert not {
        "causal_protocol",
        "balance_positivity",
        "causal_contrast",
    } & set(contract.required_roles)


def test_run_contract_uses_planner_final_analysis_type_over_question_inference(
    ra,
    tmp_path,
):
    from easyicu.research_agent.reporting.article_contract import (
        summarize_article_contract_coverage,
    )

    context = _context(
        ra,
        "Describe longitudinal ICU measurements.",
        exposure="marker",
    )
    plan = ra.AnalysisPlan(
        research_question=context.research_question,
        analysis_type="trajectory_clustering",
        steps=[
            ra.AnalysisStep(
                step_id="01_primary_phenotypes",
                intent="Discover longitudinal phenotypes.",
                method="trajectory_clustering",
                planned_analysis_role="primary",
                expected_outputs=["table:cluster_assignments"],
            )
        ],
    )

    status = summarize_article_contract_coverage(
        context=context,
        plan=plan,
        evidence_records=[],
        per_step_records=[],
        run_dir=tmp_path,
    )

    assert status["article_contract_family"] == "phenotyping"
    assert status["article_contract"]["source_analysis_type"] == "trajectory_clustering"


def test_headline_roles_across_method_families_require_primary_lineage(ra):
    from easyicu.research_agent.reporting.article_contract import (
        build_article_analysis_contract,
        roles_covered_by_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    cases = (
        (
            "Build a mortality prediction model and report AUROC and calibration.",
            "model_performance",
            ["table:model_performance", "figure:roc_curve"],
        ),
        (
            "Build a mortality prediction model and report AUROC and calibration.",
            "calibration",
            ["table:model_performance", "figure:calibration_plot"],
        ),
        (
            "Estimate time-to-event survival using Cox regression.",
            "survival_effect",
            ["table:cox_summary"],
        ),
        (
            "Discover ICU phenotypes using clustering and trajectory features.",
            "phenotype_structure",
            ["table:cluster_assignments", "figure:cluster_heatmap"],
        ),
        (
            "Emulate a target trial using IPTW and estimate a causal contrast.",
            "causal_contrast",
            ["table:causal_contrast"],
        ),
    )
    for question, role, outputs in cases:
        context = _context(ra, question, exposure="x")
        contract = build_article_analysis_contract(context)
        sensitivity = AnalysisStep(
            step_id="02_sensitivity",
            intent="Produce an alternative result.",
            method="sensitivity_analysis",
            planned_analysis_role="sensitivity",
            expected_outputs=outputs,
        )
        assert role not in roles_covered_by_plan(
            AnalysisPlan(research_question=question, steps=[sensitivity]),
            contract,
        )
        primary = sensitivity.model_copy(
            update={
                "step_id": "01_primary",
                "planned_analysis_role": "primary",
            }
        )
        assert role in roles_covered_by_plan(
            AnalysisPlan(research_question=question, steps=[primary]),
            contract,
        )


def test_association_plan_validator_accepts_article_level_display_suite(ra):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
    from easyicu.research_agent.planning.study_design import (
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
                expected_outputs=[
                    "table:primary_adjusted_effect_estimate",
                    "statistic:exposure_prevalence",
                    "table:absolute_outcome_risk_by_exposure",
                ],
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
    from easyicu.research_agent.planning.study_design import (
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
                expected_outputs=[
                    "table:modelling_cohort_flow",
                    "table:validation_split",
                ],
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
    from easyicu.research_agent.planning.study_design import build_study_design_brief

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
        roles = {
            module.role for module in brief.display_modules if module.tier == "core"
        }
        role_sets[family] = roles
        assert len(roles) >= 4
        assert roles != {"primary_estimand"}

    assert "calibration" in role_sets["prediction"]
    assert "temporal_absolute_risk" in role_sets["time_to_event"]
    assert "phenotype_structure" in role_sets["phenotyping"]
    assert "causal_protocol" in role_sets["causal_emulation"]


def test_article_figure_strategies_are_family_specific(ra):
    from easyicu.research_agent.planning.figure_strategy import (
        build_article_figure_strategy,
        render_article_figure_strategy_for_prompt,
    )

    cases = {
        "association": (
            "Is exposure X associated with mortality after adjustment?",
            "descriptive_result",
        ),
        "prediction": (
            "Build a mortality prediction model with AUROC and calibration.",
            "calibration",
        ),
        "time_to_event": (
            "Estimate survival time-to-event with Cox regression.",
            "temporal_absolute_risk",
        ),
        "phenotyping": (
            "Discover ICU trajectory phenotypes using clustering.",
            "phenotype_structure",
        ),
        "causal_emulation": (
            "Emulate a target trial for treatment effect using propensity weighting.",
            "causal_protocol",
        ),
        "descriptive": (
            "Describe age, sex, LOS, organ support, and mortality distributions.",
            "distribution",
        ),
    }

    for expected_family, (question, hero_role) in cases.items():
        strategy = build_article_figure_strategy(_context(ra, question))
        prompt = render_article_figure_strategy_for_prompt(strategy)
        roles = {role.role for role in strategy.role_strategies if role.required}

        assert strategy.analysis_family == expected_family
        assert strategy.hero_role == hero_role
        assert hero_role in roles
        assert "ARTICLE FIGURE STRATEGY" in prompt

    association = build_article_figure_strategy(
        _context(ra, cases["association"][0], exposure="x")
    )
    assert any(
        "risk-difference sensitivity" in anti_pattern.lower()
        for anti_pattern in association.anti_patterns
    )


def test_cross_database_context_adds_transportability_module(ra):
    from easyicu.research_agent.planning.study_design import build_study_design_brief

    context = _context(
        ra,
        "Compare mortality distributions across six ICU databases.",
    ).model_copy(update={"cross_database_validation": ["eicu", "hirid"]})

    brief = build_study_design_brief(context)

    assert any("cross_database" in trigger for trigger in brief.adaptive_triggers)
    modules = {module.module_id: module for module in brief.display_modules}
    assert "cross_database_heterogeneity" in modules
    assert modules["cross_database_heterogeneity"].role == "transportability"


def test_analysis_blueprint_combines_prior_art_contract_and_figure_strategy(ra):
    from easyicu.research_agent.agents.core import PlannerAgent
    from easyicu.research_agent.planning.analysis_blueprint import (
        build_analysis_blueprint,
        render_analysis_blueprint_for_prompt,
    )

    context = _context(
        ra,
        "Build a mortality prediction model with external validation, AUROC, "
        "calibration, and triage threshold utility.",
    )

    blueprint = build_analysis_blueprint(context)
    prompt = render_analysis_blueprint_for_prompt(blueprint)

    assert blueprint.analysis_family == "prediction"
    assert blueprint.prior_art_design_brief.source_mode == (
        "deterministic_family_playbook"
    )
    assert "calibration" in blueprint.required_article_roles
    assert "model_performance" in blueprint.required_article_roles
    assert "validation" in blueprint.required_article_roles
    assert blueprint.figure_hero_role == "calibration"
    assert any(role.role == "calibration" for role in blueprint.visual_roles)
    assert any(
        "supplement" in item.lower()
        for item in blueprint.prior_art_design_brief.design_questions
    )
    assert "ANALYSIS BLUEPRINT" in prompt
    assert "PRIOR-ART DESIGN BRIEF" in prompt
    assert "ARTICLE ANALYSIS CONTRACT" in prompt
    assert "typed_example=table:" in prompt
    assert "Intent-only prose does not count" in prompt
    assert "ARTICLE FIGURE STRATEGY" in prompt
    outbound_prompt = PlannerAgent.request_messages(
        context,
        planning_contract_context=prompt,
    )[1].content
    outbound_metrics = PlannerAgent.request_metrics(
        context,
        planning_contract_context=prompt,
    )
    assert "HOST-DERIVED PRE-PLAN DESIGN PROFILE" in outbound_prompt
    assert prompt in outbound_prompt
    assert outbound_metrics["total_bytes"] < outbound_metrics["limit_bytes"]
    dumped = blueprint.model_dump_json()
    assert "E1" not in dumped
    assert "Sepsis" not in dumped
    assert "MIMIC" not in dumped


def test_analysis_blueprint_flags_auroc_only_prediction_plan(ra):
    from easyicu.research_agent.planning.analysis_blueprint import (
        build_analysis_blueprint,
        validate_plan_against_analysis_blueprint,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    context = _context(
        ra,
        "Build a mortality prediction model with external validation, AUROC, "
        "calibration, and triage threshold utility.",
    )
    blueprint = build_analysis_blueprint(context)
    plan = AnalysisPlan(
        research_question=context.research_question,
        steps=[
            AnalysisStep(
                step_id="01_auroc",
                intent="Fit a model and report AUROC.",
                method="logistic regression",
                expected_outputs=["figure:roc_curve", "table:auroc"],
            )
        ],
    )

    findings = validate_plan_against_analysis_blueprint(
        plan=plan,
        blueprint=blueprint,
    )

    assert findings
    details = " ".join(str(f.detail) for f in findings if f.detail)
    assert "calibration" in details
    assert "validation" in details
    assert "data_quality" in details
    assert any("too narrow" in finding.message for finding in findings)

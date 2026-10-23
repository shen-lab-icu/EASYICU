"""Signed survival and causal suites run unchanged through the host backend."""

import hashlib
import json

import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from tests.support.host_runner_parity import (
    select_for_parity as select_standard_executor,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ResearchContext,
    CohortDescriptor,
    ConceptDescriptor,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.robustness.panel import (
    RobustnessSpec,
    write_locked_robustness_specs,
)
from tests.support.continuous_survival import continuous_authority_body
from tests.support.host_runner_signed_cases import (
    binary_frame,
    binary_plan,
    categorical_projection,
    categorical_draft_plan,
    continuous_rows,
    rmst_authority,
    rmst_frame,
    rmst_step,
)
from tests.support.host_runner_parity import assert_parity, table_binding
from tests.support.target_trial import (
    target_trial_authority_body,
    synthetic_target_trial_cohort,
)
from tests.support.survival_sealed import sealed_survival, synthetic_survival_rows


def _cohort_binding(path, frame, key):
    return {
        "relative_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "declared_kind": key.split(":")[0],
        "evidence_kind": "table",
        "product": "analysis_cohort",
        "evidence_id": "synthetic_cohort",
        "produced_by_step": "source",
        "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
        "consumption_contract": {
            "input_key": key,
            "mode": "all_rows",
            "verified_row_count": len(frame),
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }


@pytest.mark.parametrize(
    "family",
    [
        "rmst",
        "continuous_survival",
        "target_trial",
        "binary_survival",
        "source_feasibility",
        "spline",
        "categorical_grid",
        "binary_sensitivity",
    ],
)
def test_signed_suite_products_match(tmp_path, family):
    if family == "rmst":
        authority, frame = rmst_authority(), rmst_frame()
        step = rmst_step()
        plan = AnalysisPlan(
            research_question="Synthetic survival contrast", steps=[step]
        )
        plan = authority.bind_plan(plan)
        steps = plan.steps
    elif family == "binary_sensitivity":
        authority, frame, plan = None, binary_frame(), binary_plan()
        steps = plan.steps[1:]
    elif family == "source_feasibility":
        authority = build_current_case_scientific_runtime_authority(
            {
                "schema_version": "easyicu.source_feasibility_runtime_authority/1",
                "authority_kind": "source_feasibility_fail_closed",
                "protocol_content_sha256": "a" * 64,
                "plan_method": "signed_source_feasibility_fail_closed",
                "plan_intent": "Record the synthetic source's unavailable comparator",
                "plan_outputs": [
                    "table:h2_source_feasibility",
                    "log:h2_scientific_runtime_receipt",
                ],
                "source": "invented_sparse_source",
                "audited_window_hours": [0, 24],
                "decision": "fail_closed",
                "reason_code": "H2_VERIFIED_NON_USE_UNAVAILABLE",
                "verified_non_use_available": False,
                "binary_control_arm_authorized": False,
                "causal_contrast_authorized": False,
                "future_design_authorized": False,
                "forbidden_plan_tokens": ["causal_effect"],
            }
        )
        step = AnalysisStep(
            step_id="feasibility",
            intent=authority.plan_intent,
            method=authority.plan_method,
            planned_analysis_role="auxiliary",
            expected_outputs=list(authority.plan_outputs),
            icu_rule_refs=[authority.plan_rule_ref],
        )
        plan = AnalysisPlan(
            research_question="Synthetic source availability", steps=[step]
        )
        steps, frame = plan.steps, pd.DataFrame({"measurement": [1.0, 2.0]})
    elif family == "spline":
        authority = build_current_case_scientific_runtime_authority(
            {
                "schema_version": "easyicu.landmark_spline_runtime_authority/1",
                "authority_kind": "landmark_spline_association",
                "protocol_content_sha256": "a" * 64,
                "plan_method": "signed_landmark_restricted_cubic_spline",
                "plan_intent": "Estimate a synthetic nonlinear association after the declared landmark",
                "plan_outputs": [
                    "table:exposure_contrasts",
                    "table:spline_curve",
                    "table:linear_sensitivity",
                    "log:spline_receipt",
                ],
                "exposure_column": "lab_max",
                "outcome_column": "mort_28d",
                "outcome_time_column": "event_hours",
                "observation_duration_column": "observation_days",
                "observation_duration_unit": "days",
                "landmark_hours": 24,
                "required_adjustment_columns": ["age", "sex"],
                "categorical_adjustment_columns": ["sex"],
                "alternative_exposure_columns": [],
                "spline_knot_quantiles": [0.1, 0.5, 0.9],
                "spline_reference": "median_in_primary_population",
                "curve_quantile_range": [0.1, 0.9],
                "curve_points": 21,
                "linear_sensitivity_per_unit": 1.0,
                "interpretation": "descriptive_prognostic_association_not_causal",
            }
        )
        frame = continuous_rows(400)
        frame["event_hours"] = frame["followup_days_28d"] * 24
        frame["observation_days"] = frame["followup_days_28d"]
        complete = RobustnessSpec(
            spec_id="complete_cases",
            axis="missing",
            description="Document the synthetic complete-case model set",
            missing_override={
                "strategy": "complete_case",
                "variables": list(authority.model_complete_case_columns),
            },
        )
        output_map = {
            "primary_or": "primary_effect",
            "complete_case_n": "complete_case_n",
            "robustness_summary": "robustness_summary",
            "missingness_strategy_notes": "missingness_strategy_notes",
            "robustness_matrix": "robustness_matrix",
        }
        plan = authority.bind_plan(
            AnalysisPlan(
                research_question="Synthetic nonlinear association",
                robustness_specs=[complete],
                steps=[
                    AnalysisStep(
                        step_id="spline",
                        intent="Estimate the synthetic association",
                        method="draft_method",
                        planned_analysis_role="primary",
                        inputs=["artifact:analysis_cohort"],
                        expected_outputs=["table:draft"],
                    ),
                    AnalysisStep(
                        step_id="shape",
                        intent="Check the synthetic exposure shape",
                        planned_analysis_role="sensitivity",
                        method="restricted_cubic_spline_sensitivity",
                        inputs=["table:draft"],
                        expected_outputs=["table:functional_form"],
                        sensitivity_spec_ids=["shape"],
                        functional_form_spec={
                            "target_column": "lab_max",
                            "knot_quantiles": [0.1, 0.5, 0.9],
                        },
                    ),
                    AnalysisStep(
                        step_id="robustness",
                        intent="Compare the registered synthetic estimates",
                        planned_analysis_role="sensitivity",
                        method="robustness_sensitivity",
                        sensitivity_spec_ids=["complete_cases"],
                        expected_outputs=[
                            "statistic:primary_or",
                            "statistic:complete_case_n",
                            "table:robustness_summary",
                            "log:missingness_strategy_notes",
                            "table:robustness_matrix",
                        ],
                        robustness_replay_spec={
                            "products": [
                                {"product_id": key, "output": value}
                                for key, value in output_map.items()
                            ]
                        },
                    ),
                ],
            )
        )
        write_locked_robustness_specs(
            run_dir=tmp_path,
            plan=plan,
            evidence=EvidenceStore(tmp_path),
            prompt_pack_version=None,
            llm_signature="synthetic-host-parity",
        )
        steps = plan.steps
    elif family == "categorical_grid":
        source, _, original = categorical_projection(tmp_path)
        grid = build_current_case_scientific_runtime_authority(
            {
                "schema_version": "easyicu.association_model_grid_runtime_authority/1",
                "authority_kind": "association_model_grid",
                "protocol_content_sha256": original.protocol_content_sha256,
                "plan_method": "verified_association_model_grid",
                "plan_intent": "Compare two synthetic category definitions",
                "cohort_product": original.cohort_product,
                "parent_product": original.primary_product,
                "output_product": "table:synthetic_definition_grid",
                "reference_variant_id": "reference",
                "metadata_columns": ["definition"],
                "output_aliases": {},
                "variants": [
                    {"analysis_id": "reference", "metadata": {"definition": "primary"}},
                    {
                        "analysis_id": "alternate",
                        "exposure_column": "alternate",
                        "metadata": {"definition": "alternate"},
                    },
                ],
            }
        )
        authority = build_current_case_scientific_runtime_authority(
            {
                **original.model_dump(
                    mode="json",
                    exclude={"execution_contract_sha256", "association_model_grid"},
                ),
                "schema_version": "easyicu.landmark_categorical_association_runtime_authority/2",
                "association_model_grid": grid.model_dump(mode="json"),
            }
        )
        frame = pd.read_parquet(source)
        frame["alternate"] = frame["aki_stage_strict"].astype(float)
        frame.loc[:20, "alternate"] = None
        plan = authority.bind_plan(categorical_draft_plan())
        risk_inputs = list(authority.absolute_risk_population_inputs())
        risk = AnalysisStep(
            step_id="risk_context",
            intent="Describe the synthetic model population",
            method="primary_population_absolute_risk_context",
            planned_analysis_role="auxiliary",
            inputs=risk_inputs,
            expected_outputs=["table:absolute_risk_context"],
            icu_rule_refs=[authority.plan_rule_ref],
            runtime_outcome_contract={
                "owner_ref": authority.plan_rule_ref,
                "outcomes": [authority.outcome_column],
            },
            input_consumption_contracts=[
                {"input_key": key, "mode": "all_rows"}
                for key in risk_inputs
                if ":" in key
            ],
        )
        plan = plan.model_copy(update={"steps": [*plan.steps, risk]})
        steps = [step for step in plan.steps if step.method != "scientific_reporting"]
    else:
        authority = (
            sealed_survival(tmp_path)[1]
            if family == "binary_survival"
            else build_current_case_scientific_runtime_authority(
                continuous_authority_body()
                if family == "continuous_survival"
                else target_trial_authority_body()
            )
        )
        frame = (
            synthetic_survival_rows(400)
            if family == "binary_survival"
            else (
                continuous_rows(400)
                if family == "continuous_survival"
                else synthetic_target_trial_cohort(n=3000)
            )
        )
        plan = authority.bind_plan(
            AnalysisPlan(
                research_question="Synthetic survival contrast",
                analysis_type="survival"
                if family != "target_trial"
                else "causal_inference",
                steps=[
                    AnalysisStep(
                        step_id="primary",
                        intent="Estimate the declared contrast",
                        method="draft_method",
                        planned_analysis_role="primary",
                        inputs=["table:analysis_cohort"],
                        expected_outputs=["table:draft"],
                    )
                ],
            )
        )
        steps = plan.steps[1:]
    cohort = tmp_path / "cohort.parquet"
    frame.to_parquet(cohort, index=False)
    products = {
        key: _cohort_binding(cohort, frame, key)
        for key in ("artifact:analysis_cohort", "table:analysis_cohort")
    }
    shared = {}
    if family == "spline":
        context = ResearchContext.model_validate(
            {
                "research_question": plan.research_question,
                "cohort": {
                    "cohort_name": "synthetic",
                    "database": "synthetic",
                    "n_stays": len(frame),
                },
                "variables": [
                    {
                        "name": "lab_max",
                        "description": "Generated laboratory value",
                        "dtype": "float64",
                        "unit": "units",
                    }
                ],
            }
        ).model_dump(mode="json")
        for name, payload in (
            ("context", context),
            ("plan", plan.model_dump(mode="json")),
        ):
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps(payload))
            shared[name] = {
                "relative_path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    if family == "categorical_grid":
        (tmp_path / "analysis_plan.json").write_text(plan.model_dump_json())
        (tmp_path / "manifest_partial.json").write_text(
            json.dumps({"plan_path": "analysis_plan.json"})
        )
    for step in steps:
        bindings = {key: products[key] for key in step.inputs if ":" in key}
        selected = select_standard_executor(
            step,
            plan=plan,
            resolved_bindings=bindings,
            current_case_scientific_runtime_authority=authority,
            scientific_runtime_projection_sha256="b" * 64,
        )
        assert selected is not None, step
        manifest = tmp_path / "resolved_inputs.json"
        manifest.write_text(
            json.dumps({**shared, "step_id": step.step_id, "inputs": bindings})
        )
        assert_parity(
            tmp_path,
            cohort,
            step,
            selected.code,
            manifest,
            {"SOURCE_DATE_EPOCH": "1700000000"},
        )
        output = tmp_path / "steps" / step.step_id / "outputs"
        summary = json.loads((output / "step_summary.json").read_text())
        for key, filename in summary.get("output_files", {}).items():
            path = output / filename
            if key.startswith("table:"):
                products[key] = table_binding(tmp_path, key, pd.read_csv(path))
            elif key.startswith("artifact:") and path.suffix == ".parquet":
                products[key] = _cohort_binding(path, pd.read_parquet(path), key)
                products[key]["relative_path"] = path.relative_to(tmp_path).as_posix()

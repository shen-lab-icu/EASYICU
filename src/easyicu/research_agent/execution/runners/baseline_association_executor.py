"""Host execution of the signed baseline model and its absolute-risk context."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

from ...authority.baseline_association_runtime import BASELINE_METHOD, BASELINE_RISK_METHOD, BaselineAssociationRuntimeAuthority
from ...authority.current_case_scientific_runtime import load_current_case_scientific_runtime_authority
from ...contracts.exposure_outcome_distribution import ExposureOutcomeDistributionSpec
from ...contracts.model_terms import serialise_model_terms
from ...schema import AnalysisPlan
from .adjusted_association_executor import _coefficient_rows
from .association_effect_modification import baseline_model_population, fit_baseline_association
from .exposure_outcome_distribution_executor import _distribution_rows, _verify_product
from .typed_input_binding import load_step_cohort_frame, load_typed_input


def baseline_executor_owns_step(step, *, plan, authority):
    if not isinstance(authority, BaselineAssociationRuntimeAuthority):
        return False
    authority.validate_plan(plan)
    if step not in plan.steps:
        return False
    if step.method in {BASELINE_METHOD, BASELINE_RISK_METHOD}:
        return True
    # This step is ordinarily adopted before execution. The standard host
    # runner can also publish its already verified materialization; it cannot
    # infer eligibility or select a new population.
    from ...authority.run_input import _declares_host_cohort_products

    return (step.step_id == "cohort_definition" and step.method == "cohort_definition_and_attrition"
            and step.planned_analysis_role == "auxiliary" and step.cohort_definition_spec is not None
            and not step.model_requirements and _declares_host_cohort_products(step))


def _publish_baseline_cohort(plan, out):
    from ...authority.run_input import _publish_immutable_host_step_output
    from ...cohort.schema import load_materialized_analysis_cohort_result

    result = load_materialized_analysis_cohort_result(
        run_dir=Path(os.environ["EASYICU_RUN_DIR"]), plan=plan,
    )
    if result is None:
        raise ValueError("baseline cohort requires a verified plan materialization")
    output_files = {}
    for product, key in (("artifact:analysis_cohort", "path"), ("table:cohort_flow", "flow_path")):
        source = Path(result[key])
        _publish_immutable_host_step_output(source=source, destination=out / source.name, label=product)
        output_files[product] = source.name
    summary = dict(status="ok", output_files=output_files, n_universe=result["n_universe"],
                   n_analysis_cohort=result["n_cohort"], cohort_definition_sha256=result["cohort_definition_sha256"])
    if result.get("authority_ref") is not None:
        summary["materialized_cohort_authority_ref"] = result["authority_ref"]
    (out / "step_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def baseline_executor_code(step, *, plan, authority, runtime_projection_sha256, plausibility_scope=None):
    authority.validate_plan(plan)
    if len(runtime_projection_sha256) != 64:
        raise ValueError("baseline execution requires a projection digest")
    body = textwrap.dedent(f"""
        import json
        from easyicu.research_agent.schema import AnalysisPlan
        from easyicu.research_agent.authority.current_case_scientific_runtime import load_current_case_scientific_runtime_authority
        from easyicu.research_agent.execution.runners.baseline_association_executor import run_baseline_step
        run_baseline_step(
            step_id={step.step_id!r},
            plan=AnalysisPlan.model_validate_json({plan.model_dump_json()!r}),
            authority=load_current_case_scientific_runtime_authority(json.loads({authority.model_dump_json()!r})),
            runtime_projection_sha256={runtime_projection_sha256!r},
        )
    """).strip()
    from .plausibility_receipt import host_plausibility_receipt_injected

    return host_plausibility_receipt_injected(body, scope=plausibility_scope, already_satisfied=False)


def baseline_risk_rows(frame, primary_table, *, requirement, authority):
    population, _ = baseline_model_population(frame, requirement, authority)
    if (primary_table.empty or not primary_table["n"].eq(len(population)).all()
        or not primary_table["n_events"].eq(int(population[authority.outcome_column].sum())).all()
        or not primary_table["requirement_id"].eq(requirement.requirement_id).all()
        or not primary_table["execution_contract_sha256"].eq(authority.execution_contract_sha256).all()
        or int(primary_table["is_primary_contrast"].sum()) != 1):
        raise ValueError("baseline absolute-risk population differs from the primary counts")
    if authority.exposure_encoding == "restricted_cubic_spline":
        rows = primary_table.loc[primary_table["subgroup"].fillna("").eq("")].copy()
        if not ((rows["absolute_risk"] - rows["reference_risk"] - rows["risk_difference"]).abs() < 1e-10).all():
            raise ValueError("baseline standardized risk-difference cross-check failed")
        return rows.to_dict("records")
    working = population.copy()
    # The model requires a 0/1 endpoint; canonicalizing bool here preserves
    # precisely that event interpretation for the count-based owner.
    working[authority.outcome_column] = working[authority.outcome_column].astype(int)
    from ...contracts.model_terms import level_spelling

    working[authority.exposure_column] = working[authority.exposure_column].map(level_spelling)
    spec = ExposureOutcomeDistributionSpec(
        exposure=authority.exposure_column, exposure_levels=list(authority.exposure_levels),
        outcome=authority.outcome_column, outcome_levels=[0, 1], outcome_positive_value=1,
        level_match_policy="exact_typed", denominator_policy="all_declared_rows",
        missing_exposure_policy="fail_closed", missing_outcome_policy="fail_closed", confidence_level=0.95,
        risk_difference_contrast={"reference_exposure_level": authority.exposure_reference_level,
                                  "comparison_exposure_level": authority.primary_contrast_level},
        dependence=authority.dependence,
    )
    rows = _distribution_rows(working, spec=spec)
    _verify_product(rows, spec=spec)
    return rows


def run_baseline_step(*, step_id, plan: AnalysisPlan, authority, runtime_projection_sha256,
                      frame=None, output_dir=None, primary_table=None):
    import pandas as pd

    authority = load_current_case_scientific_runtime_authority(authority)
    authority.validate_plan(plan)
    if len(runtime_projection_sha256) != 64:
        raise ValueError("baseline execution requires a projection digest")
    primary = authority.governed_primary_step(plan)
    requirement = primary.model_requirements[0]
    step = next(step for step in plan.steps if step.step_id == step_id)
    if not baseline_executor_owns_step(step, plan=plan, authority=authority):
        raise ValueError("baseline executor does not own this step")
    out = Path(output_dir or os.environ["STEP_OUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    if step.step_id == "cohort_definition":
        return _publish_baseline_cohort(plan, out)
    if frame is None:
        frame, _ = load_step_cohort_frame(typed_cohort_input="artifact:analysis_cohort", require_consumption_contract=True)
    if step.method == BASELINE_RISK_METHOD:
        if primary_table is None:
            loaded = load_typed_input(input_key="table:adjusted_association_estimates",
                                      run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
                                      resolved_inputs=json.loads(Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]).read_text()),
                                      step_id=step_id, expected_declared_kind="table", expected_evidence_kind="table",
                                      minimum_row_count=1, require_consumption_contract=True)
            primary_table = loaded.frame
        rows = baseline_risk_rows(frame, primary_table, requirement=requirement, authority=authority)
        table = "absolute_risk_context.csv"
        summary = {"status": "ok", "analysis_family": "absolute_risk_context", "adjusted_effect": None,
                   "outcome": authority.outcome_column, "interpretation_ceiling": "descriptive_not_causal",
                   "population_scope": "primary_model", "analysis_role": "secondary"}
        if authority.exposure_encoding == "treatment_contrast":
            summary["descriptive_contrast"] = {
                "effect_measure": "risk_difference", "estimate_pct": rows[0]["risk_difference_pct"],
                "ci_low_pct": rows[0]["risk_difference_ci_low_pct"], "ci_high_pct": rows[0]["risk_difference_ci_high_pct"],
                "interval_method": rows[0]["risk_difference_interval_method"],
                "interpretation_ceiling": "descriptive_unadjusted_not_causal",
            }
        else:
            headline = next(row for row in rows if row["is_primary_contrast"])
            summary["descriptive_contrast"] = {
                "effect_measure": "risk_difference", "estimate": headline["risk_difference"],
                "ci_low": headline["risk_difference_ci_low"], "ci_high": headline["risk_difference_ci_high"],
                "estimand": "model_standardized_difference", "interpretation_ceiling": "descriptive_not_causal",
            }
    else:
        population, design, result, rows, interaction = fit_baseline_association(frame, requirement=requirement, authority=authority)
        table = "adjusted_association_estimates.csv"
        coefficient_rows = _coefficient_rows(result.terms, model_id=requirement.requirement_id,
                                             exposure=authority.exposure_column, adjustment=list(requirement.covariates),
                                             effect_modifiers=tuple(term.name for term in requirement.effect_modifiers),
                                             effect_scale="odds_ratio", exposure_contrast_columns=tuple(
                                                 t.term for t in result.terms if t.source_variable == authority.exposure_column))
        pd.DataFrame(coefficient_rows).to_csv(out / "adjusted_association_coefficients.csv", index=False)
        headline = next(row for row in rows if row["is_primary_contrast"])
        contract = dict(
            model_id=requirement.requirement_id, requirement_id=requirement.requirement_id,
            outcome=requirement.outcome, outcome_type="binary", method_family=requirement.method_family,
            exposure_source=requirement.exposure_source, exposure_expression=headline["contrast"],
            exposure_role="primary", analysis_role="primary", analysis_set=requirement.analysis_set,
            baseline_missing_policy="drop_missing_baseline", n=result.n, event_n=result.n_events,
            fit_status="fitted", converged=result.converged, separation_detected=result.separation_detected,
            penalized=False, fit_method=requirement.method_family, model_terms=serialise_model_terms(requirement.model_terms),
            design_columns=list(design.columns), dependence=authority.dependence.model_dump(mode="json") if authority.dependence else None,
            effect_modifiers=serialise_model_terms(requirement.effect_modifiers),
            cluster_count=result.cluster_count,
        )
        summary = dict(status="ok", analysis_family="association", interpretation_class="adjusted_association",
                       exposure=authority.exposure_column, outcome=authority.outcome_column, covariates=list(requirement.covariates),
                       adjustment_covariates=list(requirement.covariates), model_contracts=[contract],
                       coefficient_table="adjusted_association_coefficients.csv", n_total=len(population), n_events=result.n_events,
                       adjusted_effect=headline["estimate"], effect_scale="odds_ratio", primary_estimate=headline["estimate"],
                       primary_ci_low=headline["ci_low"], primary_ci_high=headline["ci_high"],
                       primary_or=headline["estimate"], primary_estimate_interval=[headline["ci_low"], headline["ci_high"]],
                       variance_estimator=result.variance_estimator, cluster_count=result.cluster_count,
                       model_terms=serialise_model_terms(requirement.model_terms), design_columns=list(design.columns),
                       estimator_kind="logistic", analysis_role="primary", analysis_set=requirement.analysis_set,
                       interaction_tests=interaction, interpretation_ceiling="descriptive_not_causal")
    for row in rows:
        row["execution_contract_sha256"] = authority.execution_contract_sha256
    pd.DataFrame(rows).to_csv(out / table, index=False)
    summary["output_files"] = {step.expected_outputs[0]: table}
    summary["execution_contract_sha256"] = authority.execution_contract_sha256
    population, _ = baseline_model_population(frame, requirement, authority)
    summary["scientific_runtime_receipt"] = {
        "schema_version": "easyicu.baseline_association_runtime_receipt/1",
        "protocol_content_sha256": authority.protocol_content_sha256,
        "execution_contract_sha256": authority.execution_contract_sha256,
        "runtime_projection_sha256": runtime_projection_sha256,
        "primary_population_n": len(population), "events": int(population[authority.outcome_column].sum()),
        "adjustment_columns": list(authority.required_adjustment_columns),
        "effect_modifiers": serialise_model_terms(requirement.effect_modifiers),
        "exposure_encoding": authority.exposure_encoding,
        "observed_knots": list(population[authority.exposure_column].quantile(authority.spline_knot_quantiles))
        if authority.exposure_encoding == "restricted_cubic_spline" else [],
        "interpretation": authority.interpretation,
    }
    (out / "step_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary

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
from ...authority.evidence_store import sha256_of_file
from ...contracts.typed_schema import merge_host_table_contract, typed_product_schema_receipt
from ...script_runtime import _sanitized, to_jsonable
from ...schema import AnalysisPlan
from .association_binary_sensitivity_executor import run_association_binary_sensitivity
from .adjusted_association_figure_executor import run_adjusted_association_figure
from .scientific_reporting_executor import run_scientific_reporting
from .adjusted_association_executor import _coefficient_rows
from .association_effect_modification import (
    BINARY_EFFECT_SCALES, baseline_estimator_kind, baseline_model_population, fit_baseline_association,
)
from .exposure_outcome_distribution_executor import _distribution_rows, _verify_product
from .typed_input_binding import contained_regular_file, load_step_cohort_frame, load_typed_input


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
        kind = baseline_estimator_kind(requirement)
        effect_scale = BINARY_EFFECT_SCALES[kind]
        table = "adjusted_association_estimates.csv"
        coefficient_rows = _coefficient_rows(result.terms, model_id=requirement.requirement_id,
                                             exposure=authority.exposure_column, adjustment=list(requirement.covariates),
                                             effect_modifiers=tuple(term.name for term in requirement.effect_modifiers),
                                             effect_scale=effect_scale, exposure_contrast_columns=tuple(
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
                       adjusted_effect=headline["estimate"], effect_scale=effect_scale, primary_estimate=headline["estimate"],
                       primary_ci_low=headline["ci_low"], primary_ci_high=headline["ci_high"],
                       # One headline key per scale, as the generic association
                       # owner writes: a risk ratio never carries the OR name.
                       **({"primary_rr": headline["estimate"], "primary_rr_ci": [headline["ci_low"], headline["ci_high"]]}
                          if kind == "glm_poisson" else {"primary_or": headline["estimate"]}),
                       primary_estimate_interval=[headline["ci_low"], headline["ci_high"]],
                       variance_estimator=result.variance_estimator, cluster_count=result.cluster_count,
                       model_terms=serialise_model_terms(requirement.model_terms), design_columns=list(design.columns),
                       estimator_kind=kind, analysis_role="primary", analysis_set=requirement.analysis_set,
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


def append_input_binding(key, record, path):
    binding = dict(relative_path=record.relative_path, sha256=record.sha256, evidence_id=record.evidence_id,
                   declared_kind=key.split(":", 1)[0], product=key.split(":", 1)[1], evidence_kind=record.kind,
                   produced_by_step=record.produced_by_step,
                   consumption_contract={"input_key": key, "mode": "all_rows", "artifact_sha256": record.sha256})
    binding["identity_row"] = {"input_key": key, **{k: binding[k] for k in ("declared_kind", "product", "evidence_id", "sha256")}}
    if record.kind == "table":
        receipt = typed_product_schema_receipt(artifact_path=path, expected_sha256=record.sha256)
        if receipt is None:
            raise ValueError("The sealed input has no readable table schema.")
        binding["product_contract"] = merge_host_table_contract(None, receipt)
        binding["consumption_contract"]["verified_row_count"] = binding["product_contract"]["row_count"]
    return binding


def _path(run_dir, binding):
    path = contained_regular_file(run_dir / binding["relative_path"], run_dir)
    if path is None or sha256_of_file(path) != binding["sha256"]:
        raise ValueError("Adopted append input changed.")
    return path


def execute_result_append(*, run_dir: Path, plan: AnalysisPlan, authority, inputs, evidence, progress=None):
    import pandas as pd

    authority.validate_plan(plan)
    compiled = authority.compilation()
    parent = authority.baseline.governed_primary_step(AnalysisPlan.model_validate_json(authority.source_plan_json))
    if compiled.kind == "sensitivity":
        from ...planning.family_spec.baseline_association_template import SensitivityAppend

        parent = SensitivityAppend.parent_step(AnalysisPlan.model_validate_json(authority.source_plan_json), authority.baseline)
    from ...authority.runtime_artifacts import AuditLogger

    audit = AuditLogger(run_dir / "audit_log.jsonl")
    records = []

    def event(step, index, verb):
        row = audit.emit(phase="execute", event=f"Step {index}/{len(plan.steps)} {verb}: {step.step_id}.",
                         status="completed" if verb == "complete" else "running", step_id=step.step_id)
        if progress:
            progress(row.model_dump(mode="json"))

    bindings = dict(inputs)
    results = []
    for index, step in enumerate(plan.steps, 1):
        event(step, index, "started")
        selected = {key: bindings[key] for key in step.inputs if ":" in key}
        for binding in selected.values():
            _path(run_dir, binding)
        out = run_dir / "steps" / step.step_id / "outputs"
        out.mkdir(parents=True, exist_ok=False)
        manifest = {"step_id": step.step_id, "inputs": selected}
        resolved = out / "resolved_inputs.json"
        resolved.write_text(json.dumps(manifest))
        if step.method == "adjusted_association_figure":
            summary = run_adjusted_association_figure(out_dir=out, run_dir=run_dir, resolved_inputs=manifest,
                                                       step_id=step.step_id, figure_product="append_article_figure")
        elif compiled.kind == "article":
            summary = run_scientific_reporting(
                out_dir=out, run_dir=run_dir, resolved_inputs=manifest, step_id=step.step_id,
                planned_analysis_role=step.planned_analysis_role, intent=step.intent,
                report_product=step.expected_outputs[0].split(":", 1)[1], declared_inputs=step.inputs,
            )
        else:
            cohort_key = next(key for key in selected if key.startswith("artifact:"))
            cohort_path = _path(run_dir, selected[cohort_key])
            frame = pd.read_parquet(cohort_path)
            if compiled.kind == "sensitivity":
                summary = run_association_binary_sensitivity(
                    frame=frame, cohort_path=cohort_path,
                    step=step.model_copy(update={"scientific_capability": "association_freeform_v1"}), parent_step=parent,
                    run_dir=run_dir, resolved_inputs=resolved, out_dir=out,
                )
            elif step.method == "append_exposure_definition":
                frame["append_exposure_value"] = pd.to_numeric(frame[compiled.exposure_source], errors="raise")
                path = out / "append_exposure_cohort.parquet"
                frame.to_parquet(path, index=False)
                summary = {"status": "ok", "output_files": {step.expected_outputs[0]: path.name},
                           "source_column": compiled.exposure_source, "derived_column": "append_exposure_value"}
            elif step.method == "append_exposure_absolute_risk":
                requirement = next(item.model_requirements[0] for item in plan.steps if item.model_requirements)
                primary_table = pd.read_csv(_path(run_dir, selected["table:adjusted_association_estimates"]))
                rows = baseline_risk_rows(frame, primary_table, requirement=requirement, authority=compiled.model_authority)
                path = out / "append_exposure_absolute_risk.csv"
                pd.DataFrame(rows).to_csv(path, index=False)
                summary = dict(status="ok", interpretation="different_exposure_definition",
                               analysis_rows=rows, output_files={step.expected_outputs[0]: path.name})
            else:
                runtime = compiled.model_authority
                # The model owner checks the same typed model on the child
                # cohort. The append runtime has already sealed its distinct
                # product name and the derivation that produced it.
                model_step = step.model_copy(update={"inputs": ["artifact:analysis_cohort", *runtime.required_columns]})
                requirement = runtime._requirement(model_step)
                _, _, fit, rows, interaction = fit_baseline_association(frame, requirement=requirement, authority=runtime)
                if compiled.kind == "subgroup":
                    rows = [row for row in rows if row["subgroup"]]
                for row in rows:
                    row["execution_contract_sha256"] = runtime.execution_contract_sha256
                    row["append_contract_sha256"] = authority.execution_contract_sha256
                path = out / (step.expected_outputs[0].split(":", 1)[1] + ".csv")
                pd.DataFrame(rows).to_csv(path, index=False)
                summary = dict(status="ok", output_files={step.expected_outputs[0]: path.name},
                               interpretation="different_exposure_definition" if compiled.kind == "exposure" else "effect_modification",
                               source_run_id=authority.source_run_id, source_run_sha256=authority.source_run_sha256,
                               n=fit.n, interaction_tests=interaction,
                               stratum_estimates=[row for row in rows if row["subgroup"]],
                               analysis_rows=rows, scientific_action_id=step.scientific_action_id)
        summary.update(step_id=step.step_id, execution_contract_sha256=authority.execution_contract_sha256,
                       deterministic_standard_analysis="result_append_" + compiled.kind)
        summary = json.loads(json.dumps(_sanitized(summary), default=to_jsonable, allow_nan=False))
        for binding in selected.values():
            _path(run_dir, binding)
        output_ids = []
        for key, name in summary["output_files"].items():
            record = evidence.register_file(
                kind="table" if key.startswith(("table:", "artifact:")) else "figure" if key.startswith("figure:") else "log",
                description=step.intent, source_path=out / name, produced_by_step=step.step_id,
                inputs=[value["evidence_id"] for value in selected.values()],
                metadata={"source_run_id": authority.source_run_id, "source_run_sha256": authority.source_run_sha256},
            )
            output_ids.append(record.evidence_id)
            bindings[key] = append_input_binding(key, record, run_dir / record.relative_path)
            if key.startswith("figure:"):
                gallery = {"figures": [{"figure_id": key.split(":", 1)[1], "label": step.intent,
                                        "relative_path": record.relative_path, "status": "available",
                                        "caption": "Existing association estimate and interval; no model refit."}]}
                (run_dir / "figure_gallery.json").write_text(json.dumps(gallery))
        for name in summary.get("contract_files", ()):
            evidence.register_file(kind="log", description="Append output contract", source_path=out / name,
                                   produced_by_step=step.step_id)
        (out / "step_summary.json").write_text(json.dumps(summary, allow_nan=False))
        evidence.register_file(kind="statistic", description="Append execution summary", source_path=out / "step_summary.json",
                               produced_by_step=step.step_id)
        records.append(dict(step_id=step.step_id, status="ok", evidence_ids=output_ids, step_summary=summary,
                            analysis_request={"step": step.model_dump(mode="json")}))
        (run_dir / "manifest.json").write_text(json.dumps(dict(run_id=run_dir.name, per_step_records=records)))
        event(step, index, "complete")
        results.append(summary)
    return results

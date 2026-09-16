"""Descriptive risk on the exact population of a bound primary model.

The first supported population adapter reuses the landmark model's own row
selection. It never infers a population from the question or copies the model's
adjusted risks into an unadjusted descriptive table.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from ...authority.current_case_scientific_runtime import (
    LandmarkCategoricalAssociationRuntimeAuthority,
    LandmarkSplineRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ...authority.declared_levels import execution_model_requirement
from ...contracts.association_execution import sole_primary_model_requirement
from ...contracts.cohort_product_keys import (
    is_closed_cohort_product_key,
    sole_typed_cohort_input,
)
from ...schema import AnalysisPlan, AnalysisStep
from ...contracts.runtime_outcomes import RuntimeOutcomeContract
from ...authority.plausibility import FlagOnlyPlausibilityScope
from .deterministic_descriptive import run_absolute_risk_context
from .landmark_spline_fit import prepare_landmark_model_population
from ..model_matrix import compile_model_terms
from .typed_input_binding import contained_regular_file, load_typed_input
from .plausibility_receipt import render_standard_plausibility_receipt_code

PRIMARY_POPULATION_RISK = "primary_population_absolute_risk_context"


def _bound_categorical_plan(run_dir: Path) -> AnalysisPlan:
    root = run_dir.resolve()
    manifest_path = contained_regular_file(root / "manifest_partial.json", root)
    if manifest_path is None:
        raise ValueError("Primary population requires the current run manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan_name = manifest.get("plan_path") or "analysis_plan.json"
    if not isinstance(plan_name, str) or not plan_name.endswith(".json"):
        raise ValueError("Primary population plan path is invalid")
    plan_path = contained_regular_file(root / plan_name, root)
    if plan_path is None:
        raise ValueError("Primary population plan is not inside the run")
    return AnalysisPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))


def _categorical_model_population(
    cohort: Any,
    *,
    plan: AnalysisPlan,
    sealed: LandmarkCategoricalAssociationRuntimeAuthority,
    primary_table: Any,
) -> Any:
    primary = sealed.governed_primary_step(plan)
    requirement = sole_primary_model_requirement(primary)
    if requirement is None:
        raise ValueError("Categorical primary model requirement is missing")
    requirement = execution_model_requirement(primary, requirement)
    needed = [
        requirement.outcome,
        *(term.name for term in requirement.model_terms or ()),
        *((requirement.dependence.group_source,) if requirement.dependence else ()),
    ]
    if any(column not in cohort.columns for column in needed):
        raise ValueError("Categorical primary model columns are missing")
    model = cohort.loc[:, list(dict.fromkeys(needed))]
    if requirement.dependence and model[requirement.dependence.group_source].isna().any():
        raise ValueError("Categorical primary grouping contains missing values")
    design = compile_model_terms(
        model, terms=requirement.model_terms or (), exposure=requirement.exposure_source
    ).design
    complete = design.notna().all(axis=1) & model[requirement.outcome].notna()
    selected = cohort.loc[complete]
    events = int(selected[sealed.outcome_column].sum())
    required = {"n", "n_events", "requirement_id", "is_primary_contrast", "analysis_role"}
    if primary_table.empty or not required.issubset(primary_table.columns):
        raise ValueError("Categorical primary result lacks its population contract")
    if (
        not primary_table["requirement_id"].eq(requirement.requirement_id).all()
        or not primary_table["analysis_role"].eq("primary").all()
        or int(primary_table["is_primary_contrast"].sum()) != 1
        or not primary_table["n"].eq(len(selected)).all()
        or not primary_table["n_events"].eq(events).all()
    ):
        raise ValueError("Descriptive population does not match the bound primary model")
    return selected


def primary_population_risk_owns_step(
    step: AnalysisStep, *, plan: AnalysisPlan, authority: Any
) -> bool:
    if step.method != PRIMARY_POPULATION_RISK:
        return False
    if isinstance(authority, LandmarkSplineRuntimeAuthority):
        primary = authority.governed_step(plan)
        expected = authority.absolute_risk_population_inputs(
            sole_typed_cohort_input(primary)
        )
    elif isinstance(authority, LandmarkCategoricalAssociationRuntimeAuthority):
        authority.governed_primary_step(plan)
        expected = authority.absolute_risk_population_inputs()
    else:
        return False
    return (
        tuple(step.inputs) == expected
        and step.expected_outputs == ["table:absolute_risk_context"]
        and authority.plan_rule_ref in step.icu_rule_refs
        and step.runtime_outcome_contract == RuntimeOutcomeContract(
            owner_ref=authority.plan_rule_ref, outcomes=(authority.outcome_column,)
        )
        and all(
            any(
                c.input_key == key and c.mode == "all_rows"
                for c in step.input_consumption_contracts
            )
            for key in expected
            if ":" in key
        )
    )


def primary_population_risk_code(
    step: AnalysisStep,
    *,
    authority: Any,
    runtime_projection_sha256: str,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    payload = json.dumps(authority.model_dump(mode="json"), sort_keys=True)
    step_payload = json.dumps(step.model_dump(mode="json"), sort_keys=True)
    body = textwrap.dedent(f"""
        import json
        import os
        from pathlib import Path
        from easyicu.research_agent.schema import AnalysisStep
        from easyicu.research_agent.execution.runners.primary_population_descriptive import run_primary_population_risk
        run_primary_population_risk(
            step=AnalysisStep.model_validate(json.loads({step_payload!r})),
            authority=json.loads({payload!r}),
            runtime_projection_sha256={runtime_projection_sha256!r},
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
        )
    """).strip()
    if plausibility_scope is None or not plausibility_scope.expected_columns:
        return body
    plausibility_scope.require_step(step.step_id)
    receipt = render_standard_plausibility_receipt_code(
        plausibility_scope, frame_name="plausibility_frame"
    )
    prefix = 'import os\nimport pandas as pd\nplausibility_frame = pd.read_parquet(os.environ["COHORT_PARQUET"])'
    suffix = textwrap.dedent("""
        summary_path = Path(os.environ["STEP_OUT_DIR"]) / "step_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["plausibility_audit"] = plausibility_audit
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    """).strip()
    return "\n\n".join([prefix, receipt, body, suffix])


def run_primary_population_risk(
    *,
    step: AnalysisStep,
    authority: Any,
    runtime_projection_sha256: str,
    run_dir: Path,
    resolved_inputs: Path,
) -> dict[str, Any]:
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, (LandmarkSplineRuntimeAuthority, LandmarkCategoricalAssociationRuntimeAuthority)):
        raise ValueError("No primary population adapter exists for this runtime")
    cohort_keys = [key for key in step.inputs if is_closed_cohort_product_key(key)]
    if len(cohort_keys) != 1:
        raise ValueError("Primary population requires exactly one bound cohort")
    cohort_key = cohort_keys[0]
    categorical = isinstance(sealed, LandmarkCategoricalAssociationRuntimeAuthority)
    expected = (
        sealed.absolute_risk_population_inputs()
        if categorical else sealed.absolute_risk_population_inputs(cohort_key)
    )
    if (
        step.method != PRIMARY_POPULATION_RISK
        or tuple(step.inputs) != expected
        or sealed.plan_rule_ref not in step.icu_rule_refs
        or step.expected_outputs != ["table:absolute_risk_context"]
        or step.runtime_outcome_contract
        != RuntimeOutcomeContract(
            owner_ref=sealed.plan_rule_ref, outcomes=(sealed.outcome_column,)
        )
        or any(
            not any(
                c.input_key == key and c.mode == "all_rows"
                for c in step.input_consumption_contracts
            )
            for key in expected
            if ":" in key
        )
    ):
        raise ValueError("Primary population input contract drifted")
    if len(runtime_projection_sha256) != 64:
        raise ValueError("Primary population requires a runtime projection digest")
    manifest = json.loads(resolved_inputs.read_text())
    loaded = {}
    parent_key = sealed.primary_product if categorical else sealed.linear_sensitivity_product
    for key in (cohort_key, parent_key):
        loaded[key] = load_typed_input(
            input_key=key,
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=step.step_id,
            expected_declared_kind=key.partition(":")[0],
            expected_evidence_kind=None if key == cohort_key else "table",
            minimum_row_count=30 if key == cohort_key else 1,
            require_consumption_contract=True,
        )
    cohort = loaded[cohort_key].frame
    parent = loaded[parent_key].frame
    if categorical:
        plan = _bound_categorical_plan(run_dir)
        sealed.validate_plan(plan)
        if not any(planned == step for planned in plan.steps):
            raise ValueError("Primary population step differs from the bound plan")
        selected = _categorical_model_population(
            cohort, plan=plan, sealed=sealed, primary_table=parent
        )
    else:
        if len(parent) != 1 or not {"n", "events"}.issubset(parent.columns):
            raise ValueError(
                "Primary population diagnostic must contain exactly one n/events record"
            )
        population = prepare_landmark_model_population(cohort, sealed)
        selected = cohort.loc[population.model_frame.index]
        if (
            float(parent.iloc[0]["n"]) != len(selected)
            or float(parent.iloc[0]["events"])
            != int(selected[sealed.outcome_column].sum())
        ):
            raise ValueError("Descriptive population does not match the bound primary model")
    events = int(selected[sealed.outcome_column].sum())
    receipt = {
        "schema_version": "easyicu.primary_population_descriptive/1",
        "scope": "primary_model_complete_cases",
        "owner_ref": sealed.plan_rule_ref,
        "runtime_projection_sha256": runtime_projection_sha256,
        "source_cohort_n": len(cohort),
        "population_n": len(selected),
        "event_n": events,
        "input_bindings": [
            {
                "input_key": key,
                "evidence_id": value.evidence_id,
                "sha256": value.sha256,
                "row_count": value.row_count,
                "loaded": True,
            }
            for key, value in loaded.items()
        ],
    }
    return run_absolute_risk_context(
        population_frame=selected,
        population_receipt=receipt,
        exposure_columns=[sealed.exposure_column],
        outcome_column=sealed.outcome_column,
    )

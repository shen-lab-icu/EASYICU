"""Describe the approved clinical/outcome roster without learning new clusters."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...contracts.phenotype_comparison import (
    ASSIGNMENTS_PRODUCT,
    COMPARISON_ACTION,
    COMPARISON_KIND,
    COMPARISON_PRODUCT,
    comparison_cohort_input,
    comparison_spec_sha256,
    comparison_table_spec,
    validate_comparison_step,
)
from ...methods.table_one import build_grouped_table_one
from ...research_context.typed import parse_research_context_json
from ...schema import AnalysisStep, PhenotypeComparisonSpec
from .typed_input_binding import load_typed_input, sha256_file


def phenotype_comparison_executor_owns_step(step: AnalysisStep) -> bool:
    if step.scientific_action_id != COMPARISON_ACTION:
        return False
    try:
        validate_comparison_step(step)
    except ValueError:
        return False
    return True


def phenotype_comparison_executor_code(step: AnalysisStep) -> str:
    validate_comparison_step(step)
    return textwrap.dedent(f"""
        import json
        import os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.phenotype_comparison_executor import run_phenotype_comparison

        summary = run_phenotype_comparison(
            spec=json.loads({step.phenotype_comparison_spec.model_dump_json()!r}),
            typed_cohort_input={comparison_cohort_input(step)!r}, step_id={step.step_id!r},
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]), out_dir=Path(os.environ["STEP_OUT_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    """).strip()


def _identities(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("phenotype_comparison_identity_invalid")
    values = series.astype(str)
    if values.str.strip().eq("").any() or values.duplicated().any():
        raise ValueError("phenotype_comparison_identity_invalid")
    return values


def run_phenotype_comparison(
    *,
    spec: PhenotypeComparisonSpec | dict,
    typed_cohort_input: str,
    step_id: str,
    run_dir: Path,
    out_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
) -> dict[str, Any]:
    spec = (
        spec
        if isinstance(spec, PhenotypeComparisonSpec)
        else PhenotypeComparisonSpec.model_validate(spec)
    )
    context = parse_research_context_json(
        (Path(run_dir) / "research_context.json").read_text("utf-8")
    )
    step = AnalysisStep(
        step_id=step_id,
        planned_analysis_role="secondary",
        intent="Describe the frozen cluster memberships.",
        method="descriptive_profile_by_frozen_cluster",
        scientific_action_id=COMPARISON_ACTION,
        inputs=[
            spec.identity_column,
            *(v.name for v in spec.variables),
            typed_cohort_input,
            ASSIGNMENTS_PRODUCT,
        ],
        expected_outputs=[COMPARISON_PRODUCT],
        phenotype_comparison_spec=spec,
    )
    validate_comparison_step(step, context)
    cohort = load_typed_input(
        input_key=typed_cohort_input,
        run_dir=run_dir,
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        expected_declared_kind=typed_cohort_input.partition(":")[0],
        expected_evidence_kind="table",
        require_consumption_contract=True,
        minimum_row_count=1,
        text_columns=(spec.identity_column,),
    )
    assignments = load_typed_input(
        input_key=ASSIGNMENTS_PRODUCT,
        run_dir=run_dir,
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        require_consumption_contract=True,
        minimum_row_count=1,
        text_columns=("unit_id",),
    )
    labels = assignments.frame
    if not {
        "source_cohort_sha256",
        "source_identity_column",
        "unit_id",
        "cluster",
    }.issubset(labels.columns):
        raise ValueError(
            "phenotype_comparison_source_mismatch: assignments must seal their source"
        )
    if (
        labels.source_cohort_sha256.isna().any()
        or labels.source_identity_column.isna().any()
        or set(labels.source_cohort_sha256) != {cohort.sha256}
        or set(labels.source_identity_column) != {spec.identity_column}
    ):
        raise ValueError("phenotype_comparison_source_mismatch")
    required = [spec.identity_column, *(v.name for v in spec.variables)]
    if not set(required).issubset(cohort.frame.columns):
        raise ValueError("phenotype_comparison_input_mismatch")
    cohort_ids = _identities(cohort.frame[spec.identity_column])
    assignment_ids = _identities(labels.unit_id)
    if len(cohort_ids) != len(assignment_ids) or set(cohort_ids) != set(assignment_ids):
        raise ValueError("phenotype_comparison_membership_mismatch")
    numeric = pd.to_numeric(labels.cluster, errors="coerce")
    if (
        not np.isfinite(numeric).all()
        or not numeric.eq(np.floor(numeric)).all()
        or numeric.lt(0).any()
        or numeric.nunique() < 2
    ):
        raise ValueError("phenotype_comparison_cluster_invalid")
    groups = sorted(int(value) for value in numeric.unique())
    execution_spec = comparison_table_spec(spec, context, groups)
    frame = cohort.frame.loc[:, required].copy()
    # Map by the exact unique key, never by row order or a partial inner join.
    frozen = pd.Series(numeric.astype(int).to_numpy(), index=assignment_ids)
    frame[execution_spec.group_by] = cohort_ids.map(frozen).to_numpy()
    result = build_grouped_table_one(frame, execution_spec)
    coordinates = {
        "comparison_spec_sha256": comparison_spec_sha256(spec),
        "source_cohort_sha256": cohort.sha256,
        "source_assignments_sha256": assignments.sha256,
    }
    for key, value in coordinates.items():
        result[key] = value
    result["variable_role"] = result.variable.map(
        {
            v.name: "outcome" if v.name in spec.outcome_columns else "clinical_profile"
            for v in spec.variables
        }
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "outcome_by_cluster.csv"
    result.to_csv(result_path, index=False)
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "descriptive_profile_by_frozen_cluster",
        "analysis_family": "phenotyping",
        "deterministic_standard_analysis": COMPARISON_KIND,
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "refit_performed": False,
        "n_rows": len(frame),
        "cluster_counts": {
            str(group): int(frame[execution_spec.group_by].eq(group).sum())
            for group in groups
        },
        "execution_table_spec": execution_spec.model_dump(mode="json"),
        **coordinates,
        "source_inputs": [typed_cohort_input, ASSIGNMENTS_PRODUCT],
        "input_bindings": [
            {"input_key": key, "loaded": True}
            for key in (typed_cohort_input, ASSIGNMENTS_PRODUCT)
        ],
        "output_files": {COMPARISON_PRODUCT: result_path.name},
        "output_sha256": sha256_file(result_path),
        "limitations": [
            "Within-cohort descriptive comparison; not a causal effect or external phenotype validation.",
            "No inference after data-derived clustering; missing outcomes are not coded as non-events.",
        ],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary

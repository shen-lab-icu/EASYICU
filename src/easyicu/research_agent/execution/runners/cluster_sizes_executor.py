"""Host-owned sizes of verified existing clusters, without refitting them."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Mapping

import pandas as pd

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...canonical_json import canonical_sha256
from ...contracts.diagnostic_specs import (
    CLUSTER_SIZE_ACTION,
    CLUSTER_SIZE_KIND,
    CLUSTER_SIZE_OUTPUTS,
    ClusterSizeSpec,
    cluster_sizes_cohort_input,
)
from ...contracts.host_scaffold import HostScaffoldedScript
from ...methods.cluster_sizes import (
    _load_sources as _load_sources,
    cluster_sizes_consumed_input_keys as cluster_sizes_consumed_input_keys,
    cluster_sizes_executor_owns_step as cluster_sizes_executor_owns_step,
    cluster_sizes_executor_verdict as cluster_sizes_executor_verdict,
    cluster_sizes_output_findings as cluster_sizes_output_findings,
    compute_cluster_sizes,
)
from ...schema import AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sha256_file

CLUSTER_SIZES_ANALYSIS_KIND = CLUSTER_SIZE_KIND








def cluster_sizes_executor_scaffold(
    step: AnalysisStep,
    *,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> HostScaffoldedScript:
    if not cluster_sizes_executor_owns_step(step):
        raise ValueError("cluster_sizes_step_contract_invalid")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    code = textwrap.dedent(f"""
        import json, os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.typed_input_binding import load_step_cohort_frame
        from easyicu.research_agent.execution.runners.cluster_sizes_executor import run_cluster_sizes_from_bound

        frame, cohort_path = load_step_cohort_frame(
            typed_cohort_input={cluster_sizes_cohort_input(step)!r}, require_consumption_contract=True,
        )
    """).strip()
    if plausibility_scope is not None and plausibility_scope.expected_columns:
        code += (
            "\n\n"
            + render_standard_plausibility_receipt_code(
                plausibility_scope, frame_name="frame"
            ).strip()
        )
    epilogue = textwrap.dedent(f"""
        summary = run_cluster_sizes_from_bound(
            frame=frame, source_cohort=cohort_path,
            spec=json.loads({step.cluster_size_spec.model_dump_json()!r}),
            step_id={step.step_id!r}, typed_cohort_input={cluster_sizes_cohort_input(step)!r},
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    """).strip()
    return HostScaffoldedScript(prologue=code, epilogue=epilogue)


def cluster_sizes_executor_code(step: AnalysisStep, **kwargs: Any) -> str:
    return cluster_sizes_executor_scaffold(step, **kwargs).assembled()




def run_cluster_sizes(
    *,
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    spec: ClusterSizeSpec | Mapping[str, Any],
    out_dir: Path,
    source_cohort: Path | None = None,
    source_assignments: Path | None = None,
    source_receipt: Mapping[str, Any] | None = None,
    step_id: str = "cluster_sizes",
) -> dict[str, Any]:
    spec = ClusterSizeSpec.model_validate(spec)
    if set(source_receipt or {}) - {
        "source_assignments_sha256",
        "assignment_source",
        "source_freeze_sha256",
        "freeze_status",
    }:
        raise ValueError("cluster_sizes_source_receipt_fields_invalid")
    if source_receipt is not None and (
        source_receipt.get("assignment_source") != spec.assignment_source
        or source_assignments is None
        or source_receipt.get("source_assignments_sha256")
        != sha256_file(source_assignments)
    ):
        raise ValueError("cluster_sizes_source_receipt_mismatch")
    result = compute_cluster_sizes(frame, assignments, spec)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "cluster_sizes.csv"
    pd.DataFrame([row.model_dump(mode="json") for row in result.rows]).to_csv(
        table_path, index=False
    )
    receipt = {
        "schema_version": "easyicu.cluster_sizes_receipt/1",
        "specification": spec.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
        "result_sha256": canonical_sha256(result.model_dump(mode="json")),
        "source_cohort_sha256": sha256_file(source_cohort) if source_cohort else None,
        "source_assignments_sha256": sha256_file(source_assignments)
        if source_assignments
        else None,
        "output_sha256": sha256_file(table_path),
        "assignment_source": spec.assignment_source,
        **dict(source_receipt or {}),
        "authority_scope": "analysis_only",
        "refit_performed": False,
        "near_empty_rule": "strictly_below_any_declared_threshold"
        if spec.near_empty_n is not None or spec.near_empty_fraction is not None
        else "not_requested",
    }
    receipt_path = out_dir / "cluster_sizes_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "analysis_family": "phenotyping",
        "deterministic_standard_analysis": CLUSTER_SIZE_KIND,
        "scientific_action_id": CLUSTER_SIZE_ACTION,
        "authority_scope": "analysis_only",
        "claim_ceiling": "analysis_only",
        "paper_authorization_allowed": False,
        "refit_performed": False,
        "analysis_unit": spec.analysis_unit,
        "n_total": result.input_n,
        "assigned_n": result.assigned_n,
        "missing_label_n": result.missing_label_n,
        "degenerate_single_cluster": result.degenerate_single_cluster,
        "specification_sha256": result.specification_sha256,
        "result_sha256": receipt["result_sha256"],
        "source_cohort_sha256": receipt["source_cohort_sha256"],
        "source_assignments_sha256": receipt["source_assignments_sha256"],
        "assignment_source": spec.assignment_source,
        **dict(source_receipt or {}),
        "output_files": dict(
            zip(CLUSTER_SIZE_OUTPUTS, (table_path.name, receipt_path.name))
        ),
        "interpretation": "descriptive_existing_frozen_cluster_memberships",
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def run_cluster_sizes_from_bound(
    *,
    frame: pd.DataFrame,
    source_cohort: Path,
    spec: ClusterSizeSpec | Mapping[str, Any],
    step_id: str,
    typed_cohort_input: str,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    spec = ClusterSizeSpec.model_validate(spec)
    cohort, assignments, source = _load_sources(
        spec=spec,
        typed_cohort_input=typed_cohort_input,
        step_id=step_id,
        run_dir=run_dir,
        resolved_inputs=resolved_inputs,
    )
    if Path(source_cohort).resolve() != cohort.path.resolve() or not frame.equals(
        cohort.frame
    ):
        raise ValueError("cluster_sizes_loaded_cohort_mismatch")
    return run_cluster_sizes(
        frame=cohort.frame,
        assignments=assignments.frame,
        spec=spec,
        out_dir=out_dir,
        source_cohort=cohort.path,
        source_assignments=assignments.path,
        source_receipt=source,
        step_id=step_id,
    )

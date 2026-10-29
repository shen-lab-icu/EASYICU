"""Host-owned sizes of verified existing clusters, without refitting them."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Mapping

import pandas as pd

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...canonical_json import canonical_sha256
from ...contracts.cluster_sizes import (
    CLUSTER_SIZE_ACTION,
    CLUSTER_SIZE_KIND,
    CLUSTER_SIZE_OUTPUTS,
    ClusterSizeSpec,
    cluster_sizes_cohort_input,
)
from ...contracts.host_scaffold import HostScaffoldedScript
from ...contracts.ownership_verdict import OwnershipVerdict
from ...contracts.phenotype_comparison import (
    TRAJECTORY_FREEZE_PRODUCT,
    TRAJECTORY_FROZEN_STATUS,
    bound_freeze_status,
)
from ...methods.cluster_sizes import compute_cluster_sizes
from ...schema import AnalysisStep, ValidationFinding
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import load_typed_input, sha256_file

CLUSTER_SIZES_ANALYSIS_KIND = CLUSTER_SIZE_KIND


def cluster_sizes_consumed_input_keys(step: AnalysisStep) -> tuple[str, ...]:
    spec = getattr(step, "cluster_size_spec", None)
    cohort = cluster_sizes_cohort_input(step)
    if spec is None or cohort is None:
        return ()
    return (
        cohort,
        spec.assignment_source,
        *(
            (TRAJECTORY_FREEZE_PRODUCT,)
            if spec.assignment_source == "table:cluster_assignments"
            else ()
        ),
    )


def cluster_sizes_executor_verdict(step: AnalysisStep) -> OwnershipVerdict:
    spec = getattr(step, "cluster_size_spec", None)
    if step.scientific_action_id != CLUSTER_SIZE_ACTION and spec is None:
        return OwnershipVerdict.wrong_shape(
            CLUSTER_SIZE_KIND, reason="different scientific action"
        )
    if spec is None:
        return OwnershipVerdict.incomplete_declaration(
            CLUSTER_SIZE_KIND,
            missing=("cluster_size_spec",),
            reason="Declare the unique analysis unit, frozen label source and expected cluster roster.",
        )
    consumed = cluster_sizes_consumed_input_keys(step)
    valid = (
        step.scientific_action_id == CLUSTER_SIZE_ACTION
        and bool(consumed)
        and len(step.inputs) == len(set(step.inputs))
        and {key for key in step.inputs if ":" in key} == set(consumed)
        and {key for key in step.inputs if ":" not in key} == {spec.unit_id_column}
        and tuple(step.expected_outputs) == CLUSTER_SIZE_OUTPUTS
        and step.planned_analysis_role in {"secondary", "auxiliary"}
        and not step.model_requirements
        and not step.family_primary_result_requirement
        and not any(
            getattr(step, name, None) is not None
            for name in type(step).model_fields
            if name.endswith("_spec") and name != "cluster_size_spec"
        )
    )
    if not valid:
        return OwnershipVerdict.incomplete_declaration(
            CLUSTER_SIZE_KIND,
            missing=("cluster_sizes_step_contract",),
            reason="Declare one typed cohort roster, frozen assignments, a trajectory freeze when applicable, and the size table/receipt outputs.",
        )
    return OwnershipVerdict.claim(
        CLUSTER_SIZE_KIND,
        reason="exact descriptive count of frozen cluster memberships",
    )


def cluster_sizes_executor_owns_step(step: AnalysisStep) -> bool:
    return cluster_sizes_executor_verdict(step).claimed


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


def _load_sources(
    *,
    spec: ClusterSizeSpec,
    typed_cohort_input: str,
    step_id: str,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    run_dir = Path(run_dir).resolve()
    cohort = load_typed_input(
        input_key=typed_cohort_input,
        run_dir=run_dir,
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        require_consumption_contract=True,
        minimum_row_count=1,
    )
    contract = cohort.binding.get("product_contract", {})
    if (
        contract.get("row_identity_column", spec.unit_id_column) != spec.unit_id_column
        or contract.get("analysis_unit", spec.analysis_unit) != spec.analysis_unit
    ):
        raise ValueError("cluster_sizes_analysis_unit_contract_mismatch")
    assignments = load_typed_input(
        input_key=spec.assignment_source,
        run_dir=run_dir,
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        require_consumption_contract=True,
        text_columns=(spec.assignment_unit_id_column,),
    )
    source: dict[str, Any] = {
        "source_assignments_sha256": assignments.sha256,
        "assignment_source": spec.assignment_source,
    }
    if spec.assignment_source == "table:phenotype_assignments":
        labels = assignments.frame
        required = {"source_cohort_sha256", "source_identity_column"}
        if (
            not required.issubset(labels.columns)
            or labels[list(required)].isna().any().any()
            or set(labels.source_cohort_sha256) != {cohort.sha256}
            or set(labels.source_identity_column) != {spec.unit_id_column}
        ):
            raise ValueError("cluster_sizes_assignment_source_mismatch")
        if (
            spec.assignment_unit_id_column != "unit_id"
            or spec.label_column != "cluster"
        ):
            raise ValueError("cluster_sizes_phenotype_assignment_schema_mismatch")
    else:
        if (
            spec.assignment_unit_id_column != spec.unit_id_column
            or spec.label_column != "cluster"
        ):
            raise ValueError("cluster_sizes_trajectory_assignment_schema_mismatch")
        payload = (
            resolved_inputs
            if isinstance(resolved_inputs, Mapping)
            else json.loads(Path(resolved_inputs).read_text("utf-8"))
        )
        freeze = payload.get("inputs", {}).get(TRAJECTORY_FREEZE_PRODUCT)
        if (
            not isinstance(freeze, Mapping)
            or freeze.get("declared_kind") != "artifact"
            or freeze.get("product") != "stability_freeze"
        ):
            raise ValueError("cluster_sizes_freeze_binding_missing")
        status = bound_freeze_status(freeze, run_dir)
        if status != TRAJECTORY_FROZEN_STATUS:
            raise ValueError("cluster_sizes_no_frozen_solution")
        freeze_path = (Path(run_dir) / str(freeze["relative_path"])).resolve()
        freeze_record = json.loads(freeze_path.read_text("utf-8"))
        assignment_producer = assignments.binding.get("produced_by_step")
        if (
            not assignment_producer
            or freeze.get("produced_by_step") != assignment_producer
        ):
            raise ValueError("cluster_sizes_freeze_assignment_producer_mismatch")
        if freeze_record.get("selected_n_clusters") != len(spec.expected_clusters):
            raise ValueError("cluster_sizes_expected_roster_differs_from_frozen_count")
        source.update(source_freeze_sha256=freeze["sha256"], freeze_status=status)
    return cohort, assignments, source


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


def cluster_sizes_output_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    context: Any = None,
    resolved_input_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    out_dir: Path | None = None,
) -> list[ValidationFinding]:
    if step.scientific_action_id != CLUSTER_SIZE_ACTION:
        return []
    try:
        if (
            not cluster_sizes_executor_owns_step(step)
            or out_dir is None
            or resolved_input_bindings is None
        ):
            raise ValueError(
                "size validation requires the exact declared step and verified inputs"
            )
        spec = step.cluster_size_spec
        if context is not None and spec.unit_id_column not in context.cohort.id_columns:
            raise ValueError("size identity is not a declared cohort identity")
        cohort, assignments, source = _load_sources(
            spec=spec,
            typed_cohort_input=cluster_sizes_cohort_input(step),
            step_id=step.step_id,
            run_dir=Path(out_dir).parents[2],
            resolved_inputs={
                "step_id": step.step_id,
                "inputs": dict(resolved_input_bindings),
            },
        )
        expected = compute_cluster_sizes(
            cohort.frame, assignments.frame, spec
        ).model_dump(mode="json")
        receipt_path = Path(out_dir) / "cluster_sizes_receipt.json"
        table_path = Path(out_dir) / "cluster_sizes.csv"
        if receipt_path.is_symlink() or table_path.is_symlink():
            raise ValueError("size outputs must be regular local files")
        receipt = json.loads(receipt_path.read_text("utf-8"))
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("refit_performed") is not False
        ):
            raise ValueError("size receipt must preserve its typed no-refit contract")
        expected_csv = pd.DataFrame(expected["rows"]).to_csv(index=False)
        receipt_metadata = {
            "schema_version": "easyicu.cluster_sizes_receipt/1",
            "authority_scope": "analysis_only",
            "refit_performed": False,
            "near_empty_rule": "strictly_below_any_declared_threshold"
            if spec.near_empty_n is not None or spec.near_empty_fraction is not None
            else "not_requested",
        }
        if any(receipt.get(key) != value for key, value in receipt_metadata.items()):
            raise ValueError(
                "size receipt changes its calculation or authority contract"
            )
        if (
            receipt.get("specification") != spec.model_dump(mode="json")
            or canonical_sha256(receipt.get("result")) != canonical_sha256(expected)
            or receipt.get("result_sha256") != canonical_sha256(expected)
            or receipt.get("source_cohort_sha256") != cohort.sha256
            or receipt.get("source_assignments_sha256") != assignments.sha256
            or receipt.get("output_sha256") != sha256_file(table_path)
            or table_path.read_text("utf-8") != expected_csv
            or any(receipt.get(key) != value for key, value in source.items())
        ):
            raise ValueError(
                "size result or provenance differs from cohort/assignment replay"
            )
        if (
            step_summary.get("step_id") != step.step_id
            or step_summary.get("deterministic_standard_analysis") != CLUSTER_SIZE_KIND
            or step_summary.get("specification_sha256")
            != expected["specification_sha256"]
            or step_summary.get("result_sha256") != canonical_sha256(expected)
            or step_summary.get("source_cohort_sha256") != cohort.sha256
            or step_summary.get("source_assignments_sha256") != assignments.sha256
            or step_summary.get("n_total") != expected["input_n"]
            or step_summary.get("assigned_n") != expected["assigned_n"]
            or step_summary.get("missing_label_n") != expected["missing_label_n"]
            or step_summary.get("analysis_unit") != spec.analysis_unit
            or step_summary.get("authority_scope") != "analysis_only"
            or step_summary.get("paper_authorization_allowed") is not False
            or step_summary.get("refit_performed") is not False
            or step_summary.get("output_files")
            != dict(zip(CLUSTER_SIZE_OUTPUTS, (table_path.name, receipt_path.name)))
            or any(step_summary.get(key) != value for key, value in source.items())
        ):
            raise ValueError("size summary differs from its sealed result")
        expected_summary = {
            "scientific_action_id": CLUSTER_SIZE_ACTION,
            "claim_ceiling": "analysis_only",
            "analysis_family": "phenotyping",
            "status": "ok",
            "analysis_status": "ok",
            "interpretation": "descriptive_existing_frozen_cluster_memberships",
            "degenerate_single_cluster": expected["degenerate_single_cluster"],
        }
        if any(
            step_summary.get(key) != value for key, value in expected_summary.items()
        ):
            raise ValueError("size summary changes its action or interpretation")
        return []
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        return [
            ValidationFinding(
                validator="cluster_sizes_output",
                severity="error",
                message=str(exc),
                detail={
                    "step_id": step.step_id,
                    "reason_code": "cluster_sizes_output_mismatch",
                },
            )
        ]

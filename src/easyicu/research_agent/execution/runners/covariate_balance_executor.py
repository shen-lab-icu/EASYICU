"""Host-owned standalone balance diagnostic on a verified cohort artifact."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Mapping

import pandas as pd

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...canonical_json import canonical_sha256
from ...contracts.covariate_balance import (
    BALANCE_ACTION,
    BALANCE_KIND,
    BALANCE_OUTPUTS,
    CovariateBalanceSpec,
)
from ...contracts.host_scaffold import HostScaffoldedScript
from ...contracts.ownership_verdict import OwnershipVerdict
from ...methods.covariate_balance import compute_covariate_balance
from ...schema import AnalysisStep, ValidationFinding
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import load_typed_input, sha256_file, sole_typed_cohort_input

COVARIATE_BALANCE_ANALYSIS_KIND = BALANCE_KIND


def covariate_balance_executor_verdict(step: AnalysisStep) -> OwnershipVerdict:
    spec = getattr(step, "covariate_balance_spec", None)
    if step.scientific_action_id != BALANCE_ACTION and spec is None:
        return OwnershipVerdict.wrong_shape(
            BALANCE_KIND, reason="different scientific action"
        )
    if spec is None:
        return OwnershipVerdict.incomplete_declaration(
            BALANCE_KIND,
            missing=("covariate_balance_spec",),
            reason="Declare the exact covariates, units, treatment and existing weights.",
        )
    required = {
        spec.unit_id_column,
        spec.treatment_column,
        *(item.column for item in spec.covariates),
    }
    if spec.weight_column:
        required.add(spec.weight_column)
    cohort = sole_typed_cohort_input(step)
    valid = (
        step.scientific_action_id == BALANCE_ACTION
        and cohort is not None
        and len(step.inputs) == len(set(step.inputs))
        and {key for key in step.inputs if ":" in key} == {cohort}
        and {key for key in step.inputs if ":" not in key} == required
        and tuple(step.expected_outputs) == BALANCE_OUTPUTS
        and step.planned_analysis_role in {"secondary", "auxiliary"}
        and not step.model_requirements
        and not step.family_primary_result_requirement
        and not any(
            getattr(step, name, None) is not None
            for name in type(step).model_fields
            if name.endswith("_spec") and name != "covariate_balance_spec"
        )
    )
    if not valid:
        return OwnershipVerdict.incomplete_declaration(
            BALANCE_KIND,
            missing=("covariate_balance_step_contract",),
            reason="Declare one typed cohort, the exact spec columns and balance table/receipt outputs without a model refit.",
        )
    return OwnershipVerdict.claim(
        BALANCE_KIND, reason="exact declared descriptive balance diagnostic"
    )


def covariate_balance_executor_owns_step(step: AnalysisStep) -> bool:
    return covariate_balance_executor_verdict(step).claimed


def covariate_balance_executor_scaffold(
    step: AnalysisStep,
    *,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> HostScaffoldedScript:
    if not covariate_balance_executor_owns_step(step):
        raise ValueError("covariate_balance_step_contract_invalid")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    code = textwrap.dedent(f"""
        import json, os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.typed_input_binding import load_step_cohort_frame
        from easyicu.research_agent.execution.runners.covariate_balance_executor import run_covariate_balance_from_bound

        frame, cohort_path = load_step_cohort_frame(
            typed_cohort_input={sole_typed_cohort_input(step)!r}, require_consumption_contract=True,
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
        summary = run_covariate_balance_from_bound(
            frame=frame, source_cohort=cohort_path,
            spec=json.loads({step.covariate_balance_spec.model_dump_json()!r}),
            step_id={step.step_id!r}, typed_cohort_input={sole_typed_cohort_input(step)!r},
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    """).strip()
    return HostScaffoldedScript(prologue=code, epilogue=epilogue)


def covariate_balance_executor_code(step: AnalysisStep, **kwargs: Any) -> str:
    return covariate_balance_executor_scaffold(step, **kwargs).assembled()


def _check_unit_contract(
    binding: Mapping[str, Any], spec: CovariateBalanceSpec
) -> None:
    contract = binding.get("product_contract", {})
    if contract.get("row_identity_column", spec.unit_id_column) != spec.unit_id_column:
        raise ValueError("covariate_balance_input_identity_contract_mismatch")
    if contract.get("analysis_unit", spec.analysis_unit) != spec.analysis_unit:
        raise ValueError("covariate_balance_input_analysis_unit_mismatch")


def run_covariate_balance(
    *,
    frame: pd.DataFrame,
    spec: CovariateBalanceSpec | Mapping[str, Any],
    out_dir: Path,
    source_cohort: Path | None = None,
    step_id: str = "covariate_balance",
) -> dict[str, Any]:
    spec = CovariateBalanceSpec.model_validate(spec)
    result = compute_covariate_balance(frame, spec)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "covariate_balance.csv"
    pd.DataFrame([row.model_dump(mode="json") for row in result.rows]).to_csv(
        table_path, index=False
    )
    receipt = {
        "schema_version": "easyicu.covariate_balance_receipt/1",
        "specification": spec.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
        "result_sha256": canonical_sha256(result.model_dump(mode="json")),
        "source_cohort_sha256": sha256_file(source_cohort) if source_cohort else None,
        "output_sha256": sha256_file(table_path),
        "variance_convention": "equal_weight_pooled_unweighted_sample_variance_ddof1_fixed_before_after",
        "ess_formula": "(sum(w))^2/sum(w^2)",
        "authority_scope": "analysis_only",
        "refit_performed": False,
    }
    receipt_path = out_dir / "covariate_balance_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "analysis_family": "causal_emulation",
        "deterministic_standard_analysis": BALANCE_KIND,
        "scientific_action_id": BALANCE_ACTION,
        "authority_scope": "analysis_only",
        "claim_ceiling": "analysis_only",
        "paper_authorization_allowed": False,
        "refit_performed": False,
        "analysis_unit": spec.analysis_unit,
        "n_total": result.input_n,
        "specification_sha256": result.specification_sha256,
        "result_sha256": receipt["result_sha256"],
        "source_cohort_sha256": receipt["source_cohort_sha256"],
        "output_files": dict(
            zip(BALANCE_OUTPUTS, (table_path.name, receipt_path.name))
        ),
        "interpretation": "measured_covariate_balance_descriptive_no_causal_effect",
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def run_covariate_balance_from_bound(
    *,
    frame: pd.DataFrame,
    source_cohort: Path,
    spec: CovariateBalanceSpec | Mapping[str, Any],
    step_id: str,
    typed_cohort_input: str,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    spec = CovariateBalanceSpec.model_validate(spec)
    bound = load_typed_input(
        input_key=typed_cohort_input,
        run_dir=Path(run_dir).resolve(),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        require_consumption_contract=True,
        minimum_row_count=1,
    )
    _check_unit_contract(bound.binding, spec)
    if Path(source_cohort).resolve() != bound.path.resolve() or not frame.equals(
        bound.frame
    ):
        raise ValueError("covariate_balance_loaded_cohort_mismatch")
    return run_covariate_balance(
        frame=bound.frame,
        spec=spec,
        out_dir=out_dir,
        source_cohort=bound.path,
        step_id=step_id,
    )


def covariate_balance_output_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    context: Any = None,
    resolved_input_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    out_dir: Path | None = None,
) -> list[ValidationFinding]:
    if step.scientific_action_id != BALANCE_ACTION:
        return []
    try:
        if (
            not covariate_balance_executor_owns_step(step)
            or out_dir is None
            or resolved_input_bindings is None
        ):
            raise ValueError(
                "balance validation requires the exact declared step and verified inputs"
            )
        spec = step.covariate_balance_spec
        if context is not None and spec.unit_id_column not in context.cohort.id_columns:
            raise ValueError("balance identity is not a declared cohort identity")
        cohort = load_typed_input(
            input_key=sole_typed_cohort_input(step),
            run_dir=Path(out_dir).resolve().parents[2],
            resolved_inputs={
                "step_id": step.step_id,
                "inputs": dict(resolved_input_bindings),
            },
            step_id=step.step_id,
            require_consumption_contract=True,
            minimum_row_count=1,
        )
        _check_unit_contract(cohort.binding, spec)
        expected = compute_covariate_balance(cohort.frame, spec).model_dump(mode="json")
        receipt_path = Path(out_dir) / "covariate_balance_receipt.json"
        table_path = Path(out_dir) / "covariate_balance.csv"
        if receipt_path.is_symlink() or table_path.is_symlink():
            raise ValueError("balance outputs must be regular local files")
        receipt = json.loads(receipt_path.read_text("utf-8"))
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("refit_performed") is not False
        ):
            raise ValueError(
                "balance receipt must preserve its typed no-refit contract"
            )
        expected_csv = pd.DataFrame(expected["rows"]).to_csv(index=False)
        receipt_metadata = {
            "schema_version": "easyicu.covariate_balance_receipt/1",
            "variance_convention": "equal_weight_pooled_unweighted_sample_variance_ddof1_fixed_before_after",
            "ess_formula": "(sum(w))^2/sum(w^2)",
            "authority_scope": "analysis_only",
            "refit_performed": False,
        }
        if any(receipt.get(key) != value for key, value in receipt_metadata.items()):
            raise ValueError(
                "balance receipt changes its calculation or authority contract"
            )
        if (
            receipt.get("specification") != spec.model_dump(mode="json")
            or canonical_sha256(receipt.get("result")) != canonical_sha256(expected)
            or receipt.get("result_sha256") != canonical_sha256(expected)
            or receipt.get("source_cohort_sha256") != cohort.sha256
            or receipt.get("output_sha256") != sha256_file(table_path)
            or table_path.read_text("utf-8") != expected_csv
        ):
            raise ValueError(
                "balance result or provenance differs from independent cohort replay"
            )
        if (
            step_summary.get("step_id") != step.step_id
            or step_summary.get("deterministic_standard_analysis") != BALANCE_KIND
            or step_summary.get("specification_sha256")
            != expected["specification_sha256"]
            or step_summary.get("result_sha256") != canonical_sha256(expected)
            or step_summary.get("source_cohort_sha256") != cohort.sha256
            or step_summary.get("n_total") != expected["input_n"]
            or step_summary.get("analysis_unit") != spec.analysis_unit
            or step_summary.get("authority_scope") != "analysis_only"
            or step_summary.get("paper_authorization_allowed") is not False
            or step_summary.get("refit_performed") is not False
            or step_summary.get("output_files")
            != dict(zip(BALANCE_OUTPUTS, (table_path.name, receipt_path.name)))
        ):
            raise ValueError("balance step summary differs from its sealed result")
        expected_summary = {
            "scientific_action_id": BALANCE_ACTION,
            "claim_ceiling": "analysis_only",
            "analysis_family": "causal_emulation",
            "status": "ok",
            "analysis_status": "ok",
            "interpretation": "measured_covariate_balance_descriptive_no_causal_effect",
        }
        if any(
            step_summary.get(key) != value for key, value in expected_summary.items()
        ):
            raise ValueError("balance summary changes its action or interpretation")
        return []
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        return [
            ValidationFinding(
                validator="covariate_balance_output",
                severity="error",
                message=str(exc),
                detail={
                    "step_id": step.step_id,
                    "reason_code": "covariate_balance_output_mismatch",
                },
            )
        ]

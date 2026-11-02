"""Host-owned standalone balance diagnostic on a verified cohort artifact."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Mapping

import pandas as pd

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...canonical_json import canonical_sha256
from ...contracts.diagnostic_specs import (
    BALANCE_ACTION,
    BALANCE_KIND,
    BALANCE_OUTPUTS,
    CovariateBalanceSpec,
)
from ...contracts.host_scaffold import HostScaffoldedScript
from ...methods.propensity_weighting import (
    compute_covariate_balance,
    _check_unit_contract as _check_unit_contract,
    covariate_balance_executor_owns_step as covariate_balance_executor_owns_step,
    covariate_balance_executor_verdict as covariate_balance_executor_verdict,
    covariate_balance_output_findings as covariate_balance_output_findings,
)
from ...schema import AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import load_typed_input, sha256_file, sole_typed_cohort_input

COVARIATE_BALANCE_ANALYSIS_KIND = BALANCE_KIND






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

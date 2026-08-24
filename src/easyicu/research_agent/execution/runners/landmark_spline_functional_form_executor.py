"""Deterministic functional-form projection for a signed landmark spline.

The signed primary owner already fits the nested spline and linear models on
one exact landmark population.  This executor exposes that registered model
comparison as the Planner-requested sensitivity table without refitting a
different cohort or sending the task to the Coder.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Mapping

from ...authority.current_case_scientific_runtime import (
    LandmarkSplineRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ...numeric_scalars import coerce_finite_float
from ...schema import AnalysisPlan, AnalysisStep

LANDMARK_SPLINE_FUNCTIONAL_FORM_ANALYSIS_KIND = (
    "signed_landmark_spline_functional_form"
)

_REQUIRED_DIAGNOSTIC_COLUMNS = frozenset(
    {
        "n",
        "events",
        "linear_aic",
        "spline_aic",
        "linear_bic",
        "spline_bic",
        "likelihood_ratio_statistic",
        "additional_spline_parameters",
        "nonlinearity_p_value",
    }
)


def landmark_spline_functional_form_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: LandmarkSplineRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    if authority is None:
        return False
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSplineRuntimeAuthority):
        return False
    sealed.governed_step(plan)
    outputs = tuple(str(value) for value in step.expected_outputs)
    contracts = {
        item.input_key: item.mode for item in step.input_consumption_contracts
    }
    signed_inputs = {
        sealed.downstream_parent_product,
        sealed.linear_sensitivity_product,
    }
    return bool(
        step.planned_analysis_role == "sensitivity"
        and step.scientific_capability is None
        and step.robustness_replay_spec is None
        and len(step.sensitivity_spec_ids) == 1
        and len(outputs) == 1
        and outputs[0].startswith("table:")
        and signed_inputs.issubset(step.inputs)
        and all(contracts.get(value) == "all_rows" for value in signed_inputs)
    )


def landmark_spline_functional_form_executor_code(
    step: AnalysisStep,
    *,
    authority: LandmarkSplineRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
) -> str:
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSplineRuntimeAuthority):
        raise TypeError("functional-form executor requires landmark authority")
    authority_json = json.dumps(sealed.model_dump(mode="json"), sort_keys=True)
    step_json = json.dumps(step.model_dump(mode="json"), sort_keys=True)
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.schema import AnalysisStep
        from easyicu.research_agent.execution.runners.landmark_spline_functional_form_executor import (
            run_bound_landmark_spline_functional_form,
        )

        run_bound_landmark_spline_functional_form(
            step=AnalysisStep.model_validate(json.loads({json.dumps(step_json)})),
            authority=json.loads({json.dumps(authority_json)}),
            runtime_projection_sha256={runtime_projection_sha256!r},
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
        )
        """
    ).strip()


def run_landmark_spline_functional_form(
    *,
    step: AnalysisStep,
    authority: LandmarkSplineRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    linear_sensitivity: Any,
    linear_evidence_id: str,
    out_dir: Path,
    input_bindings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    import pandas as pd

    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSplineRuntimeAuthority):
        raise TypeError("functional-form executor received wrong authority kind")
    if len(str(runtime_projection_sha256)) != 64:
        raise ValueError("runtime projection digest is required")
    if len(linear_sensitivity) != 1:
        raise ValueError("signed linear sensitivity must contain exactly one row")
    missing = sorted(_REQUIRED_DIAGNOSTIC_COLUMNS - set(linear_sensitivity.columns))
    if missing:
        raise ValueError(
            "signed linear sensitivity lacks functional-form diagnostics: "
            + ", ".join(missing)
        )
    row = linear_sensitivity.iloc[0]
    n = int(coerce_finite_float(row["n"], label="complete-case n"))
    events = int(coerce_finite_float(row["events"], label="event count"))
    extra_df = int(
        coerce_finite_float(
            row["additional_spline_parameters"],
            label="additional spline parameters",
        )
    )
    if n <= 0 or events < 0 or events > n or extra_df <= 0:
        raise ValueError("signed functional-form diagnostic counts are invalid")
    output_product = str(step.expected_outputs[0])
    output_name = output_product.partition(":")[2]
    if not output_name:
        raise ValueError("functional-form output product has no name")
    output_path = out_dir / f"{output_name}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(
        [
            {
                "check": "restricted_cubic_spline_vs_linear",
                "method": "nested_logistic_likelihood_ratio_test",
                "n_complete_case": n,
                "event_n": events,
                "linear_aic": coerce_finite_float(row["linear_aic"], label="linear AIC"),
                "spline_aic": coerce_finite_float(row["spline_aic"], label="spline AIC"),
                "linear_bic": coerce_finite_float(row["linear_bic"], label="linear BIC"),
                "spline_bic": coerce_finite_float(row["spline_bic"], label="spline BIC"),
                "likelihood_ratio_statistic": coerce_finite_float(
                    row["likelihood_ratio_statistic"], label="likelihood ratio"
                ),
                "additional_spline_parameters": extra_df,
                "nonlinearity_p_value": coerce_finite_float(
                    row["nonlinearity_p_value"], label="nonlinearity p-value"
                ),
                "source_evidence_id": linear_evidence_id,
            }
        ]
    )
    result.to_csv(output_path, index=False)
    summary = {
        "step": step.step_id,
        "status": "completed",
        "analysis_family": "association",
        "analysis_kind": LANDMARK_SPLINE_FUNCTIONAL_FORM_ANALYSIS_KIND,
        "interpretation_class": "descriptive_prognostic_association",
        "n_complete_case": n,
        "event_n": events,
        "input_bindings": input_bindings or [],
        "output_files": {output_product: output_path.name},
    }
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary


def run_bound_landmark_spline_functional_form(
    *,
    step: AnalysisStep,
    authority: LandmarkSplineRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    run_dir: Path,
    resolved_inputs: Path,
    out_dir: Path,
) -> dict[str, Any]:
    from .typed_input_binding import load_typed_input

    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSplineRuntimeAuthority):
        raise TypeError("functional-form executor received wrong authority kind")
    manifest = json.loads(resolved_inputs.read_text(encoding="utf-8"))
    receipts = []
    loaded = {}
    for input_key in (
        sealed.downstream_parent_product,
        sealed.linear_sensitivity_product,
    ):
        bound = load_typed_input(
            input_key=input_key,
            run_dir=run_dir,
            resolved_inputs=manifest,
            step_id=step.step_id,
            expected_declared_kind="table",
            expected_evidence_kind="table",
            minimum_row_count=(
                2 if input_key == sealed.downstream_parent_product else 1
            ),
            require_consumption_contract=True,
        )
        loaded[input_key] = bound
        receipts.append(
            {
                "input_key": input_key,
                "evidence_id": bound.evidence_id,
                "sha256": bound.sha256,
                "loaded": True,
                "row_count": bound.row_count,
            }
        )
    linear = loaded[sealed.linear_sensitivity_product]
    return run_landmark_spline_functional_form(
        step=step,
        authority=sealed,
        runtime_projection_sha256=runtime_projection_sha256,
        linear_sensitivity=linear.frame,
        linear_evidence_id=linear.evidence_id,
        out_dir=out_dir,
        input_bindings=receipts,
    )


__all__ = [
    "LANDMARK_SPLINE_FUNCTIONAL_FORM_ANALYSIS_KIND",
    "landmark_spline_functional_form_executor_code",
    "landmark_spline_functional_form_executor_owns_step",
    "run_bound_landmark_spline_functional_form",
    "run_landmark_spline_functional_form",
]

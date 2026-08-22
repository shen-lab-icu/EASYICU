"""Deterministic robustness projection for a signed landmark-spline result.

This owner does not fit another model. It projects the digest-bound contrast
and linear-sensitivity tables produced by ``LandmarkSplineRuntimeAuthority``
into the generic robustness products required by downstream renderers.
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any, Mapping

from ...authority.current_case_scientific_runtime import (
    LandmarkSplineRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ...schema import AnalysisPlan, AnalysisStep
from .deterministic_robustness import (
    declared_robustness_product_registrations,
    robustness_replay_spec_is_emittable,
)

LANDMARK_SPLINE_ROBUSTNESS_ANALYSIS_KIND = "signed_landmark_spline_robustness"

_REQUIRED_REPLAY_OUTPUTS = frozenset(
    {
        "robustness_matrix",
        "robustness_summary",
        "primary_effect",
        "complete_case_n",
        "missingness_strategy_notes",
    }
)


def _declared_replay_outputs(step: AnalysisStep) -> frozenset[str]:
    spec = step.robustness_replay_spec
    if spec is None:
        return frozenset()
    return frozenset(item.output for item in spec.products)


def landmark_spline_robustness_executor_owns_step(
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
    return bool(
        step.planned_analysis_role == "sensitivity"
        and sealed.downstream_parent_product in step.inputs
        and sealed.linear_sensitivity_product in step.inputs
        and robustness_replay_spec_is_emittable(step)
        and _declared_replay_outputs(step) == _REQUIRED_REPLAY_OUTPUTS
    )


def landmark_spline_robustness_executor_code(
    step: AnalysisStep,
    *,
    authority: LandmarkSplineRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
) -> str:
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSplineRuntimeAuthority):
        raise TypeError("landmark robustness executor requires landmark authority")
    authority_json = json.dumps(sealed.model_dump(mode="json"), sort_keys=True)
    step_json = json.dumps(step.model_dump(mode="json"), sort_keys=True)
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.schema import AnalysisStep
        from easyicu.research_agent.execution.runners.landmark_spline_robustness_executor import (
            run_bound_landmark_spline_robustness,
        )

        run_bound_landmark_spline_robustness(
            step=AnalysisStep.model_validate(json.loads({json.dumps(step_json)})),
            authority=json.loads({json.dumps(authority_json)}),
            runtime_projection_sha256={runtime_projection_sha256!r},
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
        )
        """
    ).strip()


def _finite(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _contrast_coordinate_column(frame: Any) -> str:
    reserved = {
        "adjusted_odds_ratio",
        "ci_low",
        "ci_high",
    }
    candidates = [
        str(column)
        for column in frame.columns
        if str(column) not in reserved and "reference" not in str(column).casefold()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "signed landmark contrasts must expose one non-reference coordinate"
        )
    return candidates[0]


def _summary_rows(matrix: Any) -> Any:
    import pandas as pd

    rows = []
    for axis, group in matrix.groupby("axis", sort=False, dropna=False):
        converged = group[group["converged"].astype(bool)]
        rows.append(
            {
                "axis": axis,
                "total_specs": int(len(group)),
                "converged_specs": int(len(converged)),
                "non_independent_specs": int(
                    (group["independent_variant"] == False).sum()  # noqa: E712
                ),
                "range_low": (
                    float(converged["ci_low"].min()) if not converged.empty else None
                ),
                "range_high": (
                    float(converged["ci_high"].max()) if not converged.empty else None
                ),
            }
        )
    return pd.DataFrame(rows)


def run_landmark_spline_robustness(
    *,
    step: AnalysisStep,
    authority: LandmarkSplineRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    contrasts: Any,
    linear_sensitivity: Any,
    contrast_evidence_id: str,
    linear_evidence_id: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Project already-fitted signed outputs into the robustness contract."""

    import pandas as pd

    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, LandmarkSplineRuntimeAuthority):
        raise TypeError("landmark robustness executor received wrong authority kind")
    if len(str(runtime_projection_sha256)) != 64:
        raise ValueError("runtime projection digest is required")
    if not robustness_replay_spec_is_emittable(step):
        raise ValueError("landmark robustness step has no emittable replay contract")
    required_effect_columns = {
        "adjusted_odds_ratio",
        "ci_low",
        "ci_high",
    }
    if not required_effect_columns <= set(contrasts.columns):
        raise ValueError("signed landmark contrasts have an incompatible schema")
    if not required_effect_columns | {"n", "events"} <= set(linear_sensitivity.columns):
        raise ValueError("signed linear sensitivity has an incompatible schema")
    if len(contrasts) < 2 or len(linear_sensitivity) != 1:
        raise ValueError("signed landmark robustness inputs have unexpected rows")

    coordinate = _contrast_coordinate_column(contrasts)
    ordered = contrasts.sort_values(coordinate)
    upper = ordered.iloc[-1]
    linear = linear_sensitivity.iloc[0]
    primary_or = _finite(upper["adjusted_odds_ratio"], label="upper contrast OR")
    primary_low = _finite(upper["ci_low"], label="upper contrast CI low")
    primary_high = _finite(upper["ci_high"], label="upper contrast CI high")
    complete_case_n = int(_finite(linear["n"], label="complete-case n"))
    events = int(_finite(linear["events"], label="event count"))
    coordinate_value = _finite(upper[coordinate], label="upper contrast coordinate")

    matrix_columns = [
        "spec_id",
        "effect_scale",
        "point_estimate",
        "ci_low",
        "ci_high",
        "modeled_analytic_n",
        "axis",
        "converged",
        "model_contract_n",
        "event_n",
        "model_id",
        "source_model_id",
        "exposure_source",
        "complete_case",
        "mean_imputation",
        "median_imputation",
        "independent_variant",
        "notes",
        "evidence_id",
    ]
    base = {
        "effect_scale": "OR",
        "modeled_analytic_n": complete_case_n,
        "converged": True,
        "model_contract_n": complete_case_n,
        "event_n": events,
        "source_model_id": "signed_landmark_spline_primary",
        "exposure_source": sealed.exposure_column,
        "complete_case": True,
        "mean_imputation": False,
        "median_imputation": False,
    }
    rows = [
        {
            **base,
            "spec_id": "signed_upper_boundary_contrast",
            "point_estimate": primary_or,
            "ci_low": primary_low,
            "ci_high": primary_high,
            "axis": "primary",
            "model_id": "signed_landmark_spline_upper_boundary_contrast",
            "independent_variant": True,
            "notes": (
                f"Predeclared upper curve-boundary contrast at {coordinate}="
                f"{coordinate_value:g}; a display anchor, not a scalar summary "
                "of the nonlinear curve."
            ),
            "evidence_id": contrast_evidence_id,
        },
        {
            **base,
            "spec_id": "signed_linear_functional_form_sensitivity",
            "point_estimate": _finite(
                linear["adjusted_odds_ratio"], label="linear sensitivity OR"
            ),
            "ci_low": _finite(linear["ci_low"], label="linear sensitivity CI low"),
            "ci_high": _finite(linear["ci_high"], label="linear sensitivity CI high"),
            "axis": "functional_form",
            "model_id": "signed_landmark_linear_sensitivity",
            "independent_variant": True,
            "notes": "Prespecified linear per-unit functional-form sensitivity.",
            "evidence_id": linear_evidence_id,
        },
        {
            **base,
            "spec_id": "signed_complete_case_analysis_set",
            "point_estimate": primary_or,
            "ci_low": primary_low,
            "ci_high": primary_high,
            "axis": "missing",
            "model_id": "signed_landmark_complete_case_documentation",
            "independent_variant": False,
            "notes": (
                "Documents the signed primary model's complete-case analysis set; "
                "this is not an independent refit and no imputation was performed."
            ),
            "evidence_id": contrast_evidence_id,
        },
    ]
    matrix = pd.DataFrame(rows, columns=matrix_columns)
    summary_table = _summary_rows(matrix)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = declared_robustness_product_registrations(step)
    expected = set(step.expected_outputs)
    if set(files) != expected:
        raise ValueError("landmark robustness output registrations are incomplete")

    matrix_identity = next(
        key
        for key in expected
        if step.robustness_replay_spec.output_for(key.partition(":")[2])
        == "robustness_matrix"
    )
    summary_identity = next(
        key
        for key in expected
        if step.robustness_replay_spec.output_for(key.partition(":")[2])
        == "robustness_summary"
    )
    primary_identity = next(
        key
        for key in expected
        if step.robustness_replay_spec.output_for(key.partition(":")[2])
        == "primary_effect"
    )
    complete_identity = next(
        key
        for key in expected
        if step.robustness_replay_spec.output_for(key.partition(":")[2])
        == "complete_case_n"
    )
    notes_identity = next(
        key
        for key in expected
        if step.robustness_replay_spec.output_for(key.partition(":")[2])
        == "missingness_strategy_notes"
    )
    matrix.to_csv(out_dir / files[matrix_identity], index=False)
    summary_table.to_csv(out_dir / files[summary_identity], index=False)
    (out_dir / files[primary_identity]).write_text(
        json.dumps(
            {
                "statistic": primary_identity.partition(":")[2],
                "value": primary_or,
                "ci_low": primary_low,
                "ci_high": primary_high,
                "effect_scale": "OR",
                "estimand_label": "upper signed curve-boundary contrast vs reference",
                "not_a_scalar_summary_of_nonlinearity": True,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (out_dir / files[complete_identity]).write_text(
        json.dumps(
            {
                "statistic": complete_identity.partition(":")[2],
                "value": complete_case_n,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    notes = (
        "The signed landmark primary and linear sensitivity use the same "
        f"complete-case analysis set (n={complete_case_n}; events={events}). "
        "No imputation was performed. The repeated missing-data row documents "
        "that analysis set and is explicitly non-independent."
    )
    (out_dir / files[notes_identity]).write_text(notes + "\n", encoding="utf-8")
    summary = {
        "status": "ok",
        "analysis_family": "robustness_sensitivity",
        "authority_kind": LANDMARK_SPLINE_ROBUSTNESS_ANALYSIS_KIND,
        "runtime_projection_sha256": runtime_projection_sha256,
        "primary_effect": primary_or,
        "primary_or": primary_or,
        "primary_ci_low": primary_low,
        "primary_ci_high": primary_high,
        "primary_effect_scale": "OR",
        "primary_effect_label": "upper signed curve-boundary contrast vs reference",
        "primary_effect_is_nonlinear_curve_summary": False,
        "complete_case_n": complete_case_n,
        "n_converged_variants": int(matrix["converged"].sum()),
        "robustness_rows": rows,
        "robustness_panel": {"rows": rows},
        "limitations": [
            "The scalar display anchor does not summarize the whole nonlinear curve.",
            "The missing-data row documents the primary analysis set and is not an independent refit.",
        ],
        "output_files": files,
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary


def run_bound_landmark_spline_robustness(
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
        raise TypeError("landmark robustness executor received wrong authority kind")
    contrast = load_typed_input(
        input_key=sealed.downstream_parent_product,
        run_dir=run_dir,
        resolved_inputs=resolved_inputs,
        step_id=step.step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        minimum_row_count=2,
        require_consumption_contract=True,
    )
    linear = load_typed_input(
        input_key=sealed.linear_sensitivity_product,
        run_dir=run_dir,
        resolved_inputs=resolved_inputs,
        step_id=step.step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        minimum_row_count=1,
        require_consumption_contract=True,
    )
    return run_landmark_spline_robustness(
        step=step,
        authority=sealed,
        runtime_projection_sha256=runtime_projection_sha256,
        contrasts=contrast.frame,
        linear_sensitivity=linear.frame,
        contrast_evidence_id=contrast.evidence_id,
        linear_evidence_id=linear.evidence_id,
        out_dir=out_dir,
    )


__all__ = [
    "LANDMARK_SPLINE_ROBUSTNESS_ANALYSIS_KIND",
    "landmark_spline_robustness_executor_code",
    "landmark_spline_robustness_executor_owns_step",
    "run_bound_landmark_spline_robustness",
    "run_landmark_spline_robustness",
]

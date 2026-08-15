"""Deterministic complete-case Spearman summaries for two declared variables.

The Planner owns the predictor/outcome pair and the decision to request a
descriptive association.  This owner only consumes the exact two bare columns
in their declared order, verifies that both are numeric, and emits one typed
scalar statistic.  It does not adjust, impute, dichotomise, or promote the
association into a causal claim.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...schema import AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sole_typed_cohort_input

__all__ = [
    "DESCRIPTIVE_ASSOCIATION_ANALYSIS_KIND",
    "descriptive_association_executor_code",
    "descriptive_association_executor_owns_step",
    "run_descriptive_association",
]


DESCRIPTIVE_ASSOCIATION_ANALYSIS_KIND = "descriptive_spearman_association"


def _declared_columns(step: AnalysisStep) -> tuple[str, ...]:
    return tuple(
        str(value).strip()
        for value in step.inputs
        if str(value).strip() and ":" not in str(value).strip()
    )


def _statistic_product(step: AnalysisStep) -> str | None:
    if len(step.expected_outputs or []) != 1:
        return None
    kind, separator, product = str(step.expected_outputs[0]).partition(":")
    if kind != "statistic" or not separator:
        return None
    return product if re.fullmatch(r"[a-z][a-z0-9_]*", product) else None


def descriptive_association_executor_owns_step(step: AnalysisStep) -> bool:
    """Own one exact, non-causal, two-variable Spearman contract."""

    columns = _declared_columns(step)
    return bool(
        str(step.method or "").strip().casefold() == "descriptive_association"
        and str(step.planned_analysis_role or "").strip().casefold()
        in {"primary", "secondary", "auxiliary"}
        and len(columns) == 2
        and len(set(columns)) == 2
        and _statistic_product(step) is not None
        and sole_typed_cohort_input(step) is not None
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
        and step.exposure_outcome_distribution_spec is None
        and step.cohort_definition_spec is None
        and step.measurement_audit_spec is None
        and step.robustness_replay_spec is None
    )


def descriptive_association_executor_code(
    step: AnalysisStep,
    *,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    """Render the attributable entrypoint for the declared variable pair."""

    if not descriptive_association_executor_owns_step(step):
        raise ValueError("step is not owned by the descriptive-association executor")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    predictor, outcome = _declared_columns(step)
    statistic_product = _statistic_product(step)
    typed_cohort_input = sole_typed_cohort_input(step)
    if statistic_product is None or typed_cohort_input is None:  # pragma: no cover
        raise ValueError("descriptive association contract is incomplete")
    receipt_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope,
            frame_name="frame",
        )
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    code = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.descriptive_association_executor import (
            run_descriptive_association,
        )
        from easyicu.research_agent.execution.runners.typed_input_binding import (
            load_step_cohort_frame,
        )

        typed_cohort_input = {typed_cohort_input!r}
        frame, cohort_path = load_step_cohort_frame(
            typed_cohort_input=typed_cohort_input,
        )
        """
    ).strip()
    if receipt_code:
        code += "\n\n" + receipt_code.strip()
    code += "\n\n" + textwrap.dedent(
        f"""
        summary = run_descriptive_association(
            frame=frame,
            predictor={predictor!r},
            outcome={outcome!r},
            statistic_product={statistic_product!r},
            typed_cohort_input=typed_cohort_input,
            source_cohort=cohort_path,
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
        )
        """
    ).strip()
    if receipt_code:
        code += '\nsummary["plausibility_audit"] = plausibility_audit'
    code += "\n" + textwrap.dedent(
        """
        out_dir = Path(os.environ["STEP_OUT_DIR"])
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        """
    ).strip()
    return code


def _numeric_complete_cases(
    frame: pd.DataFrame,
    *,
    predictor: str,
    outcome: str,
) -> pd.DataFrame:
    missing = sorted({predictor, outcome} - set(frame.columns))
    if missing:
        raise RuntimeError(
            "Declared association columns are absent: " + ", ".join(missing)
        )
    source = frame[[predictor, outcome]]
    numeric = source.apply(pd.to_numeric, errors="coerce")
    for column in (predictor, outcome):
        if int(numeric[column].notna().sum()) != int(source[column].notna().sum()):
            raise RuntimeError(
                f"Descriptive association column {column!r} contains non-numeric values"
            )
    return numeric.dropna()


def run_descriptive_association(
    *,
    frame: pd.DataFrame,
    predictor: str,
    outcome: str,
    statistic_product: str,
    typed_cohort_input: str,
    source_cohort: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Compute one finite complete-case Spearman correlation and receipt."""

    if predictor == outcome:
        raise RuntimeError("Predictor and outcome must differ")
    if frame.empty:
        raise RuntimeError("Descriptive association cohort is empty")
    complete = _numeric_complete_cases(frame, predictor=predictor, outcome=outcome)
    if len(complete) < 3:
        raise RuntimeError("Descriptive association needs at least three complete cases")
    if complete[predictor].nunique(dropna=True) < 2:
        raise RuntimeError("Descriptive association predictor is constant")
    if complete[outcome].nunique(dropna=True) < 2:
        raise RuntimeError("Descriptive association outcome is constant")
    result = spearmanr(
        complete[predictor].to_numpy(),
        complete[outcome].to_numpy(),
        nan_policy="raise",
    )
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    if not math.isfinite(rho) or not math.isfinite(p_value):
        raise RuntimeError("Descriptive association produced a non-finite result")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    statistic_path = out_dir / f"{statistic_product}.json"
    statistic_payload = {
        "name": statistic_product,
        "value": rho,
        "effect_scale": "spearman_rho",
        "unit": "unitless",
        "p_value": p_value,
        "n_complete_case": int(len(complete)),
        "n_total": int(len(frame)),
        "predictor": predictor,
        "outcome": outcome,
        "missing_data_policy": "complete_case_no_imputation",
        "interpretation_class": "descriptive_noncausal_association",
    }
    statistic_path.write_text(
        json.dumps(statistic_payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "analysis_status": "ok",
        "method": "spearman_rank_correlation",
        "analysis_family": "descriptive",
        "deterministic_standard_analysis": DESCRIPTIVE_ASSOCIATION_ANALYSIS_KIND,
        "typed_cohort_input": typed_cohort_input,
        "source_cohort": Path(source_cohort).name,
        "predictor": predictor,
        "outcome": outcome,
        "n_total": int(len(frame)),
        "n_complete_case": int(len(complete)),
        "missing_data_policy": "complete_case_no_imputation",
        "interpretation_class": "descriptive_noncausal_association",
        "output_files": {
            f"statistic:{statistic_product}": statistic_path.name,
        },
    }

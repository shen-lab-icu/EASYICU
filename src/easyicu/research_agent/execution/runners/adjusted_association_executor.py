"""Deterministic owner for a fully declared adjusted-association model.

``table:adjusted_association_estimates`` is the most frequently declared product
no deterministic owner could emit -- 233 of 1812 recorded real steps, 175 of
them declaring exactly one model requirement -- and it is the paper's primary
result.  Until now the LLM coder wrote it every time, and the accumulated
repair guidance in ``plan_utils`` records how that went: a script that dropped
the whole cohort by numeric-coercing ``sex`` before dummy encoding, object-dtype
design matrices handed to statsmodels, and contracts "satisfied" with a null
estimate.

Nothing here decides science.  The planner fixes the outcome, the outcome type,
the method family, the exposure and -- since the contract gained
``covariates`` -- the adjustment set.  This executor claims the step only when
every one of those is declared, and computes exactly what was declared.

The fit is ``robustness.estimators.fit_estimator``, which the robustness panel
already uses: it drops incomplete rows, refuses a rank-deficient design rather
than silently dropping a declared predictor, refuses a non-binary outcome for a
logistic family, and reports non-convergence instead of a number.  Reusing it
means the primary estimate and its robustness variants come from one
implementation, so a disagreement between them is a real disagreement rather
than two estimators drifting apart.

The emitted table's shape is not invented here: ``execution/output_files.py::
bind_primary_output`` already reads this product and requires exactly one row
whose ``fit_status`` is ``fitted`` with finite ``estimate``/``ci_low``/
``ci_high``, and reads ``effect_scale`` and ``exposure``.  That reader is the
contract.
"""

from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...contracts.host_scaffold import HostScaffoldedScript
from ...robustness.estimators import fit_estimator
from ...schema import (
    ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
    PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,
    AnalysisStep,
    PlannedModelRequirement,
    _normalise_model_contract_token,
)
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import load_step_cohort_frame, sole_typed_cohort_input

__all__ = [
    "ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS",
    "ADJUSTED_ASSOCIATION_OUTPUT",
    "AdjustedAssociationError",
    "adjusted_association_executor_code",
    "adjusted_association_executor_owns_step",
    "adjusted_association_executor_scaffold",
    "run_adjusted_association_from_env",
]

ADJUSTED_ASSOCIATION_OUTPUT = (
    f"{PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND}:{PLANNED_MODEL_REQUIREMENTS_OUTPUT}"
)
_TABLE_FILENAME = "adjusted_association_estimates.csv"

#: The exact header. The first six are what ``bind_primary_output`` reads; the
#: rest let a reader see what was fitted without opening the plan.
ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS = (
    "fit_status",
    "estimate",
    "ci_low",
    "ci_high",
    "effect_scale",
    "exposure",
    "requirement_id",
    "outcome",
    "covariates",
    "estimator_kind",
    "analysis_set",
    "n",
    "n_events",
    "standard_error",
    "notes",
)


class AdjustedAssociationError(RuntimeError):
    """The declared model could not be fitted as declared."""


def _requirement(step: AnalysisStep) -> Optional[PlannedModelRequirement]:
    requirements = list(step.model_requirements or [])
    return requirements[0] if len(requirements) == 1 else None


def _estimator_kind(requirement: PlannedModelRequirement) -> str:
    family = _normalise_model_contract_token(requirement.method_family)
    if requirement.outcome_type == "binary":
        return (
            "logistic" if family in ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES else ""
        )
    # The continuous families include quantile regression, which fit_estimator
    # does not implement.  Claiming it and fitting OLS instead would answer a
    # different question under the declared method's name.
    return (
        "linear"
        if family in {"linear_regression", "ordinary_least_squares", "ols"}
        else ""
    )


def _effect_scale(kind: str) -> str:
    return "odds_ratio" if kind == "logistic" else "coefficient"


def adjusted_association_executor_owns_step(step: AnalysisStep) -> bool:
    """Own only a single, completely declared adjusted-association model.

    Every clause is a thing the host would otherwise have to decide:

    * exactly one model requirement, because ``bind_primary_output`` binds a
      one-row table; a two-model step is a different product shape, not a
      bigger version of this one;
    * a declared adjustment set, because reconstructing one from ``step.inputs``
      is inference (see the contract's own tests);
    * an estimator this module actually implements -- a quantile-regression
      family fitted as OLS would answer a different question under the declared
      method's name;
    * one typed cohort input at most, so the frame that was analysed is the
      digest-bound one.
    """

    method = _normalise_model_contract_token(
        str(step.method or "").lower().split(" with ", 1)[0]
    )
    if method != PLANNED_MODEL_REQUIREMENTS_STEP_METHOD:
        return False
    if [str(value or "").strip() for value in step.expected_outputs or []] != [
        ADJUSTED_ASSOCIATION_OUTPUT
    ]:
        return False
    requirement = _requirement(step)
    if requirement is None or requirement.covariates is None:
        return False
    if not _estimator_kind(requirement):
        return False
    if sole_typed_cohort_input(step) == "":
        return False
    return not (
        step.table_one_spec is not None
        or step.trajectory_stability_spec is not None
        or step.exposure_outcome_distribution_spec is not None
    )


def adjusted_association_executor_scaffold(
    step: AnalysisStep,
    *,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None,
) -> HostScaffoldedScript:
    """A scaffold with **no agent region at all**, and that is the point.

    Everything this step does is fixed by the plan, so there is nothing for a
    model to write.  An earlier draft put the call in the body and kept the
    declared values in the prologue, which looks safe and is not: a contract
    repair rewriting the body could call the same host function with a
    different exposure or a shorter adjustment set, and the sealed declaration
    above it would sit there unread.  Protecting the *values* is worthless if
    the *call* is editable.

    So the body is empty and every byte is host property.  Any edit at all
    makes ``host_regions_intact`` false, which is the correct answer for a step
    the host computes entirely -- and the state fresh17 step 07 should have
    been in when the host handed its own robustness draft to the model.
    """

    if not adjusted_association_executor_owns_step(step):
        raise ValueError("The step is not owned by the adjusted-association executor")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    requirement = _requirement(step)
    assert requirement is not None
    kind = _estimator_kind(requirement)
    typed_cohort_input = sole_typed_cohort_input(step)
    receipt_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope, frame_name="frame"
        )
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )

    prologue = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.adjusted_association_executor import (
            run_adjusted_association_from_env,
        )
        from easyicu.research_agent.execution.runners.typed_input_binding import (
            load_step_cohort_frame,
        )

        typed_cohort_input = {typed_cohort_input!r}
        declared_model = {{
            "requirement_id": {requirement.requirement_id!r},
            "exposure": {requirement.exposure_source!r},
            "outcome": {requirement.outcome!r},
            "covariates": {list(requirement.covariates)!r},
            "estimator_kind": {kind!r},
            "analysis_set": {requirement.analysis_set!r},
        }}

        frame, cohort_path = load_step_cohort_frame(
            typed_cohort_input=typed_cohort_input,
        )
        """
    ).strip()
    if receipt_code:
        prologue = prologue + "\n\n" + receipt_code.strip()
    # The call is host property too. Sealing the declared values while leaving
    # the call editable would let a repair pass a different exposure to the
    # same function with the sealed declaration sitting unread above it.
    prologue = (
        prologue
        + "\n\n"
        + textwrap.dedent(
            """
        summary = run_adjusted_association_from_env(
            frame=frame,
            cohort_path=cohort_path,
            typed_cohort_input=typed_cohort_input,
            emit_step_summary=False,
            **declared_model,
        )
        """
        ).strip()
    )

    epilogue_lines = [
        'out_dir = Path(os.environ["STEP_OUT_DIR"])',
        "out_dir.mkdir(parents=True, exist_ok=True)",
    ]
    if receipt_code:
        epilogue_lines.append('summary["plausibility_audit"] = plausibility_audit')
    epilogue_lines += [
        '(out_dir / "step_summary.json").write_text(',
        "    json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),",
        '    encoding="utf-8",',
        ")",
        "print(json.dumps(summary, ensure_ascii=False, allow_nan=False))",
    ]

    return HostScaffoldedScript(
        prologue=prologue,
        body="",
        epilogue="\n".join(epilogue_lines),
    )


def adjusted_association_executor_code(
    step: AnalysisStep,
    *,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None,
) -> str:
    """Return the sandbox entrypoint for the exact declared model."""

    return adjusted_association_executor_scaffold(
        step,
        plausibility_scope=plausibility_scope,
    ).assembled()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def run_adjusted_association_from_env(
    *,
    requirement_id: str,
    exposure: str,
    outcome: str,
    covariates: Sequence[str],
    estimator_kind: str,
    analysis_set: str,
    typed_cohort_input: Optional[str] = None,
    frame: Any = None,
    cohort_path: Any = None,
    emit_step_summary: bool = True,
) -> Dict[str, Any]:
    """Fit the declared model and write the one-row estimates table.

    A model that cannot be fitted as declared raises rather than writing a row
    with a null estimate.  A null primary effect is not a weaker result: it is
    an absent one, and a table that carries it still satisfies the step's
    declared output while telling the reader nothing.  The estimator's own note
    (rank-deficient design, no complete rows, non-convergence, too few rows)
    travels in the message so the failure names itself.
    """

    import pandas as pd  # local, so importing this module never requires pandas

    if frame is None:
        frame, cohort_path = load_step_cohort_frame(
            typed_cohort_input=typed_cohort_input
        )
    adjustment = [str(name).strip() for name in covariates or []]
    needed = [exposure, outcome, *adjustment]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise AdjustedAssociationError(
            "declared model column(s) absent from the bound cohort: "
            + ", ".join(sorted(missing))
        )

    model_frame = frame[needed]
    result = fit_estimator(
        cohort=None,
        X=model_frame[[exposure, *adjustment]],
        y=model_frame[outcome],
        kind=estimator_kind,
        term=exposure,
    )
    estimate = _finite(result.point_estimate)
    ci_low = _finite(result.ci_low)
    ci_high = _finite(result.ci_high)
    if not result.converged or estimate is None or ci_low is None or ci_high is None:
        raise AdjustedAssociationError(
            f"declared model {requirement_id!r} could not be fitted as declared: "
            + (result.notes or "no estimate returned")
        )

    outcome_values = pd.to_numeric(model_frame[outcome], errors="coerce")
    n_events = (
        int((outcome_values == 1).sum()) if estimator_kind == "logistic" else None
    )
    row = {
        "fit_status": "fitted",
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "effect_scale": _effect_scale(estimator_kind),
        "exposure": exposure,
        "requirement_id": requirement_id,
        "outcome": outcome,
        "covariates": ";".join(adjustment),
        "estimator_kind": estimator_kind,
        "analysis_set": analysis_set,
        "n": int(result.n),
        "n_events": n_events,
        "standard_error": _finite(result.se),
        "notes": result.notes or "",
    }

    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / _TABLE_FILENAME
    pd.DataFrame([row], columns=list(ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS)).to_csv(
        table_path, index=False
    )

    summary: Dict[str, Any] = {
        "status": "ok",
        "analysis_family": "association",
        "interpretation_class": "adjusted_association",
        "requirement_id": requirement_id,
        "exposure": exposure,
        "outcome": outcome,
        "covariates": adjustment,
        "estimator_kind": estimator_kind,
        "analysis_set": analysis_set,
        "typed_cohort_input": typed_cohort_input,
        "source_cohort": Path(cohort_path).name if cohort_path is not None else None,
        "n_total": int(result.n),
        "n_events": n_events,
        "adjusted_effect": estimate,
        "effect_scale": _effect_scale(estimator_kind),
        "primary_estimate": estimate,
        "primary_estimate_interval": [ci_low, ci_high],
        "output_files": {ADJUSTED_ASSOCIATION_OUTPUT: table_path.name},
    }
    if estimator_kind == "logistic":
        summary["primary_or"] = estimate
        summary["primary_or_ci"] = [ci_low, ci_high]
    if not emit_step_summary:
        return summary
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary

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

from dataclasses import dataclass
import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...contracts.host_scaffold import HostScaffoldedScript
from ...contracts.ownership_verdict import OwnershipVerdict
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
    "ADJUSTED_ASSOCIATION_ANALYSIS_KIND",
    "ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS",
    "ADJUSTED_ASSOCIATION_OUTPUT",
    "AdjustedAssociationError",
    "adjusted_association_executor_code",
    "adjusted_association_executor_owns_step",
    "adjusted_association_executor_scaffold",
    "adjusted_association_executor_verdict",
    "run_adjusted_association_from_env",
]

ADJUSTED_ASSOCIATION_OUTPUT = (
    f"{PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND}:{PLANNED_MODEL_REQUIREMENTS_OUTPUT}"
)

#: The ``analysis_kind`` this owner reports, in selection and in its verdict.
#: One declaration, because a retyped kind literal is how two layers end up
#: disagreeing about which owner produced an artifact (see task #95/N6).
ADJUSTED_ASSOCIATION_ANALYSIS_KIND = PLANNED_MODEL_REQUIREMENTS_OUTPUT
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
    # --- the contrast this row reports -------------------------------------
    # A binary or continuous exposure has one contrast and one row, so these
    # describe it rather than change it: ``exposure_level`` and
    # ``reference_level`` are empty and ``is_primary_contrast`` is true.
    #
    # A CATEGORICAL OR ORDINAL exposure has one row per non-reference level,
    # and that is why these columns exist. A four-level AKI stage is three
    # contrasts, not one number, and collapsing it to a single term reports a
    # per-unit trend under the name of a stage comparison -- a different
    # scientific quantity carrying the declared estimand's label.
    #
    # ``is_primary_contrast`` marks the row the manuscript quotes. With more
    # than two levels that choice is the planner's (highest stage against the
    # reference? each adjacent step?), so the host reads the mark instead of
    # taking a row position, and every consumer -- the primary-output binding,
    # the robustness replay, the figure -- reads the same one.
    "exposure_level",
    "reference_level",
    "contrast",
    "is_primary_contrast",
)

_COEFFICIENT_FILENAME = "adjusted_association_coefficients.csv"

#: The exact header ``PrimaryModelContractValidator`` reads.  Every fitted
#: primary-association model owes a term-level table: the one-row estimates
#: product answers "what is the effect", and this one answers "of what, adjusted
#: for what" -- the question a reader has to be able to check against the plan.
#:
#: ``estimate`` is the effect on ``effect_scale`` -- an odds ratio for a
#: logistic fit.  The validator's own message calls this column
#: ``estimate_or_odds_ratio``, which is a description of the value rather than a
#: name it accepts: its reader takes ``estimate``, ``odds_ratio`` or ``or``.
#: One always-correct, self-describing name beats a header that changes shape
#: with the family.
ADJUSTED_ASSOCIATION_COEFFICIENT_COLUMNS = (
    "model_id",
    "term",
    "term_role",
    "source_variable",
    "estimate",
    "ci_low",
    "ci_high",
    "standard_error",
    "effect_scale",
)

#: The fields ``PrimaryModelContractValidator`` fixes for every model contract.
#: The emitted record also echoes the plan's ``requirement_id``, ``outcome``,
#: ``outcome_type`` and ``method_family``, which is how the validator matches a
#: contract to the requirement it answers.
MODEL_CONTRACT_FIELDS = (
    "model_id",
    "exposure_source",
    "exposure_expression",
    "exposure_role",
    "analysis_role",
    "analysis_set",
    "baseline_missing_policy",
    "n",
    "event_n",
    "fit_status",
    "converged",
    "separation_detected",
    "penalized",
    "fit_method",
)

#: ``fit_estimator`` drops any row with a missing predictor or outcome before
#: fitting, which is exactly this policy under the validator's vocabulary.
_BASELINE_MISSING_POLICY = "drop_missing_baseline"

_FIT_METHODS = {
    "logistic": "statsmodels_logit_maximum_likelihood",
    "linear": "statsmodels_ols",
}


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


def adjusted_association_executor_verdict(step: AnalysisStep) -> OwnershipVerdict:
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

    The clauses are unchanged from when this returned ``bool``; what is new is
    that each one says *which kind* of decline it is.  Measured over 553
    recorded steps, 54 of the 59 declines here were a field the Planner simply
    never filled in -- and a bool sent every one of them to the coder without
    telling anyone.  See :mod:`..contracts.ownership_verdict`.

    Two clauses are deliberately **not** reported as incomplete declarations,
    because more declaring is not what would fix them:

    * a step bundling this product with others is task #105's question of
      whether an owner's claim may depend on Planner bundling at all, and
      calling it "missing" would misname an over-declaration;
    * more than one typed input, or an unimplemented estimator family, are
      contracts this owner does not have.
    """

    method = _normalise_model_contract_token(
        str(step.method or "").lower().split(" with ", 1)[0]
    )
    if method != PLANNED_MODEL_REQUIREMENTS_STEP_METHOD:
        return OwnershipVerdict.wrong_shape(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            reason=f"step method {method!r} is not {PLANNED_MODEL_REQUIREMENTS_STEP_METHOD!r}",
        )
    declared_outputs = [
        str(value or "").strip() for value in step.expected_outputs or []
    ]
    if declared_outputs != [ADJUSTED_ASSOCIATION_OUTPUT]:
        return OwnershipVerdict.wrong_shape(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            reason=(
                f"step declares {len(declared_outputs)} expected output(s), not "
                f"exactly [{ADJUSTED_ASSOCIATION_OUTPUT}]"
            ),
        )
    requirements = list(step.model_requirements or [])
    if not requirements:
        return OwnershipVerdict.incomplete_declaration(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            missing=("model_requirements",),
            reason=(
                "the step declares the primary adjusted-association product but "
                "no model requirement, so the outcome, outcome type, method "
                "family and exposure are undeclared"
            ),
        )
    if len(requirements) != 1:
        return OwnershipVerdict.wrong_shape(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            reason=(
                f"step declares {len(requirements)} model requirements; a "
                "multi-model step is a different product shape"
            ),
        )
    requirement = requirements[0]
    if requirement.covariates is None:
        return OwnershipVerdict.incomplete_declaration(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            missing=("model_requirements[0].covariates",),
            reason=(
                "the model requirement declares no adjustment set, and "
                "reconstructing one from step.inputs would be inference"
            ),
        )
    if not _estimator_kind(requirement):
        return OwnershipVerdict.wrong_shape(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            reason=(
                f"method family {requirement.method_family!r} for outcome type "
                f"{requirement.outcome_type!r} is not an estimator this owner implements"
            ),
        )
    if sole_typed_cohort_input(step) == "":
        return OwnershipVerdict.wrong_shape(
            ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
            reason=(
                "the step declares more than one typed input, or one this "
                "executor family does not support"
            ),
        )
    for spec_name in (
        "table_one_spec",
        "trajectory_stability_spec",
        "exposure_outcome_distribution_spec",
    ):
        if getattr(step, spec_name) is not None:
            return OwnershipVerdict.wrong_shape(
                ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
                reason=f"the step also declares {spec_name}, which another owner claims",
            )
    return OwnershipVerdict.claim(
        ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
        reason="a single, completely declared adjusted-association model",
    )


def adjusted_association_executor_owns_step(step: AnalysisStep) -> bool:
    """Bool view of :func:`adjusted_association_executor_verdict`.

    It delegates rather than re-testing the clauses: two copies of one
    ownership rule drifting apart is the defect shape this package keeps
    paying for.
    """

    return adjusted_association_executor_verdict(step).claimed


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
            "analysis_role": {requirement.analysis_role!r},
            "method_family": {requirement.method_family!r},
            "exposure_levels": {
                list(requirement.exposure_levels)
                if requirement.exposure_levels is not None
                else None
            !r},
            "exposure_reference_level": {requirement.exposure_reference_level!r},
            "primary_contrast_level": {requirement.primary_contrast_level!r},
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


def _coefficient_rows(
    terms: Sequence[Any],
    *,
    model_id: str,
    exposure: str,
    adjustment: Sequence[str],
    effect_scale: str,
    exposure_contrast_columns: Sequence[str] = (),
) -> list[Dict[str, Any]]:
    """Label each fitted coefficient by the role the plan gave its source.

    The role is read off the declaration, never off the term's name: the
    exposure is the exposure because the model requirement says so, and an
    adjustment column is one because it is in the declared adjustment set.  A
    term whose source is in neither would mean the design and the declaration
    disagree, which the caller refuses rather than labels.
    """

    adjustment_set = {str(name) for name in adjustment}
    # A treatment-coded exposure reaches the fit as one indicator per level, so
    # the estimator reports each indicator's own name as its source. The
    # declared contrast columns are the exposure -- the host built them from it
    # -- and saying so here is not name-matching: the set comes from the
    # declaration, so a column the plan did not declare still has no role and
    # is still refused.
    contrast_columns = {str(name) for name in (exposure_contrast_columns or ())}
    rows: list[Dict[str, Any]] = []
    for term in terms:
        source = str(term.source_variable)
        if term.term == "const":
            role = "intercept"
        elif source == exposure or source in contrast_columns:
            role = "exposure"
        elif source in adjustment_set:
            role = "adjustment"
        else:
            raise AdjustedAssociationError(
                f"fitted design term {term.term!r} came from {source!r}, which "
                "is neither the declared exposure nor a declared covariate"
            )
        rows.append(
            {
                "model_id": model_id,
                "term": str(term.term),
                "term_role": role,
                "source_variable": source,
                "estimate": _finite(term.estimate),
                "ci_low": _finite(term.ci_low),
                "ci_high": _finite(term.ci_high),
                "standard_error": _finite(term.se),
                "effect_scale": effect_scale,
            }
        )
    return rows


@dataclass(frozen=True, slots=True)
class _DeclaredContrasts:
    """The level set a categorical exposure was declared with."""

    levels: Tuple[str, ...]
    reference: str
    primary: str

    @property
    def contrast_levels(self) -> Tuple[str, ...]:
        return tuple(level for level in self.levels if level != self.reference)


def _level_key(value: Any) -> str:
    """One spelling for a level, whichever side it arrives from.

    The plan declares levels as strings; the cohort column may hold them as
    floats (a real AKI stage column is ``0.0/1.0/2.0/3.0``). Comparing the two
    raw would make every declared level look absent, so both sides come through
    here. A float that is a whole number keeps its integer spelling, because
    ``"3"`` and ``"3.0"`` are the same stage and a reader should see one of
    them.
    """

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value).strip()


def _declared_contrasts(
    column: Any,
    *,
    exposure: str,
    levels: Optional[Sequence[str]],
    reference: Optional[str],
    primary: Optional[str],
) -> Optional[_DeclaredContrasts]:
    """Return the declared contrasts, or None for a single-term exposure.

    The declaration is checked against the cohort, not trusted: a level the
    plan declared that no stay has cannot be estimated, and a level the cohort
    holds that the plan never declared means the pre-specified level set and
    the analysed population disagree. Either way the fitted model would not be
    the declared one, so both fail closed rather than quietly analysing
    whichever levels happened to be present.
    """

    if levels is None and reference is None and primary is None:
        return None
    if levels is None or reference is None or primary is None:
        raise AdjustedAssociationError(
            "a categorical exposure needs its levels, its reference and its "
            "primary contrast together; the host will not choose which "
            "contrast the manuscript reports"
        )

    declared = tuple(_level_key(level) for level in levels)
    reference_key = _level_key(reference)
    primary_key = _level_key(primary)
    if len(set(declared)) != len(declared) or any(not item for item in declared):
        raise AdjustedAssociationError(
            f"declared exposure levels for {exposure!r} must be unique and non-empty"
        )
    if reference_key not in declared or primary_key not in declared:
        raise AdjustedAssociationError(
            f"the reference and primary contrast for {exposure!r} must both be "
            "declared levels"
        )
    if reference_key == primary_key:
        raise AdjustedAssociationError(
            "the primary contrast must not be the reference level"
        )

    observed = {_level_key(value) for value in column.dropna().unique().tolist()}
    unexpected = sorted(observed - set(declared))
    absent = sorted(set(declared) - observed)
    if unexpected:
        raise AdjustedAssociationError(
            f"the bound cohort holds levels of {exposure!r} the plan never "
            "declared: " + ", ".join(repr(item) for item in unexpected)
        )
    if absent:
        raise AdjustedAssociationError(
            f"the plan declared levels of {exposure!r} no stay has: "
            + ", ".join(repr(item) for item in absent)
        )
    return _DeclaredContrasts(
        levels=declared, reference=reference_key, primary=primary_key
    )


def _contrast_column(exposure: str, level: str) -> str:
    """The design-matrix name for one level's indicator."""

    return f"{exposure}__is_{level}"


def _contrast_design(
    model_frame: Any,
    *,
    exposure: str,
    adjustment: Sequence[str],
    contrasts: _DeclaredContrasts,
) -> Tuple[Any, str]:
    """Treatment-code the exposure against its declared reference.

    One design, one fit: every contrast comes from the same model, so the
    stage-2 and stage-3 estimates are conditional on each other exactly as the
    declared model says. Fitting each level separately would produce numbers
    that no single model ever computed.
    """

    import pandas as pd

    keys = model_frame[exposure].map(_level_key)
    indicators = {
        _contrast_column(exposure, level): (keys == level).astype(float)
        for level in contrasts.contrast_levels
    }
    design = pd.DataFrame(indicators, index=model_frame.index)
    for name in adjustment:
        design[name] = model_frame[name]
    return design, _contrast_column(exposure, contrasts.primary)


def _contrast_rows(
    terms: Sequence[Any],
    *,
    shared: Dict[str, Any],
    exposure: str,
    contrasts: _DeclaredContrasts,
    requirement_id: str,
) -> List[Dict[str, Any]]:
    """One estimates row per declared contrast, in the declared level order.

    Every row comes from the SAME fit, so the contrasts are mutually adjusted
    exactly as the declared model says. A contrast the fit did not return, or
    one it returned without a usable interval, raises rather than being written
    as a null row: an ordinal gradient with a hole in it is not a weaker
    gradient, it is a different one, and a reader comparing stages would not
    see that the missing stage was never estimated.
    """

    by_term = {str(item.term): item for item in terms}
    rows: List[Dict[str, Any]] = []
    for level in contrasts.contrast_levels:
        name = _contrast_column(exposure, level)
        term = by_term.get(name)
        estimate = _finite(getattr(term, "estimate", None)) if term else None
        low = _finite(getattr(term, "ci_low", None)) if term else None
        high = _finite(getattr(term, "ci_high", None)) if term else None
        if estimate is None or low is None or high is None:
            raise AdjustedAssociationError(
                f"declared model {requirement_id!r} returned no usable estimate "
                f"for the contrast {level!r} vs {contrasts.reference!r}; a "
                "gradient missing one of its levels is not the declared model"
            )
        rows.append(
            {
                **shared,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "standard_error": _finite(getattr(term, "se", None)),
                "exposure_level": level,
                "reference_level": contrasts.reference,
                "contrast": f"{level} vs {contrasts.reference}",
                "is_primary_contrast": level == contrasts.primary,
            }
        )
    if sum(1 for row in rows if row["is_primary_contrast"]) != 1:
        raise AdjustedAssociationError(
            "exactly one contrast must carry the primary mark the manuscript " "quotes"
        )
    return rows


def run_adjusted_association_from_env(
    *,
    requirement_id: str,
    exposure: str,
    outcome: str,
    covariates: Sequence[str],
    estimator_kind: str,
    analysis_set: str,
    analysis_role: str,
    method_family: str,
    exposure_levels: Optional[Sequence[str]] = None,
    exposure_reference_level: Optional[str] = None,
    primary_contrast_level: Optional[str] = None,
    typed_cohort_input: Optional[str] = None,
    frame: Any = None,
    cohort_path: Any = None,
    emit_step_summary: bool = True,
) -> Dict[str, Any]:
    """Fit the declared model and write the estimates table.

    One row for a binary or continuous exposure; one row per non-reference
    level when the planner declared a categorical or ordinal one. The declared
    ``primary_contrast_level`` is the row every downstream consumer reads as
    the headline -- see the column block above for why that is declared rather
    than inferred.

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
    contrasts = _declared_contrasts(
        model_frame[exposure],
        exposure=exposure,
        levels=exposure_levels,
        reference=exposure_reference_level,
        primary=primary_contrast_level,
    )
    if contrasts is None:
        design = model_frame[[exposure, *adjustment]]
        focal_term = exposure
    else:
        design, focal_term = _contrast_design(
            model_frame,
            exposure=exposure,
            adjustment=adjustment,
            contrasts=contrasts,
        )
    result = fit_estimator(
        cohort=None,
        X=design,
        y=model_frame[outcome],
        kind=estimator_kind,
        term=focal_term,
    )
    estimate = _finite(result.point_estimate)
    ci_low = _finite(result.ci_low)
    ci_high = _finite(result.ci_high)
    if not result.converged or estimate is None or ci_low is None or ci_high is None:
        raise AdjustedAssociationError(
            f"declared model {requirement_id!r} could not be fitted as declared: "
            + (result.notes or "no estimate returned")
        )

    # From the fit, not from ``model_frame``.  ``result.n`` is the complete-case
    # count, so its numerator has to come from the same rows; counting here
    # counts the rows the estimator dropped as well.  A real run reported
    # n=515 with event_n=102 where those 515 rows held 78 events -- the
    # analysis set's denominator with the whole cohort's numerator, a 19.8%
    # event rate reported for a 15.1% one.  The host's own primary-model
    # contract recomputes both from the bound cohort and refused the step, so
    # the study's primary estimate was computed, was correct, and was thrown
    # away over a count this function had no business deriving.
    n_events = result.n_events
    if estimator_kind == "logistic" and n_events is None:
        raise AdjustedAssociationError(
            f"declared model {requirement_id!r} fitted a binary outcome without "
            "reporting the events among the rows it used; refusing to report a "
            "denominator without its numerator"
        )
    shared = {
        "fit_status": "fitted",
        "effect_scale": _effect_scale(estimator_kind),
        "exposure": exposure,
        "requirement_id": requirement_id,
        "outcome": outcome,
        "covariates": ";".join(adjustment),
        "estimator_kind": estimator_kind,
        "analysis_set": analysis_set,
        "n": int(result.n),
        "n_events": n_events,
        "notes": result.notes or "",
    }
    if contrasts is None:
        rows = [
            {
                **shared,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "standard_error": _finite(result.se),
                "exposure_level": "",
                "reference_level": "",
                "contrast": "",
                "is_primary_contrast": True,
            }
        ]
    else:
        rows = _contrast_rows(
            result.terms,
            shared=shared,
            exposure=exposure,
            contrasts=contrasts,
            requirement_id=requirement_id,
        )

    coefficient_rows = _coefficient_rows(
        result.terms,
        model_id=requirement_id,
        exposure=exposure,
        adjustment=adjustment,
        effect_scale=_effect_scale(estimator_kind),
        exposure_contrast_columns=(
            ()
            if contrasts is None
            else [
                _contrast_column(exposure, level) for level in contrasts.contrast_levels
            ]
        ),
    )
    exposure_terms = [
        item for item in coefficient_rows if item["term_role"] == "exposure"
    ]
    # One fitted term per contrast the plan declared -- one for a binary or
    # continuous exposure, and one per non-reference level for a categorical
    # one. The count is checked against the DECLARATION rather than fixed at
    # one, because a four-level exposure legitimately fits three, while a
    # single-term model fitting two still means the design and the plan
    # disagree.
    expected_terms = 1 if contrasts is None else len(contrasts.contrast_levels)
    if len(exposure_terms) != expected_terms:
        raise AdjustedAssociationError(
            f"declared model {requirement_id!r} fitted {len(exposure_terms)} "
            f"terms for exposure {exposure!r}; the declaration calls for "
            f"{expected_terms}"
        )

    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / _TABLE_FILENAME
    pd.DataFrame(rows, columns=list(ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS)).to_csv(
        table_path, index=False
    )
    coefficient_path = out_dir / _COEFFICIENT_FILENAME
    pd.DataFrame(
        coefficient_rows, columns=list(ADJUSTED_ASSOCIATION_COEFFICIENT_COLUMNS)
    ).to_csv(coefficient_path, index=False)

    model_contract: Dict[str, Any] = {
        "model_id": requirement_id,
        # The plan's roster is keyed by requirement_id, so a contract without
        # one is a model nobody asked for and a requirement nobody answered.
        "requirement_id": requirement_id,
        "outcome": outcome,
        # Derived from the estimator that actually ran, not copied from the
        # plan: a logistic fit only reaches here after refusing anything but a
        # binary 0/1 outcome, so this reports what was fitted.
        "outcome_type": "binary" if estimator_kind == "logistic" else "continuous",
        # Passed through, because several declared families map to one
        # implemented estimator.  ``_estimator_kind`` already refused to claim
        # the step unless this family is one of them, so the echo is checked.
        "method_family": method_family,
        "exposure_source": exposure,
        "exposure_expression": exposure_terms[0]["term"],
        "exposure_role": "primary" if analysis_role == "primary" else "secondary",
        "analysis_role": analysis_role,
        "analysis_set": analysis_set,
        "baseline_missing_policy": _BASELINE_MISSING_POLICY,
        "n": int(result.n),
        "event_n": n_events,
        "fit_status": "fitted",
        # Reaching here means the fit converged with a finite estimate and a
        # finite interval; a separated design cannot satisfy both, and the
        # branches above return rather than reporting one that does not.
        "converged": True,
        "separation_detected": False,
        "penalized": False,
        "fit_method": _FIT_METHODS[estimator_kind],
    }

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
        "analysis_role": analysis_role,
        "model_contracts": [model_contract],
        "adjustment_covariates": list(adjustment),
        "coefficient_table": coefficient_path.name,
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

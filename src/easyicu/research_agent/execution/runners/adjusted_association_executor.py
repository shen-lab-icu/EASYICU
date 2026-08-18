"""Deterministic owner for a fully declared adjusted-association model.

``table:adjusted_association_estimates`` is the most frequently declared product
no deterministic owner could emit -- 233 of 1812 recorded real steps, 175 of
them declaring exactly one model requirement -- and it is the paper's primary
result.  Until now the LLM coder wrote it every time, and the accumulated
repair guidance in ``plan_utils`` records how that went: a script that dropped
the whole cohort by numeric-coercing ``sex`` before dummy encoding, object-dtype
design matrices handed to statsmodels, and contracts "satisfied" with a null
estimate.

Nothing here decides science.  The planner fixes the outcome, exact estimator,
exposure, adjustment set, and the coding/levels/reference/transform of every
model term.  This executor claims the step only when every one of those is
declared, and computes exactly what was declared.

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
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...authority.declared_levels import execution_model_requirement
from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...contracts.association_execution import (
    association_estimator_kind,
    association_execution_verdict,
)
from ...contracts.dependence import (
    PatientGroupResolutionError,
    PlannedDependenceRequirement,
    resolve_patient_groups,
)
from ...contracts.host_scaffold import HostScaffoldedScript
from ...contracts.model_terms import (
    ModelTermSpec,
    serialise_model_terms,
    validate_model_term_roster,
)
from ...contracts.model_tokens import (
    ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
    ASSOCIATION_LOGIT_ESTIMATOR,
    ASSOCIATION_OLS_ESTIMATOR,
    canonical_association_method,
)
from ...contracts.ownership_verdict import OwnershipVerdict
from ..model_matrix import ModelTermCompilationError, compile_model_terms
from ...robustness.estimators import fit_estimator
from ...schema import (
    PLANNED_MODEL_REQUIREMENTS_OUTPUT,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
    AnalysisStep,
    PlannedModelRequirement,
)
from ...numeric_scalars import coerce_optional_finite_float as _finite
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
    # --- which model these rows came from -----------------------------------
    # ``model_id`` is the identity this step's own model contract publishes and
    # the first column of its sibling coefficient table.  A reader that has
    # only the product name ``adjusted_association_estimates`` cannot tell
    # whether these rows are the primary, a secondary or a sensitivity
    # estimate; the figure-lineage reader answers that by matching this column
    # against the parent's ``model_contracts``, and with no such column it
    # inherits no estimand tier and a primary result figure fails its own
    # effect obligation.  ``requirement_id`` answers a different question --
    # which planned requirement these rows satisfy -- and the contract
    # publishes both for the same reason.
    "model_id",
    "requirement_id",
    "outcome",
    "covariates",
    "estimator_kind",
    # Which row the study designated primary. `MODEL_CONTRACT_FIELDS` has
    # carried this all along and this executor writes it into the step summary's
    # model contract; the TABLE did not carry it, and the figure step that
    # consumes `table:adjusted_association_estimates` declares its role-column
    # contract on the table, not on the contract.
    #
    # MEASURED (e2 lactate, 9 of 11 steps): step
    # 06_lactate_mortality_association_figure was refused with
    # `artifact_consumption_contract_invalid: role column 'analysis_role' is
    # absent from the verified schema`. Step 05 produced the table through this
    # executor and reported ok; the value sat in the requirement object the
    # executor was already holding, and it writes that same value into the
    # sibling table a hundred lines below. `analysis_set` is a different field
    # -- which population -- not a spelling of this one, so nothing covered.
    "analysis_role",
    "analysis_set",
    "n",
    "n_events",
    "standard_error",
    "variance_estimator",
    "cluster_count",
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
    "logistic": ASSOCIATION_LOGIT_ESTIMATOR,
    "linear": ASSOCIATION_OLS_ESTIMATOR,
}


class AdjustedAssociationError(RuntimeError):
    """The declared model could not be fitted as declared."""


def _requirement(step: AnalysisStep) -> Optional[PlannedModelRequirement]:
    requirements = list(step.model_requirements or [])
    return requirements[0] if len(requirements) == 1 else None


def _estimator_kind(requirement: PlannedModelRequirement) -> str:
    """Delegate to the shared claim boundary.

    This rule used to live here alone, which is how the capability registry and
    plan validation could answer "deterministic" for a GLM-binomial contract
    this owner declines.  ``contracts.association_execution`` is now the single
    statement of what is implemented; see its module docstring.
    """

    return association_estimator_kind(requirement)


def _effect_scale(kind: str) -> str:
    return "odds_ratio" if kind == "logistic" else "coefficient"


def adjusted_association_executor_verdict(step: AnalysisStep) -> OwnershipVerdict:
    """Delegate to the shared claim boundary in ``contracts``.

    The clauses moved to :func:`...contracts.association_execution
    .association_execution_verdict` unchanged.  They had to move because plan
    validation, the capability registry and readiness all need this same
    answer, and while it lived in an execution runner each of them re-derived
    an approximation of it -- which is how a GLM-binomial contract was labelled
    a deterministic host capability and then executed by the LLM coder.
    """

    return association_execution_verdict(step)


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
    # The declaration as it must EXECUTE.  A categorical level set arrives in
    # the host's own opaque placeholders whenever the Planner was told the
    # column's cardinality and not its values, and only the host can put the
    # levels back.  Reading ``requirement`` directly here is what handed the
    # sandbox four placeholders and a cohort holding 0/1/2/3.
    requirement = execution_model_requirement(step, requirement)
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
            "model_terms": {serialise_model_terms(requirement.model_terms or ())!r},
            "primary_contrast_level": {requirement.primary_contrast_level!r},
            "dependence": {requirement.dependence.model_dump(mode="json") if requirement.dependence is not None else None!r},
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
            # A treatment-coded indicator IS the exposure, encoded -- and the
            # primary-model contract reads ``source_variable`` as "the unique
            # original authoritative cohort column", allowing ``term`` itself
            # to be "an encoded or transformed design column".
            # ``<exposure>__is_<level>`` is a design column the host built and
            # no cohort carries, so reporting it as the source made every
            # contrast unresolvable: canary13's step 08 was computed correctly
            # by this owner and then refused by that contract, three issues for
            # three stages, over a name rather than a number.
            source = exposure
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


def _contrast_column(exposure: str, level: str) -> str:
    """The design-matrix name for one level's indicator."""

    return f"{exposure}__is_{level}"


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
            "exactly one contrast must carry the primary mark the manuscript quotes"
        )
    return rows


def _cluster_groups(
    *,
    frame: Any,
    dependence: PlannedDependenceRequirement | None,
) -> tuple[Any, str]:
    """Return runtime groups from the exact plan declaration.

    The derivation is intentionally closed.  In particular, the executor does
    not search for id-like columns or parse a naming convention that the plan
    did not carry.  Malformed composite identities fail before any model output
    is written.
    """

    if dependence is None:
        return None, "model_based"
    source = dependence.group_source
    if source not in frame.columns:
        raise AdjustedAssociationError(
            f"declared cluster group source {source!r} is absent from the bound cohort"
        )
    series = frame[source]
    if bool(series.isna().any()):
        raise AdjustedAssociationError(
            f"declared cluster group source {source!r} contains missing values"
        )
    try:
        resolved = resolve_patient_groups(
            series.astype("object").tolist(),
            requirement=dependence,
        )
    except PatientGroupResolutionError as exc:
        raise AdjustedAssociationError(str(exc)) from exc
    groups = series.astype("object").copy()
    groups.iloc[:] = list(resolved.groups)
    return groups, dependence.variance_estimator


def run_adjusted_association_from_env(
    *,
    requirement_id: str,
    exposure: str,
    outcome: str,
    covariates: Sequence[str],
    model_terms: Sequence[ModelTermSpec | Dict[str, Any]],
    estimator_kind: str,
    analysis_set: str,
    analysis_role: str,
    method_family: str,
    primary_contrast_level: Optional[str] = None,
    dependence: PlannedDependenceRequirement | Dict[str, Any] | None = None,
    typed_cohort_input: Optional[str] = None,
    frame: Any = None,
    cohort_path: Any = None,
    emit_step_summary: bool = True,
    output_dir: Any = None,
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
    canonical_method = canonical_association_method(method_family)
    expected_method = {
        "logistic": ASSOCIATION_LOGIT_ESTIMATOR,
        "linear": ASSOCIATION_OLS_ESTIMATOR,
    }.get(estimator_kind)
    if expected_method is None or canonical_method != expected_method:
        raise AdjustedAssociationError(
            f"declared estimator {canonical_method!r} does not exactly dispatch "
            f"to runtime kind {estimator_kind!r}"
        )
    try:
        parsed_dependence = (
            dependence
            if isinstance(dependence, PlannedDependenceRequirement)
            else PlannedDependenceRequirement.model_validate(dependence)
            if dependence is not None
            else None
        )
    except ValueError as exc:
        raise AdjustedAssociationError(
            "declared dependence contract is invalid: " + str(exc)
        ) from exc
    parsed_terms = [
        item if isinstance(item, ModelTermSpec) else ModelTermSpec.model_validate(item)
        for item in model_terms
    ]
    try:
        exposure_term, adjustment_terms = validate_model_term_roster(
            terms=parsed_terms,
            exposure=exposure,
            covariates=covariates,
        )
    except ValueError as exc:
        raise AdjustedAssociationError(
            "declared model term roster is invalid: " + str(exc)
        ) from exc
    adjustment = [item.name for item in adjustment_terms]
    needed = [outcome, *[item.name for item in parsed_terms]]
    if parsed_dependence is not None:
        needed.append(parsed_dependence.group_source)
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise AdjustedAssociationError(
            "declared model column(s) absent from the bound cohort: "
            + ", ".join(sorted(missing))
        )

    model_frame = frame[list(dict.fromkeys(needed))]
    cluster_groups, variance_estimator = _cluster_groups(
        frame=model_frame,
        dependence=parsed_dependence,
    )
    try:
        compiled = compile_model_terms(
            model_frame,
            terms=parsed_terms,
            exposure=exposure,
        )
    except ModelTermCompilationError as exc:
        raise AdjustedAssociationError(str(exc)) from exc
    contrasts: Optional[_DeclaredContrasts] = None
    if exposure_term.transform == "treatment_contrast":
        contrast_levels = exposure_term.contrast_levels
        if exposure_term.coding == "binary":
            primary = contrast_levels[0]
        else:
            primary = str(primary_contrast_level or "").strip()
            if primary not in contrast_levels:
                raise AdjustedAssociationError(
                    "the categorical exposure primary_contrast_level must be one "
                    "of its declared non-reference levels"
                )
        contrasts = _DeclaredContrasts(
            levels=tuple(exposure_term.levels or ()),
            reference=str(exposure_term.reference_level),
            primary=primary,
        )
        focal_term = _contrast_column(exposure, primary)
    else:
        focal_term = compiled.exposure_columns[0]
    result = fit_estimator(
        cohort=None,
        X=compiled.design,
        y=model_frame[outcome],
        kind=estimator_kind,
        term=focal_term,
        source_by_design_column=compiled.source_by_design_column,
        variance_estimator=variance_estimator,
        cluster_groups=cluster_groups,
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
        # The same expression the model contract below publishes, so the rows
        # and the contract can never name the model differently.
        "model_id": requirement_id,
        "requirement_id": requirement_id,
        "outcome": outcome,
        "covariates": ";".join(adjustment),
        "estimator_kind": estimator_kind,
        "analysis_role": analysis_role,
        "analysis_set": analysis_set,
        "n": int(result.n),
        "n_events": n_events,
        "variance_estimator": result.variance_estimator,
        "cluster_count": result.cluster_count,
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
            compiled.exposure_columns if contrasts is not None else ()
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
    expected_terms = len(compiled.exposure_columns)
    if len(exposure_terms) != expected_terms:
        raise AdjustedAssociationError(
            f"declared model {requirement_id!r} fitted {len(exposure_terms)} "
            f"terms for exposure {exposure!r}; the declaration calls for "
            f"{expected_terms}"
        )

    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(os.environ["STEP_OUT_DIR"])
    )
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
        # Passed through after exact-token resolution. ``_estimator_kind``
        # already refused every method family except the canonical estimator
        # (or one of its reviewed aliases), so this echo names the fit that ran.
        "method_family": method_family,
        "exposure_source": exposure,
        # The term the manuscript quotes, not whichever contrast was fitted
        # first. With one exposure term the two coincide; with a declared
        # gradient they do not, and every downstream reader of this contract
        # is asking "which single coefficient IS the primary result". The
        # estimates table already marks that row `is_primary_contrast`, and
        # the loop that built it refuses unless exactly one carries the mark,
        # so `contrasts.primary` is present and unambiguous here.
        "exposure_expression": (
            exposure_terms[0]["term"]
            if contrasts is None
            else _contrast_column(exposure, contrasts.primary)
        ),
        "exposure_role": "primary" if analysis_role == "primary" else "secondary",
        "analysis_role": analysis_role,
        "analysis_set": analysis_set,
        "baseline_missing_policy": _BASELINE_MISSING_POLICY,
        "n": int(result.n),
        "event_n": n_events,
        "fit_status": "fitted",
        "converged": True,
        # ASSERTED AFTER CHECKING, NOT BECAUSE THE FIT RETURNED NUMBERS.
        #
        # This used to be a literal False, justified by "a separated design
        # cannot satisfy both a finite estimate and a finite interval". That is
        # not true: quasi-separation routinely returns an enormous coefficient,
        # an interval spanning orders of magnitude, and converged=True. The
        # figure renderer beside this already had a test for exactly that state
        # (estimate 2.9e7, interval 1e-8 to 8.4e22), so one layer knew it
        # existed while the producer asserted it could not.
        #
        # The contract's own validator refuses a missing value, so an answer is
        # obligatory -- which is the reason it has to be computed. The fit now
        # reports it; anything that is not a logistic fit has no separation to
        # report and keeps the field False rather than inventing a verdict.
        "separation_detected": bool(result.separation_detected),
        "penalized": False,
        "fit_method": _FIT_METHODS[estimator_kind],
        "model_terms": serialise_model_terms(parsed_terms),
        "design_columns": list(compiled.design.columns),
        "dependence": (
            parsed_dependence.model_dump(mode="json")
            if parsed_dependence is not None
            else None
        ),
        "cluster_count": result.cluster_count,
    }

    summary: Dict[str, Any] = {
        "status": "ok",
        "analysis_family": "association",
        "interpretation_class": "adjusted_association",
        "requirement_id": requirement_id,
        "exposure": exposure,
        "outcome": outcome,
        "covariates": adjustment,
        "model_terms": serialise_model_terms(parsed_terms),
        "design_columns": list(compiled.design.columns),
        "estimator_kind": estimator_kind,
        "analysis_set": analysis_set,
        "variance_estimator": result.variance_estimator,
        "cluster_count": result.cluster_count,
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

"""Deterministic owner for closed binary-association sensitivity variants.

Two prespecified sensitivity strategies that previously went to the Coder are
executed by the host: the **first-stay restriction** (refit the primary model
on first ICU stays only) and the **covariate functional-form check** (refit
with one continuous covariate expressed as a restricted cubic spline and test
its nonlinearity). Both inherit the parent's exact model declaration --
exposure, outcome, covariates, dependence, primary contrast -- from the plan's
host-owned adjusted-association step; they restrict or re-express only what
the typed step names, and write the ``analysis_rows`` result the
``association_freeform_v1`` contract already validates. The owner never
chooses a variable: a step whose restriction column or spline target cannot be
derived unambiguously from typed coordinates is not owned and stays on the
agent-coded path exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import json
import math
from pathlib import Path
import textwrap
from typing import Any, Literal, Mapping, Optional

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...contracts.association_execution import (
    ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT,
    association_binary_sensitivity_contract,
    association_binary_sensitivity_plan_verdict,
    sole_primary_model_requirement,
)
from ...contracts.cohort_product_keys import is_closed_cohort_product_key
from ...contracts.dependence import (
    PatientGroupResolutionError,
    PlannedDependenceRequirement,
    resolve_patient_groups,
)
from ...contracts.model_terms import ModelTermSpec
from ...numeric_scalars import coerce_optional_finite_float
from ...schema import AnalysisPlan, AnalysisStep
from .plausibility_receipt import render_standard_plausibility_receipt_code

_finite = functools.partial(coerce_optional_finite_float, allow_bool=False)

ASSOCIATION_BINARY_SENSITIVITY_ANALYSIS_KIND = "association_binary_sensitivity"
FIRST_STAY_SENSITIVITY_METHODS = frozenset(
    {"first_stay_association", "one_stay_per_patient_association"}
)
FUNCTIONAL_FORM_SENSITIVITY_METHODS = frozenset({"restricted_cubic_spline_sensitivity"})
_TRUE_SPELLINGS = frozenset({"1", "1.0", "true", "t", "yes", "y"})
_FALSE_SPELLINGS = frozenset({"0", "0.0", "false", "f", "no", "n"})


class AssociationBinarySensitivityError(RuntimeError):
    """The closed sensitivity variant could not be executed as declared."""


@dataclass(frozen=True, slots=True)
class BinarySensitivityVariant:
    """One host-executable variant resolved from typed plan coordinates only."""

    strategy: Literal["first_stay", "functional_form"]
    spec_id: str
    parent_step_id: str
    output_product: str
    restriction_column: Optional[str] = None
    target_column: Optional[str] = None
    knot_quantiles: tuple[float, ...] = ()


def _method_head(method: str) -> str:
    return str(method or "").strip().lower().split(" with ", 1)[0].strip()


def bound_cohort_input(step: AnalysisStep) -> Optional[str]:
    """The one closed cohort product the step reads beside its parent table.

    A sensitivity child legitimately declares two typed inputs -- the digest-
    bound cohort and the parent estimates table -- so the generic sole-input
    rule cannot answer here. Anything other than exactly one closed cohort
    product beside the parent product is not owned.
    """

    typed = [
        value
        for value in (str(item or "").strip() for item in step.inputs)
        if ":" in value and value != ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT
    ]
    if len(typed) != 1 or not is_closed_cohort_product_key(typed[0]):
        return None
    return typed[0]


def association_binary_sensitivity_consumed_input_keys(
    step: AnalysisStep,
) -> tuple[str, ...]:
    """The typed inputs the refit reads: its bound cohort and the parent table.

    The host seals a receipt for exactly these keys.  The generic sole-cohort
    rule answers "none" for a step that declares two typed inputs, which left
    the cohort without a receipt and the integrity gate refusing every
    host-owned refit whose plan bound it to a typed cohort.
    """

    cohort = bound_cohort_input(step)
    return (
        *((cohort,) if cohort is not None else ()),
        ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT,
    )


def _parent_step(step: AnalysisStep, plan: AnalysisPlan) -> Optional[AnalysisStep]:
    producers = [
        candidate
        for candidate in plan.steps
        if ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT
        in {str(value or "").strip() for value in candidate.expected_outputs}
    ]
    return producers[0] if len(producers) == 1 else None


def resolve_binary_sensitivity_variant(
    step: AnalysisStep, *, plan: AnalysisPlan | None
) -> Optional[BinarySensitivityVariant]:
    """Return the variant the host can execute, or ``None`` to leave the Coder path.

    Every clause reads a typed coordinate. The first-stay restriction column is
    the one raw input the step declares beyond its parent model's own columns;
    the functional-form target is the ``functional_form_spec`` column, which
    must be a continuous covariate of the parent model. Anything ambiguous is
    not owned.
    """

    contract = association_binary_sensitivity_contract(step)
    if contract is None or len(contract.sensitivity_ids) != 1 or plan is None:
        return None
    verdict = association_binary_sensitivity_plan_verdict(step, plan_steps=plan.steps)
    if not verdict.claimed:
        return None
    if bound_cohort_input(step) is None:
        return None
    parent = _parent_step(step, plan)
    requirement = sole_primary_model_requirement(parent) if parent is not None else None
    if parent is None or requirement is None or requirement.covariates is None:
        return None
    spec_id = contract.sensitivity_ids[0]
    head = _method_head(step.method)
    model_columns = {
        str(requirement.exposure_source),
        str(requirement.outcome),
        *(str(value) for value in requirement.covariates),
        *(str(term.name) for term in (requirement.model_terms or ())),
    }
    if requirement.dependence is not None:
        model_columns.add(str(requirement.dependence.group_source))
    if head in FIRST_STAY_SENSITIVITY_METHODS:
        if step.functional_form_spec is not None:
            return None
        extra = [
            value
            for value in (str(item or "").strip() for item in step.inputs)
            if value and ":" not in value and value not in model_columns
        ]
        if len(extra) != 1:
            return None
        return BinarySensitivityVariant(
            strategy="first_stay",
            spec_id=spec_id,
            parent_step_id=parent.step_id,
            output_product=contract.output_product,
            restriction_column=extra[0],
        )
    if head in FUNCTIONAL_FORM_SENSITIVITY_METHODS:
        spec = step.functional_form_spec
        if spec is None or spec.comparison != "restricted_cubic_spline_vs_linear":
            return None
        target = str(spec.target_column)
        continuous = {
            str(term.name)
            for term in (requirement.model_terms or ())
            if term.role == "covariate" and term.coding == "continuous"
        }
        if target not in continuous or target not in set(map(str, requirement.covariates)):
            return None
        return BinarySensitivityVariant(
            strategy="functional_form",
            spec_id=spec_id,
            parent_step_id=parent.step_id,
            output_product=contract.output_product,
            target_column=target,
            knot_quantiles=tuple(float(value) for value in spec.knot_quantiles),
        )
    return None


def association_binary_sensitivity_executor_owns_step(
    step: AnalysisStep, *, plan: AnalysisPlan | None
) -> bool:
    return resolve_binary_sensitivity_variant(step, plan=plan) is not None


def association_binary_sensitivity_executor_code(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    variant = resolve_binary_sensitivity_variant(step, plan=plan)
    if variant is None:
        raise ValueError("step is not owned by the binary sensitivity executor")
    parent = _parent_step(step, plan)
    assert parent is not None
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    receipt_code = (
        render_standard_plausibility_receipt_code(plausibility_scope, frame_name="frame")
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    prologue = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.association_binary_sensitivity_executor import (
            run_association_binary_sensitivity,
        )
        from easyicu.research_agent.execution.runners.typed_input_binding import (
            load_step_cohort_frame,
        )

        frame, cohort_path = load_step_cohort_frame(
            typed_cohort_input={bound_cohort_input(step)!r},
        )
        """
    ).strip()
    execution = textwrap.dedent(
        f"""
        summary = run_association_binary_sensitivity(
            frame=frame,
            cohort_path=cohort_path,
            step={step.model_dump(mode="json")!r},
            parent_step={parent.model_dump(mode="json")!r},
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
        )
        {"summary['plausibility_audit'] = plausibility_audit" if receipt_code else ""}
        out_dir = Path(os.environ["STEP_OUT_DIR"])
        (out_dir / "step_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        """
    ).strip()
    return "\n\n".join(block for block in (prologue, receipt_code.strip(), execution) if block)


def _cluster_groups(frame: Any, dependence: PlannedDependenceRequirement | None):
    import pandas as pd

    if dependence is None:
        return None, "model_based"
    source = dependence.group_source
    if source not in frame.columns or bool(frame[source].isna().any()):
        raise AssociationBinarySensitivityError(
            f"declared cluster group source {source!r} is absent or incomplete"
        )
    try:
        resolved = resolve_patient_groups(
            frame[source].astype("object").tolist(), requirement=dependence
        )
    except PatientGroupResolutionError as exc:
        raise AssociationBinarySensitivityError(str(exc)) from exc
    return pd.Series(list(resolved.groups), index=frame.index, dtype="object"), dependence.variance_estimator


def _first_stay_mask(series: Any):
    """Exact truthy spellings only; a missing flag is not evidence of a first stay."""

    import pandas as pd

    spelled = series.astype("object").where(series.notna(), None).map(
        lambda value: "" if value is None else str(value).strip().lower()
    )
    unknown = sorted(set(spelled) - _TRUE_SPELLINGS - _FALSE_SPELLINGS - {""})
    if unknown:
        raise AssociationBinarySensitivityError(
            "first-stay restriction column is not a 0/1 indicator; unknown spellings: "
            + ", ".join(unknown[:6])
        )
    return pd.Series(spelled.isin(_TRUE_SPELLINGS).to_numpy(), index=series.index)


def _primary_row(estimates: Any, *, requirement_id: str) -> dict[str, Any]:
    primary = estimates.loc[estimates["is_primary_contrast"].astype(bool)]
    if len(primary) != 1:
        raise AssociationBinarySensitivityError(
            f"model {requirement_id!r} did not mark exactly one primary contrast"
        )
    return dict(primary.iloc[0])


def _refit_first_stay(
    *,
    frame: Any,
    requirement: Any,
    variant: BinarySensitivityVariant,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .adjusted_association_executor import (
        AdjustedAssociationError,
        run_adjusted_association_from_env,
    )
    import pandas as pd

    column = str(variant.restriction_column)
    if column not in frame.columns:
        raise AssociationBinarySensitivityError(
            f"first-stay restriction column {column!r} is absent from the bound cohort"
        )
    mask = _first_stay_mask(frame[column])
    restricted = frame.loc[mask].copy()
    if restricted.empty:
        raise AssociationBinarySensitivityError(
            f"no rows satisfy the first-stay restriction {column!r}"
        )
    try:
        summary = run_adjusted_association_from_env(
            requirement_id=f"{requirement.requirement_id}__{variant.spec_id}",
            exposure=requirement.exposure_source,
            outcome=requirement.outcome,
            covariates=list(requirement.covariates or ()),
            model_terms=[term.model_dump(mode="json") for term in (requirement.model_terms or ())],
            estimator_kind="logistic",
            analysis_set=requirement.analysis_set,
            analysis_role="sensitivity",
            method_family=requirement.method_family,
            primary_contrast_level=requirement.primary_contrast_level,
            dependence=requirement.dependence,
            typed_cohort_input=None,
            frame=restricted,
            cohort_path=None,
            emit_step_summary=False,
            output_dir=out_dir / "kernel",
        )
    except AdjustedAssociationError as exc:
        raise AssociationBinarySensitivityError(str(exc)) from exc
    estimates = pd.read_csv(out_dir / "kernel" / "adjusted_association_estimates.csv")
    primary = _primary_row(estimates, requirement_id=requirement.requirement_id)
    row = {
        "analysis_id": variant.spec_id,
        "strategy": "first_stay",
        "restriction": f"{column} in {{1, true}}",
        "n_rows_offered": int(len(frame)),
        "n_restricted_rows": int(len(restricted)),
        "n_stays": int(summary["n_total"]),
        "n_deaths": int(summary["n_events"]),
        "odds_ratio": _finite(primary["estimate"]),
        "ci_low": _finite(primary["ci_low"]),
        "ci_high": _finite(primary["ci_high"]),
        "effect_measure": "odds_ratio",
        "exposure_level": str(primary.get("exposure_level", "")),
        "reference_level": str(primary.get("reference_level", "")),
        "variance_estimator": summary.get("variance_estimator"),
        "cluster_count": summary.get("cluster_count"),
        "fit_status": "fitted",
    }
    receipt = {
        "restriction_column": column,
        "n_rows_offered": int(len(frame)),
        "n_restricted_rows": int(len(restricted)),
        "model_contract": summary["model_contracts"][0],
    }
    return row, receipt


def _refit_functional_form(
    *,
    frame: Any,
    requirement: Any,
    variant: BinarySensitivityVariant,
    linear_reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import pandas as pd

    from ...methods.rcs_dose_response import (
        RCSError,
        nonlinearity_wald_test,
        rcs_basis,
        rcs_fit,
    )
    from ...robustness.estimators import fit_estimator
    from ..model_matrix import ModelTermCompilationError, compile_model_terms

    target = str(variant.target_column)
    terms = [
        term if isinstance(term, ModelTermSpec) else ModelTermSpec.model_validate(term)
        for term in (requirement.model_terms or ())
    ]
    needed = [
        requirement.outcome,
        *(term.name for term in terms),
        *((requirement.dependence.group_source,) if requirement.dependence is not None else ()),
    ]
    missing = sorted(set(needed) - set(frame.columns))
    if missing:
        raise AssociationBinarySensitivityError(
            "declared model column(s) absent from the bound cohort: " + ", ".join(missing)
        )
    complete = frame.dropna(subset=list(dict.fromkeys(needed)))
    if complete.empty:
        raise AssociationBinarySensitivityError("no complete rows for the functional-form refit")
    try:
        compiled = compile_model_terms(
            complete,
            terms=[term for term in terms if term.name != target],
            exposure=requirement.exposure_source,
        )
    except ModelTermCompilationError as exc:
        raise AssociationBinarySensitivityError(str(exc)) from exc
    values = pd.to_numeric(complete[target], errors="coerce")
    if values.isna().any():
        raise AssociationBinarySensitivityError(
            f"functional-form target {target!r} is not numeric on every complete row"
        )
    knots = np.quantile(values.to_numpy(dtype=float), np.asarray(variant.knot_quantiles))
    try:
        basis = rcs_basis(values.to_numpy(dtype=float), knots=knots)
    except (RCSError, ValueError) as exc:
        raise AssociationBinarySensitivityError(f"{type(exc).__name__}: {exc}") from exc
    spline = pd.DataFrame(
        np.asarray(basis.matrix, dtype=float),
        index=complete.index,
        columns=[target if name == "x" else f"{target}__rcs_{name}" for name in basis.column_names],
    )
    design = pd.concat([compiled.design, spline], axis=1)
    source_by_column = dict(compiled.source_by_design_column)
    for name in spline.columns:
        source_by_column[name] = target
    groups, variance = _cluster_groups(complete, requirement.dependence)
    exposure_term = next(term for term in terms if term.role == "exposure")
    if exposure_term.transform == "treatment_contrast":
        if exposure_term.coding == "binary":
            focal_level = exposure_term.contrast_levels[0]
        else:
            focal_level = str(requirement.primary_contrast_level or "")
        focal = f"{requirement.exposure_source}__is_{focal_level}"
    else:
        focal = compiled.exposure_columns[0]
    if focal not in design.columns:
        raise AssociationBinarySensitivityError(
            f"primary contrast column {focal!r} is absent from the spline design"
        )
    outcome = pd.to_numeric(complete[requirement.outcome], errors="coerce")
    result = fit_estimator(
        cohort=None,
        X=design,
        y=outcome,
        kind="logistic",
        term=focal,
        source_by_design_column=source_by_column,
        variance_estimator=variance,
        cluster_groups=groups,
    )
    estimate = _finite(result.point_estimate)
    ci_low = _finite(result.ci_low)
    ci_high = _finite(result.ci_high)
    if not result.converged or estimate is None or ci_low is None or ci_high is None:
        raise AssociationBinarySensitivityError(
            "spline refit did not converge: " + (result.notes or "no estimate returned")
        )
    if result.n_events is None:
        raise AssociationBinarySensitivityError(
            "spline refit reported no event count for the rows it used"
        )
    linear_or = _finite(linear_reference.get("estimate"))
    row: dict[str, Any] = {
        "analysis_id": variant.spec_id,
        "strategy": "functional_form",
        "covariate": target,
        "basis": "restricted_cubic_spline",
        "knot_quantiles": "|".join(f"{value:.6g}" for value in variant.knot_quantiles),
        "knots": "|".join(f"{float(value):.6g}" for value in basis.knots),
        "n_stays": int(result.n),
        "n_deaths": int(result.n_events),
        "odds_ratio": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "effect_measure": "odds_ratio",
        "primary_or_linear_covariate": linear_or,
        "log_or_delta_vs_linear": (
            math.log(estimate) - math.log(linear_or)
            if linear_or is not None and linear_or > 0 and estimate > 0
            else None
        ),
        "nonlinearity_wald_statistic": None,
        "nonlinearity_df": None,
        "nonlinearity_p_value": None,
        "nonlinearity_covariance": "model_based_without_clustering",
        "variance_estimator": result.variance_estimator,
        "cluster_count": result.cluster_count,
        "fit_status": "fitted",
        "note": "",
    }
    try:
        spline_fit = rcs_fit(
            outcome.to_numpy(dtype=float),
            basis,
            covariates=compiled.design.to_numpy(dtype=float),
            covariate_names=list(compiled.design.columns),
            family="binomial",
        )
        wald = nonlinearity_wald_test(spline_fit)
        row.update(
            {
                "nonlinearity_wald_statistic": _finite(wald.statistic),
                "nonlinearity_df": int(wald.df),
                "nonlinearity_p_value": _finite(wald.p_value),
            }
        )
    except (RCSError, ValueError) as exc:
        row["note"] = f"nonlinearity test not estimable: {exc}"
    receipt = {
        "target_column": target,
        "knots": [float(value) for value in basis.knots],
        "design_columns": list(design.columns),
        "n_complete_rows": int(len(complete)),
    }
    return row, receipt


def run_association_binary_sensitivity(
    *,
    frame: Any,
    cohort_path: Any,
    step: AnalysisStep | Mapping[str, Any],
    parent_step: AnalysisStep | Mapping[str, Any],
    run_dir: Path,
    resolved_inputs: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Execute one closed sensitivity variant and write its typed result table."""

    import pandas as pd

    from .typed_input_binding import load_typed_input

    parsed_step = step if isinstance(step, AnalysisStep) else AnalysisStep.model_validate(step)
    parsed_parent = (
        parent_step
        if isinstance(parent_step, AnalysisStep)
        else AnalysisStep.model_validate(parent_step)
    )
    try:
        plan = AnalysisPlan(
            research_question="binary sensitivity variant",
            steps=[parsed_parent, parsed_step],
        )
    except ValueError as exc:
        raise AssociationBinarySensitivityError(
            "the step is not one host-executable binary sensitivity variant: " + str(exc)[:300]
        ) from exc
    variant = resolve_binary_sensitivity_variant(parsed_step, plan=plan)
    if variant is None or variant.parent_step_id != parsed_parent.step_id:
        raise AssociationBinarySensitivityError(
            "the step is not one host-executable binary sensitivity variant"
        )
    requirement = sole_primary_model_requirement(parsed_parent)
    assert requirement is not None
    manifest = json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    parent_binding = load_typed_input(
        input_key=ASSOCIATION_BINARY_SENSITIVITY_PARENT_PRODUCT,
        run_dir=Path(run_dir),
        resolved_inputs=manifest,
        step_id=parsed_step.step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        minimum_row_count=1,
        require_consumption_contract=True,
    )
    linear_reference = _primary_row(
        parent_binding.frame, requirement_id=requirement.requirement_id
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if variant.strategy == "first_stay":
        row, receipt = _refit_first_stay(
            frame=frame, requirement=requirement, variant=variant, out_dir=out_dir
        )
    else:
        row, receipt = _refit_functional_form(
            frame=frame,
            requirement=requirement,
            variant=variant,
            linear_reference=linear_reference,
        )
    product_name = variant.output_product.partition(":")[2]
    table_path = out_dir / f"{product_name}.csv"
    pd.DataFrame([row]).to_csv(table_path, index=False)
    return {
        "status": "ok",
        "deterministic_standard_analysis": ASSOCIATION_BINARY_SENSITIVITY_ANALYSIS_KIND,
        "analysis_family": "association",
        "interpretation_class": "prespecified_sensitivity",
        "sensitivity_strategy": variant.strategy,
        "sensitivity_spec_id": variant.spec_id,
        "parent_step_id": variant.parent_step_id,
        "parent_requirement_id": requirement.requirement_id,
        "typed_cohort_input": bound_cohort_input(parsed_step),
        "source_cohort": Path(cohort_path).name if cohort_path is not None else None,
        "analysis_rows": [row],
        "primary_reference": {
            "odds_ratio": _finite(linear_reference.get("estimate")),
            "ci_low": _finite(linear_reference.get("ci_low")),
            "ci_high": _finite(linear_reference.get("ci_high")),
            "n": int(linear_reference.get("n")) if _finite(linear_reference.get("n")) is not None else None,
            "evidence_id": parent_binding.evidence_id,
            "sha256": parent_binding.sha256,
        },
        "sensitivity_runtime_receipt": {
            "schema_version": "easyicu.association_binary_sensitivity_runtime_receipt/1",
            "strategy": variant.strategy,
            **receipt,
        },
        "output_files": {variant.output_product: table_path.name},
    }


__all__ = [
    "ASSOCIATION_BINARY_SENSITIVITY_ANALYSIS_KIND",
    "AssociationBinarySensitivityError",
    "BinarySensitivityVariant",
    "association_binary_sensitivity_consumed_input_keys",
    "association_binary_sensitivity_executor_code",
    "association_binary_sensitivity_executor_owns_step",
    "bound_cohort_input",
    "resolve_binary_sensitivity_variant",
    "run_association_binary_sensitivity",
]

"""Deterministic owner for a fully declared primary Cox analysis.

This owner chooses no scientific coordinate.  It claims only a primary step
whose Planner contract fixes the bound cohort, exposure encoding, adjustment
set, event code, time unit/horizon, censoring, missingness, estimator,
uncertainty and PH diagnostic.  The generated sandbox script has no model-
editable region.  Its receipt binds the input bytes, canonical analysis frame,
result table and PH diagnostic table by SHA-256.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...contracts.family_primary import FamilyPrimaryResultRequirement
from ...contracts.host_scaffold import HostScaffoldedScript
from ...contracts.model_terms import (
    DataPreparationPlan,
    ModelTermSpec,
    PreparationRefusal,
    preparation_for_step,
    require_matching_survival_policy,
    serialise_model_terms,
    validate_model_term_roster,
)
from ...contracts.model_tokens import SURVIVAL_COX_ESTIMATOR, SURVIVAL_PH_DIAGNOSTIC
from ...contracts.ownership_verdict import OwnershipVerdict
from ...contracts.survival import (
    CIF_PRODUCT,
    FINE_GRAY_PRODUCT,
    GRAY_PRODUCT,
    LOGRANK_PRODUCT,
    SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
    SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
    SURVIVAL_PRIMARY_OWNER,
    SurvivalAnalysisReceipt,
    canonical_survival_applied_filter,
    canonical_survival_formula,
    validate_competing_event,
)
from ...contracts.survival_execution import (
    SURVIVAL_PRIMARY_ANALYSIS_KIND,
    survival_execution_verdict,
)
from ...methods.missing_data import multiple_impute, rubin_pool
from ...methods.ph_schoenfeld import ph_test
from ...schema import AnalysisStep
from ..model_matrix import (
    ModelTermCompilationError,
    compile_model_terms,
    compile_prepared_terms,
    prepare_analysis_frame,
    write_preparation_receipt,
)
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import read_frame, sha256_file


def logrank_by_group(durations, events, groups) -> Any:
    """Use lifelines for inference; accumulate expected events at tied risk sets.

    Follow-up and administrative censoring must already be closed by the owner.
    At each event time, expected group events are d(t) Y_g(t) / Y(t).
    All subjects censored at that time remain in its risk set.
    """
    import numpy as np
    import pandas as pd
    from lifelines.statistics import logrank_test, multivariate_logrank_test

    t, e, g = np.asarray(durations, float), np.asarray(events, float), np.asarray(groups)
    if any(v.ndim != 1 for v in (t, e, g)) or not (len(t) == len(e) == len(g)):
        raise ValueError("log-rank requires equally sized one-dimensional vectors")
    if not len(t) or not np.isfinite(t).all() or (t < 0).any():
        raise ValueError("log-rank requires finite nonnegative follow-up")
    if not np.isin(e, [0, 1]).all() or not e.sum() or pd.isna(g).any():
        raise ValueError("log-rank requires binary events, observed events and complete groups")
    labels = sorted(set(g), key=repr)
    if len(labels) < 2:
        raise ValueError("log-rank requires at least two observed groups")
    if len(labels) == 2:
        a, b = g == labels[0], g == labels[1]
        test = logrank_test(t[a], t[b], event_observed_A=e[a], event_observed_B=e[b])
        method = "lifelines_logrank_test"
    else:
        test = multivariate_logrank_test(t, g, e)
        method = "lifelines_multivariate_logrank_test"
    if not np.isfinite([test.test_statistic, test.p_value]).all():
        raise ValueError("log-rank returned nonfinite inference")
    exits = pd.crosstab(t, g).reindex(columns=labels, fill_value=0).sort_index()
    risk = exits.iloc[::-1].cumsum().iloc[::-1]
    failures = pd.crosstab(t[e == 1], g[e == 1]).reindex(
        index=exits.index, columns=labels, fill_value=0,
    )
    expected = risk.mul(failures.sum(axis=1) / risk.sum(axis=1), axis=0).sum()
    return pd.DataFrame({
        "group": labels, "n": exits.sum().to_numpy(),
        "observed_events": failures.sum().to_numpy(),
        "expected_events": expected.to_numpy(),
        "statistic": float(test.test_statistic), "df": len(labels) - 1,
        "p_value": float(test.p_value), "method": method,
    })


CAUSE_SPECIFIC_QUESTION = "Instantaneous event rate among subjects still event-free."
SUBDISTRIBUTION_QUESTION = "Association with cumulative incidence in the presence of the competing event."


@dataclass(frozen=True)
class SurvivalGroupProducts:
    tables: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


def survival_group_products(
    analysis: Any, *, time_column: str, event_column: str,
    groups: Any, design_columns: list[str], competing: bool, discrete: bool,
) -> SurvivalGroupProducts:
    """Never choose a grouping cutpoint or reinterpret competing events as censoring."""
    import pandas as pd

    if not discrete:
        if competing:
            raise ValueError("competing-risk group comparisons require declared discrete exposure levels")
        return SurvivalGroupProducts(details={"logrank_status": "not_applicable_continuous_exposure"})
    time, events = analysis[time_column], analysis[event_column]
    if not competing:
        return SurvivalGroupProducts({LOGRANK_PRODUCT: logrank_by_group(time, events, groups)})
    from ...methods.competing_risks import estimate_cif, fit_fine_gray
    from ...methods.gray_test import gray_test

    curves = []
    for group in sorted(groups.unique(), key=repr):
        mask = groups.eq(group)
        if events[mask].eq(1).any():
            cif = estimate_cif(time[mask], events[mask], event_of_interest=1, random_state=0)
            knots, values = cif.times, cif.cif
        else:
            # No observed target event in this group: its empirical CIF is zero.
            # The shared kernel deliberately refuses that special case.
            knots = sorted({0.0, *time[mask].tolist()})
            values = [0.0] * len(knots)
        curves.append(pd.DataFrame({
            "group": group, "time": knots, "cumulative_incidence": values,
            "n": int(mask.sum()), "n_events": int(events[mask].eq(1).sum()),
            "n_competing": int(events[mask].eq(2).sum()), "method": "aalen_johansen_lifelines",
        }))
    gray = gray_test(time, events, groups)
    fg = fit_fine_gray(analysis, duration_col=time_column, event_col=event_column,
                       covariates=design_columns)
    estimates = fg.estimates.assign(
        estimand="subdistribution_hazard_ratio", estimand_question=SUBDISTRIBUTION_QUESTION,
        n_analysis_rows=len(analysis), n_events=int(events.eq(1).sum()),
    )
    return SurvivalGroupProducts(
        tables={CIF_PRODUCT: pd.concat(curves, ignore_index=True),
                GRAY_PRODUCT: pd.DataFrame([{
                    "statistic": gray.statistic, "df": gray.df, "p_value": gray.p_value,
                    "n": gray.n, "n_events": gray.n_interest, "n_competing": gray.n_competing,
                    "method": gray.method, "claim_ceiling": gray.claim_ceiling,
                    "censoring_weights": "pooled",
                }]), FINE_GRAY_PRODUCT: estimates},
        details={"fine_gray": fg.receipt, "gray_claim_ceiling": gray.claim_ceiling,
                 "logrank_status": "replaced_by_gray_for_competing_events",
                 "cause_specific_question": CAUSE_SPECIFIC_QUESTION,
                 "subdistribution_question": SUBDISTRIBUTION_QUESTION},
    )

_RESULT_FILENAME = "survival_effect_estimates.csv"
_PH_FILENAME = "survival_ph_diagnostic.csv"
_RECEIPT_FILENAME = "survival_analysis_receipt.json"


class SurvivalPrimaryExecutionError(RuntimeError):
    """The exact declared survival model could not be executed reportably."""

    owner = SURVIVAL_PRIMARY_OWNER
    phase = "survival_primary_execution"

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {message}")


@dataclass(frozen=True, slots=True)
class _CoxEstimate:
    hazard_ratio: float
    ci_low: float
    ci_high: float
    standard_error: float
    coefficient: float
    p_value: float


def _requirement(step: AnalysisStep) -> Optional[FamilyPrimaryResultRequirement]:
    requirement = step.family_primary_result_requirement
    return (
        requirement
        if requirement is not None and requirement.analysis_family == "survival"
        else None
    )


def survival_primary_executor_verdict(step: AnalysisStep) -> OwnershipVerdict:
    """Claim only the exact host-supported survival contract."""

    return survival_execution_verdict(
        requirement=_requirement(step),
        planned_analysis_role=step.planned_analysis_role,
        expected_outputs=step.expected_outputs,
        inputs=step.inputs,
    )


def survival_primary_executor_owns_step(step: AnalysisStep) -> bool:
    return survival_primary_executor_verdict(step).claimed


def survival_primary_executor_scaffold(
    step: AnalysisStep,
    *,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None,
) -> HostScaffoldedScript:
    """Render a fully sealed script; the Coder owns no byte of this analysis."""

    verdict = survival_primary_executor_verdict(step)
    if not verdict.claimed:
        raise ValueError(verdict.reason)
    require_matching_survival_policy(step)
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    requirement = _requirement(step)
    assert requirement is not None
    typed_input = str(requirement.input_product)
    receipt_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope, frame_name="bound.frame"
        )
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    declared = {
        "result_product": requirement.expected_result_product,
        "exposure": requirement.exposure_source,
        "outcome": requirement.outcome,
        "effect_scale": requirement.effect_scale,
        "population": requirement.population,
        "time_origin": requirement.time_origin,
        "time_column": requirement.time_column,
        "time_unit": requirement.time_unit,
        "event_column": requirement.event_column,
        "event_value": requirement.event_value,
        "event_definition": requirement.event_definition,
        "censoring_strategy": requirement.censoring_strategy,
        "competing_risk_strategy": requirement.competing_risk_strategy,
        "competing_event_value": requirement.competing_event_value,
        "competing_event_definition": requirement.competing_event_definition,
        "time_horizon": requirement.time_horizon,
        "time_horizon_value": requirement.time_horizon_value,
        "estimator": requirement.estimator,
        "effect_measure": requirement.effect_measure,
        "covariates": list(requirement.covariates or ()),
        "model_terms": serialise_model_terms(requirement.model_terms or ()),
        "ph_diagnostic": requirement.proportional_hazards_diagnostic,
        "ph_alpha": requirement.proportional_hazards_alpha,
        "ph_policy": requirement.proportional_hazards_policy,
        "data_preparation": preparation_for_step(step).model_dump(mode="json") if preparation_for_step(step) else None,
    }
    prologue = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.survival_primary_executor import (
            run_survival_primary,
        )
        from easyicu.research_agent.execution.runners.typed_input_binding import (
            load_typed_input,
            run_dir_from_env,
        )

        typed_cohort_input = {typed_input!r}
        bound = load_typed_input(
            input_key=typed_cohort_input,
            run_dir=run_dir_from_env(),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]).resolve(),
            expected_evidence_kind="table",
            exclusive=True,
        )
        declared_survival = {declared!r}
        """
    ).strip()
    if receipt_code:
        prologue += "\n\n" + receipt_code.strip()
    prologue += (
        "\n\n"
        + textwrap.dedent(
            """
        summary = run_survival_primary(
            input_path=bound.path,
            input_product=bound.input_key,
            input_evidence_id=bound.evidence_id,
            input_sha256=bound.sha256,
            emit_step_summary=False,
            **declared_survival,
        )
        """
        ).strip()
    )
    epilogue = [
        'out_dir = Path(os.environ["STEP_OUT_DIR"])',
        "out_dir.mkdir(parents=True, exist_ok=True)",
    ]
    if receipt_code:
        epilogue.append('summary["plausibility_audit"] = plausibility_audit')
    epilogue.extend(
        [
            '(out_dir / "step_summary.json").write_text(',
            "    json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),",
            '    encoding="utf-8",',
            ")",
            "print(json.dumps(summary, ensure_ascii=False, allow_nan=False))",
        ]
    )
    return HostScaffoldedScript(
        prologue=prologue,
        body="",
        epilogue="\n".join(epilogue),
    )


def survival_primary_executor_code(
    step: AnalysisStep,
    *,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None,
) -> str:
    return survival_primary_executor_scaffold(
        step, plausibility_scope=plausibility_scope
    ).assembled()


def _fit_declared_cox(
    frame: Any,
    *,
    duration_column: str,
    event_column: str,
    exposure: str,
) -> _CoxEstimate:
    try:
        from lifelines import CoxPHFitter
    except Exception as exc:  # pragma: no cover - reference image owns dependency
        raise SurvivalPrimaryExecutionError(
            "survival_dependency_missing",
            "The approved runner image does not provide lifelines",
        ) from exc

    fitter = CoxPHFitter()
    try:
        fitter.fit(frame, duration_col=duration_column, event_col=event_column)
        row = fitter.summary.loc[exposure]
        estimate = _CoxEstimate(
            hazard_ratio=float(row["exp(coef)"]),
            ci_low=float(row["exp(coef) lower 95%"]),
            ci_high=float(row["exp(coef) upper 95%"]),
            standard_error=float(row["se(coef)"]),
            coefficient=float(row["coef"]),
            p_value=float(row["p"]),
        )
    except Exception as exc:
        raise SurvivalPrimaryExecutionError(
            "survival_cox_fit_failed",
            "The declared Cox model could not be fitted",
        ) from exc
    if not all(
        math.isfinite(value)
        for value in (
            estimate.hazard_ratio,
            estimate.ci_low,
            estimate.ci_high,
            estimate.standard_error,
            estimate.coefficient,
            estimate.p_value,
        )
    ):
        raise SurvivalPrimaryExecutionError(
            "survival_cox_result_nonfinite",
            "The Cox fit returned a non-finite result",
        )
    if (
        estimate.hazard_ratio <= 0
        or estimate.ci_low <= 0
        or estimate.ci_high <= 0
        or not estimate.ci_low <= estimate.hazard_ratio <= estimate.ci_high
    ):
        raise SurvivalPrimaryExecutionError(
            "survival_cox_interval_invalid",
            "The Cox fit returned an invalid interval",
        )
    return estimate


def _canonical_frame_sha256(frame: Any) -> str:
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package in ("easyicu", "lifelines", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise SurvivalPrimaryExecutionError(
                "survival_package_version_unresolved",
                f"Cannot bind the installed {package!r} version",
            ) from exc
    return versions


def run_survival_primary(
    *,
    input_path: Path,
    input_product: str,
    input_evidence_id: str,
    input_sha256: str,
    result_product: str,
    exposure: str,
    outcome: str,
    effect_scale: str,
    population: str,
    time_origin: str,
    time_column: str,
    time_unit: str,
    event_column: str,
    event_value: int,
    event_definition: str,
    censoring_strategy: str,
    competing_risk_strategy: str,
    time_horizon: str,
    time_horizon_value: float,
    estimator: str,
    effect_measure: str,
    covariates: Sequence[str],
    model_terms: Sequence[ModelTermSpec | Dict[str, Any]],
    ph_diagnostic: str,
    ph_alpha: float,
    ph_policy: str,
    competing_event_value: int | None = None,
    competing_event_definition: str | None = None,
    emit_step_summary: bool = True,
    data_preparation: DataPreparationPlan | Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Fit, diagnose and receipt-bind the exact Planner-declared Cox model."""

    import pandas as pd

    try:
        competing = validate_competing_event(
            competing_risk_strategy, event_value,
            competing_event_value, competing_event_definition,
        )
    except ValueError as exc:
        raise SurvivalPrimaryExecutionError("survival_competing_declaration_invalid", str(exc)) from exc
    input_path = Path(input_path)
    if sha256_file(input_path) != input_sha256:
        raise SurvivalPrimaryExecutionError(
            "survival_input_digest_mismatch_before_fit",
            "Bound cohort digest changed before fit",
        )
    frame = read_frame(input_path)
    preparation = DataPreparationPlan.model_validate(data_preparation) if data_preparation is not None else None
    preparation_execution, imputed, imputation_diagnostics, missing_sensitivity = {}, None, None, None
    if preparation is not None:
        frame, preparation_execution = prepare_analysis_frame(frame, preparation, run_dir=os.environ.get("EASYICU_RUN_DIR"))
    if not input_evidence_id:
        raise SurvivalPrimaryExecutionError(
            "survival_input_evidence_missing",
            "Bound cohort has no evidence identity",
        )
    if estimator != SURVIVAL_COX_ESTIMATOR:
        raise SurvivalPrimaryExecutionError(
            "survival_estimator_not_exact",
            f"The sealed owner implements only {SURVIVAL_COX_ESTIMATOR!r}",
        )
    if ph_diagnostic != SURVIVAL_PH_DIAGNOSTIC:
        raise SurvivalPrimaryExecutionError(
            "survival_ph_diagnostic_not_exact",
            f"The sealed owner implements only {SURVIVAL_PH_DIAGNOSTIC!r}",
        )
    if ph_policy not in {
        "report_only",
        "block_paper_authorization",
    }:
        raise SurvivalPrimaryExecutionError(
            "survival_ph_policy_unsupported",
            "The declared PH handling policy is unsupported",
        )
    if not math.isfinite(float(ph_alpha)) or not 0 < float(ph_alpha) < 1:
        raise SurvivalPrimaryExecutionError(
            "survival_ph_alpha_invalid",
            "The declared PH alpha must be finite and strictly between zero and one",
        )
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
        raise SurvivalPrimaryExecutionError(
            "survival_model_term_roster_invalid", str(exc)
        ) from exc
    if exposure_term.coding == "categorical":
        raise SurvivalPrimaryExecutionError(
            "survival_categorical_exposure_shape_unsupported",
            "The v1 primary result cannot report multiple exposure contrasts",
        )
    adjustment = [term.name for term in adjustment_terms]
    source_terms = [term.name for term in parsed_terms]
    needed = [time_column, event_column, *source_terms]
    if preparation is not None and preparation.missing_data is not None:
        needed.extend(preparation.missing_data.auxiliaries)
    if len(needed) != len(set(needed)):
        raise SurvivalPrimaryExecutionError(
            "survival_columns_not_distinct",
            "Time, event, exposure and covariate columns must be distinct",
        )
    missing = sorted(set(needed) - set(frame.columns))
    if missing:
        raise SurvivalPrimaryExecutionError(
            "survival_declared_columns_missing",
            "Declared survival columns are absent from the bound cohort: "
            + ", ".join(missing),
        )
    source = frame.loc[:, needed].copy()
    n_source_rows = len(source)
    for column in (time_column, event_column):
        original = source[column]
        numeric = pd.to_numeric(original, errors="coerce")
        conversion_loss = original.notna() & numeric.isna()
        if conversion_loss.any():
            raise SurvivalPrimaryExecutionError(
                "survival_numeric_conversion_loss",
                f"Declared numeric column {column!r} contains non-numeric values",
            )
        source[column] = numeric
    source_before_imputation = source.copy()
    if preparation is not None and preparation.missing_data is not None:
        try:
            imputed = impute_survival_source(source, plan=preparation, time_column=time_column, event_column=event_column,
                                             event_value=event_value, horizon=time_horizon_value, exposure=exposure, competing=competing)
        except PreparationRefusal as exc:
            if preparation.missing_data.strategy == "multiple_imputation":
                raise
            missing_sensitivity = {"strategy": "multiple_imputation", "status": "not_estimable", "reason_code": exc.reason_code}
        if imputed is not None and preparation.missing_data.strategy == "multiple_imputation":
            source = imputed.datasets[0]
    try:
        compiled = compile_model_terms(
            source,
            terms=parsed_terms,
            exposure=exposure,
        )
        if preparation is not None:
            from ..model_matrix import compile_prepared_terms

            compiled = compile_prepared_terms(source, plan=preparation, terms=parsed_terms, exposure=exposure, outcome=event_column)
    except ModelTermCompilationError as exc:
        raise SurvivalPrimaryExecutionError(exc.reason_code, str(exc)) from exc
    if len(compiled.exposure_columns) != 1:
        raise SurvivalPrimaryExecutionError(
            "survival_exposure_design_shape_invalid",
            "The v1 primary survival result requires one exposure coefficient",
        )
    exposure_design_column = compiled.exposure_columns[0]
    analysis = source[[time_column, event_column]].join(compiled.design)
    analysis = analysis.dropna(subset=list(analysis.columns)).copy()
    if analysis.empty:
        raise SurvivalPrimaryExecutionError(
            "survival_complete_cases_empty",
            "No complete rows remain for Cox analysis",
        )
    if not all(
        bool(pd.Series(analysis[column], copy=False).map(math.isfinite).all())
        for column in analysis.columns
    ):
        raise SurvivalPrimaryExecutionError(
            "survival_analysis_value_nonfinite",
            "The complete-case analysis frame contains non-finite values",
        )
    if not (analysis[time_column] > 0).all():
        raise SurvivalPrimaryExecutionError(
            "survival_follow_up_nonpositive",
            "Follow-up time must be strictly positive",
        )
    event_codes = analysis[event_column]
    if not (event_codes == event_codes.round()).all():
        raise SurvivalPrimaryExecutionError(
            "survival_event_code_nonintegral",
            "Event codes must be integral",
        )
    observed_codes = set(event_codes.astype(int).unique().tolist())
    allowed_codes = {0, int(event_value)} | ({competing_event_value} if competing else set())
    if not observed_codes.issubset(allowed_codes):
        raise SurvivalPrimaryExecutionError(
            "survival_event_code_unexpected",
            "Observed event codes do not match the declared endpoint",
        )
    if analysis[exposure_design_column].nunique(dropna=True) < 2:
        raise SurvivalPrimaryExecutionError(
            "survival_exposure_contrast_absent",
            "The exposure has no estimable contrast",
        )

    censored_at_horizon = analysis[time_column] > float(time_horizon_value)
    analysis[time_column] = analysis[time_column].clip(upper=float(time_horizon_value))
    analysis[event_column] = (
        event_codes.eq(int(event_value)) & ~censored_at_horizon
    ).astype(int)
    competing_analysis = analysis.copy()
    if competing:
        competing_rows = event_codes.eq(competing_event_value) & ~censored_at_horizon
        if not competing_rows.any():
            raise SurvivalPrimaryExecutionError(
                "survival_competing_event_absent",
                "No declared competing event remains in the complete-case horizon cohort",
            )
        competing_analysis.loc[competing_rows, event_column] = 2
    n_events = int(analysis[event_column].sum())
    if n_events < 2:
        raise SurvivalPrimaryExecutionError(
            "survival_event_support_insufficient",
            "Fewer than two declared events remain after administrative censoring",
        )
    if sha256_file(input_path) != input_sha256:
        raise SurvivalPrimaryExecutionError(
            "survival_input_digest_mismatch_during_setup",
            "Bound cohort digest changed during fit setup",
        )

    formula = canonical_survival_formula(
        time_column=time_column,
        event_column=event_column,
        event_value=event_value,
        exposure_source=exposure,
        covariates=adjustment,
        design_columns=list(compiled.design.columns),
    )
    analysis_frame_sha256 = _canonical_frame_sha256(analysis)
    estimate = _fit_declared_cox(
        analysis,
        duration_column=time_column,
        event_column=event_column,
        exposure=exposure_design_column,
    )
    try:
        ph_table = ph_test(
            analysis,
            duration_col=time_column,
            event_col=event_column,
            covariates=list(compiled.design.columns),
            time_transform="km",
        )
    except Exception as exc:
        raise SurvivalPrimaryExecutionError(
            "survival_ph_diagnostic_failed",
            "The declared Schoenfeld PH diagnostic could not be executed",
        ) from exc
    if imputed is not None:
        estimate, ph_table, imputation_diagnostics, missing_sensitivity = pool_survival_imputations(
            source=source_before_imputation, imputed=imputed, plan=preparation, terms=parsed_terms, exposure=exposure,
            time_column=time_column, event_column=event_column, event_value=event_value, horizon=time_horizon_value,
            fit_cox=_fit_declared_cox, ph_test=ph_test, primary_estimate=estimate, primary_ph=ph_table,
        )
    global_rows = ph_table.loc[ph_table["covariate"].astype(str) == "global"]
    if len(global_rows) != 1:
        raise SurvivalPrimaryExecutionError(
            "survival_ph_global_result_invalid",
            "The PH diagnostic did not return exactly one global result",
        )
    global_p = float(global_rows["p_value"].iloc[0])
    if not math.isfinite(global_p) or not 0 <= global_p <= 1:
        raise SurvivalPrimaryExecutionError(
            "survival_ph_global_p_invalid",
            "The PH diagnostic returned an invalid global p value",
        )
    # The exposure's own PH row. The global row is Bonferroni -- min(1, k *
    # min_j p_j) -- so it cannot see a time-varying exposure effect once the
    # model carries a handful of covariates: an exposure at p=0.01 leaves the
    # global at 0.05 with 5 covariates and 0.08 with 8, both "not rejected" at
    # alpha=0.05 while the single coefficient the manuscript reports is the one
    # violating the assumption the estimate depends on.
    #
    # The contract already refuses a categorical exposure, so the exposure
    # generates exactly one design column and this lookup is exact rather than
    # a choice among contrasts.
    exposure_rows = ph_table.loc[
        ph_table["covariate"].astype(str) == str(exposure_design_column)
    ]
    if len(exposure_rows) != 1:
        raise SurvivalPrimaryExecutionError(
            "survival_ph_exposure_result_missing",
            "The PH diagnostic did not return exactly one row for the primary "
            "exposure design column",
        )
    exposure_p = float(exposure_rows["p_value"].iloc[0])
    if not math.isfinite(exposure_p) or not 0 <= exposure_p <= 1:
        raise SurvivalPrimaryExecutionError(
            "survival_ph_exposure_p_invalid",
            "The PH diagnostic returned an invalid exposure p value",
        )
    ph_violation = global_p < float(ph_alpha) or exposure_p < float(ph_alpha)
    if not ph_violation:
        ph_status = "not_rejected"
    elif ph_policy == "report_only":
        ph_status = "violation_report_only"
    else:
        ph_status = "violation_block_paper_authorization"
    # A rejected assumption is never self-authorizing, whatever the plan asked
    # for: the policy was chosen before the diagnostic ran.
    paper_authorization_allowed = not ph_violation
    ph_table = ph_table.copy()
    ph_table["declared_alpha"] = float(ph_alpha)
    ph_table["handling_policy"] = ph_policy
    ph_table["ph_decision_rule"] = "exposure_or_global"
    ph_table["ph_status"] = ph_status
    ph_table["paper_authorization_allowed"] = paper_authorization_allowed
    ph_table["estimand"] = "cause_specific_hazard_ratio" if competing else "hazard_ratio"

    try:
        supplements = survival_group_products(
            competing_analysis, time_column=time_column, event_column=event_column,
            groups=source.loc[analysis.index, exposure].astype(str),
            design_columns=list(compiled.design.columns), competing=competing,
            discrete=exposure_term.coding != "continuous",
        )
    except ValueError as exc:
        raise SurvivalPrimaryExecutionError(
            getattr(exc, "code", "survival_group_comparison_failed"), str(exc),
        ) from exc
    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / _RESULT_FILENAME
    result_row = {
        "exposure_source": exposure,
        "outcome": outcome,
        "effect_scale": effect_scale,
        "hazard_ratio": estimate.hazard_ratio,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "standard_error": estimate.standard_error,
        "coefficient": estimate.coefficient,
        "p_value": estimate.p_value,
        "n_analysis_rows": len(analysis),
        "n_events": n_events,
        "formula": formula,
        "covariates": ";".join(adjustment),
        "exposure_design_column": exposure_design_column,
        "estimand": "cause_specific_hazard_ratio" if competing else "hazard_ratio",
        "estimand_question": CAUSE_SPECIFIC_QUESTION,
        "competing_event_definition": competing_event_definition,
    }
    if competing:
        fine_gray = supplements.tables[FINE_GRAY_PRODUCT].set_index("term").loc[exposure_design_column]
        result_row.update({
            "subdistribution_hazard_ratio": float(fine_gray.hazard_ratio),
            "subdistribution_ci_low": float(fine_gray.ci_low),
            "subdistribution_ci_high": float(fine_gray.ci_high),
            "subdistribution_p_value": float(fine_gray.p_value),
            "subdistribution_question": SUBDISTRIBUTION_QUESTION,
        })
    pd.DataFrame([result_row]).to_csv(result_path, index=False)
    result_sha256 = sha256_file(result_path)
    ph_path = out_dir / _PH_FILENAME
    ph_table.to_csv(ph_path, index=False)
    ph_sha256 = sha256_file(ph_path)
    supplementary_files, supplementary_sha256 = {}, {}
    for product, table in supplements.tables.items():
        path = out_dir / (product.partition(":")[2] + ".csv")
        table.to_csv(path, index=False)
        supplementary_files[product] = path.name
        supplementary_sha256[product] = sha256_file(path)
    package_versions = _package_versions()
    if competing:
        package_versions.update({
            "R": supplements.details["fine_gray"]["r_version"],
            "R:survival": supplements.details["fine_gray"]["survival_version"],
        })
    applied_filter = canonical_survival_applied_filter(
        time_column=time_column,
        event_column=event_column,
        event_value=event_value,
        exposure_source=exposure,
        covariates=adjustment,
        model_terms=parsed_terms,
        time_horizon_value=time_horizon_value,
        time_unit=time_unit,
        missing_data_plan=preparation.missing_data if preparation else None,
    )
    receipt = SurvivalAnalysisReceipt(
        issuer=SURVIVAL_PRIMARY_OWNER,
        execution_mode="deterministic_standard",
        result_product=result_product,
        result_evidence_id=f"sha256:{result_sha256}",
        result_sha256=result_sha256,
        input_product=input_product,
        input_evidence_id=input_evidence_id,
        input_sha256=input_sha256,
        analysis_frame_sha256=analysis_frame_sha256,
        ph_diagnostic_product=SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
        ph_diagnostic_evidence_id=f"sha256:{ph_sha256}",
        ph_diagnostic_sha256=ph_sha256,
        exposure_source=exposure,
        outcome=outcome,
        effect_scale=effect_scale,
        analysis_population=population,
        n_source_rows=n_source_rows,
        n_analysis_rows=len(analysis),
        n_complete_case_dropped=n_source_rows - len(analysis),
        n_censored_at_horizon=int(censored_at_horizon.sum()),
        n_events=n_events,
        time_origin=time_origin,
        time_column=time_column,
        time_unit=time_unit,
        event_column=event_column,
        event_value=event_value,
        event_definition=event_definition,
        censoring_strategy=censoring_strategy,
        competing_risk_strategy=competing_risk_strategy,
        competing_event_value=competing_event_value,
        competing_event_definition=competing_event_definition,
        supplementary_sha256=supplementary_sha256,
        competing_analysis_frame_sha256=_canonical_frame_sha256(competing_analysis) if competing else None,
        time_horizon=time_horizon,
        time_horizon_value=time_horizon_value,
        estimator=estimator,
        effect_measure=effect_measure,
        formula=formula,
        covariates=adjustment,
        model_terms=parsed_terms,
        design_columns=list(compiled.design.columns),
        exposure_design_column=exposure_design_column,
        applied_filter=applied_filter,
        package_versions=package_versions,
        proportional_hazards_diagnostic=ph_diagnostic,
        proportional_hazards_tested=True,
        proportional_hazards_p_value=global_p,
        proportional_hazards_exposure_p_value=exposure_p,
        proportional_hazards_decision_rule="exposure_or_global",
        proportional_hazards_alpha=float(ph_alpha),
        proportional_hazards_policy=ph_policy,
        proportional_hazards_status=ph_status,
        paper_authorization_allowed=paper_authorization_allowed,
        missing_data_plan=preparation.missing_data if preparation else None,
    )
    receipt_path = out_dir / _RECEIPT_FILENAME
    receipt_path.write_text(
        json.dumps(
            receipt.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    if sha256_file(input_path) != input_sha256:
        raise SurvivalPrimaryExecutionError(
            "survival_input_digest_mismatch_after_execution",
            "Bound cohort digest changed during execution",
        )

    summary: Dict[str, Any] = {
        "status": "ok",
        "analysis_family": "survival",
        "analysis_role": "primary",
        "deterministic_standard_analysis": SURVIVAL_PRIMARY_ANALYSIS_KIND,
        "receipt_issuer": SURVIVAL_PRIMARY_OWNER,
        "typed_cohort_input": input_product,
        "input_evidence_id": input_evidence_id,
        "input_sha256": input_sha256,
        "primary_predictor": exposure,
        "hazard_ratio": estimate.hazard_ratio,
        "hazard_ratio_ci_low": estimate.ci_low,
        "hazard_ratio_ci_high": estimate.ci_high,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "sample_size": len(analysis),
        "n_analysis": len(analysis),
        "n_events": n_events,
        "formula": formula,
        "covariates": adjustment,
        "model_terms": serialise_model_terms(parsed_terms),
        "design_columns": list(compiled.design.columns),
        "exposure_design_column": exposure_design_column,
        "proportional_hazards_p_value": global_p,
        "proportional_hazards_exposure_p_value": exposure_p,
        "proportional_hazards_decision_rule": "exposure_or_global",
        "proportional_hazards_alpha": float(ph_alpha),
        "proportional_hazards_policy": ph_policy,
        "proportional_hazards_status": ph_status,
        "paper_authorization_allowed": paper_authorization_allowed,
        "estimand": result_row["estimand"],
        "estimand_question": CAUSE_SPECIFIC_QUESTION,
        "competing_event_value": competing_event_value,
        "competing_event_definition": competing_event_definition,
        "survival_group_comparisons": supplements.details,
        "output_files": {
            result_product: result_path.name,
            SURVIVAL_PH_DIAGNOSTIC_PRODUCT: ph_path.name,
            SURVIVAL_ANALYSIS_RECEIPT_PRODUCT: receipt_path.name,
            **supplementary_files,
        },
    }
    if preparation is not None:
        summary["data_preparation"] = write_preparation_receipt(
            out_dir, plan=preparation, execution=preparation_execution,
            missing_data=imputation_diagnostics, sensitivity=missing_sensitivity,
        )
        summary["data_preparation_provenance"] = "data_preparation_provenance.json"
        summary["executed_method_design"] = summary["data_preparation"]["executed_method_design"]
    if not emit_step_summary:
        return summary
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary


def impute_survival_source(
    source,
    *,
    plan,
    time_column,
    event_column,
    event_value,
    horizon,
    exposure,
    competing=False,
):
    missing = plan.missing_data
    if competing:
        raise PreparationRefusal(
            "mi_competing_risks_unsupported",
            "The MI owner supports ordinary Cox, not pooled competing-risk supplements",
        )
    if source[[time_column, event_column]].isna().any().any():
        raise PreparationRefusal(
            "mi_outcome_missing", "Cox MI requires observed follow-up and event status"
        )
    if {time_column, event_column} & {item.name for item in missing.variables}:
        raise PreparationRefusal(
            "mi_survival_endpoint_imputation_unsupported",
            "Follow-up and event status cannot be imputed by this owner",
        )
    if source[exposure].isna().any():
        raise PreparationRefusal(
            "mi_survival_exposure_missing",
            "This Cox owner imputes covariates and retains observed exposure groups",
        )
    times = source[time_column].clip(upper=horizon)
    event = source[event_column].eq(event_value) & source[time_column].le(horizon)
    counts = (
        pd.DataFrame({"time": times, "event": event})
        .groupby("time")
        .agg(n=("event", "size"), d=("event", "sum"))
    )
    risks = counts.n.iloc[::-1].cumsum().iloc[::-1]
    cumulative = (counts.d / risks).cumsum()
    model = source.drop(columns=time_column).copy()
    model[event_column] = event.astype(int)
    hazard = "__mi_nelson_aalen_hazard"
    if hazard in model:
        raise PreparationRefusal(
            "mi_reserved_column",
            "The cumulative-hazard coordinate collides with an input",
        )
    model[hazard] = times.map(cumulative)
    result = multiple_impute(model, plan=missing, outcome=event_column)
    datasets = []
    for data in result.datasets:
        data = data.drop(columns=hazard)
        data[time_column] = source[time_column]
        data[event_column] = source[event_column]
        datasets.append(data)
    result.datasets = tuple(datasets)
    result.diagnostics["survival_predictors"] = (
        "Observed event indicator and Nelson–Aalen cumulative hazard at administratively censored follow-up"
    )
    return result


def _analysis(
    source, *, plan, terms, exposure, time_column, event_column, event_value, horizon
):
    compiled = compile_prepared_terms(
        source, plan=plan, terms=terms, exposure=exposure, outcome=event_column
    )
    frame = source.loc[compiled.design.index, [time_column, event_column]].join(
        compiled.design
    )
    frame[event_column] = (
        frame[event_column].eq(event_value) & frame[time_column].le(horizon)
    ).astype(int)
    frame[time_column] = frame[time_column].clip(upper=horizon)
    return frame, compiled


def pool_survival_imputations(
    *,
    source,
    imputed,
    plan,
    terms,
    exposure,
    time_column,
    event_column,
    event_value,
    horizon,
    fit_cox,
    ph_test,
    primary_estimate,
    primary_ph,
):
    missing = plan.missing_data
    fits, diagnostics = [], []
    for index, data in enumerate(imputed.datasets):
        analysis, compiled = _analysis(
            data,
            plan=plan,
            terms=terms,
            exposure=exposure,
            time_column=time_column,
            event_column=event_column,
            event_value=event_value,
            horizon=horizon,
        )
        if len(analysis) != len(source):
            raise PreparationRefusal(
                "mi_rows_dropped", "Imputed Cox analyses may not drop rows"
            )
        if index == 0 and missing.strategy == "multiple_imputation":
            fit, ph = primary_estimate, primary_ph
        else:
            fit = fit_cox(
                analysis,
                duration_column=time_column,
                event_column=event_column,
                exposure=compiled.exposure_columns[0],
            )
            ph = ph_test(
                analysis,
                duration_col=time_column,
                event_col=event_column,
                covariates=list(compiled.design.columns),
                time_transform="km",
            )
        fits.append(fit)
        diagnostics.append(ph)
    pool = rubin_pool(
        [fit.coefficient for fit in fits],
        [fit.standard_error**2 for fit in fits],
        complete_df=int(
            (
                source[event_column].eq(event_value) & source[time_column].le(horizon)
            ).sum()
        )
        - len(compiled.design.columns),
    )
    estimate = replace(
        fits[0],
        hazard_ratio=math.exp(pool.estimate),
        ci_low=math.exp(pool.ci_low),
        ci_high=math.exp(pool.ci_high),
        standard_error=pool.standard_error,
        coefficient=pool.estimate,
        p_value=pool.p_value,
    )
    aggregate = (
        pd.concat(diagnostics)
        .groupby("covariate", sort=False)
        .agg(
            test_statistic=("test_statistic", "max"),
            p_value=("p_value", "min"),
        )
        .reset_index()
    )
    aggregate["p_value"] = (aggregate.p_value * len(fits)).clip(upper=1)
    aggregate["method"] = "across_imputations_Bonferroni_minimum_p"
    aggregate["m"] = len(fits)
    diagnostic = {**imputed.diagnostics, "pooling": pool.to_json()}
    if missing.strategy == "complete_case":
        sensitivity = {
            "strategy": "multiple_imputation",
            "status": "fitted",
            "n": len(source),
            "estimate": estimate.hazard_ratio,
            "ci_low": estimate.ci_low,
            "ci_high": estimate.ci_high,
        }
        return primary_estimate, primary_ph, diagnostic, sensitivity
    cc, cc_compiled = _analysis(
        source,
        plan=plan,
        terms=terms,
        exposure=exposure,
        time_column=time_column,
        event_column=event_column,
        event_value=event_value,
        horizon=horizon,
    )
    try:
        fit = fit_cox(
            cc,
            duration_column=time_column,
            event_column=event_column,
            exposure=cc_compiled.exposure_columns[0],
        )
        sensitivity = {
            "strategy": "complete_case",
            "status": "fitted",
            "n": len(cc),
            "estimate": fit.hazard_ratio,
            "ci_low": fit.ci_low,
            "ci_high": fit.ci_high,
        }
    except RuntimeError:
        sensitivity = {
            "strategy": "complete_case",
            "status": "not_estimable",
            "reason_code": "complete_case_cox_fit_failed",
            "n": len(cc),
        }
    return estimate, aggregate, diagnostic, sensitivity


__all__ = [
    "SURVIVAL_PRIMARY_ANALYSIS_KIND",
    "SurvivalPrimaryExecutionError",
    "run_survival_primary",
    "survival_primary_executor_code",
    "survival_primary_executor_owns_step",
    "survival_primary_executor_scaffold",
    "survival_primary_executor_verdict",
]

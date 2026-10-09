"""Deterministic clone-censor-weight emulation of one signed target trial.

The caller-reviewed runtime authority (``TargetTrialRuntimeAuthority``) owns
every coordinate of the trial and the host policy
(``contracts.target_trial_design``) every constant of its emulation.  This
module reads the input's columns into the stays of the trial, applies the
time-zero rules, codes the confounders for the weight models, runs the
estimate (``methods.clone_censor_weight``) and its diagnostics
(``methods.clone_censor_weight_diagnostics``), judges the prespecified stops,
runs the bootstrap and writes the products, the reporting envelope and the
receipt.  It contains no case identifier and no model-editable code.  The
composite figure is rendered by ``target_trial_figure`` from the tables
written here.

Reading the input
-----------------
* The treatment starts at the earliest onset any of its columns records.
* A death by the horizon is placed at its recorded death time when the input
  records one; otherwise at its follow-up time, which is read by calendar
  day, or at the ICU exit when that time precedes the exit.  A death after
  hospital discharge follows an exit it ties with; a hospital death the
  input records no time for is, at that time, a death in the ICU.  The
  receipt counts both kinds and the ones placed at the exit; a day that ends
  before the ICU exit contradicts the stay.
* The ICU exit is the ICU length of stay, in days, read as hours.
* A stay is resampled with every stay of its patient when the authority
  names a patient grouping, else on its own.
* The confounders enter the weight models as the shared model-term compiler
  codes them; a confounder the authority keeps as its own unmeasured state is
  coded so, and a stay without another confounder measured, or whose
  unmeasured state cannot be estimated (``contracts.model_retention``), is not
  in the trial.  The eligibility flow counts those stays.

An input that contradicts itself -- a death time within the horizon on a
stay that survived it, a day of death that ended before the ICU exit, a value
no column can hold -- fails.  A condition of
the data that leaves the prespecified estimate without a sound value stops
the suite: it raises ``ExecutorStop`` with one of
``TARGET_TRIAL_STOP_REASONS`` and leaves the stop's record, codes only, for
the host (``contracts.executor_stop``).  The point estimate, its diagnostics
and the first seven stops come before the bootstrap; the last stop reads it.

The estimate is an analysis under the emulation's assumptions: no
unmeasured confounding, positivity, correctly specified weight models and
censoring at ICU exit that the baseline covariates explain.  Its evidence
ceiling is ``analysis_only``, and its causal sentences are the fixed
templates of its host claims (``authority.target_trial_scientific_claims``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from ...authority.current_case_scientific_runtime import (
    load_current_case_scientific_runtime_authority,
)
from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...authority.target_trial_runtime import (
    TARGET_TRIAL_PLAN_METHOD,
    TargetTrialRuntimeAuthority,
)
from ...authority.target_trial_scientific_claims import (
    TARGET_TRIAL_REPORTING_KEY,
    TARGET_TRIAL_REPORTING_SCHEMA_VERSION,
    build_target_trial_manuscript_projection,
)
from ...contracts.dependence import PatientGroupResolutionError, resolve_patient_groups
from ...contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
    TargetTrialDesign,
    executed_method_design_payload,
)
from ...contracts.executor_stop import ExecutorStop, write_executor_stop_record
from ...contracts.host_scaffold import HostScaffoldedScript
from ...contracts.manuscript_tables import (
    MANUSCRIPT_TABLE_SCHEMA_VERSION,
    MANUSCRIPT_TABLES_KEY,
    validate_manuscript_table_declarations,
)
from ...contracts.target_trial_design import (
    TARGET_TRIAL_HOST_POLICY,
    TARGET_TRIAL_STOP_THRESHOLDS,
    target_trial_host_policy_sha256,
)
from ..model_matrix import ModelTermCompilationError, covariate_model_rows
from .plausibility_receipt import render_standard_plausibility_receipt_code
from .typed_input_binding import sole_typed_cohort_input
from ...schema import AnalysisPlan, AnalysisStep

TARGET_TRIAL_ANALYSIS_KIND = TARGET_TRIAL_PLAN_METHOD
#: The 0/1 column the coding of the confounders judges an unmeasured state's
#: estimability on: the stay started the treatment in the grace period.
_STARTED_COLUMN = "__target_trial_started_in_grace"
#: A recorded death time and the endpoint's follow-up may come from sources
#: with different resolution; within this many hours they agree.
_DEATH_TIME_TOLERANCE_HOURS = 1.0
#: Model-term compilation findings that are a condition of the eligible stays,
#: not a defect of the input: the prespecified weight model has no estimate.
_SINGULAR_COVARIATE_FINDINGS = frozenset(
    {"model_term_declared_level_absent", "missing_category_covariate_unobserved"}
)
_FILES = {
    "protocol": "target_trial_protocol.csv",
    "eligibility": "target_trial_eligibility_flow.csv",
    "table_one": "target_trial_table_one.csv",
    "risk_curve": "target_trial_risk_curves.csv",
    "effect": "target_trial_effect_estimates.csv",
    "weight_model": "target_trial_weight_models.csv",
    "weight": "target_trial_weight_diagnostics.csv",
    "balance": "target_trial_covariate_balance.csv",
    "positivity": "target_trial_positivity.csv",
    "adherence": "target_trial_adherence.csv",
    "bootstrap": "target_trial_bootstrap_replicates.csv",
    "analysis": "target_trial_analysis_cohort.parquet",
    "receipt": "target_trial_runtime_receipt.json",
}
#: The eligibility flow, in protocol order; the estimator's time-zero rules
#: (``methods.clone_censor_weight.TIME_ZERO_INCLUSIONS`` and ``_EXCLUSIONS``)
#: fill the middle stages.
_ELIGIBILITY_STAGES = (
    "source_rows",
    "endpoint_observed",
    "alive_at_time_zero",
    "in_icu_at_time_zero",
    "no_treatment_start_before_time_zero",
    "baseline_covariates_usable",
)
_ARM_PREFIXES = {"initiate": "started", "defer": "not_started"}


def _sealed(
    authority: TargetTrialRuntimeAuthority | Mapping[str, Any] | None,
) -> TargetTrialRuntimeAuthority | None:
    if authority is None:
        return None
    sealed = load_current_case_scientific_runtime_authority(authority)
    if not isinstance(sealed, TargetTrialRuntimeAuthority):
        return None
    return sealed


def target_trial_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: TargetTrialRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    sealed = _sealed(authority)
    return sealed is not None and sealed.governed_step(plan) == step


def target_trial_executor_scaffold(
    step: AnalysisStep,
    *,
    authority: TargetTrialRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None,
) -> HostScaffoldedScript:
    sealed = _sealed(authority)
    if sealed is None:
        raise TypeError("target trial executor requires its sealed authority")
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    typed_input = sole_typed_cohort_input(step)
    if typed_input is None:
        raise ValueError("target trial suite requires one typed cohort input")
    authority_json = json.dumps(sealed.model_dump(mode="json"), sort_keys=True)
    receipt_code = (
        render_standard_plausibility_receipt_code(
            plausibility_scope, frame_name="analysis_frame"
        )
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    prologue = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.target_trial_executor import (
            run_target_trial_suite,
        )
        from easyicu.research_agent.execution.runners.typed_input_binding import (
            load_typed_input,
            run_dir_from_env,
        )

        typed_cohort_input = {typed_input!r}
        authority = json.loads({json.dumps(authority_json)})
        bound = load_typed_input(
            input_key=typed_cohort_input,
            run_dir=run_dir_from_env(),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]).resolve(),
            expected_evidence_kind="table",
            exclusive=True,
        )
        analysis_frame = bound.frame
        """
    ).strip()
    if receipt_code:
        prologue += "\n\n" + receipt_code.strip()
    prologue += (
        "\n\n"
        + textwrap.dedent(
            f"""
        summary = run_target_trial_suite(
            frame=analysis_frame,
            authority=authority,
            runtime_projection_sha256={runtime_projection_sha256!r},
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            input_product=bound.input_key,
            input_evidence_id=bound.evidence_id,
            input_sha256=bound.sha256,
        )
        """
        ).strip()
    )
    epilogue: list[str] = []
    if receipt_code:
        epilogue.append('summary["plausibility_audit"] = plausibility_audit')
    epilogue.extend(
        [
            'out_dir = Path(os.environ["STEP_OUT_DIR"])',
            '(out_dir / "step_summary.json").write_text(',
            "    json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),",
            '    encoding="utf-8",',
            ")",
            "print(json.dumps(summary, ensure_ascii=False, allow_nan=False))",
        ]
    )
    return HostScaffoldedScript(
        prologue=prologue, body="", epilogue="\n".join(epilogue)
    )


def target_trial_executor_code(
    step: AnalysisStep,
    *,
    authority: TargetTrialRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope] = None,
) -> str:
    return target_trial_executor_scaffold(
        step,
        authority=authority,
        runtime_projection_sha256=runtime_projection_sha256,
        plausibility_scope=plausibility_scope,
    ).assembled()


# -- reading the input --------------------------------------------------------


def _reader_words(text: str) -> str:
    """Plain reader words for a table caption or label (no markup characters)."""

    cleaned = re.sub(r"[{}\[\]<>`\\|*_#]+", " ", str(text))
    return " ".join(cleaned.split())[:200] or "Unnamed"


def _numeric(frame: Any, column: str) -> Any:
    import numpy as np
    import pandas as pd

    source = frame[column]
    values = pd.to_numeric(source, errors="coerce")
    if bool((source.notna() & values.isna()).any()):
        raise ValueError(f"target trial column {column!r} holds non-numeric values")
    array = values.to_numpy(dtype=float)
    if bool(np.isinf(array).any()):
        raise ValueError(f"target trial column {column!r} holds infinite values")
    return array


@dataclass(frozen=True)
class _TrialInputs:
    """One value per input row, in the input's order."""

    unit_ids: Any
    group_ids: Any
    onset_hours: Any
    death_hours: Any
    icu_exit_hours: Any
    endpoint_observed: Any
    #: Deaths by the horizon after hospital discharge, placed by their
    #: follow-up time, read by day.
    deaths_timed_by_calendar_day: Any
    #: Other deaths by the horizon without a recorded time, placed the same way.
    other_deaths_timed_by_calendar_day: Any
    #: Of both, the deaths whose day preceded the ICU exit, placed at it.
    deaths_placed_at_icu_exit: Any


def _trial_inputs(frame: Any, sealed: TargetTrialRuntimeAuthority) -> _TrialInputs:
    import numpy as np

    ids = frame[sealed.unit_id_column].astype("object")
    if bool(ids.isna().any()) or bool(ids.duplicated().any()):
        raise ValueError("target trial unit identifiers must be present and unique")
    requirement = sealed.patient_group_requirement()
    if requirement is None:
        groups = ids.to_numpy(dtype=object)
    else:
        source = frame[requirement.group_source]
        if bool(source.isna().any()):
            raise ValueError("target trial patient grouping has missing values")
        try:
            resolved = resolve_patient_groups(
                source.astype("object").tolist(), requirement=requirement
            )
        except PatientGroupResolutionError as exc:
            raise ValueError(f"target trial patient grouping: {exc}") from exc
        groups = np.asarray(resolved.groups, dtype=object)

    onset = np.full(len(frame), np.nan)
    for column in sealed.treatment_onset_columns:
        onset = np.fmin(onset, _numeric(frame, column))

    event = _numeric(frame, sealed.event_column)
    followup = _numeric(frame, sealed.followup_time_column)
    horizon_days = float(sealed.endpoint_horizon_days)
    horizon_hours = 24.0 * horizon_days
    with np.errstate(invalid="ignore"):
        observed = (
            np.isin(event, (0.0, 1.0))
            & np.isfinite(followup)
            & (followup >= 0.0)
            & (followup <= horizon_days)
            & ((event == 1.0) | (followup >= horizon_days))
        )
        status = _numeric(frame, sealed.death_status_column)
        death_time = _numeric(frame, sealed.death_time_column)
        recorded = (status == 1.0) & np.isfinite(death_time)
        died = observed & (event == 1.0)
        late = (
            died & recorded & (death_time > horizon_hours + _DEATH_TIME_TOLERANCE_HOURS)
        )
        survived = (
            observed
            & (event == 0.0)
            & recorded
            & (death_time < horizon_hours - _DEATH_TIME_TOLERANCE_HOURS)
        )
    if bool(late.any()) or bool(survived.any()):
        raise ValueError(
            "target trial death times contradict the endpoint: "
            f"{int(late.sum())} deaths by the horizon are recorded after it and "
            f"{int(survived.sum())} stays that survived it record a death within it"
        )
    # A death without a recorded time is known by its day: it happened by the
    # end of that day.  One after hospital discharge, of a stay alive at it,
    # also happened after the ICU exit; a hospital death the input records no
    # time for may have happened at the exit.
    untimed = died & ~recorded
    calendar = untimed & (status == 0.0)
    day_hours = 24.0 * followup
    exit_hours = 24.0 * _numeric(frame, sealed.icu_length_of_stay_column)
    with np.errstate(invalid="ignore"):
        ended_in_icu = untimed & (
            day_hours + 24.0 + _DEATH_TIME_TOLERANCE_HOURS <= exit_hours
        )
        before_exit = untimed & (day_hours < exit_hours)
    if bool(ended_in_icu.any()):
        raise ValueError(
            "target trial death dates contradict the ICU stay: "
            f"{int(ended_in_icu.sum())} deaths known only by their day fell a "
            "full day before the ICU exit"
        )
    death_hours = np.where(
        recorded,
        death_time,
        np.where(untimed, np.fmax(day_hours, exit_hours), np.nan),
    )
    return _TrialInputs(
        unit_ids=ids.to_numpy(dtype=object),
        group_ids=groups,
        onset_hours=onset,
        death_hours=death_hours,
        icu_exit_hours=exit_hours,
        endpoint_observed=np.asarray(observed, dtype=bool),
        deaths_timed_by_calendar_day=np.asarray(calendar, dtype=bool),
        other_deaths_timed_by_calendar_day=np.asarray(untimed & ~calendar, dtype=bool),
        deaths_placed_at_icu_exit=np.asarray(before_exit, dtype=bool),
    )


def _stop(
    out_dir: Path, reason_code: str, *, detail: str, cause_code: Optional[str] = None
) -> ExecutorStop:
    stop = ExecutorStop(reason_code, cause_code=cause_code, detail=detail)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_executor_stop_record(out_dir, stop)
    return stop


def _coded_covariates(
    eligible: Any, sealed: TargetTrialRuntimeAuthority, started: Any, out_dir: Path
):
    """The weight models' design over the eligible stays, and the stays it keeps."""

    work = eligible[[item.column for item in sealed.covariates]].copy()
    work[_STARTED_COLUMN] = started.astype(int)
    work.index = range(len(work))
    try:
        rows = covariate_model_rows(
            work,
            terms=[item.model_term() for item in sealed.covariates],
            outcome=_STARTED_COLUMN,
            missing_category_covariates=[
                item.column for item in sealed.covariates if item.unmeasured_state
            ],
        )
    except ModelTermCompilationError as exc:
        if exc.reason_code in _SINGULAR_COVARIATE_FINDINGS:
            raise _stop(
                out_dir,
                "target_trial_weight_model_not_estimable",
                cause_code="singular_design",
                detail=f"a confounder cannot be coded among the eligible stays ({exc.reason_code})",
            ) from exc
        raise ValueError(f"target trial confounders cannot be coded: {exc}") from exc
    return rows


def _design_labels(sealed: TargetTrialRuntimeAuthority, rows: Any) -> dict[str, str]:
    """Reader words for each column of the weight models' design."""

    by_column = {item.column: item for item in sealed.covariates}
    indicators = {
        item.indicator: item.covariate for item in rows.missing_category_terms
    }
    labels: dict[str, str] = {}
    for column, source in rows.source_by_design_column.items():
        covariate = by_column[source]
        if column in indicators:
            labels[column] = f"{covariate.label}, not measured"
        elif column == source:
            labels[column] = covariate.label
        else:
            level = column[len(f"{source}__is_") :]
            labels[column] = f"{covariate.label}, {level}"
    return labels


# -- tables ---------------------------------------------------------------------


def _protocol_table(sealed: TargetTrialRuntimeAuthority):
    """The target trial and its emulation, row by row, in fixed words."""

    import pandas as pd

    t0, grace, horizon = (
        sealed.time_zero_hours,
        sealed.grace_period_hours,
        sealed.endpoint_horizon_days,
    )
    treatment = sealed.treatment_label
    resampled = "patients" if sealed.resampling_unit == "patient" else "ICU stays"
    rows = [
        (
            "eligibility",
            "Eligibility",
            f"{sealed.eligibility_label}, alive and in the ICU {t0} hours after ICU "
            f"admission, with no recorded start of {treatment} before then, and "
            f"with a known vital status at day {horizon}",
        ),
        (
            "treatment_strategies",
            "Treatment strategies",
            f"{sealed.initiate_label}: start {treatment} within {grace} hours of time "
            f"zero. {sealed.defer_label}: do not start it within those {grace} hours; "
            "afterwards unrestricted",
        ),
        (
            "assignment",
            "Assignment",
            "Each eligible stay is cloned into both strategies at time zero; the "
            "strategy a stay follows is not known then",
        ),
        (
            "time_zero",
            "Time zero",
            f"{t0} hours after ICU admission, when eligibility is met and follow-up "
            "starts",
        ),
        (
            "grace_period",
            "Grace period",
            f"{grace} hours from time zero; a clone is censored when its stay "
            "deviates from its strategy, and the censoring is weighted by baseline "
            "covariates",
        ),
        (
            "follow_up",
            "Follow-up",
            f"From time zero to death or day {horizon} after ICU admission",
        ),
        (
            "outcome",
            "Outcome",
            f"{sealed.outcome_label} by day {horizon} after ICU admission",
        ),
        (
            "contrast",
            "Contrast",
            f"Per-protocol: the risk of the outcome by day {horizon} under each "
            "strategy, their difference and their ratio",
        ),
        (
            "analysis",
            "Statistical analysis",
            "Inverse probability of censoring weights from pooled logistic models, "
            "stabilized and untruncated; weighted Kaplan-Meier risks; percentile "
            f"intervals from bootstrap resamples of {resampled}",
        ),
    ]
    return pd.DataFrame(
        [
            {"item": item, "label": label, "specification": _reader_words(text)}
            for item, label, text in rows
        ]
    )


def _eligibility_table(counts: list[int]):
    import pandas as pd

    source = counts[0]
    return pd.DataFrame(
        [
            {
                "stage_order": index + 1,
                "stage": stage,
                "count": int(count),
                "source_denominator": int(source),
                "percent_of_source": 100.0 * count / source if source else None,
                "excluded_since_prior_stage": (
                    0 if index == 0 else int(counts[index - 1] - count)
                ),
            }
            for index, (stage, count) in enumerate(zip(_ELIGIBILITY_STAGES, counts))
        ]
    )


def _table_one(analysis: Any, sealed: TargetTrialRuntimeAuthority, started: Any):
    """Characteristics at time zero by observed start in the grace period.

    The groups describe the eligible stays as they were treated; they are not
    the trial's arms, which hold a clone of every stay.  The standardized mean
    difference compares the stays that started with those that did not.
    """

    import numpy as np
    import pandas as pd

    groups = {"not_started": ~started, "started": started}
    rows: list[dict[str, Any]] = []
    for item in sealed.covariates:
        source = analysis[item.column]
        if item.coding != "continuous":
            for level in item.levels:
                row: dict[str, Any] = {
                    "variable": item.column,
                    "level": level,
                    "summary_type": "categorical_n_percent",
                }
                shares: dict[str, float] = {}
                for prefix, mask in groups.items():
                    subset = source.loc[mask]
                    denominator = int(len(subset))
                    count = int(subset.astype("string").str.strip().eq(level).sum())
                    share = count / denominator if denominator else float("nan")
                    row[f"{prefix}_n"] = count
                    row[f"{prefix}_denominator"] = denominator
                    row[f"{prefix}_percent"] = 100.0 * share if denominator else None
                    shares[prefix] = share
                pooled = (shares["started"] + shares["not_started"]) / 2.0
                spread = (
                    math.sqrt(pooled * (1.0 - pooled)) if math.isfinite(pooled) else 0.0
                )
                row["standardized_mean_difference"] = (
                    (shares["started"] - shares["not_started"]) / spread
                    if spread > 0
                    else None
                )
                rows.append(row)
            continue
        numeric = pd.to_numeric(source, errors="coerce")
        row = {
            "variable": item.column,
            "level": "",
            "summary_type": "continuous_mean_sd",
        }
        means: dict[str, float] = {}
        variances: dict[str, float] = {}
        for prefix, mask in groups.items():
            values = numeric.loc[mask].dropna()
            row[f"{prefix}_n"] = int(len(values))
            row[f"{prefix}_mean"] = float(values.mean()) if len(values) else None
            row[f"{prefix}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            row[f"{prefix}_median"] = float(values.median()) if len(values) else None
            row[f"{prefix}_q1"] = float(values.quantile(0.25)) if len(values) else None
            row[f"{prefix}_q3"] = float(values.quantile(0.75)) if len(values) else None
            means[prefix] = float(values.mean()) if len(values) else float("nan")
            variances[prefix] = (
                float(values.var(ddof=1)) if len(values) > 1 else float("nan")
            )
        pooled_sd = math.sqrt(
            np.nanmean([variances["started"], variances["not_started"]])
        )
        row["standardized_mean_difference"] = (
            (means["started"] - means["not_started"]) / pooled_sd
            if pooled_sd > 0 and math.isfinite(pooled_sd)
            else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _risk_curves(estimate: Any, sealed: TargetTrialRuntimeAuthority):
    import pandas as pd

    labels = {"initiate": sealed.initiate_label, "defer": sealed.defer_label}
    rows = []
    for arm_name, arm in estimate.arms.items():
        rows.append(
            {
                "arm": arm_name,
                "arm_label": labels[arm_name],
                "hours_from_time_zero": 0.0,
                "cumulative_risk": 0.0,
            }
        )
        for hour, risk in zip(arm.curve_hours, arm.curve_risk):
            rows.append(
                {
                    "arm": arm_name,
                    "arm_label": labels[arm_name],
                    "hours_from_time_zero": float(hour),
                    "cumulative_risk": float(risk),
                }
            )
    return pd.DataFrame(rows)


def _interval_row(
    estimand: str,
    weighting: str,
    role: str,
    estimate: Optional[float],
    interval: Optional[tuple[float, float]],
) -> dict[str, Any]:
    return {
        "estimand": estimand,
        "weighting": weighting,
        "analysis_role": role,
        "estimate": estimate,
        "ci_low": interval[0] if interval else None,
        "ci_high": interval[1] if interval else None,
    }


def _effect_table(estimate: Any, bootstrap: Any):
    import pandas as pd

    from ...methods.clone_censor_weight import DEFER, INITIATE

    arms = estimate.arms
    rows = []
    for weighting, role, risk_of, intervals in (
        ("stabilized", "primary", lambda arm: arm.risk, bootstrap.intervals),
        (
            "stabilized_truncated",
            "sensitivity",
            lambda arm: arm.risk_truncated,
            bootstrap.truncated_intervals,
        ),
        ("unweighted", "sensitivity", lambda arm: arm.risk_crude, None),
    ):
        initiate, defer = risk_of(arms[INITIATE]), risk_of(arms[DEFER])
        rows.append(
            _interval_row(
                "risk_initiate",
                weighting,
                role,
                initiate,
                intervals.risk.get(INITIATE) if intervals else None,
            )
        )
        rows.append(
            _interval_row(
                "risk_defer",
                weighting,
                role,
                defer,
                intervals.risk.get(DEFER) if intervals else None,
            )
        )
        rows.append(
            _interval_row(
                "risk_difference",
                weighting,
                role,
                initiate - defer,
                intervals.risk_difference if intervals else None,
            )
        )
        rows.append(
            _interval_row(
                "risk_ratio",
                weighting,
                role,
                initiate / defer if defer > 0 else None,
                intervals.risk_ratio if intervals else None,
            )
        )
    return pd.DataFrame(rows)


def _weight_model_table(estimate: Any):
    import pandas as pd

    rows = []
    for model in (
        estimate.initiation_model,
        estimate.initiation_numerator,
        estimate.icu_exit_model,
        estimate.icu_exit_numerator,
    ):
        if model is None:
            continue
        for term, coefficient in zip(model.terms, model.params):
            rows.append(
                {
                    "model": model.name,
                    "term": term,
                    "coefficient": float(coefficient),
                    "n_person_hours": int(model.n_rows),
                    "n_events": int(model.n_events),
                    "n_parameters": int(model.n_parameters),
                    "iterations": int(model.iterations),
                }
            )
    frame = pd.DataFrame(rows)
    frame["icu_exit_model_form"] = estimate.icu_exit_model_form
    return frame


def _weight_table(summaries: Any, estimate: Any):
    import pandas as pd

    rows = []
    for item in summaries:
        low, high = estimate.arms[item.arm].truncation_bounds
        rows.append(
            {
                "arm": item.arm,
                "truncated": bool(item.truncated),
                "n": item.n,
                "mean": item.mean,
                "sd": item.sd,
                "minimum": item.minimum,
                "p01": item.p01,
                "p50": item.p50,
                "p99": item.p99,
                "maximum": item.maximum,
                "effective_sample_size": item.ess,
                "effective_sample_share": item.ess_share,
                "n_clipped_low": item.n_clipped_low,
                "n_clipped_high": item.n_clipped_high,
                "truncation_low": low,
                "truncation_high": high,
            }
        )
    return pd.DataFrame(rows)


def _balance_table(rows: Any, labels: Mapping[str, str]):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "design_column": item.covariate,
                "label": labels.get(item.covariate, item.covariate),
                "arm": item.arm,
                "eligible_mean": item.eligible_mean,
                "eligible_sd": item.eligible_sd,
                "unweighted_mean": item.unweighted_mean,
                "weighted_mean": item.weighted_mean,
                "smd_unweighted": item.smd_unweighted,
                "smd_weighted": item.smd_weighted,
            }
            for item in rows
        ]
    )


_QUANTILE_COLUMNS = ("p000", "p001", "p025", "p050", "p075", "p099", "p100")


def _positivity_table(summary: Any, estimate: Any):
    import numpy as np
    import pandas as pd

    def quantiles(values: tuple[float, ...]) -> dict[str, Optional[float]]:
        return {
            name: (values[index] if values else None)
            for index, name in enumerate(_QUANTILE_COLUMNS)
        }

    low, high = summary.window
    probability = estimate.start_probability
    everyone = tuple(
        float(value)
        for value in np.percentile(
            probability, [0.0, 1.0, 25.0, 50.0, 75.0, 99.0, 100.0]
        )
    )
    return pd.DataFrame(
        [
            {
                "group": "all_eligible",
                "n": summary.n,
                "n_outside_window": summary.n_outside,
                "window_low": low,
                "window_high": high,
                **quantiles(everyone),
            },
            {
                "group": "started",
                "n": summary.n_started,
                "n_outside_window": int(
                    ((probability <= low) | (probability >= high))[
                        estimate.course.started
                    ].sum()
                ),
                "window_low": low,
                "window_high": high,
                **quantiles(summary.started),
            },
            {
                "group": "not_started",
                "n": summary.n_not_started,
                "n_outside_window": int(
                    ((probability <= low) | (probability >= high))[
                        ~estimate.course.started
                    ].sum()
                ),
                "window_low": low,
                "window_high": high,
                **quantiles(summary.not_started),
            },
        ]
    )


def _adherence_table(summaries: Any):
    import pandas as pd

    rows = []
    for item in summaries:
        for hour, count in enumerate(item.censored_by_hour):
            rows.append(
                {
                    "arm": item.arm,
                    "quantity": "censored_in_grace_hour",
                    "grace_hour": hour,
                    "count": int(count),
                }
            )
        for quantity, count in (
            ("clones", item.n_clones),
            ("censored_at_grace_end", item.censored_at_grace_end),
            ("deaths_in_grace", item.deaths_in_grace),
            ("followed_past_grace", item.followed_past_grace),
            ("events_by_horizon", item.events_by_horizon),
        ):
            rows.append(
                {
                    "arm": item.arm,
                    "quantity": quantity,
                    "grace_hour": None,
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_table(bootstrap: Any):
    import pandas as pd

    risks, truncated = bootstrap.replicate_risks, bootstrap.replicate_risks_truncated
    return pd.DataFrame(
        {
            "replicate": range(1, risks.shape[0] + 1),
            "risk_initiate": risks[:, 0],
            "risk_defer": risks[:, 1],
            "risk_initiate_truncated": truncated[:, 0],
            "risk_defer_truncated": truncated[:, 1],
        }
    )


def _estimates_outside_intervals(estimate: Any, bootstrap: Any) -> list[str]:
    """The reported estimates that lie outside their own percentile interval."""

    from ...methods.clone_censor_weight import DEFER, INITIATE

    initiate, defer = estimate.arms[INITIATE], estimate.arms[DEFER]
    checks = []
    for name, intervals, risk_of in (
        ("stabilized", bootstrap.intervals, lambda arm: arm.risk),
        ("truncated", bootstrap.truncated_intervals, lambda arm: arm.risk_truncated),
    ):
        first, second = risk_of(initiate), risk_of(defer)
        checks.extend(
            [
                (f"{name}_risk_initiate", first, intervals.risk.get(INITIATE)),
                (f"{name}_risk_defer", second, intervals.risk.get(DEFER)),
                (f"{name}_risk_difference", first - second, intervals.risk_difference),
                (
                    f"{name}_risk_ratio",
                    first / second if second > 0 else None,
                    intervals.risk_ratio,
                ),
            ]
        )
    return [
        name
        for name, value, interval in checks
        if value is None or interval is None or not interval[0] <= value <= interval[1]
    ]


def _canonical_frame_sha256(frame: Any) -> str:
    payload = frame.to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# -- the envelope ---------------------------------------------------------------


def _hour_terms(grace: int) -> str:
    if grace >= int(TARGET_TRIAL_HOST_POLICY["hour_spline_min_grace_hours"]):
        return "restricted_cubic_spline"
    return "linear" if grace >= 2 else "none"


def _percent(value: float) -> float:
    return 100.0 * float(value)


def _percent_estimate(
    value: Optional[float], interval: Optional[tuple[float, float]]
) -> Optional[dict[str, float]]:
    """An estimate on the percent scale with its percentile interval.

    The envelope states each estimate once, on the scale its sentence prints:
    a step summary registers a bounded number of values, and every one a
    sentence prints must be among them.
    """

    if value is None or interval is None or not math.isfinite(value):
        return None
    return {
        "estimate": _percent(value),
        "ci_low": _percent(interval[0]),
        "ci_high": _percent(interval[1]),
    }


def _ratio_estimate(
    numerator: float, denominator: float, interval: Optional[tuple[float, float]]
) -> Optional[dict[str, float]]:
    if denominator <= 0 or interval is None:
        return None
    return {
        "estimate": float(numerator / denominator),
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
    }


def _reportable(
    sealed: TargetTrialRuntimeAuthority,
    *,
    estimate: Any,
    bootstrap: Any,
    weights: Any,
    positivity: Any,
    balance: Any,
    exit_share: float,
    late_starts: int,
    e_value: Any,
    calendar_deaths: int,
    other_calendar_deaths: int,
    exit_deaths: int,
) -> dict[str, Any]:
    from ...methods.clone_censor_weight import DEFER, INITIATE

    arms = estimate.arms
    labels = {INITIATE: sealed.initiate_label, DEFER: sealed.defer_label}
    untruncated = {item.arm: item for item in weights if not item.truncated}
    smd_unweighted = [
        abs(row.smd_unweighted) for row in balance if row.smd_unweighted is not None
    ]
    smd_weighted = [
        abs(row.smd_weighted) for row in balance if row.smd_weighted is not None
    ]
    flag = float(TARGET_TRIAL_STOP_THRESHOLDS["balance_smd_flag"])
    low_pct, high_pct = TARGET_TRIAL_HOST_POLICY["weight_truncation_percentiles"]
    window_low, window_high = positivity.window
    primary, truncated = bootstrap.intervals, bootstrap.truncated_intervals

    def arm_payload(name: str) -> dict[str, Any]:
        arm = arms[name]
        summary = untruncated[name]
        return {
            "label": labels[name],
            "n_clones": arm.n_clones,
            "n_followed_past_grace": arm.n_followed_past_grace,
            "n_events": arm.n_events,
            "risk_percent": _percent_estimate(arm.risk, primary.risk.get(name)),
            "risk_truncated_percent": _percent_estimate(
                arm.risk_truncated, truncated.risk.get(name)
            ),
            "risk_unweighted_percent": _percent(arm.risk_crude),
            # The weights each clone carries past the grace period, before
            # truncation.
            "weights": {
                "mean": summary.mean,
                "maximum": summary.maximum,
                "p99": summary.p99,
                "effective_sample_size": summary.ess,
            },
        }

    initiate, defer = arms[INITIATE], arms[DEFER]
    return {
        "schema_version": TARGET_TRIAL_REPORTING_SCHEMA_VERSION,
        "execution_owner": "target_trial_executor_v1",
        "interpretation_ceiling": sealed.interpretation,
        "evidence_ceiling": sealed.evidence_ceiling,
        "treatment": sealed.treatment_label,
        "initiate_label": sealed.initiate_label,
        "defer_label": sealed.defer_label,
        "outcome": sealed.event_column,
        "outcome_label": sealed.outcome_label,
        "analysis_unit": sealed.analysis_unit_label,
        "population": sealed.eligibility_label,
        "time_zero_hours": sealed.time_zero_hours,
        "grace_period_hours": sealed.grace_period_hours,
        "horizon_days": sealed.endpoint_horizon_days,
        "adjustment_columns": [item.column for item in sealed.covariates],
        "adjustment_labels": [item.label for item in sealed.covariates],
        "n_eligible": int(estimate.eligible.n),
        "arms": {INITIATE: arm_payload(INITIATE), DEFER: arm_payload(DEFER)},
        "risk_difference_percentage_points": _percent_estimate(
            estimate.risk_difference, primary.risk_difference
        ),
        "risk_ratio": _ratio_estimate(initiate.risk, defer.risk, primary.risk_ratio),
        "truncated_risk_difference_percentage_points": _percent_estimate(
            initiate.risk_truncated - defer.risk_truncated, truncated.risk_difference
        ),
        "truncated_risk_ratio": _ratio_estimate(
            initiate.risk_truncated, defer.risk_truncated, truncated.risk_ratio
        ),
        "unweighted_risk_difference_percentage_points": _percent(
            initiate.risk_crude - defer.risk_crude
        ),
        "unweighted_risk_ratio": (
            float(initiate.risk_crude / defer.risk_crude)
            if defer.risk_crude > 0
            else None
        ),
        "weight_truncation_percentiles": [float(low_pct), float(high_pct)],
        # The E-value of the risk ratio, and of its interval bound nearer the
        # null when the interval excludes it.
        "e_value": (
            None
            if e_value is None
            else {
                "point": float(e_value.e_value),
                **(
                    {"interval_bound": float(e_value.e_value_lower_bound)}
                    if e_value.e_value_lower_bound is not None
                    else {}
                ),
            }
        ),
        "positivity": {
            "window_percent": [_percent(window_low), _percent(window_high)],
            "n_outside": positivity.n_outside,
            "percent_outside": _percent(positivity.share_outside),
        },
        "balance": {
            "max_abs_smd_unweighted": max(smd_unweighted) if smd_unweighted else None,
            "max_abs_smd_weighted": max(smd_weighted) if smd_weighted else None,
            "flag_threshold": flag,
            "n_flagged_weighted": int(sum(value > flag for value in smd_weighted)),
        },
        "icu_exit": {
            "percent_in_grace": _percent(exit_share),
            "model_form": estimate.icu_exit_model_form,
        },
        "late_starts": int(late_starts),
        "deaths_timed_by_calendar_day": int(calendar_deaths),
        "other_deaths_timed_by_calendar_day": int(other_calendar_deaths),
        "deaths_placed_at_icu_exit": int(exit_deaths),
        "bootstrap": {
            "resamples": bootstrap.resamples,
            "n_failed": bootstrap.n_failed,
            "confidence_level": bootstrap.confidence_level,
            "interval_method": "bootstrap_percentile",
        },
    }


def _executed_design(
    sealed: TargetTrialRuntimeAuthority,
    *,
    estimate: Any,
    calendar_deaths: int,
    other_calendar_deaths: int,
    exit_deaths: int,
    resamples: int,
) -> dict[str, Any]:
    low_pct, high_pct = TARGET_TRIAL_HOST_POLICY["weight_truncation_percentiles"]
    window_low, window_high = TARGET_TRIAL_HOST_POLICY["positivity_window"]
    return executed_method_design_payload(
        TargetTrialDesign(
            schema_version=EXECUTED_METHOD_DESIGN_SCHEMA_VERSION,
            design_kind="target_trial_clone_censor_weight",
            time_zero_hours=sealed.time_zero_hours,
            grace_period_hours=sealed.grace_period_hours,
            endpoint_horizon_days=sealed.endpoint_horizon_days,
            n_adjustment_covariates=len(sealed.covariates),
            n_unmeasured_state_covariates=sum(
                1 for item in sealed.covariates if item.unmeasured_state
            ),
            hour_terms=_hour_terms(sealed.grace_period_hours),
            icu_exit_model_form=estimate.icu_exit_model_form,
            weight_truncation_percentiles=[float(low_pct), float(high_pct)],
            positivity_window_percent=[100.0 * window_low, 100.0 * window_high],
            bootstrap_resamples=int(resamples),
            resampling_unit=sealed.resampling_unit,
            deaths_timed_by_calendar_day=int(calendar_deaths),
            other_deaths_timed_by_calendar_day=int(other_calendar_deaths),
            deaths_placed_at_icu_exit=int(exit_deaths),
        )
    )


def _manuscript_tables(
    sealed: TargetTrialRuntimeAuthority,
    *,
    analysis_events: Any,
    started: Any,
) -> list[dict[str, Any]]:
    """Declare the protocol, the eligibility flow and Table 1 as reader tables."""

    treatment = _reader_words(sealed.treatment_label)
    grace = sealed.grace_period_hours
    horizon = sealed.endpoint_horizon_days

    def group(prefix: str, label: str, mask: Any) -> dict[str, Any]:
        n = int(mask.sum())
        deaths = int(analysis_events[mask].sum())
        return {
            "prefix": prefix,
            "label": _reader_words(label),
            "n": n,
            "events": deaths,
            "events_percent": 100.0 * deaths / n if n else 0.0,
        }

    declarations = [
        {
            "schema_version": MANUSCRIPT_TABLE_SCHEMA_VERSION,
            "product": sealed.protocol_product,
            "caption": _reader_words(
                "Protocol of the target trial and of its emulation"
            ),
            "body": {
                "layout": "protocol_rows",
                "item_labels": {
                    "eligibility": "Eligibility",
                    "treatment_strategies": "Treatment strategies",
                    "assignment": "Assignment",
                    "time_zero": "Time zero",
                    "grace_period": "Grace period",
                    "follow_up": "Follow-up",
                    "outcome": "Outcome",
                    "contrast": "Contrast",
                    "analysis": "Statistical analysis",
                },
            },
            "notes": [
                "The emulation rests on assumptions the data cannot test: no "
                "unmeasured confounding, positivity, correctly specified weight "
                "models, and censoring at ICU exit that the baseline covariates "
                "explain.",
            ],
        },
        {
            "schema_version": MANUSCRIPT_TABLE_SCHEMA_VERSION,
            "product": sealed.eligibility_product,
            "caption": "Eligibility at time zero, from the source cohort to the trial",
            "body": {
                "layout": "stage_flow",
                "stage_labels": {
                    "source_rows": "Source cohort",
                    "endpoint_observed": "Endpoint observed",
                    "alive_at_time_zero": "Alive at time zero",
                    "in_icu_at_time_zero": "In the ICU at time zero",
                    "no_treatment_start_before_time_zero": (
                        f"No start of {treatment} before time zero"
                    ),
                    "baseline_covariates_usable": "Baseline covariates usable",
                },
            },
            "notes": [
                "Excluded counts are the stays removed since the stage before.",
                "A stay whose baseline covariate was not measured is kept as its own "
                "state when the covariate is so declared; otherwise it is excluded.",
            ],
        },
        {
            "schema_version": MANUSCRIPT_TABLE_SCHEMA_VERSION,
            "product": sealed.table_one_product,
            "caption": _reader_words(
                f"Characteristics at time zero of the eligible {sealed.analysis_unit_label}, "
                f"by start of {treatment} in the grace period"
            ),
            "body": {
                "layout": "grouped_summary",
                "groups": [
                    group("not_started", f"No start within {grace} hours", ~started),
                    group("started", f"Started within {grace} hours", started),
                ],
                "events_label": f"Deaths by day {horizon}, n (%)",
            },
            "notes": [
                "The groups describe the eligible stays as they were treated; the "
                "trial's strategies each hold a clone of every eligible stay.",
                "Categorical percentages use every stay of the group as the "
                "denominator, so levels need not sum to 100% when a value is missing.",
                "The standardized mean difference compares the stays that started "
                "with those that did not; it is not a significance test.",
            ],
        },
    ]
    return [
        declaration.model_dump(mode="json")
        for declaration in validate_manuscript_table_declarations(declarations)
    ]


# -- the suite ------------------------------------------------------------------


def run_target_trial_suite(
    *,
    frame: Any,
    authority: TargetTrialRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    out_dir: Path,
    input_product: str,
    input_evidence_id: str,
    input_sha256: str,
) -> dict[str, Any]:
    """Execute the exact sealed target trial emulation."""

    import numpy as np
    import pandas as pd

    from ...methods.clone_censor_weight import (
        DEFER,
        INITIATE,
        CloneCensorWeightError,
        TrialTiming,
        bootstrap_clone_censor_weight,
        estimate_clone_censor_weight,
        time_zero_eligibility,
        trial_course,
        trial_stays,
    )
    from ...methods.clone_censor_weight_diagnostics import (
        adherence_summaries,
        covariate_balance,
        grace_icu_exit_share,
        late_start_count,
        positivity_summary,
        risk_ratio_e_value,
        weight_summaries,
    )

    sealed = _sealed(authority)
    if sealed is None:
        raise TypeError("target trial runner received the wrong authority kind")
    if len(str(runtime_projection_sha256)) != 64:
        raise ValueError("target trial runtime projection digest is required")
    if sealed.host_policy_sha256 != target_trial_host_policy_sha256():
        raise ValueError("target trial was signed under another host policy")
    missing = sorted(set(sealed.required_columns) - set(frame.columns))
    if missing:
        raise ValueError("target trial input lacks columns: " + ", ".join(missing))
    if _STARTED_COLUMN in frame.columns:
        raise ValueError(f"target trial input already holds {_STARTED_COLUMN!r}")
    out_dir = Path(out_dir)
    thresholds = TARGET_TRIAL_STOP_THRESHOLDS
    timing = TrialTiming(
        time_zero_hours=sealed.time_zero_hours,
        grace_period_hours=sealed.grace_period_hours,
        horizon_hours=sealed.horizon_hours,
    )

    working = frame[list(sealed.required_columns)].reset_index(drop=True)
    inputs = _trial_inputs(working, sealed)
    n_source = len(working)
    no_covariates = np.empty((n_source, 0))

    def stays_at(positions: Any, covariates: Any, names: tuple[str, ...]):
        return trial_stays(
            stay_ids=inputs.unit_ids[positions],
            group_ids=inputs.group_ids[positions],
            onset_hours=inputs.onset_hours[positions],
            death_hours=inputs.death_hours[positions],
            icu_exit_hours=inputs.icu_exit_hours[positions],
            death_follows_icu_exit=inputs.deaths_timed_by_calendar_day[positions],
            endpoint_observed=inputs.endpoint_observed[positions],
            covariates=covariates,
            covariate_names=names,
            onset_window_hours=tuple(sealed.treatment_onset_window_hours),
        )

    try:
        everyone = stays_at(np.arange(n_source), no_covariates, ())
        eligibility = time_zero_eligibility(everyone, timing)
    except CloneCensorWeightError as exc:
        if exc.code == "ccw_no_eligible_stay":
            raise _stop(
                out_dir,
                "target_trial_sample_insufficient",
                detail="no stay is eligible at time zero",
            ) from exc
        raise ValueError(f"target trial input: {exc}") from exc
    eligible_positions = np.asarray(eligibility.analysis_positions, dtype=int)
    eligible_stays = everyone.take(eligible_positions)
    started_eligible = trial_course(eligible_stays, timing).started
    rows = _coded_covariates(
        working.iloc[eligible_positions], sealed, started_eligible, out_dir
    )
    kept = rows.design.index.to_numpy(dtype=int)
    positions = eligible_positions[kept]
    design = rows.design.astype(float)
    counts = [
        n_source,
        *(int(step.n_remaining) for step in eligibility.steps),
        int(positions.shape[0]),
    ]
    if len(counts) != len(_ELIGIBILITY_STAGES):
        raise ValueError("target trial eligibility stages drifted from the estimator")
    minimum = int(thresholds["min_eligible_stays"])
    if positions.shape[0] < minimum:
        raise _stop(
            out_dir,
            "target_trial_sample_insufficient",
            detail=f"{positions.shape[0]} stays are eligible; at least {minimum} are needed",
        )
    stays = stays_at(
        positions, design.to_numpy(), tuple(str(c) for c in design.columns)
    )
    try:
        estimate = estimate_clone_censor_weight(stays, timing)
    except CloneCensorWeightError as exc:
        if exc.code == "ccw_strategy_unobserved":
            raise _stop(
                out_dir, "target_trial_strategy_unobserved", detail=str(exc)
            ) from exc
        if exc.code == "ccw_weight_model_not_estimable":
            raise _stop(
                out_dir,
                "target_trial_weight_model_not_estimable",
                cause_code=exc.cause,
                detail=str(exc),
            ) from exc
        raise ValueError(f"target trial estimate failed on its input: {exc}") from exc

    weights = weight_summaries(estimate)
    positivity = positivity_summary(estimate)
    balance = covariate_balance(estimate)
    adherence = adherence_summaries(estimate)
    exit_share = grace_icu_exit_share(estimate)
    late_starts = late_start_count(estimate)

    # The prespecified stops on the point estimate, in protocol order.
    events = int(thresholds["min_events_per_arm"])
    short = [arm for arm in (INITIATE, DEFER) if estimate.arms[arm].n_events < events]
    if short:
        raise _stop(
            out_dir,
            "target_trial_events_insufficient",
            detail=f"the {short[0]} strategy has {estimate.arms[short[0]].n_events} "
            f"deaths by the horizon; at least {events} are needed in each",
        )
    initiation = estimate.initiation_model
    per_parameter = float(thresholds["min_initiation_events_per_parameter"])
    if initiation.n_events < per_parameter * initiation.n_parameters:
        raise _stop(
            out_dir,
            "target_trial_strategy_unobserved",
            detail=f"{initiation.n_events} starts in the grace period for "
            f"{initiation.n_parameters} parameters of the initiation model",
        )
    if positivity.share_outside > float(thresholds["max_positivity_outside_share"]):
        raise _stop(
            out_dir,
            "target_trial_positivity_violated",
            detail=f"{positivity.n_outside} of {positivity.n} eligible stays have a "
            "modelled probability of starting outside the positivity window",
        )
    ess_floor = float(thresholds["min_ess_share_of_adherers"])
    extreme = [
        item for item in weights if not item.truncated and item.ess_share < ess_floor
    ]
    if extreme:
        raise _stop(
            out_dir,
            "target_trial_weights_extreme",
            detail=f"the {extreme[0].arm} strategy's weights have an effective sample "
            f"size of {extreme[0].ess:.1f} for {extreme[0].n} clones",
        )
    if exit_share > float(thresholds["max_grace_icu_exit_share"]):
        leaving = int(estimate.course.leaves_icu_before_start.sum())
        raise _stop(
            out_dir,
            "target_trial_icu_exit_excessive",
            detail=f"{leaving} of {estimate.eligible.n} eligible stays "
            f"({exit_share:.1%}) leave the ICU in the grace period before starting",
        )

    bootstrap = bootstrap_clone_censor_weight(estimate)
    if bootstrap.failure_share > float(thresholds["max_bootstrap_failure_share"]):
        raise _stop(
            out_dir,
            "target_trial_bootstrap_unstable",
            cause_code="resamples_failed",
            detail=f"{bootstrap.n_failed} of {bootstrap.resamples} resamples failed "
            f"({', '.join(sorted(bootstrap.failures))})",
        )
    outside = _estimates_outside_intervals(estimate, bootstrap)
    if outside:
        raise _stop(
            out_dir,
            "target_trial_bootstrap_unstable",
            cause_code="estimate_outside_interval",
            detail="estimates outside their own percentile interval: "
            + ", ".join(outside),
        )

    e_value = risk_ratio_e_value(estimate.risk_ratio, bootstrap.intervals.risk_ratio)
    calendar_deaths = int(inputs.deaths_timed_by_calendar_day[positions].sum())
    other_calendar_deaths = int(
        inputs.other_deaths_timed_by_calendar_day[positions].sum()
    )
    exit_deaths = int(inputs.deaths_placed_at_icu_exit[positions].sum())
    reportable = _reportable(
        sealed,
        estimate=estimate,
        bootstrap=bootstrap,
        weights=weights,
        positivity=positivity,
        balance=balance,
        exit_share=exit_share,
        late_starts=late_starts,
        e_value=e_value,
        calendar_deaths=calendar_deaths,
        other_calendar_deaths=other_calendar_deaths,
        exit_deaths=exit_deaths,
    )
    reportable["manuscript_projection"] = build_target_trial_manuscript_projection(
        reportable
    )

    started = np.asarray(estimate.course.started, dtype=bool)
    analysis = working.iloc[positions].reset_index(drop=True)
    labels = _design_labels(sealed, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: out_dir / name for key, name in _FILES.items()}
    tables = {
        "protocol": _protocol_table(sealed),
        "eligibility": _eligibility_table(counts),
        "table_one": _table_one(analysis, sealed, pd.Series(started)),
        "risk_curve": _risk_curves(estimate, sealed),
        "effect": _effect_table(estimate, bootstrap),
        "weight_model": _weight_model_table(estimate),
        "weight": _weight_table(weights, estimate),
        "balance": _balance_table(balance, labels),
        "positivity": _positivity_table(positivity, estimate),
        "adherence": _adherence_table(adherence),
        "bootstrap": _bootstrap_table(bootstrap),
    }
    for key, table in tables.items():
        table.to_csv(paths[key], index=False)
    cohort = pd.DataFrame(
        {
            sealed.unit_id_column: analysis[sealed.unit_id_column],
            "resampling_unit": [str(value) for value in estimate.eligible.group_ids],
            "onset_hours": estimate.eligible.onset_hours,
            "death_hours": estimate.eligible.death_hours,
            "icu_exit_hours": estimate.eligible.icu_exit_hours,
            "started_in_grace": started,
            "start_probability": estimate.start_probability,
        }
    )
    cohort = pd.concat([cohort, design.reset_index(drop=True)], axis=1)
    cohort.to_parquet(paths["analysis"], index=False)

    deaths = np.isfinite(estimate.eligible.death_hours) & (
        estimate.eligible.death_hours <= sealed.horizon_hours
    )
    receipt = {
        "schema_version": "easyicu.target_trial_runtime_receipt/1",
        "protocol_content_sha256": sealed.protocol_content_sha256,
        "execution_contract_sha256": sealed.execution_contract_sha256,
        "target_trial_compile_sha256": sealed.target_trial_compile_sha256,
        "host_policy_sha256": sealed.host_policy_sha256,
        "approval_event_id": sealed.confirmation.approval_event_id,
        "runtime_projection_sha256": runtime_projection_sha256,
        "input_product": input_product,
        "input_evidence_id": input_evidence_id,
        "input_sha256": input_sha256,
        "analysis_frame_sha256": _canonical_frame_sha256(cohort),
        "eligibility_counts": dict(zip(_ELIGIBILITY_STAGES, counts)),
        "design_columns": [str(column) for column in design.columns],
        "unmeasured_state_terms": [
            item.public() for item in rows.missing_category_terms
        ],
        "unmeasured_rows_excluded": [
            item.public() for item in rows.unmeasured_rows_dropped
        ],
        "icu_exit_model_form": estimate.icu_exit_model_form,
        "deaths_timed_by_calendar_day": calendar_deaths,
        "other_deaths_timed_by_calendar_day": other_calendar_deaths,
        "deaths_placed_at_icu_exit": exit_deaths,
        "late_starts": late_starts,
        "bootstrap_resamples": bootstrap.resamples,
        "bootstrap_seed": bootstrap.seed,
        "bootstrap_failures": dict(bootstrap.failures),
        "risk_difference": float(estimate.risk_difference),
        "evidence_ceiling": sealed.evidence_ceiling,
        "interpretation": sealed.interpretation,
        "paper_authorization_allowed": False,
        "analysis_only": True,
        "human_attestation_required": True,
    }
    paths["receipt"].write_text(
        json.dumps(
            receipt, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
        ),
        encoding="utf-8",
    )
    output_files = {
        sealed.protocol_product: paths["protocol"].name,
        sealed.eligibility_product: paths["eligibility"].name,
        sealed.table_one_product: paths["table_one"].name,
        sealed.risk_curve_product: paths["risk_curve"].name,
        sealed.effect_product: paths["effect"].name,
        sealed.weight_model_product: paths["weight_model"].name,
        sealed.weight_product: paths["weight"].name,
        sealed.balance_product: paths["balance"].name,
        sealed.positivity_product: paths["positivity"].name,
        sealed.adherence_product: paths["adherence"].name,
        sealed.bootstrap_product: paths["bootstrap"].name,
        sealed.receipt_product: paths["receipt"].name,
    }
    # The per-step numeric cap registers headline roots first and then this
    # order: the design's numbers bind as one Methods fact, the envelope's
    # estimates and the counts follow.
    return {
        "status": "ok",
        "analysis_family": "causal_inference",
        "analysis_role": "primary",
        "deterministic_standard_analysis": TARGET_TRIAL_ANALYSIS_KIND,
        "interpretation_class": "per_protocol_effect_under_emulation_assumptions",
        EXECUTED_METHOD_DESIGN_KEY: _executed_design(
            sealed,
            estimate=estimate,
            calendar_deaths=calendar_deaths,
            other_calendar_deaths=other_calendar_deaths,
            exit_deaths=exit_deaths,
            resamples=bootstrap.resamples,
        ),
        TARGET_TRIAL_REPORTING_KEY: reportable,
        "n_source": n_source,
        "n_eligible": int(positions.shape[0]),
        "n_events": int(deaths.sum()),
        "typed_cohort_input": input_product,
        "input_evidence_id": input_evidence_id,
        "input_sha256": input_sha256,
        "paper_authorization_allowed": False,
        "analysis_only": True,
        "human_attestation_required": True,
        "analysis_cohort_file": paths["analysis"].name,
        "scientific_runtime_receipt": receipt,
        MANUSCRIPT_TABLES_KEY: _manuscript_tables(
            sealed, analysis_events=deaths, started=started
        ),
        "output_files": output_files,
    }


__all__ = [
    "TARGET_TRIAL_ANALYSIS_KIND",
    "run_target_trial_suite",
    "target_trial_executor_code",
    "target_trial_executor_owns_step",
    "target_trial_executor_scaffold",
]

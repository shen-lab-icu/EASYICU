"""Case-neutral primary-result integrity gates.

The reporting owner calls these pure-ish validators after execution.  They
inspect only authoritative step summaries and registered result evidence,
returning stable human-review messages without deciding readiness or mutating
run artifacts.  Ledger versus legacy filesystem lookup belongs to
``reporting.step_summaries``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..authority.runtime_artifacts import current_run_evidence_paths
from ..schema import AnalysisPlan
from .step_summaries import authoritative_step_summaries, step_authority_records


_PLAUSIBILITY_EVENT_KEYS = (
    "events",
    "n_events",
    "n_events_model",
    "num_events",
    "event_count",
)
_PLAUSIBILITY_N_KEYS = (
    "n",
    "n_model",
    "n_analysis",
    "n_analytic",
    "modeled_analytic_n",
    "n_complete_case",
    "n_complete_case_primary_model",
    "n_primary_complete_case",
    "n_stays",
    "n_patients",
    "n_obs",
    "n_full",
)
_PLAUSIBILITY_RATIO_KEYS = ("hazard_ratio", "odds_ratio", "risk_ratio")
_PLAUSIBILITY_RATE_KEYS = (
    "event_rate",
    "outcome_rate",
    "death_rate",
    "mortality_rate",
)
_PLAUSIBILITY_RESULT_MARKERS = (
    "hazard_ratio",
    "odds_ratio",
    "risk_ratio",
    "estimate",
    "point_estimate",
    "p_value",
    "pvalue",
    "log_hazard_ratio",
)
_PLAUSIBILITY_RESULT_CSVS = (
    "cox_summary.csv",
    "cox_model.csv",
    "adjusted_cox_model.csv",
    "hazard_ratio.csv",
    "adjusted_association.csv",
    "association_model_summary.csv",
    "crude_vs_adjusted_association.csv",
)


def _plausibility_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        x = float(value)
    elif isinstance(value, str):
        try:
            x = float(value.strip())
        except (ValueError, AttributeError):
            return None
    else:
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def _plausibility_first(
    mapping: Dict[str, Any], keys: Sequence[str]
) -> Optional[float]:
    for key in keys:
        if key in mapping:
            num = _plausibility_number(mapping[key])
            if num is not None:
                return num
    return None


def _plausibility_errors_for_row(where: str, row: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    if any(marker in row for marker in _PLAUSIBILITY_RESULT_MARKERS):
        events = _plausibility_first(row, _PLAUSIBILITY_EVENT_KEYS)
        n = _plausibility_first(row, _PLAUSIBILITY_N_KEYS)
        if events is not None and n is not None and n > 0 and events > n:
            errs.append(
                f"{where}: implausible primary result — {int(events)} events "
                f"exceed {int(n)} analysis units; an event count cannot exceed "
                "the sample (a corrupted/column-swapped result table)."
            )
    rate = _plausibility_first(row, _PLAUSIBILITY_RATE_KEYS)
    if rate is not None and (rate < 0.0 or rate > 1.0):
        errs.append(
            f"{where}: implausible event rate {rate} (a proportion must be "
            "within [0, 1])."
        )
    ratio = _plausibility_first(row, _PLAUSIBILITY_RATIO_KEYS)
    if ratio is not None and ratio <= 0.0:
        errs.append(
            f"{where}: implausible ratio estimate {ratio} (a hazard/odds/risk "
            "ratio must be > 0)."
        )
    lo = _plausibility_first(row, ("ci_low",))
    hi = _plausibility_first(row, ("ci_high",))
    if lo is not None and hi is not None and lo > hi:
        errs.append(
            f"{where}: inverted confidence interval (ci_low {lo} > ci_high {hi})."
        )
    return errs


def _plausibility_walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _plausibility_walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _plausibility_walk(item)


def primary_result_plausibility_errors(
    run_dir: Path,
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[str]:
    """Return case-neutral ``table == reality`` violations in primary artifacts."""

    per_step_records = step_authority_records(run_dir, per_step_records)
    errors: List[str] = []
    seen: set = set()

    def _add(new_errors: List[str]) -> None:
        for err in new_errors:
            if err not in seen:
                seen.add(err)
                errors.append(err)

    for label, payload in authoritative_step_summaries(run_dir, per_step_records):
        for mapping in _plausibility_walk(payload):
            _add(_plausibility_errors_for_row(label, mapping))

    if per_step_records is None:
        csv_paths = [
            path
            for outputs_dir in sorted((run_dir / "steps").glob("*/outputs"))
            for name in _PLAUSIBILITY_RESULT_CSVS
            if (path := outputs_dir / name).exists()
        ]
    else:
        csv_paths = [
            path
            for path in (
                current_run_evidence_paths(
                    run_dir,
                    per_step_records=per_step_records,
                )
                or []
            )
            if path.name.split("__", 1)[-1] in _PLAUSIBILITY_RESULT_CSVS
        ]
    for path in csv_paths:
        basename = path.name.split("__", 1)[-1]
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    _add(_plausibility_errors_for_row(basename, dict(row)))
        except Exception:
            continue
    return errors


_SURVIVAL_RESULT_KEYS = ("hazard_ratio", "cox_terms", "log_hazard_ratio")
_NESTED_SURVIVAL_RESULT_KEYS = (
    "hazard_ratio",
    "hr",
    "log_hazard_ratio",
    "log_hr",
    "cox_terms",
)


def _has_survival_result_structure(payload: Mapping[str, Any]) -> bool:
    if any(key in payload for key in _SURVIVAL_RESULT_KEYS):
        return True
    primary_model = payload.get("primary_model")
    return isinstance(primary_model, Mapping) and any(
        key in primary_model for key in _NESTED_SURVIVAL_RESULT_KEYS
    )


def _survival_summary_scalar(payload: Dict[str, Any], *keys: str) -> Optional[float]:
    containers: List[Dict[str, Any]] = [payload]
    primary_model = payload.get("primary_model")
    if isinstance(primary_model, dict):
        containers.append(primary_model)
    for container in containers:
        for key in keys:
            value = container.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")
    return None


def primary_survival_estimate_integrity_errors(
    plan: Optional[AnalysisPlan],
    run_dir: Optional[Path],
    per_step_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[str]:
    """Return impossible agent-produced survival summary values, fail closed."""

    if plan is None or run_dir is None:
        return []
    per_step_records = step_authority_records(run_dir, per_step_records)
    errors: List[str] = []
    active_summaries = dict(authoritative_step_summaries(run_dir, per_step_records))
    for step in getattr(plan, "steps", None) or []:
        step_id = str(getattr(step, "step_id", "") or "")
        if "figure" in step_id.lower():
            continue
        payload = active_summaries.get(step_id)
        if payload is None:
            continue
        payload = dict(payload)
        if not _has_survival_result_structure(payload):
            continue
        hr = _survival_summary_scalar(payload, "hazard_ratio", "hr", "point_estimate")
        if hr is not None and (not math.isfinite(hr) or hr <= 0):
            errors.append(
                f"primary survival step {step_id} reported an invalid hazard "
                f"ratio ({hr}); HR must be finite and > 0"
            )
        ci_low = _survival_summary_scalar(
            payload, "ci_low", "lower", "hr_ci_low", "lower_ci"
        )
        ci_high = _survival_summary_scalar(
            payload, "ci_high", "upper", "hr_ci_high", "upper_ci"
        )
        if ci_low is not None and ci_high is not None:
            if (
                not math.isfinite(ci_low)
                or not math.isfinite(ci_high)
                or ci_low <= 0
                or ci_high <= 0
                or ci_low > ci_high
                or (
                    hr is not None
                    and math.isfinite(hr)
                    and not (ci_low <= hr <= ci_high)
                )
            ):
                errors.append(
                    f"primary survival step {step_id} reported an invalid HR "
                    f"confidence interval ({ci_low}, {ci_high}) for estimate {hr}"
                )
        n_analysis = _survival_summary_scalar(
            payload, "n_analysis", "analysis_n", "n", "cohort_n"
        )
        n_events = _survival_summary_scalar(payload, "n_events", "events", "event_n")
        if n_analysis is not None and n_events is not None:
            if (
                not math.isfinite(n_analysis)
                or not math.isfinite(n_events)
                or n_analysis < 0
                or n_events < 0
                or n_events > n_analysis
            ):
                errors.append(
                    f"primary survival step {step_id} reported impossible event "
                    f"counts (events={n_events}, analysis_n={n_analysis})"
                )
    return errors


__all__ = [
    "primary_result_plausibility_errors",
    "primary_survival_estimate_integrity_errors",
]

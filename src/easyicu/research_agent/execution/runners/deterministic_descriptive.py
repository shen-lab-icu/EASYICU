"""Deterministic exposure-prevalence and absolute-risk context runner.

The runner is case-neutral: the current plan step's structured ``inputs`` name
the exposures, ``research_context.json`` names the outcome, and
``COHORT_PARQUET`` supplies the materialised analysis cohort. Continuous
exposures are summarised without creating post-hoc groups; categorical or
ordinal exposures retain their observed levels.
"""

from __future__ import annotations

import json
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...authority.plausibility import FlagOnlyPlausibilityScope
from .plausibility_receipt import render_standard_plausibility_receipt_code

__all__ = ["absolute_risk_context_code", "run_absolute_risk_context"]

_COMPANION_SUFFIXES = (
    "_measured",
    "_n",
    "_count",
    "_time",
    "_first_time",
    "_last_time",
)
_VALUE_SUFFIXES = (
    "_first_24h",
    "_first24h",
    "_24h",
    "_maximum",
    "_minimum",
    "_median",
    "_mean",
    "_max",
    "_min",
    "_first",
    "_last",
    "_peak",
    "_sum",
)


def absolute_risk_context_code(
    *,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    """Return the small script consumed by the instrumented step runner.

    The body is one call into host code, but the plausibility receipt is
    RENDERED here rather than computed inside it.  The obligation gate reads
    the script's own source to find a comparison against bounds taken from the
    contract, and a call into an imported function shows it nothing -- so a
    delegating stub is refused with ``plausibility_check_not_attributable``
    however correctly the callee behaves.

    That refusal was invisible until 2026-08-04, because this owner had never
    claimed a step: its ownership predicate carried a method allowlist the
    Planner never matched, 0 claims in 89 opportunities. The first run in which
    it did claim died here. The six sibling runners that were reachable all
    render the receipt the same way.
    """

    receipt = (
        render_standard_plausibility_receipt_code(
            plausibility_scope,
            frame_name="plausibility_frame",
        )
        if plausibility_scope is not None and plausibility_scope.expected_columns
        else ""
    )
    body = textwrap.dedent(
        """
        from easyicu.research_agent.execution.runners.deterministic_descriptive import (
            run_absolute_risk_context,
        )

        run_absolute_risk_context()
        """
    ).strip()
    if not receipt:
        return body
    # One read-modify-write after the body has written its summary, matching
    # the robustness runner: a second canonical write would give the static
    # obligation gate two writers to reason about.
    return "\n\n".join(
        (
            textwrap.dedent(
                """
                import json
                import os
                from pathlib import Path

                import pandas as pd

                plausibility_frame = pd.read_parquet(os.environ["COHORT_PARQUET"])
                """
            ).strip(),
            receipt,
            body,
            textwrap.dedent(
                """
                summary_path = Path(os.environ["STEP_OUT_DIR"]) / "step_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["plausibility_audit"] = plausibility_audit
                summary_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
                )
                """
            ).strip(),
        )
    )


#: The products this runner is competent to emit, identical to the set its
#: ownership predicate admits.  The Planner picks one of these NAMES; the
#: science is the same table either way.
_SUPPORTED_PRODUCTS = (
    "absolute_risk_context",
    "absolute_risk",
    "exposure_prevalence_and_absolute_risk",
    "exposure_outcome_summary",
)


def _declared_product(step: Mapping[str, Any]) -> str:
    """Return the product name the plan promised, not this runner's habit.

    The runner wrote ``exposure_outcome_summary`` unconditionally while the
    capability registry advertises it to the Planner as ``absolute_risk_context``
    -- so a plan that promised the advertised name got a step that registered a
    different one, and ``declared_product_contract`` refused it with
    ``declared_product_missing``. That went unseen for as long as this owner
    never claimed a step (0 of 89); e2's first real claim died on it.

    Only the four names this runner's ownership predicate admits are honoured,
    so a plan cannot rename the product into something this runner does not
    compute. With nothing declared the historical name is kept.
    """

    for value in step.get("expected_outputs") or []:
        kind, separator, name = str(value or "").strip().lower().partition(":")
        # The kind matters as much as the name. This runner writes ONE csv, and
        # a plan naming ``figure:absolute_risk_context`` is asking a renderer
        # for a picture -- 1 of the 92 recorded declarations does exactly that,
        # and honouring it would register a table under a figure's name.
        if separator and kind == "table" and name in _SUPPORTED_PRODUCTS:
            return name
    return "exposure_outcome_summary"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _resolve_column(
    requested: Any,
    columns: Sequence[Any],
    *,
    excluded: Sequence[str] = (),
) -> Optional[str]:
    """Resolve only exact/case-normalised structured names, never guess."""

    excluded_set = {str(value) for value in excluded}
    names = [str(value) for value in columns if str(value) not in excluded_set]
    if str(requested) in names:
        return str(requested)
    wanted = _normalise(requested)
    matches = [name for name in names if _normalise(name) == wanted]
    return matches[0] if len(matches) == 1 else None


def _load_current_step(run_dir: Path, step_id: str) -> Dict[str, Any]:
    manifest = _read_json(run_dir / "manifest_partial.json")
    plan_name = str(manifest.get("plan_path") or "analysis_plan.json")
    plan = _read_json(run_dir / plan_name)
    for step in plan.get("steps") or []:
        if isinstance(step, dict) and str(step.get("step_id") or "") == step_id:
            return step
    return {}


def _variable_metadata(context: Mapping[str, Any], column: str) -> Dict[str, Any]:
    for variable in context.get("variables") or []:
        if isinstance(variable, dict) and str(variable.get("name") or "") == str(
            column
        ):
            return variable
    return {}


def _is_companion_column(column: str) -> bool:
    lowered = str(column).lower()
    return any(lowered.endswith(suffix) for suffix in _COMPANION_SUFFIXES)


def _candidate_bases(column: str) -> List[str]:
    bases = [str(column)]
    current = str(column)
    while True:
        suffix = next(
            (item for item in _VALUE_SUFFIXES if current.lower().endswith(item)),
            None,
        )
        if suffix is None or len(current) <= len(suffix):
            return bases
        current = current[: -len(suffix)]
        if current not in bases:
            bases.append(current)


def _companion_column(
    exposure: str, suffix: str, columns: Sequence[Any]
) -> Optional[str]:
    names = {str(value) for value in columns}
    return next(
        (
            base + suffix
            for base in _candidate_bases(exposure)
            if base + suffix in names
        ),
        None,
    )


def _wilson(
    count: int, n: int, z: float = 1.959963984540054
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if n <= 0:
        return None, None, None
    rate = float(count / n)
    denominator = 1.0 + z * z / n
    centre = (rate + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n)) / denominator
    return rate, max(0.0, centre - half), min(1.0, centre + half)


def _source_states(
    frame: pd.DataFrame,
    exposure: str,
    measured_col: Optional[str],
    count_col: Optional[str],
) -> pd.Series:
    """Classify availability without replacing a missing value with zero."""

    value_present = frame[exposure].notna()
    if measured_col is not None:
        measured = pd.to_numeric(frame[measured_col], errors="coerce")
        flag_valid = measured.isin([0, 1])
        flag_one = measured.eq(1)
        flag_zero = measured.eq(0)
    else:
        flag_valid = pd.Series(True, index=frame.index)
        flag_one = pd.Series(True, index=frame.index)
        flag_zero = pd.Series(False, index=frame.index)

    if count_col is not None:
        count = pd.to_numeric(frame[count_col], errors="coerce")
        count_positive = count.gt(0)
        count_absent = count.isna() | count.le(0)
    else:
        count_positive = pd.Series(False, index=frame.index)
        count_absent = pd.Series(True, index=frame.index)

    if measured_col is not None:
        observed = value_present & flag_valid & flag_one
        no_source = (~value_present) & flag_valid & flag_zero
        measurement_missing = (~value_present) & flag_valid & flag_one
        if count_col is not None:
            observed &= count_positive
            no_source &= count_absent
            measurement_missing &= count_positive
    else:
        observed = value_present
        no_source = (~value_present) & count_absent
        measurement_missing = (~value_present) & count_positive
        if count_col is not None:
            observed &= count_positive

    state = pd.Series("inconsistent", index=frame.index, dtype="object")
    state.loc[observed] = "observed"
    state.loc[no_source] = "no_source"
    state.loc[measurement_missing] = "measurement_missing"
    return state


def _is_categorical(
    context: Mapping[str, Any], exposure: str, values: pd.Series
) -> bool:
    metadata = _variable_metadata(context, exposure)
    if bool(metadata.get("is_ordinal")):
        return True
    domain = metadata.get("observed_domain") or {}
    if isinstance(domain, dict) and bool(domain.get("is_binary")):
        return True
    dtype = str(metadata.get("dtype") or "").lower()
    role = str(metadata.get("role") or "").lower()
    if any(token in dtype for token in ("object", "category", "bool", "string")):
        return True
    if any(token in role for token in ("category", "categorical", "stage")):
        return True
    if metadata and "is_ordinal" in metadata:
        return False
    if pd.api.types.is_bool_dtype(values.dtype) or isinstance(
        values.dtype, pd.CategoricalDtype
    ):
        return True
    if not pd.api.types.is_numeric_dtype(values.dropna()):
        return True
    # Numeric cardinality is not a scientific type declaration. A 0/1 or
    # low-cardinality integer column may be a continuous/count exposure, so an
    # AuxiliaryRunner must not invent groups from observed values alone.
    return False


def _ordered_levels(
    context: Mapping[str, Any], exposure: str, values: pd.Series
) -> List[Any]:
    metadata = _variable_metadata(context, exposure)
    declared = metadata.get("ordinal_levels")
    observed = set(values.dropna().tolist())
    if isinstance(declared, (list, tuple)):
        levels = [level for level in declared if level in observed]
        return [*levels, *sorted(observed - set(levels), key=str)]
    try:
        return sorted(observed)
    except TypeError:
        return sorted(observed, key=str)


def _base_row(
    *,
    exposure: str,
    group_type: str,
    group_value: Any,
    label: str,
    n_total: int,
    measured_col: Optional[str],
    count_col: Optional[str],
) -> Dict[str, Any]:
    return {
        "exposure": exposure,
        "group_type": group_type,
        "group_value": str(group_value),
        "label": label,
        "n_denominator": n_total,
        "source_measured_column": measured_col,
        "source_count_column": count_col,
        "median": None,
        "q25": None,
        "q75": None,
        "minimum": None,
        "maximum": None,
    }


def _group_rows(
    *,
    exposure: str,
    group_type: str,
    group_value: Any,
    label: str,
    mask: pd.Series,
    outcome: pd.Series,
    n_total: int,
    measured_col: Optional[str],
    count_col: Optional[str],
) -> List[Dict[str, Any]]:
    mask = pd.Series(mask, index=outcome.index).fillna(False).astype(bool)
    group_n = int(mask.sum())
    prevalence, prev_low, prev_high = _wilson(group_n, n_total)
    base = _base_row(
        exposure=exposure,
        group_type=group_type,
        group_value=group_value,
        label=label,
        n_total=n_total,
        measured_col=measured_col,
        count_col=count_col,
    )
    prevalence_row = {
        **base,
        "estimate_type": "prevalence",
        "n": group_n,
        "n_positive": group_n,
        "event_n": None,
        "prevalence": prevalence,
        "prevalence_pct": prevalence * 100.0 if prevalence is not None else None,
        "outcome_risk": None,
        "outcome_risk_pct": None,
        "estimate": prevalence,
        "ci_low": prev_low,
        "ci_high": prev_high,
    }

    valid_outcome = mask & outcome.notna()
    risk_n = int(valid_outcome.sum())
    events = int(outcome.loc[valid_outcome].sum()) if risk_n else 0
    risk, risk_low, risk_high = _wilson(events, risk_n)
    risk_row = {
        **base,
        "estimate_type": "outcome_risk",
        "n": risk_n,
        "n_positive": None,
        "event_n": events,
        "prevalence": None,
        "prevalence_pct": None,
        "outcome_risk": risk,
        "outcome_risk_pct": risk * 100.0 if risk is not None else None,
        "estimate": risk,
        "ci_low": risk_low,
        "ci_high": risk_high,
    }
    return [prevalence_row, risk_row]


def _finite_or_none(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _reportable_group(
    *, prevalence_row: Mapping[str, Any], risk_row: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "group_type": str(prevalence_row["group_type"]),
        "group_value": str(prevalence_row["group_value"]),
        "label": str(prevalence_row["label"]),
        "n": int(prevalence_row["n"]),
        "prevalence_pct": _finite_or_none(prevalence_row["prevalence_pct"]),
        "prevalence_ci_low_pct": (
            float(prevalence_row["ci_low"]) * 100.0
            if _finite_or_none(prevalence_row["ci_low"]) is not None
            else None
        ),
        "prevalence_ci_high_pct": (
            float(prevalence_row["ci_high"]) * 100.0
            if _finite_or_none(prevalence_row["ci_high"]) is not None
            else None
        ),
        "outcome_n": int(risk_row["n"]),
        "outcome_event_n": int(risk_row["event_n"]),
        "outcome_risk_pct": _finite_or_none(risk_row["outcome_risk_pct"]),
        "outcome_risk_ci_low_pct": (
            float(risk_row["ci_low"]) * 100.0
            if _finite_or_none(risk_row["ci_low"]) is not None
            else None
        ),
        "outcome_risk_ci_high_pct": (
            float(risk_row["ci_high"]) * 100.0
            if _finite_or_none(risk_row["ci_high"]) is not None
            else None
        ),
    }


def _reportable_descriptive_results(
    *,
    table: pd.DataFrame,
    exposures: Sequence[str],
    context: Mapping[str, Any],
    outcome: pd.Series,
    outcome_col: str,
) -> Dict[str, Any]:
    valid_outcome = outcome.dropna()
    outcome_n = int(len(valid_outcome))
    outcome_event_n = int(valid_outcome.sum()) if outcome_n else 0
    risk, risk_low, risk_high = _wilson(outcome_event_n, outcome_n)
    exposure_results: List[Dict[str, Any]] = []
    for exposure in exposures:
        exposure_table = table.loc[table["exposure"].eq(exposure)]
        groups: List[Dict[str, Any]] = []
        pair_rows = exposure_table.loc[
            exposure_table["estimate_type"].isin(["prevalence", "outcome_risk"])
        ]
        for (group_type, group_value), pair in pair_rows.groupby(
            ["group_type", "group_value"], sort=False, dropna=False
        ):
            prevalence_rows = pair.loc[pair["estimate_type"].eq("prevalence")]
            risk_rows = pair.loc[pair["estimate_type"].eq("outcome_risk")]
            if len(prevalence_rows) != 1 or len(risk_rows) != 1:
                continue
            groups.append(
                _reportable_group(
                    prevalence_row=prevalence_rows.iloc[0].to_dict(),
                    risk_row=risk_rows.iloc[0].to_dict(),
                )
            )
        distribution_rows = exposure_table.loc[
            exposure_table["estimate_type"].eq("continuous_distribution")
        ]
        distribution = None
        if len(distribution_rows) == 1:
            row = distribution_rows.iloc[0]
            distribution = {
                key: _finite_or_none(row[key])
                for key in ("n", "median", "q25", "q75", "minimum", "maximum")
            }
        metadata = _variable_metadata(context, exposure)
        exposure_results.append(
            {
                "exposure": exposure,
                "description": metadata.get("description"),
                "unit": metadata.get("unit"),
                "materialized_representation": metadata.get("unit_normalization"),
                "groups": groups,
                "continuous_distribution": distribution,
            }
        )
    return {
        "schema_version": "easyicu.absolute_risk_reporting/1",
        "execution_owner": "absolute_risk_context_executor_v1",
        "interpretation_ceiling": "descriptive_not_causal",
        "overall_outcome": {
            "outcome": outcome_col,
            "n": outcome_n,
            "event_n": outcome_event_n,
            "risk_pct": risk * 100.0 if risk is not None else None,
            "risk_ci_low_pct": risk_low * 100.0 if risk_low is not None else None,
            "risk_ci_high_pct": risk_high * 100.0 if risk_high is not None else None,
        },
        "exposures": exposure_results,
    }


def _write_blocked(
    *,
    out_dir: Path,
    step_id: str,
    cohort_path: Path,
    reason: str,
    n_total: int = 0,
    outcome: Optional[str] = None,
) -> Dict[str, Any]:
    summary = {
        "step_id": step_id,
        "status": "blocked",
        "analysis_family": "absolute_risk_context",
        "interpretation_class": "absolute_risk_context",
        "blocking_reason": reason,
        "cohort_path": str(cohort_path),
        "n_total": n_total,
        "outcome": outcome,
        "exposure_columns": [],
        "adjusted_effect": None,
        "output_files": {},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def run_absolute_risk_context() -> Dict[str, Any]:
    """Execute the descriptive role using the standard runner environment."""

    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(os.environ.get("EASYICU_RUN_DIR") or out_dir.parents[2])
    step_id = os.environ.get("EASYICU_STEP_ID") or out_dir.parent.name
    cohort_path = Path(os.environ["COHORT_PARQUET"])
    context = _read_json(run_dir / "research_context.json")
    step = _load_current_step(run_dir, step_id)
    frame = pd.read_parquet(cohort_path).copy()
    n_total = int(len(frame))
    if n_total == 0:
        return _write_blocked(
            out_dir=out_dir,
            step_id=step_id,
            cohort_path=cohort_path,
            reason="Analysis cohort is empty; absolute risk is undefined.",
        )

    requested_outcome = os.environ.get("OUTCOME_COL") or context.get("target_outcome")
    outcome_col = _resolve_column(requested_outcome, frame.columns)
    if outcome_col is None:
        return _write_blocked(
            out_dir=out_dir,
            step_id=step_id,
            cohort_path=cohort_path,
            reason="The structured run context did not resolve to an outcome column.",
            n_total=n_total,
        )
    outcome = pd.to_numeric(frame[outcome_col], errors="coerce")
    outcome_values = set(outcome.dropna().astype(float).unique().tolist())
    if not outcome_values or not outcome_values <= {0.0, 1.0}:
        return _write_blocked(
            out_dir=out_dir,
            step_id=step_id,
            cohort_path=cohort_path,
            reason="Absolute-risk context requires a binary 0/1 outcome.",
            n_total=n_total,
            outcome=outcome_col,
        )

    exposures: List[str] = []
    for item in step.get("inputs") or []:
        column = _resolve_column(item, frame.columns, excluded=[outcome_col])
        if column is None or column in exposures or _is_companion_column(column):
            continue
        role = str(_variable_metadata(context, column).get("role") or "").lower()
        if not any(role.endswith(value) for value in ("id", "time", "outcome")):
            exposures.append(column)
    if not exposures:
        fallback = _resolve_column(context.get("primary_exposure"), frame.columns)
        if (
            fallback is not None
            and fallback != outcome_col
            and not _is_companion_column(fallback)
        ):
            exposures.append(fallback)
    if not exposures:
        return _write_blocked(
            out_dir=out_dir,
            step_id=step_id,
            cohort_path=cohort_path,
            reason="The current step has no structured exposure input in the cohort.",
            n_total=n_total,
            outcome=outcome_col,
        )

    rows: List[Dict[str, Any]] = []
    source_columns: Dict[str, Dict[str, Optional[str]]] = {}
    for exposure in exposures:
        values = frame[exposure]
        measured_col = _companion_column(exposure, "_measured", frame.columns)
        count_col = _companion_column(exposure, "_n", frame.columns)
        state: Optional[pd.Series] = None
        if measured_col is not None or count_col is not None:
            state = _source_states(frame, exposure, measured_col, count_col)
            source_columns[exposure] = {
                "measured": measured_col,
                "count": count_col,
            }
            for value in (
                "observed",
                "no_source",
                "measurement_missing",
                "inconsistent",
            ):
                mask = state.eq(value)
                if mask.any():
                    rows.extend(
                        _group_rows(
                            exposure=exposure,
                            group_type="source_state",
                            group_value=value,
                            label=f"{exposure} — {value}",
                            mask=mask,
                            outcome=outcome,
                            n_total=n_total,
                            measured_col=measured_col,
                            count_col=count_col,
                        )
                    )
        else:
            for value, mask in (
                ("observed", values.notna()),
                ("missing_summary", values.isna()),
            ):
                if mask.any():
                    rows.extend(
                        _group_rows(
                            exposure=exposure,
                            group_type="availability",
                            group_value=value,
                            label=f"{exposure} — {value}",
                            mask=mask,
                            outcome=outcome,
                            n_total=n_total,
                            measured_col=None,
                            count_col=None,
                        )
                    )

        observed_mask = state.eq("observed") if state is not None else values.notna()
        observed_values = values.loc[observed_mask].dropna()
        if _is_categorical(context, exposure, observed_values):
            for level in _ordered_levels(context, exposure, observed_values):
                rows.extend(
                    _group_rows(
                        exposure=exposure,
                        group_type="exposure_level",
                        group_value=level,
                        label=f"{exposure} = {level}",
                        mask=observed_mask & values.eq(level),
                        outcome=outcome,
                        n_total=n_total,
                        measured_col=measured_col,
                        count_col=count_col,
                    )
                )
        else:
            numeric = pd.to_numeric(observed_values, errors="coerce").dropna()
            rows.append(
                {
                    **_base_row(
                        exposure=exposure,
                        group_type="continuous_summary",
                        group_value="observed",
                        label=f"{exposure} — observed distribution",
                        n_total=n_total,
                        measured_col=measured_col,
                        count_col=count_col,
                    ),
                    "estimate_type": "continuous_distribution",
                    "n": int(len(numeric)),
                    "n_positive": None,
                    "event_n": None,
                    "prevalence": None,
                    "prevalence_pct": None,
                    "outcome_risk": None,
                    "outcome_risk_pct": None,
                    "estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "median": float(numeric.median()) if len(numeric) else None,
                    "q25": float(numeric.quantile(0.25)) if len(numeric) else None,
                    "q75": float(numeric.quantile(0.75)) if len(numeric) else None,
                    "minimum": float(numeric.min()) if len(numeric) else None,
                    "maximum": float(numeric.max()) if len(numeric) else None,
                }
            )

    table = pd.DataFrame(rows)
    product = _declared_product(step)
    table_path = out_dir / f"{product}.csv"
    table.to_csv(table_path, index=False)
    reportable_results = _reportable_descriptive_results(
        table=table,
        exposures=exposures,
        context=context,
        outcome=outcome,
        outcome_col=outcome_col,
    )
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_family": "absolute_risk_context",
        "interpretation_class": "absolute_risk_context",
        "method": (
            "Deterministic exposure prevalence and binary absolute-risk summary "
            "with Wilson 95% confidence intervals."
        ),
        "cohort_path": str(cohort_path),
        "n_total": n_total,
        "outcome": outcome_col,
        "outcome_nonmissing_n": int(outcome.notna().sum()),
        "outcome_missing_n": int(outcome.isna().sum()),
        "exposure_columns": exposures,
        "source_state_columns": source_columns,
        "n_summary_rows": int(len(table)),
        "reportable_descriptive_results": reportable_results,
        "adjusted_effect": None,
        "output_files": {f"table:{product}": table_path.name},
        "notes": [
            "Exposure columns came from the current plan step's structured inputs.",
            "Continuous exposures use median and IQR; no post-hoc bins were created.",
            "Missing exposure values were never imputed to zero.",
            "Source states used paired measured/count columns when available.",
        ],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "absolute_risk_context": "ok",
                "n_total": n_total,
                "n_summary_rows": int(len(table)),
            },
            ensure_ascii=False,
        )
    )
    return summary

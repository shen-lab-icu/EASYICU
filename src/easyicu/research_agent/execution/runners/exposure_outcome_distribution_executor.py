"""Deterministic executor for one closed exposure-by-outcome distribution.

Every scientific choice is read from the Planner's
``exposure_outcome_distribution_spec``: which column is the exposure, which
levels of it are reported, which column is the outcome, which observed value
counts as the event, whose rows form each denominator, and how the interval is
built. This module decides none of them. In particular it never infers the
exposure from column names, from the order of ``inputs``, or from the intent
text -- a study's exposure and outcome are its design, and an engine that
picks them has taken the investigator's decision.

The product is deliberately **self-contained**: each row carries its own
denominator, its missing count, its event count and its rate with an interval.
A renderer can therefore draw the whole figure from this one table, with no
second lookup into a cohort summary. That is what makes a figure step's input
contract closable before its parent has run.
"""

from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...schema import AnalysisStep, ExposureOutcomeDistributionSpec
from .typed_cohort_binding import load_step_cohort_frame

__all__ = [
    "EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS",
    "EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT",
    "exposure_outcome_distribution_executor_code",
    "exposure_outcome_distribution_executor_owns_step",
    "run_exposure_outcome_distribution_from_env",
]

EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT = "table:exposure_outcome_distribution"

#: The closed product schema. A renderer binds on this, never on the table's
#: name: two studies may call the product different things and still be the
#: same shape, and the same name may be given to a different shape.
EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS = (
    "row_role",
    "exposure_level",
    "n_rows",
    "exposure_denominator",
    "exposure_pct",
    "outcome_observed_n",
    "outcome_missing_n",
    "outcome_events",
    "outcome_denominator",
    "outcome_rate_pct",
    "ci_low_pct",
    "ci_high_pct",
)

_OVERALL_ROLE = "overall"
_LEVEL_ROLE = "exposure_level"


def _typed_cohort_input(step: AnalysisStep) -> str | None:
    """Return the single typed cohort input, ``""`` when the contract is open."""

    typed_inputs = {
        str(value or "").strip()
        for value in step.inputs
        if ":" in str(value or "").strip()
    }
    if not typed_inputs:
        return None
    if len(typed_inputs) != 1:
        return ""
    input_key = next(iter(typed_inputs))
    kind, separator, product = input_key.partition(":")
    if (
        separator
        and product
        and (kind == "cohort" or input_key == "artifact:analysis_cohort")
    ):
        return input_key
    return ""


def exposure_outcome_distribution_executor_owns_step(step: AnalysisStep) -> bool:
    """Own only a step whose distribution design is completely declared."""

    spec = step.exposure_outcome_distribution_spec
    if spec is None:
        return False
    return bool(
        str(step.method or "").strip().casefold() in {"descriptive", "distribution"}
        and str(step.planned_analysis_role or "").strip().casefold() == "auxiliary"
        and list(step.expected_outputs or []) == [EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT]
        and _typed_cohort_input(step) != ""
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
    )


def exposure_outcome_distribution_executor_code(step: AnalysisStep) -> str:
    """Return the small sandbox entrypoint for the exact declared design."""

    if not exposure_outcome_distribution_executor_owns_step(step):
        raise ValueError(
            "The step is not owned by the exposure-outcome distribution executor"
        )
    spec = step.exposure_outcome_distribution_spec
    assert spec is not None  # narrowed by owns_step
    return textwrap.dedent(
        f"""
        from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
            run_exposure_outcome_distribution_from_env,
        )

        run_exposure_outcome_distribution_from_env(
            spec_payload={spec.model_dump(mode="json")!r},
            typed_cohort_input={_typed_cohort_input(step)!r},
        )
        """
    ).strip()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentage(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(100.0 * numerator / denominator, 6)


def _wilson_interval(
    events: int, denominator: int
) -> tuple[Optional[float], Optional[float]]:
    """Wilson score interval, the method the spec declares.

    Chosen by the Planner rather than here. It is stated in the product so a
    reader never has to guess which interval a percentage carries.
    """

    if denominator <= 0:
        return (None, None)
    z = 1.959963984540054  # two-sided 95%
    proportion = events / denominator
    centre = proportion + z * z / (2 * denominator)
    spread = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4 * denominator)) / denominator
    )
    factor = 1.0 + z * z / denominator
    low = max(0.0, (centre - spread) / factor)
    high = min(1.0, (centre + spread) / factor)
    return (round(100.0 * low, 6), round(100.0 * high, 6))


def _matches(values: pd.Series, level: Any) -> pd.Series:
    """Match a declared level against observed values by the level's own type.

    A declared *number* matches the same number however the column stores it,
    including a string-typed ``"1"``: prepared columns arrive in whatever dtype
    the export produced, and refusing one would fail-close a study whose design
    is entirely correct. A declared *string* is compared as a string, so levels
    of ``0``/``1`` never quietly absorb a ``yes``/``no`` column -- those rows
    match nothing and the fail-closed policy stops the step.

    Booleans are checked before numbers because ``isinstance(True, int)`` is
    true in Python, and a boolean level must not be widened into ``1``.
    """

    if isinstance(level, bool):
        return values.map(lambda item: isinstance(item, bool) and item == level)
    if isinstance(level, (int, float)):
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.notna() & (numeric == float(level))
    return values.astype("object").map(
        lambda item: (not isinstance(item, bool))
        and item is not None
        and not (isinstance(item, float) and math.isnan(item))
        and str(item) == str(level)
    )


def _distribution_rows(
    frame: pd.DataFrame,
    *,
    spec: ExposureOutcomeDistributionSpec,
) -> List[Dict[str, Any]]:
    exposure_values = frame[spec.exposure]
    outcome_values = frame[spec.outcome]
    observed_outcome = outcome_values.notna()
    event = _matches(outcome_values, spec.outcome_positive_value) & observed_outcome

    level_masks = [
        (level, _matches(exposure_values, level)) for level in spec.exposure_levels
    ]

    covered = None
    for _, mask in level_masks:
        covered = mask if covered is None else (covered | mask)
    if covered is None:  # unreachable: the spec requires >= 2 levels
        raise RuntimeError("exposure_outcome_distribution spec declared no levels")
    unclassified = int((~covered).sum())
    if unclassified and spec.missing_exposure_policy == "fail_closed":
        raise RuntimeError(
            f"{unclassified} rows do not match any declared exposure level of "
            f"{spec.exposure!r}; the spec declares missing_exposure_policy="
            "'fail_closed', so they are not silently dropped or pooled"
        )

    total_rows = int(len(frame))
    exposure_denominator = total_rows

    rows: List[Dict[str, Any]] = []
    for level, mask in level_masks:
        n_rows = int(mask.sum())
        observed_n = int((mask & observed_outcome).sum())
        missing_n = n_rows - observed_n
        events = int((mask & event).sum())
        outcome_denominator = (
            n_rows if spec.denominator_policy == "all_declared_rows" else observed_n
        )
        low, high = _wilson_interval(events, outcome_denominator)
        rows.append(
            {
                "row_role": _LEVEL_ROLE,
                "exposure_level": level,
                "n_rows": n_rows,
                "exposure_denominator": exposure_denominator,
                "exposure_pct": _percentage(n_rows, exposure_denominator),
                "outcome_observed_n": observed_n,
                "outcome_missing_n": missing_n,
                "outcome_events": events,
                "outcome_denominator": outcome_denominator,
                "outcome_rate_pct": _percentage(events, outcome_denominator),
                "ci_low_pct": low,
                "ci_high_pct": high,
            }
        )

    overall_observed = int(observed_outcome.sum())
    overall_events = int(event.sum())
    overall_denominator = (
        total_rows
        if spec.denominator_policy == "all_declared_rows"
        else overall_observed
    )
    low, high = _wilson_interval(overall_events, overall_denominator)
    rows.append(
        {
            "row_role": _OVERALL_ROLE,
            "exposure_level": None,
            "n_rows": total_rows,
            "exposure_denominator": exposure_denominator,
            "exposure_pct": _percentage(total_rows, exposure_denominator),
            "outcome_observed_n": overall_observed,
            "outcome_missing_n": total_rows - overall_observed,
            "outcome_events": overall_events,
            "outcome_denominator": overall_denominator,
            "outcome_rate_pct": _percentage(overall_events, overall_denominator),
            "ci_low_pct": low,
            "ci_high_pct": high,
        }
    )
    return rows


def _verify_product(rows: List[Dict[str, Any]]) -> None:
    """Refuse to publish a table that does not add up.

    The executor computed these numbers, so this is not defensive noise: it is
    the difference between a bug that surfaces here and one that surfaces as a
    wrong figure in a manuscript.
    """

    level_rows = [row for row in rows if row["row_role"] == _LEVEL_ROLE]
    overall = [row for row in rows if row["row_role"] == _OVERALL_ROLE]
    if len(overall) != 1:
        raise RuntimeError(
            "exposure_outcome_distribution needs exactly one overall row"
        )
    total = overall[0]
    if sum(row["n_rows"] for row in level_rows) != total["n_rows"]:
        raise RuntimeError("declared exposure levels do not partition the cohort")
    if sum(row["outcome_events"] for row in level_rows) != total["outcome_events"]:
        raise RuntimeError("level events do not sum to the overall events")
    if (
        sum(row["outcome_observed_n"] for row in level_rows)
        != total["outcome_observed_n"]
    ):
        raise RuntimeError("level observed counts do not sum to the overall observed")
    for row in rows:
        if row["outcome_events"] > row["outcome_denominator"]:
            raise RuntimeError("more events than the denominator they are taken over")
        if row["outcome_observed_n"] + row["outcome_missing_n"] != row["n_rows"]:
            raise RuntimeError("observed plus missing does not equal the row count")
        rate = _finite(row["outcome_rate_pct"])
        low = _finite(row["ci_low_pct"])
        high = _finite(row["ci_high_pct"])
        if rate is not None and low is not None and high is not None:
            if not (low - 1e-6 <= rate <= high + 1e-6):
                raise RuntimeError("the reported rate falls outside its own interval")


def run_exposure_outcome_distribution_from_env(
    *,
    spec_payload: Dict[str, Any],
    typed_cohort_input: str | None,
) -> Dict[str, Any]:
    """Execute the declared distribution from the standard runner environment."""

    spec = ExposureOutcomeDistributionSpec.model_validate(spec_payload)
    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, cohort_path = load_step_cohort_frame(typed_cohort_input=typed_cohort_input)

    missing = [
        column
        for column in (spec.exposure, spec.outcome)
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            "Declared exposure/outcome columns are absent from the bound cohort: "
            + ", ".join(missing)
        )

    rows = _distribution_rows(frame, spec=spec)
    _verify_product(rows)

    table = pd.DataFrame(rows, columns=list(EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS))
    table_path = out_dir / "exposure_outcome_distribution.csv"
    table.to_csv(table_path, index=False)

    summary = {
        "status": "ok",
        "analysis_family": "descriptive",
        "interpretation_class": "exposure_outcome_distribution",
        "cohort_n": int(len(frame)),
        "exposure": spec.exposure,
        "outcome": spec.outcome,
        "denominator_policy": spec.denominator_policy,
        "interval_method": spec.interval_method,
        "typed_cohort_input": typed_cohort_input,
        "source_cohort": cohort_path.name,
        "source_row_count_reconciliation": {
            "source_rows": int(len(frame)),
            "analyzed_rows": int(len(frame)),
            "filtering_performed": False,
        },
        "adjusted_effect": None,
        "output_files": {EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT: table_path.name},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return summary

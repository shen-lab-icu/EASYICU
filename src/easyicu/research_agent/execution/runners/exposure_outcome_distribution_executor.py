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
denominator, its missing count, its event count and its rate with an interval,
*and* the design that produced them -- the exposure and outcome columns, the
closed level sets, the event value, the denominator, missing-data and matching
policies, the interval method and its confidence level. A renderer can
therefore draw and re-derive the whole figure from this one table, with no
second lookup into a cohort summary and no out-of-band knowledge of the spec.
That is what makes a figure step's input contract closable before its parent
has run.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ...authority.declared_levels import execution_distribution_spec
from ...schema import AnalysisStep, ExposureOutcomeDistributionSpec, _typed_level_key
from .typed_input_binding import (
    load_typed_cohort,
    run_dir_from_env,
    sole_typed_cohort_input,
)

__all__ = [
    "EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS",
    "EXPOSURE_OUTCOME_DISTRIBUTION_DESIGN_COLUMNS",
    "EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT",
    "exposure_outcome_distribution_executor_code",
    "exposure_outcome_distribution_executor_owns_step",
    "percentage",
    "run_exposure_outcome_distribution_from_env",
    "wilson_interval",
]

EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT = "table:exposure_outcome_distribution"

#: The design columns. Every row repeats them, which is the price of a product
#: that a downstream consumer can check without being told anything else.
EXPOSURE_OUTCOME_DISTRIBUTION_DESIGN_COLUMNS = (
    "exposure_column",
    "exposure_levels_declared",
    "outcome_column",
    "outcome_levels_declared",
    # A *pointer* into outcome_levels_declared, not the value itself. A CSV
    # cell cannot carry a typed scalar: ``1`` and ``"1"`` are written
    # identically and both read back as a number, which would silently undo the
    # one distinction the level policy exists to preserve. The levels array
    # survives the round trip intact, so the event is identified by position
    # within it.
    "outcome_positive_index",
    "level_match_policy",
    "denominator_policy",
    "missing_exposure_policy",
    "missing_outcome_policy",
    "interval_method",
    "confidence_level",
)

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
    # How many rows the exposure could not place. Repeated on every row, the
    # way Table 1 carries ``group_missing_excluded_n``: a denominator that
    # silently shrank is a denominator a reader cannot check.
    "missing_exposure_excluded_n",
) + EXPOSURE_OUTCOME_DISTRIBUTION_DESIGN_COLUMNS

_OVERALL_ROLE = "overall"
_LEVEL_ROLE = "exposure_level"


def _typed_cohort_input(step: AnalysisStep) -> str:
    """Return the single typed cohort input, or ``""`` when there is not one.

    An empty answer is disqualifying rather than a fallback. A step whose
    cohort arrives by bare ``COHORT_PARQUET`` has no digest, no product
    contract and no named producer for the rows it counted, so the table it
    would emit could not be bound to the plan that asked for it -- and calling
    that "deterministically owned" would be a claim the run cannot support.

    Only that last policy is this owner's own: *which* keys name the closed
    cohort product is one published vocabulary, so it is read from there rather
    than spelled out again here.
    """

    return sole_typed_cohort_input(step) or ""


def exposure_outcome_distribution_executor_owns_step(step: AnalysisStep) -> bool:
    """Own only a step whose distribution design is completely declared."""

    spec = step.exposure_outcome_distribution_spec
    if spec is None:
        return False
    return bool(
        str(step.method or "").strip().casefold() in {"descriptive", "distribution"}
        and str(step.planned_analysis_role or "").strip().casefold() == "auxiliary"
        and list(step.expected_outputs or []) == [EXPOSURE_OUTCOME_DISTRIBUTION_OUTPUT]
        and _typed_cohort_input(step)
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
    # The design as it must EXECUTE.  The Planner declares levels in the
    # host's own opaque placeholders whenever it was told a column's
    # cardinality and not its values; two recorded runs died here and were
    # rescued only by a replan that guessed ``[0, 1]``.
    spec = execution_distribution_spec(step)
    assert spec is not None  # narrowed by owns_step
    return textwrap.dedent(
        f"""
        from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
            run_exposure_outcome_distribution_from_env,
        )

        run_exposure_outcome_distribution_from_env(
            spec_payload={spec.model_dump(mode="json")!r},
            typed_cohort_input={_typed_cohort_input(step)!r},
            require_consumption_contract={
                bool(
                    [
                        contract
                        for contract in step.input_consumption_contracts
                        if contract.input_key == _typed_cohort_input(step)
                    ]
                )
            !r},
        )
        """
    ).strip()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentage(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(100.0 * numerator / denominator, 6)


def wilson_interval(
    events: int, denominator: int, *, confidence_level: float
) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval at the confidence level the spec declares.

    Both the method and its coverage are the Planner's choice, not this
    module's; the ``z`` multiplier is derived from ``confidence_level`` rather
    than written down, so there is no coverage baked into the code that a study
    never asked for. Both travel in the product, so a reader never has to guess
    what a percentage's interval means.
    """

    if denominator <= 0:
        return (None, None)
    if not (0.0 < confidence_level < 1.0):
        raise RuntimeError("confidence_level must lie strictly between 0 and 1")
    z = statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence_level) / 2.0)
    proportion = events / denominator
    centre = proportion + z * z / (2 * denominator)
    spread = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4 * denominator)) / denominator
    )
    factor = 1.0 + z * z / denominator
    low = max(0.0, (centre - spread) / factor)
    high = min(1.0, (centre + spread) / factor)
    return (round(100.0 * low, 6), round(100.0 * high, 6))


def _is_boolean(item: Any) -> bool:
    """True for a real boolean, including the numpy one pandas hands back."""

    return isinstance(item, bool) or type(item).__name__ == "bool_"


def _number(item: Any) -> Optional[float]:
    """The finite numeric value of a genuinely numeric scalar, else ``None``.

    This is the single place that decides a boolean is not a number, and it is
    load-bearing rather than defensive: ``isinstance(True, int)`` is true in
    Python, so without it a column of ``True``/``False`` answers a declared
    level of ``1``/``0`` and the study silently reports a different variable
    from the one it declared. A second copy of this guard elsewhere would make
    both removable without a test noticing, so there is only one.
    """

    if _is_boolean(item) or item is None:
        return None
    if isinstance(item, (int, float)) or type(item).__name__.startswith(
        ("int", "float", "uint")
    ):
        try:
            value = float(item)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
    return None


def _parsed_number(text: str) -> Optional[float]:
    try:
        value = float(text.strip())
    except (TypeError, ValueError, AttributeError):
        return None
    return value if math.isfinite(value) else None


def _matches_scalar(item: Any, level: Any, *, policy: str) -> bool:
    """Whether one observed value is one declared level, under ``policy``.

    No policy lets a boolean and a number match each other in either
    direction. ``numeric_string_equivalent`` only widens the *text spelling* of
    a number, which is a storage difference; a boolean is a different variable.
    """

    if _is_boolean(level):
        return _is_boolean(item) and bool(item) == bool(level)
    # No guard against a boolean *item* is needed here: a non-boolean level is
    # either numeric, where ``_number`` refuses booleans, or a string, which a
    # boolean is not. Repeating the rule would only let one copy be deleted
    # while tests stayed green on the other.
    numeric_equivalence = policy == "numeric_string_equivalent"
    if isinstance(level, (int, float)):
        number = _number(item)
        if number is not None:
            return number == float(level)
        if numeric_equivalence and isinstance(item, str):
            parsed = _parsed_number(item)
            return parsed is not None and parsed == float(level)
        return False
    if isinstance(item, str):
        if item == str(level):
            return True
        if numeric_equivalence:
            parsed_item = _parsed_number(item)
            parsed_level = _parsed_number(str(level))
            return (
                parsed_item is not None
                and parsed_level is not None
                and parsed_item == parsed_level
            )
        return False
    if numeric_equivalence:
        number = _number(item)
        parsed_level = _parsed_number(str(level))
        return (
            number is not None and parsed_level is not None and number == parsed_level
        )
    return False


def _matches(values: pd.Series, level: Any, *, policy: str) -> pd.Series:
    """Mask of the rows whose observed value is the declared ``level``."""

    return values.astype("object").map(
        lambda item: _matches_scalar(item, level, policy=policy)
    )


def _closed_level_masks(
    values: pd.Series,
    levels: List[Any],
    *,
    policy: str,
    column: str,
    role: str,
    observed: pd.Series,
) -> Dict[Tuple[str, str], pd.Series]:
    """Partition observed values across a closed level set, or refuse.

    Two failures are possible and both are fatal. A value matching *two*
    declared levels means the declaration is ambiguous, so every count built
    from it would be arbitrary. A non-missing value matching *none* of them
    means the study met data it never described -- and for an outcome that is
    the dangerous one, because an undeclared value is observed, is not the
    event, and would therefore be counted as a non-event, deflating the rate
    with nothing downstream able to see it.

    The refusal reports how many rows and how many distinct values were
    undeclared, and not the values themselves: the count is what makes the
    failure actionable, while the values are cohort data and a mis-declared
    column could put real measurements into a log.
    """

    masks: Dict[Tuple[str, str], pd.Series] = {}
    for level in levels:
        masks[_typed_level_key(level)] = _matches(values, level, policy=policy)

    counted = None
    for mask in masks.values():
        counted = mask.astype(int) if counted is None else counted + mask.astype(int)
    if counted is None:  # unreachable: the spec requires >= 2 levels
        raise RuntimeError(f"exposure_outcome_distribution declared no {role} levels")
    ambiguous = int((counted > 1).sum())
    if ambiguous:
        raise RuntimeError(
            f"{ambiguous} rows of {column!r} match more than one declared {role} "
            "level; the declared levels must be mutually exclusive"
        )
    undeclared = observed & (counted == 0)
    undeclared_rows = int(undeclared.sum())
    if undeclared_rows:
        distinct = int(values[undeclared].astype("object").nunique(dropna=False))
        raise RuntimeError(
            f"{undeclared_rows} observed rows of {column!r} carry a value that "
            f"is not one of the declared {role} levels ({distinct} distinct "
            "undeclared values). The closed level set is what stops an "
            "undeclared value from being silently counted as a non-event, so "
            "the step fails rather than reporting a deflated rate"
        )
    return masks


def _design_fields(spec: ExposureOutcomeDistributionSpec) -> Dict[str, Any]:
    """The declaration, in the form every product row carries."""

    return {
        "exposure_column": spec.exposure,
        "exposure_levels_declared": json.dumps(
            spec.exposure_levels, ensure_ascii=False
        ),
        "outcome_column": spec.outcome,
        "outcome_levels_declared": json.dumps(spec.outcome_levels, ensure_ascii=False),
        "outcome_positive_index": [
            _typed_level_key(level) for level in spec.outcome_levels
        ].index(_typed_level_key(spec.outcome_positive_value)),
        "level_match_policy": spec.level_match_policy,
        "denominator_policy": spec.denominator_policy,
        "missing_exposure_policy": spec.missing_exposure_policy,
        "missing_outcome_policy": spec.missing_outcome_policy,
        "interval_method": spec.interval_method,
        "confidence_level": spec.confidence_level,
    }


def _distribution_rows(
    frame: pd.DataFrame,
    *,
    spec: ExposureOutcomeDistributionSpec,
) -> List[Dict[str, Any]]:
    exposure_values = frame[spec.exposure]
    outcome_values = frame[spec.outcome]
    observed_outcome = outcome_values.notna()
    observed_exposure = exposure_values.notna()

    # A missing exposure is refused on its own terms rather than swept into the
    # undeclared-level bucket below. Both stop the step, but they send a reader
    # to different places: one means the data holds a category the study never
    # described, the other means the study has rows it cannot group at all.
    # Reporting the second as the first sends someone hunting for a stray code
    # that does not exist.
    missing_exposure_rows = int((~observed_exposure).sum())
    if missing_exposure_rows and spec.missing_exposure_policy == "fail_closed":
        raise RuntimeError(
            f"{missing_exposure_rows} rows have no observed value for "
            f"{spec.exposure!r}; the spec declares missing_exposure_policy="
            "'fail_closed', so they are neither dropped nor pooled into a "
            "declared exposure level"
        )
    if missing_exposure_rows:
        # Complete-case on the exposure. The rows leave the frame entirely --
        # every denominator below is then taken over the same rows, which is
        # the property a reader checks -- and the count they left behind
        # travels in the product so the shrink is visible rather than inferred
        # from a total that does not add up.
        #
        # canary13 is why this option exists at all: 8 of 1000 stays had no
        # AKI stage, `fail_closed` was the ONLY value the field could take, and
        # the step died with no result the Planner had any way to avoid.
        frame = frame.loc[observed_exposure]
        exposure_values = frame[spec.exposure]
        outcome_values = frame[spec.outcome]
        observed_outcome = outcome_values.notna()
        observed_exposure = exposure_values.notna()
    exposure_masks = _closed_level_masks(
        exposure_values,
        spec.exposure_levels,
        policy=spec.level_match_policy,
        column=spec.exposure,
        role="exposure",
        observed=observed_exposure,
    )
    outcome_masks = _closed_level_masks(
        outcome_values,
        spec.outcome_levels,
        policy=spec.level_match_policy,
        column=spec.outcome,
        role="outcome",
        observed=observed_outcome,
    )
    # The event mask is *the* declared positive level's mask, not a second
    # match: the spec guarantees the positive value is one of the closed
    # levels, so reusing it makes the events and the level counts one fact.
    event = outcome_masks[_typed_level_key(spec.outcome_positive_value)]

    missing_outcome_rows = int((~observed_outcome).sum())
    if missing_outcome_rows and spec.missing_outcome_policy == "fail_closed":
        raise RuntimeError(
            f"{missing_outcome_rows} rows have no observed value for "
            f"{spec.outcome!r}; the spec declares missing_outcome_policy="
            "'fail_closed', so they are neither dropped nor counted as "
            "non-events without an explicit decision"
        )

    total_rows = int(len(frame))
    exposure_denominator = total_rows
    # Carried beside the declaration on every row, not merged into it: the
    # policy is what the plan asked for, this is what the data cost.
    design = {
        **_design_fields(spec),
        "missing_exposure_excluded_n": missing_exposure_rows,
    }

    rows: List[Dict[str, Any]] = []
    for level in spec.exposure_levels:
        mask = exposure_masks[_typed_level_key(level)]
        n_rows = int(mask.sum())
        observed_n = int((mask & observed_outcome).sum())
        missing_n = n_rows - observed_n
        events = int((mask & event).sum())
        outcome_denominator = (
            n_rows if spec.denominator_policy == "all_declared_rows" else observed_n
        )
        low, high = wilson_interval(
            events, outcome_denominator, confidence_level=spec.confidence_level
        )
        rows.append(
            {
                "row_role": _LEVEL_ROLE,
                "exposure_level": level,
                "n_rows": n_rows,
                "exposure_denominator": exposure_denominator,
                "exposure_pct": percentage(n_rows, exposure_denominator),
                "outcome_observed_n": observed_n,
                "outcome_missing_n": missing_n,
                "outcome_events": events,
                "outcome_denominator": outcome_denominator,
                "outcome_rate_pct": percentage(events, outcome_denominator),
                "ci_low_pct": low,
                "ci_high_pct": high,
                **design,
            }
        )

    overall_observed = int(observed_outcome.sum())
    overall_events = int(event.sum())
    overall_denominator = (
        total_rows
        if spec.denominator_policy == "all_declared_rows"
        else overall_observed
    )
    low, high = wilson_interval(
        overall_events, overall_denominator, confidence_level=spec.confidence_level
    )
    rows.append(
        {
            "row_role": _OVERALL_ROLE,
            "exposure_level": None,
            "n_rows": total_rows,
            "exposure_denominator": exposure_denominator,
            "exposure_pct": percentage(total_rows, exposure_denominator),
            "outcome_observed_n": overall_observed,
            "outcome_missing_n": total_rows - overall_observed,
            "outcome_events": overall_events,
            "outcome_denominator": overall_denominator,
            "outcome_rate_pct": percentage(overall_events, overall_denominator),
            "ci_low_pct": low,
            "ci_high_pct": high,
            **design,
        }
    )
    return rows


def _verify_product(
    rows: List[Dict[str, Any]], *, spec: ExposureOutcomeDistributionSpec
) -> None:
    """Refuse to publish a table that does not add up.

    The executor computed these numbers, so this is not defensive noise: it is
    the difference between a bug that surfaces here and one that surfaces as a
    wrong figure in a manuscript. Every published quantity is re-derived from
    the counts beside it, so a wrong percentage cannot pass merely by being
    plausible.
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
    if (
        sum(row["outcome_missing_n"] for row in level_rows)
        != total["outcome_missing_n"]
    ):
        raise RuntimeError("level missing counts do not sum to the overall missing")
    for row in rows:
        if row["outcome_events"] > row["outcome_denominator"]:
            raise RuntimeError("more events than the denominator they are taken over")
        if row["outcome_observed_n"] + row["outcome_missing_n"] != row["n_rows"]:
            raise RuntimeError("observed plus missing does not equal the row count")
        expected_denominator = (
            row["n_rows"]
            if spec.denominator_policy == "all_declared_rows"
            else row["outcome_observed_n"]
        )
        if row["outcome_denominator"] != expected_denominator:
            raise RuntimeError(
                "the outcome denominator does not follow the declared "
                f"denominator_policy={spec.denominator_policy!r}"
            )
        if row["exposure_pct"] != percentage(
            row["n_rows"], row["exposure_denominator"]
        ):
            raise RuntimeError("an exposure percentage is not its own counts")
        if row["outcome_rate_pct"] != percentage(
            row["outcome_events"], row["outcome_denominator"]
        ):
            raise RuntimeError("an outcome rate is not its own events over denominator")
        if (row["ci_low_pct"], row["ci_high_pct"]) != wilson_interval(
            row["outcome_events"],
            row["outcome_denominator"],
            confidence_level=spec.confidence_level,
        ):
            raise RuntimeError("an interval is not the declared method at that level")
        rate = _finite(row["outcome_rate_pct"])
        low = _finite(row["ci_low_pct"])
        high = _finite(row["ci_high_pct"])
        if rate is not None and low is not None and high is not None:
            if not (low - 1e-6 <= rate <= high + 1e-6):
                raise RuntimeError("the reported rate falls outside its own interval")


def run_exposure_outcome_distribution_from_env(
    *,
    spec_payload: Dict[str, Any],
    typed_cohort_input: str,
    require_consumption_contract: bool = False,
) -> Dict[str, Any]:
    """Execute the declared distribution from the standard runner environment."""

    spec = ExposureOutcomeDistributionSpec.model_validate(spec_payload)
    if not str(typed_cohort_input or "").strip():
        raise RuntimeError(
            "exposure_outcome_distribution requires an exact typed cohort "
            "binding; it does not read an unbound cohort from the environment, "
            "because a table counted from unverified bytes cannot be bound to "
            "the plan that asked for it"
        )
    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, cohort_path = load_typed_cohort(
        input_key=typed_cohort_input,
        run_dir=run_dir_from_env(),
        resolved_inputs_path=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]).resolve(),
        require_consumption_contract=require_consumption_contract,
    )

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
    _verify_product(rows, spec=spec)

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
        "missing_outcome_policy": spec.missing_outcome_policy,
        "level_match_policy": spec.level_match_policy,
        "interval_method": spec.interval_method,
        "confidence_level": spec.confidence_level,
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

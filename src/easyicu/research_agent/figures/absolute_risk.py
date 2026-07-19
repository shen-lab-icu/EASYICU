"""Render an agent-planned absolute-risk and source-prevalence figure.

This is a sealed, rendering-only adapter for one controlled parent method.  It
parses exactly two digest-bound parent tables and the parent summary.  The
adapter does not read a cohort, discover sibling files, choose an exposure or
outcome, define groups, or fit a statistical model.  Any row family that is not
part of the closed parent schema makes the adapter decline the render.
"""

from __future__ import annotations

import io
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from ..contracts.declared_product import read_digest_bound_artifact_snapshot
from .publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)


REPAIR_ID = "absolute_risk_incidence_prevalence_publication_bundle_v1"
CONTROLLED_METHOD = "binary_outcome_incidence_and_absolute_risk"
_OUTCOME_TABLE = "outcome_incidence.csv"
_PREVALENCE_TABLE = "exposure_prevalence.csv"
_SNAPSHOT_NAMES = {"step_summary.json", _OUTCOME_TABLE, _PREVALENCE_TABLE}

_RISK_REQUIRED_COLUMNS = {
    "estimate_type",
    "outcome",
    "outcome_definition",
    "group_type",
    "group_value",
    "source_status",
    "group_definition",
    "stratum_n",
    "denominator_type",
    "denominator_n",
    "n",
    "event_n",
    "non_event_n",
    "outcome_risk",
    "outcome_risk_percentage",
    "outcome_risk_fraction",
    "ci_low",
    "ci_high",
    "ci_method",
    "ci_alpha",
    "percentage_of_locked_cohort",
    "fraction_of_locked_cohort",
    "risk_status",
}
_PREVALENCE_REQUIRED_COLUMNS = {
    "estimate_type",
    "variable",
    "group_type",
    "group_value",
    "group_label",
    "source_status",
    "denominator_type",
    "denominator_n",
    "n",
    "event_n",
    "non_event_n",
    "percentage_of_denominator",
    "fraction_of_denominator",
    "percentage_of_locked_cohort",
    "fraction_of_locked_cohort",
    "group_definition",
    "summary_status",
}


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _finite(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _nonnegative_integer(value: Any) -> Optional[int]:
    parsed = _finite(value)
    if parsed is None or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def _missing(value: Any) -> bool:
    return value is None or pd.isna(value) or not str(value).strip()


def _same(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=tolerance)


def _read_csv(source: Path | bytes, required: set[str]) -> Optional[pd.DataFrame]:
    reader: Path | io.BytesIO
    if isinstance(source, bytes):
        reader = io.BytesIO(source)
    else:
        path = Path(source)
        if path.is_symlink() or not path.is_file():
            return None
        reader = path
    try:
        frame = pd.read_csv(reader)
    except Exception:
        return None
    if frame.empty or not required.issubset(set(map(str, frame.columns))):
        return None
    return frame


def _raw_strings(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _unique_nonempty(series: pd.Series) -> Optional[str]:
    values = _raw_strings(series)
    if values.eq("").any() or values.nunique(dropna=False) != 1:
        return None
    return str(values.iloc[0])


def _embedded_rows_match(
    summary: Mapping[str, Any],
    *,
    key: str,
    frame: pd.DataFrame,
) -> bool:
    """Reject a declared table mirror that disagrees with the sealed CSV.

    Embedded row mirrors are optional legacy metadata.  When a parent emits one,
    however, silently preferring either copy would make the renderer choose
    between two contradictory scientific products.  Row order is material
    because it is also the source-row provenance used by the figure export.
    """

    if key not in summary:
        return True
    raw_rows = summary.get(key)
    if not isinstance(raw_rows, list) or not raw_rows:
        return False
    if any(not isinstance(row, Mapping) for row in raw_rows):
        return False
    embedded = pd.DataFrame(raw_rows)
    if len(embedded) != len(frame) or set(embedded.columns) != set(frame.columns):
        return False
    embedded = embedded.loc[:, list(frame.columns)].reset_index(drop=True)
    observed = frame.reset_index(drop=True)
    for column in observed.columns:
        for left, right in zip(observed[column], embedded[column]):
            left_missing = left is None or (
                not isinstance(left, (Mapping, list, tuple, set)) and bool(pd.isna(left))
            )
            right_missing = right is None or (
                not isinstance(right, (Mapping, list, tuple, set)) and bool(pd.isna(right))
            )
            if left_missing or right_missing:
                if left_missing and right_missing:
                    continue
                return False
            if isinstance(left, str) or isinstance(right, str):
                if not isinstance(left, str) or not isinstance(right, str) or left != right:
                    return False
                continue
            left_number = _finite(left)
            right_number = _finite(right)
            if left_number is not None and right_number is not None:
                if not _same(left_number, right_number):
                    return False
            elif str(left) != str(right):
                return False
    return True


def _wrapped(value: Any, *, width: int = 23) -> str:
    label = str(value or "").strip()
    lines = textwrap.wrap(
        label,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines) if lines else label


def _definition_mentions_identity(definition: Any, identity: str) -> bool:
    """Match one declared scientific identifier as a complete token."""

    value = str(definition or "")
    return (
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(identity)}(?![A-Za-z0-9_])",
            value,
        )
        is not None
    )


def _summary_declared_files(summary: Mapping[str, Any]) -> set[str]:
    declared: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, str) and Path(value).name == value:
            declared.add(value)

    visit(summary.get("output_files"))
    return declared


def _validate_count_row(
    row: pd.Series,
    *,
    locked_denominator: int,
    risk_row: bool,
) -> Optional[tuple[int, int, int, int]]:
    denominator = _nonnegative_integer(row.get("denominator_n"))
    n = _nonnegative_integer(row.get("n"))
    event_n = _nonnegative_integer(row.get("event_n"))
    non_event_n = _nonnegative_integer(row.get("non_event_n"))
    if None in {denominator, n, event_n, non_event_n}:
        return None
    assert denominator is not None
    assert n is not None
    assert event_n is not None
    assert non_event_n is not None
    if event_n + non_event_n != n or n > denominator:
        return None
    if not str(row.get("denominator_type") or "").strip():
        return None

    if risk_row:
        stratum_n = _nonnegative_integer(row.get("stratum_n"))
        if stratum_n != n or n != denominator:
            return None
        risk_fields = (
            "outcome_risk",
            "outcome_risk_percentage",
            "outcome_risk_fraction",
        )
        if denominator == 0:
            if (
                any(
                    not _missing(row.get(field))
                    for field in (
                        *risk_fields,
                        "ci_low",
                        "ci_high",
                        "ci_method",
                        "ci_alpha",
                    )
                )
                or _normalise(row.get("risk_status"))
                != "not_estimable_zero_denominator"
            ):
                return None
        else:
            risk = _finite(row.get("outcome_risk"))
            percentage = _finite(row.get("outcome_risk_percentage"))
            fraction = _finite(row.get("outcome_risk_fraction"))
            expected = event_n / denominator
            if (
                risk is None
                or percentage is None
                or fraction is None
                or not 0 <= risk <= 1
                or not _same(risk, expected)
                or not _same(fraction, expected)
                or not _same(percentage, 100.0 * expected, tolerance=1e-6)
                or _normalise(row.get("risk_status")) != "available"
            ):
                return None
            ci_low = _finite(row.get("ci_low"))
            ci_high = _finite(row.get("ci_high"))
            ci_present = ci_low is not None or ci_high is not None
            # A non-empty binomial stratum is estimable even when it has zero
            # events.  Its uncertainty interval must therefore be explicit;
            # treating a zero-event row like a zero-denominator row would erase
            # the upper confidence bound in the publication figure.
            if ci_low is None or ci_high is None:
                return None
            alpha = _finite(row.get("ci_alpha"))
            if (
                not ci_present
                or not 0 <= ci_low <= risk <= ci_high <= 1
                or _missing(row.get("ci_method"))
                or alpha is None
                or not 0 < alpha < 1
            ):
                return None

        locked_percentage = row.get("percentage_of_locked_cohort")
        locked_fraction = row.get("fraction_of_locked_cohort")
        if n == 0:
            for value in (locked_percentage, locked_fraction):
                if not _missing(value) and _finite(value) != 0:
                    return None
        else:
            percentage = _finite(locked_percentage)
            fraction = _finite(locked_fraction)
            if (
                percentage is None
                or fraction is None
                or not _same(fraction, n / locked_denominator)
                or not _same(
                    percentage,
                    100.0 * n / locked_denominator,
                    tolerance=1e-6,
                )
            ):
                return None
    return denominator, n, event_n, non_event_n


def _validate_prevalence_rates(
    row: pd.Series,
    *,
    denominator: int,
    count: int,
    locked_denominator: int,
) -> bool:
    for percentage_field, fraction_field, rate_denominator in (
        ("percentage_of_denominator", "fraction_of_denominator", denominator),
        (
            "percentage_of_locked_cohort",
            "fraction_of_locked_cohort",
            locked_denominator,
        ),
    ):
        percentage_raw = row.get(percentage_field)
        fraction_raw = row.get(fraction_field)
        if count == 0:
            for value in (percentage_raw, fraction_raw):
                if not _missing(value) and _finite(value) != 0:
                    return False
            continue
        percentage = _finite(percentage_raw)
        fraction = _finite(fraction_raw)
        if (
            percentage is None
            or fraction is None
            or not _same(fraction, count / rate_denominator)
            or not _same(
                percentage,
                100.0 * count / rate_denominator,
                tolerance=1e-6,
            )
        ):
            return False
    return True


@dataclass(frozen=True)
class AbsoluteRiskInputs:
    risk_rows: pd.DataFrame
    prevalence_rows: pd.DataFrame
    risk_group_type: str
    risk_group_values: tuple[str, ...]
    risk_group_labels: tuple[str, ...]
    status_schema: tuple[str, ...]
    locked_denominator: int
    valid_observed_n: int
    outcome_label: str


def prepare_absolute_risk_inputs(
    parent_summary: Mapping[str, Any],
    outcome_source: Path | bytes,
    prevalence_source: Path | bytes,
    *,
    expected_primary_exposure: Optional[str] = None,
    expected_target_outcome: Optional[str] = None,
) -> Optional[AbsoluteRiskInputs]:
    """Validate the complete two-table parent schema before renderer routing.

    ``outcome_source`` and ``prevalence_source`` should normally be immutable
    byte payloads from a host-verified artifact snapshot.  ``Path`` remains
    supported for focused tests, but the public render entry point never uses
    unsealed paths.
    """

    if not isinstance(parent_summary, Mapping):
        return None
    if _normalise(parent_summary.get("method")) != CONTROLLED_METHOD:
        return None
    if _normalise(parent_summary.get("analysis_status")) != "ok":
        return None
    if not {_OUTCOME_TABLE, _PREVALENCE_TABLE} <= _summary_declared_files(parent_summary):
        return None
    if any(
        parent_summary.get(flag) is not True
        for flag in (
            "outcome_data_available",
            "source_status_available",
            "measurement_provenance_ok",
        )
    ):
        return None

    schema_raw = parent_summary.get("source_status_schema")
    counts_raw = parent_summary.get("source_status_counts")
    if not isinstance(schema_raw, list) or not isinstance(counts_raw, Mapping):
        return None
    status_schema = tuple(str(value).strip() for value in schema_raw)
    if (
        not status_schema
        or any(not value for value in status_schema)
        or len(set(status_schema)) != len(status_schema)
        or len({_normalise(value) for value in status_schema}) != len(status_schema)
    ):
        return None
    valid_statuses = [
        status for status in status_schema if _normalise(status) == "valid_observed"
    ]
    if len(valid_statuses) != 1:
        return None
    valid_status = valid_statuses[0]
    declared_count_keys = {str(key).strip() for key in counts_raw}
    if declared_count_keys != set(status_schema):
        return None
    declared_counts = {
        status: _nonnegative_integer(counts_raw.get(status)) for status in status_schema
    }
    if any(value is None for value in declared_counts.values()):
        return None

    outcome = _read_csv(outcome_source, _RISK_REQUIRED_COLUMNS)
    prevalence = _read_csv(prevalence_source, _PREVALENCE_REQUIRED_COLUMNS)
    if outcome is None or prevalence is None:
        return None
    if not _embedded_rows_match(
        parent_summary,
        key="outcome_incidence_rows",
        frame=outcome,
    ) or not _embedded_rows_match(
        parent_summary,
        key="exposure_prevalence_rows",
        frame=prevalence,
    ):
        return None
    outcome = outcome.copy()
    prevalence = prevalence.copy()
    outcome["__source_row_index"] = outcome.index.astype(int)
    prevalence["__source_row_index"] = prevalence.index.astype(int)

    if not _raw_strings(outcome["estimate_type"]).map(_normalise).eq("outcome_risk").all():
        return None
    outcome_group_types = _raw_strings(outcome["group_type"])
    if outcome_group_types.eq("").any():
        return None
    normalized_outcome_groups = outcome_group_types.map(_normalise)
    other_groups = tuple(
        dict.fromkeys(
            raw
            for raw, normalized in zip(outcome_group_types, normalized_outcome_groups)
            if normalized not in {"overall", "source_status"}
        )
    )
    if len(other_groups) != 1:
        return None
    risk_group_type = other_groups[0]
    if sum(_normalise(value) == _normalise(risk_group_type) for value in other_groups) != 1:
        return None
    allowed_risk_groups = {"overall", "source_status", _normalise(risk_group_type)}
    if set(normalized_outcome_groups) != allowed_risk_groups:
        return None

    overall_rows = outcome.loc[normalized_outcome_groups.eq("overall")].copy()
    source_risk_rows = outcome.loc[normalized_outcome_groups.eq("source_status")].copy()
    risk_rows = outcome.loc[
        normalized_outcome_groups.eq(_normalise(risk_group_type))
    ].copy()
    if len(overall_rows) != 1 or len(source_risk_rows) != len(status_schema) or len(risk_rows) < 2:
        return None
    if _normalise(overall_rows.iloc[0]["group_value"]) != "overall":
        return None
    if not _missing(overall_rows.iloc[0].get("source_status")):
        return None

    source_risk_statuses = _raw_strings(source_risk_rows["source_status"])
    source_risk_values = _raw_strings(source_risk_rows["group_value"])
    if (
        source_risk_statuses.duplicated().any()
        or set(source_risk_statuses) != set(status_schema)
        or not source_risk_statuses.equals(source_risk_values)
    ):
        return None
    source_risk_rows = (
        source_risk_rows.assign(
            __status_order=source_risk_statuses.map(
                {status: index for index, status in enumerate(status_schema)}
            )
        )
        .sort_values("__status_order")
        .drop(columns="__status_order")
    )

    risk_values = _raw_strings(risk_rows["group_value"])
    risk_definitions = _raw_strings(risk_rows["group_definition"])
    risk_status_values = _raw_strings(risk_rows["source_status"])
    if (
        _raw_strings(outcome["group_definition"]).eq("").any()
        or risk_values.eq("").any()
        or risk_values.duplicated().any()
        or risk_definitions.eq("").any()
        or not risk_status_values.eq(valid_status).all()
    ):
        return None
    outcome_identity = _unique_nonempty(outcome["outcome"])
    summary_outcome = str(parent_summary.get("target_outcome") or "").strip()
    host_outcome = (
        str(expected_target_outcome).strip()
        if expected_target_outcome is not None
        else None
    )
    if (
        outcome_identity is None
        or not summary_outcome
        or outcome_identity != summary_outcome
        or (host_outcome is not None and summary_outcome != host_outcome)
        or _unique_nonempty(outcome["outcome_definition"]) is None
    ):
        return None
    outcome_label = outcome_identity

    # The prevalence table has exactly three structural row families: the
    # complete source-status partition, one valid-observed summary row, and the
    # same planned grouping rendered in the risk panel.  No fourth family is
    # silently discarded.
    prevalence_group_types = _raw_strings(prevalence["group_type"])
    prevalence_estimates = _raw_strings(prevalence["estimate_type"])
    if (
        prevalence_group_types.eq("").any()
        or prevalence_estimates.eq("").any()
        or _raw_strings(prevalence["group_definition"]).eq("").any()
        or _raw_strings(prevalence["group_label"]).eq("").any()
        or _raw_strings(prevalence["variable"]).eq("").any()
        or not _raw_strings(prevalence["summary_status"])
        .map(_normalise)
        .eq("available")
        .all()
    ):
        return None
    summary_exposure = str(parent_summary.get("primary_exposure") or "").strip()
    host_exposure = (
        str(expected_primary_exposure).strip()
        if expected_primary_exposure is not None
        else None
    )
    if not summary_exposure or (
        host_exposure is not None and summary_exposure != host_exposure
    ):
        return None
    normalized_prevalence_groups = prevalence_group_types.map(_normalise)
    normalized_prevalence_estimates = prevalence_estimates.map(_normalise)
    source_mask = normalized_prevalence_groups.eq("source_status")
    source_estimate_mask = normalized_prevalence_estimates.eq(
        "source_status_prevalence"
    )
    if not source_mask.equals(source_estimate_mask):
        return None
    grouped_mask = normalized_prevalence_groups.eq(_normalise(risk_group_type))
    if not prevalence_group_types.loc[grouped_mask].eq(risk_group_type).all():
        return None
    remaining_mask = ~(source_mask | grouped_mask)
    if remaining_mask.sum() != 1 or grouped_mask.sum() != len(risk_rows):
        return None
    remaining_row = prevalence.loc[remaining_mask].iloc[0]
    valid_summary_group = _normalise(remaining_row.get("group_type"))
    valid_summary_estimate = _normalise(remaining_row.get("estimate_type"))
    if (
        not (
            valid_summary_group == "valid_observed"
            or valid_summary_group.startswith("valid_observed_")
        )
        or valid_summary_estimate != f"{valid_summary_group}_distribution"
        or _normalise(remaining_row.get("group_value")) != "valid_observed"
        or str(remaining_row.get("group_label") or "").strip()
        != str(remaining_row.get("group_value") or "").strip()
        or str(remaining_row.get("variable") or "").strip()
        != str(remaining_row.get("group_type") or "").strip()
        or str(remaining_row.get("source_status") or "").strip() != valid_status
    ):
        return None
    allowed_prevalence_groups = {
        "source_status",
        _normalise(risk_group_type),
        valid_summary_group,
    }
    if set(normalized_prevalence_groups) != allowed_prevalence_groups:
        return None
    grouped_estimate_types = set(normalized_prevalence_estimates.loc[grouped_mask])
    if grouped_estimate_types != {f"{_normalise(risk_group_type)}_prevalence"}:
        return None

    prevalence_rows = prevalence.loc[source_mask].copy()
    prevalence_statuses = _raw_strings(prevalence_rows["source_status"])
    prevalence_values = _raw_strings(prevalence_rows["group_value"])
    prevalence_labels = _raw_strings(prevalence_rows["group_label"])
    if (
        len(prevalence_rows) != len(status_schema)
        or prevalence_statuses.duplicated().any()
        or set(prevalence_statuses) != set(status_schema)
        or not prevalence_statuses.equals(prevalence_values)
        or prevalence_labels.tolist() != prevalence_values.tolist()
        or _unique_nonempty(prevalence_rows["variable"]) is None
    ):
        return None
    prevalence_rows = (
        prevalence_rows.assign(
            __status_order=prevalence_statuses.map(
                {status: index for index, status in enumerate(status_schema)}
            )
        )
        .sort_values("__status_order")
        .drop(columns="__status_order")
    )

    locked_denominators = [
        _nonnegative_integer(value) for value in prevalence_rows["denominator_n"]
    ]
    if (
        any(value is None or value <= 0 for value in locked_denominators)
        or len(set(locked_denominators)) != 1
    ):
        return None
    locked_denominator = int(locked_denominators[0])  # type: ignore[arg-type]
    summary_locked_n = parent_summary.get("locked_cohort_n")
    if summary_locked_n is not None and _nonnegative_integer(summary_locked_n) != locked_denominator:
        return None

    source_counts: dict[str, tuple[int, int, int]] = {}
    for status, (_, row) in zip(status_schema, prevalence_rows.iterrows()):
        validated = _validate_count_row(
            row,
            locked_denominator=locked_denominator,
            risk_row=False,
        )
        if validated is None:
            return None
        denominator, n, event_n, non_event_n = validated
        if denominator != locked_denominator or declared_counts[status] != n:
            return None
        if not _validate_prevalence_rates(
            row,
            denominator=denominator,
            count=n,
            locked_denominator=locked_denominator,
        ):
            return None
        source_counts[status] = (n, event_n, non_event_n)
    if sum(value[0] for value in source_counts.values()) != locked_denominator:
        return None
    valid_observed_n, valid_event_n, valid_non_event_n = source_counts[valid_status]
    if valid_observed_n <= 0:
        return None

    # Any top-level generic ``valid_observed_*_n`` declaration must agree with
    # the explicit source-status row; the renderer never guesses which subject
    # a differently named field represents.
    declared_observed_values: list[int] = []
    for raw_key, raw_value in parent_summary.items():
        key = _normalise(raw_key)
        if key == "valid_observed_n" or (
            key.startswith("valid_observed_") and key.endswith("_n")
        ):
            parsed = _nonnegative_integer(raw_value)
            if parsed is None:
                return None
            declared_observed_values.append(parsed)
    if declared_observed_values and any(
        value != valid_observed_n for value in declared_observed_values
    ):
        return None

    for _, row in outcome.iterrows():
        if _validate_count_row(
            row,
            locked_denominator=locked_denominator,
            risk_row=True,
        ) is None:
            return None
    overall = overall_rows.iloc[0]
    if (
        _nonnegative_integer(overall["n"]) != locked_denominator
        or _nonnegative_integer(overall["event_n"])
        != sum(value[1] for value in source_counts.values())
        or _nonnegative_integer(overall["non_event_n"])
        != sum(value[2] for value in source_counts.values())
    ):
        return None
    for status, (_, row) in zip(status_schema, source_risk_rows.iterrows()):
        if (
            _nonnegative_integer(row["n"]),
            _nonnegative_integer(row["event_n"]),
            _nonnegative_integer(row["non_event_n"]),
        ) != source_counts[status]:
            return None

    grouped_prevalence = prevalence.loc[grouped_mask].copy()
    grouped_values = _raw_strings(grouped_prevalence["group_value"])
    grouped_labels = _raw_strings(grouped_prevalence["group_label"])
    grouped_variables = _raw_strings(grouped_prevalence["variable"])
    grouped_definitions = _raw_strings(grouped_prevalence["group_definition"])
    source_risk_definitions = _raw_strings(source_risk_rows["group_definition"])
    source_prevalence_definitions = _raw_strings(
        prevalence_rows["group_definition"]
    )
    if (
        grouped_values.tolist() != risk_values.tolist()
        or grouped_labels.tolist() != risk_values.tolist()
        or not grouped_variables.eq(risk_group_type).all()
        or grouped_definitions.tolist() != risk_definitions.tolist()
        or source_prevalence_definitions.tolist()
        != source_risk_definitions.tolist()
        or not _raw_strings(grouped_prevalence["source_status"])
        .eq(valid_status)
        .all()
    ):
        return None
    if not all(
        _definition_mentions_identity(definition, summary_exposure)
        for definition in risk_definitions
    ):
        return None
    for (_, risk_row), (_, prevalence_row) in zip(
        risk_rows.iterrows(), grouped_prevalence.iterrows()
    ):
        risk_counts = tuple(
            _nonnegative_integer(risk_row[field])
            for field in ("n", "event_n", "non_event_n")
        )
        prevalence_counts = tuple(
            _nonnegative_integer(prevalence_row[field])
            for field in ("n", "event_n", "non_event_n")
        )
        if risk_counts != prevalence_counts or None in risk_counts:
            return None
        denominator = _nonnegative_integer(prevalence_row["denominator_n"])
        if denominator != valid_observed_n:
            return None
        if not _validate_prevalence_rates(
            prevalence_row,
            denominator=denominator,
            count=int(prevalence_counts[0]),  # type: ignore[arg-type]
            locked_denominator=locked_denominator,
        ):
            return None
    if (
        sum(_nonnegative_integer(value) or 0 for value in risk_rows["n"])
        != valid_observed_n
        or sum(_nonnegative_integer(value) or 0 for value in risk_rows["event_n"])
        != valid_event_n
        or sum(_nonnegative_integer(value) or 0 for value in risk_rows["non_event_n"])
        != valid_non_event_n
    ):
        return None

    valid_summary_counts = _validate_count_row(
        remaining_row,
        locked_denominator=locked_denominator,
        risk_row=False,
    )
    if valid_summary_counts is None:
        return None
    summary_denominator, summary_n, summary_event_n, summary_non_event_n = (
        valid_summary_counts
    )
    if (
        summary_denominator != valid_observed_n
        or summary_n != valid_observed_n
        or summary_event_n != valid_event_n
        or summary_non_event_n != valid_non_event_n
        or not _validate_prevalence_rates(
            remaining_row,
            denominator=valid_observed_n,
            count=valid_observed_n,
            locked_denominator=locked_denominator,
        )
    ):
        return None

    return AbsoluteRiskInputs(
        risk_rows=risk_rows,
        prevalence_rows=prevalence_rows,
        risk_group_type=risk_group_type,
        risk_group_values=tuple(risk_values),
        risk_group_labels=tuple(grouped_labels),
        status_schema=status_schema,
        locked_denominator=locked_denominator,
        valid_observed_n=valid_observed_n,
        outcome_label=outcome_label,
    )


def _snapshot_from_inputs(
    *,
    parent_out: Path,
    preverified_parent_digests: Optional[Mapping[str, str]],
    preverified_parent_artifacts: Optional[Mapping[str, bytes]],
) -> Optional[dict[str, bytes]]:
    if preverified_parent_digests is not None and preverified_parent_artifacts is not None:
        return None
    if preverified_parent_artifacts is not None:
        if {str(name) for name in preverified_parent_artifacts} != _SNAPSHOT_NAMES:
            return None
        if any(not isinstance(payload, bytes) for payload in preverified_parent_artifacts.values()):
            return None
        return {str(name): payload for name, payload in preverified_parent_artifacts.items()}
    if preverified_parent_digests is None:
        return None
    if {str(name) for name in preverified_parent_digests} != _SNAPSHOT_NAMES:
        return None
    try:
        return read_digest_bound_artifact_snapshot(
            parent_out=parent_out,
            artifact_digests=preverified_parent_digests,
        )
    except ValueError:
        return None


def render_absolute_risk_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    preverified_parent_artifacts: Optional[Mapping[str, bytes]] = None,
    preverified_parent_digests: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Render the exact direct parent's digest-bound two-table product."""

    step_id = str(current_step_id).strip()
    parent_step_id = step_id.removesuffix("_figure")
    if (
        not parent_step_id
        or parent_step_id == step_id
        or Path(parent_step_id).name != parent_step_id
        or Path(step_id).name != step_id
    ):
        return None
    parent_out = Path(run_dir) / "steps" / parent_step_id / "outputs"
    snapshot = _snapshot_from_inputs(
        parent_out=parent_out,
        preverified_parent_digests=preverified_parent_digests,
        preverified_parent_artifacts=preverified_parent_artifacts,
    )
    if snapshot is None:
        return None
    try:
        summary = json.loads(snapshot["step_summary.json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(summary, Mapping):
        return None
    prepared = prepare_absolute_risk_inputs(
        summary,
        snapshot[_OUTCOME_TABLE],
        snapshot[_PREVALENCE_TABLE],
    )
    if prepared is None:
        return None

    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    risk_source = output_root / "absolute_risk_panel_source_data.csv"
    prevalence_source = output_root / "source_status_prevalence_panel_source_data.csv"

    risk_export = prepared.risk_rows.copy()
    source_indices = risk_export.pop("__source_row_index").astype(int)
    risk_export.insert(0, "source_row_index", source_indices)
    risk_export.insert(1, "source_table", _OUTCOME_TABLE)
    risk_export.insert(
        2,
        "source_transform",
        "select_outcome_risk_rows_from_agent_planned_group",
    )
    prevalence_export = prepared.prevalence_rows.copy()
    prevalence_indices = prevalence_export.pop("__source_row_index").astype(int)
    # ``variable`` is constant across the complete source-status partition and
    # therefore is not a row key.  Preserve it as provenance while letting the
    # explicit parent-row position provide the unambiguous trace.
    prevalence_export = prevalence_export.rename(
        columns={"variable": "source_variable"}
    )
    prevalence_export.insert(0, "source_row_index", prevalence_indices)
    prevalence_export.insert(1, "source_table", _PREVALENCE_TABLE)
    prevalence_export.insert(
        2,
        "source_transform",
        "select_complete_agent_declared_source_status_partition",
    )
    risk_export.to_csv(risk_source, index=False)
    prevalence_export.to_csv(prevalence_source, index=False)

    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 96 / 25.4),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )

    risk_percent = pd.to_numeric(prepared.risk_rows["outcome_risk"]) * 100.0
    ci_low = pd.to_numeric(prepared.risk_rows["ci_low"]) * 100.0
    ci_high = pd.to_numeric(prepared.risk_rows["ci_high"]) * 100.0
    counts = pd.to_numeric(prepared.risk_rows["n"]).astype(int)
    positions = list(range(len(prepared.risk_rows)))
    ax_a.errorbar(
        risk_percent,
        positions,
        xerr=[risk_percent - ci_low, ci_high - risk_percent],
        fmt="o",
        color=palette["blue"],
        ecolor=palette["neutral"],
        elinewidth=1.0,
        capsize=2.0,
        markersize=4.2,
    )
    ax_a.set_yticks(positions)
    ax_a.set_yticklabels([_wrapped(label) for label in prepared.risk_group_labels])
    ax_a.invert_yaxis()
    upper = min(100.0, max(5.0, float(ci_high.max()) * 1.18))
    ax_a.set_xlim(0, upper)
    ax_a.set_xlabel("Absolute risk (%)")
    ax_a.set_title("Outcome risk by planned group", loc="left", pad=4)
    ax_a.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for position, estimate, count in zip(positions, risk_percent, counts):
        ax_a.text(
            min(float(estimate) + upper * 0.025, upper * 0.98),
            position,
            f"{float(estimate):.1f}%  n={int(count):,}",
            va="center",
            ha="left" if estimate < upper * 0.88 else "right",
            fontsize=6.1,
        )
    add_panel_label(ax_a, "A", x=-0.14, y=1.03)

    prevalence_counts = pd.to_numeric(prepared.prevalence_rows["n"]).astype(int)
    prevalence_percent = prevalence_counts.astype(float) * 100.0 / prepared.locked_denominator
    prevalence_positions = list(range(len(prepared.prevalence_rows)))
    bars = ax_b.barh(
        prevalence_positions,
        prevalence_percent,
        color=palette["blue_soft"],
        height=0.58,
    )
    ax_b.set_yticks(prevalence_positions)
    ax_b.set_yticklabels(
        [_wrapped(label) for label in _raw_strings(prepared.prevalence_rows["group_label"])]
    )
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Locked analysis cohort (%)")
    ax_b.set_title("Source-status prevalence", loc="left", pad=4)
    ax_b.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for bar, percentage, count in zip(bars, prevalence_percent, prevalence_counts):
        ax_b.text(
            min(float(percentage) + 1.0, 97.0),
            bar.get_y() + bar.get_height() / 2,
            f"{float(percentage):.1f}%  n={int(count):,}",
            va="center",
            ha="left" if percentage < 94 else "right",
            fontsize=6.1,
        )
    add_panel_label(ax_b, "B", x=-0.14, y=1.03)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.87, wspace=0.56)

    stem = "absolute_risk_incidence_prevalence"
    contract = make_figure_contract(
        figure_id=f"figure:{stem}",
        core_claim=(
            "The agent-planned absolute risks and complete source-status partition "
            "are rendered from two verified direct-parent tables."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=96.0,
        panels=[
            {
                "panel_id": "A",
                "title": "Outcome risk by planned group",
                "role": "descriptive_result",
                "claim": "Absolute risks and confidence intervals use the parent-defined groups.",
                "evidence_ids": [risk_source.name],
                "metadata": {
                    "source_data": [risk_source.name],
                    "planner_product_slots": ["absolute_risk"],
                },
            },
            {
                "panel_id": "B",
                "title": "Source-status prevalence",
                "role": "data_quality",
                "claim": "Every agent-declared source-status category is shown, including zero-count categories.",
                "evidence_ids": [prevalence_source.name],
                "metadata": {"source_data": [prevalence_source.name]},
            },
        ],
        source_data=[risk_source.name, prevalence_source.name],
        statistics_note=(
            "The adapter selects no cohort, exposure, outcome, group, or method; "
            "it only renders the closed, reconciled parent-table contract."
        ),
    )
    outputs = save_publication_figure(
        fig,
        output_root / stem,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
    )
    plt.close(fig)
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    rendered_summary = {
        "step_id": step_id,
        "method": "deterministic_absolute_risk_incidence_prevalence_figure",
        "analysis_family": "descriptive",
        "analysis_status": "ok",
        "status": "completed",
        "rendering_only": True,
        "repair_id": REPAIR_ID,
        "source_step_id": parent_step_id,
        "source_parent_method": CONTROLLED_METHOD,
        "source_tables": [_OUTCOME_TABLE, _PREVALENCE_TABLE],
        "source_data_files": [risk_source.name, prevalence_source.name],
        "figure_files": figure_files,
        "figure_path": f"{stem}.png",
        "figure_contract": f"{stem}.figure_contract.json",
        "output_files": {"figure:publication_figure": f"{stem}.png"},
        "warnings": [],
        "skipped": [],
        "errors": [],
    }
    (output_root / "step_summary.json").write_text(
        json.dumps(rendered_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return REPAIR_ID


__all__ = [
    "AbsoluteRiskInputs",
    "CONTROLLED_METHOD",
    "REPAIR_ID",
    "prepare_absolute_risk_inputs",
    "render_absolute_risk_bundle_from_prior_outputs",
]

"""Case-neutral reconciliation for positive-only binary event exports.

The caller owns the scientific choice of concept and columns.  This module
only validates and renders that declared choice into a complete binary event
status when the count, companion flag, and representative value agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .descriptive_inputs import measurement_provenance_receipt


@dataclass(frozen=True)
class BinaryEventPresenceResult:
    """Validated binary event values plus replayable reconciliation evidence."""

    values: pd.Series
    row_status: pd.Series
    audit: dict[str, Any]
    status_table: pd.DataFrame


@dataclass(frozen=True)
class MeasurementSourceStatusResult:
    """Validated source states for one continuous/ordinal summary column.

    The result is audit-only.  It never replaces, filters, or imputes the
    authoritative value column supplied by the Planner.
    """

    row_status: pd.Series
    provenance_receipt: dict[str, Any]
    audit: dict[str, Any]
    status_table: pd.DataFrame


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def reconcile_binary_event_presence(
    frame: pd.DataFrame,
    *,
    count_column: str,
    measured_column: str,
    representative_column: str,
) -> BinaryEventPresenceResult:
    """Validate an explicitly selected sparse-event triad.

    The positive-only representative may be missing (or explicitly zero) on
    reconciled negative rows.  Every positive row must carry representative
    value one.  Any invalid or discordant row fails closed; this helper never
    guesses a replacement column or silently drops rows.
    """

    selected = (count_column, measured_column, representative_column)
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"sparse event triad columns missing: {missing}")

    count = _numeric(frame[count_column])
    measured = _numeric(frame[measured_column])
    representative_raw = frame[representative_column]
    representative = _numeric(representative_raw)

    count_valid = (
        count.notna()
        & np.isfinite(count.to_numpy(dtype=float))
        & count.ge(0)
        & np.isclose(count, np.rint(count), rtol=0.0, atol=1e-9)
    )
    measured_valid = (
        measured.notna()
        & np.isfinite(measured.to_numpy(dtype=float))
        & measured.isin([0, 1])
    )
    pair_valid = count_valid & measured_valid
    event_present = count.gt(0)
    pair_discordant = pair_valid & measured.ne(event_present.astype(int))

    representative_coercion_invalid = representative_raw.notna() & representative.isna()
    representative_valid = ~representative_coercion_invalid & (
        representative.isna() | representative.isin([0, 1])
    )
    positive_missing = event_present & representative.ne(1)
    negative_positive = ~event_present & representative.eq(1)
    invalid = (
        ~pair_valid
        | pair_discordant
        | ~representative_valid
        | positive_missing
        | negative_positive
    )

    audit = {
        "indicator_semantics": "binary_event_presence",
        "count_column": count_column,
        "measured_column": measured_column,
        "representative_column": representative_column,
        "comparison_n": int(pair_valid.sum()),
        "invalid_pair_n": int((~pair_valid).sum()),
        "discordant_n": int(pair_discordant.sum()),
        "representative_invalid_n": int((~representative_valid).sum()),
        "representative_coercion_invalid_n": int(representative_coercion_invalid.sum()),
        "positive_representative_missing_n": int(positive_missing.sum()),
        "negative_representative_positive_n": int(negative_positive.sum()),
        "event_present_n": int(event_present.sum()),
        "event_absent_n": int((~event_present).sum()),
        "n_total": int(len(frame)),
    }
    if bool(invalid.any()):
        raise ValueError(f"sparse event triad is invalid or discordant: {audit}")

    values = event_present.astype("Int64").rename("binary_event_presence")
    row_status = pd.Series(
        np.where(event_present, "event_present", "event_absent"),
        index=frame.index,
        dtype="string",
        name="source_status",
    )
    status_table = pd.DataFrame(
        [
            {
                "source_status": label,
                "count": count_value,
                "denominator": int(len(frame)),
                "percentage": (
                    100.0 * count_value / len(frame) if len(frame) else np.nan
                ),
                "indicator_semantics": "binary_event_presence",
            }
            for label, count_value in (
                ("event_present", int(event_present.sum())),
                ("event_absent", int((~event_present).sum())),
            )
        ]
    )
    return BinaryEventPresenceResult(
        values=values,
        row_status=row_status,
        audit=audit,
        status_table=status_table,
    )


def reconcile_measurement_source_status(
    frame: pd.DataFrame,
    *,
    measured_column: str,
    count_column: str,
    value_column: str,
) -> MeasurementSourceStatusResult:
    """Return mutually exclusive source states for one declared summary.

    ``measurement_provenance_receipt`` first proves the measured/count pair.
    The value column is then used only to distinguish an observed summary,
    an unmeasured row, and a measured row whose requested summary is missing.
    A value on a proven-unmeasured row is contradictory and fails closed.
    """

    selected = (measured_column, count_column, value_column)
    if len(set(selected)) != len(selected):
        raise ValueError("measurement source-status roles require distinct columns")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"measurement source-status columns missing: {missing}")

    receipt = measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    )
    measured = pd.to_numeric(frame[measured_column], errors="raise").eq(1)
    value_present = frame[value_column].notna()

    observed = measured & value_present
    no_source = ~measured & ~value_present
    summary_missing = measured & ~value_present
    contradictory = ~measured & value_present
    audit = {
        "summary_semantics": "measurement_source_status",
        "measured_column": measured_column,
        "count_column": count_column,
        "value_column": value_column,
        "n_total": int(len(frame)),
        "valid_observed_n": int(observed.sum()),
        "no_source_n": int(no_source.sum()),
        "measured_source_present_summary_missing_n": int(summary_missing.sum()),
        "contradictory_invalid_n": int(contradictory.sum()),
    }
    if audit["contradictory_invalid_n"]:
        raise ValueError(f"measurement source status is contradictory: {audit}")

    labels = pd.Series(index=frame.index, dtype="string", name="source_status")
    labels.loc[observed] = "valid observed"
    labels.loc[no_source] = "no source"
    labels.loc[summary_missing] = "measured/source present but summary missing"
    if bool(labels.isna().any()):
        raise ValueError("measurement source status did not form a closed partition")

    counts = (
        labels.value_counts(dropna=False)
        .reindex(
            [
                "valid observed",
                "no source",
                "measured/source present but summary missing",
                "contradictory/invalid",
            ],
            fill_value=0,
        )
        .astype(int)
    )
    status_table = pd.DataFrame(
        {
            "source_status": counts.index.astype(str),
            "count": counts.to_numpy(dtype=int),
            "denominator": int(len(frame)),
            "percentage": (
                100.0 * counts.to_numpy(dtype=float) / len(frame)
                if len(frame)
                else np.full(len(counts), np.nan)
            ),
        }
    )
    return MeasurementSourceStatusResult(
        row_status=labels,
        provenance_receipt=receipt,
        audit=audit,
        status_table=status_table,
    )


__all__ = [
    "BinaryEventPresenceResult",
    "MeasurementSourceStatusResult",
    "reconcile_binary_event_presence",
    "reconcile_measurement_source_status",
]

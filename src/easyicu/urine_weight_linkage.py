"""Proof-bound linkage for urine-output and body-weight inputs.

Owner: the data layer owns this narrow decision, shared by KDIGO and the
urine-output callbacks.  Its public contract is deliberately small: when two
tables have no common identifier, return a weight only if the input itself
proves a single-entity scope.  It never invents a patient mapping or selects a
representative row from a multi-patient table.

The stable diagnostic codes are for caller logs and assessment receipts; the
function has no dependencies on score or callback implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


_IDENTIFIER_NAMES = frozenset(
    {
        "stay_id",
        "icustay_id",
        "patientunitstayid",
        "admissionid",
        "patientid",
        "caseid",
        "hadm_id",
        "subject_id",
    }
)


@dataclass(frozen=True)
class UnkeyedWeightResolution:
    """A conservative resolution for an otherwise unjoinable weight table."""

    weight: float | None
    diagnostic_code: str | None


class ConflictingKeyedWeightError(RuntimeError):
    """A keyed entity has multiple different valid weights without a selector."""

    diagnostic_code = "urine_weight_conflicting_keyed_values"

    def __init__(self, *, conflicting_entity_count: int) -> None:
        self.conflicting_entity_count = conflicting_entity_count
        super().__init__(
            "Multiple different valid weights exist for "
            f"{conflicting_entity_count} keyed entity/entities"
        )


def resolve_keyed_unique_weights(
    weight: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    weight_column: str = "weight",
) -> pd.DataFrame:
    """Return one weight per key only when the observed value is unique.

    Repeated identical observations are harmless duplicates. Different valid
    values require an explicit timestamp/clinical selection contract and are
    therefore rejected instead of being resolved by row order or an implicit
    aggregate.
    """

    ids = [column for column in id_columns if column in weight.columns]
    if not ids or weight_column not in weight.columns:
        return pd.DataFrame(columns=[*ids, weight_column])

    resolved = weight[ids + [weight_column]].copy()
    resolved[weight_column] = pd.to_numeric(
        resolved[weight_column], errors="coerce"
    )
    resolved = resolved.dropna(subset=[*ids, weight_column])
    resolved = resolved[resolved[weight_column] > 0]
    if resolved.empty:
        return resolved

    unique_counts = resolved.groupby(ids, dropna=False)[weight_column].nunique()
    conflicting = unique_counts.gt(1)
    if conflicting.any():
        raise ConflictingKeyedWeightError(
            conflicting_entity_count=int(conflicting.sum())
        )
    return resolved.drop_duplicates(ids + [weight_column]).drop_duplicates(ids)


def resolve_unkeyed_single_entity_weight(
    urine: pd.DataFrame,
    weight: pd.DataFrame,
    *,
    urine_id_columns: Sequence[str],
    weight_column: str = "weight",
) -> UnkeyedWeightResolution:
    """Return a broadcastable weight only for a proved one-entity scope.

    A missing join key is not evidence that the first row in ``weight`` belongs
    to every row in ``urine``.  We accept the exceptional one-entity input only
    when urine has exactly one complete identifier tuple, the weight table has
    no identifier at all, and there is exactly one valid numeric weight.
    """

    ids = [column for column in urine_id_columns if column in urine.columns]
    if not ids:
        return UnkeyedWeightResolution(None, "urine_entity_unproven")

    entities = urine[ids].dropna().drop_duplicates()
    if len(entities) != 1 or urine[ids].isna().any(axis=None):
        return UnkeyedWeightResolution(None, "urine_entity_unproven")

    weight_identifiers = [
        column
        for column in weight.columns
        if str(column).strip().lower() in _IDENTIFIER_NAMES
        or str(column).strip().lower().endswith("_id")
    ]
    if weight_identifiers:
        return UnkeyedWeightResolution(None, "weight_identifier_not_joinable")

    if weight_column not in weight.columns:
        return UnkeyedWeightResolution(None, "weight_value_unavailable")
    values = pd.to_numeric(weight[weight_column], errors="coerce").dropna()
    values = values[values > 0]
    if len(values) != 1:
        return UnkeyedWeightResolution(None, "weight_value_not_singleton")
    return UnkeyedWeightResolution(float(values.iloc[0]), None)


__all__ = [
    "ConflictingKeyedWeightError",
    "UnkeyedWeightResolution",
    "resolve_keyed_unique_weights",
    "resolve_unkeyed_single_entity_weight",
]

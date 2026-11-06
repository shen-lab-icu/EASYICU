"""Order-independent grouped sums and means.

A floating-point sum depends on the order of its terms, so a plain grouped
``sum``/``mean`` publishes different last bits when the same rows reach it in
a different order (another batch split, thread count or load path). Here every
group total is the correctly rounded value of the exact sum (``math.fsum``),
which is the same for any order of the same multiset of values.

Groups of one or two values are already correctly rounded by a single IEEE
addition (``0 + a`` is exact, ``a + b`` is correctly rounded and commutative),
so only groups of three or more values take the ``fsum`` path. A group with
an infinity keeps its IEEE result (``inf + -inf`` is NaN), decided directly.
A group of finite values whose partial sums overflow (``fsum`` then raises,
in some orders only) is summed as exact rationals. Its correctly rounded total
is the same for every order. A group of finite values whose exact total is
beyond the largest finite float64 (compared exactly, before rounding), of any
size, is refused with
:class:`ExactSumOverflowError`, reason ``exact_sum_overflow``. No path falls
back to a floating-point summation. The former fallback published an
order-dependent result, finite for one order and infinite for another.

Pandas nullable numeric columns (``Int64``, ``Float64`` and the like) are
converted to float64 with ``NA`` as NaN and take the same path, so they
publish the same float64 values as a plain float column.
"""

from __future__ import annotations

import math
from typing import Hashable, List, Mapping, Sequence

import numpy as np
import pandas as pd

ORDER_SENSITIVE_AGGREGATES = frozenset({"sum", "mean"})

__all__ = [
    "EXACT_SUM_OVERFLOW",
    "ExactSumOverflowError",
    "ORDER_SENSITIVE_AGGREGATES",
    "apply_exact_columns",
    "exact_group_totals",
    "exact_groupby_agg",
    "exact_groupby_sum",
    "exact_terms_to_values",
]


EXACT_SUM_OVERFLOW = "exact_sum_overflow"


class ExactSumOverflowError(ArithmeticError):
    """Finite values whose exact sum lies beyond the float64 range.

    Reason code ``exact_sum_overflow`` (lead-r13 delta2 D2-EXACT). The former
    fallback summed the values in floating point, so the same multiset could
    publish a finite total in one order and an infinite one in another.
    """

    reason = EXACT_SUM_OVERFLOW

    def __init__(self, n_values: int) -> None:
        self.n_values = int(n_values)
        super().__init__(
            f"{self.reason}: the exact sum of {self.n_values} finite values is beyond "
            "the float64 range; refusing to publish an invented infinity"
        )


def _fsum(values: np.ndarray) -> float:
    if not np.isfinite(values).all():
        return _ieee_non_finite_sum(values)
    try:
        total = math.fsum(values.tolist())
    except OverflowError:
        # fsum raises on an overflowing partial sum, which depends on the
        # order: [M, -M, M] passes and [M, M, -M] raises. Sum the exact
        # rationals instead (each float is an integer multiple of 2**-1074);
        # the correctly rounded total is the same for every order.
        return _exact_rational_sum(values)
    if abs(total) == _MAX_FINITE:
        # A correctly rounded max may stand for an exact total beyond it.
        return _exact_rational_sum(values)
    return total


def _ieee_non_finite_sum(values: np.ndarray) -> float:
    """The IEEE sum of a group holding NaN or an infinity, in any order.

    NaN, or both infinities, give NaN; otherwise the one infinity's sign wins
    over every finite term. Decided directly, no floating-point summation.
    """

    if np.isnan(values).any():
        return math.nan
    positive = bool(np.isposinf(values).any())
    negative = bool(np.isneginf(values).any())
    if positive and negative:
        return math.nan
    return math.inf if positive else -math.inf


_SCALE_BITS = 1074  # every finite float64 is an integer multiple of 2**-1074


def _units(value: float) -> int:
    """``value`` as an exact integer number of 2**-1074 units."""

    numerator, denominator = value.as_integer_ratio()
    return numerator * ((1 << _SCALE_BITS) // denominator)


_MAX_FINITE = float(np.finfo(np.float64).max)
_MAX_UNITS = _units(_MAX_FINITE)


def _exact_rational_sum(values: np.ndarray) -> float:
    """Correctly rounded sum of finite floats, or the typed refusal.

    Reached when a finite group's float result is not trustworthy at the
    range edge: ``fsum`` overflowed, or a rounded total equals the largest
    finite float64. The total is the integer sum of the values in units of
    2**-1074, so it is exact in any order. It is compared with the largest
    finite float64 *before* rounding (lead-r13 delta4 D4-EXACT-RANGE). An
    exact total beyond it is refused, whether it would round to inf or back
    to the maximum, because publishing either would invent a value. Otherwise
    it is converted once, with correct rounding.
    """

    total = sum(_units(value) for value in values.tolist())
    if abs(total) > _MAX_UNITS:
        raise ExactSumOverflowError(len(values))
    return total / (1 << _SCALE_BITS)


def exact_group_totals(
    codes: np.ndarray, values: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(totals, counts)`` per group code ``0..n_groups-1``.

    Rows with a negative code or a NaN value are ignored. A group without
    values has total 0.0 and count 0.
    """

    codes = np.asarray(codes, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    keep = (codes >= 0) & ~np.isnan(values)
    codes = codes[keep]
    values = values[keep]
    counts = np.bincount(codes, minlength=n_groups)
    # Exact for groups of at most two values.
    totals = np.bincount(codes, weights=values, minlength=n_groups)
    # Two finite values whose rounded total reaches the range edge (inf or the
    # largest finite value) are re-decided exactly: a total beyond the range
    # is refused, as for larger groups.
    non_finite_terms = np.bincount(codes, weights=~np.isfinite(values), minlength=n_groups)
    edge = (
        (counts <= 2) & (non_finite_terms == 0)
        & (~np.isfinite(totals) | (np.abs(totals) == _MAX_FINITE))
    )
    for code in np.flatnonzero(edge):
        totals[code] = _exact_rational_sum(values[codes == code])
    many = counts >= 3
    if many.any():
        rows = many[codes]
        many_codes = codes[rows]
        many_values = values[rows]
        order = np.argsort(many_codes, kind="stable")
        many_codes = many_codes[order]
        many_values = many_values[order]
        starts = np.flatnonzero(
            np.concatenate(([True], many_codes[1:] != many_codes[:-1]))
        )
        ends = np.append(starts[1:], len(many_codes))
        totals[many_codes[starts]] = [
            _fsum(many_values[start:end]) for start, end in zip(starts, ends)
        ]
    return totals, counts


def _codes_from_ngroup(grouped) -> tuple[np.ndarray, int]:
    numbers = grouped.ngroup()
    codes = np.asarray(
        pd.to_numeric(numbers, errors="coerce").fillna(-1).to_numpy(), dtype=np.int64
    )
    return codes, int(grouped.ngroups)


def _is_summable(series: pd.Series) -> bool:
    """Float or integer (not bool) column: its sum or mean publishes float64.

    The SQL path returns DOUBLE for SUM/AVG; an integer source column summed
    on the pandas path must publish the same float64 dtype. Pandas nullable
    numeric dtypes (``Int64``, ``Float64``, ...) qualify too (D2-EXACT).
    """

    dtype = series.dtype
    if isinstance(dtype, np.dtype):
        return dtype.kind in "fiu"
    return (
        isinstance(dtype, pd.api.extensions.ExtensionDtype)
        and pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
    )


def _as_float64(series: pd.Series) -> np.ndarray:
    """float64 values of a summable column; a nullable ``NA`` becomes NaN."""

    return series.to_numpy(dtype=np.float64, na_value=np.nan)


def apply_exact_columns(
    result: pd.DataFrame,
    frame: pd.DataFrame,
    codes: np.ndarray,
    n_groups: int,
    agg_dict: Mapping[Hashable, object],
) -> pd.DataFrame:
    """Overwrite the order-sensitive float columns of a grouped ``result``.

    ``result`` row *i* must be group code *i* (``sort=False`` first-appearance
    order, as from ``ngroup``/``factorize``). ``sum`` follows pandas
    ``agg('sum')`` (empty group is 0.0); ``mean`` is NaN for an empty group.
    """

    if len(result) != n_groups:
        raise ValueError("grouped result rows do not match the group codes")
    for column, how in agg_dict.items():
        if how not in ORDER_SENSITIVE_AGGREGATES or column not in frame.columns:
            continue
        if column not in result.columns or not _is_summable(frame[column]):
            continue
        totals, counts = exact_group_totals(codes, _as_float64(frame[column]), n_groups)
        if how == "mean":
            with np.errstate(invalid="ignore", divide="ignore"):
                totals = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
        result[column] = totals.astype(np.float64, copy=False)
    return result


def exact_groupby_agg(
    frame: pd.DataFrame,
    keys: Sequence[Hashable],
    agg_dict: Mapping[Hashable, object],
    *,
    dropna: bool = True,
    sort: bool = False,
) -> pd.DataFrame:
    """``frame.groupby(keys, sort=sort, as_index=False).agg(agg_dict)``
    with order-independent ``sum`` and ``mean``."""

    keys_list: List[Hashable] = list(keys)
    grouped = frame.groupby(keys_list, sort=sort, as_index=False, dropna=dropna)
    result = grouped.agg(dict(agg_dict))
    if not any(how in ORDER_SENSITIVE_AGGREGATES for how in agg_dict.values()):
        return result
    codes, n_groups = _codes_from_ngroup(grouped)
    return apply_exact_columns(result, frame, codes, n_groups, agg_dict)


def exact_groupby_sum(
    frame: pd.DataFrame,
    keys: Sequence[Hashable],
    column: Hashable,
    *,
    min_count: int = 0,
    dropna: bool = True,
    sort: bool = False,
) -> pd.DataFrame:
    """Order-independent ``groupby(keys, sort=sort, as_index=False)[column]
    .sum(min_count=min_count)``."""

    keys_list: List[Hashable] = list(keys)
    grouped = frame.groupby(keys_list, sort=sort, as_index=False, dropna=dropna)
    result = grouped[column].sum(min_count=min_count)
    if not _is_summable(frame[column]):
        return result
    codes, n_groups = _codes_from_ngroup(grouped)
    totals, counts = exact_group_totals(codes, _as_float64(frame[column]), n_groups)
    if min_count > 0:
        totals = np.where(counts >= min_count, totals, np.nan)
    if len(result) != n_groups:
        raise ValueError("grouped result rows do not match the group codes")
    result[column] = totals
    return result


def exact_terms_to_values(terms: pd.Series, how: str) -> pd.Series:
    """Collapse per-group term lists (SQL ``LIST``) into exact sums or means.

    A NULL or empty list is NaN, as SQL ``SUM``/``AVG`` over no values is NULL.
    """

    if how not in ORDER_SENSITIVE_AGGREGATES:
        raise ValueError(f"unsupported exact aggregate {how!r}")
    cells = terms.to_numpy(dtype=object)
    lengths = np.zeros(len(cells), dtype=np.int64)
    arrays = []
    for position, cell in enumerate(cells):
        if cell is None or (isinstance(cell, float) and math.isnan(cell)):
            continue
        values = np.asarray(cell, dtype=np.float64).ravel()
        lengths[position] = len(values)
        arrays.append(values)
    values = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)
    codes = np.repeat(np.arange(len(cells), dtype=np.int64), lengths)
    totals, counts = exact_group_totals(codes, values, len(cells))
    with np.errstate(invalid="ignore", divide="ignore"):
        if how == "mean":
            totals = totals / np.maximum(counts, 1)
        out = np.where(counts > 0, totals, np.nan)
    return pd.Series(out, index=terms.index, name=terms.name, dtype=np.float64)

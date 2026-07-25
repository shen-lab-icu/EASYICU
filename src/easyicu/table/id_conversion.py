"""Canonical ID-system conversion for typed ICU tables.

This module is the **single** home for ``change_id``/``upgrade_id``/
``downgrade_id``. There used to be a second set in ``easyicu.table.utils`` that
used the words "upgrade" and "downgrade" in the opposite sense — ``utils``'
``downgrade_id`` expanded rows coarse-to-fine while this one aggregates
fine-to-coarse — so which implementation an import resolved to changed what the
call actually did. ``table.utils`` is now a deprecated shim that says so.

Everything here operates on plain ``DataFrame`` objects, which is why it can sit
below the typed-table classes without importing them.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import pandas as pd

__all__ = [
    "IdMapRelationError",
    "change_id",
    "classify_id_relation",
    "downgrade_id",
    "upgrade_id",
]


def upgrade_id(
    data: pd.DataFrame,
    id_map: pd.DataFrame,
    from_id: str,
    to_id: str,
    keep_old_id: bool = False,
) -> pd.DataFrame:
    """Upgrade ID type to a higher level (R ricu upgrade_id).

    Converts IDs to a higher-level identifier (e.g., hadm_id -> icustay_id).
    This is a one-to-many relationship.

    Args:
        data: Input DataFrame with lower-level IDs
        id_map: Mapping DataFrame with both ID columns
        from_id: Current ID column name (lower level)
        to_id: Target ID column name (higher level)
        keep_old_id: Whether to keep the original ID column

    Returns:
        DataFrame with upgraded IDs

    Examples:
        >>> # Upgrade from hadm_id to icustay_id
        >>> vitals = pd.DataFrame({'hadm_id': [1, 1, 2], 'hr': [80, 85, 90]})
        >>> mapping = pd.DataFrame({'hadm_id': [1, 1, 2], 'icustay_id': [10, 11, 20]})
        >>> upgrade_id(vitals, mapping, 'hadm_id', 'icustay_id')
        # Result: 3 rows become 4 rows (hadm_id=1 duplicated for 2 stays)
    """
    # Validate inputs
    if from_id not in data.columns:
        raise ValueError(f"Column '{from_id}' not found in data")
    if from_id not in id_map.columns or to_id not in id_map.columns:
        raise ValueError(f"Columns '{from_id}' and '{to_id}' must be in id_map")

    # Get unique mapping (remove duplicates in id_map)
    mapping = id_map[[from_id, to_id]].drop_duplicates()

    # Merge to add new ID
    result = data.merge(mapping, on=from_id, how="left")

    # Optionally remove old ID
    if not keep_old_id:
        result = result.drop(columns=[from_id])

    return result


def downgrade_id(
    data: pd.DataFrame,
    id_map: pd.DataFrame,
    from_id: str,
    to_id: str,
    agg_funcs: Optional[Dict[str, Union[str, Callable]]] = None,
    keep_old_id: bool = False,
) -> pd.DataFrame:
    """Downgrade ID type to a lower level (R ricu downgrade_id).

    Converts IDs to a lower-level identifier (e.g., icustay_id -> hadm_id).
    This is a many-to-one relationship requiring aggregation.

    Args:
        data: Input DataFrame with higher-level IDs
        id_map: Mapping DataFrame with both ID columns
        from_id: Current ID column name (higher level)
        to_id: Target ID column name (lower level)
        agg_funcs: Dictionary mapping column names to aggregation functions
                   (default: first for non-numeric, mean for numeric)
        keep_old_id: Whether to keep the original ID column

    Returns:
        DataFrame with downgraded IDs and aggregated data

    Examples:
        >>> # Downgrade from icustay_id to hadm_id
        >>> vitals = pd.DataFrame({
        ...     'icustay_id': [10, 11, 20],
        ...     'hr': [80, 85, 90],
        ...     'temp': [36.5, 37.0, 36.8]
        ... })
        >>> mapping = pd.DataFrame({
        ...     'icustay_id': [10, 11, 20],
        ...     'hadm_id': [1, 1, 2]
        ... })
        >>> downgrade_id(vitals, mapping, 'icustay_id', 'hadm_id',
        ...              agg_funcs={'hr': 'mean', 'temp': 'mean'})
        # Result: 3 rows become 2 rows (stays 10 and 11 merged into hadm 1)
    """
    # Validate inputs
    if from_id not in data.columns:
        raise ValueError(f"Column '{from_id}' not found in data")
    if from_id not in id_map.columns or to_id not in id_map.columns:
        raise ValueError(f"Columns '{from_id}' and '{to_id}' must be in id_map")

    # Get unique mapping
    mapping = id_map[[from_id, to_id]].drop_duplicates()

    # Merge to add new ID
    result = data.merge(mapping, on=from_id, how="left")

    # Determine columns to aggregate
    group_cols = [to_id]
    if keep_old_id:
        group_cols.append(from_id)

    data_cols = [col for col in result.columns if col not in [from_id, to_id]]

    # Build aggregation dict
    if agg_funcs is None:
        agg_funcs = {}
        ambiguous = []
        for col in data_cols:
            if pd.api.types.is_numeric_dtype(result[col]):
                agg_funcs[col] = "mean"
                if not _is_continuous_measurement(result[col]):
                    ambiguous.append(col)
            else:
                agg_funcs[col] = "first"
        if ambiguous:
            # The mean of a SOFA score across two stays is not a SOFA score,
            # and the mean of a 0/1 death flag is a proportion wearing an
            # outcome's name. Both used to happen silently because "numeric"
            # was taken to mean "continuous measurement".
            raise ValueError(
                f"downgrading {from_id!r} to {to_id!r} would average column(s) "
                f"{ambiguous}, which are stored as integers or 0/1 flags — an "
                "ordinal score, an indicator, a category code or a count does "
                "not survive a mean. Pass agg_funcs explicitly, e.g. "
                f"agg_funcs={{{ambiguous[0]!r}: 'max'}}."
            )

    # Apply aggregation if needed
    if not keep_old_id:
        result = result.drop(columns=[from_id])
        result = result.groupby(to_id, as_index=False).agg(agg_funcs)
    else:
        result = result.groupby([to_id, from_id], as_index=False).agg(agg_funcs)

    return result


def _is_continuous_measurement(values: pd.Series) -> bool:
    """Whether averaging this column is defensible without being told.

    Heart rate and creatinine are measurements and are stored as floats;
    SOFA components, KDIGO stages, event counts and numeric category codes are
    stored as integers, and their mean is a different quantity from the thing
    measured. A 0/1 flag is a flag whichever dtype it arrived in, so it is
    caught by value as well.

    Storage dtype is a heuristic, not proof — which is why it only decides
    whether the caller is *asked*, never what is computed. Note the deliberate
    non-rule: an integer-valued float column such as a heart rate recorded as
    ``80.0`` stays continuous, because flagging it would make every vitals
    downgrade require boilerplate and the check would be turned off.
    """

    if not pd.api.types.is_float_dtype(values):
        return False
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return False
    return not bool(numeric.isin((0, 1)).all())


class IdMapRelationError(ValueError):
    """The id map does not describe a relation ``change_id`` can apply alone."""


def classify_id_relation(id_map: pd.DataFrame, from_id: str, to_id: str) -> str:
    """Name the relation an id map describes.

    Counting distinct values on each side is not enough to tell the direction.
    ``A→X, A→Y, B→X, B→Y`` has two distinct values on both sides and is
    many-to-many, yet a count comparison reads it as one-to-one — after which
    ``dict(zip(...))`` keeps only the last pair for each key and the rest of
    the mapping disappears with no error. Ask each side how far it fans out
    instead.
    """

    for column, frame, label in (
        (from_id, id_map, "id_map"),
        (to_id, id_map, "id_map"),
    ):
        if column not in frame.columns:
            raise ValueError(f"Column '{column}' must be in {label}")

    mapping = id_map[[from_id, to_id]].drop_duplicates().dropna()
    fans_out = bool((mapping.groupby(from_id)[to_id].nunique() > 1).any())
    fans_in = bool((mapping.groupby(to_id)[from_id].nunique() > 1).any())
    if fans_out and fans_in:
        return "many_to_many"
    if fans_out:
        return "one_to_many"
    if fans_in:
        return "many_to_one"
    return "one_to_one"


def change_id(
    data: pd.DataFrame,
    id_map: pd.DataFrame,
    from_id: str,
    to_id: str,
    keep_old_id: bool = False,
    agg_funcs: Optional[Dict[str, Union[str, Callable]]] = None,
    on_many_to_many: Optional[str] = None,
) -> pd.DataFrame:
    """Change ID type (auto-detect upgrade vs downgrade) (R ricu change_id).

    The direction is read from how the mapping fans out, not from how many
    distinct values each side happens to have.

    Args:
        data: Input DataFrame
        id_map: Mapping DataFrame with both ID columns
        from_id: Current ID column name
        to_id: Target ID column name
        keep_old_id: Whether to keep the original ID column
        agg_funcs: Aggregation functions (for downgrade only)
        on_many_to_many: What to do when the map is many-to-many, which has no
            single correct answer: ``'expand'`` emits one row per matching
            target id, ``'aggregate'`` collapses to one row per target id using
            ``agg_funcs``. Omitting it refuses the conversion rather than
            picking one silently.

    Returns:
        DataFrame with changed IDs

    Raises:
        IdMapRelationError: many-to-many map with no strategy given.

    Examples:
        >>> # Auto-detect direction
        >>> change_id(data, mapping, 'hadm_id', 'icustay_id')
    """
    relation = classify_id_relation(id_map, from_id, to_id)

    if relation == "many_to_many":
        if on_many_to_many == "expand":
            return upgrade_id(data, id_map, from_id, to_id, keep_old_id)
        if on_many_to_many == "aggregate":
            return downgrade_id(data, id_map, from_id, to_id, agg_funcs, keep_old_id)
        raise IdMapRelationError(
            f"the map from {from_id!r} to {to_id!r} is many-to-many: at least "
            f"one {from_id} reaches several {to_id} and at least one {to_id} is "
            f"reached by several {from_id}. Row counts and per-row values both "
            "depend on which way it is resolved, so pass "
            "on_many_to_many='expand' or on_many_to_many='aggregate'."
        )

    if relation == "one_to_many":
        return upgrade_id(data, id_map, from_id, to_id, keep_old_id)
    if relation == "many_to_one":
        return downgrade_id(data, id_map, from_id, to_id, agg_funcs, keep_old_id)

    # Proven one-to-one, so no key in the dict can shadow another.
    mapping = id_map[[from_id, to_id]].drop_duplicates()
    mapping_dict = dict(zip(mapping[from_id], mapping[to_id]))
    result = data.copy()
    result[to_id] = result[from_id].map(mapping_dict)

    if not keep_old_id:
        result = result.drop(columns=[from_id])

    return result

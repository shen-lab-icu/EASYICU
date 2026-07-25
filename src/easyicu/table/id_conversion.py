"""Canonical ID-system conversion for typed ICU tables.

This module is the **single** home for ``change_id``/``upgrade_id``/
``downgrade_id``. There used to be a second set in ``easyicu.table.utils`` that
used the words "upgrade" and "downgrade" in the opposite sense — ``utils``'
``downgrade_id`` expanded rows coarse-to-fine while this one aggregates
fine-to-coarse — so which implementation an import resolved to changed what the
call actually did. ``table.utils`` is now a deprecated shim that says so.

The vocabulary here is deliberately **coarse-grained / fine-grained**, never
"higher level" / "lower level": the two deprecated modules both use the
level wording and they mean opposite things by it, which is how the direction
got inverted in the first place. ``hadm_id`` is coarse, ``icustay_id`` is fine.

Everything here operates on plain ``DataFrame`` objects, which is why it can sit
below the typed-table classes without importing them.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import pandas as pd

__all__ = [
    "IdMapRelationError",
    "UnmappedIdError",
    "change_id",
    "classify_id_relation",
    "downgrade_id",
    "upgrade_id",
]

ON_UNMAPPED_POLICIES = ("error", "keep", "drop")


class UnmappedIdError(ValueError):
    """The id map does not cover every id in the data, and no policy was given.

    Left-joining an incomplete map and then dropping the original id column
    replaces a real identifier with ``NaN``: the rows survive, look like data,
    and can no longer be traced back to a patient. In ``downgrade_id`` it is
    worse than that — ``groupby`` drops null keys, so the unmapped rows leave
    the result entirely and only the row count records that anything happened.
    Neither is something a caller should have to discover afterwards.
    """


def _apply_unmapped_policy(
    result: pd.DataFrame,
    from_id: str,
    to_id: str,
    on_unmapped: str,
    *,
    operation: str,
) -> pd.DataFrame:
    """Decide what happens to rows the map does not cover.

    ``keep`` reproduces the historical behaviour and is the reason this is a
    parameter rather than an unconditional raise: dropping ids that are outside
    a cohort on purpose is legitimate. Choosing it silently is not.
    """

    if on_unmapped not in ON_UNMAPPED_POLICIES:
        raise ValueError(
            f"on_unmapped must be one of {ON_UNMAPPED_POLICIES}, got {on_unmapped!r}"
        )

    unmapped = result[to_id].isna()
    count = int(unmapped.sum())
    if count == 0 or on_unmapped == "keep":
        return result
    if on_unmapped == "drop":
        return result.loc[~unmapped].reset_index(drop=True)

    if from_id in result.columns:
        missing = result.loc[unmapped, from_id].dropna().unique().tolist()
        examples = ", ".join(repr(value) for value in missing[:5])
        if len(missing) > 5:
            examples += f", ... ({len(missing)} distinct)"
        detail = f" ({from_id} = {examples})" if examples else ""
    else:
        detail = ""

    raise UnmappedIdError(
        f"{operation} from {from_id!r} to {to_id!r}: the id map covers no "
        f"{to_id} for {count} of {len(result)} row(s){detail}. Those rows would "
        f"keep their measurements while losing their identity. Pass "
        "on_unmapped='drop' to remove them or on_unmapped='keep' to accept a "
        f"null {to_id}."
    )


def _require_non_empty_map(
    id_map: pd.DataFrame, from_id: str, to_id: str
) -> pd.DataFrame:
    """An empty map is a failed load, not a relation with no rows.

    It is worth separating from the incomplete-map case: ``on_unmapped`` says
    what to do about ids outside the map, whereas an empty map means the map
    itself never arrived — wrong column names, a filter that matched nothing, a
    read that returned no rows — and every answer to "what should happen to the
    unmapped rows" is wrong when the real answer is "fix the map".
    """

    mapping = id_map[[from_id, to_id]].drop_duplicates()
    if mapping.dropna().empty:
        raise IdMapRelationError(
            f"the id map from {from_id!r} to {to_id!r} contains no usable pairs, "
            f"so every row would be assigned a null {to_id}. Check that the map "
            "was loaded and that both column names are right."
        )
    return mapping


def upgrade_id(
    data: pd.DataFrame,
    id_map: pd.DataFrame,
    from_id: str,
    to_id: str,
    keep_old_id: bool = False,
    on_unmapped: str = "error",
) -> pd.DataFrame:
    """Convert to a finer-grained ID, expanding rows (R ricu upgrade_id).

    Converts IDs to a finer-grained identifier (e.g., hadm_id -> icustay_id),
    which is a one-to-many relationship: one row can become several.

    Note that :mod:`easyicu.table.utils` and :mod:`easyicu.io.data_utils` each
    ship an ``upgrade_id`` that converts the *other* way. This module is the
    canonical one; those two are deprecated and warn when called.

    Args:
        data: Input DataFrame with coarse-grained IDs
        id_map: Mapping DataFrame with both ID columns
        from_id: Current ID column name (coarse-grained)
        to_id: Target ID column name (fine-grained)
        keep_old_id: Whether to keep the original ID column
        on_unmapped: What to do with rows the map does not cover — ``'error'``
            (default), ``'drop'``, or ``'keep'`` for a null ``to_id``.

    Returns:
        DataFrame with upgraded IDs

    Raises:
        IdMapRelationError: the map contains no usable pairs.
        UnmappedIdError: rows are unmapped and ``on_unmapped='error'``.

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
    mapping = _require_non_empty_map(id_map, from_id, to_id)

    # Merge to add new ID
    result = data.merge(mapping, on=from_id, how="left")
    result = _apply_unmapped_policy(
        result, from_id, to_id, on_unmapped, operation="upgrade_id"
    )

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
    on_unmapped: str = "error",
) -> pd.DataFrame:
    """Convert to a coarser-grained ID, aggregating rows (R ricu downgrade_id).

    Converts IDs to a coarser-grained identifier (e.g., icustay_id -> hadm_id),
    which is a many-to-one relationship and therefore requires aggregation.

    Note that :mod:`easyicu.table.utils` ships a ``downgrade_id`` that converts
    the *other* way and expands rows instead of aggregating them. This module is
    the canonical one; that one is deprecated and warns when called.

    Args:
        data: Input DataFrame with fine-grained IDs
        id_map: Mapping DataFrame with both ID columns
        from_id: Current ID column name (fine-grained)
        to_id: Target ID column name (coarse-grained)
        agg_funcs: Dictionary mapping column names to aggregation functions.
            When omitted, non-numeric columns take ``first`` and continuous
            float measurements take ``mean``; integer, 0/1 and other
            non-continuous numeric columns are **refused** rather than averaged,
            because the mean of an ordinal score or an indicator is a different
            quantity from the thing it names. Name them explicitly to proceed.
        keep_old_id: Whether to keep the original ID column
        on_unmapped: What to do with rows the map does not cover — ``'error'``
            (default), ``'drop'``, or ``'keep'``, which groups them together
            under a null ``to_id`` rather than letting ``groupby`` discard them.

    Returns:
        DataFrame with downgraded IDs and aggregated data

    Raises:
        IdMapRelationError: the map contains no usable pairs.
        UnmappedIdError: rows are unmapped and ``on_unmapped='error'``.
        ValueError: a numeric column would be averaged without being named.

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
    mapping = _require_non_empty_map(id_map, from_id, to_id)

    # Merge to add new ID
    result = data.merge(mapping, on=from_id, how="left")
    result = _apply_unmapped_policy(
        result, from_id, to_id, on_unmapped, operation="downgrade_id"
    )

    # Determine columns to aggregate
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

    # ``dropna=False`` so that whether unmapped rows survive is decided by
    # ``on_unmapped`` above and not by a pandas default several lines away.
    if not keep_old_id:
        result = result.drop(columns=[from_id])
        result = result.groupby(to_id, as_index=False, dropna=False).agg(agg_funcs)
    else:
        result = result.groupby([to_id, from_id], as_index=False, dropna=False).agg(
            agg_funcs
        )

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

    Returns one of ``empty``, ``one_to_one``, ``one_to_many``, ``many_to_one``
    or ``many_to_many``. ``empty`` is reported separately because a map with no
    pairs fans out nowhere on either side, which the fan-out test would
    otherwise read as a clean one-to-one relation.
    """

    for column, frame, label in (
        (from_id, id_map, "id_map"),
        (to_id, id_map, "id_map"),
    ):
        if column not in frame.columns:
            raise ValueError(f"Column '{column}' must be in {label}")

    mapping = id_map[[from_id, to_id]].drop_duplicates().dropna()
    if mapping.empty:
        return "empty"
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
    on_unmapped: str = "error",
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
        on_unmapped: What to do with rows the map does not cover — ``'error'``
            (default), ``'drop'``, or ``'keep'`` for a null ``to_id``.

    Returns:
        DataFrame with changed IDs

    Raises:
        IdMapRelationError: many-to-many map with no strategy given, or a map
            with no usable pairs.
        UnmappedIdError: rows are unmapped and ``on_unmapped='error'``.

    Examples:
        >>> # Auto-detect direction
        >>> change_id(data, mapping, 'hadm_id', 'icustay_id')
    """
    relation = classify_id_relation(id_map, from_id, to_id)

    if relation == "empty":
        # Reached before the direction is known: with no pairs there is nothing
        # to read a direction from, so neither branch below could be chosen.
        _require_non_empty_map(id_map, from_id, to_id)

    if relation == "many_to_many":
        if on_many_to_many == "expand":
            return upgrade_id(
                data, id_map, from_id, to_id, keep_old_id, on_unmapped=on_unmapped
            )
        if on_many_to_many == "aggregate":
            return downgrade_id(
                data,
                id_map,
                from_id,
                to_id,
                agg_funcs,
                keep_old_id,
                on_unmapped=on_unmapped,
            )
        raise IdMapRelationError(
            f"the map from {from_id!r} to {to_id!r} is many-to-many: at least "
            f"one {from_id} reaches several {to_id} and at least one {to_id} is "
            f"reached by several {from_id}. Row counts and per-row values both "
            "depend on which way it is resolved, so pass "
            "on_many_to_many='expand' or on_many_to_many='aggregate'."
        )

    if relation == "one_to_many":
        return upgrade_id(
            data, id_map, from_id, to_id, keep_old_id, on_unmapped=on_unmapped
        )
    if relation == "many_to_one":
        return downgrade_id(
            data,
            id_map,
            from_id,
            to_id,
            agg_funcs,
            keep_old_id,
            on_unmapped=on_unmapped,
        )

    # Proven one-to-one, so no key in the dict can shadow another.
    mapping = _require_non_empty_map(id_map, from_id, to_id)
    mapping_dict = dict(zip(mapping[from_id], mapping[to_id]))
    result = data.copy()
    result[to_id] = result[from_id].map(mapping_dict)
    result = _apply_unmapped_policy(
        result, from_id, to_id, on_unmapped, operation="change_id"
    )

    if not keep_old_id:
        result = result.drop(columns=[from_id])

    return result

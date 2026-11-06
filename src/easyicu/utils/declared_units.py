"""Units come from a declaration, never from the size of the values.

A rule such as "every value is at most 1, so these are fractions" depends on
which rows share a batch or a table. One outlier, or one homogeneous chunk,
flips the reading of every row beside it (lead-r13 magnitude/format sweep).
Where a declaration exists, the owner uses it. Where a caller reaches a
value-based guess without one, the owner raises this error instead of guessing.
"""

from __future__ import annotations

__all__ = [
    "UndeclaredUnitError",
    "FIO2_UNITS",
    "declared_fio2_unit",
    "BOUNDS_UNIT_SUSPECT_MIN_VALUES",
    "bounds_unit_suspect",
    "refuse_bounds_unit_suspect",
]

#: Accepted spellings of a declared FiO2 unit, normalised to one name.
FIO2_UNITS = {
    "fraction": "fraction",
    "percentage": "percent",
    "percent": "percent",
    "%": "percent",
}


class UndeclaredUnitError(RuntimeError):
    """A unit would have to be guessed from the values; nothing declared it.

    Not a ``ValueError``: readers catch those as "source not available here",
    which would turn a unit question into a silently absent result. ``reason``
    is a stable code; ``quantity`` and ``site`` name what and where.
    """

    def __init__(self, *, reason: str, quantity: str, site: str, detail: str):
        self.reason = reason
        self.quantity = quantity
        self.site = site
        self.detail = detail
        super().__init__(
            f"{reason}: {quantity} at {site}: {detail}. Declare the unit; "
            "it is not inferred from the values"
        )


#: Evidence floor of the bounds-unit-suspect refusal: below it, out-of-bounds
#: values are dropped as ordinary implausible values.
BOUNDS_UNIT_SUSPECT_MIN_VALUES = 100


def bounds_unit_suspect(
    nonzero_values: int, bounded_non_null: int, bounded_aggregate_non_null: int = 0,
) -> bool:
    """At least 100 non-zero values in one read and none within the declared bounds.

    The values then contradict the declared bounds as a whole (another unit),
    and dropping them would publish the concept as absent. Zeros are not
    counted: every unit conversion maps 0 to 0, so a zero carries no unit
    evidence (eICU monitors record 0 mmHg end-tidal CO2 for no signal, and a
    50-stay read can hold only those). Every bounded read path decides with
    this one rule (lead-r13 sweep #14; Codex delta4 V6).
    """
    return bool(
        nonzero_values >= BOUNDS_UNIT_SUSPECT_MIN_VALUES
        and bounded_non_null == 0
        and bounded_aggregate_non_null == 0
    )


def refuse_bounds_unit_suspect(
    *,
    quantity: str,
    site: str,
    nonzero_values: int,
    bounded_non_null: int,
    bounded_aggregate_non_null: int = 0,
    value_min=None,
    value_max=None,
) -> None:
    """Raise ``UndeclaredUnitError(bounds_unit_suspect)`` when the rule holds."""
    if bounds_unit_suspect(nonzero_values, bounded_non_null, bounded_aggregate_non_null):
        raise UndeclaredUnitError(
            reason="bounds_unit_suspect",
            quantity=str(quantity),
            site=site,
            detail=(
                f"{int(nonzero_values)} values in this batch (zeros not counted) and "
                f"none within the declared bounds [{value_min}, {value_max}]"
            ),
        )


def declared_fio2_unit(value, *, site: str):
    """Return ``"fraction"``, ``"percent"`` or ``None`` (undeclared)."""
    if value is None:
        return None
    unit = FIO2_UNITS.get(str(value).strip().lower())
    if unit is None:
        raise UndeclaredUnitError(
            reason="fio2_unit_invalid", quantity="fio2", site=site,
            detail=f"fio2_unit={value!r} is not one of {sorted(FIO2_UNITS)}",
        )
    return unit

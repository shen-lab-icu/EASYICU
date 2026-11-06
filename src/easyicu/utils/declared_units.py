"""Units come from a declaration, never from the size of the values.

A rule such as "every value is at most 1, so these are fractions" depends on
which rows share a batch or a table. One outlier, or one homogeneous chunk,
flips the reading of every row beside it (lead-r13 magnitude/format sweep).
Where a declaration exists, the owner uses it. Where a caller reaches a
value-based guess without one, the owner raises this error instead of guessing.
"""

from __future__ import annotations

__all__ = ["UndeclaredUnitError", "FIO2_UNITS", "declared_fio2_unit"]

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

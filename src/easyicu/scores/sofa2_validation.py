"""Fail-closed input contracts for the SOFA-2 score owner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class SOFA2InputError(ValueError):
    """A typed SOFA-2 input-contract failure with a stable reason code."""

    def __init__(
        self,
        *,
        component: str,
        field: str,
        reason_code: str,
        message: str,
        invalid_count: int = 0,
    ) -> None:
        self.component = component
        self.field = field
        self.reason_code = reason_code
        self.invalid_count = int(invalid_count)
        super().__init__(
            f"{message} (component={component}, field={field}, "
            f"reason_code={reason_code}, invalid_count={self.invalid_count})"
        )


@dataclass(frozen=True)
class NormalizedFiO2:
    """Normalized FiO2 fractions plus the single detected source-unit domain."""

    values: pd.Series
    source_unit: str


def validate_aligned_input(
    value: pd.Series,
    *,
    component: str,
    field: str,
    index: pd.Index | None = None,
) -> pd.Series:
    """Require a pandas Series aligned exactly with the component anchor."""

    if not isinstance(value, pd.Series):
        raise SOFA2InputError(
            component=component,
            field=field,
            reason_code=f"{component}_{field}_series_required",
            message="SOFA-2 inputs must be pandas Series",
            invalid_count=1,
        )
    if index is not None and not value.index.equals(index):
        raise SOFA2InputError(
            component=component,
            field=field,
            reason_code=f"{component}_{field}_index_mismatch",
            message="SOFA-2 input indices must align exactly",
            invalid_count=len(value),
        )
    return value


def validate_numeric_input(
    value: pd.Series,
    *,
    component: str,
    field: str,
    index: pd.Index | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
    integer: bool = False,
) -> pd.Series:
    """Coerce numeric encodings and reject non-finite or out-of-domain values."""

    raw = validate_aligned_input(
        value,
        component=component,
        field=field,
        index=index,
    )
    # ``pd.to_numeric`` silently turns temporal dtypes into epoch/duration
    # integers, so a mis-wired datetime column would score as an enormous but
    # perfectly "valid" physiological value. Allow-list the dtype families a
    # SOFA-2 input may legitimately arrive in instead: numeric (including the
    # nullable and boolean extensions), object/string (parsed below), and
    # categorical (the receipt/coverage encoding used elsewhere in the
    # package). Anything else fails closed rather than being coerced.
    if not (
        pd.api.types.is_numeric_dtype(raw)
        # Arrow-backed booleans are not is_numeric_dtype even though numpy and
        # nullable booleans are, and this package converts end-to-end through
        # pyarrow -- component availability/observed receipts routinely arrive
        # as bool[pyarrow].
        or pd.api.types.is_bool_dtype(raw)
        or pd.api.types.is_object_dtype(raw)
        or pd.api.types.is_string_dtype(raw)
        or isinstance(raw.dtype, pd.CategoricalDtype)
    ):
        raise SOFA2InputError(
            component=component,
            field=field,
            reason_code=f"{component}_{field}_numeric_dtype_invalid",
            message=(
                "SOFA-2 numeric input must be numeric, string or categorical, "
                f"not {raw.dtype}"
            ),
            invalid_count=len(raw),
        )
    numeric = pd.to_numeric(raw, errors="coerce")
    bad_encoding = raw.notna() & numeric.isna()
    if bad_encoding.any():
        raise SOFA2InputError(
            component=component,
            field=field,
            reason_code=f"{component}_{field}_numeric_encoding_invalid",
            message="SOFA-2 numeric input contains an unparseable value",
            invalid_count=int(bad_encoding.sum()),
        )

    nonfinite = numeric.notna() & ~np.isfinite(numeric.astype(float))
    if nonfinite.any():
        raise SOFA2InputError(
            component=component,
            field=field,
            reason_code=f"{component}_{field}_nonfinite",
            message="SOFA-2 numeric input must be finite",
            invalid_count=int(nonfinite.sum()),
        )

    out_of_domain = pd.Series(False, index=numeric.index, dtype=bool)
    if minimum is not None:
        below = numeric < minimum if minimum_inclusive else numeric <= minimum
        out_of_domain |= numeric.notna() & below
    if maximum is not None:
        above = numeric > maximum if maximum_inclusive else numeric >= maximum
        out_of_domain |= numeric.notna() & above
    if out_of_domain.any():
        raise SOFA2InputError(
            component=component,
            field=field,
            reason_code=f"{component}_{field}_domain_invalid",
            message="SOFA-2 numeric input is outside its allowed clinical domain",
            invalid_count=int(out_of_domain.sum()),
        )

    if integer:
        fractional = numeric.notna() & ((numeric % 1).abs() > 1e-9)
        if fractional.any():
            raise SOFA2InputError(
                component=component,
                field=field,
                reason_code=f"{component}_{field}_integer_required",
                message="SOFA-2 ordinal input must be integer-valued",
                invalid_count=int(fractional.sum()),
            )
    return numeric


def normalize_fio2_input(
    value: pd.Series,
    *,
    index: pd.Index,
) -> NormalizedFiO2:
    """Accept one unambiguous FiO2 unit domain and return fractions.

    Valid non-missing inputs are either fractions in ``[0.21, 1.0]`` or
    percentages in ``[21, 100]``. A single Series cannot mix the two domains.

    The two domains are disjoint, so the *returned fractions* are independent
    of how the caller partitioned its rows. The mixed-unit rejection, however,
    can only see the rows in this call: a source that encodes units
    inconsistently fails closed when the inconsistent rows arrive together and
    passes when a chunk happens to be homogeneous. Treat it as a per-call data
    hygiene gate, not a cohort-level invariant -- a cohort-wide guarantee needs
    the unit declared once at the owning extraction layer.
    """

    numeric = validate_numeric_input(
        value,
        component="sofa2_resp",
        field="fio2",
        index=index,
    )
    present = numeric.notna()
    fractional = present & numeric.between(0.21, 1.0, inclusive="both")
    percentage = present & numeric.between(21.0, 100.0, inclusive="both")
    invalid = present & ~(fractional | percentage)
    if invalid.any():
        raise SOFA2InputError(
            component="sofa2_resp",
            field="fio2",
            reason_code="sofa2_resp_fio2_domain_invalid",
            message=("FiO2 must use fractions 0.21-1.0 or percentages 21-100"),
            invalid_count=int(invalid.sum()),
        )
    if fractional.any() and percentage.any():
        raise SOFA2InputError(
            component="sofa2_resp",
            field="fio2",
            reason_code="sofa2_resp_fio2_units_mixed",
            message="FiO2 cannot mix fraction and percentage units in one Series",
            invalid_count=int(present.sum()),
        )
    if percentage.any():
        return NormalizedFiO2(values=numeric / 100.0, source_unit="percent")
    if fractional.any():
        return NormalizedFiO2(values=numeric, source_unit="fraction")
    return NormalizedFiO2(values=numeric, source_unit="unavailable")


__all__ = [
    "NormalizedFiO2",
    "SOFA2InputError",
    "normalize_fio2_input",
    "validate_aligned_input",
    "validate_numeric_input",
]

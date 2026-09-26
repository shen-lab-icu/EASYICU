"""Apply clinical volume bounds to a signed total, never to its input terms."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bound_signed_total(frame, value_column, minimum, maximum):
    """Keep valid totals, including observed zero; invalid totals remain unknown.

    Removing an invalid total is the loader's missing-observation convention.
    Counts retain the distinction between negative, excessive and nonfinite
    totals. No clipping, zero filling, or reassignment to another hour occurs.
    """
    values = pd.to_numeric(frame[value_column], errors="coerce")
    finite = np.isfinite(values)
    below = finite & values.lt(minimum) if minimum is not None else finite & False
    above = finite & values.gt(maximum) if maximum is not None else finite & False
    valid = finite & ~below & ~above
    result = frame.loc[valid].copy()
    result[value_column] = values.loc[valid]
    result.attrs["easyicu_signed_sum_bounds"] = {
        "policy": "SIGNED_SUM_THEN_BOUNDS_V1",
        "total_rows": len(frame),
        "below_minimum_rows": int(below.sum()),
        "above_maximum_rows": int(above.sum()),
        "nonfinite_rows": int((~finite).sum()),
        "observed_zero_rows": int((valid & values.eq(0)).sum()),
        "retained_rows": len(result),
    }
    return result

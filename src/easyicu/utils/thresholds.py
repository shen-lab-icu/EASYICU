"""Decimal-threshold comparisons for derived clinical quantities.

A clinical criterion is written in decimals ("creatinine rise >= 0.3 mg/dL",
">= 1.5 x baseline", "urine < 0.5 mL/kg/h"). When the compared quantity is
the result of arithmetic (a difference, ratio, product or sum), binary
floating point puts an exact decimal boundary a few ulps to either side:
``1.1 + 0.3`` is ``1.4000000000000001`` and ``0.8 * 1.5`` is
``1.2000000000000002``, so a creatinine of 1.4 after 1.1, or of 1.2 against a
baseline of 0.8, would not be AKI.

These helpers decide the comparison as the clinical text reads it: a value
equal to the threshold within ``eps = 1e-6 * max(1, |threshold|)`` is equal.
So ``ge``/``le`` include the boundary and ``gt``/``lt`` exclude it, whatever
the float noise. 1e-6 also absorbs float32 input rounding (about 6e-8
relative), so a float32 and a float64 load decide alike. Charted values (0.01 mg/dL,
1 mL, 1 s clocks) are far coarser, so a real difference is never absorbed:
on every 0.1 mg/dL creatinine pair from 0.1 to 14.9, ``+0.3``, ``x1.5`` and
``x3`` agree with exact decimal arithmetic in float32 and float64 alike.

Use them only where arithmetic produced the compared quantity (or the
threshold side, as in ``baseline * 1.5``). A charted value compared directly
with a constant needs no tolerance.

Both arguments may be scalars, numpy arrays or pandas Series. Only the
threshold moves; the operator is the plain one, so missing values behave
exactly as before: a NaN in a numpy/float64 column compares False, and a
pandas nullable column (``Float64``, ``Int64``) propagates ``<NA>``
(three-valued logic), so "not evaluated" stays distinct from "negative".
Time-window limits (48 h, 7 d, 6 h urine windows, the 12 h record gap) use
the same helpers in hours: 48 h x 1e-6 is 0.17 s and 12 h is 0.04 s, below
the 1 s source clocks. Outcome horizons are in days (365 d x 1e-6 is 32 s).
"""

from __future__ import annotations

import numpy as np

REL_EPS = 1e-6


def _eps(threshold):
    return REL_EPS * np.maximum(1.0, np.abs(threshold))


def ge(x, threshold):
    """``x >= threshold`` with an exact decimal boundary included."""
    return x >= threshold - _eps(threshold)


def gt(x, threshold):
    """``x > threshold`` with an exact decimal boundary excluded."""
    return x > threshold + _eps(threshold)


def le(x, threshold):
    """``x <= threshold`` with an exact decimal boundary included."""
    return x <= threshold + _eps(threshold)


def lt(x, threshold):
    """``x < threshold`` with an exact decimal boundary excluded."""
    return x < threshold - _eps(threshold)

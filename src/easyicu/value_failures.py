"""Typed value-path failures that no fallback may absorb.

A broad ``except Exception`` on a value path (a fallback read, a best-effort
alignment, a per-concept error report) must let these through unchanged: an
interval aggregation that could not be computed exactly, or a storage
installation that contradicts its own manifest, is a failure of the result,
never a reason to return the unaggregated frame, an empty table or a partial
report. Handlers re-raise them first::

    except TYPED_VALUE_FAILURES:
        raise
    except Exception:
        ...
"""

from __future__ import annotations

from .io.ts_utils import IntervalAggregationError
from .storage_layout import StorageIntegrityError

TYPED_VALUE_FAILURES: tuple[type[Exception], ...] = (
    IntervalAggregationError,
    StorageIntegrityError,
)

__all__ = ["TYPED_VALUE_FAILURES"]

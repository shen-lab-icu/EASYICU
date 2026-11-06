"""Typed value-path failures that no fallback may absorb.

A broad ``except Exception`` on a value path (a fallback read, a best-effort
alignment, a per-concept error report) must let these through unchanged: an
interval aggregation that could not be computed exactly, a numeric time axis
whose unit is undeclared or unknown, or a storage installation that
contradicts its own manifest, is a failure of the result,
never a reason to return the unaggregated frame, an empty table or a partial
report. The same holds for an exact sum refused because its exact total is
beyond float64 (``ExactSumOverflowError``, lead-r13 delta4 D4-EXACT-CATCH).
A resolver refusing a concurrent top-level call (``ResolverBusyError``,
B18-REENTRY-DEADLOCK) is one too: a fallback must not turn it into an empty
result. Handlers re-raise them first::

    except TYPED_VALUE_FAILURES:
        raise
    except Exception:
        ...
"""

from __future__ import annotations

from .io.ts_utils import IntervalAggregationError, TimeAxisUnitError
from .resolver_errors import ResolverBusyError
from .storage_layout import StorageIntegrityError
from .utils.exact_sum import ExactSumOverflowError

TYPED_VALUE_FAILURES: tuple[type[Exception], ...] = (
    IntervalAggregationError,
    StorageIntegrityError,
    ExactSumOverflowError,
    TimeAxisUnitError,
    ResolverBusyError,
)

__all__ = ["TYPED_VALUE_FAILURES"]

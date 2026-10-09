"""Why the Writer left an executed run without a draft, as codes.

Owner
-----
The write phase records a Writer failure as a ``writer_agent`` error finding.
This module owns the one Writer stop that names its cause: the model
provider's transport failed with a typed server status (5xx), after the
attempts its transport policy allowed.  The status is read only from the
exception's typed fields (``safe_provider_http_status_code``), never from its
text, so a message that quotes "503" names nothing.  Any other Writer failure
names no stop, and its finding stays as it was.

The stop travels in the finding's detail.  Readiness reports the stop of the
current Writer error (the last one supersession has not retired) as
``writer_stop``, re-validated here, so the Web host can name it as what
stopped a failed-closed run instead of the gate axis the missing draft failed.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..providers.clients import safe_provider_http_status_code
from ..providers.transport_retry import RETRY_STOP_REASONS

WRITER_PROVIDER_TRANSPORT_UNAVAILABLE = "writer_provider_transport_unavailable"


def _server_status(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and 500 <= value <= 599:
        return value
    return None


def _stop(status: Any, attempts: Any, exhausted: Any) -> Optional[dict[str, Any]]:
    server_status = _server_status(status)
    if server_status is None:
        return None
    stop: dict[str, Any] = {
        "reason_code": WRITER_PROVIDER_TRANSPORT_UNAVAILABLE,
        "provider_http_status": server_status,
    }
    if isinstance(attempts, int) and not isinstance(attempts, bool) and attempts >= 1:
        stop["transport_attempts"] = attempts
    if isinstance(exhausted, str) and exhausted in RETRY_STOP_REASONS:
        stop["transport_retry_exhausted"] = exhausted
    return stop


def writer_transport_stop(exc: BaseException) -> Optional[dict[str, Any]]:
    """The stop a Writer exception names, or ``None``.

    Only the exception the Writer raised is read, not its causes: a failure
    another owner wrapped is that owner's to name.
    """

    return _stop(
        safe_provider_http_status_code(exc),
        getattr(exc, "easyicu_transport_attempts", None),
        getattr(exc, "easyicu_transport_retry_exhausted", None),
    )


def writer_failure_detail(exc: BaseException, **detail: Any) -> dict[str, Any]:
    """A ``writer_agent`` finding's detail: the write phase's fields, then the
    stop ``exc`` names, if any."""

    return {**detail, **(writer_transport_stop(exc) or {})}


def writer_stop(detail: Any) -> Optional[dict[str, Any]]:
    """A recorded stop, re-validated, or ``None``.

    Reads a finding's detail or a projection of it; keys other than the
    stop's own are dropped.
    """

    if not isinstance(detail, Mapping):
        return None
    if detail.get("reason_code") != WRITER_PROVIDER_TRANSPORT_UNAVAILABLE:
        return None
    return _stop(
        detail.get("provider_http_status"),
        detail.get("transport_attempts"),
        detail.get("transport_retry_exhausted"),
    )


def current_writer_stop(active_findings: Sequence[Any]) -> Optional[dict[str, Any]]:
    """The stop the last current Writer error names, or ``None``.

    ``active_findings`` are the findings supersession left active, in the
    order they were recorded; a later Writer failure replaces an earlier
    one as the cause, whatever it names.
    """

    for finding in reversed(active_findings):
        if (
            getattr(finding, "validator", None) == "writer_agent"
            and getattr(finding, "severity", None) == "error"
        ):
            return writer_stop(getattr(finding, "detail", None))
    return None


__all__ = [
    "WRITER_PROVIDER_TRANSPORT_UNAVAILABLE",
    "current_writer_stop",
    "writer_failure_detail",
    "writer_stop",
    "writer_transport_stop",
]

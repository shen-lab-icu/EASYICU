"""A deterministic standard executor's typed stop, as a record the host reads.

A standard executor stops with :class:`ExecutorStop` when a condition of the
data it was given leaves its declared result without an estimate, and it can
name that condition.  A defect of its input contract is not such a stop: it
still fails with its own error.  Before raising, the executor writes a record
of the stop into its output directory (:func:`write_executor_stop_record`).
The record holds codes only: the schema version, the executor that stopped,
the stop's reason code and its lower-layer cause code.  The process still
fails, so a reader that does not know the record still sees a failure.

The record is a claim of the step's own process, which can write any file in
its output directory.  The host therefore reads it only for the standard
executor that owns the step, and :func:`parse_executor_stop_record` accepts
only the reasons and causes registered in :data:`EXECUTOR_STOP_REASONS` for
that executor.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

from ..methods.time_varying_cox import TIME_VARYING_NOT_ESTIMABLE_REASONS

__all__ = [
    "EXECUTOR_STOP_REASONS",
    "EXECUTOR_STOP_RECORD_MAX_BYTES",
    "EXECUTOR_STOP_RECORD_NAME",
    "EXECUTOR_STOP_RECORD_REJECTIONS",
    "EXECUTOR_STOP_SCHEMA_VERSION",
    "ExecutorStop",
    "ExecutorStopReason",
    "ExecutorStopRecordError",
    "RecordedExecutorStop",
    "executor_stop_codes",
    "parse_executor_stop_record",
    "recorded_executor_stop",
    "registered_executor_stop",
    "write_executor_stop_record",
]

EXECUTOR_STOP_SCHEMA_VERSION = "easyicu.executor_stop/1"
#: The record's file name in the step's output directory.  It is a private
#: work product of the failed attempt: the host reads it, then removes it with
#: the executor's other internal files, so it never becomes evidence.
EXECUTOR_STOP_RECORD_NAME = "_executor_stop.json"
EXECUTOR_STOP_RECORD_MAX_BYTES = 4096
_RECORD_KEYS = frozenset({"schema_version", "owner", "reason_code", "cause_code"})


@dataclass(frozen=True)
class ExecutorStopReason:
    """One registered stop of one standard executor."""

    #: The standard executor's analysis kind, as the step record names it
    #: (``deterministic_standard_analysis``).
    owner: str
    #: The lower-layer causes the stop names; empty when it names none.
    cause_codes: frozenset[str]
    #: Whether the stop follows from the approved plan, the bound data and the
    #: execution identity alone, so a retry with all three unchanged repeats it.
    repeats_on_unchanged_retry: bool


EXECUTOR_STOP_REASONS: Mapping[str, ExecutorStopReason] = MappingProxyType(
    {
        # The proportional-hazards check rejected a constant hazard ratio, so
        # the interval-specific hazard ratios are the primary result, and the
        # interval model could not be estimated on these data.
        "continuous_survival_interval_result_not_estimable": ExecutorStopReason(
            owner="signed_landmark_continuous_survival_suite",
            cause_codes=frozenset(TIME_VARYING_NOT_ESTIMABLE_REASONS),
            repeats_on_unchanged_retry=True,
        ),
        # The same rule in the binary landmark survival suite: its rejected
        # PH test leaves the interval-specific hazard ratios as the result,
        # and the interval model could not be estimated on these data.
        "landmark_survival_interval_result_not_estimable": ExecutorStopReason(
            owner="signed_landmark_survival_suite",
            cause_codes=frozenset(TIME_VARYING_NOT_ESTIMABLE_REASONS),
            repeats_on_unchanged_retry=True,
        ),
        # Every modelled record has the same exposure value, so the suite
        # has no step to report its hazard ratios per and no association to
        # estimate.
        "continuous_survival_exposure_has_one_value": ExecutorStopReason(
            owner="signed_landmark_continuous_survival_suite",
            cause_codes=frozenset(),
            repeats_on_unchanged_retry=True,
        ),
    }
)
#: Why a record was not accepted.  A rejected record names no stop.
EXECUTOR_STOP_RECORD_REJECTIONS = frozenset(
    {
        "record_unreadable",
        "record_too_large",
        "record_shape_invalid",
        "record_schema_unknown",
        "record_owner_mismatch",
        "record_reason_unregistered",
        "record_cause_unregistered",
    }
)


def registered_executor_stop(
    reason_code: Any, cause_code: Any, *, owner: Optional[str] = None
) -> Optional[tuple[str, Optional[str]]]:
    """``(reason_code, cause_code)`` when registered together, else ``None``.

    With ``owner``, the reason must also be that executor's.
    """

    if not isinstance(reason_code, str):
        return None
    reason = EXECUTOR_STOP_REASONS.get(reason_code)
    if reason is None or (owner is not None and owner != reason.owner):
        return None
    if reason.cause_codes:
        if not isinstance(cause_code, str) or cause_code not in reason.cause_codes:
            return None
        return reason_code, cause_code
    return (reason_code, None) if cause_code is None else None


class ExecutorStop(ValueError):
    """A standard executor's typed stop.

    The message starts with the stop's reason code.  Raise it ``from`` the
    lower-layer error, so the cause stays attached.
    """

    def __init__(
        self, reason_code: str, *, cause_code: Optional[str] = None, detail: str = ""
    ) -> None:
        reason = EXECUTOR_STOP_REASONS.get(reason_code)
        if reason is None:
            raise ValueError(f"unregistered executor stop reason {reason_code!r}")
        if registered_executor_stop(reason_code, cause_code) is None:
            raise ValueError(
                f"executor stop {reason_code!r} does not name cause {cause_code!r}"
            )
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)
        self.owner = reason.owner
        self.reason_code = reason_code
        self.cause_code = cause_code


def write_executor_stop_record(out_dir: Path | str, stop: ExecutorStop) -> bool:
    """Write ``stop``'s record into ``out_dir``; whether it was written.

    A failure to write never replaces the stop: it is reported on stderr and
    the caller raises the stop as it would have.  The record is created
    exclusively beside its final name and renamed into place, so a file or
    link already at the temporary name is never followed.
    """

    payload = json.dumps(
        {
            "schema_version": EXECUTOR_STOP_SCHEMA_VERSION,
            "owner": stop.owner,
            "reason_code": stop.reason_code,
            "cause_code": stop.cause_code,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    directory = Path(out_dir)
    temporary = directory / f".{EXECUTOR_STOP_RECORD_NAME}.{os.getpid()}.tmp"
    try:
        with open(temporary, "xb") as handle:
            handle.write(payload)
        os.replace(temporary, directory / EXECUTOR_STOP_RECORD_NAME)
    except OSError as exc:
        print(
            f"executor stop record not written ({type(exc).__name__})",
            file=sys.stderr,
        )
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True


@dataclass(frozen=True)
class RecordedExecutorStop:
    """An accepted record: the stop the step's executor named."""

    owner: str
    reason_code: str
    cause_code: Optional[str]


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite number {token}")


class ExecutorStopRecordError(ValueError):
    """A record that names no stop; ``code`` is one of the record rejections."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def parse_executor_stop_record(
    payload: bytes, *, expected_owner: str
) -> RecordedExecutorStop:
    """Read one record written for ``expected_owner``'s step."""

    if len(payload) > EXECUTOR_STOP_RECORD_MAX_BYTES:
        raise ExecutorStopRecordError("record_too_large")
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError):
        raise ExecutorStopRecordError("record_unreadable") from None
    if not isinstance(value, dict) or set(value) != _RECORD_KEYS:
        raise ExecutorStopRecordError("record_shape_invalid")
    if value["schema_version"] != EXECUTOR_STOP_SCHEMA_VERSION:
        raise ExecutorStopRecordError("record_schema_unknown")
    if value["owner"] != expected_owner:
        raise ExecutorStopRecordError("record_owner_mismatch")
    reason_code = value["reason_code"]
    reason = (
        EXECUTOR_STOP_REASONS.get(reason_code) if isinstance(reason_code, str) else None
    )
    if reason is None or reason.owner != expected_owner:
        raise ExecutorStopRecordError("record_reason_unregistered")
    if registered_executor_stop(reason_code, value["cause_code"]) is None:
        raise ExecutorStopRecordError("record_cause_unregistered")
    return RecordedExecutorStop(
        owner=expected_owner, reason_code=reason_code, cause_code=value["cause_code"]
    )


def recorded_executor_stop(
    step_record: Mapping[str, Any],
) -> Optional[tuple[str, Optional[str]]]:
    """The registered stop a step record carries, or ``None``.

    The candidate loop writes ``executor_stop_reason_code`` and
    ``executor_stop_cause_code`` onto the step record from an accepted record;
    a reader checks them again against the step's own executor.
    """

    if "executor_stop_reason_code" not in step_record:
        return None
    return registered_executor_stop(
        step_record.get("executor_stop_reason_code"),
        step_record.get("executor_stop_cause_code"),
        owner=str(step_record.get("deterministic_standard_analysis") or ""),
    )


def executor_stop_codes(step_record: Mapping[str, Any]) -> dict[str, str]:
    """The stop a step record carries, as a failed step's report keys.

    ``reason_code``, and ``cause_code`` when the stop names one; empty for a
    step without a registered stop, which keeps the keys it always had.
    """

    stop = recorded_executor_stop(step_record)
    if stop is None:
        return {}
    reason_code, cause_code = stop
    if cause_code is None:
        return {"reason_code": reason_code}
    return {"reason_code": reason_code, "cause_code": cause_code}

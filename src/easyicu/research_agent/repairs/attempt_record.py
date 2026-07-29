"""An account of the deterministic repairs a step was offered.

fresh19, step ``07_primary_adjusted_association``. The step died with::

    TypeError: float() argument must be a string or a real number,
               not 'StrictNumericInput'

having spent 2 of 2 LLM repairs while leaving 5 of 9 provider calls unused.
The deterministic patch for that exact error already existed and was already
wired (``repairs/strict_numeric_result.py`` via ``repairs/source.py``), and
applying it by hand to that run's on-disk ``analysis.py`` and ``run.log``
produced the correct one-line fix immediately.

So one of two things happened, and the run could not say which:

* the patch was offered this failure, declined it, and the LLM repair that
  followed introduced the defect with no budget left to fix it; or
* the patch was never offered this failure at all.

Only the *firing* case was recorded (``step_record["runner_repair"]``). A
decline wrote nothing, the enabling flag was not recorded anywhere in the run,
and every attempt overwrites the same ``analysis.py`` / ``run.log``, so the
losing drafts are gone. A repair loop that keeps no account of the free repairs
it was offered cannot be debugged from its own artifacts -- which is how a
whole diagnosis stalled.

This module only writes that account. It decides nothing: it does not choose a
repair, does not gate one, and never raises into the step it is describing.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

ATTEMPT_KEY = "deterministic_runner_repair_attempts"

# ``Traceback`` tail lines look like ``SomeError: detail`` at column 0.
_TERMINAL_ERROR = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_.]*Error|[A-Za-z_][A-Za-z0-9_.]*Exception|SystemExit|KeyboardInterrupt)\b.*$"
)


def _digest(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()


def failure_signature(run_log: Optional[str]) -> Optional[str]:
    """The last terminal exception line in a run log, trimmed.

    This is the fact a reader needs first: *which* failure the patch was asked
    about. Without it a decline is indistinguishable from a decline about
    something else entirely.
    """

    if not run_log:
        return None
    for line in reversed(str(run_log).splitlines()):
        stripped = line.strip()
        if _TERMINAL_ERROR.match(stripped):
            return stripped[:300]
    return None


def record_deterministic_runner_repair_attempt(
    step_record: MutableMapping[str, Any],
    *,
    code: Optional[str],
    run_log: Optional[str],
    previous_repair: Optional[str],
    outcome: str,
    repair_id: Optional[str] = None,
) -> None:
    """Append one attempt to the step's deterministic-repair account.

    ``outcome`` is ``applied`` / ``declined`` / ``disabled``. ``declined`` and
    ``disabled`` are the two the run previously could not distinguish from each
    other or from "never reached".
    """

    try:
        attempts = step_record.get(ATTEMPT_KEY)
        if not isinstance(attempts, list):
            attempts = []
            step_record[ATTEMPT_KEY] = attempts
        entry: Dict[str, Any] = {
            "sequence": len(attempts) + 1,
            "outcome": str(outcome),
            "repair_id": str(repair_id) if repair_id else None,
            "previous_repair": str(previous_repair) if previous_repair else None,
            "code_sha256": _digest(code),
            "run_log_sha256": _digest(run_log),
            "failure_signature": failure_signature(run_log),
        }
        attempts.append(entry)
    except Exception:  # noqa: BLE001 - an account must never fail its subject
        return


def declined_attempts(step_record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Attempts where a free repair was offered a failure and passed on it."""

    attempts = step_record.get(ATTEMPT_KEY)
    if not isinstance(attempts, list):
        return []
    return [
        attempt
        for attempt in attempts
        if isinstance(attempt, Mapping) and attempt.get("outcome") == "declined"
    ]


__all__ = [
    "ATTEMPT_KEY",
    "declined_attempts",
    "failure_signature",
    "record_deterministic_runner_repair_attempt",
]

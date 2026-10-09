"""The host decisions whose job is being started, per study, in this process.

A submission runs its start checks before its job exists -- for a plan's data
preparation, tens of seconds.  Until the job exists the study has no job
pointer, so a page reopened in that window would offer the same button again.
The job submission owner (``host_action_jobs``) registers the decision here for
that window and is the only writer; the workflow projection reads it.

The registry is process state on purpose: the start checks die with the
process, so after a restart it is empty, which is then the truth.

Leaf module: it imports no Copilot or job owner.
"""

from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

from easyicu.webserver.host_action_contracts import StartingEntry

_STRIPES = tuple(threading.Lock() for _ in range(32))
_STARTING: Dict[str, StartingEntry] = {}


def _stripe(study_context_id: str) -> threading.Lock:
    digest = hashlib.sha256(study_context_id.encode("utf-8")).digest()
    return _STRIPES[digest[0] % len(_STRIPES)]


class StudyStarting:
    """One study's entry, read and written while the study's lock is held."""

    def __init__(self, study_context_id: str) -> None:
        self.study_context_id = study_context_id
        self._held = True

    def _require_lock(self) -> None:
        if not self._held:
            raise RuntimeError("host_action_study_lock_released")

    def current(self) -> Optional[StartingEntry]:
        self._require_lock()
        return _STARTING.get(self.study_context_id)

    def register(self, entry: StartingEntry) -> None:
        """Register the one decision whose job this study is starting."""

        self._require_lock()
        if entry.study_context_id != self.study_context_id:
            raise ValueError("host_action_starting_study_mismatch")
        if self.study_context_id in _STARTING:
            raise RuntimeError("host_action_already_starting")
        _STARTING[self.study_context_id] = entry

    def clear(self, entry: StartingEntry) -> bool:
        """Clear the study's entry if it is still ``entry``."""

        self._require_lock()
        if _STARTING.get(self.study_context_id) != entry:
            return False
        del _STARTING[self.study_context_id]
        return True


@contextmanager
def study_lock(study_context_id: str) -> Iterator[StudyStarting]:
    """Hold one study's lock for a short read or write of its entry.

    It is never held across a submission's start checks or a workflow
    projection, and it does not nest.
    """

    with _stripe(study_context_id):
        starting = StudyStarting(study_context_id)
        try:
            yield starting
        finally:
            starting._held = False


def starting_for(study_context_id: str) -> Optional[StartingEntry]:
    """The decision whose job the study is starting now, if any."""

    with _stripe(study_context_id):
        return _STARTING.get(study_context_id)


def clear_all_for_tests() -> None:
    for stripe in _STRIPES:
        stripe.acquire()
    try:
        _STARTING.clear()
    finally:
        for stripe in _STRIPES:
            stripe.release()


__all__ = [
    "StudyStarting",
    "clear_all_for_tests",
    "starting_for",
    "study_lock",
]

"""Bound terminal pipeline errors to host progress, never exception messages."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
from contextlib import contextmanager
import re


_PHASES = {"planning": "plan", "plan": "plan", "execute": "execute", "execution": "execute",
           "context": "context", "audit": "context", "hypothesis": "plan", "runtime": "runtime",
           "human_review": "human_review", "write": "write", "finalise": "finalise"}
_RUNNERS = {"host", "docker", "subprocess", "host_sandbox", "not_started", "unknown"}
_EXCEPTIONS = {kind: kind.__name__ for kind in (
    ValueError, KeyError, TypeError, IndexError, AttributeError, RuntimeError,
    AssertionError, OSError, FileNotFoundError, PermissionError, TimeoutError,
    ImportError, ModuleNotFoundError, ZeroDivisionError, MemoryError,
)}
EXECUTION_FAILURE_CODES = frozenset({
    "research_pipeline_execution_failed", "research_pipeline_approved_run_failed",
    "research_pipeline_review_resume_failed",
})


def _step(value):
    return value if isinstance(value, str) and re.fullmatch(r"[a-z0-9][a-z0-9_]{0,79}", value) else "unknown"


def _phase(value, default="unknown"):
    return _PHASES.get(value, default) if isinstance(value, str) else default


def _runner(value):
    return value if isinstance(value, str) and value in _RUNNERS else "unknown"


@dataclass(frozen=True)
class FailureLocation:
    """Only codes observed at the host operation boundary, no event prose."""

    phase: str = "unknown"
    step_id: str = "unknown"
    runner_kind: str = "unknown"

    def observe(self, event):
        if not isinstance(event, Mapping):
            return self
        phase = _phase(event.get("phase") or event.get("stage") or event.get("step"), self.phase)
        step = _step(event.get("step_id"))
        if step == "unknown" and phase == self.phase:
            step = self.step_id
        if phase == "plan":
            step = "plan"
        elif event.get("status") == "complete":
            step = "unknown"
        return replace(self, phase=phase, step_id=step)


@contextmanager
def track_failure_location(*, runner_kind=None, progress_callback=None):
    """Keep the failing operation's coordinates through outer error wrappers."""
    location = FailureLocation(runner_kind=_runner(runner_kind))

    def progress(event):
        nonlocal location
        location = location.observe(event)
        if progress_callback is not None:
            progress_callback(event)

    try:
        yield progress
    except Exception as exc:
        # Review-resume wrappers expose their original exception as ``cause``.
        # Bind both so that the review owner may continue to unwrap it.
        for error in (exc, getattr(exc, "cause", None)):
            if isinstance(error, BaseException) and not isinstance(getattr(error, "easyicu_failure_location", None), FailureLocation):
                error.easyicu_failure_location = location
        raise


def call_with_failure_location(operation, *args, runner_kind=None, **kwargs):
    """Call an operation with the same bounded tracking used by direct calls."""
    with track_failure_location(runner_kind=runner_kind, progress_callback=kwargs.get("progress_callback")) as progress:
        return operation(*args, **{**kwargs, "progress_callback": progress})


def execution_failure_diagnostic(chain, *, observed_coordinates_only=False):
    """A closed fallback when no more specific typed owner diagnosed the stop.

    Missing coordinates are explicitly unknown for execution fallbacks. A
    provider timeout takes only observed coordinates, preserving its typed
    owner's classification. Classifying from exception
    prose, traceback locals, or an arbitrary diagnostic dict would turn data
    into authority and potentially disclose it.
    """
    error = next((error for error in reversed(chain) if isinstance(getattr(error, "easyicu_failure_location", None), FailureLocation)), chain[-1])
    location = getattr(error, "easyicu_failure_location", None)
    if not isinstance(location, FailureLocation):
        location = FailureLocation()
    phase = _phase(location.phase)
    step = _step(location.step_id)
    runner = _runner(location.runner_kind)
    if phase in {"context", "plan", "human_review"}:
        runner = "not_started"
    coordinates = {"step_id": step, "phase": phase, "runner_kind": runner}
    if observed_coordinates_only:
        return {key: value for key, value in coordinates.items() if value != "unknown"}
    return {
        "owner": "easyicu.execution.pipeline_v1",
        "reason_code": "execution_step_raised" if phase == "execute" and step != "unknown" else "pipeline_operation_raised",
        **coordinates,
        "exception_type": _EXCEPTIONS.get(type(error), "UnclassifiedError"),
    }

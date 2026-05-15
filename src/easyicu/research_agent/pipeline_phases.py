"""Public phase-runner wrappers around the pipeline's three phases.

:class:`ResearchAgentPipeline.run` factors execution into three private
phase methods (``_run_plan_phase``, ``_run_execute_phase``,
``_run_write_phase``). They share state via the :mod:`pipeline_state`
dataclasses, but the methods themselves still live on the giant
``ResearchAgentPipeline`` class. Splitting them out physically is a
multi-session refactor that requires identifying which ``self.*``
fields each phase touches and threading them through a runner-style
abstraction.

This module is a smaller step in that direction: it exposes three
thin runner classes that **wrap** the existing phase methods. Calling
``PlanPhaseRunner(pipeline).run(**kwargs)`` is exactly equivalent to
calling ``pipeline._run_plan_phase(**kwargs)``. The wrappers add:

* a stable public name for the phase ("plan" / "execute" / "write");
* a :class:`Protocol`-friendly ``run`` signature suitable for
  alternative orchestrators (LangGraph, retry shims, benchmark
  harnesses) that want to invoke phases independently;
* a documented contract (input shape, return type) without committing
  to a particular implementation strategy yet.

The actual phase logic stays in :class:`ResearchAgentPipeline` for
now. When the underlying methods are eventually moved out, only the
``run`` implementations here will change — callers can already use
this API.

Example::

    pipeline = ResearchAgentPipeline.from_config(config)
    plan_runner = PlanPhaseRunner(pipeline)
    plan_state = plan_runner.run(question="...", cohort_path=...)
    execute_state = ExecutePhaseRunner(pipeline).run(plan_result=plan_state, ...)
    write_state = WritePhaseRunner(pipeline).run(plan_result=plan_state, execute_result=execute_state, ...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from .pipeline_state import ExecutePhaseState, PlanPhaseState, WritePhaseState

if TYPE_CHECKING:
    from .pipeline import ResearchAgentPipeline


class PlanPhaseRunner:
    """Wraps :meth:`ResearchAgentPipeline._run_plan_phase` with a stable name."""

    name: ClassVar[str] = "plan"

    def __init__(self, pipeline: "ResearchAgentPipeline") -> None:
        self._pipeline = pipeline

    def run(self, **kwargs: Any) -> PlanPhaseState:
        return self._pipeline._run_plan_phase(**kwargs)

    def __repr__(self) -> str:
        return f"PlanPhaseRunner(pipeline={type(self._pipeline).__name__})"


class ExecutePhaseRunner:
    """Wraps :meth:`ResearchAgentPipeline._run_execute_phase`."""

    name: ClassVar[str] = "execute"

    def __init__(self, pipeline: "ResearchAgentPipeline") -> None:
        self._pipeline = pipeline

    def run(self, **kwargs: Any) -> ExecutePhaseState:
        return self._pipeline._run_execute_phase(**kwargs)

    def __repr__(self) -> str:
        return f"ExecutePhaseRunner(pipeline={type(self._pipeline).__name__})"


class WritePhaseRunner:
    """Wraps :meth:`ResearchAgentPipeline._run_write_phase`."""

    name: ClassVar[str] = "write"

    def __init__(self, pipeline: "ResearchAgentPipeline") -> None:
        self._pipeline = pipeline

    def run(self, **kwargs: Any) -> WritePhaseState:
        return self._pipeline._run_write_phase(**kwargs)

    def __repr__(self) -> str:
        return f"WritePhaseRunner(pipeline={type(self._pipeline).__name__})"


__all__ = [
    "ExecutePhaseRunner",
    "PlanPhaseRunner",
    "WritePhaseRunner",
]

"""The reviewed requirement sets a run's configuration binds onto its context.

Owner
-----
A reviewed plan revision can carry three host-accepted requirement sets: the
baseline table's variables, the study population and the analysis inputs.  A
new run binds each onto its research context before the context is sealed.
A resumed run restores its sealed context only when that context already
carries exactly the configured sets; otherwise the set's owner raises its
drift reason.  Each planning owner decides its own binding; this module
applies the three, in one order, for :mod:`easyicu.research_agent.pipeline`.
"""

from __future__ import annotations

from ..planning.accepted_analysis_inputs import bind_analysis_inputs
from ..planning.baseline_requirements import bind_baseline_requirements
from ..planning.population_requirements import bind_population_requirements
from ..schema import ResearchContext
from .config import PipelineConfig


def bind_reviewed_requirements(
    context: ResearchContext,
    config: PipelineConfig,
    *,
    restoring: bool = False,
) -> ResearchContext:
    """Bind ``config``'s reviewed requirement sets onto ``context``.

    With ``restoring``, ``context`` is a sealed context being restored: it is
    returned unchanged when it carries exactly the configured sets.
    """

    context = bind_baseline_requirements(
        context, config.bound_baseline_requirements, restoring=restoring
    )
    context = bind_population_requirements(
        context, config.bound_population_requirements, restoring=restoring
    )
    return bind_analysis_inputs(
        context, config.bound_analysis_inputs, restoring=restoring
    )


__all__ = ["bind_reviewed_requirements"]

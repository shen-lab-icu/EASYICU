"""The run log's record of how a run's analysis family was chosen.

The planner plans on one family (``analysis_types.infer_analysis_type``).
Whether the study's typed design stated it, the question's words read it, or
the fallback rule chose it for a question no family's words read
(``infer_analysis_type_with_basis``) is written to the run log beside the
family, so the plan card can say when the default rule decided.  The plan
itself carries no such field.
"""

from __future__ import annotations

from typing import Any, Callable

from ..schema import ResearchContext
from .analysis_types import infer_analysis_type_with_basis

__all__ = ["traced_analysis_type_key"]


def traced_analysis_type_key(
    context: ResearchContext, emit_progress: Callable[..., Any], run_id: str
) -> str:
    """The family ``context`` is planned on; how it was chosen goes to the run log."""

    spec, basis = infer_analysis_type_with_basis(context)
    emit_progress(
        "plan",
        f"Analysis family {spec.key} ({basis}).",
        run_id=run_id,
        analysis_family=spec.key,
        family_basis=basis,
    )
    return spec.key

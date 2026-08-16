"""Biomni-style governed one-shot entry point.

Biomni's ``A1().go(prompt)`` is a useful usability benchmark, but EasyICU
deliberately does not copy its "full system privileges" execution model.
This module provides the same one-call ergonomics while keeping every
EasyICU invariant: the returned run stays fail-closed, analysis-only, and
never gains manuscript/publication authority through this facade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from .pipeline import ResearchAgentPipeline
from .providers.mocks import MockLLMClient

__all__ = ["go"]


def go(
    question: str,
    *,
    cohort: Union[str, Path, Any],
    llm: Optional[Any] = None,
    workdir: Optional[Union[str, Path]] = None,
    database: str = "miiv",
    target_outcome: Optional[str] = None,
    primary_exposure: Optional[str] = None,
    inclusion_criteria: Optional[list[str]] = None,
    exclusion_criteria: Optional[list[str]] = None,
    time_windows: Optional[list[Any]] = None,
    notes: Optional[str] = None,
    manuscript_language: Optional[str] = None,
    stop_after_analysis: bool = True,
    progress_callback: Optional[Any] = None,
    **run_kwargs: Any,
) -> Any:
    """Run one research question through the governed Plan → Execute → Write path.

    This is a convenience wrapper over :class:`ResearchAgentPipeline`. It
    intentionally does not expose publication authority, formal-mode toggles,
    or provider-budget overrides: those stay on the explicit pipeline/CLI
    surface. ``llm`` defaults to the deterministic offline mock; pass a real
    provider client only when the surrounding process has already satisfied
    the opt-in gate.
    """
    client = llm if llm is not None else MockLLMClient()
    pipeline = ResearchAgentPipeline(
        workdir=workdir,
        llm=client,
    )
    return pipeline.run(
        question=question,
        cohort=cohort,
        database=database,
        target_outcome=target_outcome,
        primary_exposure=primary_exposure,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        time_windows=time_windows,
        notes=notes,
        manuscript_language=manuscript_language,
        stop_after_analysis=stop_after_analysis,
        progress_callback=progress_callback,
        **run_kwargs,
    )

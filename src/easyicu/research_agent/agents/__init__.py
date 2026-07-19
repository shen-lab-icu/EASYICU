"""Agent roles and the optional CLI-backed coder.

The package surface is deliberately lazy: importing :mod:`agents` must not
load prompts, provider clients, or the large role implementation until a
public agent class is requested. Internal helpers live in :mod:`agents.core`
and are not part of this package API.
"""

from __future__ import annotations

from typing import Any

_CORE_EXPORTS = frozenset(
    {
        "PlannerAgent",
        "ReplannerAgent",
        "ClinicalSemanticsAgent",
        "DataExtractionAgent",
        "StatisticalAnalysisAgent",
        "VisualizationAgent",
        "ManuscriptAgent",
        "CriticAgent",
        "RuntimeSupervisor",
        "CoderAgent",
        "AnalyzerAgent",
        "WriterAgent",
    }
)
_AGENTIC_CODER_EXPORTS = frozenset({"AgenticCoderAgent", "maybe_wrap_coder"})

__all__ = sorted(_CORE_EXPORTS | _AGENTIC_CODER_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _CORE_EXPORTS:
        from . import core

        value = getattr(core, name)
    elif name in _AGENTIC_CODER_EXPORTS:
        from . import agentic_coder

        value = getattr(agentic_coder, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

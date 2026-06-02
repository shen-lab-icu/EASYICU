"""Generic fallback / case-plugin infrastructure for ``ResearchAgentPipeline``.

The pipeline needs deterministic Python fallbacks for the moments when
LLM-generated code fails (or omits a required numeric field in its
``step_summary``). Historically those fallbacks were case-specific Python
strings (the lactate / MAP / vasopressor study) hardcoded directly into
``pipeline.py``. That worked for one paper but bakes a research design
into a tool that is supposed to be paper-agnostic.

This module replaces those hardcoded dispatchers with a small plugin
protocol:

* :class:`CasePlugin` — protocol every case plugin satisfies.
* :class:`CasePluginRegistry` — pipeline-side registry that the runtime
  asks "is there a plugin that handles this step / column / failure?".
* The :class:`ResearchAgentPipeline` consults the registry at every
  former hardcoded dispatch point. Plugins are **opt-in**: a default
  ``ResearchAgentPipeline()`` carries no plugins and therefore no bias
  toward any specific paper's design.

Concrete plugins can live under :mod:`easyicu.research_agent.case_plugins`,
but the package intentionally bundles no paper-specific plugins.
"""

from __future__ import annotations

from .protocol import CasePlugin, CasePluginRegistry, NullCasePlugin

__all__ = [
    "CasePlugin",
    "CasePluginRegistry",
    "NullCasePlugin",
]

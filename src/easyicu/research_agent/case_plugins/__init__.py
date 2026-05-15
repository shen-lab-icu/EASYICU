"""Case-specific plugins for ``ResearchAgentPipeline``.

Each subdirectory under this package is one research-case plugin that
supplies deterministic Python fallbacks / repairs / column aliases for
a specific paper or study design. Plugins are **opt-in** — a default
``ResearchAgentPipeline()`` constructs with no plugins registered, so
it carries no case-specific bias.

Bundled plugins:

* :mod:`.lactate_map_vaso` — the original lactate / mean arterial
  pressure / vasopressor → mortality study, whose deterministic
  fallbacks were previously hardcoded directly into ``pipeline.py``.

Adding a new plugin
-------------------
1. Create ``case_plugins/<name>/__init__.py`` that exports a
   :class:`~easyicu.research_agent.fallback.CasePlugin` instance.
2. Optional: split fallback bodies into ``fallbacks.py`` /
   ``repairs.py`` / ``column_map.py`` siblings.
3. Users opt in by passing the plugin instance to the pipeline (see
   :class:`~easyicu.research_agent.pipeline.ResearchAgentPipeline`).
"""

from __future__ import annotations

__all__: list[str] = []

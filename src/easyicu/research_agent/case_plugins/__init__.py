"""Case-specific plugins for ``ResearchAgentPipeline``.

Each subdirectory under this package is one research-case plugin that
supplies deterministic Python fallbacks / repairs / column aliases for
a specific paper or study design. Plugins are **opt-in** — a default
``ResearchAgentPipeline()`` constructs with no plugins registered, so
it carries no case-specific bias.

No case-specific plugins are bundled with the package: paper-specific
fallback scripts are dead weight for general users and risk laundering
hand-written analyses as autonomous agent output. This package ships
only the *mechanism*; users supply their own plugins.

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

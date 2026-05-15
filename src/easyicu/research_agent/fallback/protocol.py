"""CasePlugin Protocol + registry.

A :class:`CasePlugin` exposes optional deterministic hooks that the
pipeline can call at well-known failure / dispatch points. The hooks
return ``None`` when the plugin does not handle the case at hand; the
registry tries each plugin in registration order and returns the first
non-``None`` result.

Hook surface (all optional, all may return ``None``)
----------------------------------------------------
``fallback_code(step)``
    Deterministic Python script to run when the LLM-generated code for
    a given :class:`AnalysisStep` fails repeatedly.

``repair_code(step, code, run_log)``
    Targeted repair of LLM-generated code given the original script
    and its run log / traceback.

``summary_repair(step, step_summary, df)``
    Deterministically patch missing numeric fields in a step's
    ``step_summary`` (e.g. primary odds ratio + CI for an association
    step).

``column_aliases()``
    Map of canonical concept name → accepted column-name aliases the
    plugin understands. The pipeline merges these into its column
    detection layer (currently a hardcoded list in
    ``_generic_clustering_fallback_code``).

``v15_task_template(task_key)``
    Inline Python template for one of the v15 task families
    (``"lactate"``, ``"kdigo"``, ``"creatinine"``, ``"vitals"``...).

Naming
------
Plugins use ``snake_case`` ``name`` strings (e.g. ``"lactate_map_vaso"``)
which become both the plugin id and the directory name under
``case_plugins/``.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from ..schema import AnalysisStep, ResearchContext


@runtime_checkable
class CasePlugin(Protocol):
    """Protocol for a case-specific deterministic-fallback plugin.

    Every method is **optional**: implementations may return ``None``
    when they don't handle the given dispatch. The registry skips to
    the next plugin in that case.
    """

    name: ClassVar[str]

    def matches(
        self,
        *,
        context: "ResearchContext",
        step: Optional["AnalysisStep"] = None,
    ) -> bool:
        """Return ``True`` if this plugin recognises the
        research question / step. The registry uses this as a coarse
        filter before consulting any hook below.
        """
        ...

    # All hooks below are optional. The Protocol can't enforce
    # "method may or may not be defined"; concrete plugins simply do
    # not implement what they don't support. ``CasePluginRegistry``
    # checks via ``hasattr`` before invoking.


class NullCasePlugin:
    """No-op plugin used as a sentinel in tests / type checks.

    Always reports ``matches() -> False``; implements no hooks.
    Useful when constructing a pipeline that explicitly disables any
    case-specific behaviour while still satisfying optional typing.
    """

    name: ClassVar[str] = "_null"

    def matches(
        self,
        *,
        context: "ResearchContext",
        step: Optional["AnalysisStep"] = None,
    ) -> bool:
        return False


class CasePluginRegistry:
    """Registry the pipeline asks at every former hardcoded dispatch.

    Plugins are tried **in registration order** for any single hook;
    the first plugin to return a non-``None`` result wins. This is the
    same precedence model used by middleware chains and routing tables.

    The registry is intentionally tiny and dependency-free so it can
    be constructed before the pipeline (e.g. in a fixture) and shared
    across multiple ``ResearchAgentPipeline`` instances.
    """

    def __init__(self, plugins: Optional[Sequence[CasePlugin]] = None) -> None:
        self._plugins: List[CasePlugin] = list(plugins or [])

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, plugin: CasePlugin) -> None:
        """Append ``plugin`` to the chain. Order matters: the first
        matching plugin handles the dispatch.
        """
        self._plugins.append(plugin)

    def names(self) -> List[str]:
        """Return the registered plugin names in order. Handy for
        manifests and debug logs.
        """
        return [getattr(p, "name", type(p).__name__) for p in self._plugins]

    def __len__(self) -> int:
        return len(self._plugins)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self._plugins)

    # ------------------------------------------------------------------
    # Hook dispatch
    # ------------------------------------------------------------------

    def _matching(
        self,
        *,
        context: "ResearchContext",
        step: Optional["AnalysisStep"] = None,
    ) -> List[CasePlugin]:
        return [p for p in self._plugins if p.matches(context=context, step=step)]

    def fallback_code(
        self,
        *,
        context: "ResearchContext",
        step: "AnalysisStep",
    ) -> Optional[str]:
        """First plugin that produces a deterministic Python script."""
        for plugin in self._matching(context=context, step=step):
            fn = getattr(plugin, "fallback_code", None)
            if fn is None:
                continue
            result = fn(step)
            if result is not None:
                return result
        return None

    def repair_code(
        self,
        *,
        context: "ResearchContext",
        step: "AnalysisStep",
        code: str,
        run_log: str,
    ) -> Optional[str]:
        """First plugin that produces a repaired Python script for a
        specific failure mode of the original LLM-generated code.
        """
        for plugin in self._matching(context=context, step=step):
            fn = getattr(plugin, "repair_code", None)
            if fn is None:
                continue
            result = fn(step, code, run_log)
            if result is not None:
                return result
        return None

    def summary_repair(
        self,
        *,
        context: "ResearchContext",
        step: "AnalysisStep",
        step_summary: Dict[str, Any],
        df: "Optional[pd.DataFrame]" = None,
    ) -> Optional[Dict[str, Any]]:
        """First plugin that produces a patched ``step_summary`` for a
        step whose original summary is missing required numeric fields.
        """
        for plugin in self._matching(context=context, step=step):
            fn = getattr(plugin, "summary_repair", None)
            if fn is None:
                continue
            result = fn(step, step_summary, df)
            if result is not None:
                return result
        return None

    def v15_task_template(
        self,
        *,
        context: "ResearchContext",
        task_key: str,
    ) -> Optional[str]:
        """First plugin that supplies an inline Python template for
        a v15 task family (``"lactate"``, ``"kdigo"`` ...).
        """
        for plugin in self._matching(context=context, step=None):
            fn = getattr(plugin, "v15_task_template", None)
            if fn is None:
                continue
            result = fn(task_key)
            if result is not None:
                return result
        return None

    def column_aliases(
        self,
        *,
        context: "ResearchContext",
    ) -> Dict[str, List[str]]:
        """Merge the column-alias maps of every plugin that matches.

        Order matters: later plugins extend / override earlier ones for
        the same canonical key. Plugins that don't implement
        ``column_aliases`` contribute nothing.
        """
        merged: Dict[str, List[str]] = {}
        for plugin in self._matching(context=context, step=None):
            fn = getattr(plugin, "column_aliases", None)
            if fn is None:
                continue
            for canonical, aliases in (fn() or {}).items():
                merged.setdefault(canonical, []).extend(aliases)
        return merged


__all__ = ["CasePlugin", "CasePluginRegistry", "NullCasePlugin"]

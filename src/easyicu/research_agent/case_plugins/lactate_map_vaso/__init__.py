"""Lactate / MAP / vasopressor → mortality case plugin.

Implements :class:`~easyicu.research_agent.fallback.CasePlugin` so a
``ResearchAgentPipeline`` configured with this plugin gains the
deterministic fallbacks and column aliases that used to be hardcoded
directly into ``pipeline.py``.

Activate via::

    from easyicu.research_agent.case_plugins.lactate_map_vaso import plugin
    from easyicu.research_agent.fallback import CasePluginRegistry

    registry = CasePluginRegistry([plugin])
    # ... pass registry to ResearchAgentPipeline once P4 wires it through.

The plugin is **inactive** until it is explicitly registered, so a
default ``ResearchAgentPipeline()`` carries no bias toward this paper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional

from . import fallbacks as _fb

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...schema import AnalysisStep, ResearchContext


class LactateMapVasoPlugin:
    """Case plugin for the lactate / MAP / vasopressor mortality paper."""

    name: ClassVar[str] = "lactate_map_vaso"

    # ------------------------------------------------------------------
    # Plugin protocol hooks
    # ------------------------------------------------------------------

    def matches(
        self,
        *,
        context: "ResearchContext",
        step: "Optional[AnalysisStep]" = None,
    ) -> bool:
        """Coarse filter: this plugin only handles steps / contexts
        whose research design matches the lactate / MAP / vasopressor
        study.

        The detection is intentionally permissive: any of the columns
        ``lactate_max_24h`` / ``map_min_24h`` / ``vaso_any_24h`` in the
        cohort, or any of the step ids the original hardcoded
        dispatcher recognised (``t04_lactate_mortality_association``,
        ``t15_norepinephrine_dose_response``, ``t02_age_stratified*``),
        flips the match on.
        """
        cohort = getattr(context, "cohort", None)
        column_names = set()
        for attr in ("variables", "id_columns", "time_columns", "outcome_columns"):
            seq = getattr(cohort, attr, None) or []
            for entry in seq:
                column_names.add(getattr(entry, "name", None) or str(entry))
        lactate_columns = {"lactate_max_24h", "map_min_24h", "vaso_any_24h"}
        if lactate_columns & column_names:
            return True
        if step is not None:
            sid = (step.step_id or "").lower()
            if any(
                token in sid
                for token in (
                    "lactate_mortality_association",
                    "norepinephrine_dose_response",
                    "age_stratified",
                )
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Deterministic fallbacks
    # ------------------------------------------------------------------

    def fallback_code(self, step: "AnalysisStep") -> Optional[str]:
        """Pick a fallback by step intent / id.

        Currently no step-level dispatch is wired through this hook —
        the pipeline still calls the underlying helpers directly via
        ``v15_task_template`` and the named code-repair helpers below.
        Returning ``None`` lets the existing pipeline logic continue.
        """
        return None

    def repair_code(
        self,
        step: "AnalysisStep",
        code: str,
        run_log: str,
    ) -> Optional[str]:
        """No-op for now. Named repairs (e.g. norepinephrine /
        age-stratified) are still dispatched from
        ``_deterministic_runner_repair`` in :mod:`pipeline`.
        """
        return None

    def v15_task_template(self, task_key: str) -> Optional[str]:
        """Inline analysis template for the v15 task families that
        the lactate / MAP / vasopressor study covers
        (``"lactate"``, ``"kdigo"``, ``"creatinine"``, ``"vitals"``).
        Returns ``None`` for task keys this plugin doesn't recognise.
        """
        return _fb._generic_v15_task_fallback_code(task_key)

    # ------------------------------------------------------------------
    # Column aliases
    # ------------------------------------------------------------------

    def column_aliases(self) -> Dict[str, List[str]]:
        """Canonical concept → accepted column-name aliases for the
        lactate / MAP / vasopressor study.

        Picked up by the column-detection layer (currently the
        hardcoded ``column_specs`` block in
        ``_generic_clustering_fallback_code``); merging through the
        registry rather than hardcoding lets new studies add their
        own aliases without touching pipeline source.
        """
        return {
            "lactate": ["lactate_max_24h", "lactate", "lact"],
            "map": ["map_min_24h", "map", "mean_arterial_pressure"],
            "vasopressor": [
                "vaso_any_24h",
                "vasopressor",
                "norepi",
                "norepinephrine",
                "pressor",
            ],
            "sofa2": ["sofa2_max_24h", "sofa2", "sofa_total", "sofa"],
            "creatinine": ["creat_max_24h", "creat_median_24h", "creatinine"],
            "death": ["death", "icu_mortality", "hospital_mortality"],
        }


# Module-level instance — the public entry point users import.
plugin = LactateMapVasoPlugin()


__all__ = ["LactateMapVasoPlugin", "plugin", "fallbacks"]


from . import fallbacks  # noqa: E402,F401  (re-export for tests)

"""Main page registry for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Callable


def build_main_page_registry(get_text: Callable[[str], str]) -> list[dict[str, str]]:
    """Return the ordered top-level pages shown in the main tab bar.

    Tab IA (since the 2026-05 Phase A/B redesign, then revised: Tutorial
    restored as a top tab):

    * **Tutorial** is the leftmost top tab again so first-time users can
      find the data-preparation workflow guide without digging into the
      sidebar. The page is also still reachable via the sidebar
      "📚 Workflow Help" button and via ``_scroll_to_tab='tutorial'``
      nav requests.
    * **Cross-DB Benchmark** is a top tab because it requires ≥2
      database roots, structurally different from the other Cohort
      Statistics panels.
    * **Cohort Analysis → Cohort Statistics** (rename) shows 4 subtabs
      (Groups / Coverage / Snapshot / SOFA Δ).
    """
    return [
        {"key": "tutorial", "label": get_text("home")},
        {"key": "quick_viz", "label": get_text("quick_visualization")},
        {"key": "cohort", "label": get_text("cohort_compare")},
        {"key": "cross_db", "label": get_text("cross_db_benchmark")},
        {"key": "research_agent", "label": get_text("research_agent")},
    ]

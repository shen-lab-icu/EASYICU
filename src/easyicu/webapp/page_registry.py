"""Main page registry for the EasyICU Streamlit app."""

from __future__ import annotations

from typing import Callable


def build_main_page_registry(get_text: Callable[[str], str]) -> list[dict[str, str]]:
    """Return the ordered top-level pages shown in the main tab bar.

    Tab IA (2026-05 revision):

    * **Data Extraction** is the first page and owns the demo/local/code
      entry decisions plus the concept dictionary.
    * **Data Visualization** is represented in the sidebar as a group
      containing Patient Review, Cohort Statistics, and Cross-DB Benchmark.
      The hidden fallback radio keeps those pages as routable top-level keys.
    * **Agent Projects** is the user-facing project/run layer. The route key
      remains ``research_agent`` for deep-link and test compatibility.
    """
    return [
        {"key": "tutorial", "label": get_text("home")},
        {"key": "quick_viz", "label": get_text("quick_visualization")},
        {"key": "cohort", "label": get_text("cohort_compare")},
        {"key": "cross_db", "label": get_text("cross_db_benchmark")},
        {"key": "research_agent", "label": get_text("agent_projects")},
    ]

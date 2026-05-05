"""EHRFlowBench-style benchmark fixtures for the EasyICU research agent.

The bench harness lives in ``tools/run_research_agent_bench.py`` and
the per-item fixtures in ``tests/bench/items.py``. See the docstring
of either for orientation.
"""

from .items import BENCH_ITEMS, BenchItem  # noqa: F401

__all__ = ["BENCH_ITEMS", "BenchItem"]

"""Register Case B cohort patterns for the research-agent pilot runner."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from easyicu.research_agent.cohort.schema import (
    PatternRegistry,
    default_pattern_registry,
)


CASE_DIR = Path(__file__).resolve().parent
COHORT_PATTERNS_PATH = CASE_DIR / "cohort_patterns.json"
CASE_CONFIG_PATH = CASE_DIR / "case_config.yaml"


def register_patterns(registry: Optional[PatternRegistry] = None) -> None:
    """Register case-owned CTAS patterns for Case B.

    The shared framework intentionally ships with no named cohort patterns.
    This generic function is the explicit case-level hook used by the benchmark runner
    before planning.
    """

    target = registry or default_pattern_registry()
    target.register_from_file(COHORT_PATTERNS_PATH)


register_case_b_patterns = register_patterns


__all__ = [
    "CASE_CONFIG_PATH",
    "COHORT_PATTERNS_PATH",
    "register_case_b_patterns",
    "register_patterns",
]

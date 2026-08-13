"""Dependency-neutral contract for independent novelty appraisal.

Owner: research-agent planning authority.
Public contract: the exact comparison dimensions that an external review must
complete before either a plan or an executed manuscript may claim novelty.
Allowed dependencies: none.  Consumers must not maintain local subsets.
"""

from __future__ import annotations


NOVELTY_REVIEW_DIMENSIONS: tuple[str, ...] = (
    "population_and_setting",
    "exposure_definition_and_time_zero",
    "outcome_and_estimand",
    "analysis_and_robustness_route",
    "data_source_and_transportability",
    "clinical_decision_or_methodological_contribution",
)


__all__ = ["NOVELTY_REVIEW_DIMENSIONS"]

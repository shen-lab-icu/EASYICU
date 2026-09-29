"""Which outline shapes can describe phenotypes by outcome.

Owner
-----
This module owns the outline-stage rules for ``phenotyping.outcome_by_cluster``,
checked before a foundation is sealed or a step call is spent.  Requested
outcomes after a cross-sectional cluster solution need one separate
description step.  A model-coded trajectory primary has no host-owned label
source to describe.  The executed step itself belongs to the host comparison
contract (:mod:`..contracts.phenotype_comparison`).
"""

from __future__ import annotations

from typing import Iterable

from ..contracts.phenotype_comparison import COMPARISON_ACTION
from ..contracts.trajectory_design import (
    TRAJECTORY_OUTCOME_DESCRIPTION_RULE,
    TRAJECTORY_PRIMARY_ACTION,
)
from .progressive_contract import ProgressivePlanCompileError, ProgressivePlanOutline

__all__ = ["validate_outline_phenotype_comparison"]

_CLUSTER_PRIMARY_ACTION = "phenotyping.cluster_solution"


def _primaries(outline: ProgressivePlanOutline, action: str) -> list[str]:
    return [
        step.step_id for step in outline.steps
        if step.scientific_action_id == action
        and step.planned_analysis_role == "primary"
    ]


def validate_outline_phenotype_comparison(
    outline: ProgressivePlanOutline, *, requested_outcomes: Iterable[str]
) -> None:
    """Refuse an outline whose outcome description no host owner can execute."""

    primary_clusters = _primaries(outline, _CLUSTER_PRIMARY_ACTION)
    required_cluster_outcomes = set(requested_outcomes)
    if primary_clusters and required_cluster_outcomes:
        comparisons = [step for step in outline.steps if step.scientific_action_id == COMPARISON_ACTION]
        if not (
            len(primary_clusters) == 1 and len(comparisons) == 1
            and comparisons[0].planned_analysis_role == "secondary"
            and comparisons[0].module_id == "custom_analysis"
            and primary_clusters[0] in comparisons[0].depends_on
            and required_cluster_outcomes.issubset(comparisons[0].variable_names)
        ):
            raise ProgressivePlanCompileError(
                "progressive_outline_phenotype_comparison_incomplete",
                "Requested post-clustering outcomes require one separate secondary phenotyping.outcome_by_cluster step, "
                "directly dependent on the primary cluster solution and naming every requested outcome. "
                "A fit/profile/figure step is not the comparison owner.", path="steps",
                findings=({"required_outcomes": sorted(required_cluster_outcomes), "primary_step_ids": primary_clusters},),
            )
    trajectory_primaries = _primaries(outline, TRAJECTORY_PRIMARY_ACTION)
    if trajectory_primaries and any(
        step.scientific_action_id == COMPARISON_ACTION for step in outline.steps
    ):
        raise ProgressivePlanCompileError(
            "progressive_outline_trajectory_comparison_unowned",
            "phenotyping.outcome_by_cluster describes only frozen cross-sectional assignments or the "
            "signed fixed-window suite's frozen trajectory labels, which the host wires. A model-coded "
            "phenotyping.trajectory_feature_clustering primary has neither. "
            + TRAJECTORY_OUTCOME_DESCRIPTION_RULE, path="steps",
            findings=({"primary_step_ids": trajectory_primaries},),
        )

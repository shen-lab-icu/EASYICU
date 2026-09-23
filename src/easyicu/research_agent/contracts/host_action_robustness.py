"""Robustness axes a host-owned scientific action executes under a published policy.

Owner
-----
Sibling of :mod:`sealed_suite_robustness`.  A sealed suite prespecifies its
sensitivity design under a runtime-contract digest.  Some host actions
prespecify theirs in a published, versioned host policy instead: a plan may
include or omit the action, but it cannot choose its settings, and the owning
executor claims the step only in its exact declared shape.  The
cross-sectional phenotyping owner replays its silhouette selection over the
fixed candidate-k grid of ``easyicu.cross_sectional_phenotyping_policy`` and
measures fixed-seed subsample and diagonal-GMM agreement at the primary K --
the "alternative cluster number" and "resampling stability" alternatives the
phenotyping playbook asks for.  The static prediction owner evaluates its
sealed held-out predictions across a fixed threshold grid against treat-all
and treat-none -- the prediction playbook's "threshold and decision-curve
analysis".

The review layer only recognised StudyContext specs and sealed suites, so a
plan that executed both of those still read as "no typed, executable
sensitivity authority was prespecified".  This module is the closed join
between such a step and the axes its owner actually computes.  It credits
nothing on an action id alone: an axis counts only for a step the owning
executor would claim, through the same dependency-neutral predicate the
executor uses, and only when the plan also carries exactly one host-owned
primary whose sealed product that step reads.

The axis names reuse the sealed-suite review vocabulary; like those, they are
not values a user can declare as ``PrespecifiedSensitivitySpec.axis``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from .phenotyping_execution import cross_sectional_phenotyping_owns_step
from .phenotyping_features import PHENOTYPING_PRIMARY_ACTION
from .prediction_execution import PREDICTION_PRIMARY_ACTION, static_prediction_owns_step

__all__ = [
    "HOST_ACTION_ROBUSTNESS_AXES",
    "host_action_prespecified_axes",
]

#: Host action -> the robustness axes its owner computes under the published
#: policy.  Each entry names only what the executor actually runs.
HOST_ACTION_ROBUSTNESS_AXES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        # Silhouette selection replayed over every admissible cluster count of
        # the fixed candidate grid on the sealed primary matrix.
        "phenotyping.k_selection": ("model_specification",),
        # Fixed-seed subsampling without replacement plus diagonal-GMM
        # agreement at the primary K; conditional agreement, not external
        # reproducibility, and the owner says so in its receipt.
        "phenotyping.cluster_stability": ("resampling_stability",),
        # Net benefit over the owner's fixed threshold grid on the sealed
        # held-out predictions, against treat-all and treat-none.
        "prediction.decision_curve": ("decision_threshold",),
    }
)

#: Host action -> (claim predicate, primary action whose product it replays).
_OWNERS: Mapping[str, tuple[Callable[[Any], bool], str]] = MappingProxyType(
    {
        "phenotyping.k_selection": (
            cross_sectional_phenotyping_owns_step,
            PHENOTYPING_PRIMARY_ACTION,
        ),
        "phenotyping.cluster_stability": (
            cross_sectional_phenotyping_owns_step,
            PHENOTYPING_PRIMARY_ACTION,
        ),
        "prediction.decision_curve": (
            static_prediction_owns_step,
            PREDICTION_PRIMARY_ACTION,
        ),
    }
)


def _action(step: Any) -> str:
    return str(getattr(step, "scientific_action_id", "") or "").strip()


def host_action_prespecified_axes(steps: Iterable[Any]) -> tuple[str, ...]:
    """The distinct axes host-owned robustness actions contribute to a plan.

    Returns ``()`` for anything the owning executor would not claim, and for
    every downstream action when the plan does not carry exactly one
    host-owned primary for it to replay.
    """

    plan_steps = tuple(steps or ())
    axes: dict[str, None] = {}
    for step in plan_steps:
        action = _action(step)
        declared = HOST_ACTION_ROBUSTNESS_AXES.get(action)
        owner = _OWNERS.get(action)
        if not declared or owner is None:
            continue
        owns, primary_action = owner
        if not owns(step):
            continue
        primaries = [
            candidate
            for candidate in plan_steps
            if _action(candidate) == primary_action and owns(candidate)
        ]
        if len(primaries) != 1:
            continue
        for axis in declared:
            axes.setdefault(axis, None)
    return tuple(axes)

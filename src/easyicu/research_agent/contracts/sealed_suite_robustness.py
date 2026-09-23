"""Robustness axes that a sealed host suite prespecifies and executes itself.

Owner
-----
A sealed scientific runtime authority fixes its own sensitivity design before
any result is seen, and its executors refuse to run without the contract digest
that design is sealed under.  That is a *stronger* prespecification than a
StudyContext sensitivity spec, which the reviewer can still revise -- but the
review layer only recognised the StudyContext form, so a study whose robustness
is carried entirely by a signed suite read as "no typed, executable sensitivity
authority was prespecified".

This module is the closed join between a signed step and the axes its suite
actually executes.  It credits nothing on a method name alone: an unsigned
draft may spell the same method, so an axis is credited only for a step that
carries the runtime-contract rule reference, which only ``bind_plan`` attaches
and ``validate_plan`` verifies.

The axis names below extend the *review* vocabulary only. They are deliberately
not values of ``PrespecifiedSensitivitySpec.axis``: a user cannot declare them,
because a user cannot prespecify a sealed suite's internals.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Iterable

__all__ = [
    "SEALED_RUNTIME_CONTRACT_REF_PATTERN",
    "SEALED_SUITE_ROBUSTNESS_AXES",
    "sealed_runtime_contract_sealed",
    "sealed_suite_prespecified_axes",
]

SEALED_RUNTIME_CONTRACT_REF_PATTERN = re.compile(
    r"^scientific_runtime_contract:[0-9a-f]{64}$"
)

#: Signed method -> the robustness axes that method's suite prespecifies and
#: executes. Each entry names only what the executor actually computes.
SEALED_SUITE_ROBUSTNESS_AXES = MappingProxyType(
    {
        # Landmark eligibility (alive at the landmark, negative event times
        # excluded) is a prespecified timing design; the interval time-varying
        # Cox fitted beside the constant-hazard model is an alternative hazard
        # specification of the same estimand.
        "signed_landmark_survival_suite": ("timing", "model_specification"),
        # The sealed candidate grid fits every admissible cluster count and
        # fails closed when the BIC optimum sits at the upper boundary, which
        # is the "alternative cluster number" the phenotyping playbook asks for.
        "observed_data_diagonal_gaussian_mixture_candidate_selection": (
            "model_specification",
        ),
        # Subsampling with a sealed seed derivation, a mean adjusted-Rand
        # threshold, and no post-hoc rescue.
        "trajectory_cluster_stability_characterization": ("resampling_stability",),
    }
)


def sealed_runtime_contract_sealed(rule_refs: Iterable[str]) -> bool:
    """Whether a step carries a runtime-contract reference from ``bind_plan``."""

    return any(
        SEALED_RUNTIME_CONTRACT_REF_PATTERN.match(str(ref or "").strip())
        for ref in rule_refs or ()
    )


def sealed_suite_prespecified_axes(
    *, method: str, rule_refs: Iterable[str]
) -> tuple[str, ...]:
    """The axes one signed step contributes, or ``()`` for anything unsigned."""

    head = str(method or "").strip().casefold().split(" with ", 1)[0]
    axes = SEALED_SUITE_ROBUSTNESS_AXES.get(head)
    if not axes or not sealed_runtime_contract_sealed(rule_refs):
        return ()
    return tuple(axes)

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

A suite may also declare an analysis per signing rather than per method: the
landmark suite's prevalence-definition sensitivity analysis.  Its axis is
credited only for a step that declares the product, and this module owns the
rule that fixes its hours.
"""

from __future__ import annotations

import math
import re
from types import MappingProxyType
from typing import Iterable

__all__ = [
    "EXPOSURE_ONSET_HOURS_PRODUCT",
    "PREVALENCE_SENSITIVITY_PRODUCT",
    "PREVALENCE_SENSITIVITY_RULE",
    "SEALED_RUNTIME_CONTRACT_REF_PATTERN",
    "SEALED_SUITE_PRODUCT_AXES",
    "SEALED_SUITE_ROBUSTNESS_AXES",
    "prevalence_sensitivity_cutoffs_hours",
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
        # The continuous-exposure suite applies the same landmark design; its
        # interval model and the spline check of its linear term are
        # alternative specifications of the same per-unit estimand.
        "signed_landmark_continuous_survival_suite": ("timing", "model_specification"),
        # The sealed candidate grid fits every admissible cluster count and
        # fails closed when the BIC optimum sits at the upper boundary, which
        # is the "alternative cluster number" the phenotyping playbook asks for.
        "observed_data_diagonal_gaussian_mixture_candidate_selection": (
            "model_specification",
        ),
        "observed_data_mixed_mode_latent_class_candidate_selection": (
            "model_specification",
        ),
        # Subsampling with a sealed seed derivation, a mean adjusted-Rand
        # threshold, and no post-hoc rescue.
        "trajectory_cluster_stability_characterization": ("resampling_stability",),
    }
)


#: The landmark suite's prevalence-definition sensitivity analysis: each fit
#: also excludes exposed records first recorded by a later hour after time
#: zero and refits the adjusted Cox model.  Exposure present on arrival but
#: first recorded hours later is otherwise counted as incident; no single
#: hour separates the two, so the suite reports how the estimate moves.
PREVALENCE_SENSITIVITY_PRODUCT = "table:landmark_prevalence_sensitivity"
#: Its outcome-blind companion: the exposed group's first-record hours.
EXPOSURE_ONSET_HOURS_PRODUCT = "table:landmark_exposure_onset_hours"
#: The hours: whole hours at a quarter and at a half of the exposure window.
PREVALENCE_SENSITIVITY_RULE = "whole_hours_at_quarter_and_half_of_exposure_window"

#: Signed method -> a product it declares per signing -> the axis that product
#: executes.  A suite signed without the product is not credited with it.
SEALED_SUITE_PRODUCT_AXES = MappingProxyType(
    {
        "signed_landmark_survival_suite": MappingProxyType(
            {PREVALENCE_SENSITIVITY_PRODUCT: "prevalence_definition"}
        ),
    }
)


def prevalence_sensitivity_cutoffs_hours(window_end_hours: float) -> tuple[float, ...]:
    """The hours ``PREVALENCE_SENSITIVITY_RULE`` gives for one exposure window.

    Whole hours at a quarter and at a half of the window, after time zero and
    before the window's end, without a duplicate; empty when the window is too
    short to hold one.
    """

    hours = {
        float(math.floor(window_end_hours / 4.0)),
        float(math.floor(window_end_hours / 2.0)),
    }
    return tuple(sorted(hour for hour in hours if 0.0 < hour < window_end_hours))


def sealed_runtime_contract_sealed(rule_refs: Iterable[str]) -> bool:
    """Whether a step carries a runtime-contract reference from ``bind_plan``."""

    return any(
        SEALED_RUNTIME_CONTRACT_REF_PATTERN.match(str(ref or "").strip())
        for ref in rule_refs or ()
    )


def sealed_suite_prespecified_axes(
    *, method: str, rule_refs: Iterable[str], expected_outputs: Iterable[str] = ()
) -> tuple[str, ...]:
    """The axes one signed step contributes, or ``()`` for anything unsigned."""

    head = str(method or "").strip().casefold().split(" with ", 1)[0]
    axes = SEALED_SUITE_ROBUSTNESS_AXES.get(head)
    if not axes or not sealed_runtime_contract_sealed(rule_refs):
        return ()
    declared = set(expected_outputs or ())
    return tuple(axes) + tuple(
        axis
        for product, axis in SEALED_SUITE_PRODUCT_AXES.get(head, {}).items()
        if product in declared and axis not in axes
    )

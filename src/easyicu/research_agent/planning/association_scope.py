"""Shared scope of the minimal baseline association, independent of plan transport."""

from typing import Optional

from ..authority.declared_levels import closed_planning_levels_for
from ..contracts.model_terms import level_spelling
from ..schema import ResearchContext, VariableRole
from ..research_context.materialization_window import host_materialization_window_hours
from ..research_context.temporal_semantics import window_extends_after_anchor
from .adjustment_authority import (
    baseline_static_grouping, owner_declared_baseline_static, primary_landmark_hours,
)
from .progressive_contract import ProgressivePlanCompileError

MAX_ASSOCIATION_EXPOSURE_LEVELS = 24


def baseline_association_context(context) -> bool:
    exposure, outcome = context.primary_exposure, context.target_outcome
    variables = {item.name: item for item in context.variables}
    if (not exposure or not outcome or exposure not in variables or outcome not in variables
        or len(context.cohort.id_columns) != 1 or primary_landmark_hours(context) is not None
        or post_baseline_exposure(context)[0]):
        return False
    outcome_levels = [level_spelling(value) for value in closed_planning_levels_for(name=outcome, variables=variables)]
    if outcome_levels not in (["0", "1"], ["false", "true"]):
        return False
    exposure_levels = closed_planning_levels_for(name=exposure, variables=variables)
    if exposure_levels:
        return 2 <= len(exposure_levels) <= MAX_ASSOCIATION_EXPOSURE_LEVELS
    variable = variables[exposure]
    role = str(getattr(variable.role, "value", variable.role))
    return (role in {"demographic", "lab", "vital", "composite_score", "other"}
            and not variable.is_ordinal and any(token in str(variable.dtype).lower() for token in ("float", "int", "double", "decimal")))


def post_baseline_exposure(
    context: ResearchContext, exposure: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Classify post-zero exposure opportunity from owner-issued coordinates.

    When no exposure is supplied, classify the context's primary exposure.
    A concept's clinical definition window remains the first authority.  The
    Web host may also bind an exact physical feature-materialization window in
    ``data_constraints`` before a concrete wide-column descriptor exists (for
    example while candidate planning is metadata-only).  That outer window is
    *not* promoted to a clinical-definition anchor; it only establishes that
    a primary feature can be observed after ICU admission, so association
    plans must close early-event/exposure opportunity.  The window is read by
    ``host_materialization_window_hours``, as the cohort-predicate rule reads
    it. A lab or vital without either window has unknown timing: stop through
    the typed planning refusal rather than claiming baseline availability.
    """

    exposure = str(
        (context.primary_exposure if exposure is None else exposure) or ""
    ).strip()
    if not exposure:
        return False, None
    descriptor = context.variable(exposure)
    window = str(getattr(descriptor, "analysis_window", "") or "").strip()
    if window:
        return window_extends_after_anchor(window), window
    if descriptor is not None and (
        owner_declared_baseline_static(descriptor)
        or baseline_static_grouping(descriptor, context)
    ):
        # The outer window bounds *measurement* opportunity inside the stay.
        # It cannot move an owner-declared baseline attribute (age, sex,
        # admission type) after time zero, nor a grouping formed only from
        # them (age groups), and the host already relies on that same
        # declaration for baseline covariate timing.  A concept whose own
        # window is post-baseline was returned above, so this never overrides
        # physical window evidence.
        return False, None

    # This detects a risk in the host-declared physical window; it grants no
    # execution authority. An absent confirmation cannot hide that risk.
    hours = host_materialization_window_hours(context)
    if hours is None:
        if descriptor is not None and descriptor.role in {VariableRole.LAB, VariableRole.VITAL}:
            raise ProgressivePlanCompileError(
                "progressive_exposure_window_required",
                "the exposure is a lab or vital measurement with no analysis window and no "
                "host materialization window, so its timing relative to time zero is unknown",
                path=f"variables.{exposure}.analysis_window",
                concepts=(exposure,), element="exposure_window", question_required=True,
            )
        return False, None
    # This label deliberately names the physical coordinate, rather than
    # implying a phenotype definition or a follow-up horizon.
    return True, f"outer_materialization:icu_admission[0,{hours:g}]h"

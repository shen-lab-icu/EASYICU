"""Origin-aware adjustment-set authority for analysis planning.

Owner
-----
The typed :class:`~easyicu.research_agent.schema.UserPreferences` contract owns
whether an adjustment set is still open to Planner selection or was fixed by
the user-facing study configuration.  This module compiles that small public
contract and validates Planner output.  It does not select covariates, inspect
rows, or infer an adjustment set from available demographic columns.

The validator is intentionally called both while structured Planner retry is
available and again at the execution boundary.  A stored/resumed plan must not
be able to bypass the same scientific authority that constrained a fresh plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

from ..authority.declared_levels import observed_levels_for
from ..research_context.typed import declared_domain_for_variable


# A repeated concept is materialized into explicit value summaries before the
# Planner sees it.  Those suffixes change representation, not the user-owned
# covariate identity.  Measurement/count/time companions are deliberately not
# included: ``charlson_n`` or ``charlson_measured`` is not the Charlson score.
_VALUE_AGGREGATION_SUFFIXES = ("_first", "_mean", "_max", "_min")
_MODEL_TERM_INELIGIBLE_ROLES = frozenset(
    {"id", "time", "index", "meta", "outcome"}
)
_MODEL_TERM_DYNAMIC_ROLES = frozenset(
    {"vital", "lab", "intervention", "ordinal_score", "composite_score"}
)
HostTemporalRole = Literal["baseline_static", "at_or_before_time_zero"]
# ``first_24h``, ``0-24h``, ``0_24h``, ``24h``: the trailing hour bound of a
# named window is the only fact this owner reads from a window label.
_WINDOW_END_HOURS = re.compile(r"(?:^|[^0-9])(\d+(?:\.\d+)?)\s*h(?:ours?)?\s*$", re.IGNORECASE)
#: The research-context builder's own label form, ``icu_admission[start,end]h``.
_WINDOW_INTERVAL_HOURS = re.compile(
    r"icu_admission\[\s*-?\d+(?:\.\d+)?\s*,\s*(\d+(?:\.\d+)?)\s*\]\s*h(?:ours?)?\s*$",
    re.IGNORECASE,
)


def primary_landmark_hours(context: Any) -> Optional[float]:
    """Return the typed landmark (hours after ICU admission) or ``None``.

    The landmark is a study-context fact: an explicit ``landmark_hours``
    preference or the prespecified landmark timing sensitivity.  Prose is not
    authority, so no text is parsed here.
    """

    preferences = getattr(context, "user_preferences", None)
    if preferences is None:
        return None
    declared = getattr(preferences, "landmark_hours", None)
    if declared is not None:
        return float(declared)
    for spec in getattr(preferences, "sensitivity_specs", ()) or ():
        if (
            str(getattr(spec, "axis", "") or "") == "timing"
            and str(getattr(spec, "strategy", "") or "") == "landmark"
            and getattr(spec, "landmark_hours", None) is not None
        ):
            return float(spec.landmark_hours)
    return None


def host_outer_feature_window_end_hours(context: Any) -> Optional[float]:
    """Return the end (hours after ICU admission) of the host-bound feature window.

    ``ResearchContext.time_windows`` are the windows the host materialized; the
    cohort carries no measurement outside the widest of them.  Windows with a
    different anchor cannot be compared with an ICU-admission landmark and are
    ignored.
    """

    ends = [
        float(window.end_hours)
        for window in (getattr(context, "time_windows", ()) or ())
        if str(getattr(window, "anchor", "") or "") == "icu_admission"
        and getattr(window, "end_hours", None) is not None
    ]
    return max(ends) if ends else None


def _analysis_window_end_hours(label: Any) -> Optional[float]:
    text = str(label or "").strip()
    if not text:
        return None
    interval = _WINDOW_INTERVAL_HOURS.search(text)
    if interval:
        return float(interval.group(1))
    match = _WINDOW_END_HOURS.search(text)
    return float(match.group(1)) if match else None


def host_window_bound_roles(
    context: Any,
    *,
    reference_hours: Optional[float],
    dynamic_roles: frozenset[str] = _MODEL_TERM_DYNAMIC_ROLES,
    outer_window_fallback_roles: frozenset[str] = _MODEL_TERM_DYNAMIC_ROLES,
) -> dict[str, HostTemporalRole]:
    """Project which variables the host can prove observed by ``reference_hours``.

    An owner-declared baseline demographic is static. A window-derived variable
    with one of ``dynamic_roles`` is available at or before the reference time
    only when its materialization window ends at or before it: its own
    ``analysis_window`` label, or -- for the clinical roles listed in
    ``outer_window_fallback_roles`` -- the outer host-bound feature window.
    Every other variable is absent from the mapping. With no reference time
    only the demographics are provable.
    """

    outer_end = host_outer_feature_window_end_hours(context)
    roles: dict[str, HostTemporalRole] = {}
    for variable in getattr(context, "variables", ()) or ():
        name = str(getattr(variable, "name", "") or "").strip()
        role = str(getattr(variable.role, "value", variable.role) or "")
        if not name:
            continue
        if role == "demographic":
            roles[name] = "baseline_static"
            continue
        if role not in dynamic_roles or reference_hours is None:
            continue
        window_end = _analysis_window_end_hours(
            getattr(variable, "analysis_window", None)
        )
        if window_end is None and role in outer_window_fallback_roles:
            window_end = outer_end
        if window_end is not None and window_end <= reference_hours:
            roles[name] = "at_or_before_time_zero"
    return roles


def host_proven_temporal_roles(context: Any) -> dict[str, HostTemporalRole]:
    """Project the covariate timing the host can prove without a rationale.

    An owner-declared baseline demographic is static.  A window-derived clinical
    measurement or score is available at or before time zero only when its
    materialization window (its own ``analysis_window`` label, else the outer
    host-bound feature window) ends at or before the typed landmark.  Every
    other variable is absent from the mapping: a generated clinical rationale
    never replaces this timing authority, so such a variable can enter an
    adjustment set only through an exact user-reviewed roster.
    """

    return host_window_bound_roles(
        context, reference_hours=primary_landmark_hours(context)
    )


def _matches_declared_covariate(declared: str, observed: str) -> bool:
    if observed == declared:
        return True
    return any(
        observed == f"{declared}{suffix}"
        for suffix in _VALUE_AGGREGATION_SUFFIXES
    )


class AdjustmentAuthorityError(ValueError):
    """A plan changed a user-locked adjustment set."""

    code = "adjustment_set_authority_mismatch"


@dataclass(frozen=True)
class AdjustmentSetAuthority:
    """Immutable projection of the exact roster and its declared author."""

    selection: Literal["planner_selectable", "exact"]
    covariates: tuple[str, ...]
    authority: Optional[Literal["user", "agent_plan"]] = None
    rationales: tuple[tuple[str, str], ...] = ()
    temporal_roles: tuple[tuple[str, str], ...] = ()
    operationalizations: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_context(cls, context: Any) -> "AdjustmentSetAuthority":
        preferences = getattr(context, "user_preferences", None)
        if preferences is None:
            return cls(selection="planner_selectable", covariates=())
        selection = str(
            getattr(preferences, "covariate_selection", "planner_selectable")
            or "planner_selectable"
        ).strip()
        covariates = tuple(
            str(value or "").strip()
            for value in (getattr(preferences, "covariates", ()) or ())
            if str(value or "").strip()
        )
        if selection not in {"planner_selectable", "exact"}:
            raise AdjustmentAuthorityError(
                "adjustment_set_authority_invalid: covariate_selection must be "
                "'planner_selectable' or 'exact'"
            )
        rationales = getattr(preferences, "covariate_rationales", {}) or {}
        temporal_roles = getattr(preferences, "covariate_temporal_roles", {}) or {}
        operationalizations = (
            getattr(preferences, "covariate_operationalizations", {}) or {}
        )
        authority = getattr(preferences, "covariate_authority", None)
        return cls(
            selection=selection,
            covariates=covariates,
            authority=authority,
            rationales=tuple((name, str(rationales[name])) for name in covariates if name in rationales),
            temporal_roles=tuple(
                (name, str(temporal_roles[name]))
                for name in covariates
                if name in temporal_roles
            ),
            operationalizations=tuple(
                (name, str(operationalizations[name]))
                for name in covariates
                if name in operationalizations
            ),
        )

    @property
    def operational_covariates(self) -> tuple[str, ...]:
        mapping = dict(self.operationalizations)
        return tuple(mapping.get(name, name) for name in self.covariates)

    @property
    def operational_rationales(self) -> dict[str, str]:
        mapping = dict(self.operationalizations)
        return {mapping.get(name, name): value for name, value in self.rationales}

    @property
    def operational_temporal_roles(self) -> dict[str, str]:
        mapping = dict(self.operationalizations)
        return {mapping.get(name, name): value for name, value in self.temporal_roles}

    def prompt_projection(self) -> dict[str, Any]:
        """Preserve authorship wherever a bound adjustment set is described.

        Exactness is a constraint on the roster, not evidence that the user
        authored it or that a plan-review action has occurred.
        """

        if self.selection != "exact":
            boundary = "Planner-selectable; available variables are not user-specified covariates."
        elif self.authority == "agent_plan":
            boundary = "Agent-selected, not user-specified; describe this as the Agent's proposed adjustment set."
        elif self.authority == "user":
            boundary = "User-authored adjustment set; preserve the exact roster."
        else:
            boundary = "Authorship is unrecorded; do not infer user or Agent authorship."
        return {
            "selection": self.selection,
            "authority": self.authority or "unrecorded",
            "scientific_covariates": list(self.covariates),
            "operational_covariates": list(self.operational_covariates),
            "operationalizations": dict(self.operationalizations),
            "authorship_boundary": boundary + " This binding is not a plan-approval receipt.",
        }

    def validate_plan(self, plan: Any) -> None:
        """Require every declared fitted model to honor an exact roster.

        ``covariates=[]`` is a meaningful scientific decision: an unadjusted
        model.  It is not equivalent to an omitted roster, and available age or
        sex columns do not authorize the Planner to add them.
        """

        if self.selection != "exact":
            return

        mismatches: list[str] = []
        operationalizations = dict(self.operationalizations)
        for step in getattr(plan, "steps", ()) or ():
            step_id = str(getattr(step, "step_id", "") or "<unnamed>")
            for requirement in getattr(step, "model_requirements", ()) or ():
                declared = getattr(requirement, "covariates", None)
                observed: Optional[tuple[str, ...]] = (
                    None
                    if declared is None
                    else tuple(str(value or "").strip() for value in declared)
                )
                roster_matches = (
                    observed is not None
                    and len(observed) == len(self.covariates)
                    and all(
                        (
                            actual == operationalizations[expected]
                            if expected in operationalizations
                            else _matches_declared_covariate(expected, actual)
                        )
                        for expected, actual in zip(self.covariates, observed)
                    )
                )
                if not roster_matches:
                    mismatches.append(
                        f"{step_id}/{getattr(requirement, 'requirement_id', '<unnamed>')}: "
                        f"declared={list(observed) if observed is not None else None!r}"
                    )

        if not mismatches:
            return
        raise AdjustmentAuthorityError(
            f"{AdjustmentAuthorityError.code}: "
            "user_preferences.covariate_selection='exact' binds "
            f"every planned model to covariates={list(self.covariates)!r}; "
            "the Planner may not add, remove, infer, or reorder covariates. "
            "Mismatches: "
            + "; ".join(mismatches[:6])
        )


def adjusted_model_term_planning_authority(
    context: Any,
    variable_names: Sequence[str],
) -> dict[str, Any]:
    """Project model-term eligibility for a bounded Planner prompt.

    This metadata-only projection cannot select an adjustment set or infer
    clinical timing. It exposes which catalog variables the current typed
    authority permits as model terms and the encodings supported by their
    sealed domains.
    """

    adjustment = AdjustmentSetAuthority.from_context(context)
    temporal_roles = adjustment.operational_temporal_roles
    authorized_time_zero = frozenset(
        name
        for name in adjustment.operational_covariates
        if adjustment.selection == "exact"
        and temporal_roles.get(name) in {"baseline_static", "at_or_before_time_zero"}
    )
    primary_exposure = str(context.primary_exposure or "").strip()
    outcome = str(context.target_outcome or "").strip()
    variables = {item.name: item for item in context.variables}
    host_timed: Mapping[str, HostTemporalRole] = host_proven_temporal_roles(context)
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for name in dict.fromkeys(str(value or "").strip() for value in variable_names):
        if not name or name in {primary_exposure, outcome}:
            continue
        variable = variables.get(name)
        if variable is None:
            continue
        role = str(getattr(variable.role, "value", variable.role) or "")
        if role in _MODEL_TERM_INELIGIBLE_ROLES:
            excluded.append({"name": name, "reason": f"semantic_role:{role}"})
            continue
        if adjustment.selection != "exact" and name not in host_timed:
            excluded.append(
                {"name": name, "reason": "planner_baseline_authority_missing"}
            )
            continue
        if (
            role in _MODEL_TERM_DYNAMIC_ROLES
            and name not in authorized_time_zero
            and host_timed.get(name) != "at_or_before_time_zero"
        ):
            excluded.append({"name": name, "reason": "time_zero_authority_missing"})
            continue
        observed = observed_levels_for(name=name, variables=variables)
        declared, declared_basis = declared_domain_for_variable(variable)
        closed_domain = observed or list(declared or ())
        eligible.append(
            {
                "name": name,
                "semantic_role": role,
                "host_temporal_role": host_timed.get(name),
                "allowed_codings": [
                    "binary" if len(closed_domain) == 2 else "categorical"
                ]
                if closed_domain
                else ["continuous"],
                "closed_domain_size": len(closed_domain) or None,
                "domain_authority": "observed" if observed else declared_basis,
            }
        )
    return {
        "selection": adjustment.selection,
        "primary_exposure": primary_exposure,
        "outcome": outcome,
        "eligible_covariates": eligible,
        "excluded_covariates": excluded,
    }


def validate_plan_against_adjustment_authority(*, plan: Any, context: Any) -> None:
    """Public fail-closed boundary used by planning and execution."""

    AdjustmentSetAuthority.from_context(context).validate_plan(plan)


__all__ = [
    "AdjustmentAuthorityError",
    "AdjustmentSetAuthority",
    "HostTemporalRole",
    "adjusted_model_term_planning_authority",
    "host_outer_feature_window_end_hours",
    "host_proven_temporal_roles",
    "host_window_bound_roles",
    "primary_landmark_hours",
    "validate_plan_against_adjustment_authority",
]

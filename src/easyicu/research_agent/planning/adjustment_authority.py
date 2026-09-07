"""User-owned adjustment-set authority for analysis planning.

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

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

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
    """Immutable projection of the user-owned adjustment-set decision."""

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
        if adjustment.selection != "exact" and role != "demographic":
            excluded.append(
                {"name": name, "reason": "planner_baseline_authority_missing"}
            )
            continue
        if role in _MODEL_TERM_DYNAMIC_ROLES and name not in authorized_time_zero:
            excluded.append({"name": name, "reason": "time_zero_authority_missing"})
            continue
        observed = observed_levels_for(name=name, variables=variables)
        declared, declared_basis = declared_domain_for_variable(variable)
        closed_domain = observed or list(declared or ())
        eligible.append(
            {
                "name": name,
                "semantic_role": role,
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
    "adjusted_model_term_planning_authority",
    "validate_plan_against_adjustment_authority",
]

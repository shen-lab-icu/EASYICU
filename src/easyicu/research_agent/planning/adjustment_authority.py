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
from typing import Any, Literal, Optional


class AdjustmentAuthorityError(ValueError):
    """A plan changed a user-locked adjustment set."""

    code = "adjustment_set_authority_mismatch"


@dataclass(frozen=True)
class AdjustmentSetAuthority:
    """Immutable projection of the user-owned adjustment-set decision."""

    selection: Literal["planner_selectable", "exact"]
    covariates: tuple[str, ...]
    rationales: tuple[tuple[str, str], ...] = ()
    temporal_roles: tuple[tuple[str, str], ...] = ()

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
        return cls(
            selection=selection,
            covariates=covariates,
            rationales=tuple((name, str(rationales[name])) for name in covariates if name in rationales),
            temporal_roles=tuple(
                (name, str(temporal_roles[name]))
                for name in covariates
                if name in temporal_roles
            ),
        )

    def validate_plan(self, plan: Any) -> None:
        """Require every declared fitted model to honor an exact roster.

        ``covariates=[]`` is a meaningful scientific decision: an unadjusted
        model.  It is not equivalent to an omitted roster, and available age or
        sex columns do not authorize the Planner to add them.
        """

        if self.selection != "exact":
            return

        mismatches: list[str] = []
        for step in getattr(plan, "steps", ()) or ():
            step_id = str(getattr(step, "step_id", "") or "<unnamed>")
            for requirement in getattr(step, "model_requirements", ()) or ():
                declared = getattr(requirement, "covariates", None)
                observed: Optional[tuple[str, ...]] = (
                    None
                    if declared is None
                    else tuple(str(value or "").strip() for value in declared)
                )
                if observed != self.covariates:
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


def validate_plan_against_adjustment_authority(*, plan: Any, context: Any) -> None:
    """Public fail-closed boundary used by planning and execution."""

    AdjustmentSetAuthority.from_context(context).validate_plan(plan)


__all__ = [
    "AdjustmentAuthorityError",
    "AdjustmentSetAuthority",
    "validate_plan_against_adjustment_authority",
]

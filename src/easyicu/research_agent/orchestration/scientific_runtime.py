"""Bind optional caller-reviewed scientific authorities for one pipeline run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..authority.current_case_scientific_runtime import (
    AssociationModelGridRuntimeAuthority,
    CurrentCaseScientificRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ..schema import AnalysisPlan, ValidationFinding
from ..trajectory.scientific_runtime_authority import (
    TrajectoryScientificRuntimeAuthority,
    load_trajectory_scientific_runtime_authority,
)


@dataclass(frozen=True)
class ScientificRuntimeAuthorities:
    """Immutable pair compiled once and shared with planners and executors."""

    trajectory: TrajectoryScientificRuntimeAuthority | None
    current_case: CurrentCaseScientificRuntimeAuthority | None

    @classmethod
    def load(
        cls,
        *,
        trajectory: Mapping[str, Any] | None,
        current_case: Mapping[str, Any] | None,
    ) -> "ScientificRuntimeAuthorities":
        return cls(
            trajectory=(
                load_trajectory_scientific_runtime_authority(trajectory)
                if trajectory is not None
                else None
            ),
            current_case=(
                load_current_case_scientific_runtime_authority(current_case)
                if current_case is not None
                else None
            ),
        )

    def validate_plan(self, plan: AnalysisPlan) -> None:
        """Preserve the precise authority-owner error for any plan drift."""

        if self.trajectory is not None:
            self.trajectory.validate_plan(plan)
        if self.current_case is not None:
            self.current_case.validate_plan(plan)

    def bind_plan(
        self,
        plan: AnalysisPlan,
    ) -> tuple[AnalysisPlan, list[ValidationFinding]]:
        """Compile host-owned wiring before the final plan is reviewed.

        Only the generic model-grid authority currently needs this operation:
        its caller owns the variant science, while product names, exact inputs,
        and the verified-tool route are mechanical consequences.  Other signed
        authorities continue to require the Planner's exact step unchanged.
        """

        authority = self.current_case
        if not isinstance(authority, AssociationModelGridRuntimeAuthority):
            return plan, []
        bound = authority.bind_plan(plan)
        step = authority.governed_step(bound)
        return bound, [
            ValidationFinding(
                validator="scientific_runtime_plan_compiler",
                severity="warning",
                message=(
                    "Compiled the prespecified association model grid into the "
                    "verified host-tool route."
                ),
                detail={
                    "reason_code": "association_model_grid_host_compiled",
                    "step_id": step.step_id,
                    "output_product": authority.output_product,
                    "variant_ids": list(authority.sensitivity_ids),
                    "execution_contract_sha256": (
                        authority.execution_contract_sha256
                    ),
                },
            )
        ]


__all__ = ["ScientificRuntimeAuthorities"]

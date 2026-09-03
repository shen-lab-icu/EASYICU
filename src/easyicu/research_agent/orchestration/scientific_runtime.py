"""Bind optional caller-reviewed scientific authorities for one pipeline run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..authority.current_case_scientific_runtime import (
    AssociationModelGridRuntimeAuthority,
    CurrentCaseScientificRuntimeAuthority,
    LandmarkSplineRuntimeAuthority,
    LandmarkSurvivalRuntimeAuthority,
    SourceFeasibilityRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ..schema import AnalysisPlan, ValidationFinding
from ..authority.time_varying_runtime import TimeVaryingRuntimeAuthority
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

        trajectory_authority = self.trajectory
        if (
            trajectory_authority is not None
            and trajectory_authority.is_development_execution_only_plan(plan)
        ):
            bound = trajectory_authority.development_execution_only_plan(
                research_question=plan.research_question
            )
            return bound, [
                ValidationFinding(
                    validator="scientific_runtime_plan_compiler",
                    severity="warning",
                    message=(
                        "Removed generic article-shaping additions and compiled "
                        "the four signed trajectory execution owners."
                    ),
                    detail={
                        "reason_code": (
                            "trajectory_development_execution_only_authority_compiled"
                        ),
                        "step_ids": [step.step_id for step in bound.steps],
                        "execution_contract_sha256": (
                            trajectory_authority.execution_contract_sha256
                        ),
                    },
                )
            ]

        authority = self.current_case
        if isinstance(authority, TimeVaryingRuntimeAuthority):
            bound = authority.bind_plan(plan)
            return bound, [ValidationFinding(
                validator="scientific_runtime_plan_compiler", severity="warning",
                message="Compiled the explicit time-updated analysis-only plan; incompatible static-model analyses are not inherited.",
                detail={"reason_code": "time_varying_exposure_host_compiled", "analysis_only": True,
                        "execution_contract_sha256": authority.execution_contract_sha256},
            )]
        if isinstance(authority, SourceFeasibilityRuntimeAuthority):
            bound = authority.development_execution_only_plan(
                research_question=plan.research_question
            )
            step = authority.governed_step(bound)
            return bound, [
                ValidationFinding(
                    validator="scientific_runtime_plan_compiler",
                    severity="warning",
                    message=(
                        "Removed generic article-shaping additions and compiled "
                        "the signed source-feasibility non-use decision."
                    ),
                    detail={
                        "reason_code": "source_feasibility_fail_closed_host_compiled",
                        "step_id": step.step_id,
                        "output_products": list(authority.plan_outputs),
                        "execution_contract_sha256": (
                            authority.execution_contract_sha256
                        ),
                    },
                )
            ]
        if isinstance(authority, LandmarkSurvivalRuntimeAuthority):
            bound = authority.bind_plan(plan)
            step = authority.governed_step(bound)
            return bound, [
                ValidationFinding(
                    validator="scientific_runtime_plan_compiler",
                    severity="warning",
                    message=(
                        "Compiled the signed landmark survival suite into one "
                        "deterministic host-tool route."
                    ),
                    detail={
                        "reason_code": "landmark_survival_suite_host_compiled",
                        "step_id": step.step_id,
                        "output_products": list(authority.plan_outputs),
                        "execution_contract_sha256": (
                            authority.execution_contract_sha256
                        ),
                    },
                )
            ]
        if isinstance(authority, LandmarkSplineRuntimeAuthority):
            bound = authority.bind_plan(plan)
            step = authority.governed_step(bound)
            return bound, [
                ValidationFinding(
                    validator="scientific_runtime_plan_compiler",
                    severity="warning",
                    message=(
                        "Compiled the signed landmark spline authority into the "
                        "verified host-tool route."
                    ),
                    detail={
                        "reason_code": "landmark_spline_host_compiled",
                        "step_id": step.step_id,
                        "output_products": list(authority.plan_outputs),
                        "execution_contract_sha256": (
                            authority.execution_contract_sha256
                        ),
                    },
                )
            ]
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

    def development_execution_only_plan(
        self,
        *,
        research_question: str,
    ) -> tuple[AnalysisPlan, ValidationFinding] | None:
        """Return a host-projected plan only when its sealed authority opts in."""

        authority = self.current_case
        trajectory_authority = self.trajectory
        if authority is None and trajectory_authority is not None:
            plan = trajectory_authority.development_execution_only_plan(
                research_question=research_question
            )
            return plan, ValidationFinding(
                validator="scientific_runtime_plan_compiler",
                severity="warning",
                message=(
                    "Used the digest-bound trajectory authority for an explicit "
                    "development execution-only run without a Planner call."
                ),
                detail={
                    "reason_code": (
                        "trajectory_development_execution_only_authority_compiled"
                    ),
                    "analysis_only": True,
                    "step_ids": [step.step_id for step in plan.steps],
                    "execution_contract_sha256": (
                        trajectory_authority.execution_contract_sha256
                    ),
                },
            )
        if isinstance(authority, LandmarkSurvivalRuntimeAuthority):
            if not authority.development_execution_only_allowed:
                return None
        elif not isinstance(authority, (SourceFeasibilityRuntimeAuthority, TimeVaryingRuntimeAuthority)):
            return None
        plan = authority.development_execution_only_plan(
            research_question=research_question
        )
        step = authority.governed_step(plan)
        return plan, ValidationFinding(
            validator="scientific_runtime_plan_compiler",
            severity="warning",
            message=(
                "Used the digest-bound current-case authority for an explicit "
                "development execution-only run without another Planner call."
            ),
            detail={
                "reason_code": (
                    "source_feasibility_development_execution_only_authority_compiled"
                    if isinstance(authority, SourceFeasibilityRuntimeAuthority)
                    else "development_execution_only_authority_compiled"
                ),
                "analysis_only": True,
                "step_id": step.step_id,
                "execution_contract_sha256": authority.execution_contract_sha256,
            },
        )


__all__ = ["ScientificRuntimeAuthorities"]

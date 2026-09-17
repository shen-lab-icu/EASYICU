"""Bind optional caller-reviewed scientific authorities for one pipeline run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..authority.current_case_scientific_runtime import (
    CurrentCaseScientificRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ..schema import AnalysisPlan, ValidationFinding
from ..trajectory.scientific_runtime_authority import (
    TrajectoryScientificRuntimeAuthority,
    load_trajectory_scientific_runtime_authority,
)


@dataclass(frozen=True)
class _RuntimePlanCompilerSpec:
    """Declarative bridge from one typed authority to orchestration output."""

    message: str
    reason_code: str
    project_research_question: bool = False
    enabled_flag: str | None = None
    governed_steps: tuple[tuple[str, str], ...] = ()
    scalar_details: tuple[tuple[str, str], ...] = ()
    sequence_details: tuple[tuple[str, str], ...] = ()
    analysis_only: bool = False


_CURRENT_CASE_PLAN_COMPILERS: Mapping[str, _RuntimePlanCompilerSpec] = {
    "time_varying_exposure_association": _RuntimePlanCompilerSpec(
        message=(
            "Compiled the explicit time-updated analysis-only plan; incompatible "
            "static-model analyses are not inherited."
        ),
        reason_code="time_varying_exposure_host_compiled",
        analysis_only=True,
    ),
    "restricted_mean_survival_difference": _RuntimePlanCompilerSpec(
        message=(
            "Validated the reviewed RMST sensitivity against its deterministic "
            "host executor."
        ),
        reason_code="rmst_contrast_host_validated",
        governed_steps=(("step_id", "governed_step"),),
        sequence_details=(("output_products", "plan_outputs"),),
        analysis_only=True,
    ),
    "source_feasibility_fail_closed": _RuntimePlanCompilerSpec(
        message=(
            "Removed generic article-shaping additions and compiled the signed "
            "source-feasibility non-use decision."
        ),
        reason_code="source_feasibility_fail_closed_host_compiled",
        project_research_question=True,
        governed_steps=(("step_id", "governed_step"),),
        sequence_details=(("output_products", "plan_outputs"),),
    ),
    "landmark_survival_suite": _RuntimePlanCompilerSpec(
        message=(
            "Compiled the signed landmark survival suite into one deterministic "
            "host-tool route."
        ),
        reason_code="landmark_survival_suite_host_compiled",
        governed_steps=(("step_id", "governed_step"),),
        sequence_details=(("output_products", "plan_outputs"),),
    ),
    "landmark_categorical_association": _RuntimePlanCompilerSpec(
        message=(
            "Compiled the categorical landmark cohort and primary association "
            "into verified host-tool routes."
        ),
        reason_code="landmark_categorical_association_host_compiled",
        governed_steps=(
            ("cohort_step_id", "governed_cohort_step"),
            ("primary_step_id", "governed_primary_step"),
        ),
    ),
    "landmark_spline_association": _RuntimePlanCompilerSpec(
        message=(
            "Compiled the signed landmark spline authority into the verified "
            "host-tool route."
        ),
        reason_code="landmark_spline_host_compiled",
        governed_steps=(("step_id", "governed_step"),),
        sequence_details=(("output_products", "plan_outputs"),),
    ),
    "association_model_grid": _RuntimePlanCompilerSpec(
        message=(
            "Compiled the prespecified association model grid into the verified "
            "host-tool route."
        ),
        reason_code="association_model_grid_host_compiled",
        governed_steps=(("step_id", "governed_step"),),
        scalar_details=(("output_product", "output_product"),),
        sequence_details=(("variant_ids", "sensitivity_ids"),),
    ),
}


_DEVELOPMENT_MESSAGE = (
    "Used the digest-bound current-case authority for an explicit development "
    "execution-only run without another Planner call."
)
_CURRENT_CASE_DEVELOPMENT_PLAN_COMPILERS: Mapping[
    str, _RuntimePlanCompilerSpec
] = {
    "source_feasibility_fail_closed": _RuntimePlanCompilerSpec(
        message=_DEVELOPMENT_MESSAGE,
        reason_code=(
            "source_feasibility_development_execution_only_authority_compiled"
        ),
        project_research_question=True,
        governed_steps=(("step_id", "governed_step"),),
        analysis_only=True,
    ),
    "time_varying_exposure_association": _RuntimePlanCompilerSpec(
        message=_DEVELOPMENT_MESSAGE,
        reason_code="development_execution_only_authority_compiled",
        project_research_question=True,
        governed_steps=(("step_id", "governed_step"),),
        analysis_only=True,
    ),
    "landmark_survival_suite": _RuntimePlanCompilerSpec(
        message=_DEVELOPMENT_MESSAGE,
        reason_code="development_execution_only_authority_compiled",
        project_research_question=True,
        enabled_flag="development_execution_only_allowed",
        governed_steps=(("step_id", "governed_step"),),
        analysis_only=True,
    ),
}


def _compile_current_case_plan(
    authority: CurrentCaseScientificRuntimeAuthority,
    plan: AnalysisPlan,
    *,
    development_execution_only: bool = False,
) -> tuple[AnalysisPlan, _RuntimePlanCompilerSpec] | None:
    registry = (
        _CURRENT_CASE_DEVELOPMENT_PLAN_COMPILERS
        if development_execution_only
        else _CURRENT_CASE_PLAN_COMPILERS
    )
    spec = registry.get(authority.authority_kind)
    if spec is None:
        if development_execution_only:
            return None
        raise TypeError(
            "current-case scientific runtime authority has no plan compiler: "
            f"{authority.authority_kind}"
        )
    if spec.enabled_flag is not None and not getattr(authority, spec.enabled_flag):
        return None
    if spec.project_research_question:
        bound = authority.development_execution_only_plan(
            research_question=plan.research_question
        )
    else:
        bound = authority.bind_plan(plan)
    return bound, spec


def _compilation_finding(
    authority: CurrentCaseScientificRuntimeAuthority,
    plan: AnalysisPlan,
    spec: _RuntimePlanCompilerSpec,
) -> ValidationFinding:
    detail: dict[str, Any] = {
        "reason_code": spec.reason_code,
        "execution_contract_sha256": authority.execution_contract_sha256,
    }
    if spec.analysis_only:
        detail["analysis_only"] = True
    for detail_key, method_name in spec.governed_steps:
        detail[detail_key] = getattr(authority, method_name)(plan).step_id
    for detail_key, attribute_name in spec.scalar_details:
        detail[detail_key] = getattr(authority, attribute_name)
    for detail_key, attribute_name in spec.sequence_details:
        detail[detail_key] = list(getattr(authority, attribute_name))
    return ValidationFinding(
        validator="scientific_runtime_plan_compiler",
        severity="warning",
        message=spec.message,
        detail=detail,
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

    def planning_contract_context(self) -> str:
        """Let the authority owner disclose otherwise hidden planner choices."""

        authority = self.current_case
        context_builder = getattr(authority, "planning_contract_context", None)
        return context_builder() if callable(context_builder) else ""

    def bind_plan(
        self,
        plan: AnalysisPlan,
    ) -> tuple[AnalysisPlan, list[ValidationFinding]]:
        """Compile host-owned wiring before the final plan is reviewed.

        Each authority owns its scientific coordinates and any mechanical
        product/input wiring. Binding does not authorize the Planner to change
        a sealed scientific coordinate.
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
        if authority is None:
            return plan, []
        compiled = _compile_current_case_plan(authority, plan)
        assert compiled is not None
        bound, spec = compiled
        return bound, [_compilation_finding(authority, bound, spec)]

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
        if authority is None:
            return None
        seed = AnalysisPlan(research_question=research_question, steps=[])
        compiled = _compile_current_case_plan(
            authority,
            seed,
            development_execution_only=True,
        )
        if compiled is None:
            return None
        plan, spec = compiled
        return plan, _compilation_finding(authority, plan, spec)


__all__ = ["ScientificRuntimeAuthorities"]

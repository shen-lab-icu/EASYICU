"""Bind optional caller-reviewed scientific authorities for one pipeline run."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from ..authority.current_case_scientific_runtime import (
    CurrentCaseScientificAuthorityError,
    CurrentCaseScientificRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ..contracts.endpoint import EndpointSpec
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
    *,
    sealed_from: CurrentCaseScientificRuntimeAuthority | None = None,
) -> ValidationFinding:
    detail: dict[str, Any] = {
        "reason_code": spec.reason_code,
        "execution_contract_sha256": authority.execution_contract_sha256,
    }
    roster = getattr(authority, "plan_bound_adjustment_roster", None)
    if roster is not None:
        # The roster was Planner-selected and sealed by the host at bind
        # time. Record it, and when this call performed the sealing, the
        # unsealed digest too, so run lineage joins the Web projection to the
        # executed contract.
        detail["adjustment_roster_authority"] = roster.authority
        detail["adjustment_roster"] = list(
            getattr(authority, "required_adjustment_columns", ())
        )
        if sealed_from is not None:
            detail["unsealed_execution_contract_sha256"] = (
                sealed_from.execution_contract_sha256
            )
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

    def bind_run_inputs(
        self,
        *,
        endpoint: EndpointSpec | None,
        primary_exposure: str | None,
        user_preferences: Mapping[str, Any] | None,
    ) -> tuple[EndpointSpec | None, str | None, dict[str, Any] | None]:
        """Project a sealed authority's declarations onto the run inputs.

        A survival suite declares the event/time endpoint and the exposure
        status column that the caller may otherwise only know as a binary
        event-status outcome. The caller's coordinates are kept when they
        agree; an absent coordinate takes the sealed value; a binary endpoint
        on the sealed event column is the same event without its time axis
        and is upgraded; anything else conflicts and fails closed here rather
        than at execution. A source-feasibility authority declares the run's
        formal result scope instead, so every planning contract narrows to
        the fail-closed decision the reviewed protocol allows.
        """

        authority = self.current_case
        scope = getattr(authority, "formal_result_scope", None)
        preferences = dict(user_preferences) if user_preferences is not None else None
        if isinstance(scope, str) and scope:
            declared = str((preferences or {}).get("formal_result_scope") or "")
            if declared and declared != scope:
                raise CurrentCaseScientificAuthorityError(
                    "run formal result scope conflicts with the sealed authority: "
                    f"{declared} versus {scope}"
                )
            preferences = {**(preferences or {}), "formal_result_scope": scope}
        projector = getattr(authority, "research_context_endpoint", None)
        if authority is None or not callable(projector):
            return endpoint, primary_exposure, preferences
        sealed_endpoint = projector()
        sealed_exposure = str(getattr(authority, "exposure_status_column", "") or "")
        if endpoint is not None and endpoint != sealed_endpoint:
            same_event_without_time_axis = (
                endpoint.kind == "binary"
                and endpoint.name == sealed_endpoint.event_column
                and list(endpoint.levels or []) == list(sealed_endpoint.levels or [])
            )
            if not same_event_without_time_axis:
                raise CurrentCaseScientificAuthorityError(
                    "run endpoint conflicts with the sealed survival authority: "
                    f"{endpoint.name} ({endpoint.kind}) versus "
                    f"{sealed_endpoint.event_column} (time_to_event)"
                )
        if primary_exposure and sealed_exposure and primary_exposure != sealed_exposure:
            raise CurrentCaseScientificAuthorityError(
                "run primary exposure conflicts with the sealed survival authority: "
                f"{primary_exposure} versus {sealed_exposure}"
            )
        return sealed_endpoint, (sealed_exposure or primary_exposure), preferences

    def planning_contract_context(self) -> str:
        """Let each sealed authority disclose otherwise hidden planner choices."""

        disclosures: list[str] = []
        for authority in (self.trajectory, self.current_case):
            context_builder = getattr(authority, "planning_contract_context", None)
            text = context_builder() if callable(context_builder) else ""
            if text:
                disclosures.append(text)
        return "\n\n".join(disclosures)

    def seal_for_plan(self, plan: AnalysisPlan) -> "ScientificRuntimeAuthorities":
        """Return the authorities with any plan-bound coordinate sealed.

        A contract that defers its adjustment roster to the reviewed plan
        (``plan_bound_adjustment_roster``) is re-signed here from the primary
        model's covariates. The pipeline keeps the returned value for
        validation, execution, review, and finalization so every phase holds
        the same digest the bound plan references. Exact contracts return
        ``self`` unchanged; sealing is deterministic and idempotent.
        """

        authority = self.current_case
        sealer = getattr(authority, "seal_adjustment_roster", None)
        if authority is None or not callable(sealer):
            return self
        sealed = sealer(plan)
        if sealed is authority:
            return self
        return replace(self, current_case=sealed)

    def bind_plan(
        self,
        plan: AnalysisPlan,
    ) -> tuple[AnalysisPlan, list[ValidationFinding]]:
        """Compile host-owned wiring before the final plan is reviewed.

        Each authority owns its scientific coordinates and any mechanical
        product/input wiring. Binding does not authorize the Planner to change
        a sealed scientific coordinate. A plan-bound roster is sealed first so
        the bound plan references the executed digest; callers persist that
        sealed value through :meth:`seal_for_plan`.
        """

        trajectory_authority = self.trajectory
        if trajectory_authority is not None and (
            trajectory_authority.is_development_execution_only_plan(plan)
            or trajectory_authority.names_signed_owners(plan)
        ):
            # A Planner draft that names the signed owners is compiled the
            # same way as the development projection: every scientific
            # coordinate comes from the digest-bound authority, never from
            # the draft's inputs, outputs or prose.
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
        sealed = self.seal_for_plan(plan).current_case
        assert sealed is not None
        compiled = _compile_current_case_plan(sealed, plan)
        assert compiled is not None
        bound, spec = compiled
        return bound, [
            _compilation_finding(
                sealed,
                bound,
                spec,
                sealed_from=authority if sealed is not authority else None,
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

"""Cross-field scientific ceilings shared by configuration and execution."""


from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exposure_outcome_distribution import DescriptiveIntervalLimitation

AnalysisRole = Literal["primary", "secondary", "sensitivity", "auxiliary"]


class PlanShapeRemediation(BaseModel):
    """One use of an exact action/role/input ceiling from a host finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["add_step", "replace_module"]
    action: str | None = Field(default=None, min_length=1)
    module: str | None = Field(default=None, min_length=1)
    planned_analysis_role: AnalysisRole
    inputs: tuple[str, ...]
    step_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _bounded_operation(self):
        if self.action is None and self.module is None:
            raise ValueError("structural remediation needs an action or module")
        if (self.operation == "replace_module") != (self.step_id is not None):
            raise ValueError("only replace_module names a source step_id")
        if len(set(self.inputs)) != len(self.inputs) or any(not value.strip() for value in self.inputs):
            raise ValueError("structural remediation inputs must be distinct nonempty identities")
        return self


ScientificReviewSeverity = Literal["blocker", "major", "minor"]
ScientificRemediationRoute = Literal[
    "agent_plan_revision",
    "runtime_capability",
    "study_authority_change",
    "external_evidence",
    "independent_review",
    "unclassified",
]

class PlanScientificFinding(BaseModel):
    """One stable, reviewable defect in the proposed scientific plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: ScientificReviewSeverity
    dimension: str
    message: str
    evidence_refs: list[str] = Field(default_factory=list, max_length=20)
    remediation: str
    structural_remediations: tuple[PlanShapeRemediation, ...] = Field(
        default=(), exclude_if=lambda value: not value,
    )
    remediation_route: ScientificRemediationRoute = "unclassified"
    requires_user_authorization: bool = False
    authorization_question: Optional[str] = None
    descriptive_interval_limitation: DescriptiveIntervalLimitation | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )


CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION = "easyicu.plan_scientific_review/21"


class PlanScientificReview(BaseModel):
    """Digest-bound pre-approval review of an exact context/plan/literature set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Archived reviews remain readable, but cannot substitute for a /21
    # execution review, which checks source screening and curated identity.
    schema_version: Literal[
        "easyicu.plan_scientific_review/10", "easyicu.plan_scientific_review/11",
        "easyicu.plan_scientific_review/12",
        "easyicu.plan_scientific_review/13",
        "easyicu.plan_scientific_review/14",
        "easyicu.plan_scientific_review/15",
        "easyicu.plan_scientific_review/16",
        "easyicu.plan_scientific_review/17",
        "easyicu.plan_scientific_review/18",
        "easyicu.plan_scientific_review/19",
        "easyicu.plan_scientific_review/20",
        "easyicu.plan_scientific_review/21",
    ] = (
        CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION
    )
    status: Literal["changes_required", "analysis_only", "ready_for_approval"]
    review_scope: Literal["pre_execution_plan"] = "pre_execution_plan"
    rendered_outputs_assessed: Literal[False] = False
    approval_allowed: bool
    top_journal_candidate: bool
    score: int = Field(ge=0, le=100)
    dimension_scores: dict[str, int]
    findings: list[PlanScientificFinding] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    literature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    figure_strategy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: str


class AnalysisDesignConflict(ValueError):
    code = "analysis_design_counts_only_family_conflict"


class BootstrapFamilyConflict(AnalysisDesignConflict):
    code = "analysis_design_bootstrap_family_conflict"


def validate_analysis_family_ceiling(
    *, analysis_family: str | None, variance_estimator: str
) -> None:
    """Counts-only is not a way to bypass a prediction/association contract.

    The caller canonicalizes family names. An absent family is a legacy
    coordinate, not permission to infer one; downstream step guards still run.
    A bootstrap is the interval only the causal inference suite computes, so
    only a declared causal family states it; no legacy design names one.
    """
    if (
        variance_estimator == "none_counts_only"
        and analysis_family
        and analysis_family != "descriptive_epidemiology"
    ):
        raise AnalysisDesignConflict(
            "A counts-only ceiling is compatible only with descriptive epidemiology; the agent must revise the design without changing the research question."
        )
    if variance_estimator == "bootstrap" and analysis_family != "causal_inference":
        raise BootstrapFamilyConflict(
            "A bootstrap is the interval of the causal inference suite only; the agent must revise the design without changing the research question."
        )

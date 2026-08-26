"""Case-neutral policy for article display placement and scientific purpose.

The Planner declares an article role and, when needed, a preferred placement.
This owner compiles those typed coordinates into one immutable display decision.
It never inspects task ids, variable names, database names, titles, or numeric
results, so benchmark cases and future held-out questions share the same rules.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ARTICLE_DISPLAY_POLICY_SCHEMA_VERSION = "easyicu.article_display_policy/1"
ARTICLE_DISPLAY_POLICY_VALIDATOR = "article_display_policy"

DisplayPlacement = Literal["main", "supplementary"]
DisplayPurpose = Literal["scientific_result", "diagnostic", "context", "audit"]

ARTICLE_DISPLAY_ROLE_UNSUPPORTED = "ARTICLE_DISPLAY_ROLE_UNSUPPORTED"
ARTICLE_DISPLAY_FAILED_CLOSED_RESULT_FORBIDDEN = (
    "ARTICLE_DISPLAY_FAILED_CLOSED_RESULT_FORBIDDEN"
)
ARTICLE_DISPLAY_PURPOSE_CONFLICT = "ARTICLE_DISPLAY_PURPOSE_CONFLICT"


_SCIENTIFIC_RESULT_ROLES = frozenset(
    {
        "relationship",
        "validation",
        "descriptive_result",
        "primary_estimand",
        "robustness",
        "heterogeneity",
        "model_performance",
        "calibration",
        "clinical_utility",
        "explainability",
        "temporal_absolute_risk",
        "survival_effect",
        "phenotype_structure",
        "phenotype_profile",
        "downstream_characterization",
        "causal_contrast",
        "distribution",
        "transportability",
    }
)
_DIAGNOSTIC_ROLES = frozenset(
    {
        "deviation",
        "diagnostics",
        "balance_positivity",
        "stability",
        "cluster_selection",
    }
)
_CONTEXT_ROLES = frozenset(
    {
        "overview",
        "mechanism",
        "workflow",
        "cohort_accounting",
        "baseline_context",
        "validation_design",
        "causal_protocol",
    }
)
_AUDIT_ROLES = frozenset(
    {
        "audit",
        "data_quality",
        "measurement_missingness",
        "measurement_process",
        "supplementary_provenance",
    }
)


class ArticleDisplayPolicyError(ValueError):
    """Owner-attributable display-policy failure with a stable reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.validator = ARTICLE_DISPLAY_POLICY_VALIDATOR


class ArticleDisplayPolicyRequest(BaseModel):
    """Typed inputs used to decide one panel or table's article role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    article_role: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    requested_placement: DisplayPlacement | None = None
    analysis_type: str = ""
    scientific_status: str = "analysis_only"
    central_to_question: bool = False
    interpretation_critical: bool = False
    terminal_diagnostic: bool = False


class ArticleDisplayDecision(BaseModel):
    """Immutable receipt compiled by the display-policy owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ARTICLE_DISPLAY_POLICY_SCHEMA_VERSION
    placement: DisplayPlacement
    display_purpose: DisplayPurpose
    reason_code: str


def _base_purpose(article_role: str) -> DisplayPurpose:
    if article_role in _SCIENTIFIC_RESULT_ROLES:
        return "scientific_result"
    if article_role in _DIAGNOSTIC_ROLES:
        return "diagnostic"
    if article_role in _CONTEXT_ROLES:
        return "context"
    if article_role in _AUDIT_ROLES:
        return "audit"
    raise ArticleDisplayPolicyError(
        ARTICLE_DISPLAY_ROLE_UNSUPPORTED,
        f"Unsupported typed article role: {article_role!r}",
    )


def decide_article_display(
    request: ArticleDisplayPolicyRequest,
) -> ArticleDisplayDecision:
    """Compile role, placement, and status without case or title inference."""

    role = request.article_role
    purpose = _base_purpose(role)
    failed_closed = request.scientific_status in {"failed_closed", "blocked"}

    if failed_closed:
        if request.terminal_diagnostic:
            return ArticleDisplayDecision(
                placement=(
                    "main"
                    if request.central_to_question or request.interpretation_critical
                    else "supplementary"
                ),
                display_purpose="diagnostic",
                reason_code="TERMINAL_DIAGNOSTIC_DISPLAY",
            )
        if purpose == "scientific_result":
            raise ArticleDisplayPolicyError(
                ARTICLE_DISPLAY_FAILED_CLOSED_RESULT_FORBIDDEN,
                "A failed-closed or blocked analysis cannot emit a scientific-result display.",
            )
        return ArticleDisplayDecision(
            placement="supplementary",
            display_purpose=purpose,
            reason_code="FAILED_CLOSED_SUPPORTING_DISPLAY",
        )

    if role in _AUDIT_ROLES:
        if request.central_to_question or request.analysis_type == "data_quality_audit":
            return ArticleDisplayDecision(
                placement=request.requested_placement or "main",
                display_purpose="scientific_result",
                reason_code="MEASUREMENT_PROCESS_IS_RESEARCH_RESULT",
            )
        if request.interpretation_critical:
            return ArticleDisplayDecision(
                placement="main",
                display_purpose="diagnostic",
                reason_code="INTERPRETATION_CRITICAL_DATA_DIAGNOSTIC",
            )
        return ArticleDisplayDecision(
            placement="supplementary",
            display_purpose="audit",
            reason_code="ROUTINE_DATA_AUDIT_SUPPLEMENTARY",
        )

    default_placement: DisplayPlacement = (
        "supplementary" if purpose == "diagnostic" else "main"
    )
    placement = request.requested_placement or default_placement
    return ArticleDisplayDecision(
        placement=placement,
        display_purpose=purpose,
        reason_code={
            "scientific_result": "SCIENTIFIC_RESULT_DISPLAY",
            "diagnostic": "DIAGNOSTIC_DISPLAY",
            "context": "READER_CONTEXT_DISPLAY",
            "audit": "AUDIT_DISPLAY",
        }[purpose],
    )


__all__ = [
    "ARTICLE_DISPLAY_FAILED_CLOSED_RESULT_FORBIDDEN",
    "ARTICLE_DISPLAY_POLICY_SCHEMA_VERSION",
    "ARTICLE_DISPLAY_POLICY_VALIDATOR",
    "ARTICLE_DISPLAY_PURPOSE_CONFLICT",
    "ARTICLE_DISPLAY_ROLE_UNSUPPORTED",
    "ArticleDisplayDecision",
    "ArticleDisplayPolicyError",
    "ArticleDisplayPolicyRequest",
    "DisplayPlacement",
    "DisplayPurpose",
    "decide_article_display",
]

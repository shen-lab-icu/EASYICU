"""Compatibility imports for the retired LangGraph dispatcher.

Production orchestration now lives in :mod:`easyicu.research_agent.orchestration.workflow`.
This module preserves human-review model imports for EasyICU 1.x; it does not
provide a second phase dispatcher.
"""

from __future__ import annotations

from .orchestration.workflow import (
    HUMAN_REVIEW_FINDING_REASONS,
    HUMAN_REVIEW_RESUME_SCOPE,
    HumanReviewAuthorityError,
    HumanReviewDecision,
    HumanReviewPending,
    HumanReviewRejected,
    HumanReviewRequest,
    OrchestrationRuntimeReceipt,
    PipelineWorkflow,
    WorkflowEngine,
    WorkflowCompleted,
    WorkflowPaused,
    human_review_requests_for_plan,
    orchestration_runtime_receipt,
)


def build_pipeline_graph(*args, **kwargs):
    """Refuse the retired dual-dispatcher construction API."""

    del args, kwargs
    raise RuntimeError(
        "build_pipeline_graph() was retired because EasyICU phase handoffs are "
        "not LangGraph-checkpoint serializable. Use "
        "orchestration.workflow.build_pipeline_workflow()."
    )


__all__ = [
    "HUMAN_REVIEW_FINDING_REASONS",
    "HUMAN_REVIEW_RESUME_SCOPE",
    "HumanReviewAuthorityError",
    "HumanReviewDecision",
    "HumanReviewPending",
    "HumanReviewRejected",
    "HumanReviewRequest",
    "OrchestrationRuntimeReceipt",
    "PipelineWorkflow",
    "WorkflowEngine",
    "WorkflowCompleted",
    "WorkflowPaused",
    "build_pipeline_graph",
    "human_review_requests_for_plan",
    "orchestration_runtime_receipt",
]

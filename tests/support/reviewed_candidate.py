"""A reviewed candidate plan snapshot with runtime findings (B8/B10 routing).

The snapshot the workflow builds for one approvable metadata-only plan whose
scientific review names only runtime-capability findings. Synthetic; no
benchmark item.
"""

from __future__ import annotations

from typing import Any

from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot

_RUN_ID = "run-candidate-repeated-stays"


def reviewed_candidate(study: dict[str, Any], runtime_codes: list[str]):
    digest = study_context_owner.scientific_configuration_sha256(study)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "run_id": _RUN_ID,
            "study_id": study["id"],
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": "human_plan_review_required",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "scientific_configuration_sha256": digest,
            "artifact_names": [
                "agent_plan.json",
                "scientific_plan_review.json",
                "source_run_manifest.json",
            ],
        },
        plan_review_authority={
            "run_id": _RUN_ID,
            "resumable_here": True,
            "scientific_configuration_sha256": digest,
            "budget_mode": "planner_canary",
            "research_input_state": "metadata_only",
            "plan_approval_allowed": True,
            "scientific_plan_review": {
                "status": "analysis_only",
                "approval_allowed": True,
                "findings": [],
                "facts": {
                    "remediation_buckets": {
                        "agent_plan_revision": [],
                        "runtime_capability": runtime_codes,
                        "study_authority_change": [],
                        "external_evidence": [],
                        "independent_review": [],
                    }
                },
            },
        },
    )
    assert snapshot.next_action_code == "plan_execution_upgrade_required"
    return snapshot

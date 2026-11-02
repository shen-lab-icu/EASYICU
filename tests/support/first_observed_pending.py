"""Synthetic unapproved first-observed checkpoint fixture."""
import hashlib
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.webserver import first_observed_preparation as owner
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding

PATIENT_SHA = hashlib.sha256(b"synthetic patient bytes").hexdigest()

def _pending():
    plan = AnalysisPlan(research_question="Synthetic source only", steps=[AnalysisStep(step_id="one", intent="Describe data")])
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256={"scientific_plan_review": "c" * 64})
    scope = {"purpose": "prepare_only", "analysis_authorized": False, "database": "eicu_demo",
             "source_identity_sha256": "a" * 64, "plan_sha256": authority.plan_sha256,
             "context_sha256": "b" * 64, "scientific_review_sha256": "c" * 64,
             "archive_sha256": "d" * 64, "read_scope_version": 2,
             "export_manifest_sha256": "1" * 64}
    payload = {"plan_review_authority": authority.model_dump(mode="json"), owner._SCOPE_KEY: scope}
    request = HumanReviewRequest.create(kind="scientific_stop", summary=owner._SUMMARY,
        payload=payload, authority_sha256=canonical_sha256(payload))
    checkpoint = RawPreparationCheckpoint.pending(plan_sha256=authority.plan_sha256, request=request)
    binding = RawPreparationHostBinding(project_id="project", session_id="session", study_context_id="study",
                                        run_id="run", nonce="e" * 64)
    return scope, checkpoint, binding

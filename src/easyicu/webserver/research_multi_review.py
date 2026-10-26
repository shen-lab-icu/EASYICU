"""Read the existing run-input and review authorities without minting approval."""
from pathlib import Path

from easyicu.research_agent.authority.evidence_store import EvidenceStore, sha256_of_file
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.run_input import load_verified_run_input_capsule
from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, load_checkpoint
from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.research_agent.schema import AnalysisPlan, ResearchContext
from . import study_contexts
from .research_append_source import contained, read_json
from .research_multi_contracts import MultiDatabaseError, ReviewedRun


def read_review(wrapper, run_id, study):
    run_dir = contained(Path(wrapper), "pipeline/" + run_id)
    checkpoint = HumanReviewCheckpoint.model_validate(read_json(run_dir / "human_review_checkpoint.json"))
    if checkpoint.state != "completed":
        checkpoint = load_checkpoint(run_dir / "human_review_checkpoint.json", require_pending=False)
    if checkpoint.run_id != run_id or not checkpoint.requests:
        raise MultiDatabaseError("multi_database_review_changed")
    capsule = read_json(run_dir / "run_input_capsule.json")
    if sha256_of_file(run_dir / "run_input_capsule.json") != checkpoint.run_input_capsule_sha256:
        raise MultiDatabaseError("multi_database_input_changed")
    verified = load_verified_run_input_capsule(run_dir=run_dir, scientific_identity=capsule["scientific_identity"])
    context = ResearchContext.model_validate_json(verified.context_evidence_path.read_bytes())
    plan = AnalysisPlan.model_validate(checkpoint.plan_handoff["plan"])
    evidence = EvidenceStore(run_dir)
    record = evidence.get("scientific_plan_review")
    path = verified_run_evidence_path(run_dir, record) if record else None
    if path is None:
        raise MultiDatabaseError("multi_database_review_changed")
    review = PlanScientificReview.model_validate_json(path.read_bytes())
    plan_sha = canonical_sha256(plan.model_dump(mode="json"))
    context_sha = canonical_sha256(context.model_dump(mode="json"))
    if not review.approval_allowed or review.plan_sha256 != plan_sha or review.context_sha256 != context_sha:
        raise MultiDatabaseError("multi_database_review_changed")
    for request in checkpoint.requests:
        authority = PlanReviewAuthority.model_validate(request.payload["plan_review_authority"])
        if authority.plan_sha256 != plan_sha or authority.evidence_sha256.get("scientific_plan_review") != record.sha256:
            raise MultiDatabaseError("multi_database_review_changed")
        for key, digest in authority.evidence_sha256.items():
            item = evidence.get(key)
            if item is None or item.sha256 != digest or verified_run_evidence_path(run_dir, item) is None:
                raise MultiDatabaseError("multi_database_input_changed")
    binding = ReviewedRun(run_id=run_id, study_context_id=study["id"], plan_sha256=plan_sha,
        context_sha256=context_sha, scientific_review_sha256=record.sha256,
        requests_sha256=canonical_sha256([row.model_dump(mode="json") for row in checkpoint.requests]),
        input_sha256=checkpoint.run_input_capsule_sha256,
        configuration_sha256=study_contexts.scientific_configuration_sha256(study))
    return binding, context, plan, checkpoint, run_dir

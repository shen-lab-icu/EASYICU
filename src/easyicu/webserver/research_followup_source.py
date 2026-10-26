"""Read completed approvals and result identities without reading result rows."""
from pathlib import Path

from easyicu.research_agent.authority.runtime_artifacts import load_run_artifact_authority
from easyicu.research_agent.authority.typed_binding import _resolve_typed_input_evidence
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.schema import AnalysisPlan
from .research_append_history import resolve_result_source
from .research_append_source import SEAL_NAME, _approved_source, contained, read_json
from .research_followup_contracts import CompletedResult


def load_completed_result(wrapper: Path, *, run_id: str, study_id: str) -> CompletedResult:
    """Use t5's seal where present; otherwise verify the completed review capsule.

    An invalid t5 seal never falls back to the generic path. Only aggregate
    product identities are used: free-form answers and table cells are not
    suggestion authority.
    """
    if run_id.startswith("append-") or (wrapper / SEAL_NAME).exists():
        source, predecessor = resolve_result_source(wrapper, run_id=run_id, study_id=study_id)
        steps, kinds = list(source.plan.steps), set()
        prior = predecessor
        while prior:
            from .research_append_run import load_append_run

            child = load_append_run(Path(prior["wrapper"]))
            steps.extend(AnalysisPlan.model_validate(child.record["plan"]).steps)
            kinds.add(child.authority.kind)
            prior = child.record.get("predecessor")
        return CompletedResult(run_id, predecessor["sha256"] if predecessor else source.digest,
                               source.plan, source.context, frozenset(source.seal["products"]),
                               tuple(steps), source.baseline, frozenset(kinds))
    run = contained(wrapper, "pipeline/" + run_id)
    checkpoint, plan, context, verified = _approved_source(run)
    status = read_json(run / "run_status.json")
    if checkpoint.run_id != run_id or any(status.get("gates", {}).get(key) is not True
                                         for key in ("execution_complete", "analysis_validated")):
        raise ValueError("followup_result_incomplete")
    from easyicu.research_agent.planning.scientific_review import PlanScientificReview

    review = PlanScientificReview.model_validate(read_json(wrapper / "scientific_plan_review.json"))
    if not review.approval_allowed or review.plan_sha256 != canonical_sha256(plan.model_dump(mode="json")):
        raise ValueError("followup_result_unapproved")
    ledger = load_run_artifact_authority(run)
    if ledger is None:
        raise ValueError("followup_result_unverified")
    products = {}
    for step in plan.steps:
        for key in step.expected_outputs:
            ref, failure = _resolve_typed_input_evidence(
                input_name=key, plan=plan, evidence_records=list(verified.evidence_records.values()),
                per_step_records=ledger["per_step_records"], run_dir=run,
            )
            if failure is None and ref is not None:
                # The shared resolver checks producer success, approved scope,
                # containment and registered bytes before issuing this ref.
                products[key] = verified.evidence_records[ref.evidence_id]["sha256"]
    primary = {key for step in plan.steps if step.planned_analysis_role == "primary" for key in step.expected_outputs}
    if not primary or not primary <= products.keys():
        raise ValueError("followup_primary_result_unverified")
    digest = canonical_sha256(dict(checkpoint=checkpoint.model_dump(mode="json"),
                                  context=context.model_dump(mode="json"), status=status, products=products))
    return CompletedResult(run_id, digest, plan, context, frozenset(products), tuple(plan.steps))

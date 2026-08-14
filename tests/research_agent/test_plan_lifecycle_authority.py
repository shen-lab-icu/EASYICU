from __future__ import annotations

import json

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plan_lifecycle import (
    ApprovedExecutablePlan,
    NormalizedPlan,
    PlanLifecycleAuthorityError,
    PlanTransformationReceipt,
    ProposedPlan,
    load_normalized_plan,
    persist_approved_executable_plan,
    persist_normalized_plan,
)
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _plan(*, intent: str = "Summarize the cohort.", revision: int = 1) -> AnalysisPlan:
    return AnalysisPlan(
        revision=revision,
        research_question="What is observed in this ICU cohort?",
        analysis_type="descriptive_epidemiology",
        steps=[
            AnalysisStep(
                step_id="01_summary",
                intent=intent,
                method="descriptive",
                expected_outputs=["table:cohort_summary"],
            )
        ],
    )


def _normalized() -> NormalizedPlan:
    proposed_plan = _plan()
    final_plan = _plan(intent="Summarize the verified cohort denominator.")
    receipt = PlanTransformationReceipt.create(
        transformer="host.cohort_definition",
        reason="Bind the verified analysis denominator.",
        input_plan=proposed_plan,
        output_plan=final_plan,
        scientific_semantics_changed=True,
    )
    return NormalizedPlan.create(
        proposed=ProposedPlan.create(plan=proposed_plan, source="planner_llm"),
        transformation_receipts=(receipt,),
        plan=final_plan,
    )


def test_transformation_receipt_names_exact_changed_fields_and_semantics() -> None:
    receipt = _normalized().transformation_receipts[0]

    assert receipt.changed_fields == ("/steps/0/intent",)
    assert receipt.scientific_semantics_changed is True
    assert receipt.input_sha256 != receipt.output_sha256


def test_normalized_plan_rejects_a_noncontiguous_or_tampered_chain() -> None:
    normalized = _normalized()
    payload = normalized.model_dump(mode="json")
    payload["transformation_receipts"][0]["input_sha256"] = "0" * 64

    with pytest.raises(ValueError):
        NormalizedPlan.model_validate(payload)


def test_normalized_and_approved_stages_are_immutable_evidence(
    tmp_path,
) -> None:
    evidence = EvidenceStore(tmp_path)
    normalized = _normalized()

    lineage_path = persist_normalized_plan(
        run_dir=tmp_path,
        evidence=evidence,
        normalized=normalized,
    )
    assert lineage_path.name == "plan_lifecycle_revision_1.json"
    assert load_normalized_plan(
        run_dir=tmp_path,
        evidence=evidence,
        revision=1,
    ) == normalized

    review = PlanReviewAuthority.create(plan=normalized.plan_payload)
    approved = ApprovedExecutablePlan.create(
        normalized=normalized,
        plan_review_authority=review,
        decision_set_sha256="d" * 64,
    )
    approved_path = persist_approved_executable_plan(
        run_dir=tmp_path,
        evidence=evidence,
        approved=approved,
    )
    assert approved_path.name == "approved_executable_plan_revision_1.json"
    assert evidence.get("approved_executable_plan_revision_1") is not None

    tampered = json.loads(lineage_path.read_text(encoding="utf-8"))
    tampered["plan_payload"]["research_question"] = "Changed after approval."
    lineage_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert load_normalized_plan(
        run_dir=tmp_path,
        evidence=evidence,
        revision=1,
    ) == normalized
    with pytest.raises(PlanLifecycleAuthorityError, match="cannot be overwritten"):
        persist_normalized_plan(
            run_dir=tmp_path,
            evidence=evidence,
            normalized=NormalizedPlan.create(
                proposed=ProposedPlan.create(
                    plan=_plan(intent="Different proposal."),
                    source="planner_llm",
                ),
                transformation_receipts=(),
                plan=_plan(intent="Different proposal."),
            ),
        )


def test_approval_refuses_a_review_of_a_different_plan() -> None:
    normalized = _normalized()
    different = _plan(intent="A different scientific plan.")

    with pytest.raises(
        PlanLifecycleAuthorityError,
        match="does not bind the normalized plan",
    ):
        ApprovedExecutablePlan.create(
            normalized=normalized,
            plan_review_authority=PlanReviewAuthority.create(plan=different),
            decision_set_sha256="e" * 64,
        )

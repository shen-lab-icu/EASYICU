"""Same-revision dual-plan variant binding (P1 fix regression test).

Covers: load_normalized_plan(plan_sha256=...) reads the variant,
approve binds the specified digest, unknown/malformed digests fail closed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plan_lifecycle import (
    NormalizedPlan,
    ProposedPlan,
    approve_normalized_plan_for_execution,
    load_normalized_plan,
    persist_normalized_plan,
)
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _plan(*, intent: str, revision: int = 1) -> AnalysisPlan:
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


def _normalized(intent: str) -> NormalizedPlan:
    plan = _plan(intent=intent)
    return NormalizedPlan.create(
        proposed=ProposedPlan.create(plan=plan, source="planner_llm"),
        transformation_receipts=(),
        plan=plan,
    )


def _setup_two_plans(tmp_path: Path):
    evidence = EvidenceStore(tmp_path)
    first = _normalized("First proposal narrative.")
    second = _normalized("Second proposal narrative.")
    assert first.plan_sha256 != second.plan_sha256
    persist_normalized_plan(run_dir=tmp_path, evidence=evidence, normalized=first)
    persist_normalized_plan(run_dir=tmp_path, evidence=evidence, normalized=second)
    return evidence, first, second


def test_default_reads_baseline(tmp_path: Path) -> None:
    evidence, first, _ = _setup_two_plans(tmp_path)
    baseline = load_normalized_plan(run_dir=tmp_path, evidence=evidence, revision=1)
    assert baseline.plan_sha256 == first.plan_sha256


def test_explicit_digest_reads_variant(tmp_path: Path) -> None:
    evidence, _, second = _setup_two_plans(tmp_path)
    variant = load_normalized_plan(
        run_dir=tmp_path, evidence=evidence, revision=1, plan_sha256=second.plan_sha256
    )
    assert variant.plan_sha256 == second.plan_sha256


def test_approve_binds_specified_digest(tmp_path: Path) -> None:
    evidence, _, second = _setup_two_plans(tmp_path)
    review = PlanReviewAuthority.create(plan=second.plan_payload)
    assert review.plan_sha256 == second.plan_sha256
    approved = approve_normalized_plan_for_execution(
        run_dir=tmp_path,
        evidence=evidence,
        revision=1,
        review_requests=[{"payload": {"plan_review_authority": review.model_dump(mode="json")}}],
        decision_set_sha256=canonical_sha256([]),
        plan_sha256=second.plan_sha256,
    )
    assert approved.plan_sha256 == second.plan_sha256


def test_unknown_digest_fails_closed(tmp_path: Path) -> None:
    evidence, _, _ = _setup_two_plans(tmp_path)
    # E-P2-7: pytest.raises instead of try/except:pass so an unexpected
    # success fails loudly instead of falling into an else-raise.
    with pytest.raises(Exception):
        load_normalized_plan(
            run_dir=tmp_path, evidence=evidence, revision=1, plan_sha256="0" * 64
        )


def test_malformed_digest_fails_closed(tmp_path: Path) -> None:
    evidence, _, _ = _setup_two_plans(tmp_path)
    # E-P2-7: same as above; "xyz" violates the sha256 pattern.
    with pytest.raises(Exception):
        load_normalized_plan(
            run_dir=tmp_path, evidence=evidence, revision=1, plan_sha256="xyz"
        )


def test_tmp_scratch_dir_also_works() -> None:
    # Same flow on a non-pytest tmp dir (parity with the original ad-hoc script).
    tmp = Path(tempfile.mkdtemp(prefix="task1_verify_"))
    evidence, first, second = _setup_two_plans(tmp)
    assert (
        load_normalized_plan(run_dir=tmp, evidence=evidence, revision=1).plan_sha256
        == first.plan_sha256
    )
    assert (
        load_normalized_plan(
            run_dir=tmp, evidence=evidence, revision=1, plan_sha256=second.plan_sha256
        ).plan_sha256
        == second.plan_sha256
    )

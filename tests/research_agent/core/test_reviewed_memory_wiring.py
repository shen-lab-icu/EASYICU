"""Phase-2 production wiring for permissioned reviewed memory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.coder_authority import HostCoderAuthority
from easyicu.research_agent.learning.runtime import (
    REVIEWED_MEMORY_PROMPT_LIMIT_BYTES,
    ReviewedMemoryIntegrityError,
    attach_step_reviewed_memory,
    build_reviewed_memory_bundle,
)
from easyicu.research_agent.learning.store import (
    FileSystemMemoryStore,
    MemoryObject,
    MemoryReviewAttestation,
    payload_sha256,
)
from easyicu.research_agent.orchestration.profiles import get_submission_profile

PROFILE_REF = "npj_dm_framework_v2_memory_dev/20260722"
NAMESPACES = (
    "reviewed_knowledge/framework_v2",
    "promoted_lessons/1.0.0",
)


def _reviewed(*, profile_ref: str = PROFILE_REF) -> MemoryObject:
    payload = {
        "title": "Explicit time-zero checks",
        "advice": "Verify the planner-owned time zero before fitting a model.",
    }
    digest = payload_sha256(payload)
    return MemoryObject.create(
        namespace=NAMESPACES[0],
        key="time-zero-check",
        version="1.0.0",
        payload=payload,
        source="held-out-regression-suite",
        producer="human-review-workflow",
        review_status="reviewed",
        created_at="2026-07-22T08:00:00-04:00",
        applicable_scope=("global",),
        invalidation=("time_zero_contract_change",),
        profile_ref=profile_ref,
        attestation=MemoryReviewAttestation(
            reviewer="clinical-and-methods-review",
            reviewed_at="2026-07-22T08:00:00-04:00",
            review_scope="held-out design checks",
            payload_sha256=digest,
            evidence_refs=("review://time-zero/1",),
        ),
    )


def _bundle(store: FileSystemMemoryStore, *, profile_ref: str = PROFILE_REF):
    return build_reviewed_memory_bundle(
        store=store,
        profile_ref=profile_ref,
        allowed_namespaces=NAMESPACES,
        step_id="02_primary_model",
        analysis_family="association",
        step_role="primary",
        question="Estimate an adjusted association.",
        method="adjusted_association_models",
    )


def test_reviewed_memory_selects_current_profile_only_and_zero_llm(
    tmp_path: Path,
) -> None:
    store = FileSystemMemoryStore(tmp_path)
    current = _reviewed()
    wrong_profile = _reviewed(profile_ref="another_profile/20260722").model_copy(
        update={"key": "wrong-profile"}
    )
    store.put(current)
    store.put(wrong_profile)

    bundle = _bundle(store)

    assert bundle.provider_calls == 0
    assert bundle.prompt_bytes <= REVIEWED_MEMORY_PROMPT_LIMIT_BYTES
    assert [item.key for item in bundle.selected] == [current.key]
    assert bundle.selected[0].profile_ref == PROFILE_REF
    assert "wrong-profile" not in bundle.prompt_projection


def test_reviewed_memory_zero_match_is_legal(tmp_path: Path) -> None:
    bundle = _bundle(FileSystemMemoryStore(tmp_path))
    assert bundle.selected == ()
    assert bundle.prompt_projection == ""
    assert bundle.prompt_bytes == 0


def test_reviewed_memory_receipt_tamper_and_wrong_profile_fail_closed(
    tmp_path: Path,
) -> None:
    store = FileSystemMemoryStore(tmp_path / "store")
    store.put(_reviewed())
    bundle = _bundle(store)
    _authority, path = attach_step_reviewed_memory(
        authority=HostCoderAuthority(), run_dir=tmp_path / "run", bundle=bundle
    )
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ReviewedMemoryIntegrityError, match="changed"):
        attach_step_reviewed_memory(
            authority=HostCoderAuthority(), run_dir=tmp_path / "run", bundle=bundle
        )

    path.unlink()
    attach_step_reviewed_memory(
        authority=HostCoderAuthority(), run_dir=tmp_path / "run", bundle=bundle
    )
    with pytest.raises(ReviewedMemoryIntegrityError, match="changed"):
        attach_step_reviewed_memory(
            authority=HostCoderAuthority(),
            run_dir=tmp_path / "run",
            bundle=bundle.model_copy(update={"profile_ref": "wrong/20260722"}),
        )


def test_reviewed_memory_requires_additive_profile_and_exact_namespaces(
    ra, tmp_path: Path
) -> None:
    profile = get_submission_profile(PROFILE_REF)
    options = profile.as_pipeline_options()
    assert options["enable_reviewed_memory"] is True
    assert options["reviewed_memory_namespaces"] == NAMESPACES
    assert (
        "enable_reviewed_memory"
        not in get_submission_profile("npj_dm/20260719").as_pipeline_options()
    )
    with pytest.raises(ValueError, match="requires an additive submission profile"):
        ra.ResearchAgentPipeline(
            workdir=tmp_path,
            enable_memory=False,
            enable_reviewed_memory=True,
            reviewed_memory_namespaces=NAMESPACES,
        )


def test_pipeline_production_callsite_binds_only_reviewed_current_profile_memory(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from easyicu.research_agent.authority.step_capsule import (
        StepAuthorityCapsuleRef,
        load_verified_step_authority_capsule,
        read_verified_content,
    )
    from easyicu.research_agent.execution import phase as execute_phase
    from tests.research_agent import (
        test_trajectory_stability_pipeline_success as fixture,
    )

    store = FileSystemMemoryStore(tmp_path / ".memory_v2")
    store.put(_reviewed())
    fixture._disable_unrelated_audits(monkeypatch)
    coder_authorities: list[str] = []
    original_coder_run = execute_phase.CoderAgent.run

    def observed_coder_run(self, *args, **kwargs):
        coder_authorities.append(kwargs["host_authority"].render())
        return original_coder_run(self, *args, **kwargs)

    monkeypatch.setattr(execute_phase.CoderAgent, "run", observed_coder_run)
    runners: dict[float, object] = {}

    def runner_factory(*, workdir, timeout_seconds, **_kwargs):
        timeout = float(timeout_seconds)
        if timeout not in runners:
            runners[timeout] = fixture._HybridTrajectoryRunner(workdir=Path(workdir))
        return runners[timeout]

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=fixture._PlanAndCoderLLM(),
        timeout_seconds=17.0,
        standard_executor_timeout_seconds=1_234.0,
        runner_factory=runner_factory,
        enable_memory=False,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_replanning=False,
        enable_deterministic_code_fallback=True,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=2,
        submission_profile_name="npj_dm_framework_v2_memory_dev",
        submission_profile_version="20260722",
        enable_coder_resources=True,
        enable_reviewed_memory=True,
        reviewed_memory_namespaces=NAMESPACES,
    )
    cohort = pd.DataFrame(
        {
            "stay_id": list(range(1, 25)),
            "marker_h0_6": np.linspace(-1.0, 1.0, 24),
            "marker_h6_12": np.linspace(-0.5, 1.5, 24),
            "death": [0, 1] * 12,
        }
    )
    result = pipeline.run(
        question="Assess fixed-window trajectory phenotypes.",
        cohort=cohort,
        cohort_name="reviewed_memory_production_wiring",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_representation",
        stop_after_analysis=True,
    )

    run_dir = Path(result.workdir)
    receipt_path = run_dir / "memory_selections" / "coder" / "01_representation.json"
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["profile_ref"] == PROFILE_REF
    assert receipt["provider_calls"] == 0
    assert [item["key"] for item in receipt["selected"]] == ["time-zero-check"]
    manifest = json.loads(
        (run_dir / "manifest_partial.json").read_text(encoding="utf-8")
    )
    record = next(
        row
        for row in manifest["per_step_records"]
        if row.get("step_id") == "01_representation"
    )
    assert record["reviewed_memory_provider_calls"] == 0
    assert coder_authorities
    assert "easyicu.reviewed_memory_attachment/1" in coder_authorities[0]
    capsule_ref = StepAuthorityCapsuleRef.model_validate(
        record["step_authority_capsule_ref"]
    )
    verified = load_verified_step_authority_capsule(run_dir, ref=capsule_ref)
    scoped_payload = json.loads(
        read_verified_content(run_dir, verified.capsule.scoped_coder_context)
    )
    attachments = scoped_payload["host_coder_authority"]["attachments"]
    assert any("easyicu.reviewed_memory_attachment/1" in item for item in attachments)

"""Image-only development recovery must retain exact scientific authority."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.human_review_checkpoint import (
    HumanReviewCheckpoint,
    HumanReviewCheckpointError,
    write_checkpoint,
)
from easyicu.research_agent.orchestration.runtime_revision import (
    prepare_execution_runtime_revision,
)
from easyicu.research_agent.orchestration.services import PipelineServices
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.research_agent.pipeline import _pipeline_run___human_review_invoker
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


@pytest.fixture
def approved(tmp_path):
    config = PipelineConfig(
        workdir=tmp_path,
        runner_kind="docker",
        runner_image="engine:old",
        require_human_plan_review=True,
        max_provider_attempts_per_run=100,
        max_provider_attempts_per_batch=100,
        max_total_tokens_per_run=1000000,
        max_total_tokens_per_batch=1000000,
        max_estimated_cost_usd_per_batch=100.0,
        max_wall_clock_seconds_per_task=600.0,
        provider_input_cost_usd_per_million_tokens=20.0,
        provider_output_cost_usd_per_million_tokens=120.0,
    )
    run_dir = tmp_path / "run-recovery"
    run_dir.mkdir()
    evidence = EvidenceStore(run_dir)
    capsule_path = run_dir / "run_input_capsule.json"
    capsule_path.write_text('{"test_input_identity": "sealed"}')
    capsule = evidence.register_file(
        kind="log",
        description="Input identity fixture",
        source_path=capsule_path,
        evidence_id="run_input_capsule",
    )
    plan = AnalysisPlan(
        research_question="How does the measured exposure relate to the outcome?",
        steps=[
            AnalysisStep(step_id="model", intent="Estimate the reviewed association.")
        ],
    )
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review plan",
        authority_sha256="b" * 64,
        payload={},
    )
    cp = HumanReviewCheckpoint.create(
        run_id=run_dir.name,
        pipeline_config_sha256=config.canonical_digest(),
        environment_identity={},
        llm_signature_sha256="c" * 64,
        run_input_capsule_sha256=capsule.sha256,
        capability_activation_sha256="d" * 64,
        runtime_capabilities=("numpy",),
        runtime_bundle=None,
        requests=(request,),
        plan_handoff={"plan": plan.model_dump(mode="json")},
        execution_coordinates={},
    )
    decisions = [
        {
            "review_id": request.review_id,
            "authority_sha256": request.authority_sha256,
            "decision": "approved",
        }
    ]
    cp = (
        cp.approved(
            decisions=decisions,
            decision_records=decisions,
            decision_sha256=canonical_sha256(decisions),
        )
        .execution_started()
        .transitioned("completed")
    )
    write_checkpoint(run_dir / "human_review_checkpoint.json", cp)
    config_json = config.recovery_payload()
    updated, revision = prepare_execution_runtime_revision(
        approved_config=config, runner_image="engine:fixed", run_dir=run_dir
    )
    assert config.recovery_payload() == config_json
    return SimpleNamespace(
        config=config,
        updated=updated,
        revision=revision,
        run_dir=run_dir,
        checkpoint=cp,
        plan=plan,
        evidence=evidence,
        capsule=capsule,
    )


def _bundle(image="engine:fixed"):
    return {
        "schema": "easyicu.docker_runtime_preflight/3",
        "requirements": "numpy==1",
        "provenance": {
            "runtime": "docker",
            "image_reference": image,
            "image_id": "sha256:" + "e" * 64,
            "network": "none",
            "execution_kernel_identity_sha256": "f" * 64,
        },
    }


def _authorize(a, **overrides):
    kwargs = dict(
        config=a.updated,
        run_dir=a.run_dir,
        plan_payload=a.plan.model_dump(mode="json"),
        run_input_capsule_sha256=a.capsule.sha256,
        runtime_bundle=_bundle(),
        runtime_capabilities=("numpy", "pandas"),
        evidence=a.evidence,
    )
    kwargs.update(overrides)
    a.revision.authorize_and_record(**kwargs)


def test_real_pipeline_review_bridge_records_revision_without_new_approval(approved):
    a = approved
    checkpoint_path = a.run_dir / "human_review_checkpoint.json"
    original = checkpoint_path.read_bytes()
    pipeline = SimpleNamespace(
        _config=a.updated,
        _services=PipelineServices(execution_runtime_revision=a.revision),
        _validated_runtime_bundle=_bundle(),
        _validated_runtime_capabilities=("numpy", "pandas"),
    )
    result = SimpleNamespace(evidence=a.evidence, plan=a.plan, resume_state={})
    for _ in range(2):
        assert (
            _pipeline_run___human_review_invoker(
                result, reviewed_plan=[], self=pipeline
            )
            == ()
        )
    assert checkpoint_path.read_bytes() == original
    files = list(a.run_dir.glob("execution_runtime_revision_*.json"))
    assert len(files) == 1
    receipt = json.loads(files[0].read_text())
    assert receipt["approved_config_sha256"] == a.config.canonical_digest()
    assert receipt["target_config_sha256"] == a.updated.canonical_digest()
    assert receipt["changed_fields"] == ["runner_image"]
    assert receipt["paper_authority"] is False
    assert (
        receipt["validated_runtime_bundle"]["provenance"]["image_id"]
        == "sha256:" + "e" * 64
    )
    assert a.evidence.get(files[0].stem) is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("runner_network", "bridge"),
        ("runner_kind", "subprocess"),
        ("max_total_tokens_per_run", 999999),
        ("require_human_plan_review", False),
        ("manuscript_language", "zh"),
        ("enable_pdf_render", True),
    ],
)
def test_even_self_consistent_target_hash_cannot_expand_config_scope(
    approved, field, value
):
    a = approved
    config = PipelineConfig(**{**a.updated.recovery_payload(), field: value})
    revision = dataclasses.replace(
        a.revision, target_config_sha256=config.canonical_digest()
    )
    with pytest.raises(HumanReviewCheckpointError, match="change only"):
        revision.validate(config=config, run_dir=a.run_dir)
    assert not list(a.run_dir.glob("execution_runtime_revision_*.json"))


@pytest.mark.parametrize(
    "change", ["plan", "capsule", "runtime", "image", "packages", "kernel"]
)
def test_changed_science_or_unvalidated_runtime_never_gets_a_receipt(approved, change):
    a = approved
    kwargs = {}
    if change == "plan":
        kwargs["plan_payload"] = {
            **a.plan.model_dump(mode="json"),
            "research_question": "Another question",
        }
    if change == "capsule":
        kwargs["run_input_capsule_sha256"] = "a" * 64
    if change == "runtime":
        kwargs["runtime_bundle"] = None
    if change == "image":
        kwargs["runtime_bundle"] = _bundle("engine:other")
    if change == "packages":
        kwargs["runtime_capabilities"] = ()
    if change == "kernel":
        kwargs["runtime_bundle"] = _bundle()
        kwargs["runtime_bundle"]["provenance"]["execution_kernel_identity_sha256"] = ""
    with pytest.raises(HumanReviewCheckpointError):
        _authorize(a, **kwargs)
    assert not list(a.run_dir.glob("execution_runtime_revision_*.json"))


def test_checkpoint_drift_and_cross_run_reuse_are_rejected(approved):
    a = approved
    with pytest.raises(HumanReviewCheckpointError, match="another run"):
        a.revision.validate(config=a.updated, run_dir=a.run_dir.parent / "another")
    write_checkpoint(
        a.run_dir / "human_review_checkpoint.json", a.checkpoint.transitioned("failed")
    )
    with pytest.raises(HumanReviewCheckpointError, match="checkpoint changed"):
        _authorize(a)


def test_unchanged_image_uses_existing_exact_retry(approved):
    a = approved
    config, revision = prepare_execution_runtime_revision(
        approved_config=a.config, runner_image=a.config.runner_image, run_dir=a.run_dir
    )
    assert config is a.config and revision is None


def test_paper_profile_cannot_use_development_runtime_revision(approved):
    from easyicu.research_agent.orchestration.profiles import get_submission_profile

    a = approved
    formal = PipelineConfig(
        **{**a.config.recovery_payload(), **get_submission_profile().pipeline_options()}
    )
    with pytest.raises(HumanReviewCheckpointError, match="unpinned development"):
        prepare_execution_runtime_revision(
            approved_config=formal,
            runner_image="engine:fixed",
            run_dir=a.run_dir,
        )


def test_receipt_tampering_is_not_overwritten(approved):
    a = approved
    _authorize(a)
    receipt = next(a.run_dir.glob("execution_runtime_revision_*.json"))
    receipt.write_text("changed")
    with pytest.raises(HumanReviewCheckpointError, match="receipt changed"):
        _authorize(a)
    assert receipt.read_text() == "changed"

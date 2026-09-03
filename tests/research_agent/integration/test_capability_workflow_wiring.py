"""Production contracts for the no-install capability workflow."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.orchestration.profiles import (
    SUBMISSION_PROFILE_REGISTRY,
    SubmissionProfile,
    get_submission_profile,
)
from easyicu.research_agent.concept_dict_audit import (
    compute_concept_dict_fingerprint,
)
from easyicu.research_agent.resources import (
    CapabilityApproval,
    build_capability_activation,
    build_capability_request,
)

PENDING_PROFILE_REF = "npj_dm_framework_v2_capability_dev/20260722"
IMAGE_DIGEST = "sha256:" + "a" * 64
BASE_IMPORTS = ("numpy", "pandas", "scipy", "sklearn", "statsmodels")


class _CountingLLM:
    name = "capability-zero-provider-test"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("provider must not run while capability review is pending")


def _request(*, runtime_imports=BASE_IMPORTS):
    return build_capability_request(
        method_name="missingness-aware trajectory representation",
        package_name="trajectory-extra",
        import_name="trajectory_extra",
        version_spec="==1.0.0",
        purpose="Execute the planner-declared trajectory representation",
        analysis_families=("trajectory_clustering", "descriptive_epidemiology"),
        license_spdx="BSD-3-Clause",
        upstream_source="https://example.org/trajectory-extra",
        validation_test_refs=("tests/methods/test_trajectory_extra.py",),
        requested_by="maintainer:framework-v2-test",
        requested_at="2026-07-22T10:00:00-04:00",
        runtime_import_names=runtime_imports,
    )


def _approval(request):
    return CapabilityApproval(
        request_id=request.request_id,
        request_sha256=request.sha256,
        decision="approved",
        reviewer="maintainer",
        reviewed_at="2026-07-22T10:30:00-04:00",
        installed_version="1.0.0",
        image_reference="easyicu/research-agent@" + IMAGE_DIGEST,
        image_digest=IMAGE_DIGEST,
        validation_receipt_sha256="b" * 64,
    )


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": list(range(1, 25)),
            "marker_h0_6": np.linspace(-1.0, 1.0, 24),
            "marker_h6_12": np.linspace(-0.5, 1.5, 24),
            "death": [0, 1] * 12,
        }
    )


def _target_profile(*, name: str = "capability_activated") -> SubmissionProfile:
    fingerprint = compute_concept_dict_fingerprint()
    return SubmissionProfile(
        name=name,
        version="1",
        locked_at="2026-07-22T11:00:00-04:00",
        evidence_enforcement_mode="strict",
        writer_digest_widened=True,
        enable_reproducibility_envelope=True,
        requires_arm="aware",
        expected_concept_dict_sha=fingerprint.concept_dict_sha,
        expected_sofa2_dict_sha=fingerprint.sofa2_dict_sha,
        enable_memory=False,
        enable_experience_bank=False,
        enable_know_how=False,
        enable_coder_resources=True,
        enable_reviewed_memory=False,
        enable_capability_workflow=True,
        expected_runner_image_digest=IMAGE_DIGEST,
    )


def test_missing_capability_pauses_real_pipeline_before_provider(
    ra, tmp_path: Path
) -> None:
    from tests.research_agent.integration import (
        test_trajectory_stability_pipeline_success as fixture,
    )

    llm = _CountingLLM()
    runners: dict[float, object] = {}

    def runner_factory(*, workdir, timeout_seconds, **_kwargs):
        timeout = float(timeout_seconds)
        if timeout not in runners:
            runners[timeout] = fixture._HybridTrajectoryRunner(workdir=Path(workdir))
        return runners[timeout]

    fingerprint = compute_concept_dict_fingerprint()
    profile = replace(
        get_submission_profile(PENDING_PROFILE_REF),
        expected_concept_dict_sha=fingerprint.concept_dict_sha,
        expected_sofa2_dict_sha=fingerprint.sofa2_dict_sha,
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        runner_factory=runner_factory,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        **profile.pipeline_options(),
        capability_request=_request().model_dump(mode="json"),
    )
    result = pipeline.run(
        question="Assess fixed-window trajectory phenotypes.",
        cohort=_cohort(),
        cohort_name="capability_review_pending",
        database="synthetic",
        target_outcome="death",
        stop_after_analysis=True,
    )

    run_dir = Path(result.workdir)
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert llm.calls == 0
    assert manifest["notes"] == "aborted: capability_review_required"
    assert (run_dir / "capability" / "request.json").is_file()
    finding = next(
        item
        for item in manifest["findings"]
        if item["validator"] == "capability_workflow"
    )
    assert finding["detail"]["provider_calls"] == 0
    assert finding["detail"]["runtime_install_allowed"] is False
    capsule = json.loads((run_dir / "run_input_capsule.json").read_text())
    coordinate = capsule["scientific_identity"]["capability_workflow"]
    assert coordinate["request_sha256"] == _request().sha256
    assert coordinate["status"] == "review_required"


def test_approved_capability_requires_new_run_not_resume(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target_profile(name="capability_resume_forbidden")
    monkeypatch.setitem(SUBMISSION_PROFILE_REGISTRY, target.ref, target)
    request = _request()
    approval = _approval(request)
    activation = build_capability_activation(
        request=request,
        approval=approval,
        source_profile_ref=PENDING_PROFILE_REF,
        target_profile_ref=target.ref,
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=_CountingLLM(),
        **target.pipeline_options(),
        capability_request=request.model_dump(mode="json"),
        capability_approval=approval.model_dump(mode="json"),
        capability_activation=activation.model_dump(mode="json"),
    )
    with pytest.raises(ValueError, match="new run"):
        pipeline.run(
            question="test",
            cohort=_cohort(),
            database="synthetic",
            resume_run_id="old-run",
        )


def test_approved_capability_enters_coder_only_on_pinned_new_image(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from easyicu.research_agent.execution import phase as execute_phase
    from tests.research_agent.integration import (
        test_trajectory_stability_pipeline_success as fixture,
    )

    target = _target_profile()
    monkeypatch.setitem(SUBMISSION_PROFILE_REGISTRY, target.ref, target)
    request = _request()
    approval = _approval(request)
    activation = build_capability_activation(
        request=request,
        approval=approval,
        source_profile_ref=PENDING_PROFILE_REF,
        target_profile_ref=target.ref,
    )

    class ImageBoundRunner(fixture._HybridTrajectoryRunner):
        @staticmethod
        def validate_runtime_capabilities():
            return (*BASE_IMPORTS, "trajectory_extra")

        @staticmethod
        def export_validated_runtime_bundle():
            return {"image_id": IMAGE_DIGEST, "repo_digests": []}

        def adopt_validated_runtime_bundle(self, bundle):
            assert bundle["image_id"] == IMAGE_DIGEST

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
            runners[timeout] = ImageBoundRunner(workdir=Path(workdir))
        return runners[timeout]

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=fixture._PlanAndCoderLLM(),
        runner_factory=runner_factory,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_replanning=False,
        enable_deterministic_code_fallback=True,
        enable_deterministic_runner_repair=False,
        max_code_repair_attempts=2,
        **target.pipeline_options(),
        capability_request=request.model_dump(mode="json"),
        capability_approval=approval.model_dump(mode="json"),
        capability_activation=activation.model_dump(mode="json"),
    )
    result = pipeline.run(
        question="Assess fixed-window trajectory phenotypes.",
        cohort=_cohort(),
        cohort_name="capability_activated_new_run",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_representation",
        stop_after_analysis=True,
    )

    run_dir = Path(result.workdir)
    assert (run_dir / "capability" / "approval.json").is_file()
    assert (run_dir / "capability" / "activation.json").is_file()
    assert coder_authorities
    assert "trajectory_extra" in coder_authorities[0]
    resource_receipt = json.loads(
        (
            run_dir / "resource_selections" / "coder" / "01_representation.json"
        ).read_text(encoding="utf-8")
    )
    software = resource_receipt["selections"][1]["selected"]
    assert any(item["resource_id"] == "software:trajectory-extra" for item in software)


def test_wrong_runtime_image_fails_before_provider(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target_profile(name="capability_wrong_image")
    monkeypatch.setitem(SUBMISSION_PROFILE_REGISTRY, target.ref, target)
    request = _request()
    approval = _approval(request)
    activation = build_capability_activation(
        request=request,
        approval=approval,
        source_profile_ref=PENDING_PROFILE_REF,
        target_profile_ref=target.ref,
    )
    from tests.research_agent.integration import (
        test_trajectory_stability_pipeline_success as fixture,
    )

    class WrongImageRunner(fixture._HybridTrajectoryRunner):
        @staticmethod
        def validate_runtime_capabilities():
            return (*BASE_IMPORTS, "trajectory_extra")

        @staticmethod
        def export_validated_runtime_bundle():
            return {"image_id": "sha256:" + "c" * 64, "repo_digests": []}

    llm = _CountingLLM()
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        runner_factory=lambda **kwargs: WrongImageRunner(
            workdir=Path(kwargs["workdir"])
        ),
        enable_literature=False,
        **target.pipeline_options(),
        capability_request=request.model_dump(mode="json"),
        capability_approval=approval.model_dump(mode="json"),
        capability_activation=activation.model_dump(mode="json"),
    )
    with pytest.raises(ValueError, match="runtime image"):
        pipeline.run(
            question="Assess fixed-window trajectory phenotypes.",
            cohort=_cohort(),
            database="synthetic",
            target_outcome="death",
        )
    assert llm.calls == 0

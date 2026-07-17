"""Concept-dictionary drift detection for reproducible research-agent runs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from easyicu.research_agent.concept_dict_audit import (
    ConceptDictDriftError,
    assert_dict_matches,
    compute_concept_dict_fingerprint,
    verify_recorded_dict_match,
    verify_replay_dict_match,
)
from easyicu.research_agent.pipeline_profiles import NPJ_DM_2026_05
from easyicu.research_agent.pipeline_profiles import (
    DEFAULT_SUBMISSION_PROFILE_REF,
    get_submission_profile,
)


def test_fingerprint_is_stable_across_calls() -> None:
    first = compute_concept_dict_fingerprint()
    second = compute_concept_dict_fingerprint()
    assert first.concept_dict_sha == second.concept_dict_sha
    assert first.sofa2_dict_sha == second.sofa2_dict_sha
    assert first.concept_dict_path == "easyicu/data/concept-dict.json"
    assert first.sofa2_dict_path == "easyicu/data/sofa2-dict.json"


def test_assert_dict_matches_passes_on_identical_sha() -> None:
    fingerprint = compute_concept_dict_fingerprint()
    assert (
        assert_dict_matches(
            fingerprint,
            expected_concept_dict_sha=fingerprint.concept_dict_sha,
            expected_sofa2_dict_sha=fingerprint.sofa2_dict_sha,
            mode="strict",
        )
        == []
    )


def test_default_submission_profile_matches_packaged_dictionaries() -> None:
    fingerprint = compute_concept_dict_fingerprint()
    profile = get_submission_profile(DEFAULT_SUBMISSION_PROFILE_REF)
    assert_dict_matches(
        fingerprint,
        expected_concept_dict_sha=profile.expected_concept_dict_sha,
        expected_sofa2_dict_sha=profile.expected_sofa2_dict_sha,
        mode="strict",
    )


def test_assert_dict_matches_strict_raises_on_mismatch() -> None:
    fingerprint = compute_concept_dict_fingerprint()
    with pytest.raises(ConceptDictDriftError) as excinfo:
        assert_dict_matches(
            fingerprint,
            expected_concept_dict_sha="0" * 64,
            mode="strict",
        )
    message = str(excinfo.value)
    assert "expected=" + ("0" * 64) in message
    assert f"actual={fingerprint.concept_dict_sha}" in message


def test_assert_dict_matches_soft_returns_warnings() -> None:
    warnings = assert_dict_matches(
        compute_concept_dict_fingerprint(),
        expected_concept_dict_sha="0" * 64,
        mode="soft",
    )
    assert warnings
    assert "concept-dict.json SHA mismatch" in warnings[0]


def test_profile_locked_sha_enforced_at_plan_start(
    ra,
    synthetic_cohort: Path,
    tmp_path: Path,
) -> None:
    profile = replace(NPJ_DM_2026_05, expected_concept_dict_sha="0" * 64)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ra.MockLLMClient(),
        **profile.pipeline_options(),
    )
    with pytest.raises(ConceptDictDriftError, match="concept-dict.json SHA mismatch"):
        pipeline.run(
            question="Is age associated with mortality?",
            cohort=synthetic_cohort,
            cohort_name="synthetic",
            database="synthetic",
            target_outcome="death",
        )


def test_manifest_records_full_fingerprint(
    ra, synthetic_cohort: Path, tmp_path: Path
) -> None:
    result = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient()).run(
        question="Is age associated with mortality?",
        cohort=synthetic_cohort,
        cohort_name="synthetic",
        database="synthetic",
        target_outcome="death",
    )
    manifest = json.loads((Path(result.workdir) / "manifest.json").read_text())
    fingerprint = manifest["concept_dict_fingerprint"]
    assert fingerprint["concept_dict_sha"] == manifest["concept_dict_sha"]
    assert fingerprint["sofa2_dict_sha"] == manifest["sofa2_dict_sha"]
    assert len(fingerprint["concept_dict_sha"]) == 64
    assert len(fingerprint["sofa2_dict_sha"]) == 64


def test_replay_rejects_modified_dict(tmp_path: Path) -> None:
    fingerprint = compute_concept_dict_fingerprint()
    _write_manifest(
        tmp_path,
        concept_sha="0" * 64,
        sofa2_sha=fingerprint.sofa2_dict_sha,
    )
    with pytest.raises(ConceptDictDriftError, match="concept-dict.json SHA mismatch"):
        verify_replay_dict_match(tmp_path)


def test_replay_passes_when_unchanged(tmp_path: Path) -> None:
    fingerprint = compute_concept_dict_fingerprint()
    _write_manifest(
        tmp_path,
        concept_sha=fingerprint.concept_dict_sha,
        sofa2_sha=fingerprint.sofa2_dict_sha,
    )
    assert verify_replay_dict_match(tmp_path) == []


def test_checkpoint_manifest_dict_authority_rejects_mismatch() -> None:
    fingerprint = compute_concept_dict_fingerprint()
    manifest = {
        "concept_dict_fingerprint": {
            "concept_dict_sha": "0" * 64,
            "sofa2_dict_sha": fingerprint.sofa2_dict_sha,
        }
    }

    with pytest.raises(ConceptDictDriftError, match="concept-dict.json SHA mismatch"):
        verify_recorded_dict_match(manifest, mode="strict")


def test_checkpoint_manifest_without_legacy_dict_authority_remains_compatible() -> None:
    assert verify_recorded_dict_match({}, mode="strict") == []
    assert verify_recorded_dict_match({}, mode="soft") == [
        "manifest has no concept dictionary fingerprint to verify"
    ]


def test_resume_rejects_dict_drift_before_writing_receipt(
    ra,
    synthetic_cohort: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from easyicu.research_agent import concept_dict_audit

    kwargs = {
        "question": "Is age associated with mortality?",
        "cohort": synthetic_cohort,
        "cohort_name": "synthetic",
        "database": "synthetic",
        "target_outcome": "death",
    }
    first = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ra.MockLLMClient(),
    ).run(**kwargs)
    run_dir = Path(first.workdir)
    root_fingerprint_before = (run_dir / "concept_dict_fingerprint.json").read_bytes()
    receipts_before = sorted((run_dir / "evidence").glob("*resume_environment*"))
    fingerprint = concept_dict_audit.compute_concept_dict_fingerprint()
    monkeypatch.setattr(
        concept_dict_audit,
        "compute_concept_dict_fingerprint",
        lambda: replace(fingerprint, concept_dict_sha="0" * 64),
    )

    with pytest.raises(ConceptDictDriftError, match="concept-dict.json SHA mismatch"):
        ra.ResearchAgentPipeline(
            workdir=tmp_path,
            llm=ra.MockLLMClient(),
        ).run(**kwargs, resume_run_id=first.run_id)

    assert (run_dir / "concept_dict_fingerprint.json").read_bytes() == (
        root_fingerprint_before
    )
    assert sorted((run_dir / "evidence").glob("*resume_environment*")) == (
        receipts_before
    )


def test_sofa2_and_concept_dict_tracked_independently(tmp_path: Path) -> None:
    fingerprint = compute_concept_dict_fingerprint()
    _write_manifest(
        tmp_path,
        concept_sha=fingerprint.concept_dict_sha,
        sofa2_sha="1" * 64,
    )
    with pytest.raises(ConceptDictDriftError) as excinfo:
        verify_replay_dict_match(tmp_path)
    message = str(excinfo.value)
    assert "sofa2-dict.json SHA mismatch" in message
    assert "concept-dict.json SHA mismatch" not in message


def _write_manifest(tmp_path: Path, *, concept_sha: str, sofa2_sha: str) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "concept_dict_fingerprint": {
                    "concept_dict_path": "easyicu/data/concept-dict.json",
                    "concept_dict_sha": concept_sha,
                    "sofa2_dict_path": "easyicu/data/sofa2-dict.json",
                    "sofa2_dict_sha": sofa2_sha,
                    "computed_at": "2026-05-27T00:00:00Z",
                }
            }
        ),
        encoding="utf-8",
    )

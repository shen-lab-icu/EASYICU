"""Submission-profile guards for the research-agent benchmark runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_DIR))
try:
    from run_research_agent_bench import (  # type: ignore[import-not-found]
        _benchmark_pipeline_options,
        _enforce_mock_aware_provider,
        _enforce_submission_profile_arms,
        _enforce_submission_profile_runner,
    )
finally:
    sys.path.pop(0)

from easyicu.research_agent.pipeline_profiles import (
    DEFAULT_SUBMISSION_PROFILE_REF,
    NPJ_DM_2026_05,
    NPJ_DM_2026_06,
    get_submission_profile,
)

CANONICAL_PROFILE = NPJ_DM_2026_06


def test_submission_profile_forces_canonical_pipeline_options() -> None:
    options = _benchmark_pipeline_options(
        max_total_steps=None,
        disable_replanning=False,
        max_code_repair_attempts=None,
        enable_repro_envelope=False,
        submission_profile=CANONICAL_PROFILE,
    )
    assert options["evidence_enforcement_mode"] == "strict"
    assert options["enable_reproducibility_envelope"] is True
    assert options["writer_digest_widened"] is True
    assert options["submission_profile_name"] == "npj_dm"
    assert options["submission_profile_version"] == "20260611"
    assert options["submission_profile_locked_at"] == "2026-06-11T00:00:00Z"


def test_submission_profile_requires_aware_only_arm() -> None:
    assert _enforce_submission_profile_arms(
        ["aware"],
        profile=CANONICAL_PROFILE,
    ) == ["aware"]
    with pytest.raises(SystemExit, match="--arms aware"):
        _enforce_submission_profile_arms(
            ["naive", "aware"],
            profile=CANONICAL_PROFILE,
        )


def test_submission_profile_requires_docker_runner() -> None:
    # No profile: host subprocess stays the default.
    assert _enforce_submission_profile_runner(None, profile=None) == "subprocess"
    # Profile + no explicit runner defaults to the required docker runner.
    assert _enforce_submission_profile_runner(
        None, profile=CANONICAL_PROFILE
    ) == "docker"
    assert _enforce_submission_profile_runner(
        "docker", profile=CANONICAL_PROFILE
    ) == "docker"
    # Profile + host runner with no escape hatch is rejected.
    with pytest.raises(SystemExit, match="--runner docker"):
        _enforce_submission_profile_runner("subprocess", profile=CANONICAL_PROFILE)
    # The development escape hatch is honoured but yields a non-canonical run.
    assert _enforce_submission_profile_runner(
        "subprocess", profile=CANONICAL_PROFILE, allow_host_runner=True
    ) == "subprocess"


def test_benchmark_options_record_runner_kind() -> None:
    options = _benchmark_pipeline_options(
        max_total_steps=None,
        disable_replanning=False,
        max_code_repair_attempts=None,
        enable_repro_envelope=False,
        submission_profile=CANONICAL_PROFILE,
        runner_kind="docker",
    )
    assert options["runner_kind"] == "docker"
    # runner_kind stays out of the profile's own option bundle.
    assert "runner_kind" not in CANONICAL_PROFILE.as_pipeline_options()


def test_mock_provider_aware_arm_requires_explicit_smoke_opt_in() -> None:
    _enforce_mock_aware_provider(["naive"], provider="mock")
    _enforce_mock_aware_provider(["aware"], provider="openrouter")
    _enforce_mock_aware_provider(
        ["aware"], provider="mock", allow_mock_aware=True,
    )
    with pytest.raises(SystemExit, match="--allow-mock-aware"):
        _enforce_mock_aware_provider(["aware"], provider="mock")


def test_submission_profile_registry_is_versioned() -> None:
    old_profile = get_submission_profile("npj_dm/20260527")
    assert old_profile is NPJ_DM_2026_05
    profile = get_submission_profile("npj_dm/20260611")
    assert profile is CANONICAL_PROFILE
    assert DEFAULT_SUBMISSION_PROFILE_REF == profile.ref
    with pytest.raises(ValueError, match="Unknown submission profile"):
        get_submission_profile("npj_dm/main")


def test_submission_profile_as_pipeline_options_matches_canonical() -> None:
    opts = CANONICAL_PROFILE.as_pipeline_options()
    assert opts == {
        "evidence_enforcement_mode": "strict",
        "writer_digest_widened": True,
        "enable_reproducibility_envelope": True,
    }


def test_benchmark_options_merges_profile_overrides() -> None:
    options = _benchmark_pipeline_options(
        max_total_steps=None,
        disable_replanning=False,
        max_code_repair_attempts=None,
        enable_repro_envelope=False,
        submission_profile=CANONICAL_PROFILE,
    )
    for key, value in CANONICAL_PROFILE.as_pipeline_options().items():
        assert options[key] == value

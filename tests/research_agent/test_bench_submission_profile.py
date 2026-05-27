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
        _enforce_submission_profile_arms,
    )
finally:
    sys.path.pop(0)

from easyicu.research_agent.pipeline_profiles import (
    DEFAULT_SUBMISSION_PROFILE_REF,
    NPJ_DM_2026_05,
    get_submission_profile,
)


def test_submission_profile_forces_canonical_pipeline_options() -> None:
    options = _benchmark_pipeline_options(
        max_total_steps=None,
        disable_replanning=False,
        max_code_repair_attempts=None,
        enable_repro_envelope=False,
        submission_profile=NPJ_DM_2026_05,
    )
    assert options["evidence_enforcement_mode"] == "strict"
    assert options["enable_reproducibility_envelope"] is True
    assert options["writer_digest_widened"] is True
    assert options["submission_profile_name"] == "npj_dm"
    assert options["submission_profile_version"] == "20260527"
    assert options["submission_profile_locked_at"] == "2026-05-27T00:00:00Z"


def test_submission_profile_requires_aware_only_arm() -> None:
    assert _enforce_submission_profile_arms(
        ["aware"],
        profile=NPJ_DM_2026_05,
    ) == ["aware"]
    with pytest.raises(SystemExit, match="--arms aware"):
        _enforce_submission_profile_arms(
            ["naive", "aware"],
            profile=NPJ_DM_2026_05,
        )


def test_submission_profile_registry_is_versioned() -> None:
    profile = get_submission_profile("npj_dm/20260527")
    assert profile is NPJ_DM_2026_05
    assert profile.ref == "npj_dm/20260527"
    assert DEFAULT_SUBMISSION_PROFILE_REF == profile.ref
    with pytest.raises(ValueError, match="Unknown submission profile"):
        get_submission_profile("npj_dm/main")


def test_submission_profile_as_pipeline_options_matches_canonical() -> None:
    opts = NPJ_DM_2026_05.as_pipeline_options()
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
        submission_profile=NPJ_DM_2026_05,
    )
    for key, value in NPJ_DM_2026_05.as_pipeline_options().items():
        assert options[key] == value

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
        _enforce_development_resume_repair_budget,
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
    NPJ_DM_2026_07,
    NPJ_DM_2026_07_16,
    NPJ_DM_2026_07_17,
    SUBMISSION_PROFILE_REGISTRY,
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
    # No profile: capability-probed safe auto-selection is the default.
    assert _enforce_submission_profile_runner(None, profile=None) == "auto"
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


def test_benchmark_options_keep_execution_timeouts_independent() -> None:
    options = _benchmark_pipeline_options(
        max_total_steps=None,
        disable_replanning=False,
        max_code_repair_attempts=None,
        timeout_seconds=29.0,
        standard_executor_timeout_seconds=2_345.0,
        enable_repro_envelope=False,
    )

    assert options["timeout_seconds"] == 29.0
    assert options["standard_executor_timeout_seconds"] == 2_345.0


def test_benchmark_options_disable_cross_run_memory_by_default() -> None:
    # Canonical/benchmark runs must NOT inject cross-run RunMemory (StrategyCard)
    # into the planner: every resume reuses the workdir, so prior-run cards would
    # pollute a fresh/resumed run and undermine reproducibility. Default off; the
    # explicit opt-in flag is the only way to turn it back on. A submission
    # profile never re-enables it.
    default_options = _benchmark_pipeline_options(
        max_total_steps=None,
        disable_replanning=False,
        max_code_repair_attempts=None,
    )
    assert default_options["enable_memory"] is False
    # ExperienceBank is explicitly pinned off (guards against profile/default
    # drift silently re-opening cross-run experience injection).
    assert default_options["enable_experience_bank"] is False

    with_profile = _benchmark_pipeline_options(
        max_total_steps=None,
        disable_replanning=False,
        max_code_repair_attempts=None,
        submission_profile=CANONICAL_PROFILE,
    )
    assert with_profile["enable_memory"] is False
    assert with_profile["enable_experience_bank"] is False

    # The opt-in flag is for exploratory (profile-less) runs only.
    opted_in = _benchmark_pipeline_options(
        max_total_steps=None,
        disable_replanning=False,
        max_code_repair_attempts=None,
        enable_cross_run_memory=True,
    )
    assert opted_in["enable_memory"] is True
    assert opted_in["enable_experience_bank"] is False


def test_cross_run_memory_optin_is_rejected_for_a_submission_profile() -> None:
    # The profile pins the flags off; a CLI flag must never silently re-open
    # cross-run memory on a paper-facing run. Fail closed instead.
    with pytest.raises(SystemExit):
        _benchmark_pipeline_options(
            max_total_steps=None,
            disable_replanning=False,
            max_code_repair_attempts=None,
            submission_profile=CANONICAL_PROFILE,
            enable_cross_run_memory=True,
        )


def test_development_resume_can_raise_durable_step_repair_ceiling() -> None:
    value = _enforce_development_resume_repair_budget(
        3,
        resume_run_id="run_existing",
        resume_from_step_id="02_table_one",
        profile=None,
    )
    options = _benchmark_pipeline_options(
        max_total_steps=None,
        disable_replanning=False,
        max_code_repair_attempts=None,
        max_step_llm_repair_attempts=value,
        enable_repro_envelope=False,
    )

    assert options["max_step_llm_repair_attempts"] == 3


def test_step_repair_ceiling_override_requires_noncanonical_explicit_resume() -> None:
    with pytest.raises(SystemExit, match="requires both"):
        _enforce_development_resume_repair_budget(
            3,
            resume_run_id=None,
            resume_from_step_id=None,
            profile=None,
        )
    with pytest.raises(SystemExit, match="submission profile"):
        _enforce_development_resume_repair_budget(
            3,
            resume_run_id="run_existing",
            resume_from_step_id="02_table_one",
            profile=CANONICAL_PROFILE,
        )
    with pytest.raises(SystemExit, match="must be exactly 3"):
        _enforce_development_resume_repair_budget(
            -1,
            resume_run_id="run_existing",
            resume_from_step_id="02_table_one",
            profile=None,
        )
    with pytest.raises(SystemExit, match="must be exactly 3"):
        _enforce_development_resume_repair_budget(
            4,
            resume_run_id="run_existing",
            resume_from_step_id="02_table_one",
            profile=None,
        )


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
    bounds_profile = get_submission_profile("npj_dm/20260708")
    assert bounds_profile is NPJ_DM_2026_07
    assert (
        bounds_profile.expected_concept_dict_sha
        == "bc377779ce0f6b7983b2f8f527a37c1c394cc38e4a64055c9d9268b5f4d451ea"
    )
    # Prior default (20260716) stays retrievable as an immutable archival contract.
    assert get_submission_profile("npj_dm/20260716") is NPJ_DM_2026_07_16
    current_profile = get_submission_profile()
    assert current_profile is NPJ_DM_2026_07_17
    assert (
        current_profile.expected_concept_dict_sha
        == "b930e4384a07df16bc642a1e7df48d9fb5248c6bdac27f60fd78882ce612df54"
    )
    assert DEFAULT_SUBMISSION_PROFILE_REF == current_profile.ref
    with pytest.raises(ValueError, match="Unknown submission profile"):
        get_submission_profile("npj_dm/main")


def test_submission_profile_as_pipeline_options_matches_canonical() -> None:
    opts = CANONICAL_PROFILE.as_pipeline_options()
    assert opts == {
        "evidence_enforcement_mode": "strict",
        "writer_digest_widened": True,
        "enable_reproducibility_envelope": True,
        # Cross-run agent memory is submission-defining: a paper-facing run is
        # never steered by prior-run StrategyCards / ExperienceBank cards.
        "enable_memory": False,
        "enable_experience_bank": False,
    }


def test_every_submission_profile_pins_cross_run_memory_off() -> None:
    # Structural, not per-tool: the guarantee must hold for EVERY entrypoint
    # that applies a profile, and for every registered (incl. archival) profile.
    for ref, profile in SUBMISSION_PROFILE_REGISTRY.items():
        opts = profile.as_pipeline_options()
        assert opts["enable_memory"] is False, ref
        assert opts["enable_experience_bank"] is False, ref


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

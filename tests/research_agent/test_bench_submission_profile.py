"""Submission-profile guards for the research-agent benchmark runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_DIR))
try:
    from run_research_agent_bench import (  # type: ignore[import-not-found]
        _benchmark_pipeline_options,
        _default_submission_profile_ref,
        _enforce_development_resume_repair_budget,
        _enforce_mock_aware_provider,
        _enforce_submission_profile_arms,
        _enforce_submission_profile_runner,
    )
finally:
    sys.path.pop(0)

from easyicu.research_agent.orchestration.profiles import (
    DEFAULT_SUBMISSION_PROFILE_REF,
    NPJ_DM_2026_05,
    NPJ_DM_2026_06,
    NPJ_DM_2026_07,
    NPJ_DM_2026_07_16,
    NPJ_DM_2026_07_17,
    NPJ_DM_2026_07_18,
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
    # The 20260717 profile remains retrievable after the additive re-lock.
    assert get_submission_profile("npj_dm/20260717") is NPJ_DM_2026_07_17
    current_profile = get_submission_profile()
    assert current_profile is NPJ_DM_2026_07_18
    assert (
        current_profile.expected_concept_dict_sha
        == "fccadc53622dc82fe1dc8696617e52044168b6a84a9255e97e59df9e53bc5803"
    )
    assert DEFAULT_SUBMISSION_PROFILE_REF == current_profile.ref
    with pytest.raises(ValueError, match="Unknown submission profile"):
        get_submission_profile("npj_dm/main")


def test_pre_existing_profile_options_are_immutable_three_keys() -> None:
    # (d), not (a): the archival replay contracts must be byte-identical to
    # before the memory fields existed — exactly the original three keys, with
    # NO enable_memory / enable_experience_bank. Changing an old profile's
    # option bundle would silently change how its archived runs replay.
    for profile in (
        NPJ_DM_2026_05,
        NPJ_DM_2026_06,
        NPJ_DM_2026_07,
        NPJ_DM_2026_07_16,
    ):
        assert profile.as_pipeline_options() == {
            "evidence_enforcement_mode": "strict",
            "writer_digest_widened": True,
            "enable_reproducibility_envelope": True,
        }, profile.ref


def test_new_canonical_profile_pins_cross_run_memory_off() -> None:
    # The current profile pins cross-run memory OFF as a submission-defining
    # option — exactly five keys.  The archived 20260717 profile also retains
    # the explicit False values it was originally sealed with.
    assert NPJ_DM_2026_07_18.as_pipeline_options() == {
        "evidence_enforcement_mode": "strict",
        "writer_digest_widened": True,
        "enable_reproducibility_envelope": True,
        "enable_memory": False,
        "enable_experience_bank": False,
    }
    # The default profile is the one that pins it off.
    assert DEFAULT_SUBMISSION_PROFILE_REF == NPJ_DM_2026_07_18.ref


def test_every_profile_either_omits_or_pins_memory_off_never_on() -> None:
    # Structural guard: no registered profile may ever emit enable_memory=True.
    for ref, profile in SUBMISSION_PROFILE_REGISTRY.items():
        opts = profile.as_pipeline_options()
        assert opts.get("enable_memory", False) is False, ref
        assert opts.get("enable_experience_bank", False) is False, ref


def test_pre_existing_profile_to_dict_omits_memory_fields() -> None:
    # (d), not (a), at the PUBLIC-serialization layer too: to_dict() is the
    # public replay representation. Adding two Optional dataclass fields must NOT
    # leak "enable_memory": null / "enable_experience_bank": null into an
    # archival profile's to_dict() — that would silently change its serialized
    # form even though the value is unset. Old profiles keep exactly the
    # original field set (the ten dataclass fields + ref), with neither
    # memory key present.
    expected_keys = {
        "name",
        "version",
        "locked_at",
        "evidence_enforcement_mode",
        "writer_digest_widened",
        "enable_reproducibility_envelope",
        "requires_arm",
        "requires_runner",
        "expected_concept_dict_sha",
        "expected_sofa2_dict_sha",
        "ref",
    }
    for profile in (
        NPJ_DM_2026_05,
        NPJ_DM_2026_06,
        NPJ_DM_2026_07,
        NPJ_DM_2026_07_16,
    ):
        payload = profile.to_dict()
        assert set(payload) == expected_keys, profile.ref
        assert "enable_memory" not in payload, profile.ref
        assert "enable_experience_bank" not in payload, profile.ref


def test_new_canonical_profile_to_dict_surfaces_pinned_memory_fields() -> None:
    # The current profile surfaces both pinned keys in its public serialization
    # — explicitly False, not absent.
    payload = NPJ_DM_2026_07_18.to_dict()
    assert payload["enable_memory"] is False
    assert payload["enable_experience_bank"] is False


def test_20260717_profile_to_dict_remains_immutable_after_relock() -> None:
    expected = {
        "enable_experience_bank": False,
        "enable_memory": False,
        "enable_reproducibility_envelope": True,
        "evidence_enforcement_mode": "strict",
        "expected_concept_dict_sha": "b930e4384a07df16bc642a1e7df48d9fb5248c6bdac27f60fd78882ce612df54",
        "expected_sofa2_dict_sha": "65075a691ef103112d9df0df452601299c37603c1c075742fe211bb75d2f92cc",
        "locked_at": "2026-07-17T00:00:00Z",
        "name": "npj_dm",
        "ref": "npj_dm/20260717",
        "requires_arm": "aware",
        "requires_runner": "docker",
        "version": "20260717",
        "writer_digest_widened": True,
    }
    assert NPJ_DM_2026_07_17.to_dict() == expected
    assert json.dumps(NPJ_DM_2026_07_17.to_dict(), sort_keys=True) == json.dumps(
        expected, sort_keys=True
    )


def test_benchmark_default_profile_is_registry_owned() -> None:
    assert _default_submission_profile_ref() == DEFAULT_SUBMISSION_PROFILE_REF


def test_current_profile_is_reexported_by_package_identity() -> None:
    import easyicu.research_agent as research_agent

    assert research_agent.NPJ_DM_2026_07_18 is NPJ_DM_2026_07_18


# Frozen canonical to_dict() snapshots for the archival profiles. Key-set
# equality alone would miss a field-VALUE change; these lock the full public
# replay representation so "byte-identical" is actually enforced (canonical JSON
# with sort_keys makes the comparison order-independent and diffable).
_ARCHIVAL_TO_DICT_SNAPSHOTS = {
    "npj_dm/20260527": {
        "enable_reproducibility_envelope": True,
        "evidence_enforcement_mode": "strict",
        "expected_concept_dict_sha": "9ef52ed3ec51652f235c92a1394d4f4b91318cbd46e3915a5eacbbed2754e179",
        "expected_sofa2_dict_sha": "e1844deafad9151aa5069824ff335bf59e228b97040a8bd884d23e0457047b25",
        "locked_at": "2026-05-27T00:00:00Z",
        "name": "npj_dm",
        "ref": "npj_dm/20260527",
        "requires_arm": "aware",
        "requires_runner": "docker",
        "version": "20260527",
        "writer_digest_widened": True,
    },
    "npj_dm/20260611": {
        "enable_reproducibility_envelope": True,
        "evidence_enforcement_mode": "strict",
        "expected_concept_dict_sha": "4b9c55bf9ec5dc92c39d6c14b036f0b19d4da684d9808618833b83d6b53c9ed2",
        "expected_sofa2_dict_sha": "b26e36b6ef5ea947027c8f7cd514fc5174545aa658187d6bdb8ec43f2a80b6aa",
        "locked_at": "2026-06-11T00:00:00Z",
        "name": "npj_dm",
        "ref": "npj_dm/20260611",
        "requires_arm": "aware",
        "requires_runner": "docker",
        "version": "20260611",
        "writer_digest_widened": True,
    },
    "npj_dm/20260708": {
        "enable_reproducibility_envelope": True,
        "evidence_enforcement_mode": "strict",
        "expected_concept_dict_sha": "bc377779ce0f6b7983b2f8f527a37c1c394cc38e4a64055c9d9268b5f4d451ea",
        "expected_sofa2_dict_sha": "b26e36b6ef5ea947027c8f7cd514fc5174545aa658187d6bdb8ec43f2a80b6aa",
        "locked_at": "2026-07-08T00:25:43-04:00",
        "name": "npj_dm",
        "ref": "npj_dm/20260708",
        "requires_arm": "aware",
        "requires_runner": "docker",
        "version": "20260708",
        "writer_digest_widened": True,
    },
    "npj_dm/20260716": {
        "enable_reproducibility_envelope": True,
        "evidence_enforcement_mode": "strict",
        "expected_concept_dict_sha": "095350e3d897ed6824673b229435941932bd8270b75667826e8b32538e5de146",
        "expected_sofa2_dict_sha": "b26e36b6ef5ea947027c8f7cd514fc5174545aa658187d6bdb8ec43f2a80b6aa",
        "locked_at": "2026-07-16T10:17:17-04:00",
        "name": "npj_dm",
        "ref": "npj_dm/20260716",
        "requires_arm": "aware",
        "requires_runner": "docker",
        "version": "20260716",
        "writer_digest_widened": True,
    },
}


def test_archival_profile_to_dict_matches_frozen_canonical_snapshot() -> None:
    # Value-level lock (not just key set): any change to a field value of an
    # archival profile — including accidentally surfacing a memory field — breaks
    # this, because the frozen snapshots have neither memory key and pin every
    # SHA / flag.
    for profile in (
        NPJ_DM_2026_05,
        NPJ_DM_2026_06,
        NPJ_DM_2026_07,
        NPJ_DM_2026_07_16,
    ):
        expected = _ARCHIVAL_TO_DICT_SNAPSHOTS[profile.ref]
        assert profile.to_dict() == expected, profile.ref
        # Canonical-JSON equality makes the "byte-identical" claim literal.
        assert json.dumps(profile.to_dict(), sort_keys=True) == json.dumps(
            expected, sort_keys=True
        ), profile.ref


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

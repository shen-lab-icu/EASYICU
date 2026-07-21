"""Versioned pipeline profiles for paper-facing research-agent runs.

Profiles freeze option bundles that must be reproducible across paper
drafts, reviews, and reruns. A profile is not another agent mode; it is
part of the evaluation/submission scaffold defined in
``docs/architecture_glossary.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SubmissionProfile:
    """Frozen option bundle for a paper-facing run."""

    name: str
    version: str
    locked_at: str
    evidence_enforcement_mode: str
    writer_digest_widened: bool
    enable_reproducibility_envelope: bool
    requires_arm: str
    # Paper-facing runs must execute agent-generated code in a
    # network-isolated container (``docker run --network none`` with a
    # read-only cohort mount), not on the host subprocess. Enforced by
    # the benchmark runner; ``requires_runner`` itself stays out of the
    # profile's pipeline option bundle, while the resolved ``runner_kind``
    # is recorded separately by the bench wrapper.
    requires_runner: str = "docker"
    expected_concept_dict_sha: Optional[str] = None
    expected_sofa2_dict_sha: Optional[str] = None
    # Cross-run agent-memory policy. ``None`` = the profile does not pin it (so
    # ``as_pipeline_options`` omits the key and the pre-existing profiles are
    # byte-identical replay contracts). Only a profile that explicitly sets these
    # to ``False`` pins cross-run memory OFF as a submission-defining option.
    enable_memory: Optional[bool] = None
    enable_experience_bank: Optional[bool] = None
    # Deterministic mock-generated planner/coder fallbacks are useful only for
    # tests and offline demonstrations.  ``None`` preserves pre-existing
    # profiles byte-for-byte; the current paper profile explicitly pins both
    # fallbacks off so a provider failure cannot turn into fixture science.
    enable_deterministic_code_fallback: Optional[bool] = None
    enable_deterministic_planner_fallback: Optional[bool] = None
    # Bench-wrapper policy rather than a PipelineConfig key.  When true, even
    # the explicit ``--allow-mock-aware`` smoke escape hatch is incompatible
    # with this paper-facing profile.
    requires_real_provider: Optional[bool] = None
    # Research Know-How can alter study design, so enabling it is a
    # submission-defining coordinate. Historical profiles leave this None and
    # remain byte-identical; only an additive profile may opt in.
    enable_know_how: Optional[bool] = None

    @property
    def ref(self) -> str:
        return f"{self.name}/{self.version}"

    def as_pipeline_options(self) -> Dict[str, Any]:
        """Return canonical PipelineConfig overrides for this profile.

        The profile owns only the submission-defining flags here. Run-shape
        knobs such as ``max_total_steps`` and ``llm_seed`` remain caller-owned.
        """

        options: Dict[str, Any] = {
            "evidence_enforcement_mode": self.evidence_enforcement_mode,
            "writer_digest_widened": self.writer_digest_widened,
            "enable_reproducibility_envelope": self.enable_reproducibility_envelope,
        }
        # Cross-run agent memory is submission-DEFINING, not a run-shape knob: a
        # paper-facing run must not be steered by StrategyCards distilled from a
        # prior run of the same workdir (every resume reuses it) nor by
        # ExperienceBank cards — that is unvalidated procedural memory and it
        # makes the run irreproducible. Pinning it on the PROFILE is what makes
        # the guarantee hold for every entrypoint that applies a profile; a
        # per-tool default is silently bypassed by the next entrypoint. Within-run
        # authority (StepAuthorityCapsule / checkpoints / EvidenceStore) does not
        # use RunMemory and is unaffected.
        #
        # A profile that leaves these ``None`` does NOT pin them, so its option
        # bundle is byte-identical to before this field existed — the pre-existing
        # profiles remain immutable replay contracts. Only a profile that
        # explicitly sets ``enable_memory=False`` emits the key.
        if self.enable_memory is not None:
            options["enable_memory"] = self.enable_memory
        if self.enable_experience_bank is not None:
            options["enable_experience_bank"] = self.enable_experience_bank
        if self.enable_deterministic_code_fallback is not None:
            options["enable_deterministic_code_fallback"] = (
                self.enable_deterministic_code_fallback
            )
        if self.enable_deterministic_planner_fallback is not None:
            options["enable_deterministic_planner_fallback"] = (
                self.enable_deterministic_planner_fallback
            )
        if self.enable_know_how is not None:
            options["enable_know_how"] = self.enable_know_how
        return options

    def pipeline_options(self) -> Dict[str, Any]:
        """Return PipelineConfig overrides, including manifest profile metadata."""

        return {
            **self.as_pipeline_options(),
            "submission_profile_name": self.name,
            "submission_profile_version": self.version,
            "submission_profile_locked_at": self.locked_at,
            "expected_concept_dict_sha": self.expected_concept_dict_sha,
            "expected_sofa2_dict_sha": self.expected_sofa2_dict_sha,
        }

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        # The cross-run memory fields are additive: a profile that does not pin
        # them (value ``None``) must serialize byte-identically to before the
        # fields existed, so its PUBLIC replay representation is unchanged. Only
        # a profile that explicitly pins them (``True``/``False``) surfaces the
        # keys. Mirrors ``as_pipeline_options`` — both are replay contracts.
        for field_name in (
            "enable_memory",
            "enable_experience_bank",
            "enable_deterministic_code_fallback",
            "enable_deterministic_planner_fallback",
            "requires_real_provider",
            "enable_know_how",
        ):
            if payload.get(field_name) is None:
                payload.pop(field_name, None)
        payload["ref"] = self.ref
        return payload


NPJ_DM_2026_05 = SubmissionProfile(
    name="npj_dm",
    version="20260527",
    locked_at="2026-05-27T00:00:00Z",
    evidence_enforcement_mode="strict",
    writer_digest_widened=True,
    enable_reproducibility_envelope=True,
    requires_arm="aware",
    requires_runner="docker",
    # Locked 2026-05-27 after the user's 77/77 user-added concept audit and
    # 112/112 itemid verification (see docs/concept_dict_audit_log.md).
    expected_concept_dict_sha="9ef52ed3ec51652f235c92a1394d4f4b91318cbd46e3915a5eacbbed2754e179",
    expected_sofa2_dict_sha="e1844deafad9151aa5069824ff335bf59e228b97040a8bd884d23e0457047b25",
)

NPJ_DM_2026_06 = SubmissionProfile(
    name="npj_dm",
    version="20260611",
    locked_at="2026-06-11T00:00:00Z",
    evidence_enforcement_mode="strict",
    writer_digest_widened=True,
    enable_reproducibility_envelope=True,
    requires_arm="aware",
    requires_runner="docker",
    # Locked after the 2026-06-09 concept coverage and tracker audit refresh
    # that added the SICdb ventilation mapping and SOFA-2 dictionary updates.
    # Re-locked 2026-06-22: the 22 extended-feature concepts (severity scores,
    # extra labs, comorbidity/microbiology/outcome loaders, and the 10 Tier-1
    # derived indices) are now part of the frozen submission dictionary.
    # Re-locked 2026-07-04. concept-dict.json SHA = the recall-audit pass
    # (within-concept itemid/version recovery across all 6 DBs) plus the new
    # hemodynamic/neuro/endocrine and lab concepts (icp, svo2, scvo2, pawp,
    # cortisol, pct, bnp, uric_acid, the lipid panel, iron studies, ft4, t4,
    # prealbumin, myoglobin, retic) — committed 7eb2f67..445e297. sofa2-dict.json
    # SHA tracks the in-flight 2026-07-03 RRT/CRRT + ventilation mimic_demo
    # coverage review currently in the working tree; commit that dict change and
    # this re-lock together (do not commit one without the other).
    expected_concept_dict_sha="4b9c55bf9ec5dc92c39d6c14b036f0b19d4da684d9808618833b83d6b53c9ed2",
    expected_sofa2_dict_sha="b26e36b6ef5ea947027c8f7cd514fc5174545aa658187d6bdb8ec43f2a80b6aa",
)

NPJ_DM_2026_07 = SubmissionProfile(
    name="npj_dm",
    version="20260708",
    locked_at="2026-07-08T00:25:43-04:00",
    evidence_enforcement_mode="strict",
    writer_digest_widened=True,
    enable_reproducibility_envelope=True,
    requires_arm="aware",
    requires_runner="docker",
    # Re-locked after the cross-database bounds audit added or corrected six
    # physiology/artifact ceilings in 836a896. Preserve older profiles as
    # immutable replay contracts instead of silently changing their hashes.
    expected_concept_dict_sha="bc377779ce0f6b7983b2f8f527a37c1c394cc38e4a64055c9d9268b5f4d451ea",
    expected_sofa2_dict_sha="b26e36b6ef5ea947027c8f7cd514fc5174545aa658187d6bdb8ec43f2a80b6aa",
)

NPJ_DM_2026_07_16 = SubmissionProfile(
    name="npj_dm",
    version="20260716",
    locked_at="2026-07-16T10:17:17-04:00",
    evidence_enforcement_mode="strict",
    writer_digest_widened=True,
    enable_reproducibility_envelope=True,
    requires_arm="aware",
    requires_runner="docker",
    # Re-locked after ea9fc98 made SICdb hospital-discharge type an explicit
    # input to the fail-closed in-hospital mortality callback. Keep the
    # 20260708 profile immutable so archived runs retain their original
    # dictionary authority.
    expected_concept_dict_sha="095350e3d897ed6824673b229435941932bd8270b75667826e8b32538e5de146",
    expected_sofa2_dict_sha="b26e36b6ef5ea947027c8f7cd514fc5174545aa658187d6bdb8ec43f2a80b6aa",
)

NPJ_DM_2026_07_17 = SubmissionProfile(
    name="npj_dm",
    version="20260717",
    locked_at="2026-07-17T00:00:00Z",
    evidence_enforcement_mode="strict",
    writer_digest_widened=True,
    enable_reproducibility_envelope=True,
    requires_arm="aware",
    requires_runner="docker",
    # Re-locked 2026-07-17 (定档) after the own-concept audit + new ventilator
    # concepts entered the frozen dictionary: 38 own-concept aggregate-arity/unit/
    # route fixes and 43 within-concept recall recoveries, plus vent_mode,
    # vent_breath_seq, and driving_pres_controlled (per-mode driving pressure).
    # sofa2-dict.json tracks the accompanying ventilation/SOFA-2 coverage update.
    # See concept-dict.LOCK.json. Older profiles stay immutable replay contracts.
    expected_concept_dict_sha="b930e4384a07df16bc642a1e7df48d9fb5248c6bdac27f60fd78882ce612df54",
    expected_sofa2_dict_sha="65075a691ef103112d9df0df452601299c37603c1c075742fe211bb75d2f92cc",
    # This profile pins cross-run agent memory OFF as a submission-defining
    # option (the older profiles leave it None → their option bundles stay
    # byte-identical immutable replay contracts).
    enable_memory=False,
    enable_experience_bank=False,
)

NPJ_DM_2026_07_18 = SubmissionProfile(
    name="npj_dm",
    version="20260718",
    locked_at="2026-07-18T17:55:00-04:00",
    evidence_enforcement_mode="strict",
    writer_digest_widened=True,
    enable_reproducibility_envelope=True,
    requires_arm="aware",
    requires_runner="docker",
    # Additive re-lock after the post-full6 data-foundation corrections in
    # 58d2267, 63b0967, and 8e97d31.  This profile pins the packaged dictionary
    # bytes used to materialize fresh canonical inputs; it does not rewrite the
    # historical full6_20260717 extraction lock or authorize reuse of cohorts
    # materialized under that older dictionary authority.
    expected_concept_dict_sha="fccadc53622dc82fe1dc8696617e52044168b6a84a9255e97e59df9e53bc5803",
    expected_sofa2_dict_sha="61f37a41083cd96df49a2e61d26c682e9d090d0a22d05ff97ba85a966b165b1c",
    enable_memory=False,
    enable_experience_bank=False,
)

NPJ_DM_2026_07_19 = SubmissionProfile(
    name="npj_dm",
    version="20260719",
    locked_at="2026-07-19T11:45:00-04:00",
    evidence_enforcement_mode="strict",
    writer_digest_widened=True,
    enable_reproducibility_envelope=True,
    requires_arm="aware",
    requires_runner="docker",
    # Additive protocol re-lock after Planner-owned primary-role authority.
    # Dictionary bytes are unchanged from 20260718; this profile additionally
    # prevents test/demo mock fallbacks from acquiring paper-facing authority.
    expected_concept_dict_sha="fccadc53622dc82fe1dc8696617e52044168b6a84a9255e97e59df9e53bc5803",
    expected_sofa2_dict_sha="61f37a41083cd96df49a2e61d26c682e9d090d0a22d05ff97ba85a966b165b1c",
    enable_memory=False,
    enable_experience_bank=False,
    enable_deterministic_code_fallback=False,
    enable_deterministic_planner_fallback=False,
    requires_real_provider=True,
)

NPJ_DM_2026_07_21_KNOW_HOW = SubmissionProfile(
    name="npj_dm_know_how_dev",
    version="20260721",
    locked_at="2026-07-21T12:00:00-04:00",
    evidence_enforcement_mode="strict",
    writer_digest_widened=True,
    enable_reproducibility_envelope=True,
    requires_arm="aware",
    requires_runner="docker",
    expected_concept_dict_sha=NPJ_DM_2026_07_19.expected_concept_dict_sha,
    expected_sofa2_dict_sha=NPJ_DM_2026_07_19.expected_sofa2_dict_sha,
    enable_memory=False,
    enable_experience_bank=False,
    enable_deterministic_code_fallback=False,
    enable_deterministic_planner_fallback=False,
    requires_real_provider=True,
    enable_know_how=True,
)

DEFAULT_SUBMISSION_PROFILE_REF = NPJ_DM_2026_07_19.ref
SUBMISSION_PROFILE_REGISTRY: Dict[str, SubmissionProfile] = {
    NPJ_DM_2026_05.ref: NPJ_DM_2026_05,
    NPJ_DM_2026_06.ref: NPJ_DM_2026_06,
    NPJ_DM_2026_07.ref: NPJ_DM_2026_07,
    NPJ_DM_2026_07_16.ref: NPJ_DM_2026_07_16,
    NPJ_DM_2026_07_17.ref: NPJ_DM_2026_07_17,
    NPJ_DM_2026_07_18.ref: NPJ_DM_2026_07_18,
    NPJ_DM_2026_07_19.ref: NPJ_DM_2026_07_19,
    NPJ_DM_2026_07_21_KNOW_HOW.ref: NPJ_DM_2026_07_21_KNOW_HOW,
}


def get_submission_profile(ref: Optional[str] = None) -> SubmissionProfile:
    """Return a registered submission profile by ``name/version`` ref."""

    key = ref or DEFAULT_SUBMISSION_PROFILE_REF
    try:
        return SUBMISSION_PROFILE_REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(SUBMISSION_PROFILE_REGISTRY))
        raise ValueError(
            f"Unknown submission profile {key!r}; choose from: {known}"
        ) from exc


def require_profile_know_how_setting(
    *,
    name: Optional[str],
    version: Optional[str],
    enabled: bool,
) -> None:
    """Keep the study-design-affecting Know-How flag profile-owned."""
    if name is None:
        return
    ref = f"{name}/{version}"
    expected = bool(get_submission_profile(ref).enable_know_how)
    if bool(enabled) != expected:
        raise ValueError(
            "Research Know-How changes study design and must match an additive "
            f"submission profile coordinate; profile {ref!r} pins "
            f"enable_know_how={expected}"
        )


__all__ = [
    "SubmissionProfile",
    "NPJ_DM_2026_05",
    "NPJ_DM_2026_06",
    "NPJ_DM_2026_07",
    "NPJ_DM_2026_07_16",
    "NPJ_DM_2026_07_17",
    "NPJ_DM_2026_07_18",
    "NPJ_DM_2026_07_19",
    "NPJ_DM_2026_07_21_KNOW_HOW",
    "DEFAULT_SUBMISSION_PROFILE_REF",
    "SUBMISSION_PROFILE_REGISTRY",
    "get_submission_profile",
    "require_profile_know_how_setting",
]

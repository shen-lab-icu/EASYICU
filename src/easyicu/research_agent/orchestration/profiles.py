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
    # Step-scoped Action/Software/Data resources alter Coder prompts and resume
    # authority. Historical profiles therefore omit the coordinate entirely;
    # only an additive development profile may enable Phase-2 wiring.
    enable_coder_resources: Optional[bool] = None
    # Reviewed/promoted memory is a separate, permissioned system from legacy
    # RunMemory. Both its enable flag and exact namespaces are profile-owned.
    enable_reviewed_memory: Optional[bool] = None
    reviewed_memory_namespaces: Optional[tuple[str, ...]] = None
    # New analytical software is never installed in a running sandbox. A
    # request profile may emit a reviewable capability request; a later,
    # additive profile must pin the rebuilt runner image before activation.
    enable_capability_workflow: Optional[bool] = None
    expected_runner_image_digest: Optional[str] = None
    # Whether rendered figure bytes may leave the host for external visual
    # review. Submission-defining: it decides whether a manuscript figure was
    # ever transmitted off-site, which a privacy reviewer must be able to read
    # off the profile rather than infer from a caller's kwargs. ``None``
    # preserves the existing profiles byte-for-byte.
    allow_external_figure_upload: Optional[bool] = None
    # Diagnostic profiles may authorize a real Planner call while forbidding
    # every transition into Execute, regardless of which entrypoint applies the
    # profile. ``None`` preserves historical profile replay bytes.
    planner_only: Optional[bool] = None

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
        if self.allow_external_figure_upload is not None:
            options["allow_external_figure_upload"] = self.allow_external_figure_upload
        if self.enable_know_how is not None:
            options["enable_know_how"] = self.enable_know_how
        if self.enable_coder_resources is not None:
            options["enable_coder_resources"] = self.enable_coder_resources
        if self.enable_reviewed_memory is not None:
            options["enable_reviewed_memory"] = self.enable_reviewed_memory
        if self.reviewed_memory_namespaces is not None:
            options["reviewed_memory_namespaces"] = self.reviewed_memory_namespaces
        if self.enable_capability_workflow is not None:
            options["enable_capability_workflow"] = self.enable_capability_workflow
        if self.expected_runner_image_digest is not None:
            options["expected_runner_image_digest"] = self.expected_runner_image_digest
        if self.planner_only is not None:
            options["planner_only"] = self.planner_only
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
            "enable_coder_resources",
            "enable_reviewed_memory",
            "reviewed_memory_namespaces",
            "enable_capability_workflow",
            "expected_runner_image_digest",
            "allow_external_figure_upload",
            "planner_only",
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

E1_PLANNER_CANARY_2026_08_14 = SubmissionProfile(
    name="npj_dm_e1_canary_dev",
    version="20260814",
    locked_at="2026-08-14T15:15:00-04:00",
    evidence_enforcement_mode="strict",
    writer_digest_widened=True,
    enable_reproducibility_envelope=True,
    requires_arm="aware",
    requires_runner="docker",
    # This diagnostic profile can reach Planner but cannot be approved into
    # Execute. Publication still requires a separately sealed profile and data.
    expected_concept_dict_sha="22039e19c9b499d635dce956298550cecb1fdf55059304736cca73ee42bf129a",
    expected_sofa2_dict_sha="998a14c70c8a983c71ce6af2da8408fe22063cc042e8cde69f572083880bdaf8",
    enable_memory=False,
    enable_experience_bank=False,
    enable_deterministic_code_fallback=False,
    enable_deterministic_planner_fallback=False,
    requires_real_provider=True,
    planner_only=True,
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

NPJ_DM_2026_07_22_FRAMEWORK_V2_DEV = SubmissionProfile(
    name="npj_dm_framework_v2_dev",
    version="20260722",
    locked_at="2026-07-22T06:00:00-04:00",
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
    enable_know_how=False,
    enable_coder_resources=True,
)

NPJ_DM_2026_07_22_FRAMEWORK_V2_MEMORY_DEV = SubmissionProfile(
    name="npj_dm_framework_v2_memory_dev",
    version="20260722",
    locked_at="2026-07-22T08:00:00-04:00",
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
    enable_know_how=False,
    enable_coder_resources=True,
    enable_reviewed_memory=True,
    reviewed_memory_namespaces=(
        "reviewed_knowledge/framework_v2",
        "promoted_lessons/1.0.0",
    ),
)

NPJ_DM_2026_07_22_FRAMEWORK_V2_CAPABILITY_DEV = SubmissionProfile(
    name="npj_dm_framework_v2_capability_dev",
    version="20260722",
    locked_at="2026-07-22T10:00:00-04:00",
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
    enable_know_how=False,
    enable_coder_resources=True,
    enable_reviewed_memory=True,
    reviewed_memory_namespaces=(
        "reviewed_knowledge/framework_v2",
        "promoted_lessons/1.0.0",
    ),
    enable_capability_workflow=True,
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
    E1_PLANNER_CANARY_2026_08_14.ref: E1_PLANNER_CANARY_2026_08_14,
    NPJ_DM_2026_07_21_KNOW_HOW.ref: NPJ_DM_2026_07_21_KNOW_HOW,
    NPJ_DM_2026_07_22_FRAMEWORK_V2_DEV.ref: (NPJ_DM_2026_07_22_FRAMEWORK_V2_DEV),
    NPJ_DM_2026_07_22_FRAMEWORK_V2_MEMORY_DEV.ref: (
        NPJ_DM_2026_07_22_FRAMEWORK_V2_MEMORY_DEV
    ),
    NPJ_DM_2026_07_22_FRAMEWORK_V2_CAPABILITY_DEV.ref: (
        NPJ_DM_2026_07_22_FRAMEWORK_V2_CAPABILITY_DEV
    ),
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


def is_paper_facing_profile(name: Optional[str]) -> bool:
    """Return whether a profile name makes paper-facing claims.

    The registry holds two families under the same ``npj_dm`` stem: the frozen
    submission profiles, and additive ``*_dev`` profiles that exist to exercise
    in-flight wiring. Gates that must not be relaxable — provenance
    completeness, reviewer authentication, execution resource floors — key off
    this rather than off "a profile name was supplied at all", which would
    hold development runs to submission rules, or off an allow-list of exact
    refs, which silently exempts every profile added later.
    """

    return bool(name) and not str(name).endswith("_dev")


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


def require_profile_coder_resource_setting(
    *,
    name: Optional[str],
    version: Optional[str],
    enabled: bool,
) -> None:
    """Keep Coder resource selection profile-owned and replay-safe."""
    if name is None:
        if enabled:
            raise ValueError("Coder resources require an additive submission profile")
        return
    ref = f"{name}/{version}"
    expected = bool(get_submission_profile(ref).enable_coder_resources)
    if bool(enabled) != expected:
        raise ValueError(
            "Coder resource selection changes prompt and resume authority and must "
            f"match the submission profile; profile {ref!r} pins "
            f"enable_coder_resources={expected}"
        )


def require_profile_reviewed_memory_setting(
    *,
    name: Optional[str],
    version: Optional[str],
    enabled: bool,
    namespaces: tuple[str, ...],
) -> None:
    """Keep reviewed-memory reads profile-owned and replay-safe."""

    if name is None:
        if enabled or namespaces:
            raise ValueError("Reviewed memory requires an additive submission profile")
        return
    ref = f"{name}/{version}"
    profile = get_submission_profile(ref)
    expected_enabled = bool(profile.enable_reviewed_memory)
    expected_namespaces = tuple(profile.reviewed_memory_namespaces or ())
    if bool(enabled) != expected_enabled or namespaces != expected_namespaces:
        raise ValueError(
            "Reviewed-memory consumption changes Coder and resume authority and "
            f"must match the submission profile; profile {ref!r} pins "
            f"enable_reviewed_memory={expected_enabled}, "
            f"reviewed_memory_namespaces={expected_namespaces!r}"
        )


def require_profile_capability_workflow_setting(
    *,
    name: Optional[str],
    version: Optional[str],
    enabled: bool,
    expected_runner_image_digest: Optional[str],
) -> None:
    """Keep capability requests/activations on additive profiles."""

    if name is None:
        if enabled or expected_runner_image_digest is not None:
            raise ValueError("Capability workflow requires an additive profile")
        return
    ref = f"{name}/{version}"
    profile = get_submission_profile(ref)
    expected_enabled = bool(profile.enable_capability_workflow)
    expected_digest = profile.expected_runner_image_digest
    if enabled != expected_enabled or expected_runner_image_digest != expected_digest:
        raise ValueError(
            "Capability workflow changes runtime authority and must match the "
            f"submission profile; profile {ref!r} pins "
            f"enable_capability_workflow={expected_enabled}, "
            f"expected_runner_image_digest={expected_digest!r}"
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
    "E1_PLANNER_CANARY_2026_08_14",
    "NPJ_DM_2026_07_21_KNOW_HOW",
    "NPJ_DM_2026_07_22_FRAMEWORK_V2_DEV",
    "NPJ_DM_2026_07_22_FRAMEWORK_V2_MEMORY_DEV",
    "NPJ_DM_2026_07_22_FRAMEWORK_V2_CAPABILITY_DEV",
    "DEFAULT_SUBMISSION_PROFILE_REF",
    "SUBMISSION_PROFILE_REGISTRY",
    "get_submission_profile",
    "is_paper_facing_profile",
    "require_profile_know_how_setting",
    "require_profile_coder_resource_setting",
    "require_profile_reviewed_memory_setting",
    "require_profile_capability_workflow_setting",
]

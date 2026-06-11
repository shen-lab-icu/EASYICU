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

    @property
    def ref(self) -> str:
        return f"{self.name}/{self.version}"

    def as_pipeline_options(self) -> Dict[str, Any]:
        """Return canonical PipelineConfig overrides for this profile.

        The profile owns only the submission-defining flags here. Run-shape
        knobs such as ``max_total_steps`` and ``llm_seed`` remain caller-owned.
        """

        return {
            "evidence_enforcement_mode": self.evidence_enforcement_mode,
            "writer_digest_widened": self.writer_digest_widened,
            "enable_reproducibility_envelope": self.enable_reproducibility_envelope,
        }

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
    expected_concept_dict_sha="85bf8b5fd819881e41b4ba57167ca664db5ec5b83e8524e40291bb9c56a719df",
    expected_sofa2_dict_sha="22c4de886b6cb98ccb9145ce9e310780cb1b9b3feaf7a0c3f8b283647e4777b5",
)

DEFAULT_SUBMISSION_PROFILE_REF = NPJ_DM_2026_06.ref
SUBMISSION_PROFILE_REGISTRY: Dict[str, SubmissionProfile] = {
    NPJ_DM_2026_05.ref: NPJ_DM_2026_05,
    NPJ_DM_2026_06.ref: NPJ_DM_2026_06,
}


def get_submission_profile(ref: Optional[str] = None) -> SubmissionProfile:
    """Return a registered submission profile by ``name/version`` ref."""

    key = ref or DEFAULT_SUBMISSION_PROFILE_REF
    try:
        return SUBMISSION_PROFILE_REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(SUBMISSION_PROFILE_REGISTRY))
        raise ValueError(f"Unknown submission profile {key!r}; choose from: {known}") from exc


__all__ = [
    "SubmissionProfile",
    "NPJ_DM_2026_05",
    "NPJ_DM_2026_06",
    "DEFAULT_SUBMISSION_PROFILE_REF",
    "SUBMISSION_PROFILE_REGISTRY",
    "get_submission_profile",
]

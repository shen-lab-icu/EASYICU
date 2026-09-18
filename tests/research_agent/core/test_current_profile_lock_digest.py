"""C-F10: CURRENT profile refs must bind the LOCK.json actual digests.

Asserts the CURRENT E1 dev profile coordinates equal the real
``concept-dict.LOCK.json`` digests (not a stale hardcoded copy).
"""

from __future__ import annotations

import json
from pathlib import Path


def _lock_digests() -> tuple[str, str]:
    repo = Path(__file__).resolve().parents[3]
    payload = json.loads(
        (repo / "src" / "easyicu" / "data" / "concept-dict.LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    return (
        str(payload["concept_dict_sha256"]).strip().lower(),
        str(payload["sofa2_dict_sha256"]).strip().lower(),
    )


def test_current_e1_refs_match_lock_digest() -> None:
    from easyicu.research_agent.orchestration.profiles import (
        CURRENT_E1_PLANNER_CANARY_DEV_PROFILE_REF,
        CURRENT_E1_REVIEWED_DEMO_DEV_PROFILE_REF,
        get_submission_profile,
    )

    concept_sha, sofa2_sha = _lock_digests()
    assert len(concept_sha) == 64 and len(sofa2_sha) == 64
    for ref in (
        CURRENT_E1_PLANNER_CANARY_DEV_PROFILE_REF,
        CURRENT_E1_REVIEWED_DEMO_DEV_PROFILE_REF,
    ):
        profile = get_submission_profile(ref)
        assert profile.ref == ref
        assert (profile.expected_concept_dict_sha or "").lower() == concept_sha
        assert (profile.expected_sofa2_dict_sha or "").lower() == sofa2_sha

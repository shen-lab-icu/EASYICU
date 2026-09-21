"""The real foundation state must not be masked by synthetic fixtures.

Restored from the v4 publication lock review (bfa903af).
"""
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_current_foundation_lock_matches_resources_and_allows_named_v6_seal():
    path = ROOT / "scripts/releases/EX-A01_seal_full6_release.py"
    spec = importlib.util.spec_from_file_location("current_foundation_sealer", path)
    sealer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sealer)
    lock = json.loads(sealer.FOUNDATION_LOCK_PATH.read_text())
    dictionary = sealer.FOUNDATION_RESOURCE_PATHS["concept_dictionary_sha256"]
    sofa2 = sealer.FOUNDATION_RESOURCE_PATHS["sofa2_dictionary_sha256"]

    assert lock["finalized"] is True
    assert lock["total_concepts"] == len(json.loads(dictionary.read_text()))
    assert lock["concept_dict_sha256"] == hashlib.sha256(
        dictionary.read_bytes()
    ).hexdigest()
    assert lock["sofa2_dict_sha256"] == hashlib.sha256(sofa2.read_bytes()).hexdigest()
    assert lock["pending_foundation_changes"] == []
    assert lock["finalized_foundation_changes"]
    assert lock["locked_for_extraction_run"] == (
        "full6_native_v6_clean_rebuild_97f508c6_20260921"
    )
    for field in (
        "clinical_contracts_sha256",
        "clinical_contract_validator_sha256",
        "data_sources_sha256",
    ):
        assert lock[field] == hashlib.sha256(
            sealer.FOUNDATION_RESOURCE_PATHS[field].read_bytes()
        ).hexdigest()

    validated = sealer._validate_foundation_lock()
    assert validated["lock_finalized"] is True
    assert validated["locked_for_extraction_run"] == lock["locked_for_extraction_run"]


def test_all_current_e1_profiles_bind_the_current_foundation_resources():
    from easyicu.research_agent.orchestration.profiles import (
        CURRENT_E1_PLANNER_CANARY_DEV_PROFILE_REF,
        CURRENT_E1_PLANNER_CANARY_LIVE_PUBMED_DEV_PROFILE_REF,
        CURRENT_E1_REVIEWED_DEMO_DEV_PROFILE_REF,
        CURRENT_E1_REVIEWED_DEMO_LIVE_PUBMED_DEV_PROFILE_REF,
        get_submission_profile,
    )

    lock = json.loads(
        (ROOT / "src/easyicu/data/concept-dict.LOCK.json").read_text()
    )
    current_refs = (
        CURRENT_E1_PLANNER_CANARY_DEV_PROFILE_REF,
        CURRENT_E1_REVIEWED_DEMO_DEV_PROFILE_REF,
        CURRENT_E1_PLANNER_CANARY_LIVE_PUBMED_DEV_PROFILE_REF,
        CURRENT_E1_REVIEWED_DEMO_LIVE_PUBMED_DEV_PROFILE_REF,
    )

    for profile_ref in current_refs:
        profile = get_submission_profile(profile_ref)
        assert profile.expected_concept_dict_sha == lock["concept_dict_sha256"]
        assert profile.expected_sofa2_dict_sha == lock["sofa2_dict_sha256"]

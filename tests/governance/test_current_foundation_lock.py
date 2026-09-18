"""The real foundation state must not be masked by synthetic fixtures.

Restored from the v4 publication lock review (bfa903af).
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_current_foundation_lock_matches_resources_and_blocks_unsealed_release():
    path = ROOT / "scripts/releases/EX-A01_seal_full6_release.py"
    spec = importlib.util.spec_from_file_location("current_foundation_sealer", path)
    sealer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sealer)
    lock = json.loads(sealer.FOUNDATION_LOCK_PATH.read_text())
    dictionary = sealer.FOUNDATION_RESOURCE_PATHS["concept_dictionary_sha256"]
    sofa2 = sealer.FOUNDATION_RESOURCE_PATHS["sofa2_dictionary_sha256"]

    assert lock["finalized"] is False
    assert lock["total_concepts"] == len(json.loads(dictionary.read_text()))
    assert lock["concept_dict_sha256"] == hashlib.sha256(
        dictionary.read_bytes()
    ).hexdigest()
    assert lock["sofa2_dict_sha256"] == hashlib.sha256(sofa2.read_bytes()).hexdigest()
    assert lock["pending_foundation_changes"]

    with pytest.raises(sealer.ReleaseValidationError, match="not finalized"):
        sealer._validate_foundation_lock()

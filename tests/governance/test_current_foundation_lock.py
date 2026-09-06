"""The real release foundation must not be masked by synthetic fixtures.

Restored from the v4 publication lock review (bfa903af).
"""
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_current_foundation_lock_matches_actual_release_resources():
    path = ROOT / "scripts/releases/EX-A01_seal_full6_release.py"
    spec = importlib.util.spec_from_file_location("current_foundation_sealer", path)
    sealer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sealer)
    result = sealer._validate_foundation_lock()
    assert result["lock_finalized"] is True
    lock = json.loads(sealer.FOUNDATION_LOCK_PATH.read_text())
    dictionary = sealer.FOUNDATION_RESOURCE_PATHS["concept_dictionary_sha256"]
    assert lock["total_concepts"] == len(json.loads(dictionary.read_text()))
    for field, resource in sealer.FOUNDATION_RESOURCE_PATHS.items():
        assert result[field] == hashlib.sha256(resource.read_bytes()).hexdigest()

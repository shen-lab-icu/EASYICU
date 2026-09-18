"""Keep the dependency/model upgrade canary manifest in sync with its assets.

The manifest is the machine-readable half of
``docs/dependency_model_change_regression_policy.md``.  This test proves every
declared canary/oracle path exists and that the recorded lock identities still
match the repository, so an upgrade cannot silently invalidate the policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "dependency_upgrade_canaries.json"
RUNNER_IMAGE = ROOT / "src" / "easyicu" / "research_agent" / "runner_image"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_declares_every_layer_and_asset_exists() -> None:
    payload = _load()
    assert payload["schema_version"] == "easyicu.dependency-upgrade-canaries/1"
    assert (ROOT / payload["policy_doc"]).is_file()

    layer_1 = payload["layer_1_structure_and_permission"]["canaries"]
    assert layer_1
    for relative in layer_1:
        path = ROOT / relative
        assert path.exists(), f"declared layer-1 canary is missing: {relative}"

    oracles = payload["layer_2_numeric_oracles"]["oracles"]
    assert oracles
    for row in oracles:
        assert (ROOT / row["oracle"]).is_file(), row["oracle"]
        assert row["tests"], row["domain"]
        for test in row["tests"]:
            assert (ROOT / test).is_file(), test


def test_recorded_lock_identities_match_the_repository() -> None:
    payload = _load()
    recorded = payload["verified_against"]
    assert _sha256(RUNNER_IMAGE / "requirements.lock") == recorded[
        "requirements_lock_sha256"
    ]
    assert _sha256(RUNNER_IMAGE / "base-image.lock") == recorded[
        "base_image_lock_sha256"
    ]
    base_lock = (RUNNER_IMAGE / "base-image.lock").read_text(encoding="utf-8")
    assert f"digest={recorded['base_image_digest']}" in base_lock


def test_known_uncovered_methods_stay_registered() -> None:
    payload = _load()
    not_covered = payload["layer_2_numeric_oracles"]["known_not_covered"]
    oracle = json.loads(
        (ROOT / "tests/research_agent/data/method_kernel_oracles.json").read_text(
            encoding="utf-8"
        )
    )
    assert not_covered == oracle["_provenance"]["not_covered"]

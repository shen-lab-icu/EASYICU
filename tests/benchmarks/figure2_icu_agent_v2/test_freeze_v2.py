from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.figure2_icu_agent_v2.freeze_v2 import (
    DesignFreezeError,
    PACKAGE_ROOT,
    PROTOCOL_PATH,
    REPO_ROOT,
    validate_design_freeze,
)


def test_v2_design_freeze_is_valid_but_grants_no_run_authority() -> None:
    receipt = validate_design_freeze()

    assert receipt.heldout_task_count == 27
    assert receipt.safety_task_count == 12
    assert receipt.core_formal_runs == 78
    assert receipt.formal_batch_authorized is False
    assert len(receipt.asset_sha256) == 8


def test_v2_manifest_reproduces_every_frozen_digest() -> None:
    manifest_path = PACKAGE_ROOT / "design_freeze_manifest_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = validate_design_freeze()

    assert manifest["protocol"]["sha256"] == receipt.protocol_sha256
    for relative_path, expected in manifest["frozen_assets"].items():
        observed = hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
        assert observed == expected
    validator_path = REPO_ROOT / manifest["validator"]["path"]
    assert hashlib.sha256(validator_path.read_bytes()).hexdigest() == manifest[
        "validator"
    ]["sha256"]


def test_v2_asset_digest_drift_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["frozen_assets"][0]["sha256"] = "0" * 64
    changed = tmp_path / "experiment_protocol_v2.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DesignFreezeError) as exc_info:
        validate_design_freeze(protocol_path=changed)

    assert exc_info.value.reason_code == "DESIGN_ASSET_DIGEST_MISMATCH"


def test_v2_cannot_grant_formal_authority_by_editing_protocol(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["current_authority"]["formal_batch_authorized"] = True
    changed = tmp_path / "experiment_protocol_v2.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DesignFreezeError) as exc_info:
        validate_design_freeze(protocol_path=changed)

    assert exc_info.value.reason_code == "PREMATURE_FORMAL_AUTHORITY"

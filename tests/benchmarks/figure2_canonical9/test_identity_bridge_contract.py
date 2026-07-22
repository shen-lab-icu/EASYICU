"""Synthetic, no-patient-data tests for the future identity-bridge contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.figure2_canonical9.identity_bridge_contract import (
    IdentityBridgeContractError,
    assess_identity_bridge_contract,
    load_identity_bridge_contract,
)


def _sha(token: str) -> str:
    return token.encode("utf-8").hex().ljust(64, "0")[:64]


def _payload(*, data_lane: str = "not_authorized") -> dict[str, object]:
    source_keys = (
        ("mimic_iv", "stay_id", "subject_id", "attested_icu_stay_to_patient"),
        ("mimic_iii", "icustay_id", "subject_id", "attested_icu_stay_to_patient"),
        ("eicu", "patientunitstayid", "uniquepid", "attested_icu_stay_to_patient"),
        ("amsterdamumcdb", "admissionid", "patientid", "attested_icu_stay_to_patient"),
        ("hirid", "patientid", "patientid", "attested_source_key_semantics"),
        ("sicdb", "CaseID", "CaseID", "attested_source_key_semantics"),
    )
    mappings = []
    for index, (source_id, stay_key, patient_key, semantics) in enumerate(source_keys):
        token = f"{index + 1:x}"
        mappings.append(
            {
                "source_id": source_id,
                "stay_key": stay_key,
                "patient_key": patient_key,
                "mapping_semantics": semantics,
                "source_semantics_attestation_sha256": _sha("a" + token),
                "projection": {
                    "artifact_sha256": _sha("b" + token),
                    "artifact_size_bytes": index + 1,
                    "source_snapshot_sha256": _sha("c" + token),
                    "relation_schema_sha256": _sha("d" + token),
                },
                "mapped_stay_count": index + 1,
                "unmapped_stay_count": 0,
                "duplicate_stay_count": 0,
                "max_stays_per_patient": 1,
            }
        )
    return {
        "schema_version": "easyicu.figure2_identity_bridge/1",
        "bridge_ref": "figure2_canonical9/identity-bridge/20260722-v1",
        "historical_export": {
            "export_label": "full6_20260717",
            "export_manifest_sha256": _sha("e"),
            "export_content_sha256": _sha("f"),
        },
        "data_lane": {
            "status": data_lane,
            "authorization_reference": (
                "owner-ticket-identity-bridge" if data_lane == "authorized" else None
            ),
        },
        "source_mappings": mappings,
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    return path


def test_identity_bridge_contract_is_only_materialization_handoff(
    tmp_path: Path,
) -> None:
    contract, digest = load_identity_bridge_contract(
        _write(tmp_path / "bridge.json", _payload())
    )
    report = assess_identity_bridge_contract(contract, contract_sha256=digest)
    assert report.data_lane_authorized is False
    assert report.eligible_for_native_materialization_review is False
    assert report.real_run_authorized is False
    assert report.blockers[0] == "OWNER_DATA_LANE_AUTHORIZATION_REQUIRED"
    assert "P4_PRODUCTION_INPUT_AUTHORITY_REQUIRED" in report.blockers


def test_even_authorized_bridge_never_becomes_a_real_run_permit(tmp_path: Path) -> None:
    contract, digest = load_identity_bridge_contract(
        _write(tmp_path / "bridge.json", _payload(data_lane="authorized"))
    )
    report = assess_identity_bridge_contract(contract, contract_sha256=digest)
    assert report.eligible_for_native_materialization_review is True
    assert report.real_run_authorized is False
    assert report.blockers == (
        "NATIVE_TYPED_MATERIALIZATION_REQUIRED",
        "P4_PRODUCTION_INPUT_AUTHORITY_REQUIRED",
        "FINAL_OPERATOR_FREEZE_REQUIRED",
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["source_mappings"].__setitem__(
                0, dict(value["source_mappings"][0], patient_key="forged")
            ),
            "source keys",
        ),
        (
            lambda value: value["source_mappings"].__setitem__(
                4,
                dict(
                    value["source_mappings"][4],
                    mapping_semantics="attested_icu_stay_to_patient",
                ),
            ),
            "HiRID and SICdb",
        ),
        (
            lambda value: value["source_mappings"].__setitem__(
                1, dict(value["source_mappings"][1], duplicate_stay_count=1)
            ),
            "duplicate stay",
        ),
        (
            lambda value: value.__setitem__("paper_authorized", True),
            "Extra inputs are not permitted",
        ),
    ],
)
def test_identity_bridge_rejects_semantic_and_authority_forgery(
    tmp_path: Path, mutate, message: str
) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(IdentityBridgeContractError, match=message):
        load_identity_bridge_contract(_write(tmp_path / "bridge.json", payload))


def test_identity_bridge_rejects_duplicate_json_and_symlink(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"a","schema_version":"b"}\n', encoding="utf-8"
    )
    with pytest.raises(IdentityBridgeContractError, match="duplicate key"):
        load_identity_bridge_contract(duplicate)

    target = _write(tmp_path / "target.json", _payload())
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(IdentityBridgeContractError, match="non-symlink"):
        load_identity_bridge_contract(alias)


def test_identity_bridge_rejects_empty_mapping_artifact_and_forged_digest(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["source_mappings"][0]["projection"]["artifact_size_bytes"] = 0
    with pytest.raises(IdentityBridgeContractError, match="greater than 0"):
        load_identity_bridge_contract(_write(tmp_path / "bridge.json", payload))

    contract, digest = load_identity_bridge_contract(
        _write(tmp_path / "valid-bridge.json", _payload())
    )
    with pytest.raises(IdentityBridgeContractError, match="digest is invalid"):
        assess_identity_bridge_contract(contract, contract_sha256=digest.upper())

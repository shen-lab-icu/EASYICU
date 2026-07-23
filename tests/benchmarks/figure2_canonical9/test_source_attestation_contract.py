"""Synthetic tests for the non-authorizing full0717 source-attestation handoff."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.figure2_canonical9.source_attestation_contract import (
    FULL0717_EXPORT_CONTENT_SHA256,
    FULL0717_IDENTITY_BRIDGE_CONTRACT_SHA256,
    FULL0717_RUN_MANIFEST_SHA256,
    SourceAttestationContractError,
    assess_source_attestation_contract,
    load_source_attestation_contract,
)


def _sha(token: str) -> str:
    return token.encode("utf-8").hex().ljust(64, "0")[:64]


def _payload(*, attested: bool = False) -> dict[str, object]:
    sources = (
        "mimic_iv",
        "mimic_iii",
        "eicu",
        "amsterdamumcdb",
        "hirid",
        "sicdb",
    )
    source_attestations = (
        [
            {
                "source_id": source_id,
                "source_snapshot_sha256": _sha(f"snapshot-{index}"),
                "relation_schema_sha256": _sha(f"schema-{index}"),
                "typed_column_inventory_sha256": _sha(f"inventory-{index}"),
                "source_semantics_attestation_sha256": _sha(f"semantics-{index}"),
                "inventory_record_count": index + 1,
            }
            for index, source_id in enumerate(sources)
        ]
        if attested
        else []
    )
    return {
        "schema_version": "easyicu.figure2_source_attestation/1",
        "attestation_ref": "figure2_canonical9/full0717-source-attestation/20260722-v1",
        "historical_export": {
            "export_label": "full6_20260717",
            "export_content_sha256": FULL0717_EXPORT_CONTENT_SHA256,
            "export_run_manifest_sha256": FULL0717_RUN_MANIFEST_SHA256,
        },
        "identity_bridge": {
            "contract_sha256": FULL0717_IDENTITY_BRIDGE_CONTRACT_SHA256,
            "review_handoff_only": True,
            "real_run_authorized": False,
        },
        "review": {
            "status": (
                "attested_for_native_materialization_review" if attested else "pending"
            ),
            "data_owner_reference": "data-owner/20260722" if attested else None,
            "transformation_owner_reference": (
                "transform-owner/20260722" if attested else None
            ),
            "identity_owner_reference": "identity-owner/20260722" if attested else None,
        },
        "source_attestations": source_attestations,
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    return path


def test_pending_source_attestation_stays_fail_closed(tmp_path: Path) -> None:
    contract, digest = load_source_attestation_contract(
        _write(tmp_path / "source-attestation.json", _payload())
    )

    report = assess_source_attestation_contract(contract, contract_sha256=digest)

    assert report.source_review_attested is False
    assert report.eligible_for_native_materialization_review is False
    assert report.real_run_authorized is False
    assert report.blockers == (
        "SOURCE_DATA_IDENTITY_ATTESTATION_REQUIRED",
        "TYPED_COLUMN_INVENTORY_REQUIRED",
        "NATIVE_TYPED_MATERIALIZATION_REQUIRED",
        "P4_PRODUCTION_INPUT_AUTHORITY_REQUIRED",
        "FINAL_OPERATOR_FREEZE_REQUIRED",
    )


def test_attested_source_contract_is_only_materialization_review_handoff(
    tmp_path: Path,
) -> None:
    contract, digest = load_source_attestation_contract(
        _write(tmp_path / "source-attestation.json", _payload(attested=True))
    )

    report = assess_source_attestation_contract(contract, contract_sha256=digest)

    assert report.source_review_attested is True
    assert report.eligible_for_native_materialization_review is True
    assert report.real_run_authorized is False
    assert report.blockers == (
        "NATIVE_TYPED_MATERIALIZATION_REQUIRED",
        "P4_PRODUCTION_INPUT_AUTHORITY_REQUIRED",
        "FINAL_OPERATOR_FREEZE_REQUIRED",
    )


@pytest.mark.parametrize(
    ("attested", "mutate", "message"),
    [
        (
            False,
            lambda value: value["review"].__setitem__(
                "data_owner_reference", "forged-owner"
            ),
            "pending source review",
        ),
        (
            True,
            lambda value: value.__setitem__("paper_authorized", True),
            "Extra inputs are not permitted",
        ),
        (
            True,
            lambda value: value["identity_bridge"].__setitem__(
                "real_run_authorized", True
            ),
            "False",
        ),
        (
            True,
            lambda value: value["historical_export"].__setitem__(
                "export_content_sha256", _sha("wrong-full0717-content")
            ),
            "does not bind full0717",
        ),
        (
            True,
            lambda value: value["identity_bridge"].__setitem__(
                "contract_sha256", _sha("wrong-bridge")
            ),
            "does not bind the approved bridge",
        ),
        (
            True,
            lambda value: value["source_attestations"].pop(),
            "exact six-source order",
        ),
    ],
)
def test_source_attestation_rejects_authority_forgery(
    tmp_path: Path, attested: bool, mutate, message: str
) -> None:
    payload = _payload(attested=attested)
    mutate(payload)

    with pytest.raises(SourceAttestationContractError, match=message):
        load_source_attestation_contract(
            _write(tmp_path / "source-attestation.json", payload)
        )


def test_source_attestation_rejects_duplicate_json_and_symlink(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"a","schema_version":"b"}\n', encoding="utf-8"
    )
    with pytest.raises(SourceAttestationContractError, match="duplicate key"):
        load_source_attestation_contract(duplicate)

    target = _write(tmp_path / "target.json", _payload())
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(SourceAttestationContractError, match="non-symlink"):
        load_source_attestation_contract(alias)


def test_source_attestation_rejects_forged_digest_at_assessment(tmp_path: Path) -> None:
    contract, digest = load_source_attestation_contract(
        _write(tmp_path / "source-attestation.json", _payload())
    )

    with pytest.raises(SourceAttestationContractError, match="digest is invalid"):
        assess_source_attestation_contract(contract, contract_sha256=digest.upper())


def test_source_attestation_contract_is_not_a_p4_import_or_permit() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    realrun_source = (
        repo_root / "benchmarks/figure2_canonical9/realrun_authority.py"
    ).read_text(encoding="utf-8")

    assert "source_attestation_contract" not in realrun_source

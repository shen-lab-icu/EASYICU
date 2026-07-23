"""Synthetic tests for the bounded, non-authorizing full0717 review packet."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from benchmarks.figure2_canonical9 import source_review_packet_builder as builder


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _inventory() -> dict[str, object]:
    source_ids = (
        ("mimic_iv", "miiv"),
        ("mimic_iii", "mimic"),
        ("eicu", "eicu"),
        ("amsterdamumcdb", "aumc"),
        ("hirid", "hirid"),
        ("sicdb", "sic"),
    )
    sources = []
    for index, (source_id, directory) in enumerate(source_ids):
        module = {
            "module": "demographics",
            "state": "schema_observed_not_source_attested",
            "manifest": {
                "relative_path": f"{directory}/demographics.manifest.json",
                "sha256": _sha(f"manifest-{source_id}"),
                "size_bytes": 10,
            },
            "data_file": {
                "relative_path": f"{directory}/demographics.parquet",
                "sha256": _sha(f"data-{source_id}"),
                "size_bytes": 100,
            },
        }
        if index == 0:
            module.update(
                {
                    "manifest_completeness": "partial_or_invalid",
                    "manifest_missing_fields": ["rows", "concepts"],
                    "manifest_invalid_fields": [],
                }
            )
        sources.append(
            {
                "source_id": source_id,
                "export_directory": directory,
                "module_count": 1,
                "modules": [module],
            }
        )
    selected_digest = sources[0]["modules"][0]["data_file"]["sha256"]
    duplicate_digest = _sha("unlisted-duplicate")
    return {
        "schema_version": "easyicu.figure2_schema_inventory/1",
        "inventory_ref": "figure2_canonical9/full0717-schema-inventory/20260722-v1",
        "historical_export": {
            "export_label": "full6_20260717",
            "export_content_sha256": builder.FULL0717_EXPORT_CONTENT_SHA256,
            "export_run_manifest_sha256": builder.FULL0717_RUN_MANIFEST_SHA256,
        },
        "metadata_only": True,
        "source_attested": False,
        "real_run_authorized": False,
        "requires_unlisted_member_disposition": True,
        "sources": sources,
        "unlisted_members": [
            {"relative_path": ".DS_Store", "sha256": _sha("ds"), "size_bytes": 1},
            {
                "relative_path": "eicu/.easyicu_spill/a.parquet",
                "sha256": _sha("spill"),
                "size_bytes": 2,
            },
            {
                "relative_path": "other-copy.parquet",
                "sha256": selected_digest,
                "size_bytes": 3,
            },
            {
                "relative_path": "module-summary-a.csv",
                "sha256": duplicate_digest,
                "size_bytes": 4,
            },
            {
                "relative_path": "module-summary-b.csv",
                "sha256": duplicate_digest,
                "size_bytes": 4,
            },
        ],
    }


def _write_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]
) -> Path:
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    monkeypatch.setattr(
        builder, "FULL0717_SCHEMA_INVENTORY_SHA256", hashlib.sha256(raw).hexdigest()
    )
    path = tmp_path / "schema_inventory.json"
    path.write_bytes(raw)
    return path


def test_source_review_packet_groups_actions_without_granting_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _write_inventory(tmp_path, monkeypatch, _inventory())

    result = builder.build_source_review_packet(
        schema_inventory_path=inventory, output_root=tmp_path / "review-packet"
    )

    payload = json.loads(result.packet_path.read_text(encoding="utf-8"))
    assert result.partial_manifest_count == 1
    assert result.unlisted_member_count == 5
    assert payload["metadata_only"] is True
    assert payload["source_attested"] is False
    assert payload["real_run_authorized"] is False
    assert payload["p4_integration"] == "forbidden_pending_separate_review"
    assert payload["source_review_actions"][0]["partial_or_invalid_manifests"] == [
        {
            "invalid_fields": [],
            "manifest_relative_path": "miiv/demographics.manifest.json",
            "manifest_sha256": _sha("manifest-mimic_iv"),
            "missing_fields": ["rows", "concepts"],
            "module": "demographics",
        }
    ]
    classifications = {
        item["relative_path"]: item["classification"]
        for item in payload["unlisted_member_actions"]
    }
    assert classifications == {
        ".DS_Store": "operating_system_metadata",
        "eicu/.easyicu_spill/a.parquet": "execution_spill_unselected",
        "other-copy.parquet": "exact_duplicate_of_selected_member",
        "module-summary-a.csv": "duplicate_unlisted_member",
        "module-summary-b.csv": "duplicate_unlisted_member",
    }
    assert stat.S_IMODE(result.output_root.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(result.packet_path.stat().st_mode) & 0o077 == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.__setitem__("source_attested", True),
            "non-authorizing boundary",
        ),
        (
            lambda payload: payload.__setitem__(
                "requires_unlisted_member_disposition", False
            ),
            "unlisted-member gate",
        ),
        (
            lambda payload: payload["sources"].pop(),
            "exact six-source order",
        ),
    ],
)
def test_source_review_packet_rejects_forged_or_inconsistent_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate, message: str
) -> None:
    payload = _inventory()
    mutate(payload)
    inventory = _write_inventory(tmp_path, monkeypatch, payload)

    with pytest.raises(builder.SourceReviewPacketBuildError, match=message):
        builder.build_source_review_packet(
            schema_inventory_path=inventory, output_root=tmp_path / "must-not-exist"
        )


def test_source_review_packet_rejects_symlink_and_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _write_inventory(tmp_path, monkeypatch, _inventory())
    alias = tmp_path / "alias.json"
    alias.symlink_to(inventory)
    with pytest.raises(builder.SourceReviewPacketBuildError, match="non-symlink"):
        builder.build_source_review_packet(
            schema_inventory_path=alias, output_root=tmp_path / "must-not-exist"
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(
        builder.SourceReviewPacketBuildError, match="must not already exist"
    ):
        builder.build_source_review_packet(
            schema_inventory_path=inventory, output_root=existing
        )


def test_source_review_packet_is_not_a_p4_import_or_permit() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    realrun_source = (
        repo_root / "benchmarks/figure2_canonical9/realrun_authority.py"
    ).read_text(encoding="utf-8")

    assert "source_review_packet_builder" not in realrun_source

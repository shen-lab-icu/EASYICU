"""Synthetic tests for the host-only Canonical9 identity-bridge builder."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.figure2_canonical9.identity_bridge_builder import (
    _SOURCES,
    IdentityBridgeBuildError,
    build_identity_bridge,
)
from benchmarks.figure2_canonical9.identity_bridge_contract import (
    assess_identity_bridge_contract,
    load_identity_bridge_contract,
)


def _write_relation(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _fixture_roots(tmp_path: Path) -> tuple[Path, Path]:
    export_root = tmp_path / "full6_20260717"
    raw_root = tmp_path / "sources"
    export_root.mkdir()
    (export_root / "run_manifest.json").write_text(
        '{"source":"synthetic-full0717"}\n', encoding="utf-8"
    )
    for index, spec in enumerate(_SOURCES, start=1):
        stay_values = [index * 100 + 1, index * 100 + 2]
        _write_relation(
            export_root / spec.export_directory / "demographics.parquet",
            pd.DataFrame({spec.export_stay_key: stay_values, "age": [60, 61]}),
        )
        if spec.raw_stay_key == spec.raw_patient_key:
            raw_frame = pd.DataFrame({spec.raw_stay_key: stay_values})
        else:
            raw_frame = pd.DataFrame(
                {
                    spec.raw_stay_key: stay_values,
                    spec.raw_patient_key: [f"person-{index}", f"person-{index}"],
                }
            )
        _write_relation(raw_root / spec.raw_relative_path, raw_frame)
    return export_root, raw_root


def test_build_identity_bridge_is_private_review_handoff_only(tmp_path: Path) -> None:
    export_root, raw_root = _fixture_roots(tmp_path)
    result = build_identity_bridge(
        full_export_root=export_root,
        raw_source_root=raw_root,
        output_root=tmp_path / "protected-bridge",
        owner_authorization_reference="owner-delegated-full0717-bridge",
    )

    contract, digest = load_identity_bridge_contract(result.contract_path)
    report = assess_identity_bridge_contract(contract, contract_sha256=digest)
    assert result.contract_sha256 == digest
    assert report.data_lane_authorized is True
    assert report.eligible_for_native_materialization_review is True
    assert report.real_run_authorized is False
    assert set(result.mapping_paths) == {spec.source_id for spec in _SOURCES}
    assert all(path.exists() for path in result.mapping_paths.values())
    assert all(
        stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
        for path in [result.contract_path, *result.mapping_paths.values()]
    )
    descriptor = result.contract_path.read_text(encoding="utf-8")
    assert "person-1" not in descriptor
    assert str(raw_root) not in descriptor
    receipt = json.loads(
        (result.output_root / "build_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["real_run_authorized"] is False


def test_build_identity_bridge_rejects_unmapped_full0717_stay(tmp_path: Path) -> None:
    export_root, raw_root = _fixture_roots(tmp_path)
    spec = _SOURCES[0]
    _write_relation(
        raw_root / spec.raw_relative_path,
        pd.DataFrame({spec.raw_stay_key: [101], spec.raw_patient_key: ["person-1"]}),
    )
    output = tmp_path / "must-not-exist"

    with pytest.raises(IdentityBridgeBuildError, match="without identity mapping"):
        build_identity_bridge(
            full_export_root=export_root,
            raw_source_root=raw_root,
            output_root=output,
            owner_authorization_reference="owner-delegated-full0717-bridge",
        )

    assert not output.exists()


def test_build_identity_bridge_rejects_duplicate_source_stay(tmp_path: Path) -> None:
    export_root, raw_root = _fixture_roots(tmp_path)
    spec = _SOURCES[1]
    _write_relation(
        raw_root / spec.raw_relative_path,
        pd.DataFrame(
            {
                spec.raw_stay_key: [201, 201, 202],
                spec.raw_patient_key: ["p1", "p1", "p1"],
            }
        ),
    )

    with pytest.raises(IdentityBridgeBuildError, match="duplicate stays"):
        build_identity_bridge(
            full_export_root=export_root,
            raw_source_root=raw_root,
            output_root=tmp_path / "must-not-exist",
            owner_authorization_reference="owner-delegated-full0717-bridge",
        )


def test_build_identity_bridge_requires_a_new_empty_output_root(tmp_path: Path) -> None:
    export_root, raw_root = _fixture_roots(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(IdentityBridgeBuildError, match="must not already exist"):
        build_identity_bridge(
            full_export_root=export_root,
            raw_source_root=raw_root,
            output_root=output,
            owner_authorization_reference="owner-delegated-full0717-bridge",
        )

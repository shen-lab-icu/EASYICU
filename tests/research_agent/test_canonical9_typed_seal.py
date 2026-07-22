"""Tracked unit tests for the structural typed retrofit seal (offline, synthetic).

These build tiny synthetic export files using REAL concept names (so the packaged
dictionary resolves them) and assert the B′ contract: parquet bytes are never
touched, extraction_bounds are OMITTED (current-dict ranges are not an authority
over vintage values), every physical column gets a compat verdict (no silent
skips), and the result is a valid TYPED export package.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.figure2_canonical9.typed_export_seal import (
    SEAL_KIND,
    TypedRetrofitSealError,
    seal_export_structural_typed,
)


def _write_synthetic_export(root: Path) -> dict[str, str]:
    """A 3-file untyped export with a bounded concept, an out-of-range value,
    and a boolean-vs-non-logical column. Returns pre-seal parquet SHA256s."""

    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "age": [65.0, 70.0, 55.0],
            "sex": ["Male", "Female", "Male"],
        }
    ).to_parquet(root / "demographics.parquet", index=False)
    pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 3],
            "charttime": [0.5, 6.0, 2.0, 1.0],
            # 999 is far outside the current dict's lact bound [0, 50]
            "lact": [1.2, 999.0, 2.5, 3.1],
        }
    ).to_parquet(root / "blood_gas.parquet", index=False)
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "charttime": [0.0, 0.0, 0.0],
            "death": [1.0, 0.0, 1.0],
            "mort_28d": [True, False, True],  # boolean, non-logical dict concept
        }
    ).to_parquet(root / "outcome.parquet", index=False)
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.glob("*.parquet")
    }


def _load_sidecar(root: Path, result) -> dict:
    return json.loads((root / result.sidecar_file).read_text(encoding="utf-8"))


def test_seal_produces_typed_export_with_provenance(tmp_path: Path) -> None:
    root = tmp_path / "export"
    _write_synthetic_export(root)
    result = seal_export_structural_typed(
        root, database="miiv", value_vintage="20260717"
    )

    assert result.seal_kind == SEAL_KIND
    assert result.bounds_authority == "unavailable"
    assert result.value_vintage == "20260717"
    assert result.metadata_projection_dict["submission_profile"] == "npj_dm/20260718"

    manifest = json.loads((root / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seal_kind"] == SEAL_KIND
    assert manifest["bounds_authority"] == "unavailable"
    assert manifest["value_vintage"] == "20260717"
    assert manifest["metadata_projection_dict"]["concept_dict_sha256"]
    # It is a native-schema manifest referencing the content-addressed sidecar.
    assert manifest["column_metadata"]["file"] == result.sidecar_file


def test_extraction_bounds_are_omitted(tmp_path: Path) -> None:
    root = tmp_path / "export"
    _write_synthetic_export(root)
    result = seal_export_structural_typed(root, value_vintage="20260717")
    sidecar = _load_sidecar(root, result)
    # Every sealed column must carry NO extraction_bounds (current dict ranges are
    # not an authority over vintage values).
    for file_binding in sidecar["files"]:
        for _column, binding in file_binding["columns"].items():
            assert binding["metadata"]["extraction_bounds"] is None


def test_out_of_range_value_is_preserved_with_advisory(tmp_path: Path) -> None:
    root = tmp_path / "export"
    _write_synthetic_export(root)
    result = seal_export_structural_typed(root)
    lact = next(c for c in result.columns if c.column == "lact")
    assert lact.status == "bound"
    advisory = lact.current_dict_bounds_advisory
    assert advisory is not None
    assert advisory["n_total"] == 4
    assert advisory["n_above_current_max"] == 1  # the 999 value, counted not dropped
    # The data itself is untouched: 999 still present in the parquet.
    reloaded = pd.read_parquet(root / "blood_gas.parquet")
    assert 999.0 in set(reloaded["lact"].tolist())


def test_boolean_nonlogical_recorded_as_semantic_conflict(tmp_path: Path) -> None:
    root = tmp_path / "export"
    _write_synthetic_export(root)
    result = seal_export_structural_typed(root)
    mort = next(c for c in result.columns if c.column == "mort_28d")
    assert mort.status == "semantic_conflict"
    assert mort.reason is not None
    # It must NOT be sealed into the sidecar.
    sidecar = _load_sidecar(root, result)
    sealed_columns = {col for fb in sidecar["files"] for col in fb["columns"]}
    assert "mort_28d" not in sealed_columns


def test_patient_identity_unavailable_when_no_subject_id(tmp_path: Path) -> None:
    root = tmp_path / "export"
    _write_synthetic_export(root)
    result = seal_export_structural_typed(root)
    assert result.patient_identity["subject_id_present"] is False
    assert result.patient_identity["blocker"] == "patient_identity_unavailable"
    assert result.patient_identity["patient_level_uniqueness_verified"] is False


def test_parquet_bytes_immutable(tmp_path: Path) -> None:
    root = tmp_path / "export"
    pre = _write_synthetic_export(root)
    result = seal_export_structural_typed(root)
    assert result.parquet_immutability_verified is True
    post = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.glob("*.parquet")
    }
    assert post == pre


def test_refuses_to_overwrite_native_manifest(tmp_path: Path) -> None:
    root = tmp_path / "export"
    _write_synthetic_export(root)
    seal_export_structural_typed(root)
    with pytest.raises(TypedRetrofitSealError):
        seal_export_structural_typed(root)


def test_sealed_export_is_a_typed_package(tmp_path: Path) -> None:
    root = tmp_path / "export"
    _write_synthetic_export(root)
    seal_export_structural_typed(root)
    from easyicu.research_agent.acquisition.catalog import build_available_catalog
    from easyicu.research_agent.intake.export_package import (
        is_export_package,
        open_export_package,
    )

    assert is_export_package(root)
    with open_export_package(root) as pkg:
        assert pkg.column_metadata_sha256 is not None
    catalog = build_available_catalog(str(root))
    typed = {c.concept_id for c in catalog.concepts if c.typed_metadata}
    assert {"age", "sex", "lact", "death"} <= typed
    assert "mort_28d" not in typed

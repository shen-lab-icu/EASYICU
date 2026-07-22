"""Tracked tests for the structural typed retrofit seal (offline, synthetic).

Builds tiny synthetic export files with REAL concept names (so the packaged
dictionary resolves them) and asserts the B′ authority contract: parquet bytes
untouched; numeric value-range claims OMITTED; structural fields honestly
labelled as a current-dictionary projection and paper-gated; runtime dict-SHA
verified against the profile; source evidence bound; a real no-write dry-run;
and fail-close negatives (dict drift, forged vintage, subject_id conflict,
source tamper detectability, atomic no-partial write).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.concept_dict_audit import ConceptDictDriftError
from benchmarks.figure2_canonical9 import typed_export_seal as seal_mod
from benchmarks.figure2_canonical9.typed_export_seal import (
    METADATA_PROVENANCE,
    SEAL_KIND,
    TypedRetrofitSealError,
    assert_sealed_export_paper_ready,
    seal_export_structural_typed,
)


def _write_synthetic_export(
    root: Path, *, with_manifest: bool = True
) -> dict[str, str]:
    """A 3-file untyped export: a bounded concept with an out-of-range value and a
    boolean-vs-non-logical column. Returns pre-seal parquet SHA256s."""

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
            "lact": [1.2, 999.0, 2.5, 3.1],  # 999 outside current lact bound [0,50]
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
    if with_manifest:
        (root / "easyicu_export_manifest.json").write_text(
            json.dumps(
                {"database": "miiv", "entry_mode": "module_grouped_full_export"}
            ),
            encoding="utf-8",
        )
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.glob("*.parquet")
    }


def _seal(root: Path, **kw):
    return seal_export_structural_typed(root, value_vintage="20260717", **kw)


def _load_sidecar(root: Path, result) -> dict:
    return json.loads((root / result.sidecar_file).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Positive contract
# --------------------------------------------------------------------------- #
def test_seal_provenance_and_dict_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    assert result.seal_kind == SEAL_KIND
    assert result.bounds_authority == "unavailable"
    assert result.metadata_provenance == METADATA_PROVENANCE
    assert result.dict_fingerprint["verified_against_profile"] is True
    assert result.dict_fingerprint["submission_profile"] == "npj_dm/20260718"
    manifest = json.loads((root / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata_provenance"] == METADATA_PROVENANCE
    assert manifest["bounds_authority"] == "unavailable"
    assert manifest["column_metadata"]["file"] == result.sidecar_file


def test_both_value_range_claims_omitted(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    sidecar = _load_sidecar(root, result)
    for fb in sidecar["files"]:
        for binding in fb["columns"].values():
            assert binding["metadata"]["extraction_bounds"] is None
            assert binding["metadata"]["analysis_plausibility_range"] is None


def test_out_of_range_value_preserved_with_advisory(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    lact = next(c for c in result.columns if c.column == "lact")
    assert lact.status == "bound"
    assert lact.current_dict_bounds_advisory["n_above_current_max"] == 1
    assert lact.current_dict_bounds_advisory["n_total"] == 4
    assert 999.0 in set(pd.read_parquet(root / "blood_gas.parquet")["lact"].tolist())


def test_boolean_nonlogical_is_semantic_conflict(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    mort = next(c for c in result.columns if c.column == "mort_28d")
    assert mort.status == "semantic_conflict" and mort.reason
    sidecar = _load_sidecar(root, result)
    sealed = {col for fb in sidecar["files"] for col in fb["columns"]}
    assert "mort_28d" not in sealed


def test_semantic_review_is_paper_gated(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    review = result.semantic_review
    assert review["paper_authorized"] is False and review["reviewed"] is False
    concepts = {row["concept"] for row in review["review_table"]}
    assert {"age", "sex", "death"} <= concepts
    for row in review["review_table"]:
        assert row["provenance"] == METADATA_PROVENANCE
        assert row["reviewed"] is False


def test_source_evidence_bound(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    ev = result.source_evidence
    assert (
        ev["extraction_run_manifest_sha256"]
        == hashlib.sha256(
            (root / "easyicu_export_manifest.json").read_bytes()
        ).hexdigest()
    )
    assert len(ev["sealer_code_sha256"]) == 64
    assert ev["per_module_manifest_sha256"] == {}  # synthetic per-module absent
    git = ev["sealer_git"]
    assert set(git) == {"head", "sealer_paths_dirty", "head_describes_running_code"}
    # HEAD may describe the running code ONLY when we have a HEAD and the sealer's
    # own files are clean at it; a dirty/untracked/unknown tree must never claim so.
    assert git["head_describes_running_code"] is (
        bool(git["head"]) and git["sealer_paths_dirty"] is False
    )


def test_patient_identity_unavailable_when_no_subject_id(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    assert result.patient_identity["subject_id_present"] is False
    assert result.patient_identity["blocker"] == "patient_identity_unavailable"


def test_parquet_bytes_immutable(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    pre = _write_synthetic_export(root)
    result = _seal(root)
    assert result.parquet_immutability_verified is True
    post = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.glob("*.parquet")
    }
    assert post == pre


def test_sealed_export_is_a_typed_package(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    _seal(root)
    from easyicu.research_agent.acquisition.catalog import build_available_catalog
    from easyicu.research_agent.intake.export_package import (
        is_export_package,
        open_export_package,
    )

    assert is_export_package(root)
    with open_export_package(root) as pkg:
        assert pkg.column_metadata_sha256 is not None
    typed = {
        c.concept_id
        for c in build_available_catalog(str(root)).concepts
        if c.typed_metadata
    }
    assert {"age", "sex", "lact", "death"} <= typed and "mort_28d" not in typed


# --------------------------------------------------------------------------- #
# Producer -> consumer paper-readiness gate (fail-closed)
# --------------------------------------------------------------------------- #
def _sign_manifest(root: Path, **changes) -> dict:
    """Load, mutate, and rewrite the retrofit manifest; return the new payload."""

    path = root / "_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_paper_ready_is_false_for_full6_shape(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    assert result.paper_ready is False
    manifest = json.loads((root / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["paper_ready"] is False


def test_consumer_gate_fails_closed_when_unreviewed(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    _seal(root)  # default: paper_authorized False, patient_identity_unavailable
    with pytest.raises(TypedRetrofitSealError, match="NOT paper-authorized"):
        assert_sealed_export_paper_ready(root)


def test_consumer_gate_fails_closed_on_insufficient_identity(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    # Sign the review, but identity stays stay-level only -> still not paper-ready.
    review = dict(result.semantic_review, paper_authorized=True, reviewed=True)
    _sign_manifest(root, semantic_review=review, paper_ready=False)
    with pytest.raises(TypedRetrofitSealError, match="patient identity insufficient"):
        assert_sealed_export_paper_ready(root)


def test_consumer_gate_passes_when_signed_and_identity_sufficient(
    tmp_path: Path,
) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    review = dict(result.semantic_review, paper_authorized=True, reviewed=True)
    identity = {
        "subject_id_present": True,
        "row_identity": "stay_id",
        "blocker": None,
    }
    manifest = _sign_manifest(
        root, semantic_review=review, patient_identity=identity, paper_ready=True
    )
    got = assert_sealed_export_paper_ready(root)
    assert got == manifest and got["paper_ready"] is True


def test_consumer_gate_detects_paper_ready_forgery(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    _seal(root)
    # Flip only the summary flag; review still unsigned + identity still blocked.
    _sign_manifest(root, paper_ready=True)
    with pytest.raises(TypedRetrofitSealError, match="disagrees with its own"):
        assert_sealed_export_paper_ready(root)


def test_consumer_gate_passthrough_for_non_retrofit_manifest(tmp_path: Path) -> None:
    root = tmp_path / "native_export"
    root.mkdir()
    payload = {"schema_version": "easyicu_native_export_v2", "files": []}
    (root / "_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    # No seal_kind -> official typed-authority path governs; gate returns as-is.
    assert assert_sealed_export_paper_ready(root) == payload


def test_consumer_gate_requires_a_sealed_manifest(tmp_path: Path) -> None:
    root = tmp_path / "unsealed"
    root.mkdir()
    with pytest.raises(TypedRetrofitSealError, match="not sealed"):
        assert_sealed_export_paper_ready(root)


# --------------------------------------------------------------------------- #
# (#2) Real no-write dry-run
# --------------------------------------------------------------------------- #
def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root, dry_run=True)
    assert result.dry_run is True
    assert result.sidecar_file is None and result.manifest_path is None
    assert not (root / "_manifest.json").exists()
    assert not list(root.glob("column_metadata.sha256-*.json"))
    # Full report still produced (would-be sidecar ref + compat).
    assert result.sidecar_ref["sha256"] and result.columns
    # A subsequent real seal still works (dry-run left no trace).
    real = _seal(root)
    assert real.sidecar_ref["sha256"] == result.sidecar_ref["sha256"]


# --------------------------------------------------------------------------- #
# (#9) Fail-close negatives
# --------------------------------------------------------------------------- #
class _WrongProfile:
    name = "npj_dm"
    version = "20260718"
    expected_concept_dict_sha = "0" * 64
    expected_sofa2_dict_sha = "0" * 64


def test_dict_sha_drift_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    with pytest.raises(ConceptDictDriftError):
        seal_export_structural_typed(
            root, value_vintage="20260717", submission_profile=_WrongProfile()
        )
    assert not (root / "_manifest.json").exists()


def test_forged_vintage_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "full6_20260717"
    _write_synthetic_export(root)
    with pytest.raises(TypedRetrofitSealError, match="forged vintage"):
        seal_export_structural_typed(root, value_vintage="20260718")
    assert not (root / "_manifest.json").exists()


def test_vintage_basis_records_path_token(tmp_path: Path) -> None:
    root = tmp_path / "full6_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    assert result.value_vintage_basis == "export_path_date_token:20260717"


def test_cross_file_subject_id_conflict_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    # Two files disagree on stay_id -> subject_id.
    pd.DataFrame({"stay_id": [1], "subject_id": [10], "age": [65.0]}).to_parquet(
        root / "a.parquet", index=False
    )
    pd.DataFrame(
        {"stay_id": [1], "subject_id": [11], "charttime": [0.0], "lact": [2.0]}
    ).to_parquet(root / "b.parquet", index=False)
    with pytest.raises(TypedRetrofitSealError, match="subject_id cross-file conflict"):
        _seal(root)


def test_source_manifest_tamper_is_detectable(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    recorded = result.source_evidence["extraction_run_manifest_sha256"]
    # Tamper the source manifest after sealing: the recorded SHA no longer matches.
    (root / "easyicu_export_manifest.json").write_text("{}", encoding="utf-8")
    now = hashlib.sha256(
        (root / "easyicu_export_manifest.json").read_bytes()
    ).hexdigest()
    assert now != recorded  # a downstream verifier detects the drift


def test_atomic_manifest_write_leaves_no_partial(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)

    def _boom(*_a, **_k):
        raise OSError("simulated mid-write failure")

    monkeypatch.setattr(seal_mod.os, "replace", _boom)
    with pytest.raises(OSError):
        _seal(root)
    # No partial _manifest.json and no leftover temp files.
    assert not (root / "_manifest.json").exists()
    assert not list(root.glob(".*_manifest.json*.tmp"))

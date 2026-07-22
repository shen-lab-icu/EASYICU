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
    RETROFIT_DECISION_FILE,
    SEAL_KIND,
    TypedRetrofitSealError,
    assert_sealed_export_paper_ready,
    build_retrofit_review_attestation,
    build_retrofit_review_request,
    seal_export_structural_typed,
    verify_retrofit_review_attestation,
    write_retrofit_review_decision,
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


def test_patient_level_uniqueness_not_overclaimed_for_multi_stay(
    tmp_path: Path,
) -> None:
    root = tmp_path / "export_20260717"
    root.mkdir(parents=True, exist_ok=True)
    # subject 10 has TWO stays (1, 2) -> subjects (2) != stays (3): NOT unique.
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "subject_id": [10, 10, 30],
            "age": [65.0, 66.0, 55.0],
            "sex": ["Male", "Male", "Female"],
        }
    ).to_parquet(root / "demographics.parquet", index=False)
    (root / "easyicu_export_manifest.json").write_text(
        json.dumps({"database": "miiv"}), encoding="utf-8"
    )
    pid = _seal(root).patient_identity
    assert pid["subject_id_present"] is True
    assert pid["n_stays_with_subject"] == 3 and pid["n_subjects"] == 2
    assert pid["n_multi_stay_patients"] == 1 and pid["max_stays_per_subject"] == 2
    assert pid["multi_stay_patients_present"] is True
    assert pid["patient_level_uniqueness_verified"] is False
    assert pid["first_icu_stay_verified"] is False


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
# Receipt-backed paper-readiness gate (fail-closed; no trusted manifest flags)
# --------------------------------------------------------------------------- #
def _decision_for(
    root: Path, *, reviewer: str = "dr. reviewer", decided_at: str = "2026-07-22"
) -> dict:
    """A Framework v2 HumanReviewDecision bound to the derived review request."""

    request, _authority = build_retrofit_review_request(root)
    return {
        "review_id": request.review_id,
        "authority_sha256": request.authority_sha256,
        "decision": "approved",
        "reviewer": reviewer,
        "decided_at": decided_at,
    }


def _review(root: Path) -> Path:
    """The write-once HITL sign-off (real HumanReviewDecision, not free text)."""

    return write_retrofit_review_decision(root, decision=_decision_for(root))


def _write_export_with_subject_id(root: Path) -> None:
    """Write a patient-level-unique export (each subject maps to one stay)."""

    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "subject_id": [10, 20, 30],
            "age": [65.0, 70.0, 55.0],
            "sex": ["Male", "Female", "Male"],
        }
    ).to_parquet(root / "demographics.parquet", index=False)
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "subject_id": [10, 20, 30],
            "charttime": [0.5, 2.0, 1.0],
            "lact": [1.2, 2.5, 3.1],
        }
    ).to_parquet(root / "blood_gas.parquet", index=False)
    (root / "easyicu_export_manifest.json").write_text(
        json.dumps({"database": "miiv"}), encoding="utf-8"
    )


def _write_paper_ready_export(root: Path) -> None:
    """Seal a subject-id export, then record a write-once Framework v2 HITL
    decision — the real sign-off, not a hand-edited flag."""

    _write_export_with_subject_id(root)
    seal_export_structural_typed(root, value_vintage="20260717")
    _review(root)


def test_paper_ready_is_false_for_full6_shape(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    result = _seal(root)
    assert result.paper_ready is False
    assert json.loads((root / "_manifest.json").read_text())["paper_ready"] is False


def test_review_sign_off_refuses_identity_insufficient(tmp_path: Path) -> None:
    # full6 shape (no subject_id): the HITL sign-off itself refuses to approve.
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)
    _seal(root)
    with pytest.raises(TypedRetrofitSealError, match="patient identity insufficient"):
        _review(root)
    assert not (root / RETROFIT_DECISION_FILE).exists()


def test_gate_fails_closed_without_a_decision_receipt(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_paper_ready_export(root)
    (root / RETROFIT_DECISION_FILE).unlink()
    with pytest.raises(
        TypedRetrofitSealError, match="no write-once HITL review decision"
    ):
        assert_sealed_export_paper_ready(root)


def test_gate_fails_closed_on_insufficient_identity(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)  # no subject_id
    _seal(root)
    with pytest.raises(
        TypedRetrofitSealError, match="patient identity is insufficient"
    ):
        assert_sealed_export_paper_ready(root)


def test_gate_passes_with_valid_decision_and_identity(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_paper_ready_export(root)
    manifest = assert_sealed_export_paper_ready(root)
    assert manifest["seal_kind"] == SEAL_KIND


def test_gate_ignores_hand_edited_manifest_paper_ready(tmp_path: Path) -> None:
    # subject_id present so identity passes; but NO decision receipt -> a
    # hand-flipped manifest paper_ready/paper_authorized cannot make it ready.
    root = tmp_path / "export_20260717"
    _write_paper_ready_export(root)
    (root / RETROFIT_DECISION_FILE).unlink()
    m_path = root / "_manifest.json"
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    manifest["paper_ready"] = True
    manifest["semantic_review"]["paper_authorized"] = True
    m_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        TypedRetrofitSealError, match="no write-once HITL review decision"
    ):
        assert_sealed_export_paper_ready(root)


def test_write_once_decision_rejects_reapproval(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_paper_ready_export(root)
    second = _decision_for(root, reviewer="dr. other", decided_at="2026-07-23")
    with pytest.raises(TypedRetrofitSealError, match="different bytes"):
        write_retrofit_review_decision(root, decision=second)


def test_write_rejected_decision_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_export_with_subject_id(root)
    seal_export_structural_typed(root, value_vintage="20260717")
    rejected = dict(_decision_for(root), decision="rejected")
    with pytest.raises(TypedRetrofitSealError, match="not approved"):
        write_retrofit_review_decision(root, decision=rejected)
    assert not (root / RETROFIT_DECISION_FILE).exists()


def test_write_forged_review_id_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_export_with_subject_id(root)
    seal_export_structural_typed(root, value_vintage="20260717")
    forged = {
        "review_id": "review-" + "0" * 16,  # not derived from this authority
        "authority_sha256": "a" * 64,
        "decision": "approved",
        "reviewer": "attacker",
        "decided_at": "2026-07-22",
    }
    with pytest.raises(TypedRetrofitSealError, match="does not bind|authority_sha256"):
        write_retrofit_review_decision(root, decision=forged)


def test_gate_detects_decision_tamper(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_paper_ready_export(root)
    d_path = root / RETROFIT_DECISION_FILE
    receipt = json.loads(d_path.read_text(encoding="utf-8"))
    # Change the embedded decision's reviewer without recomputing decision_sha256.
    receipt["review_decision"]["reviewer"] = "attacker"
    d_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(TypedRetrofitSealError, match="canonical sha mismatch"):
        assert_sealed_export_paper_ready(root)


def test_gate_detects_manifest_tamper_after_review(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_paper_ready_export(root)
    m_path = root / "_manifest.json"
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    manifest["benign_marker"] = "tampered"
    m_path.write_text(json.dumps(manifest), encoding="utf-8")
    # Tampering changes the reviewed-authority digest -> the derived review_id no
    # longer binds; caught before the explicit per-digest check.
    with pytest.raises(
        TypedRetrofitSealError,
        match="does not bind this source|source_manifest_sha256 mismatch",
    ):
        assert_sealed_export_paper_ready(root)


def test_consumer_gate_passthrough_for_non_retrofit_manifest(tmp_path: Path) -> None:
    root = tmp_path / "native_export"
    root.mkdir()
    payload = {"schema_version": "easyicu_native_export_v2", "files": []}
    (root / "_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert assert_sealed_export_paper_ready(root) == payload


def test_consumer_gate_requires_a_sealed_manifest(tmp_path: Path) -> None:
    root = tmp_path / "unsealed"
    root.mkdir()
    with pytest.raises(TypedRetrofitSealError, match="not sealed"):
        assert_sealed_export_paper_ready(root)


# --------------------------------------------------------------------------- #
# Review attestation: mint from the decision receipt + re-verify (offline + live)
# --------------------------------------------------------------------------- #
def test_build_attestation_fails_closed_for_full6_shape(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_synthetic_export(root)  # no subject_id + no decision receipt
    _seal(root)
    with pytest.raises(TypedRetrofitSealError):
        build_retrofit_review_attestation(root)


def test_build_and_verify_attestation_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_paper_ready_export(root)
    att = build_retrofit_review_attestation(root)
    assert att["paper_ready"] is True and att["seal_kind"] == SEAL_KIND
    assert len(att["decision_sha256"]) == 64
    assert len(att["patient_identity_authority_sha256"]) == 64
    verify_retrofit_review_attestation(att)  # offline
    verify_retrofit_review_attestation(att, export_dir=root)  # live reconcile


def test_verify_attestation_rejects_forged_unreviewed(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_paper_ready_export(root)
    att = build_retrofit_review_attestation(root)
    with pytest.raises(TypedRetrofitSealError, match="not paper_ready"):
        verify_retrofit_review_attestation(dict(att, paper_ready=False))


def test_verify_attestation_detects_manifest_tamper(tmp_path: Path) -> None:
    root = tmp_path / "export_20260717"
    _write_paper_ready_export(root)
    att = build_retrofit_review_attestation(root)
    m_path = root / "_manifest.json"
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    manifest["benign_marker"] = "tampered"
    m_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TypedRetrofitSealError):
        verify_retrofit_review_attestation(att, export_dir=root)


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

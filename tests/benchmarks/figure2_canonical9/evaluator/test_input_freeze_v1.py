"""Fail-closed contracts for the archived Figure 2 input assessment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from benchmarks.figure2_canonical9.evaluator import input_freeze_v1 as freeze

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = (
    REPO_ROOT / "benchmarks" / "figure2_canonical9" / "canonical_input_freeze_v1.json"
)


def _payload() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_manifest(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "manifest.json"
    path.write_bytes(freeze._canonical_json_bytes(payload) + b"\n")
    return path


def test_tracked_manifest_is_portable_exact_and_truthfully_blocked() -> None:
    manifest = freeze.load_canonical_input_freeze_manifest(MANIFEST_PATH)

    assert freeze.canonical_input_freeze_manifest_sha256(MANIFEST_PATH) == (
        "aa790fca1da56ca70dcd45b8f0bf4b227ecfebfee27f1ca884ce52231b738331"
    )
    assert isinstance(manifest.cases, tuple)
    assert [case.case_id for case in manifest.cases] == ["e2", "e3", "h2"]
    assert [case.state for case in manifest.cases] == ["blocked"] * 3
    assert all(isinstance(case.files, tuple) for case in manifest.cases)
    assert all(isinstance(case.blockers, tuple) for case in manifest.cases)

    raw = MANIFEST_PATH.read_text(encoding="utf-8")
    assert "/Volumes/" not in raw
    assert "expected_or_direction" not in raw
    assert "gold_answer" not in raw
    assert "forbidden_claims" not in raw


def test_tracked_manifest_records_real_missing_and_stale_authority() -> None:
    manifest = freeze.load_canonical_input_freeze_manifest(MANIFEST_PATH)
    cases = {case.case_id: case for case in manifest.cases}

    e2_codes = {item.code for item in cases["e2"].blockers}
    assert {"MISSING_BUILD_PROVENANCE", "MISSING_SELECTION_REPORT"} <= e2_codes

    e3 = cases["e3"]
    assert e3.provenance.recorded_cohort_semantic_digest_present is True
    assert e3.provenance.recorded_cohort_semantic_digest_reverified is False
    assert "RECORDED_COHORT_SEMANTIC_DIGEST_UNVERIFIED" in {
        item.code for item in e3.blockers
    }

    h2 = cases["h2"]
    h2_codes = {item.code for item in h2.blockers}
    assert h2.provenance.physical_schema_matches_provenance is False
    assert "PHYSICAL_SCHEMA_PROVENANCE_MISMATCH" in h2_codes
    assert h2.provenance.recorded_trajectory_semantic_digest_present is True
    assert h2.provenance.recorded_trajectory_semantic_digest_reverified is False
    assert "RECORDED_TRAJECTORY_SEMANTIC_DIGEST_UNVERIFIED" in h2_codes
    assert {
        "OWNER_DATABASE_REQUIRED",
        "OWNER_OPERATIONAL_EXPOSURE_REQUIRED",
    } <= h2_codes


def test_manifest_authority_is_deeply_immutable() -> None:
    manifest = freeze.load_canonical_input_freeze_manifest(MANIFEST_PATH)

    with pytest.raises(ValidationError, match="frozen"):
        manifest.manifest_ref = "figure2_canonical9/input_freeze/20990101"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        manifest.cases.append(manifest.cases[0])  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        manifest.cases[0].files.append(manifest.cases[0].files[0])  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "mutation,code",
    [
        (
            lambda payload: payload["cases"][0]["files"][0].__setitem__(
                "relative_path", "/tmp/cohort.parquet"
            ),
            "absolute_path_in_public_manifest",
        ),
        (
            lambda payload: payload.__setitem__("expected_or_direction", 1),
            "evaluator_oracle_in_public_manifest",
        ),
    ],
)
def test_public_manifest_rejects_absolute_paths_and_oracles(
    tmp_path: Path, mutation, code: str
) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(freeze.CanonicalInputFreezeError) as exc_info:
        freeze.load_canonical_input_freeze_manifest(_write_manifest(tmp_path, payload))
    assert exc_info.value.code == code


def test_manifest_rejects_duplicate_keys_and_noncanonical_bytes(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"x","schema_version":"y"}\n')
    with pytest.raises(freeze.CanonicalInputFreezeError) as exc_info:
        freeze.load_canonical_input_freeze_manifest(duplicate)
    assert exc_info.value.code == "duplicate_json_key"

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(_payload(), indent=2), encoding="utf-8")
    with pytest.raises(freeze.CanonicalInputFreezeError) as exc_info:
        freeze.load_canonical_input_freeze_manifest(pretty)
    assert exc_info.value.code == "manifest_not_canonical"


def test_archive_v1_cannot_self_attest_runnable_or_verified_typed_authority(
    tmp_path: Path,
) -> None:
    payload = _payload()
    e2 = payload["cases"][0]
    e2["state"] = "runnable"
    e2["blockers"] = []
    e2["provenance"]["typed_cohort_authority_present"] = True
    e2["provenance"]["recorded_cohort_semantic_digest_reverified"] = True

    with pytest.raises(freeze.CanonicalInputFreezeError) as exc_info:
        freeze.load_canonical_input_freeze_manifest(_write_manifest(tmp_path, payload))
    assert exc_info.value.code == "manifest_schema_invalid"


def test_role_presence_flags_cannot_be_forged(tmp_path: Path) -> None:
    payload = _payload()
    e3 = payload["cases"][1]
    e3["files"] = [
        item
        for item in e3["files"]
        if item["role"] not in {"build_provenance", "selection_report"}
    ]

    with pytest.raises(
        freeze.CanonicalInputFreezeError, match="does not match frozen files"
    ):
        freeze.load_canonical_input_freeze_manifest(_write_manifest(tmp_path, payload))


def test_row_count_provenance_mismatch_requires_typed_blocker(tmp_path: Path) -> None:
    payload = _payload()
    e3 = payload["cases"][1]
    e3["provenance"]["physical_rows_match_provenance"] = False

    with pytest.raises(freeze.CanonicalInputFreezeError) as exc_info:
        freeze.load_canonical_input_freeze_manifest(_write_manifest(tmp_path, payload))
    assert "PHYSICAL_ROW_COUNT_PROVENANCE_MISMATCH" in str(exc_info.value)


def test_trajectory_without_typed_authority_requires_blocker(tmp_path: Path) -> None:
    payload = _payload()
    h2 = payload["cases"][2]
    h2["blockers"] = [
        item
        for item in h2["blockers"]
        if item["code"] != "TYPED_TRAJECTORY_AUTHORITY_MISSING"
    ]

    with pytest.raises(freeze.CanonicalInputFreezeError) as exc_info:
        freeze.load_canonical_input_freeze_manifest(_write_manifest(tmp_path, payload))
    assert "TYPED_TRAJECTORY_AUTHORITY_MISSING" in str(exc_info.value)


def test_owner_science_blockers_are_authorized_only_for_h2(tmp_path: Path) -> None:
    payload = _payload()
    e2 = payload["cases"][0]
    e2["blockers"].append(
        {"code": "OWNER_DATABASE_REQUIRED", "resolution": "benchmark_owner"}
    )
    e2["blockers"] = sorted(e2["blockers"], key=lambda item: item["code"])

    with pytest.raises(freeze.CanonicalInputFreezeError, match="only for h2"):
        freeze.load_canonical_input_freeze_manifest(_write_manifest(tmp_path, payload))


def test_physical_verifier_checks_digest_rows_and_schema(tmp_path: Path) -> None:
    parquet_path = tmp_path / "cohort.parquet"
    pq.write_table(pa.table({"stay_id": [1, 2], "value": [3.0, 4.0]}), parquet_path)
    raw = parquet_path.read_bytes()
    parquet = pq.ParquetFile(parquet_path)
    frozen = freeze.FrozenInputFile(
        role="cohort",
        relative_path=parquet_path.name,
        format="parquet",
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        row_count=2,
        column_count=2,
        schema_sha256=freeze._schema_sha256(freeze._parquet_shape(parquet)),
    )

    verified = freeze._verify_member(tmp_path, frozen, case_id="e2")
    assert verified.path == parquet_path

    parquet_path.write_bytes(raw + b"tamper")
    with pytest.raises(freeze.CanonicalInputFreezeError) as exc_info:
        freeze._verify_member(tmp_path, frozen, case_id="e2")
    assert exc_info.value.code == "frozen_input_digest_mismatch"


def test_case_selection_must_follow_manifest_order() -> None:
    with pytest.raises(freeze.CanonicalInputFreezeError) as exc_info:
        freeze.verify_local_input_freeze(
            MANIFEST_PATH,
            local_roots={"e3": Path("/missing"), "e2": Path("/missing")},
            case_ids=["e3", "e2"],
        )
    assert exc_info.value.code == "invalid_case_selection"


def test_archive_assessment_exposes_no_agent_handoff_generator() -> None:
    assert not hasattr(freeze, "generate_agent_handoff_jsonl")
    assert not hasattr(freeze, "LocalHandoffReceipt")

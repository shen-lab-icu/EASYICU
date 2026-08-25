from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from easyicu.webserver.research_evidence_preview import (
    EvidencePreviewError,
    build_evidence_preview,
)


def _register(
    run_dir: Path,
    *,
    evidence_id: str,
    kind: str,
    name: str,
    content: bytes,
    produced_by_step: str = "analysis_step",
    script_evidence_id: str | None = None,
) -> tuple[Path, str]:
    evidence = run_dir / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    path = evidence / name
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    index_path = evidence / "evidence_index.json"
    records = json.loads(index_path.read_text()) if index_path.exists() else []
    records.append(
        {
            "evidence_id": evidence_id,
            "kind": kind,
            "description": "registered test evidence",
            "relative_path": f"evidence/{name}",
            "sha256": digest,
            "produced_by_step": produced_by_step,
            "script_evidence_id": script_evidence_id,
        }
    )
    index_path.write_text(json.dumps(records), encoding="utf-8")
    return path, digest


def test_registered_code_preview_is_digest_pinned_and_path_free(tmp_path: Path) -> None:
    _path, digest = _register(
        tmp_path,
        evidence_id="code_analysis_1",
        kind="code",
        name="code_analysis_1__analysis.py",
        content=b"value = 42\nprint(value)\n",
    )

    preview = build_evidence_preview(tmp_path, "code_analysis_1", digest)

    assert preview["renderer"] == "code"
    assert preview["language"] == "python"
    assert preview["line_count"] == 2
    assert preview["text"] == "value = 42\nprint(value)\n"
    assert "relative_path" not in preview
    assert str(tmp_path) not in json.dumps(preview)


def test_statistic_json_preview_is_structured(tmp_path: Path) -> None:
    _path, digest = _register(
        tmp_path,
        evidence_id="statistic_summary_1",
        kind="statistic",
        name="statistic_summary_1__summary.json",
        content=b'{"estimate": 1.25, "ci": [1.1, 1.4]}',
    )

    preview = build_evidence_preview(tmp_path, "statistic_summary_1", digest)

    assert preview["renderer"] == "json"
    assert preview["value"]["estimate"] == 1.25


def test_result_csv_is_bounded_and_identifier_csv_is_withheld(tmp_path: Path) -> None:
    _path, digest = _register(
        tmp_path,
        evidence_id="table_result_1",
        kind="table",
        name="table_result_1__result.csv",
        content=b"contrast,estimate\n5 vs 2.1,1.89\n",
        script_evidence_id="code_analysis_1",
    )
    preview = build_evidence_preview(tmp_path, "table_result_1", digest)
    assert preview["renderer"] == "table"
    assert preview["headers"] == ["contrast", "estimate"]
    assert preview["rows"] == [["5 vs 2.1", "1.89"]]

    _path, sensitive_digest = _register(
        tmp_path,
        evidence_id="table_sensitive_1",
        kind="table",
        name="table_sensitive_1__result.csv",
        content=b"estimate,subject_id\n1.2,123\n",
        script_evidence_id="code_analysis_1",
    )
    sensitive = build_evidence_preview(tmp_path, "table_sensitive_1", sensitive_digest)
    assert sensitive["renderer"] == "metadata"
    assert sensitive["withheld_reason"] == "direct_identifier_columns_withheld"
    assert "rows" not in sensitive


def test_raw_cohort_is_metadata_only(tmp_path: Path) -> None:
    _path, digest = _register(
        tmp_path,
        evidence_id="analysis_cohort_1",
        kind="table",
        name="analysis_cohort_1__cohort.parquet",
        content=b"PAR1 patient-level placeholder",
        produced_by_step="cohort_definition",
    )

    preview = build_evidence_preview(tmp_path, "analysis_cohort_1", digest)

    assert preview["renderer"] == "metadata"
    assert preview["withheld_reason"] == "patient_level_rows_withheld"
    assert "text" not in preview and "rows" not in preview


def test_expected_and_actual_digests_fail_closed(tmp_path: Path) -> None:
    path, digest = _register(
        tmp_path,
        evidence_id="code_analysis_1",
        kind="code",
        name="code_analysis_1__analysis.py",
        content=b"value = 42\n",
    )
    with pytest.raises(EvidencePreviewError) as wrong:
        build_evidence_preview(tmp_path, "code_analysis_1", "0" * 64)
    assert wrong.value.code == "evidence_preview_sha_mismatch"

    path.write_text("value = 43\n", encoding="utf-8")
    with pytest.raises(EvidencePreviewError) as tampered:
        build_evidence_preview(tmp_path, "code_analysis_1", digest)
    assert tampered.value.code == "evidence_preview_digest_mismatch"


def test_path_escape_and_host_path_text_are_withheld(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    outside = tmp_path.parent / "outside-evidence.py"
    outside.write_text("safe = True\n", encoding="utf-8")
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    (evidence / "evidence_index.json").write_text(
        json.dumps(
            [
                {
                    "evidence_id": "code_escape_1",
                    "kind": "code",
                    "relative_path": "../outside-evidence.py",
                    "sha256": digest,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvidencePreviewError) as escaped:
        build_evidence_preview(tmp_path, "code_escape_1", digest)
    assert escaped.value.code == "evidence_preview_path_invalid"

    _path, host_digest = _register(
        tmp_path,
        evidence_id="code_host_path_1",
        kind="code",
        name="code_host_path_1__analysis.py",
        content=b"source = '/Users/example/private.csv'\n",
    )
    withheld = build_evidence_preview(tmp_path, "code_host_path_1", host_digest)
    assert withheld["renderer"] == "metadata"
    assert withheld["withheld_reason"] == "evidence_preview_host_path_detected"


def test_registered_symlink_is_rejected_even_when_target_is_inside_run(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    target = evidence / "target.py"
    target.write_text("value = 1\n", encoding="utf-8")
    link = evidence / "linked.py"
    link.symlink_to(target.name)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (evidence / "evidence_index.json").write_text(
        json.dumps(
            [
                {
                    "evidence_id": "code_link_1",
                    "kind": "code",
                    "relative_path": "evidence/linked.py",
                    "sha256": digest,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvidencePreviewError) as blocked:
        build_evidence_preview(tmp_path, "code_link_1", digest)
    assert blocked.value.code == "evidence_preview_symlink_forbidden"

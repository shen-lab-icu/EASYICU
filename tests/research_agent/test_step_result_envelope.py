from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent.execution.result_envelope import (
    StepResultEnvelope,
    normalize_step_result_shadow,
    verify_step_result_envelope,
    write_shadow_step_result_envelope,
)


def _issue_codes(envelope: StepResultEnvelope) -> set[str]:
    return {issue.code for issue in envelope.normalization_issues}


def test_shadow_envelope_binds_statistic_without_mutating_raw_artifact(
    tmp_path: Path,
) -> None:
    statistic = tmp_path / "prevalence.json"
    raw = {
        "variable": "exposure",
        "positive_n": 25,
        "denominator_n": 100,
        "prevalence": 0.25,
        "ci_95_low": 0.17,
        "ci_95_high": 0.35,
    }
    statistic.write_text(json.dumps(raw), encoding="utf-8")
    before = statistic.read_bytes()
    summary = {
        "status": "completed",
        "output_files": {"statistic:exposure_prevalence": statistic.name},
    }

    envelope = normalize_step_result_shadow(
        step_id="03_prevalence",
        step_summary=summary,
        output_dir=tmp_path,
        status="ok",
    )

    assert statistic.read_bytes() == before
    assert verify_step_result_envelope(envelope)
    assert envelope.paper_authorized is False
    assert envelope.shadow is True
    assert len(envelope.artifacts) == 1
    assert len(envelope.statistics) == 1
    canonical = envelope.statistics[0]
    assert canonical.statistic_id == "exposure_prevalence"
    assert canonical.value == 0.25
    assert canonical.interval_low == 0.17
    assert canonical.interval_high == 0.35
    assert canonical.numerator == 25
    assert canonical.denominator == 100
    assert any(
        receipt.operation == "bind_declared_product_identity"
        for receipt in envelope.normalization_receipts
    )


def test_shadow_envelope_normalizes_numpy_nullable_and_rejects_nonfinite(
    tmp_path: Path,
) -> None:
    envelope = normalize_step_result_shadow(
        step_id="numeric",
        step_summary={
            "n": np.int64(8),
            "estimate": np.float64(1.25),
            "complete": np.bool_(True),
            "nullable": pd.NA,
            "bad": float("nan"),
            "notes": "free clinical explanation must not enter the envelope",
        },
        output_dir=tmp_path,
    )

    by_path = {item.field_path: item.value for item in envelope.observed_scalars}
    assert by_path["n"] == 8
    assert by_path["estimate"] == 1.25
    assert by_path["complete"] is True
    assert by_path["nullable"] is None
    assert "bad" not in by_path
    assert "notes" not in by_path
    assert "nonfinite_scalar" in _issue_codes(envelope)
    assert "untyped_string_omitted" in _issue_codes(envelope)
    operations = {receipt.operation for receipt in envelope.normalization_receipts}
    assert "numpy_scalar_to_builtin" in operations
    assert "nullable_to_null" in operations


def test_shadow_envelope_rejects_hostile_paths_and_accepts_explicit_container_root(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "metric.json"
    safe.write_text('{"value": 1.0}', encoding="utf-8")
    outside = tmp_path.parent / "outside.json"
    outside.write_text('{"value": 2.0}', encoding="utf-8")
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(outside)

    envelope = normalize_step_result_shadow(
        step_id="paths",
        step_summary={
            "output_files": {
                "statistic:safe": "/output/metric.json",
                "statistic:absolute": str(outside),
                "statistic:traversal": "../outside.json",
                "statistic:symlink": symlink.name,
            }
        },
        output_dir=tmp_path,
        container_output_roots=("/output",),
    )

    assert [item.product_id for item in envelope.artifacts] == ["statistic:safe"]
    assert {
        "absolute_unbound_product_path",
        "unsafe_relative_product_path",
        "symlink_product_path",
    }.issubset(_issue_codes(envelope))
    assert any(
        receipt.operation == "container_path_to_relative"
        for receipt in envelope.normalization_receipts
    )


def test_shadow_envelope_fails_closed_on_conflicting_or_nonstandard_statistic_json(
    tmp_path: Path,
) -> None:
    conflict = tmp_path / "conflict.json"
    conflict.write_text(
        '{"name":"other","estimate":1.0}',
        encoding="utf-8",
    )
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")

    envelope = normalize_step_result_shadow(
        step_id="invalid-statistics",
        step_summary={
            "output_files": {
                "statistic:expected": conflict.name,
                "statistic:nonfinite": nonfinite.name,
            }
        },
        output_dir=tmp_path,
    )

    assert envelope.statistics == ()
    assert {
        "conflicting_statistic_identity",
        "invalid_statistic_json",
    }.issubset(_issue_codes(envelope))
    assert len(envelope.artifacts) == 2


def test_shadow_envelope_records_current_vs_raw_summary_drift(
    tmp_path: Path,
) -> None:
    current = b'{"status":"completed"}'
    stale_raw = b'{"status":"failed"}'

    envelope = normalize_step_result_shadow(
        step_id="status-drift",
        step_summary={"status": "completed"},
        output_dir=tmp_path,
        source_summary_bytes=current,
        raw_summary_artifact_bytes=stale_raw,
        ledger_record_sha256="a" * 64,
    )

    assert envelope.source_summary_sha256 != envelope.raw_summary_artifact_sha256
    assert envelope.ledger_record_sha256 == "a" * 64


def test_shadow_envelope_replaces_host_bound_path_with_opaque_evidence_ref(
    tmp_path: Path,
) -> None:
    raw_path = "/easyicu-run/evidence/cohort.parquet"
    opaque_ref = "evidence:table_cohort@sha256:" + "b" * 64

    envelope = normalize_step_result_shadow(
        step_id="bound-path",
        step_summary={
            "cohort_path": raw_path,
            "input_bindings": [{"path": raw_path}],
        },
        output_dir=tmp_path,
        authorized_path_refs={raw_path: opaque_ref},
    )

    scalars = {item.field_path: item.value for item in envelope.observed_scalars}
    assert scalars["cohort_path"] == opaque_ref
    assert scalars["input_bindings[0].path"] == opaque_ref
    assert raw_path not in {str(item.value) for item in envelope.observed_scalars}
    assert envelope.input_evidence_refs == (opaque_ref,)
    assert "absolute_unbound_path" not in _issue_codes(envelope)
    assert (
        sum(
            receipt.operation == "authorized_path_to_evidence_ref"
            for receipt in envelope.normalization_receipts
        )
        == 2
    )


def test_shadow_envelope_is_strict_and_cannot_be_written_into_raw_outputs(
    tmp_path: Path,
) -> None:
    envelope = normalize_step_result_shadow(
        step_id="strict",
        step_summary={},
        output_dir=tmp_path,
    )
    with pytest.raises(ValidationError):
        StepResultEnvelope.model_validate(
            {
                **envelope.model_dump(mode="json"),
                "unexpected": "field",
            }
        )
    with pytest.raises(ValueError, match="must not be written"):
        write_shadow_step_result_envelope(
            envelope,
            tmp_path / "step_result.envelope.json",
            source_output_dir=tmp_path,
        )

    target = tmp_path.parent / "shadow" / "step_result.envelope.json"
    write_shadow_step_result_envelope(
        envelope,
        target,
        source_output_dir=tmp_path,
    )
    loaded = StepResultEnvelope.model_validate_json(target.read_text())
    assert verify_step_result_envelope(loaded)


def test_archived_replay_uses_current_ledger_and_verified_input_authority(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = tmp_path / "run"
    step_out = run_dir / "steps" / "01_analysis" / "outputs"
    evidence_dir = run_dir / "evidence"
    step_out.mkdir(parents=True)
    evidence_dir.mkdir()
    cohort = evidence_dir / "table_cohort__cohort.parquet"
    cohort.write_bytes(b"sealed cohort bytes")
    cohort_sha = hashlib.sha256(cohort.read_bytes()).hexdigest()
    statistic = step_out / "estimate.json"
    statistic.write_text('{"estimate":1.25}', encoding="utf-8")
    container_path = "/easyicu-run/evidence/table_cohort__cohort.parquet"
    current_summary = {
        "cohort_path": container_path,
        "output_files": {"statistic:primary_estimate": statistic.name},
    }
    (step_out / "step_summary.json").write_text(
        json.dumps({"status": "stale"}),
        encoding="utf-8",
    )
    manifest = {
        "per_step_records": [
            {
                "step_id": "01_analysis",
                "status": "ok",
                "planned_analysis_role": "primary_estimand",
                "resolved_input_evidence_ids": ["table_cohort"],
                "step_summary": current_summary,
            }
        ]
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    authority = {
        "records": [
            {
                "evidence_id": "table_cohort",
                "relative_path": "evidence/table_cohort__cohort.parquet",
                "sha256": cohort_sha,
            }
        ]
    }
    (evidence_dir / "evidence_authority.json").write_text(
        json.dumps(authority),
        encoding="utf-8",
    )
    shadow_dir = tmp_path / "shadow"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "replay_step_result_envelopes.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(shadow_dir),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    index = json.loads((shadow_dir / "index.json").read_text())
    assert index["normalization_error_count"] == 0
    envelope = StepResultEnvelope.model_validate_json(
        (shadow_dir / "01_analysis.step_result.envelope.json").read_text()
    )
    by_path = {item.field_path: item.value for item in envelope.observed_scalars}
    assert by_path["cohort_path"].startswith("evidence:table_cohort@sha256:")
    assert envelope.status == "ok"
    assert envelope.raw_summary_artifact_sha256 != envelope.source_summary_sha256
    assert (step_out / "step_summary.json").read_text() == json.dumps(
        {"status": "stale"}
    )

    authority["records"][0]["sha256"] = "0" * 64
    (evidence_dir / "evidence_authority.json").write_text(
        json.dumps(authority),
        encoding="utf-8",
    )
    rejected_shadow_dir = tmp_path / "shadow-rejected"
    rejected = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "replay_step_result_envelopes.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(rejected_shadow_dir),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 0, rejected.stderr
    rejected_index = json.loads((rejected_shadow_dir / "index.json").read_text())
    assert rejected_index["normalization_error_count"] == 1
    rejected_envelope = StepResultEnvelope.model_validate_json(
        (rejected_shadow_dir / "01_analysis.step_result.envelope.json").read_text()
    )
    assert rejected_envelope.input_evidence_refs == ()
    assert "absolute_unbound_path" in _issue_codes(rejected_envelope)


def test_shadow_envelope_is_not_wired_into_live_execution() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for relative in (
        "src/easyicu/research_agent/execution/phase.py",
        "src/easyicu/research_agent/pipeline.py",
    ):
        source = (repo_root / relative).read_text(encoding="utf-8")
        assert "execution.result_envelope" not in source
        assert "normalize_step_result_shadow" not in source

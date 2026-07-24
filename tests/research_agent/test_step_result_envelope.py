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

from easyicu.research_agent.audits.envelope_shadow import (
    compare_fraction_scale_shadow,
    compare_validator_shadow_inputs,
)
from easyicu.research_agent.audits.envelope_consumers import (
    CrossStepRegisteredOutputEnvelopeDualReader,
    StepSummaryFractionEnvelopeDualReader,
)
from easyicu.research_agent.audits.validators import (
    CrossStepRegisteredOutputValidator,
    StepSummaryFractionValidator,
)
from easyicu.research_agent.execution.result_envelope import (
    StepResultEnvelope,
    normalize_step_result_shadow,
    verify_step_result_envelope,
    write_shadow_step_result_envelope,
)
from easyicu.research_agent.schema import AnalysisStep


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


def test_v2_reader_preserves_v1_shadow_digest_compatibility(tmp_path: Path) -> None:
    current = normalize_step_result_shadow(
        step_id="legacy-shadow",
        step_summary={"n": 10},
        output_dir=tmp_path,
    )
    legacy = current.model_dump(mode="json", exclude={"content_sha256", "tables"})
    legacy["schema_version"] = "easyicu.step_result_envelope/1"
    legacy["content_sha256"] = hashlib.sha256(
        (
            json.dumps(
                legacy,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()

    loaded = StepResultEnvelope.model_validate(legacy)

    assert loaded.schema_version == "easyicu.step_result_envelope/1"
    assert loaded.tables == ()
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
    table = step_out / "estimate.csv"
    table.write_text("model_id,estimate\nprimary,1.25\n", encoding="utf-8")
    container_path = "/easyicu-run/evidence/table_cohort__cohort.parquet"
    current_summary = {
        "cohort_path": container_path,
        "output_files": {
            "statistic:primary_estimate": statistic.name,
            "table:primary_estimate": table.name,
        },
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
            },
            {
                "step_id": "02_reconciliation",
                "status": "ok",
                "step_summary": {
                    "registered_output": {
                        "upstream_step": "01_analysis",
                        "source_table_available": False,
                    }
                },
            },
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
    assert index["schema_version"] == "easyicu.shadow_step_result_index/4"
    assert index["normalization_error_count"] == 0
    assert index["validator_shadow_mismatch_count"] == 0
    assert index["registered_output_claim_count"] == 1
    assert index["registered_output_shadow_mismatch_count"] == 0
    assert index["fraction_shadow_mismatch_count"] == 0
    assert index["steps"][1]["registered_output_legacy_finding_count"] == 1
    assert index["steps"][1]["registered_output_shadow_exact"] is True
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
    assert rejected_index["validator_shadow_mismatch_count"] == 1
    assert rejected_index["registered_output_shadow_mismatch_count"] == 1
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
        assert "audits.envelope_consumers" not in source
        assert "audits.envelope_shadow" not in source
        assert "compare_validator_shadow_inputs" not in source


def test_registered_tables_compile_typed_population_missingness_and_estimate(
    tmp_path: Path,
) -> None:
    (tmp_path / "flow.csv").write_text(
        "stage,n_at_start,n_excluded,n_remaining,exposure_level\n"
        "input,100,0,100,all\n"
        "eligible,100,20,80,all\n"
        "eligible,100,0,50,0\n"
        "eligible,100,0,30,1\n",
        encoding="utf-8",
    )
    (tmp_path / "missing.csv").write_text(
        "variable,role,n_full,n_nonmissing,missing_n,missing_pct\n"
        "exposure,primary_exposure,80,60,20,25.0\n"
        "outcome,target_outcome,80,80,0,0.0\n"
        "age,adjustment,80,79,1,1.25\n",
        encoding="utf-8",
    )
    (tmp_path / "estimate.csv").write_text(
        "model_id,outcome,model_family,exposure,exposure_role,effect_scale,"
        "estimate,ci_low,ci_high,p_value,n,event_n,non_event_n,fit_status,"
        "converged,fit_method\n"
        "primary,outcome,logistic_regression,exposure,primary,odds_ratio,"
        "1.5,1.2,1.8,0.01,60,12,48,fitted,True,statsmodels_logit\n",
        encoding="utf-8",
    )
    summary = {
        "output_files": {
            "table:flow": "flow.csv",
            "table:missingness": "missing.csv",
            "table:primary_estimate": "estimate.csv",
        }
    }

    envelope = normalize_step_result_shadow(
        step_id="typed-tables",
        step_summary=summary,
        output_dir=tmp_path,
    )

    assert envelope.schema_version == "easyicu.step_result_envelope/2"
    assert verify_step_result_envelope(envelope)
    assert {role for table in envelope.tables for role in table.semantic_roles} >= {
        "effect_estimate",
        "missingness",
        "model_diagnostic",
        "population_flow",
    }
    assert envelope.population is not None
    assert envelope.population.eligible_n == 80
    assert envelope.population.analyzed_n == 60
    assert {
        (item.group_id, item.value) for item in envelope.population.group_counts
    } == {
        ("0", 50),
        ("1", 30),
    }
    assert envelope.missing_data is not None
    missing = {item.variable: item for item in envelope.missing_data.variables}
    assert missing["exposure"].missing_fraction == 0.25
    assert missing["exposure"].missing_n == 20
    assert envelope.variables is not None
    assert envelope.variables.exposures == ("exposure",)
    assert envelope.variables.outcomes == ("outcome",)
    assert envelope.variables.covariates == ("age",)
    assert len(envelope.statistics) == 1
    assert envelope.statistics[0].value == 1.5
    assert envelope.statistics[0].interval_low == 1.2
    assert envelope.statistics[0].interval_high == 1.8
    assert len(envelope.model_diagnostics) == 1
    assert envelope.model_diagnostics[0].converged is True
    assert not [
        issue for issue in envelope.normalization_issues if issue.severity == "error"
    ]


def test_registered_model_contract_is_typed_without_free_text(
    tmp_path: Path,
) -> None:
    model = {
        "model_contract": {
            "model_id": "primary_model",
            "outcome": "death",
            "model_family": "logistic_regression",
            "exposure_source": "lactate",
            "fit_status": "fitted",
            "converged": True,
            "separation_detected": False,
            "penalized": False,
            "fit_method": "statsmodels_glm",
            "n": 75,
            "event_n": 9,
            "fit_failure_reason": "free text that must not be copied",
        }
    }
    (tmp_path / "model.json").write_text(json.dumps(model), encoding="utf-8")

    envelope = normalize_step_result_shadow(
        step_id="model",
        step_summary={"output_files": {"artifact:model": "model.json"}},
        output_dir=tmp_path,
    )

    assert envelope.variables is not None
    assert envelope.variables.exposures == ("lactate",)
    assert envelope.variables.outcomes == ("death",)
    diagnostic = envelope.model_diagnostics[0]
    assert diagnostic.status == "fitted"
    assert diagnostic.converged is True
    assert diagnostic.separation_detected is False
    assert diagnostic.analyzed_n == 75
    assert diagnostic.event_n == 9
    assert "free text" not in envelope.model_dump_json()


def test_registered_table_parser_fails_closed_on_header_numeric_and_fraction_conflicts(
    tmp_path: Path,
) -> None:
    (tmp_path / "duplicate.csv").write_text(
        "variable,missing_n,missing_n,n_full,missing_pct\nx,1,1,10,10\n",
        encoding="utf-8",
    )
    (tmp_path / "conflict.csv").write_text(
        "variable,missing_n,n_full,fraction_missing,missing_pct\n" "x,1,10,0.1,50\n",
        encoding="utf-8",
    )
    (tmp_path / "nonfinite.csv").write_text(
        "model_id,outcome,exposure,effect_scale,estimate,ci_low,ci_high,n,"
        "fit_status,converged\n"
        "m,y,x,odds_ratio,NaN,1,2,10,fitted,yes\n",
        encoding="utf-8",
    )

    envelope = normalize_step_result_shadow(
        step_id="hostile-tables",
        step_summary={
            "output_files": {
                "table:duplicate": "duplicate.csv",
                "table:conflict": "conflict.csv",
                "table:nonfinite": "nonfinite.csv",
            }
        },
        output_dir=tmp_path,
    )

    assert {
        "conflicting_registered_fraction",
        "invalid_registered_boolean_cell",
        "invalid_registered_table_header",
        "nonfinite_registered_numeric_cell",
    }.issubset(_issue_codes(envelope))
    assert envelope.statistics == ()


def test_validator_shadow_comparison_is_exact_and_observational(
    tmp_path: Path,
) -> None:
    (tmp_path / "estimate.json").write_text('{"value":1.0}', encoding="utf-8")
    summary = {"output_files": {"statistic:estimate": "estimate.json"}}
    envelope = normalize_step_result_shadow(
        step_id="compare",
        step_summary=summary,
        output_dir=tmp_path,
        status="ok",
    )

    exact = compare_validator_shadow_inputs(
        step_summary=summary,
        envelope=envelope,
        current_status="ok",
    )
    assert exact.exact_match is True
    assert exact.decision_effect == "none"
    assert exact.mismatches == ()

    tampered = compare_validator_shadow_inputs(
        step_summary={"output_files": {"statistic:other": "estimate.json"}},
        envelope=envelope,
        current_status="failed",
    )
    assert tampered.exact_match is False
    assert tampered.decision_effect == "none"
    assert {item.code for item in tampered.mismatches} >= {
        "canonical_artifact_missing",
        "canonical_source_digest_mismatch",
        "canonical_status_mismatch",
        "canonical_unexpected_artifact",
    }


def _registered_output_record(summary: dict[str, object]) -> dict[str, object]:
    return {
        "step_id": "04_absolute_risk_context",
        "status": "ok",
        "evidence_ids": ["table_exposure_outcome_summary_8368e5ab"],
        "step_summary": summary,
    }


def _registered_output_consumer_summary() -> dict[str, object]:
    return {
        "registered_output": {
            "upstream_step": "04_absolute_risk_context",
            "source_table_available": False,
            "source_table_path": None,
        }
    }


def test_registered_output_gate_dual_read_matches_legacy_exactly(
    tmp_path: Path,
) -> None:
    table = tmp_path / "exposure_outcome_summary.csv"
    table.write_text("group,n\n0,40\n1,60\n", encoding="utf-8")
    prior_summary = {
        "status": "ok",
        "output_files": {
            "table:exposure_outcome_summary": table.name,
        },
    }
    prior = _registered_output_record(prior_summary)
    envelope = normalize_step_result_shadow(
        step_id="04_absolute_risk_context",
        step_summary=prior_summary,
        output_dir=tmp_path,
        status="ok",
    )
    validator = CrossStepRegisteredOutputValidator()
    step = AnalysisStep(
        step_id="05_reconciliation",
        intent="Audit the registered outputs of the prior risk step.",
    )

    legacy = validator.audit(
        step=step,
        step_summary=_registered_output_consumer_summary(),
        completed_step_records=[prior],
    )
    dual_read = CrossStepRegisteredOutputEnvelopeDualReader().audit(
        step=step,
        step_summary=_registered_output_consumer_summary(),
        completed_step_records=[prior],
        completed_step_envelopes={"04_absolute_risk_context": envelope},
    )

    assert [finding.model_dump(mode="json") for finding in dual_read] == [
        finding.model_dump(mode="json") for finding in legacy
    ]


def test_registered_output_gate_dual_read_fails_closed_without_envelope(
    tmp_path: Path,
) -> None:
    table = tmp_path / "exposure_outcome_summary.csv"
    table.write_text("group,n\n0,40\n1,60\n", encoding="utf-8")
    prior = _registered_output_record(
        {
            "status": "ok",
            "output_files": {
                "table:exposure_outcome_summary": table.name,
            },
        }
    )

    findings = CrossStepRegisteredOutputEnvelopeDualReader().audit(
        step=AnalysisStep(
            step_id="05_reconciliation",
            intent="Audit the registered outputs of the prior risk step.",
        ),
        step_summary=_registered_output_consumer_summary(),
        completed_step_records=[prior],
        completed_step_envelopes={},
    )

    assert len(findings) == 1
    assert findings[0].detail["canonical_shadow_blocked"] is True
    assert findings[0].detail["mismatch_codes"] == ["canonical_envelope_missing"]


def test_registered_output_gate_dual_read_rejects_source_or_presence_drift(
    tmp_path: Path,
) -> None:
    table = tmp_path / "exposure_outcome_summary.csv"
    table.write_text("group,n\n0,40\n1,60\n", encoding="utf-8")
    prior_summary = {
        "status": "ok",
        "output_files": {
            "table:exposure_outcome_summary": table.name,
        },
    }
    prior = _registered_output_record(prior_summary)
    envelope = normalize_step_result_shadow(
        step_id="04_absolute_risk_context",
        step_summary={"status": "ok"},
        output_dir=tmp_path,
        status="ok",
    )

    findings = CrossStepRegisteredOutputEnvelopeDualReader().audit(
        step=AnalysisStep(
            step_id="05_reconciliation",
            intent="Audit the registered outputs of the prior risk step.",
        ),
        step_summary=_registered_output_consumer_summary(),
        completed_step_records=[prior],
        completed_step_envelopes={"04_absolute_risk_context": envelope},
    )

    assert len(findings) == 1
    assert findings[0].detail["canonical_shadow_blocked"] is True
    assert set(findings[0].detail["mismatch_codes"]) >= {
        "canonical_artifact_missing",
        "canonical_source_digest_mismatch",
        "canonical_table_presence_mismatch",
    }


def _finding_payloads(findings: list[object]) -> list[dict[str, object]]:
    return [finding.model_dump(mode="json") for finding in findings]


@pytest.mark.parametrize(
    "summary",
    (
        {
            "missingness": {
                "missing_fraction": {"lab_max": 56.372144},
                "missing_pct": {"lab_max": 56.372144},
            }
        },
        {
            "missingness": {
                "missing_fraction": {"lab_max": 0.56372144},
                "missing_pct": {"lab_max": 56.372144},
            }
        },
        {
            "overall_outcome_prevalence": {
                "outcome_column": "death",
                "n": 94_458,
                "event_n": 9_466,
                "non_event_n": 84_992,
                "risk": 0.1002,
                "ci_low": 0.0983,
                "ci_high": 0.1022,
            }
        },
        {
            "overall_outcome_prevalence": {
                "n_events": 9_466,
                "mortality_percentage": 10.02,
                "risk": 0.1002,
            }
        },
        {
            "absolute_risk": {
                "bili_distribution": {"q25": 0.8, "q75": 1.4},
                "outcome_risk": 10.0,
            }
        },
        {"observed_fraction": {"group_a": 40.0, "risk": 0.2}},
        {"observed_fraction": {"group_a": {"value": 40.0}}},
        {
            "overall_risk": {
                "interval": {
                    "estimate": 10.0,
                    "ci_low": 9.0,
                    "ci_high": 11.0,
                }
            }
        },
        {"observed_fraction": {"by_group": [{"group": "a", "value": 40.0}]}},
        {"overall_risk": {"risk": 0.2, "bootstrap_replicates": 1_000}},
        {"observed_fraction": {"metric": "aic", "group_a": 40.0}},
        {"observed_fraction": {"unit": "unknown", "value": 40.0}},
        {
            "overall_risk": {
                "risk": 0.2,
                "effect": {
                    "measure": "odds_ratio",
                    "estimate": 2.0,
                    "ci_low": 1.2,
                    "ci_high": 3.0,
                },
            }
        },
        {
            "overall_risk": {
                "risk": 0.2,
                "estimates": [
                    {
                        "measure": "odds_ratio",
                        "estimate": 2.0,
                        "ci_low": 1.2,
                        "ci_high": 3.0,
                    }
                ],
            }
        },
        {
            "overall_risk": {
                "risk": 0.2,
                "levels": [
                    {
                        "level": 4,
                        "count": 1_256,
                        "fraction": 0.013,
                        "risk": 0.378,
                        "ci_low": 0.352,
                        "ci_high": 0.405,
                    }
                ],
            }
        },
        {
            "fractional_polynomial_power": 2,
            "sampling_fraction_denominator": 500,
            "sampling_fraction_numerator": 125,
            "attributable_fraction": -0.08,
            "observed_fraction": {
                "value": 0.25,
                "numerator": 125,
                "denominator": 500,
            },
        },
        {
            "valid_level_distribution_percent": {"0": 0.7, "1": 0.3},
            "valid_level_distribution_percent_pct": {"0": 70.0, "1": 30.0},
        },
        {"prevalence_ci_high": 1.0000000000000002},
        {
            "risk": {
                "estimate": 0.2,
                "ci_low": 0.1,
                "ci_high": 1.0000000000000002,
            }
        },
        {
            "risk_ratio": 1.8,
            "risk_ratio_ci_low": 1.4,
            "risk_ratio_ci_high": 2.2,
            "odds_ratio": 2.4,
            "hazard_ratio": 1.7,
            "risk_difference": -0.2,
            "number_at_risk": 120,
        },
        {
            "absolute_risk": 0.2,
            "risk_ratio": 1.8,
            "ci_low": 1.4,
            "ci_high": 2.2,
        },
    ),
)
def test_fraction_scale_dual_read_matches_legacy_adversarial_corpus(
    tmp_path: Path,
    summary: dict[str, object],
) -> None:
    step = AnalysisStep(step_id="04_fraction_audit", intent="Audit bounded metrics.")
    envelope = normalize_step_result_shadow(
        step_id=step.step_id,
        step_summary=summary,
        output_dir=tmp_path,
        status="ok",
    )

    legacy = StepSummaryFractionValidator().audit(
        step=step,
        step_summary=summary,
    )
    dual_read = StepSummaryFractionEnvelopeDualReader().audit(
        step=step,
        step_summary=summary,
        envelope=envelope,
        current_status="ok",
    )
    comparison = compare_fraction_scale_shadow(
        step=step,
        step_summary=summary,
        current_status="ok",
        envelope=envelope,
        legacy_findings=legacy,
    )

    assert _finding_payloads(dual_read) == _finding_payloads(legacy)
    assert comparison.exact_match is True
    assert comparison.legacy_finding_count == len(legacy)
    assert comparison.canonical_finding_count == len(legacy)


def test_fraction_scale_dual_read_fails_closed_without_envelope() -> None:
    step = AnalysisStep(step_id="04_fraction_audit", intent="Audit bounded metrics.")
    summary = {"observed_fraction": {"group_a": 40.0}}

    findings = StepSummaryFractionEnvelopeDualReader().audit(
        step=step,
        step_summary=summary,
        envelope=None,
        current_status="ok",
    )

    assert len(findings) == 1
    assert findings[0].detail["canonical_shadow_blocked"] is True
    assert findings[0].detail["mismatch_codes"] == ["canonical_envelope_missing"]


def test_fraction_scale_dual_read_fails_closed_on_source_or_status_drift(
    tmp_path: Path,
) -> None:
    step = AnalysisStep(step_id="04_fraction_audit", intent="Audit bounded metrics.")
    envelope = normalize_step_result_shadow(
        step_id=step.step_id,
        step_summary={"observed_fraction": {"group_a": 0.4}},
        output_dir=tmp_path,
        status="failed",
    )

    findings = StepSummaryFractionEnvelopeDualReader().audit(
        step=step,
        step_summary={"observed_fraction": {"group_a": 40.0}},
        envelope=envelope,
        current_status="ok",
    )

    assert len(findings) == 1
    assert findings[0].detail["canonical_shadow_blocked"] is True
    assert set(findings[0].detail["mismatch_codes"]) >= {
        "canonical_fraction_view_mismatch",
        "canonical_source_digest_mismatch",
        "canonical_status_mismatch",
    }


def test_fraction_scale_dual_read_fails_closed_on_nonfinite_normalization(
    tmp_path: Path,
) -> None:
    step = AnalysisStep(step_id="04_fraction_audit", intent="Audit bounded metrics.")
    summary = {"observed_fraction": {"group_a": float("nan")}}
    envelope = normalize_step_result_shadow(
        step_id=step.step_id,
        step_summary=summary,
        output_dir=tmp_path,
        status="ok",
    )

    findings = StepSummaryFractionEnvelopeDualReader().audit(
        step=step,
        step_summary=summary,
        envelope=envelope,
        current_status="ok",
    )

    assert len(findings) == 1
    assert findings[0].detail["canonical_shadow_blocked"] is True
    assert set(findings[0].detail["mismatch_codes"]) >= {
        "canonical_fraction_view_mismatch",
        "normalization_error",
    }


def test_fraction_scale_dual_read_preserves_legacy_numeric_string_finding(
    tmp_path: Path,
) -> None:
    step = AnalysisStep(step_id="04_fraction_audit", intent="Audit bounded metrics.")
    summary = {"observed_fraction": {"group_a": "40.0"}}
    envelope = normalize_step_result_shadow(
        step_id=step.step_id,
        step_summary=summary,
        output_dir=tmp_path,
        status="ok",
    )

    findings = StepSummaryFractionEnvelopeDualReader().audit(
        step=step,
        step_summary=summary,
        envelope=envelope,
        current_status="ok",
    )
    legacy = StepSummaryFractionValidator().audit(
        step=step,
        step_summary=summary,
    )

    assert _finding_payloads(findings) == _finding_payloads(legacy)
    assert findings[0].detail["reported_value"] == 40.0


def test_fraction_scale_dual_read_fails_closed_on_omitted_numeric_string(
    tmp_path: Path,
) -> None:
    step = AnalysisStep(step_id="04_fraction_audit", intent="Audit bounded metrics.")
    summary = {"risk": "40.0"}
    envelope = normalize_step_result_shadow(
        step_id=step.step_id,
        step_summary=summary,
        output_dir=tmp_path,
        status="ok",
    )

    findings = StepSummaryFractionEnvelopeDualReader().audit(
        step=step,
        step_summary=summary,
        envelope=envelope,
        current_status="ok",
    )

    assert len(findings) == 1
    assert findings[0].detail["canonical_shadow_blocked"] is True
    assert findings[0].detail["mismatch_codes"] == ["canonical_fraction_view_mismatch"]

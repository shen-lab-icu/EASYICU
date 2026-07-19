"""Current step checkpoints, not append-only files, own scientific authority."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from easyicu.research_agent.evaluation_scorecard import score_run_from_dir
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.icu_agent_bench import ICUAgentBenchTask
from easyicu.research_agent.pipeline_primary_effect import _extract_primary_effect_row
from easyicu.research_agent.reporting.readiness import (
    _blocked_outcome_step_ids,
    _compute_readiness_gates,
    primary_result_plausibility_errors,
    primary_survival_estimate_integrity_errors,
)
from easyicu.research_agent.schema import PipelineResult, ResearchContext
from easyicu.research_agent.validity_signals import assess_validity_signals


def _write_summary(run_dir: Path, step_id: str, payload: dict) -> Path:
    outputs = run_dir / "steps" / step_id / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / "step_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_manifest(
    run_dir: Path,
    records: list[dict],
    *,
    evidence: list[dict] | None = None,
) -> Path:
    path = run_dir / "manifest_partial.json"
    path.write_text(
        json.dumps(
            {
                "per_step_records": records,
                "evidence": evidence or [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Is exposure associated with mortality?",
        cohort={
            "cohort_name": "authority",
            "database": "miiv",
            "n_patients": 100,
            "n_stays": 100,
        },
        variables=[],
    )


def _survival_plan(step_id: str = "01_primary_survival") -> SimpleNamespace:
    return SimpleNamespace(
        steps=[
            SimpleNamespace(
                step_id=step_id,
                method="survival_analysis",
                intent="primary survival association",
                expected_outputs=[],
            )
        ]
    )


def _invalid_blocked_summary() -> dict:
    return {
        "analysis_executed": False,
        "primary_analysis_authorized": False,
        "blocked_reason": "mortality outcome analysis blocked",
        "hazard_ratio": -1.0,
        "p_value": 0.01,
        "n_analysis": 100,
        "n_events": 120,
    }


def test_failed_current_step_makes_stale_readiness_files_inert(tmp_path: Path):
    step_id = "01_primary_survival"
    stale = _invalid_blocked_summary()
    _write_summary(tmp_path, step_id, stale)
    records = [
        {"step_id": step_id, "status": "ok", "step_summary": stale},
        {
            "step_id": step_id,
            "status": "contract_failed",
            "step_summary": {"status": "rejected"},
        },
    ]
    _write_manifest(tmp_path, records)

    assert _blocked_outcome_step_ids(tmp_path) == []
    assert primary_result_plausibility_errors(tmp_path) == []
    assert primary_survival_estimate_integrity_errors(_survival_plan(), tmp_path) == []

    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        "Mortality was associated with the exposure between groups.",
        encoding="utf-8",
    )
    gates = _compute_readiness_gates(
        context=_context(),
        plan=None,
        per_step_records=records,
        findings=[],
        evidence=EvidenceStore(tmp_path),
        run_dir=tmp_path,
        manuscript_path=manuscript,
        stop_after_analysis=False,
    )
    assert gates["blocked_outcome_step_ids"] == []
    assert not any(
        token in message
        for message in gates["analysis_errors"]
        for token in ("implausible", "blocked outcome gate leaked")
    )


def test_active_and_legacy_readiness_controls_still_gate(tmp_path: Path):
    step_id = "01_primary_survival"
    stale = _invalid_blocked_summary()
    _write_summary(tmp_path, step_id, stale)
    _write_manifest(
        tmp_path,
        [{"step_id": step_id, "status": "ok", "step_summary": stale}],
    )

    assert _blocked_outcome_step_ids(tmp_path) == [step_id]
    assert primary_result_plausibility_errors(tmp_path)
    assert primary_survival_estimate_integrity_errors(_survival_plan(), tmp_path)

    (tmp_path / "manifest_partial.json").unlink()
    assert _blocked_outcome_step_ids(tmp_path) == [step_id]
    assert primary_result_plausibility_errors(tmp_path)
    assert primary_survival_estimate_integrity_errors(_survival_plan(), tmp_path)


def test_blocked_outcome_gate_fails_closed_when_evidence_is_tampered(
    tmp_path: Path,
) -> None:
    step_id = "01_outcome_gate"
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    gate_path = evidence_dir / "outcome_gate__outcome_gate.csv"
    gate_path.write_text(
        "status,primary_analysis_authorized,outcome\n" "blocked,false,mortality\n",
        encoding="utf-8",
    )
    evidence_id = "outcome_gate"
    records = [
        {
            "step_id": step_id,
            "status": "ok",
            "evidence_ids": [evidence_id],
            "step_summary": {},
        }
    ]
    _write_manifest(
        tmp_path,
        records,
        evidence=[
            {
                "evidence_id": evidence_id,
                "kind": "table",
                "relative_path": str(gate_path.relative_to(tmp_path)),
                "sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
                "produced_by_step": step_id,
            }
        ],
    )
    assert _blocked_outcome_step_ids(tmp_path, records) == [step_id]

    gate_path.write_text(
        "status,primary_analysis_authorized,outcome\n" "ok,true,mortality\n",
        encoding="utf-8",
    )

    assert _blocked_outcome_step_ids(tmp_path, records) == [step_id]


def test_validity_signals_ignore_failed_summary_and_failed_evidence(
    tmp_path: Path,
):
    step_id = "01_model_training"
    stale_summary = {
        "auroc": 0.8,
        "split_integrity": {
            "split_unit": "patient",
            "patient_overlap_n": 12,
        },
    }
    _write_summary(tmp_path, step_id, stale_summary)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    stale_balance = evidence_dir / "table_old__weighted_balance.csv"
    stale_balance.write_text("feature,abs_smd\nage,0.7\n", encoding="utf-8")
    records = [
        {
            "step_id": step_id,
            "status": "ok",
            "step_summary": stale_summary,
            "evidence_ids": ["old_balance"],
        },
        {
            "step_id": step_id,
            "status": "contract_failed",
            "step_summary": {"status": "rejected"},
            "evidence_ids": [],
        },
    ]
    _write_manifest(
        tmp_path,
        records,
        evidence=[
            {
                "evidence_id": "old_balance",
                "relative_path": str(stale_balance.relative_to(tmp_path)),
                "produced_by_step": step_id,
            }
        ],
    )

    prediction = assess_validity_signals("mortality_prediction", tmp_path)
    causal = assess_validity_signals("causal_inference", tmp_path)
    assert [(signal.name, signal.status) for signal in prediction] == [
        ("patient_level_split_no_overlap", "na")
    ]
    assert all(signal.status == "na" for signal in causal)

    clean_summary = {
        "split_integrity": {
            "split_unit": "patient",
            "patient_overlap_n": 0,
        }
    }
    _write_manifest(
        tmp_path,
        [{"step_id": step_id, "status": "ok", "step_summary": clean_summary}],
    )
    assert assess_validity_signals("mortality_prediction", tmp_path)[0].status == "pass"

    (tmp_path / "manifest_partial.json").unlink()
    assert assess_validity_signals("mortality_prediction", tmp_path)[0].status == "fail"


def _score_task() -> ICUAgentBenchTask:
    return ICUAgentBenchTask(
        task_id="authority_score",
        kind="descriptive_association",
        title="authority score",
        objective="authority score",
        expected_outputs=["table one"],
    )


def _write_score_status(run_dir: Path) -> None:
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "gates": {
                    "required_step_count": 1,
                    "completed_step_count": 0,
                    "failed_steps": [
                        {"step_id": "01_primary_model", "status": "contract_failed"}
                    ],
                    "execution_complete": False,
                    "evidence_complete": False,
                    "numeric_verified": False,
                    "missing_evidence_count": 0,
                    "manuscript_ready": False,
                }
            }
        ),
        encoding="utf-8",
    )


def test_scorecard_does_not_read_stale_failed_primary_coefficients(
    tmp_path: Path,
):
    _write_score_status(tmp_path)
    outputs = tmp_path / "steps" / "01_primary_model" / "outputs"
    outputs.mkdir(parents=True)
    coefficient_path = outputs / "model_coefficients.csv"
    with coefficient_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variable", "coef"])
        writer.writeheader()
        writer.writerow({"variable": "sepsis3", "coef": 0.8})
        writer.writerow({"variable": "sofa_max", "coef": 0.5})
    secondary_path = (
        tmp_path / "steps" / "02_secondary_model" / "outputs" / "coefficients.csv"
    )
    secondary_path.parent.mkdir(parents=True)
    secondary_path.write_text(
        "variable,coef\nsepsis3,0.7\nsofa_max,0.4\n",
        encoding="utf-8",
    )
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    registered_primary = evidence_dir / "active_coefficients__model_coefficients.csv"
    registered_secondary = evidence_dir / "secondary_coefficients__coefficients.csv"
    registered_primary.write_bytes(coefficient_path.read_bytes())
    registered_secondary.write_bytes(secondary_path.read_bytes())

    def registered_record(
        *, evidence_id: str, path: Path, produced_by_step: str
    ) -> dict:
        return {
            "evidence_id": evidence_id,
            "relative_path": str(path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "produced_by_step": produced_by_step,
        }

    primary_summary = {"primary_model": {"exposure": "sepsis3"}}
    _write_manifest(
        tmp_path,
        [
            {
                "step_id": "01_primary_model",
                "status": "ok",
                "step_summary": primary_summary,
                "evidence_ids": ["stale_coefficients"],
            },
            {
                "step_id": "01_primary_model",
                "status": "contract_failed",
                "step_summary": {"status": "rejected"},
                "evidence_ids": [],
            },
            {
                "step_id": "02_secondary_model",
                "status": "ok",
                "step_summary": {"secondary_association": {"status": "ok"}},
                "evidence_ids": ["secondary_coefficients"],
            },
        ],
        evidence=[
            registered_record(
                evidence_id="stale_coefficients",
                path=registered_primary,
                produced_by_step="01_primary_model",
            ),
            registered_record(
                evidence_id="secondary_coefficients",
                path=registered_secondary,
                produced_by_step="02_secondary_model",
            ),
        ],
    )

    failed = score_run_from_dir(
        _score_task(),
        tmp_path,
        exposure_concept="sepsis3",
    )
    assert failed.result_validity.level is None
    assert not any("overadjustment" in note for note in failed.result_validity.notes)

    _write_manifest(
        tmp_path,
        [
            {
                "step_id": "01_primary_model",
                "status": "ok",
                "step_summary": primary_summary,
                "evidence_ids": ["active_coefficients"],
            }
        ],
        evidence=[
            registered_record(
                evidence_id="active_coefficients",
                path=registered_primary,
                produced_by_step="01_primary_model",
            )
        ],
    )
    active = score_run_from_dir(
        _score_task(),
        tmp_path,
        exposure_concept="sepsis3",
    )
    assert active.result_validity.level == "Fail"
    assert any("overadjustment" in note for note in active.result_validity.notes)

    (tmp_path / "manifest_partial.json").unlink()
    legacy = score_run_from_dir(
        _score_task(),
        tmp_path,
        exposure_concept="sepsis3",
    )
    assert legacy.result_validity.level == "Fail"


def _pipeline_result(run_dir: Path) -> PipelineResult:
    return PipelineResult(
        run_id="authority_run",
        workdir=str(run_dir),
        context_path=str(run_dir / "research_context.json"),
        plan_path=str(run_dir / "analysis_plan.json"),
        manifest_path=str(run_dir / "manifest.json"),
        report_path=str(run_dir / "results_report.md"),
        manuscript_path=str(run_dir / "manuscript.md"),
        evidence_count=0,
        findings_count=0,
    )


def test_cross_database_primary_effect_uses_active_then_legacy_summary(
    tmp_path: Path,
):
    step_id = "01_primary_model"
    stale = {
        "primary_predictor": "exposure",
        "primary_or": 9.9,
        "primary_ci_low": 8.0,
        "primary_ci_high": 12.0,
        "n": 100,
    }
    _write_summary(tmp_path, step_id, stale)
    _write_manifest(
        tmp_path,
        [
            {"step_id": step_id, "status": "ok", "step_summary": stale},
            {
                "step_id": step_id,
                "status": "contract_failed",
                "step_summary": {"status": "rejected"},
            },
        ],
    )

    failed = _extract_primary_effect_row(
        database="miiv",
        result=_pipeline_result(tmp_path),
    )
    assert failed["primary_or"] is None
    assert failed["status"] == "missing_primary_association"

    active_summary = {
        **stale,
        "primary_or": 1.25,
        "primary_ci_low": 1.1,
        "primary_ci_high": 1.4,
    }
    _write_manifest(
        tmp_path,
        [{"step_id": step_id, "status": "ok", "step_summary": active_summary}],
    )
    active = _extract_primary_effect_row(
        database="miiv",
        result=_pipeline_result(tmp_path),
    )
    assert active["primary_or"] == 1.25

    (tmp_path / "manifest_partial.json").unlink()
    legacy = _extract_primary_effect_row(
        database="miiv",
        result=_pipeline_result(tmp_path),
    )
    assert legacy["primary_or"] == 9.9

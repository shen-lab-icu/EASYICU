from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = _REPO_ROOT / "tools" / "run_v14_agent_experiments.py"
_SPEC = importlib.util.spec_from_file_location("run_v14_agent_experiments", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
v14 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = v14
_SPEC.loader.exec_module(v14)


def test_v14_task_registry_has_ten_real_cohort_tasks():
    v14._bootstrap_imports()
    tasks = v14._load_task_specs_from_builder()

    assert len(tasks) == 10
    keys = {task.key for task in tasks}
    assert "t04_lactate_mortality_association" in keys
    assert "t06_shock_phenotype_clustering" in keys
    assert "t07_mortality_prediction_auroc" in keys
    assert all(task.cohort_file.endswith(".parquet") for task in tasks)
    task_by_key = {task.key: task for task in tasks}
    assert task_by_key["t03_severity_score_correlation"].target_outcome is None


def test_openrouter_env_validation_does_not_leak_credentials(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc:
        v14._validate_provider_env("openrouter")

    message = str(exc.value)
    assert "OPENROUTER_API_KEY" in message
    assert "sk-" not in message


def test_metric_extractor_handles_v14_analysis_families(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_test"
    step_dir = run_dir / "steps" / "01_model" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manuscript_scaffold_bound.md").write_text(
        "Result sentence.\n[evidence missing: calibration]\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_20260508T000000_test",
                "used_mock_llm": False,
                "llm_signature": "openrouter:test",
                "prompt_pack_version": "easyicu-research-agent-prompts/v1",
                "evidence": [
                    {"kind": "code", "evidence_id": "code"},
                    {"kind": "log", "evidence_id": "log"},
                    {"kind": "table", "evidence_id": "table"},
                    {"kind": "figure", "evidence_id": "figure"},
                    {"kind": "statistic", "evidence_id": "stat"},
                ],
                "findings": [
                    {
                        "severity": "warning",
                        "message": "Selection bias warning and SOFA zero artefact.",
                    }
                ],
                "per_step_records": [
                    {"step_id": "01_model", "status": "ok"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (step_dir / "step_summary.json").write_text(
        json.dumps(
            {
                "primary_or": 1.42,
                "held_out_auroc": 0.73,
                "brier_score": 0.18,
                "statistic:silhouette_score": 0.31,
                "statistic:cluster_count": 2,
                "spearman_rho": 0.62,
                "n_complete_case": 900,
                "n_sofa2_zero": 12,
            }
        ),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t07_mortality_prediction_auroc",
        title="Prediction",
        family="prediction_model",
        difficulty="advanced",
        cohort_file="t07_mortality_prediction_auroc.parquet",
        question="Build a prediction model.",
        expected_metrics=["auroc", "calibration"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["execution_success"] is True
    assert metrics["evidence_kinds_complete"] is True
    assert metrics["evidence_missing_count"] == 1
    assert metrics["primary_or"] == 1.42
    assert metrics["auroc"] == 0.73
    assert metrics["brier_score"] == 0.18
    assert metrics["silhouette_score"] == 0.31
    assert metrics["cluster_count"] == 2
    assert metrics["spearman_rho"] == 0.62
    assert metrics["complete_case_n"] == 900
    assert metrics["sofa_zero_count"] == 12
    assert metrics["selection_bias_warning"] is False
    assert metrics["warning_source"] is None
    assert metrics["guardrail_warning"] is True
    assert metrics["calibration"] is True
    assert metrics["expected_metric_hits"] == {"auroc": True, "calibration": True}


def test_failure_classifier_has_watchdog_bucket():
    failure = v14._classify_failure(RuntimeError("runtime stalled heartbeat missing"), None)

    assert failure == "runtime_stalled"


def test_metric_extractor_accepts_predictor_named_or_key(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_assoc"
    step_dir = run_dir / "steps" / "03_association" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run_20260508T000000_assoc", "evidence": [], "findings": []}),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (step_dir / "step_summary.json").write_text(
        json.dumps({"lactate_max_24h_or": 1.23}),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t04_lactate_mortality_association",
        title="Association",
        family="association_study",
        difficulty="intermediate",
        cohort_file="t04_lactate_mortality_association.parquet",
        question="Estimate lactate association.",
        expected_metrics=["primary_or"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["primary_or"] == 1.23
    assert metrics["expected_metric_hits"]["primary_or"] is True


def test_metric_extractor_accepts_or_estimate_key(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_or_estimate"
    step_dir = run_dir / "steps" / "04_primary_association_model" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run_20260508T000000_or_estimate", "evidence": [], "findings": []}),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (step_dir / "step_summary.json").write_text(
        json.dumps({"or_estimate": 1.2148}),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t04_lactate_mortality_association",
        title="Association",
        family="association_study",
        difficulty="intermediate",
        cohort_file="t04_lactate_mortality_association.parquet",
        question="Estimate lactate association.",
        expected_metrics=["primary_or"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["primary_or"] == 1.2148
    assert metrics["expected_metric_hits"]["primary_or"] is True


def test_metric_extractor_accepts_strategy_specific_robustness_keys(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_robust"
    step_dir = run_dir / "steps" / "03_model_fitting_complete_case" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run_20260508T000000_robust", "evidence": [], "findings": []}),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (step_dir / "step_summary.json").write_text(
        json.dumps({"statistic": {"cc_or_lactate": 1.18, "cc_n": 450}}),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t10_complete_case_robustness",
        title="Robustness",
        family="robustness",
        difficulty="advanced",
        cohort_file="t10_complete_case_robustness.parquet",
        question="Compare lactate association robustness.",
        expected_metrics=["primary_or", "complete_case_n"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["primary_or"] == 1.18
    assert metrics["complete_case_n"] == 450
    assert metrics["expected_metric_hits"]["primary_or"] is True
    assert metrics["expected_metric_hits"]["complete_case_n"] is True


def test_metric_extractor_accepts_n_complete_cases_alias(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_kdigo_alias"
    step_dir = run_dir / "steps" / "04_primary_model" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run_20260508T000000_kdigo_alias", "evidence": [], "findings": []}),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (step_dir / "step_summary.json").write_text(
        json.dumps({"primary_association_estimate": 0.41, "n_complete_cases": 992}),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t05_kdigo_renal_sensitivity",
        title="KDIGO",
        family="association_sensitivity",
        difficulty="intermediate",
        cohort_file="t05_kdigo_renal_sensitivity.parquet",
        question="Estimate KDIGO association.",
        expected_metrics=["complete_case_n"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["complete_case_n"] == 992
    assert metrics["expected_metric_hits"]["complete_case_n"] is True


def test_metric_extractor_accepts_robustness_metric_lists(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_robust_lists"
    step_dir = run_dir / "steps" / "01_robustness_analysis" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run_20260508T000000_robust_lists", "evidence": [], "findings": []}),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (step_dir / "step_summary.json").write_text(
        json.dumps(
            {
                "strategy": ["complete_case", "missing_indicator", "reduced_variable"],
                "sample_size": [217, 785, 338],
                "lactate_or": [2.35, 3.15, 2.35],
            }
        ),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t10_complete_case_robustness",
        title="Robustness",
        family="robustness",
        difficulty="advanced",
        cohort_file="t10_complete_case_robustness.parquet",
        question="Compare lactate association robustness.",
        expected_metrics=["primary_or", "complete_case_n"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["primary_or"] == 2.35
    assert metrics["complete_case_n"] == 217
    assert metrics["expected_metric_hits"]["primary_or"] is True
    assert metrics["expected_metric_hits"]["complete_case_n"] is True


def test_metric_extractor_accepts_sofa_zero_count_alias(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_sofa_zero_alias"
    step_dir = run_dir / "steps" / "03_mortality_by_sofa_zero" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run_20260508T000000_sofa_zero_alias", "evidence": [], "findings": []}),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (step_dir / "step_summary.json").write_text(
        json.dumps({"sofa2_max_24h_zero_count": 41}),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t09_sofa_zero_artefact_audit",
        title="SOFA zero",
        family="data_quality_audit",
        difficulty="advanced",
        cohort_file="t09_sofa_zero_artefact_audit.parquet",
        question="Audit SOFA-zero artefacts.",
        expected_metrics=["sofa_zero_count"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["sofa_zero_count"] == 41
    assert metrics["expected_metric_hits"]["sofa_zero_count"] is True


def test_metric_extractor_uses_probe_summary_without_masking_step_count(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_table_one"
    probe_dir = run_dir / "steps" / "00_probe" / "outputs"
    step_dir = run_dir / "steps" / "01_table_one" / "outputs"
    probe_dir.mkdir(parents=True)
    step_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run_20260508T000000_table_one", "evidence": [], "findings": []}),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (probe_dir / "probe_summary.json").write_text(
        json.dumps(
            {
                "n_rows": 1000,
                "outcome_rate": 0.096,
                "top_missing_columns": [{"variable": "lactate_max_24h", "fraction_missing": 0.487}],
            }
        ),
        encoding="utf-8",
    )
    (step_dir / "step_summary.json").write_text(
        json.dumps({"variable": "death", "missing_pct": 0.0}),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t01_table_one_descriptive",
        title="Table one",
        family="descriptive",
        difficulty="basic",
        cohort_file="t01_table_one_descriptive.parquet",
        question="Create table one.",
        expected_metrics=["n_rows", "mortality_rate", "missingness"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["step_summary_count"] == 1
    assert metrics["n_rows"] == 1000
    assert metrics["mortality_rate"] == 0.096
    assert metrics["missingness"] is True
    assert metrics["expected_metric_hits"] == {
        "n_rows": True,
        "mortality_rate": True,
        "missingness": True,
    }


def test_metric_extractor_accepts_spearman_correlation_dict(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_corr"
    step_dir = run_dir / "steps" / "02_correlation_analysis" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run_20260508T000000_corr", "evidence": [], "findings": []}),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (step_dir / "step_summary.json").write_text(
        json.dumps({"statistic": {"spearman_correlations": {"sofa2_resp": 0.71}}}),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t03_severity_score_correlation",
        title="Correlation",
        family="correlation",
        difficulty="intermediate",
        cohort_file="t03_severity_score_correlation.parquet",
        question="Estimate correlations.",
        expected_metrics=["spearman_rho"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["spearman_rho"] == 0.71
    assert metrics["expected_metric_hits"]["spearman_rho"] is True


def test_metric_extractor_accepts_textual_or_and_cc_sample_size(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_text_or"
    step_dir = run_dir / "steps" / "04_primary_association_model" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run_20260508T000000_text_or", "evidence": [], "findings": []}),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (step_dir / "step_summary.json").write_text(
        json.dumps({
            "summary": {
                "notes": [
                    "Association estimate with lactate: OR=1.219 (95% CI 1.116-1.332)."
                ]
            },
            "cc_sample_size": 217,
        }),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t10_complete_case_robustness",
        title="Robustness",
        family="robustness",
        difficulty="advanced",
        cohort_file="t10_complete_case_robustness.parquet",
        question="Compare lactate association robustness.",
        expected_metrics=["primary_or", "complete_case_n"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["primary_or"] == 1.219
    assert metrics["complete_case_n"] == 217
    assert metrics["expected_metric_hits"]["primary_or"] is True
    assert metrics["expected_metric_hits"]["complete_case_n"] is True


def test_metric_extractor_robustness_prefers_complete_case_or_over_event_rate_alias(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_robust_event_rate"
    step_dir = run_dir / "steps" / "03_complete_case_robustness" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run_20260508T000000_robust_event_rate", "evidence": [], "findings": []}),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (step_dir / "step_summary.json").write_text(
        json.dumps(
            {
                "statistic:primary_or": 19.82,
                "statistic:event_rate": 19.82,
                "statistic:lactate_or_complete_case": 1.126,
                "statistic:complete_case_n": 217,
            }
        ),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t10_complete_case_robustness",
        title="Robustness",
        family="robustness",
        difficulty="advanced",
        cohort_file="t10_complete_case_robustness.parquet",
        question="Compare lactate association robustness.",
        expected_metrics=["primary_or", "complete_case_n"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["primary_or"] == 1.126
    assert metrics["primary_metric_source"] == "statistic:lactate_or_complete_case"
    assert metrics["complete_case_n"] == 217
    assert metrics["aggregation_version"] == v14.AGGREGATION_VERSION


def test_metric_extractor_prefers_adjusted_or_for_bias_audit(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_bias"
    step_dir = run_dir / "steps" / "04_primary_association_model" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run_20260508T000000_bias", "evidence": [], "findings": []}),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (step_dir / "step_summary.json").write_text(
        json.dumps(
            {
                "estimate": 0.52,
                "adjusted_or": 1.07,
                "selection_bias_warning": "Confounded by indication; avoid causal treatment-effect language.",
            }
        ),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t08_vaso_selection_bias_audit",
        title="Bias audit",
        family="bias_audit",
        difficulty="advanced",
        cohort_file="t08_vaso_selection_bias_audit.parquet",
        question="Audit vasopressor selection bias.",
        expected_metrics=["primary_or", "selection_bias_warning"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["primary_or"] == 1.07
    assert metrics["primary_metric_source"] == "adjusted_or"
    assert metrics["selection_bias_warning"] is True
    assert metrics["warning_source"] == "selection_bias_warning"


def test_selection_bias_warning_requires_explicit_bias_language(tmp_path: Path):
    run_dir = tmp_path / "run_20260508T000000_selection_word"
    step_dir = run_dir / "steps" / "01_model" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_20260508T000000_selection_word",
                "evidence": [],
                "findings": [{"severity": "warning", "message": "Feature selection used a manual screen."}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (step_dir / "step_summary.json").write_text(
        json.dumps({"feature_selection_method": "manual pre-screen"}),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t07_mortality_prediction_auroc",
        title="Prediction",
        family="prediction_model",
        difficulty="advanced",
        cohort_file="t07_mortality_prediction_auroc.parquet",
        question="Build a prediction model.",
        expected_metrics=["auroc"],
    )

    metrics = v14._extract_metrics(run_dir, task)

    assert metrics["selection_bias_warning"] is False
    assert metrics["warning_source"] is None


def test_acceptance_status_requires_expected_metrics_and_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run_clean"
    step_dir = run_dir / "steps" / "01_model" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manuscript_scaffold_bound.md").write_text("All supported.", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_clean",
                "evidence": [
                    {"kind": "code", "evidence_id": "code"},
                    {"kind": "log", "evidence_id": "log"},
                    {"kind": "table", "evidence_id": "table"},
                    {"kind": "figure", "evidence_id": "figure"},
                    {"kind": "statistic", "evidence_id": "stat"},
                ],
                "findings": [],
                "per_step_records": [{"step_id": "01_model", "status": "ok"}],
            }
        ),
        encoding="utf-8",
    )
    (step_dir / "step_summary.json").write_text(json.dumps({"held_out_auroc": 0.8}), encoding="utf-8")
    task = v14.V14Task(
        key="t07_mortality_prediction_auroc",
        title="Prediction",
        family="prediction_model",
        difficulty="advanced",
        cohort_file="x.parquet",
        question="Build model.",
        expected_metrics=["auroc", "brier_score"],
        required_artifacts=["manifest", "bound_manuscript", "step_summary", "table", "statistic", "figure"],
    )

    metrics = v14._extract_metrics(run_dir, task)
    failure = v14._classify_failure(None, metrics, task)
    record = v14._with_status_fields({"run_dir": str(run_dir), "failure_class": failure, "metrics": metrics})

    assert failure == "metric_contract_failure"
    assert record["pipeline_status"] == "completed"
    assert record["acceptance_status"] == "partial"
    assert v14._missing_expected_metrics(metrics) == ["brier_score"]


def test_acceptance_status_clean_ok_when_contract_is_complete(tmp_path: Path):
    run_dir = tmp_path / "run_clean"
    step_dir = run_dir / "steps" / "01_model" / "outputs"
    step_dir.mkdir(parents=True)
    (run_dir / "manuscript_scaffold_bound.md").write_text("All supported.", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_clean",
                "evidence": [
                    {"kind": "code", "evidence_id": "code"},
                    {"kind": "log", "evidence_id": "log"},
                    {"kind": "table", "evidence_id": "table"},
                    {"kind": "figure", "evidence_id": "figure"},
                    {"kind": "statistic", "evidence_id": "stat"},
                ],
                "findings": [],
                "per_step_records": [{"step_id": "01_model", "status": "ok"}],
            }
        ),
        encoding="utf-8",
    )
    (step_dir / "step_summary.json").write_text(
        json.dumps(
            {
                "held_out_auroc": 0.8,
                "brier_score": 0.16,
                "baseline_prevalence": 0.1,
                "split_strategy": "held-out train/test split",
            }
        ),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="t07_mortality_prediction_auroc",
        title="Prediction",
        family="prediction_model",
        difficulty="advanced",
        cohort_file="x.parquet",
        question="Build model.",
        expected_metrics=["auroc", "brier_score", "baseline_prevalence", "split_or_cv"],
        required_artifacts=["manifest", "bound_manuscript", "step_summary", "table", "statistic", "figure"],
    )

    metrics = v14._extract_metrics(run_dir, task)
    failure = v14._classify_failure(None, metrics, task)
    record = v14._with_status_fields({"run_dir": str(run_dir), "failure_class": failure, "metrics": metrics})

    assert failure is None
    assert record["acceptance_status"] == "clean_ok"


def test_aggregate_only_recovers_stale_heartbeat(tmp_path: Path):
    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"fake")
    out_root = tmp_path / "out"
    heartbeat = out_root / "_heartbeats" / "model" / "task" / "aware" / "heartbeat.json"
    heartbeat.parent.mkdir(parents=True)
    heartbeat.write_text(
        json.dumps(
            {
                "status": "running",
                "updated_at": "2000-01-01T00:00:00+00:00",
                "pid": 99999999,
            }
        ),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="task",
        title="Task",
        family="x",
        difficulty="x",
        cohort_file="cohort.parquet",
        question="Question",
    )

    record = v14._stale_heartbeat_record(
        task=task,
        arm="aware",
        model="model",
        provider="openai",
        experiment_mode="self_repair",
        cohort_path=cohort,
        out_root=out_root,
        task_timeout=60,
    )

    assert record is not None
    assert record["acceptance_status"] == "stalled"
    assert record["failure_class"] == "runtime_stalled"


def test_aggregate_only_preserves_terminal_stalled_heartbeat(tmp_path: Path):
    cohort = tmp_path / "cohort.parquet"
    cohort.write_bytes(b"fake")
    out_root = tmp_path / "out"
    heartbeat = out_root / "_heartbeats" / "model" / "task" / "aware" / "heartbeat.json"
    heartbeat.parent.mkdir(parents=True)
    heartbeat.write_text(
        json.dumps(
            {
                "status": "runtime_stalled",
                "updated_at": "2000-01-01T00:00:00+00:00",
                "failure_class": "runtime_stalled",
                "error": "Task exceeded watchdog timeout.",
            }
        ),
        encoding="utf-8",
    )
    task = v14.V14Task(
        key="task",
        title="Task",
        family="x",
        difficulty="x",
        cohort_file="cohort.parquet",
        question="Question",
    )

    record = v14._terminal_heartbeat_record(
        task=task,
        arm="aware",
        model="model",
        provider="openai",
        experiment_mode="self_repair",
        cohort_path=cohort,
        out_root=out_root,
    )

    assert record is not None
    assert record["acceptance_status"] == "stalled"
    assert record["failure_class"] == "runtime_stalled"


def test_reuse_selects_best_run_not_latest(tmp_path: Path):
    task = v14.V14Task(
        key="t10_complete_case_robustness",
        title="Robustness",
        family="robustness",
        difficulty="advanced",
        cohort_file="t10_complete_case_robustness.parquet",
        question="Compare lactate association robustness.",
        expected_metrics=["primary_or", "complete_case_n"],
    )
    cohort_path = tmp_path / "cohort.parquet"
    cohort_path.write_bytes(b"cohort")
    arm_root = tmp_path / "qwen3-coder-30b" / task.key / "aware"
    older = arm_root / "run_20260508T000000_good"
    newer = arm_root / "run_20260508T000100_bad"
    for run_dir in (older, newer):
        (run_dir / "steps" / "01_model" / "outputs").mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "evidence": [
                        {"kind": "code", "evidence_id": "code"},
                        {"kind": "log", "evidence_id": "log"},
                        {"kind": "table", "evidence_id": "table"},
                        {"kind": "figure", "evidence_id": "figure"},
                        {"kind": "statistic", "evidence_id": "stat"},
                    ],
                    "findings": [],
                    "per_step_records": [{"step_id": "01_model", "status": "ok"}],
                }
            ),
            encoding="utf-8",
        )
    (older / "manuscript_scaffold_bound.md").write_text("ok", encoding="utf-8")
    (older / "steps" / "01_model" / "outputs" / "step_summary.json").write_text(
        json.dumps({"statistic:primary_or": 1.12, "statistic:complete_case_n": 217}),
        encoding="utf-8",
    )
    (newer / "manuscript_scaffold_bound.md").write_text(
        "[evidence missing: primary_association]",
        encoding="utf-8",
    )
    (newer / "steps" / "01_model" / "outputs" / "step_summary.json").write_text(
        json.dumps({"statistic:primary_or": 1.12}),
        encoding="utf-8",
    )

    record = v14._reuse_task_arm(
        task=task,
        arm="aware",
        model="qwen3-coder-30b",
        cohort_path=cohort_path,
        out_root=tmp_path,
    )

    assert record is not None
    assert record["run_id"] == older.name
    assert record["acceptance_status"] == "clean_ok"

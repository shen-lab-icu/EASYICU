"""Validator tests — the core of the agent's safety story.

These tests are intentionally adversarial: each one feeds the
validator something that *should* trip an ICU rule, and asserts the
right finding pops out at the right severity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _ctx_with_sofa(ra) -> "ra.ResearchContext":
    """Tiny context with a sofa2 column — the SOFA-aware validators
    fire on this shape."""
    df = pd.DataFrame({
        "stay_id": list(range(1, 11)),
        "age": [60, 70, 50, 80, 65, 75, 90, 40, 55, 60],
        "sofa2": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "lact": [1.0, 2.0, 1.5, 3.0, 4.0, 5.0, 2.5, 1.2, 3.3, 7.0],
        "death": [1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
    })
    return ra.build_research_context(
        research_question="sofa2 → death?",
        cohort=df, cohort_name="t", database="synthetic",
        target_outcome="death",
    )


def test_concept_usage_flags_mean_of_sofa(ra):
    ctx = _ctx_with_sofa(ra)
    auditor = ra.ConceptUsageAuditor()
    code = "x = df['sofa2'].mean()  # forbidden"
    findings = auditor.audit(context=ctx, script_text=code)
    assert any(f.severity == "error" and "sofa" in f.message.lower() or "ordinal" in f.message.lower()
               for f in findings), findings
    # at least one error finding with the right validator name
    assert any(f.severity == "error" and f.validator == auditor.name for f in findings)


def test_concept_usage_flags_mean_of_lact_without_median(ra):
    ctx = _ctx_with_sofa(ra)
    auditor = ra.ConceptUsageAuditor()
    code = "lact_avg = df['lact'].mean()"
    findings = auditor.audit(context=ctx, script_text=code)
    # Lab + mean without median → warning
    assert any(f.severity == "warning" and "lact" in f.message.lower() for f in findings)


def test_concept_usage_silences_lab_mean_when_median_present(ra):
    ctx = _ctx_with_sofa(ra)
    auditor = ra.ConceptUsageAuditor()
    code = "x = df['lact'].mean(); y = df['lact'].median()"
    findings = auditor.audit(context=ctx, script_text=code)
    assert all("lact" not in f.message.lower() for f in findings)


def test_concept_usage_flags_fillna_zero(ra):
    ctx = _ctx_with_sofa(ra)
    auditor = ra.ConceptUsageAuditor()
    code = "df['lact'] = df['lact'].fillna(0)"
    findings = auditor.audit(context=ctx, script_text=code)
    assert any("fillna" in f.message.lower() or "imputation" in f.message.lower()
               for f in findings)


def test_concept_usage_fillna_zero_ignores_env_string_subscripts(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='import os\npath = os.environ["COHORT_PARQUET"]',
    )
    assert not any("fillna" in f.message.lower() or "imputation" in f.message.lower()
                   for f in findings)


def test_concept_usage_flags_agg_mean_of_sofa(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='x = df["sofa2"].agg("mean")',
    )
    assert any(f.severity == "error" and "sofa" in f.message.lower() for f in findings)


def test_concept_usage_flags_numpy_mean_of_sofa(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='import numpy as np\nx = np.mean(df["sofa2"])',
    )
    assert any(f.severity == "error" and "sofa" in f.message.lower() for f in findings)


def test_concept_usage_flags_rolling_mean_of_sofa(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='x = df["sofa2"].rolling(3).mean()',
    )
    assert any(f.severity == "error" and "sofa" in f.message.lower() for f in findings)


def test_statistical_validator_flags_outcome_mismatch(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    pd.read_parquet  # touch import
    df = pd.DataFrame({
        "stay_id": list(range(1, 11)),
        "age": [60] * 10,
        "sofa2": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "lact": [1.0] * 10,
        "death": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0.2 incidence
    })
    df.to_parquet(cohort_path, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # write a placeholder so out_dir is non-empty
    (out_dir / "step_summary.json").write_text("{}", encoding="utf-8")

    schema = ra.schema
    step = schema.AnalysisStep(step_id="02_outcome_incidence",
                               intent="incidence", expected_outputs=["statistic:outcome_rate"])
    validator = ra.StatisticalValidator()
    findings = validator.audit(
        context=ctx, cohort_path=cohort_path, step=step,
        out_dir=out_dir,
        # report a clearly wrong outcome rate
        step_summary={"outcome_rate": 0.99},
    )
    msgs = " ".join(f.message for f in findings)
    assert any(f.severity == "error" for f in findings), findings
    assert "outcome rate" in msgs.lower() or "disagrees" in msgs.lower()


def test_statistical_validator_flags_sofa_zero_anomaly(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    df = pd.DataFrame({
        "stay_id": list(range(1, 21)),
        "age": [60] * 20,
        "sofa2": [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5,
        "death": [1, 1, 1, 1, 0,  # rate at 0 = 0.8
                  0, 0, 0, 0, 0,  # rate at 1 = 0.0
                  0, 0, 0, 1, 0,  # rate at 2 = 0.2
                  1, 1, 1, 1, 1], # rate at 3 = 1.0
    })
    df.to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Build a sofa_strata.csv with the anomaly: rate@0 > rate@1
    pd.DataFrame({"sofa2": [0, 1, 2, 3], "n": [5, 5, 5, 5],
                  "outcome_rate": [0.8, 0.0, 0.2, 1.0]}).to_csv(
        out_dir / "sofa_strata.csv", index=False)

    schema = ra.schema
    step = schema.AnalysisStep(step_id="05_sofa_zero_audit", intent="audit")
    validator = ra.StatisticalValidator()
    findings = validator.audit(
        context=ctx, cohort_path=cohort_path, step=step,
        out_dir=out_dir, step_summary={},
    )
    assert any("non-monotonic" in f.message.lower() or "exceeds" in f.message.lower()
               for f in findings), findings
    assert any(f.severity == "warning" for f in findings)


def test_statistical_validator_accepts_real_llm_stratum_audit_shape(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "age": [60, 61, 62, 63],
        "sofa2": [0, 0, 1, 1],
        "death": [1, 0, 0, 0],
    }).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame({
        "sofa2_score": [0, 1],
        "n_total": [2, 2],
        "n_death": [1, 0],
        "mortality_rate": [0.5, 0.0],
        "mortality_rate_ci_low": [0.1, 0.0],
        "mortality_rate_ci_high": [0.9, 0.7],
    }).to_csv(out_dir / "stratum_audit.csv", index=False)

    step = ra.schema.AnalysisStep(step_id="05_stratum_level_audit", intent="audit")
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort_path, step=step,
        out_dir=out_dir, step_summary={},
    )
    assert any("stratum_audit.csv" in str(f.detail) for f in findings), findings
    assert any("non-monotonic" in f.message.lower() for f in findings), findings


def test_statistical_validator_recomputes_sofa_zero_when_audit_omits_mortality(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "age": [60, 61, 62, 63],
        "sofa2": [0, 0, 1, 1],
        "death": [1, 0, 0, 0],
        "lact": [2.0, 2.2, 1.5, 1.7],
    }).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame({
        "variable": ["lact"],
        "sofa2_0_median": [2.1],
        "sofa2_1_median": [1.6],
    }).to_csv(out_dir / "table:sofa2_stratum_audit.csv", index=False)

    step = ra.schema.AnalysisStep(
        step_id="05_stratum_level_audit",
        intent="Audit SOFA-2 score==0 vs score==1.",
    )
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort_path, step=step,
        out_dir=out_dir, step_summary={},
    )
    assert any(
        f.detail and f.detail.get("source") == "cohort_recompute"
        for f in findings
    ), findings
    assert any("non-monotonic" in f.message.lower() for f in findings), findings


def test_statistical_validator_no_artefacts_is_error(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1], "age": [60], "sofa2": [3], "lact": [1.0], "death": [0]}
                 ).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()  # deliberately empty
    schema = ra.schema
    step = schema.AnalysisStep(step_id="99_empty", intent="x")
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort_path, step=step,
        out_dir=out_dir, step_summary={},
    )
    assert any(f.severity == "error" and "no output artefacts" in f.message.lower()
               for f in findings), findings


def test_statistical_validator_flags_primary_or_mismatch(ra, tmp_path: Path):
    """T1.6 — when the reported OR disagrees with primary_association.csv,
    the validator must surface an error finding."""
    ctx = _ctx_with_sofa(ra)
    cohort_path = tmp_path / "cohort.parquet"
    df = pd.DataFrame({
        "stay_id": list(range(1, 11)),
        "age": [60] * 10,
        "sofa2": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "lact": [1.0] * 10,
        "death": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })
    df.to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame([
        {"variable": "intercept", "coef": -2.0, "odds_ratio": 0.135},
        {"variable": "sofa2", "coef": 0.4, "odds_ratio": 1.491825},
        {"variable": "age", "coef": 0.01, "odds_ratio": 1.01005},
    ]).to_csv(out_dir / "primary_association.csv", index=False)

    schema = ra.schema
    step = schema.AnalysisStep(step_id="04_primary_association", intent="logit")
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort_path, step=step,
        out_dir=out_dir,
        # report a wildly wrong OR
        step_summary={"predictor": "sofa2", "primary_or": 5.0,
                      "outcome_rate": float(df["death"].mean())},
    )
    assert any(f.severity == "error" and "primary or" in f.message.lower()
               for f in findings), findings


def test_cohort_auditor_row_count_mismatch(ra, tmp_path: Path):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4, 5],
        "age": [60.0] * 5,
        "death": [0, 1, 0, 1, 0],
    })
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path, index=False)
    ctx = ra.build_research_context(
        research_question="x", cohort=df,
        cohort_name="c", database="synthetic", target_outcome="death",
    )
    # Pretend the descriptor was written when there were 99 rows.
    ctx.cohort.n_stays = 99
    findings = ra.CohortAuditor().audit(context=ctx, cohort_path=cohort_path)
    assert any(f.severity == "error" and "row count mismatch" in f.message.lower()
               for f in findings)


def test_cohort_auditor_allows_correlation_context_without_target_outcome(ra, tmp_path: Path):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "sofa2_max_24h": [1, 3, 5],
        "sofa2_resp_max_24h": [0, 1, 2],
    })
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path, index=False)
    ctx = ra.build_research_context(
        research_question="Correlate SOFA components.",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome=None,
    )

    findings = ra.CohortAuditor().audit(context=ctx, cohort_path=cohort_path)

    assert not any("Target outcome" in f.message for f in findings)


def test_llm_concept_auditor_parses_findings(ra):
    from easyicu.research_agent.validators import parse_llm_concept_audit_response

    raw = """```json
{"findings":[{"severity":"warning","message":"ICU mortality may be confused with hospital mortality.","detail":{"column":"death_hosp"}}]}
```"""
    findings = parse_llm_concept_audit_response(raw, step_id="04_primary")
    assert len(findings) == 1
    assert findings[0].validator == "llm_concept_auditor"
    assert findings[0].severity == "warning"
    assert findings[0].detail["step_id"] == "04_primary"


def test_clinical_constraint_validator_warns_on_missing_time_zero(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "research_question": "Estimate the effect of early vasopressor treatment on death.",
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="causal_inference"
            ),
        }
    )
    step = ra.schema.AnalysisStep(step_id="04_causal_protocol", intent="Target-trial style causal analysis")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    findings = ra.ClinicalConstraintValidator().audit(
        context=ctx,
        step=step,
        out_dir=out_dir,
        step_summary={},
    )
    assert any("immortal time bias" in f.message.lower() for f in findings), findings


def test_statistical_guard_warns_when_prediction_outputs_lack_split_metadata(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="prediction_model",
                covariates=["age", "sex", "sofa2"],
            ),
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({
        "stay_id": list(range(1, 11)),
        "age": [60] * 10,
        "sofa2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "death": [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    }).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame({
        "model": ["logit"],
        "auc": [0.76],
        "brier": [0.18],
    }).to_csv(out_dir / "model_performance_train_test.csv", index=False)
    step = ra.schema.AnalysisStep(step_id="04_prediction", intent="prediction model analysis")
    findings = ra.StatisticalGuard().audit(
        context=ctx,
        cohort_path=cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary={},
    )
    messages = " ".join(f.message.lower() for f in findings)
    assert "train/test split" in messages
    assert "calibration_slope" in messages


def test_statistical_guard_accepts_v14_cv_prediction_summary(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="prediction_model",
                covariates=["age", "sex", "sofa2"],
            ),
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame({
        "stay_id": list(range(1, 41)),
        "age": [60] * 40,
        "sofa2": list(range(10)) * 4,
        "death": [0, 1] * 20,
    }).to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    step = ra.schema.AnalysisStep(step_id="01_model_training", intent="prediction model analysis")

    findings = ra.StatisticalGuard().audit(
        context=ctx,
        cohort_path=cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary={
            "statistic:cv_auroc_mean": 0.74,
            "statistic:cv_brier_mean": 0.18,
            "cv_folds": 5,
            "split_strategy": "5-fold cross-validation",
        },
    )

    messages = " ".join(f.message.lower() for f in findings)
    assert "held-out performance" not in messages
    assert "train/test split" not in messages
    assert "calibration_slope" not in messages

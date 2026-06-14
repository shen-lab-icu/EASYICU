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
    # Impartiality contract: mean/SD of an ordinal/composite score is a
    # reporting-practice *preference*, not an objective error, so it is a
    # WARNING (advisory) that never hard-blocks a run. The caution must
    # still be raised (so a reviewer sees it), just not as severity="error".
    ctx = _ctx_with_sofa(ra)
    auditor = ra.ConceptUsageAuditor()
    code = "x = df['sofa2'].mean()  # advisory"
    findings = auditor.audit(context=ctx, script_text=code)
    matched = [
        f for f in findings
        if f.validator == auditor.name
        and ("sofa" in f.message.lower() or "ordinal" in f.message.lower())
    ]
    assert matched, findings
    assert all(f.severity == "warning" for f in matched), matched
    # ...and no forbidden-aggregation finding is escalated to a blocking error.
    assert not any(
        f.severity == "error" and "misleading" in f.message.lower()
        for f in findings
    ), findings


def test_concept_usage_mean_of_sofa_blocks_under_strict_ablation(ra, monkeypatch):
    # The historical strict fail-closed benchmark stays reproducible behind
    # EASYICU_AUDIT_ORDINAL_STRICT=1, which restores severity="error" for
    # primary-analysis / manuscript stages.
    monkeypatch.setenv("EASYICU_AUDIT_ORDINAL_STRICT", "1")
    ctx = _ctx_with_sofa(ra)
    auditor = ra.ConceptUsageAuditor()
    schema = ra.schema
    step = schema.AnalysisStep(step_id="primary_association", intent="primary")
    findings = auditor.audit(
        context=ctx, script_text="x = df['sofa2'].mean()", step=step
    )
    assert any(
        f.severity == "error" and f.validator == auditor.name for f in findings
    ), findings


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
    # Detection still fires across call forms; severity is advisory (warning).
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='x = df["sofa2"].agg("mean")',
    )
    assert any(f.severity == "warning" and "sofa" in f.message.lower() for f in findings)


def test_concept_usage_flags_numpy_mean_of_sofa(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='import numpy as np\nx = np.mean(df["sofa2"])',
    )
    assert any(f.severity == "warning" and "sofa" in f.message.lower() for f in findings)


def test_concept_usage_flags_rolling_mean_of_sofa(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='x = df["sofa2"].rolling(3).mean()',
    )
    assert any(f.severity == "warning" and "sofa" in f.message.lower() for f in findings)


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


def test_statistical_validator_ignores_outcome_blind_component_qc_table(
    ra,
    tmp_path: Path,
):
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
    pd.DataFrame({
        "variable": ["sofa2"],
        "n_rows": [20],
        "n_low_completeness": [5],
        "frac_low_completeness": [0.25],
    }).to_csv(out_dir / "component_completeness_qc.csv", index=False)

    schema = ra.schema
    step = schema.AnalysisStep(
        step_id="05_component_completeness_qc",
        intent="component completeness QC",
    )
    validator = ra.StatisticalValidator()
    findings = validator.audit(
        context=ctx, cohort_path=cohort_path, step=step,
        out_dir=out_dir, step_summary={},
    )
    assert not any(
        "non-monotonic" in f.message.lower() or "exceeds" in f.message.lower()
        for f in findings
    ), findings


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


# ---------------- cohort-hygiene flags (impartial, advisory) -------------

def _hygiene_ctx(ra, df):
    return ra.build_research_context(
        research_question="Does sepsis predict ICU mortality?",
        cohort=df, cohort_name="c", database="synthetic",
        target_outcome="death",
    )


def test_cohort_hygiene_flags_missing_patient_id_when_outcome(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "los_icu": [2.0, 3.0, 5.0],
        "death": [0, 1, 0],
    })
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    pid = [f for f in findings
           if f.detail.get("subkind") == "patient_independence_unassessable"]
    assert len(pid) == 1
    assert pid[0].severity == "warning"
    assert pid[0].detail["structural_no_source"] is True
    # Advice, not a mandate: it must not assert independence or demand a filter.
    assert "re-extract" in pid[0].message.lower()


def test_cohort_hygiene_no_patient_flag_with_patient_id(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame({
        "subject_id": [10, 10, 11],
        "stay_id": [1, 2, 3],
        "los_icu": [2.0, 3.0, 5.0],
        "death": [0, 1, 0],
    })
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    assert not any(
        f.detail.get("subkind") == "patient_independence_unassessable"
        for f in findings
    )


def test_cohort_hygiene_no_patient_flag_without_outcome(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame({"stay_id": [1, 2, 3], "los_icu": [2.0, 3.0, 5.0]})
    ctx = ra.build_research_context(
        research_question="Describe LoS.", cohort=df,
        cohort_name="c", database="synthetic", target_outcome=None,
    )
    findings = cohort_hygiene_findings(df, ctx)
    assert not any(
        f.detail.get("subkind") == "patient_independence_unassessable"
        for f in findings
    )


def test_cohort_hygiene_short_stay_reported_not_enforced(ra):
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "los_icu": [0.2, 0.5, 3.0, 5.0],  # half are <1 day
        "death": [0, 1, 0, 1],
    })
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    short = [f for f in findings
             if f.detail.get("subkind") == "short_stay_exposure"]
    assert len(short) == 1
    assert short[0].severity == "warning"
    assert short[0].detail["fraction_los_under_1_day"] == 0.5
    assert "no minimum-los filter is imposed" in short[0].message.lower()


def test_cohort_hygiene_findings_never_block(ra):
    """Impartiality: hygiene flags are advisory and must never fail-close."""
    from easyicu.research_agent.audits.validators import cohort_hygiene_findings

    df = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "los_icu": [0.1, 0.2, 5.0],
        "death": [0, 1, 0],
    })
    findings = cohort_hygiene_findings(df, _hygiene_ctx(ra, df))
    assert findings  # both flags fire
    assert all(f.severity == "warning" for f in findings)
    assert all(f.detail.get("impartial") is True for f in findings)


def test_cohort_auditor_surfaces_hygiene_flags(ra, tmp_path: Path):
    """The hygiene flags reach callers through CohortAuditor.audit."""
    df = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "los_icu": [0.2, 3.0, 5.0],
        "age": [60.0, 70.0, 80.0],
        "death": [0, 1, 0],
    })
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path, index=False)
    ctx = _hygiene_ctx(ra, df)
    findings = ra.CohortAuditor().audit(context=ctx, cohort_path=cohort_path)
    assert any(f.detail.get("kind") == "cohort_hygiene" for f in findings)


def test_llm_concept_auditor_parses_findings(ra):
    from easyicu.research_agent.audits.validators import parse_llm_concept_audit_response

    raw = """```json
{"findings":[{"severity":"warning","message":"ICU mortality may be confused with hospital mortality.","detail":{"column":"death_hosp"}}]}
```"""
    findings = parse_llm_concept_audit_response(raw, step_id="04_primary")
    assert len(findings) == 1
    assert findings[0].validator == "llm_concept_auditor"
    assert findings[0].severity == "warning"
    assert findings[0].detail["step_id"] == "04_primary"


def test_llm_concept_auditor_prompt_includes_outcome_semantics(ra):
    auditor = ra.LLMConceptAuditor(ra.MockLLMClient())
    ctx = ra.build_research_context(
        research_question="Is age associated with ICU mortality?",
        cohort=pd.DataFrame({
            "stay_id": [1, 2, 3],
            "age": [60, 70, 80],
            "death": [0, 1, 0],
        }),
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )
    prompt = auditor._prompt(context=ctx, script_text="print('hello')", step=None)
    assert "icu_mortality" in prompt
    assert "explicitly treated as ICU mortality" in prompt


def test_llm_concept_auditor_downgrades_nonblocking_outcome_confusion(ra):
    from easyicu.research_agent.audits.validators import parse_llm_concept_audit_response

    raw = """
    {
      "findings": [
        {
          "severity": "error",
          "message": "ICU vs hospital mortality confusion",
          "detail": {
            "issue": "Explicitly noted that 'death' is ICU mortality, but the script does not verify or enforce consistent usage across all downstream analyses or reporting."
          }
        }
      ]
    }
    """
    findings = parse_llm_concept_audit_response(raw, step_id="02_model")
    assert len(findings) == 1
    assert findings[0].severity == "warning"


def test_llm_concept_auditor_uses_context_to_downgrade_outcome_ambiguity(ra):
    class _FalsePositiveLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return """
            {
              "findings": [
                {
                  "severity": "error",
                  "message": "ICU vs hospital mortality confusion",
                  "detail": {
                    "context": "The script uses death without clarifying whether it is ICU, hospital, or 28-day mortality."
                  }
                }
              ]
            }
            """

    ctx = ra.build_research_context(
        research_question="Is early lactate associated with ICU mortality?",
        cohort=pd.DataFrame({
            "stay_id": [1, 2, 3],
            "lactate_max_24h": [1.0, 2.0, 3.0],
            "death": [0, 1, 0],
        }),
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )
    findings = ra.LLMConceptAuditor(_FalsePositiveLLM()).audit(
        context=ctx,
        script_text="model.fit(df[['lactate_max_24h']], df['death'])",
        step=None,
    )

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].detail["downgraded_reason"]


def test_llm_concept_auditor_preserves_error_for_conflicting_outcome_label(ra):
    class _ConfusionLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return """
            {
              "findings": [
                {
                  "severity": "error",
                  "message": "ICU vs hospital mortality confusion",
                  "detail": {"context": "The plot labels ICU death as hospital mortality."}
                }
              ]
            }
            """

    ctx = ra.build_research_context(
        research_question="Is early lactate associated with ICU mortality?",
        cohort=pd.DataFrame({
            "stay_id": [1, 2, 3],
            "lactate_max_24h": [1.0, 2.0, 3.0],
            "death": [0, 1, 0],
        }),
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )
    findings = ra.LLMConceptAuditor(_ConfusionLLM()).audit(
        context=ctx,
        script_text="ax.set_title('Adjusted association with hospital mortality')",
        step=None,
    )

    assert len(findings) == 1
    assert findings[0].severity == "error"


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


def test_clinical_constraint_validator_does_not_flag_prediction_feature_list_as_treatment_effect(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "research_question": (
                "Build a mortality prediction workflow using age, sex, SOFA-2, "
                "lactate, MAP, and vasopressor exposure."
            ),
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="prediction_model"
            ),
        }
    )
    step = ra.schema.AnalysisStep(
        step_id="01_model_training",
        intent="Train and validate the mortality prediction model with AUROC and calibration.",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    findings = ra.ClinicalConstraintValidator().audit(
        context=ctx,
        step=step,
        out_dir=out_dir,
        step_summary={"statistic:auroc": 0.8, "statistic:brier_score": 0.18},
    )
    assert not any("immortal time bias" in f.message.lower() for f in findings), findings


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


# ---------------------------------------------------------------------------
# Degenerate-partition disclosure caution (clustering / trajectory)
# ---------------------------------------------------------------------------


def _cluster_sizes_dir(tmp_path: Path, sizes) -> Path:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    total = float(sum(sizes))
    pd.DataFrame({
        "cluster": list(range(len(sizes))),
        "n": sizes,
        "pct": [s / total * 100.0 for s in sizes],
    }).to_csv(out / "cluster_sizes.csv", index=False)
    return out


def test_statistical_validator_flags_degenerate_partition(ra, tmp_path: Path):
    # The M3 scenario: a "2-cluster solution" that is really 99.5% / 0.5%.
    # silhouette/ARI on such a split are inflated by outlier isolation, so the
    # agent must be cautioned to disclose the size imbalance.
    ctx = _ctx_with_sofa(ra)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort, index=False)
    out_dir = _cluster_sizes_dir(tmp_path, [38584, 203])
    step = ra.schema.AnalysisStep(
        step_id="01_phenotype_trajectory_clustering", intent="subphenotype clustering"
    )
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort, step=step, out_dir=out_dir,
        step_summary={"silhouette": 0.808, "mean_ari": 1.0},
    )
    deg = [f for f in findings if "degenerate" in f.message.lower()]
    assert deg, findings
    assert all(f.severity == "warning" for f in deg)  # never blocks honest reporting
    assert deg[0].detail["min_cluster_fraction"] < 0.01


def test_statistical_validator_silent_on_balanced_partition(ra, tmp_path: Path):
    # A genuinely separated partition must NOT be flagged — the rule layer only
    # surfaces objective degeneracy, never imposes a "good enough" threshold.
    ctx = _ctx_with_sofa(ra)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort, index=False)
    out_dir = _cluster_sizes_dir(tmp_path, [12000, 9000, 7500, 6000])
    step = ra.schema.AnalysisStep(
        step_id="01_phenotype_trajectory_clustering", intent="subphenotype clustering"
    )
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort, step=step, out_dir=out_dir, step_summary={},
    )
    assert not [f for f in findings if "degenerate" in f.message.lower()]


def test_statistical_validator_degeneracy_silent_without_cluster_evidence(ra, tmp_path: Path):
    # Absence of a cluster-size distribution is not degeneracy: a non-clustering
    # step must never trip this caution.
    ctx = _ctx_with_sofa(ra)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "primary_association.csv").write_text("variable,odds_ratio\nage,1.1\n", encoding="utf-8")
    step = ra.schema.AnalysisStep(step_id="04_primary_association", intent="association")
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort, step=step, out_dir=out_dir, step_summary={},
    )
    assert not [f for f in findings if "degenerate" in f.message.lower()]


def test_statistical_validator_flags_single_group_partition(ra, tmp_path: Path):
    ctx = _ctx_with_sofa(ra)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(cohort, index=False)
    out_dir = _cluster_sizes_dir(tmp_path, [38787])
    step = ra.schema.AnalysisStep(
        step_id="01_phenotype_trajectory_clustering", intent="subphenotype clustering"
    )
    findings = ra.StatisticalValidator().audit(
        context=ctx, cohort_path=cohort, step=step, out_dir=out_dir, step_summary={},
    )
    deg = [f for f in findings if "single-group" in f.message.lower() or "degenerate" in f.message.lower()]
    assert deg and all(f.severity == "warning" for f in deg)

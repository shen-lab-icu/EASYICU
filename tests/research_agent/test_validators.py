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
import pytest

from easyicu.research_agent.audits.validators import FigureSourceDataValidator
from easyicu.research_agent.schema import AnalysisStep


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


def test_figure_source_data_validator_accepts_source_row_index_trace(tmp_path: Path):
    parent = tmp_path / "steps" / "02_descriptive_results" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "group_label": ["Group A", "Group B"],
            "n": [100, 120],
            "event_n": [10, 24],
            "outcome_risk_pct": [10.0, 20.0],
        }
    ).to_csv(parent / "outcome_by_group.csv", index=False)

    out = tmp_path / "steps" / "02_descriptive_results_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "group_label": ["Group B"],
            "n": [120],
            "event_n": [24],
            "outcome_risk_pct": [20.0],
            "source_table": ["outcome_by_group.csv"],
            "source_row_index": [1],
        }
    ).to_csv(out / "figure_panel_source_data.csv", index=False)

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="02_descriptive_results_figure",
            intent="Render figure for step '02_descriptive_results'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )

    assert findings == []


def test_figure_source_data_validator_handles_shared_boolean_columns(
    tmp_path: Path,
):
    parent = tmp_path / "steps" / "05_sensitivity" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "term": ["primary"],
            "estimate": [1.2],
            "converged": [True],
        }
    ).to_csv(parent / "sensitivity_results.csv", index=False)

    out = tmp_path / "steps" / "05_sensitivity_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "term": ["primary"],
            "estimate": [1.2],
            "converged": [True],
            "source_table": ["sensitivity_results.csv"],
        }
    ).to_csv(out / "figure_panel_source_data.csv", index=False)

    findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_sensitivity_figure",
            intent="Render figure for step '05_sensitivity'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )

    assert findings == []


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


def test_concept_usage_ignores_lab_missingness_fraction_mean(ra):
    ctx = _ctx_with_sofa(ra)
    findings = ra.ConceptUsageAuditor().audit(
        context=ctx,
        script_text='missing_pct = df["lact"].isna().mean() * 100',
    )
    assert all("lact" not in f.message.lower() for f in findings)


def test_concept_usage_silences_generic_helper_with_median_and_mean(ra):
    ctx = _ctx_with_sofa(ra)
    code = """
def add_continuous(series):
    vals = series.dropna().astype(float)
    return {
        "median": vals.median(),
        "q25": vals.quantile(0.25),
        "q75": vals.quantile(0.75),
        "mean": vals.mean(),
    }
summary = add_continuous(df["lact"])
"""
    findings = ra.ConceptUsageAuditor().audit(context=ctx, script_text=code)
    assert all("lact" not in f.message.lower() for f in findings)


def test_concept_usage_flags_fillna_zero(ra):
    ctx = _ctx_with_sofa(ra)
    auditor = ra.ConceptUsageAuditor()
    code = "df['lact'] = df['lact'].fillna(0)"
    findings = auditor.audit(context=ctx, script_text=code)
    assert any("fillna" in f.message.lower() or "imputation" in f.message.lower()
               for f in findings)


def test_concept_usage_allows_boolean_mask_fillna_false(ra):
    ctx = _ctx_with_sofa(ra)
    code = """
mask = pd.to_numeric(df["age"], errors="coerce") >= 18
adult = df.loc[mask.fillna(False)].copy()
"""
    findings = ra.ConceptUsageAuditor().audit(context=ctx, script_text=code)
    assert not any(
        "fillna" in f.message.lower() or "imputation" in f.message.lower()
        for f in findings
    )


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


def _figure_source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    upstream = run_dir / "steps" / "05_sensitivity_comparison" / "outputs"
    figure = run_dir / "steps" / "05_sensitivity_comparison_figure" / "outputs"
    upstream.mkdir(parents=True)
    figure.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "spec_id": "primary_modeled_or",
                "effect_scale": "odds_ratio",
                "point_estimate": 1.07,
                "ci_low": 1.01,
                "ci_high": 1.13,
            },
            {
                "spec_id": "drop_lactate_modeled_or",
                "effect_scale": "odds_ratio",
                "point_estimate": 1.24,
                "ci_low": 1.18,
                "ci_high": 1.30,
            },
            {
                "spec_id": "primary_modeled_rd",
                "effect_scale": "risk_difference",
                "point_estimate": 0.005,
                "ci_low": 0.001,
                "ci_high": 0.009,
            },
        ]
    ).to_csv(upstream / "sensitivity_comparison.csv", index=False)
    return run_dir, figure


def test_figure_source_data_validator_accepts_upstream_subset(ra, tmp_path: Path):
    run_dir, figure_out = _figure_source_fixture(tmp_path)
    pd.DataFrame(
        [
            {
                "spec_id": "primary_modeled_or",
                "effect_scale": "odds_ratio",
                "point_estimate": 1.07,
                "ci_low": 1.01,
                "ci_high": 1.13,
            },
            {
                "spec_id": "primary_modeled_rd",
                "effect_scale": "risk_difference",
                "point_estimate": 0.005,
                "ci_low": 0.001,
                "ci_high": 0.009,
            },
        ]
    ).to_csv(figure_out / "sensitivity_forest_source_data.csv", index=False)

    step = ra.schema.AnalysisStep(
        step_id="05_sensitivity_comparison_figure",
        intent="Render the publication figure declared by step '05_sensitivity_comparison'.",
    )
    findings = ra.FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )
    assert findings == []


def test_figure_source_data_validator_accepts_definition_id_key(ra, tmp_path: Path):
    run_dir = tmp_path / "run"
    upstream = (
        run_dir
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap"
        / "outputs"
    )
    figure = (
        run_dir
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap_figure"
        / "outputs"
    )
    upstream.mkdir(parents=True)
    figure.mkdir(parents=True)
    pd.DataFrame(
        {
            "definition_id": ["primary", "relax_temp"],
            "n_included": [100, 112],
            "moved_in_vs_primary_n": [0, 12],
        }
    ).to_csv(upstream / "alternative_cohort_attrition.csv", index=False)
    pd.DataFrame(
        {
            "definition_id": ["primary", "relax_temp"],
            "n_included": [100, 112],
            "moved_in_vs_primary_n": [0, 12],
        }
    ).to_csv(figure / "publication_figure_definition_source_data.csv", index=False)

    findings = ra.FigureSourceDataValidator().audit(
        step=ra.schema.AnalysisStep(
            step_id="04_alternative_eligibility_definitions_and_overlap_figure",
            intent=(
                "Render the publication figure declared by step "
                "'04_alternative_eligibility_definitions_and_overlap'."
            ),
        ),
        out_dir=figure,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )
    assert findings == []


def test_figure_source_data_validator_accepts_pairwise_definition_key(
    ra,
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    upstream = (
        run_dir
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap"
        / "outputs"
    )
    figure = (
        run_dir
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap_figure"
        / "outputs"
    )
    upstream.mkdir(parents=True)
    figure.mkdir(parents=True)
    pd.DataFrame(
        {
            "definition_a": ["primary", "primary", "relax_temp"],
            "definition_b": ["primary", "relax_temp", "primary"],
            "intersection_n": [100, 100, 100],
            "jaccard": [1.0, 0.893, 0.893],
        }
    ).to_csv(upstream / "cohort_overlap_matrix.csv", index=False)
    pd.DataFrame(
        {
            "definition_a": ["primary", "relax_temp"],
            "definition_b": ["relax_temp", "primary"],
            "intersection_n": [100, 100],
            "jaccard": [0.893, 0.893],
        }
    ).to_csv(figure / "publication_figure_overlap_source_data.csv", index=False)

    findings = ra.FigureSourceDataValidator().audit(
        step=ra.schema.AnalysisStep(
            step_id="04_alternative_eligibility_definitions_and_overlap_figure",
            intent=(
                "Render the publication figure declared by step "
                "'04_alternative_eligibility_definitions_and_overlap'."
            ),
        ),
        out_dir=figure,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )
    assert findings == []


def test_figure_source_data_validator_blocks_resume_evidence_pollution(
    ra,
    tmp_path: Path,
):
    run_dir, figure_out = _figure_source_fixture(tmp_path)
    pd.DataFrame(
        [
            {
                "spec_id": "primary_modeled_or",
                "effect_scale": "odds_ratio",
                "point_estimate": 1.07,
                "ci_low": 1.01,
                "ci_high": 1.13,
            },
            {
                "spec_id": "alt_cohort_from_old_robustness_panel",
                "effect_scale": "odds_ratio",
                "point_estimate": 1.03,
                "ci_low": 0.95,
                "ci_high": 1.11,
            },
        ]
    ).to_csv(figure_out / "sensitivity_forest_source_data.csv", index=False)

    step = ra.schema.AnalysisStep(
        step_id="05_sensitivity_comparison_figure",
        intent="Render the publication figure declared by step '05_sensitivity_comparison'.",
    )
    findings = ra.FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )
    assert any(f.severity == "error" for f in findings), findings
    assert "absent" in findings[0].message.lower() or (
        findings[0].detail
        and findings[0].detail["best_mismatch"]["reason"]
        == "source_rows_not_in_upstream"
    )


def test_figure_source_data_validator_blocks_numeric_drift(ra, tmp_path: Path):
    run_dir, figure_out = _figure_source_fixture(tmp_path)
    pd.DataFrame(
        [
            {
                "spec_id": "primary_modeled_or",
                "effect_scale": "odds_ratio",
                "point_estimate": 9.99,
                "ci_low": 1.01,
                "ci_high": 1.13,
            }
        ]
    ).to_csv(figure_out / "sensitivity_forest_source_data.csv", index=False)

    step = ra.schema.AnalysisStep(
        step_id="05_sensitivity_comparison_figure",
        intent="Render the publication figure declared by step '05_sensitivity_comparison'.",
    )
    findings = ra.FigureSourceDataValidator().audit(
        step=step,
        out_dir=figure_out,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )
    assert any(
        f.severity == "error"
        and f.detail
        and f.detail["best_mismatch"]["reason"] == "source_values_disagree"
        for f in findings
    ), findings


def test_figure_source_data_validator_blocks_inconsistent_percent_counts(
    ra,
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    upstream = (
        run_dir
        / "steps"
        / "02_baseline_characteristics_and_data_quality"
        / "outputs"
    )
    figure = (
        run_dir
        / "steps"
        / "02_baseline_characteristics_and_data_quality_figure"
        / "outputs"
    )
    upstream.mkdir(parents=True)
    figure.mkdir(parents=True)
    pd.DataFrame(
        {
            "variable": ["resp_max"],
            "missing_pct": [0.2512394927],
            "missing_n": [188],
            "total_n": [74829],
        }
    ).to_csv(upstream / "missingness_measurement_panel_source_data.csv", index=False)
    pd.DataFrame(
        {
            "variable": ["resp_max"],
            "missing_pct": [25.12394927],
            "missing_n": [188],
            "total_n": [74829],
        }
    ).to_csv(figure / "missingness_measurement_panel_source_data.csv", index=False)

    findings = ra.FigureSourceDataValidator().audit(
        step=ra.schema.AnalysisStep(
            step_id="02_baseline_characteristics_and_data_quality_figure",
            intent=(
                "Render the publication figure declared by step "
                "'02_baseline_characteristics_and_data_quality'."
            ),
        ),
        out_dir=figure,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )

    assert any(
        f.severity == "error"
        and "100*missing_n/total_n" in f.message
        and f.detail
        and f.detail["expected_pct"] == pytest.approx(0.2512394927)
        for f in findings
    ), findings


def test_figure_source_data_validator_accepts_derived_missingness_source_data(
    ra,
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    upstream = (
        run_dir
        / "steps"
        / "02_baseline_characteristics_and_data_quality"
        / "outputs"
    )
    figure = (
        run_dir
        / "steps"
        / "02_baseline_characteristics_and_data_quality_figure"
        / "outputs"
    )
    upstream.mkdir(parents=True)
    figure.mkdir(parents=True)
    pd.DataFrame(
        {
            "concept": ["resp", "lact"],
            "label": ["Respiratory rate", "Lactate"],
            "n_total": [74829, 74829],
            "value_missing_n": [188, 30490],
            "value_missing_pct": [0.2512394927100456, 40.74623474856005],
            "measured_one_n": [74641, 44339],
            "measured_one_pct": [99.74876050728996, 59.25376525143995],
        }
    ).to_csv(upstream / "missingness_measurement_audit.csv", index=False)
    pd.DataFrame(
        {
            "variable": ["resp", "lact"],
            "concept": ["resp", "lact"],
            "label": ["Respiratory rate", "Lactate"],
            "display_label": ["Respiratory rate", "Lactate"],
            "missing_pct": [0.2512394927100456, 40.74623474856005],
            "missing_n": [188, 30490],
            "total_n": [74829, 74829],
            "value_missing_pct": [0.2512394927100456, 40.74623474856005],
            "value_missing_n": [188, 30490],
            "n_total": [74829, 74829],
            "measured_pct": [99.74876050728996, 59.25376525143995],
            "measured_n": [74641, 44339],
            "measured_one_pct": [99.74876050728996, 59.25376525143995],
            "measured_one_n": [74641, 44339],
            "source_table": ["missingness_measurement_audit.csv"] * 2,
            "source_transform": ["missingness_measurement_summary_v1"] * 2,
        }
    ).to_csv(figure / "missingness_measurement_panel_source_data.csv", index=False)

    findings = ra.FigureSourceDataValidator().audit(
        step=ra.schema.AnalysisStep(
            step_id="02_baseline_characteristics_and_data_quality_figure",
            intent=(
                "Render the publication figure declared by step "
                "'02_baseline_characteristics_and_data_quality'."
            ),
        ),
        out_dir=figure,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )

    assert findings == []


def test_figure_contract_quality_blocks_rescue_publication_contract(ra, tmp_path: Path):
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "04_primary_association_figure" / "outputs"
    out_dir.mkdir(parents=True)
    contract_path = out_dir / "publication_figure.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "publication_figure",
                "core_claim": "Adjusted odds ratios are summarised from source data.",
                "statistics_note": (
                    "Deterministic rescue figure generated when the figure-only "
                    "child step did not emit exports."
                ),
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Odds-ratio forest plot",
                        "role": "relationship",
                        "claim": "Adjusted odds ratios and 95% intervals are plotted.",
                        "evidence_ids": ["table_association_results"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    step = ra.schema.AnalysisStep(
        step_id="04_primary_association_figure",
        intent="Render the manuscript publication figure.",
        method="figure rendering",
    )

    findings = ra.FigureContractQualityValidator().audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )

    assert any(
        f.severity == "error" and "fallback/rescue" in f.message.lower()
        for f in findings
    ), findings


def test_figure_contract_quality_requires_contract_for_figure_exports(ra, tmp_path: Path):
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "04_primary_association_figure" / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / "effect_estimate_forest.png").write_bytes(b"fake-png")
    step = ra.schema.AnalysisStep(
        step_id="04_primary_association_figure",
        intent="Render the manuscript publication figure.",
        method="figure rendering",
    )

    findings = ra.FigureContractQualityValidator().audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary={"rendering_only": True},
    )

    assert any(
        f.severity == "error" and "without a .figure_contract.json" in f.message
        for f in findings
    ), findings


def test_figure_contract_quality_blocks_single_panel_result_contract(ra, tmp_path: Path):
    contract_path = tmp_path / "easyicu_publication_figure.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "easyicu_publication_figure",
                "core_claim": "Primary effect and robustness range are shown.",
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Primary effect and robustness variants",
                        "role": "robustness",
                        "claim": "Primary and robustness estimates are plotted.",
                        "evidence_ids": ["robustness_panel"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    findings = ra.FigureContractQualityValidator().audit_contract_file(
        contract_path,
        manuscript_facing=True,
    )

    assert any(
        f.severity == "error" and "only 1 panel" in f.message
        for f in findings
    ), findings


def test_figure_contract_quality_accepts_multipanel_result_contract(ra, tmp_path: Path):
    contract_path = tmp_path / "easyicu_publication_figure.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "easyicu_publication_figure",
                "core_claim": "Primary effect, robustness, and denominator context are shown.",
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Primary effect and robustness variants",
                        "role": "robustness",
                        "claim": "Primary and variant estimates are shown with intervals.",
                        "evidence_ids": ["robustness_panel"],
                    },
                    {
                        "panel_id": "B",
                        "title": "Variant convergence by axis",
                        "role": "validation",
                        "claim": "Converged and non-converged variants are counted.",
                        "evidence_ids": ["robustness_panel"],
                    },
                    {
                        "panel_id": "C",
                        "title": "Analytic sample-size range",
                        "role": "audit",
                        "claim": "Sample-size ranges are shown for denominator context.",
                        "evidence_ids": ["robustness_panel"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    findings = ra.FigureContractQualityValidator().audit_contract_file(
        contract_path,
        manuscript_facing=True,
    )

    assert not any(f.severity == "error" for f in findings), findings


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


def test_clinical_constraint_validator_does_not_flag_association_named_exposure(
    ra, tmp_path: Path
):
    ctx = _ctx_with_sofa(ra).model_copy(
        update={
            "research_question": "Is Sepsis-3 status associated with ICU mortality?",
            "user_preferences": ra.schema.UserPreferences(
                inferred_analysis_family="association"
            ),
        }
    )
    step = ra.schema.AnalysisStep(
        step_id="03b_dataset_validation",
        intent="Validate the modeling dataset and named exposure before regression.",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    findings = ra.ClinicalConstraintValidator().audit(
        context=ctx,
        step=step,
        out_dir=out_dir,
        step_summary={
            "named_exposure": "sepsis3",
            "method": "post_audit_modeling_dataset_validation_and_repair",
        },
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

"""Deterministic estimator adapter tests for robustness panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _synthetic_binary_frame(n: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    logit = -0.3 + np.log(1.8) * x
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    y_alt_1 = rng.binomial(1, np.clip(p + 0.02, 0.01, 0.99))
    y_alt_2 = rng.binomial(1, np.clip(p - 0.02, 0.01, 0.99))
    return pd.DataFrame({"x": x, "y": y, "y_alt_1": y_alt_1, "y_alt_2": y_alt_2})


def _adapter_records(df: pd.DataFrame):
    return [
        {
            "step_id": "01_model",
            "status": "ok",
            "step_summary_evidence_id": "stat_model",
            "step_summary": {
                "primary_predictor": "x",
                "primary_or": 1.8,
                "primary_ci_low": 1.5,
                "primary_ci_high": 2.1,
                "n_total": int(len(df)),
                "estimator_adapter": {
                    "data": df.to_dict("records"),
                    "exposure": "x",
                    "outcome": "y",
                    "estimator_kind": "logistic",
                    "missing_strategy": "complete_case",
                    "outcome_columns": {
                        "author_defined_outcome_1": "y_alt_1",
                        "author_defined_outcome_2": "y_alt_2",
                    },
                }
            },
        }
    ]


def test_logistic_estimator_recovers_known_odds_ratio() -> None:
    from easyicu.research_agent.robustness.estimators import fit_estimator

    df = _synthetic_binary_frame(n=5000, seed=10)
    result = fit_estimator(cohort=None, X=df[["x"]], y=df["y"], kind="logistic")

    assert result.converged
    assert result.point_estimate is not None
    assert 1.6 <= result.point_estimate <= 2.1


def test_complete_case_and_mean_imputation_can_differ() -> None:
    from easyicu.research_agent.robustness.estimators import fit_estimator
    from easyicu.research_agent.methods.missing import apply_missing_strategy

    df = _synthetic_binary_frame(n=300, seed=4)
    df.loc[df["x"] > 0.8, "x"] = np.nan
    cc = apply_missing_strategy(df[["x", "y"]], "complete_case")
    mean_imp = apply_missing_strategy(df[["x", "y"]], "mean_imputation")

    cc_result = fit_estimator(cohort=None, X=cc[["x"]], y=cc["y"], kind="logistic")
    mean_result = fit_estimator(
        cohort=None,
        X=mean_imp[["x"]],
        y=mean_imp["y"],
        kind="logistic",
    )

    assert cc_result.converged
    assert mean_result.converged
    assert cc_result.point_estimate != mean_result.point_estimate


def test_non_convergence_is_captured_not_raised() -> None:
    from easyicu.research_agent.robustness.estimators import fit_estimator

    X = pd.DataFrame({f"x{i}": [0, 1, 0, 1, 0] for i in range(10)})
    y = pd.Series([0, 1, 0, 1, 0])
    result = fit_estimator(cohort=None, X=X, y=y, kind="logistic")

    assert not result.converged
    assert "sample size too small" in result.notes


def test_adapter_builds_full_eight_row_panel_and_registers_claims(ra, tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from easyicu.research_agent.robustness.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness.panel import (
        build_robustness_panel_from_records,
        default_robustness_specs,
        write_locked_robustness_specs,
        write_robustness_panel,
    )

    specs = default_robustness_specs()
    records = _adapter_records(_synthetic_binary_frame(n=900))
    rows, warnings = fit_robustness_rows_from_records(
        specs=specs,
        per_step_records=records,
        allow_implicit_cohort_refit=True,
    )
    evidence = ra.EvidenceStore(tmp_path)
    source_summary = tmp_path / "adapter_step_summary.json"
    summary_payload = dict(records[0]["step_summary"])
    summary_payload["robustness_rows"] = [
        row.to_dict() for row in rows if row.spec_id != "primary"
    ]
    source_summary.write_text(json.dumps(summary_payload), encoding="utf-8")
    evidence.register_file(
        kind="statistic",
        description="Digest-bound estimator-adapter step summary.",
        source_path=source_summary,
        produced_by_step="01_model",
        evidence_id="stat_model",
    )
    write_locked_robustness_specs(
        run_dir=tmp_path,
        plan=SimpleNamespace(robustness_specs=specs),
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )
    panel = build_robustness_panel_from_records(
        specs=specs,
        per_step_records=records,
        adapter_rows=rows,
    )
    write_robustness_panel(
        run_dir=tmp_path,
        panel=panel,
        evidence=evidence,
        prompt_pack_version="test",
    )

    assert warnings == []
    assert len(panel.rows) == 8
    assert panel.n_variants == 7
    assert all(row.converged for row in panel.rows)
    claim_fields = {claim.source_field for claim in evidence.numeric_claims()}
    assert "primary_point_estimate" in claim_fields
    assert "row_primary_point_estimate" not in claim_fields
    assert "range_low" in claim_fields
    assert "range_high" in claim_fields
    assert f"row_{specs[0].spec_id}_point_estimate" not in claim_fields


def test_step_owned_rows_prevent_adapter_refit_with_warning() -> None:
    from easyicu.research_agent.robustness.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness.panel import (
        build_robustness_panel_from_records,
        default_robustness_specs,
    )

    specs = default_robustness_specs()
    records = _adapter_records(_synthetic_binary_frame(n=500))
    records[0]["step_summary"]["robustness_rows"] = [
        {
            "spec_id": specs[0].spec_id,
            "axis": specs[0].axis,
            "n": 1,
            "point_estimate": 99.0,
            "ci_low": 98.0,
            "ci_high": 100.0,
            "se": 1.0,
            "converged": True,
        }
    ]

    rows, warnings = fit_robustness_rows_from_records(
        specs=specs,
        per_step_records=records,
        allow_implicit_cohort_refit=True,
    )
    panel = build_robustness_panel_from_records(
        specs=specs,
        per_step_records=records,
        adapter_rows=rows,
    )

    assert any(specs[0].spec_id in warning for warning in warnings)
    assert all(row.spec_id != specs[0].spec_id for row in rows)
    owned_row = next(row for row in panel.rows if row.spec_id == specs[0].spec_id)
    assert owned_row.point_estimate == 99.0


# ---------------------------------------------------------------------------
# A rank-deficient locked design must fail closed rather than silently dropping
# declared variables and changing the adjustment set.
# ---------------------------------------------------------------------------


def test_fit_estimator_blocks_rank_deficient_locked_design():
    from easyicu.research_agent.robustness.estimators import fit_estimator

    rng = np.random.default_rng(0)
    n = 1500
    sepsis3 = rng.binomial(1, 0.4, n)
    age = rng.normal(65, 15, n)
    p = 1 / (1 + np.exp(-(-2.0 + 0.8 * sepsis3 + 0.01 * age)))
    y = rng.binomial(1, p)
    X = pd.DataFrame(
        {
            "sepsis3": sepsis3,
            "age": age,
            "lact_measured": np.ones(n),  # constant -> zero variance
            "age_dup": age * 1.0,  # perfectly collinear with age
        }
    )

    # Plain MLE on this design is singular.
    import statsmodels.api as sm

    try:
        sm.Logit(y, sm.add_constant(X.astype(float))).fit(disp=0)
        raised = False
    except Exception:
        raised = True
    assert raised, "design should be singular for a plain fit"

    result = fit_estimator(cohort=None, X=X, y=pd.Series(y), kind="logistic")
    assert result.converged is False
    assert result.point_estimate is None
    assert "refusing to drop declared predictors" in result.notes
    assert "lact_measured" in result.notes and "age_dup" in result.notes


def test_fit_estimator_preserves_rows_after_filtered_nonconsecutive_index():
    from easyicu.research_agent.robustness.estimators import fit_estimator

    rng = np.random.default_rng(20260711)
    n = 240
    x = rng.normal(size=n)
    probability = 1 / (1 + np.exp(-(-0.4 + 0.8 * x)))
    y = (rng.random(n) < probability).astype(int)
    retained_index = np.arange(n) * 3 + 7
    X = pd.DataFrame({"x": x}, index=retained_index)
    outcome = pd.Series(y, index=retained_index)

    result = fit_estimator(cohort=None, X=X, y=outcome, kind="logistic")

    assert result.converged is True
    assert result.n == n


def test_robust_design_keeps_exposure_and_const():
    from easyicu.research_agent.robustness.estimators import _robust_design

    x = pd.DataFrame(
        {
            "const": np.ones(50),
            "sepsis3": np.r_[np.zeros(25), np.ones(25)],
            "flag": np.ones(50),  # constant -> dropped
            "sepsis3_copy": np.r_[np.zeros(25), np.ones(25)],  # collinear -> dropped
        }
    )
    reduced, dropped = _robust_design(x, keep=["const", "sepsis3"])
    assert list(reduced.columns) == ["const", "sepsis3"]
    assert set(dropped) == {"flag", "sepsis3_copy"}

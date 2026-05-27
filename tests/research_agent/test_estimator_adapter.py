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
    y_db = rng.binomial(1, np.clip(p + 0.02, 0.01, 0.99))
    y_28 = rng.binomial(1, np.clip(p - 0.02, 0.01, 0.99))
    return pd.DataFrame({"x": x, "y": y, "y_db": y_db, "y_28": y_28})


def _adapter_records(df: pd.DataFrame):
    return [
        {
            "step_id": "01_model",
            "step_summary_evidence_id": "stat_model",
            "step_summary": {
                "estimator_adapter": {
                    "data": df.to_dict("records"),
                    "exposure": "x",
                    "outcome": "y",
                    "estimator_kind": "logistic",
                    "missing_strategy": "complete_case",
                    "outcome_columns": {
                        "database_specific_mortality": "y_db",
                        "28_day_mortality_if_available": "y_28",
                    },
                }
            },
        }
    ]


def test_logistic_estimator_recovers_known_odds_ratio() -> None:
    from easyicu.research_agent.estimators import fit_estimator

    df = _synthetic_binary_frame(n=5000, seed=10)
    result = fit_estimator(cohort=None, X=df[["x"]], y=df["y"], kind="logistic")

    assert result.converged
    assert result.point_estimate is not None
    assert 1.6 <= result.point_estimate <= 2.1


def test_complete_case_and_mean_imputation_can_differ() -> None:
    from easyicu.research_agent.estimators import fit_estimator
    from easyicu.research_agent.missing import apply_missing_strategy

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
    from easyicu.research_agent.estimators import fit_estimator

    X = pd.DataFrame({f"x{i}": [0, 1, 0, 1, 0] for i in range(10)})
    y = pd.Series([0, 1, 0, 1, 0])
    result = fit_estimator(cohort=None, X=X, y=y, kind="logistic")

    assert not result.converged
    assert "sample size too small" in result.notes


def test_adapter_builds_full_eight_row_panel_and_registers_claims(ra, tmp_path) -> None:
    from easyicu.research_agent.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness_panel import (
        build_robustness_panel_from_records,
        default_robustness_specs,
        write_robustness_panel,
    )

    specs = default_robustness_specs()
    records = _adapter_records(_synthetic_binary_frame(n=900))
    rows, warnings = fit_robustness_rows_from_records(
        specs=specs,
        per_step_records=records,
    )
    panel = build_robustness_panel_from_records(
        specs=specs,
        per_step_records=records,
        adapter_rows=rows,
    )
    evidence = ra.EvidenceStore(tmp_path)
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
    assert "row_primary_point_estimate" in claim_fields
    assert f"row_{specs[0].spec_id}_point_estimate" in claim_fields


def test_adapter_rows_override_coder_rows_with_warning() -> None:
    from easyicu.research_agent.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness_panel import default_robustness_specs

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
    )

    assert any(specs[0].spec_id in warning for warning in warnings)
    adapter_row = next(row for row in rows if row.spec_id == specs[0].spec_id)
    assert adapter_row.point_estimate != 99.0

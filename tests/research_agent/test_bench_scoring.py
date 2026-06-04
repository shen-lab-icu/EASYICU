from __future__ import annotations

import json
from pathlib import Path

from tools.run_research_agent_bench import _artifact_substring_hits, _primary_or


def _write_summary(run_dir: Path, payload: dict) -> None:
    step_dir = run_dir / "steps" / "01_primary" / "outputs"
    step_dir.mkdir(parents=True)
    (step_dir / "step_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_panel(run_dir: Path, value: float) -> None:
    (run_dir / "robustness_panel.json").write_text(
        json.dumps(
            {
                "primary_spec_id": "primary",
                "rows": [
                    {
                        "spec_id": "primary",
                        "axis": "primary",
                        "point_estimate": value,
                        "converged": True,
                    }
                ],
                "primary_point_estimate": value,
            }
        ),
        encoding="utf-8",
    )


def test_artifact_substring_hits_scan_evidence_record_fields() -> None:
    manifest = {
        "evidence": [
            {
                "evidence_id": "table_one__summary",
                "description": "Table one baseline characteristics.",
                "relative_path": "evidence/table_one__summary.csv",
                "kind": "table",
            }
        ]
    }

    assert _artifact_substring_hits(manifest, ["table_one"]) == {"table_one": True}


def test_primary_or_accepts_nested_logistic_model_type(tmp_path: Path) -> None:
    _write_summary(
        tmp_path,
        {
            "method": None,
            "primary_model": {"model_type": "logistic_regression"},
            "primary_or": 1.1366224560031324,
        },
    )

    assert _primary_or(tmp_path, expected_predictor="sofa2") == 1.1366224560031324


def test_primary_or_leaves_non_logistic_models_unscored(tmp_path: Path) -> None:
    _write_summary(
        tmp_path,
        {
            "method": None,
            "primary_model": {"model_type": "linear_regression"},
            "primary_or": 1.1366224560031324,
        },
    )

    assert _primary_or(tmp_path, expected_predictor="sofa2") is None


def test_primary_or_prefers_manuscript_facing_panel_primary(
    tmp_path: Path,
) -> None:
    """Q1-style run: panel headline wins over trend and dummy contrast."""
    _write_panel(tmp_path, 1.0184219783832567)
    _write_summary(
        tmp_path,
        {
            "primary_association_estimate": 0.09303264484823856,
            "primary_association_term": "sofa2==1.0",
            "primary_analysis": {"model_type": "logistic_regression"},
            "core_complete_case_model": {
                "fit_method": "Logit(lbfgs)",
                "primary_or": 0.09303264484823856,
            },
            "sofa2_numeric_trend_model": {
                "fit_method": "Logit(lbfgs)",
                "sofa2_or_per_point": 1.2538866781554125,
            },
        },
    )

    assert (
        _primary_or(
            tmp_path,
            expected_predictor="sofa2",
            item_key="analysis_sofa_multisignal_mortality__miiv",
        )
        == 1.0184219783832567
    )


def test_primary_or_skips_continuous_predictor_dummy_level_contrast(
    tmp_path: Path,
) -> None:
    _write_summary(
        tmp_path,
        {
            "primary_association_estimate": 0.09303264484823856,
            "primary_association_term": "sofa2==1.0",
            "primary_analysis": {"model_type": "logistic_regression"},
            "core_complete_case_model": {
                "fit_method": "Logit(lbfgs)",
                "primary_or": 0.09303264484823856,
            },
        },
    )

    assert _primary_or(tmp_path, expected_predictor="sofa2") is None


def test_primary_or_allows_binary_predictor_level_contrast(
    tmp_path: Path,
) -> None:
    _write_summary(
        tmp_path,
        {
            "primary_association_estimate": 1.74,
            "primary_association_term": "vaso==1",
            "primary_analysis": {"model_type": "logistic_regression"},
        },
    )

    assert _primary_or(tmp_path, expected_predictor="vaso") == 1.74


def test_primary_or_leaves_non_or_benchmark_unscored(tmp_path: Path) -> None:
    _write_panel(tmp_path, 1.0184219783832567)
    _write_summary(
        tmp_path,
        {
            "method": "logistic_regression",
            "primary_or": 1.0184219783832567,
        },
    )

    assert (
        _primary_or(
            tmp_path,
            expected_predictor="sofa2",
            item_key="analysis_sofa2_time_to_mortality_cox__miiv",
        )
        is None
    )

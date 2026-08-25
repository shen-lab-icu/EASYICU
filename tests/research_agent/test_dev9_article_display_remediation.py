from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.figure2_canonical9 import (
    render_dev9_article_display_remediation as renderer,
)


def _write_h2_source(run_dir: Path, *, effect_estimate: float | None = None) -> Path:
    output_dir = run_dir / renderer.H2_FEASIBILITY_RELATIVE.parent
    output_dir.mkdir(parents=True)
    path = output_dir / renderer.H2_FEASIBILITY_RELATIVE.name
    pd.DataFrame(
        [
            {
                "source": "typed_vasopressor_source",
                "window_start_hours": 0,
                "window_end_hours": 24,
                "verified_non_use_available": False,
                "binary_control_arm_authorized": False,
                "causal_contrast_authorized": False,
                "decision": "fail_closed",
                "reason_code": "H2_VERIFIED_NON_USE_UNAVAILABLE",
                "effect_estimate": effect_estimate,
            }
        ]
    ).to_csv(path, index=False)
    return path


def test_copy_source_preserves_nested_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    frame = pd.DataFrame(
        [{"source_row_index": 7, "source_file": "upstream.csv", "value": 3.0}]
    )
    frame.to_csv(source, index=False)

    output = tmp_path / "copied.csv"
    renderer._copy_source(frame, source, output)
    copied = pd.read_csv(output)

    assert copied.loc[0, "source_row_index"] == 0
    assert copied.loc[0, "upstream_source_row_index"] == 7
    assert copied.loc[0, "upstream_source_file"] == "upstream.csv"
    assert copied.loc[0, "source_sha256"] == renderer._sha256(source)


def test_h2_renderer_emits_only_supplementary_fail_closed_figure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_h2_source(run_dir)

    summary = renderer._render_h2(run_dir, tmp_path / "out")

    assert summary["main_figure_count"] == 0
    assert summary["supplementary_figure_count"] == 1
    assert summary["scientific_status"] == "failed_closed"
    contract_path = (
        tmp_path
        / "out"
        / "h2_supplementary_figure_s1_fail_closed_feasibility.figure_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["panels"][0]["metadata"]["placement"] == "supplementary"
    assert "No effect estimate" in contract["statistics_note"] or contract["panels"][0][
        "claim"
    ].endswith("no effect estimate exists.")


def test_h2_renderer_rejects_an_effect_estimate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_h2_source(run_dir, effect_estimate=0.8)

    with pytest.raises(ValueError, match="unauthorized causal result"):
        renderer._render_h2(run_dir, tmp_path / "out")


def test_h1_renderer_adds_prespecified_time_varying_main_figure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    km_rows = []
    for group, initial in ((0, 500), (1, 350)):
        for day, survival, at_risk in (
            (0, 1.0, initial),
            (7, 0.97, initial - 20),
            (14, 0.94, initial - 40),
            (21, 0.91, initial - 60),
            (27, 0.88, initial - 80),
        ):
            km_rows.append(
                {
                    "exposure_group": group,
                    "time_from_landmark_days": day,
                    "survival_probability": survival - group * 0.02,
                    "at_risk": at_risk,
                    "group_n": initial,
                    "group_events": 80,
                }
            )
    pd.DataFrame(km_rows).to_csv(source / "landmark_km_curve.csv", index=False)
    pd.DataFrame(
        [
            {
                "term": "incident_ventilation_by_24h",
                "hazard_ratio": 1.3,
                "ci_low": 1.1,
                "ci_high": 1.5,
            }
        ]
    ).to_csv(source / "landmark_cox_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "stage_order": index,
                "stage": stage,
                "count": 1_000 - 100 * index,
                "source_denominator": 1_000,
                "percent_of_source": 100 - 10 * index,
            }
            for index, stage in enumerate(
                (
                    "source_rows",
                    "valid_fixed_horizon_endpoint",
                    "alive_and_observed_at_landmark",
                    "exposure_status_and_timing_supported",
                    "landmark_analysis_population",
                ),
                start=1,
            )
        ]
    ).to_csv(source / "landmark_risk_set_flow.csv", index=False)
    pd.DataFrame(
        [
            {
                "covariate": "global",
                "p_value": 0.001,
                "declared_alpha": 0.05,
            }
        ]
    ).to_csv(source / "landmark_ph_diagnostics.csv", index=False)
    pd.DataFrame(
        [
            {
                "tau_days_from_landmark": 27,
                "exposed_rmst_days": 24.0,
                "exposed_rmst_ci_low": 23.5,
                "exposed_rmst_ci_high": 24.5,
                "comparator_rmst_days": 25.0,
                "comparator_rmst_ci_low": 24.6,
                "comparator_rmst_ci_high": 25.4,
                "rmst_difference_days": -1.0,
                "ci_low": -1.5,
                "ci_high": -0.5,
            }
        ]
    ).to_csv(source / "landmark_rmst_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "term": "incident_ventilation_by_24h",
                "is_exposure": True,
                "interval_index": index,
                "interval_start_days": start,
                "interval_end_days": stop,
                "hazard_ratio": estimate,
                "ci_low": estimate - 0.15,
                "ci_high": estimate + 0.15,
            }
            for index, (start, stop, estimate) in enumerate(
                ((0, 7, 1.4), (7, 14, 1.2), (14, 27, 1.05)), start=1
            )
        ]
    ).to_csv(source / "landmark_time_varying_cox_summary.csv", index=False)

    summary = renderer._render_h1(source, tmp_path / "out")

    assert summary["main_figure_count"] == 4
    assert summary["reason_code"] == (
        "H1_CONSTANT_HR_WITHHELD_TIME_VARYING_SENSITIVITY_AVAILABLE"
    )
    contract = json.loads(
        (
            tmp_path
            / "out"
            / "h1_main_figure_3_time_varying_association.figure_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["panels"][0]["metadata"]["placement"] == "main"
    assert contract["panels"][0]["metadata"]["chart_type"] == (
        "time_varying_hazard_ratio_forest"
    )

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


def test_article_table_packaging_preserves_frozen_digest_and_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "source" / "run"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    source = evidence_dir / "table_step_artifact_abc__table_one.csv"
    source.write_text("name,value\nAge,63\n", encoding="utf-8")
    index = [
        {
            "evidence_id": "table_step_artifact_abc",
            "kind": "table",
            "relative_path": "evidence/table_step_artifact_abc__table_one.csv",
            "sha256": renderer._sha256(source),
            "produced_by_step": "baseline_context",
        }
    ]
    (evidence_dir / "evidence_index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    monkeypatch.setattr(renderer, "RUN_RELATIVES", {"x": Path("run")})
    monkeypatch.setattr(
        renderer,
        "ARTICLE_TABLE_SPECS",
        {
            "x": (
                (
                    "table_1_cohort_characteristics",
                    "main",
                    "baseline_context",
                    "table_one.csv",
                    "Cohort characteristics",
                ),
            )
        },
    )
    monkeypatch.setattr(
        renderer,
        "ARTICLE_TABLE_ROLES",
        {("x", "table_1_cohort_characteristics"): "baseline_context"},
    )

    summary = renderer._package_article_tables(
        source_root=tmp_path / "source", output_root=tmp_path / "out"
    )

    assert summary["x"]["main_table_count"] == 1
    packaged = tmp_path / "out/x/table_1_cohort_characteristics.csv"
    assert renderer._sha256(packaged) == renderer._sha256(source)
    contract = json.loads(
        (
            tmp_path / "out/x/table_1_cohort_characteristics.table_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["placement"] == "main"
    assert contract["display_purpose"] == "context"
    assert contract["article_role"] == "baseline_context"
    assert contract["upstream_evidence_id"] == "table_step_artifact_abc"
    assert contract["paper_authorization_allowed"] is False


def test_h2_renderer_emits_main_diagnostic_without_an_effect(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_h2_source(run_dir)

    summary = renderer._render_h2(run_dir, tmp_path / "out")

    assert summary["main_figure_count"] == 1
    assert summary["supplementary_figure_count"] == 0
    assert summary["scientific_status"] == "failed_closed"
    contract_path = (
        tmp_path
        / "out"
        / "h2_main_figure_1_causal_identifiability_diagnostic.figure_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["panels"][0]["metadata"]["placement"] == "main"
    assert contract["panels"][0]["metadata"]["display_purpose"] == "diagnostic"
    assert "No effect estimate" in contract["statistics_note"] or contract["panels"][0][
        "claim"
    ].endswith("no effect estimate exists.")
    table_id, placement, *_ = renderer.ARTICLE_TABLE_SPECS["h2"][0]
    assert (table_id, placement) == ("table_1_causal_identifiability", "main")
    assert renderer.ARTICLE_TABLE_ROLES[("h2", table_id)] == "diagnostics"


def test_h2_renderer_rejects_an_effect_estimate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_h2_source(run_dir, effect_estimate=0.8)

    with pytest.raises(ValueError, match="unauthorized causal result"):
        renderer._render_h2(run_dir, tmp_path / "out")


def test_h3_renderer_separates_main_selection_diagnostic_from_missingness_audit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        [
            {
                "n_clusters": n_clusters,
                "bic": float(1_000 - n_clusters * 100),
                "aic": float(950 - n_clusters * 100),
                "selected": n_clusters == 6,
                "upper_boundary": n_clusters == 6,
                "scientific_status": "failed_closed",
            }
            for n_clusters in range(2, 7)
        ]
    ).to_csv(source / "trajectory_selection_bic_source_data.csv", index=False)
    pd.DataFrame(
        [
            {
                "feature": f"sofa2_resp__h{start}_{start + 12}",
                "observed_n": 80 - start,
                "missing_n": 20 + start,
                "missing_fraction": (20 + start) / 100,
            }
            for start in (0, 12, 24)
        ]
    ).to_csv(source / "trajectory_selection_availability_source_data.csv", index=False)

    summary = renderer._render_h3(source, tmp_path / "out")

    assert summary["main_figure_count"] == 1
    assert summary["supplementary_figure_count"] == 1
    assert summary["scientific_status"] == "failed_closed"
    main_contract = json.loads(
        (
            tmp_path
            / "out"
            / "h3_main_figure_1_candidate_selection_diagnostic.figure_contract.json"
        ).read_text(encoding="utf-8")
    )
    supplementary_contract = json.loads(
        (
            tmp_path
            / "out"
            / "h3_supplementary_figure_s1_feature_availability.figure_contract.json"
        ).read_text(encoding="utf-8")
    )
    main_metadata = main_contract["panels"][0]["metadata"]
    assert main_metadata["placement"] == "main"
    assert main_metadata["display_purpose"] == "diagnostic"
    assert main_metadata["source_data"] == ["h3_selection_source_data.csv"]
    assert supplementary_contract["panels"][0]["metadata"]["placement"] == (
        "supplementary"
    )
    assert supplementary_contract["panels"][0]["metadata"]["display_purpose"] == (
        "audit"
    )
    assert "no class is selected" in main_contract["statistics_note"].casefold()
    table_id, placement, *_ = renderer.ARTICLE_TABLE_SPECS["h3"][0]
    assert (table_id, placement) == ("table_1_candidate_selection", "main")
    assert renderer.ARTICLE_TABLE_ROLES[("h3", table_id)] == "cluster_selection"


def test_article_table_specs_have_complete_typed_role_coverage() -> None:
    expected = {
        (task_id, table_id)
        for task_id, specs in renderer.ARTICLE_TABLE_SPECS.items()
        for table_id, *_ in specs
    }

    assert set(renderer.ARTICLE_TABLE_ROLES) == expected


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


def test_landmark_article_figure_uses_comparable_contrasts_not_robustness_ranges(
    tmp_path: Path,
) -> None:
    source = tmp_path / "visual"
    source.mkdir()
    pd.DataFrame(
        [
            {
                "group_value": group,
                "estimate_type": estimate_type,
                "prevalence_pct": value if estimate_type == "prevalence" else None,
                "outcome_risk_pct": value if estimate_type == "outcome_risk" else None,
                "estimate": value / 100.0,
                "ci_low": value / 100.0 - 0.005,
                "ci_high": value / 100.0 + 0.005,
            }
            for group, prevalence, outcome in (
                ("observed", 54.0, 14.0),
                ("no_source", 46.0, 6.0),
            )
            for estimate_type, value in (
                ("prevalence", prevalence),
                ("outcome_risk", outcome),
            )
        ]
    ).to_csv(source / "absolute_risk_context_source_data.csv", index=False)
    pd.DataFrame(
        {
            "exposure_value": [1.0, 2.1, 5.0],
            "reference_exposure_value": [2.1, 2.1, 2.1],
            "adjusted_odds_ratio": [0.76, 1.0, 1.96],
            "ci_low": [0.72, 1.0, 1.89],
            "ci_high": [0.81, 1.0, 2.03],
        }
    ).to_csv(source / "curve.csv", index=False)
    pd.DataFrame(
        {
            "concept": ["lactate"],
            "n_total": [100],
            "measured_one_n": [54],
            "repeat_measured_n": [21],
        }
    ).to_csv(source / "measurement.csv", index=False)
    contrasts = tmp_path / "contrasts.csv"
    pd.DataFrame(
        {
            "exposure_value": [1.0, 5.0],
            "reference_exposure_value": [2.1, 2.1],
            "adjusted_odds_ratio": [0.76, 1.96],
            "ci_low": [0.72, 1.89],
            "ci_high": [0.81, 2.03],
        }
    ).to_csv(contrasts, index=False)

    summary = renderer._render_landmark_association(
        task_id="generic",
        source_dir=source,
        out_dir=tmp_path / "out",
        exposure_label="Maximum biomarker (mmol/L)",
        curve_file="curve.csv",
        contrast_path=contrasts,
        measurement_file="measurement.csv",
        measurement_is_main=False,
    )

    figure_name = "generic_main_figure_2_continuous_association_and_contrasts"
    assert summary["main_figure_count"] == 1
    assert summary["supplementary_figure_count"] == 2
    contract = json.loads(
        (tmp_path / "out" / f"{figure_name}.figure_contract.json").read_text()
    )
    assert contract["panels"][1]["metadata"]["chart_type"] == "contrast_forest"
    assert contract["panels"][1]["metadata"]["effect_comparison_authorized"]
    assert all(
        "robustness" not in source.casefold() for source in contract["source_data"]
    )
    svg = (
        tmp_path / "out" / "generic_supplementary_figure_s1_measurement_context.svg"
    ).read_text(encoding="utf-8")
    assert "Not measured" in svg
    assert "No recorded source" not in svg


def test_landmark_reporting_replay_adds_adjusted_risk_and_population_accounting(
    tmp_path: Path,
) -> None:
    source = tmp_path / "visual"
    source.mkdir()
    pd.DataFrame(
        [
            {
                "group_value": group,
                "estimate_type": estimate_type,
                "prevalence_pct": value if estimate_type == "prevalence" else None,
                "outcome_risk_pct": value if estimate_type == "outcome_risk" else None,
                "estimate": value / 100.0,
                "ci_low": value / 100.0 - 0.005,
                "ci_high": value / 100.0 + 0.005,
            }
            for group, prevalence, outcome in (
                ("observed", 54.0, 14.0),
                ("no_source", 46.0, 6.0),
            )
            for estimate_type, value in (
                ("prevalence", prevalence),
                ("outcome_risk", outcome),
            )
        ]
    ).to_csv(source / "absolute_risk_context_source_data.csv", index=False)
    curve = pd.DataFrame(
        {
            "exposure_value": [1.0, 2.1, 5.0],
            "reference_exposure_value": [2.1, 2.1, 2.1],
            "adjusted_odds_ratio": [0.76, 1.0, 1.96],
            "ci_low": [0.72, 1.0, 1.89],
            "ci_high": [0.81, 1.0, 2.03],
        }
    )
    curve.to_csv(source / "curve.csv", index=False)
    pd.DataFrame(
        {
            "concept": ["bilirubin"],
            "n_total": [100],
            "measured_one_n": [54],
            "repeat_measured_n": [21],
        }
    ).to_csv(source / "measurement.csv", index=False)
    contrasts = tmp_path / "contrasts.csv"
    curve.iloc[[0, 2]].to_csv(contrasts, index=False)
    reporting = tmp_path / "reporting"
    reporting.mkdir()
    pd.DataFrame(
        {
            "exposure_value": [1.0, 2.1, 5.0],
            "reference_exposure_value": [2.1, 2.1, 2.1],
            "adjusted_absolute_risk": [0.08, 0.11, 0.22],
            "ci_low": [0.07, 0.10, 0.20],
            "ci_high": [0.09, 0.12, 0.24],
        }
    ).to_csv(reporting / "m1_adjusted_absolute_risk.csv", index=False)
    pd.DataFrame(
        {
            "stage": [
                "source_cohort",
                "alive_and_under_observation_at_landmark",
                "valid_exposure_primary_population",
                "complete_case_model_population",
            ],
            "n": [100, 80, 54, 52],
            "excluded_from_previous": [0, 20, 26, 2],
            "population_rule": ["source", "landmark", "exposure", "model"],
        }
    ).to_csv(reporting / "m1_landmark_population_flow.csv", index=False)

    summary = renderer._render_landmark_association(
        task_id="m1",
        source_dir=source,
        out_dir=tmp_path / "out",
        exposure_label="Maximum bilirubin (mg/dL)",
        curve_file="curve.csv",
        contrast_path=contrasts,
        measurement_file="measurement.csv",
        measurement_is_main=True,
        reporting_dir=reporting,
    )

    contract = json.loads(
        (
            tmp_path
            / "out"
            / "m1_main_figure_2_continuous_association_and_contrasts.figure_contract.json"
        ).read_text()
    )
    assert [panel["metadata"]["article_role"] for panel in contract["panels"]] == [
        "absolute_risk",
        "primary_estimand",
        "primary_estimand",
    ]
    assert summary["additional_main_table_count"] == 1
    assert (
        tmp_path / "out" / "table_1b_landmark_population_flow.table_contract.json"
    ).exists()

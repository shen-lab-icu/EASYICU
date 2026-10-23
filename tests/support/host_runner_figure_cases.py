"""Tiny invented result tables for host runner figure parity."""

import pandas as pd


def cohort_flow_frames() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0, "universe", 140, 0, 140],
            [1, "adult", 140, 12, 128],
            [2, "first_icu_stay", 128, 8, 120],
        ],
        columns=[
            "step_order",
            "predicate_kind",
            "n_before",
            "n_excluded",
            "n_remaining",
        ],
    )


def prevalence_outcome_frames() -> pd.DataFrame:
    columns = [
        "row_type",
        "variable",
        "category",
        "n",
        "denominator",
        "percentage",
        "estimate",
        "ci_lower",
        "ci_upper",
        "events",
        "missing_n",
        "missing_pct",
    ]
    rows = [
        [
            "cohort_context",
            "analysis_denominator",
            "locked_cohort_n",
            10,
            10,
            1.0,
            10.0,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            "cohort_context",
            "analysis_denominator",
            "exposure_available_n",
            10,
            10,
            1.0,
            10.0,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            "prevalence",
            "binary_exposure",
            0,
            6,
            10,
            0.6,
            0.6,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            "prevalence",
            "binary_exposure",
            1,
            4,
            10,
            0.4,
            0.4,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            "absolute_outcome_risk",
            "binary_outcome",
            0,
            6,
            6,
            1 / 6,
            1 / 6,
            0.03,
            0.56,
            1,
            0,
            0,
        ],
        [
            "absolute_outcome_risk",
            "binary_outcome",
            1,
            4,
            4,
            0.5,
            0.5,
            0.15,
            0.85,
            2,
            0,
            0,
        ],
    ]
    return pd.DataFrame(rows, columns=columns)


def prevalence_cohort_frames() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["locked_cohort", None, 10, 10, 100.0, None, None],
            ["exposure_available", None, 10, 10, 100.0, None, None],
            ["exposure_missing", None, 0, 10, 0.0, None, None],
            ["exposure_prevalence", 1.0, 4, 10, 40.0, 20.0, 65.0],
            ["outcome_available", None, 10, 10, 100.0, None, None],
            ["outcome_missing", None, 0, 10, 0.0, None, None],
        ],
        columns=[
            "summary",
            "exposure_level",
            "numerator",
            "denominator",
            "percentage",
            "ci_low",
            "ci_high",
        ],
    )


def prevalence_incidence_frames() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["overall", 10, 10, 3, 30.0, 10.0, 60.0],
            ["0.0", 6, 6, 1, 100.0 / 6.0, 1.0, 55.0],
            ["1.0", 4, 4, 2, 50.0, 15.0, 85.0],
        ],
        columns=[
            "exposure_level",
            "exposure_count",
            "outcome_observed_n",
            "deaths",
            "mortality_pct",
            "ci_low_pct",
            "ci_high_pct",
        ],
    )


def composite_frames() -> dict[str, pd.DataFrame]:
    return {
        "table:cohort_flow": pd.DataFrame(
            {"concept_id": ["analysis cohort"], "n_remaining": [100]}
        ),
        "table:exposure_outcome_distribution": pd.DataFrame(
            {
                "row_role": ["exposure_level", "exposure_level", "overall"],
                "exposure_level": [0, 1, -1],
                "n_rows": [60, 40, 100],
                "exposure_denominator": [100, 100, 100],
                "exposure_pct": [60.0, 40.0, 100.0],
                "outcome_events": [6, 8, 14],
                "outcome_denominator": [60, 40, 100],
                "outcome_rate_pct": [10.0, 20.0, 14.0],
            }
        ),
        "table:missingness_measurement_audit": pd.DataFrame(
            {
                "variable": ["age", "lactate"],
                "label": ["Age", "Lactate"],
                "n_total": [100, 100],
                "missing_n": [0, 20],
                "missing_pct": [0.0, 20.0],
            }
        ),
        "table:measurement_process_audit": pd.DataFrame(
            {
                "concept": ["age", "lactate"],
                "n_total": [100, 100],
                "measured_one_n": [100, 80],
                "eligible_n": [100, 100],
            }
        ),
    }


INPUTS = (
    "table:generic_landmark_rcs_curve",
    "table:generic_adjusted_absolute_risk",
    "table:robustness_summary",
    "table:measurement_process",
)


def landmark_frames() -> dict[str, pd.DataFrame]:
    return {
        INPUTS[0]: pd.DataFrame(
            {
                "exposure": ["biomarker_mg_dl"] * 3,
                "biomarker_mg_dl": [1.0, 3.0, 5.0],
                "reference_biomarker_mg_dl": [2.1, 2.1, 2.1],
                "adjusted_odds_ratio": [0.76, 1.1, 1.96],
                "ci_low": [0.72, 1.0, 1.89],
                "ci_high": [0.81, 1.21, 2.03],
                "exposure_density_n": [20, 60, 20],
                "exposure_density_fraction": [0.2, 0.6, 0.2],
            }
        ),
        INPUTS[1]: pd.DataFrame(
            {
                "exposure": ["biomarker_mg_dl"] * 3,
                "biomarker_mg_dl": [1.0, 3.0, 5.0],
                "reference_biomarker_mg_dl": [2.1, 2.1, 2.1],
                "adjusted_absolute_risk": [0.08, 0.12, 0.23],
                "ci_low": [0.07, 0.1, 0.2],
                "ci_high": [0.09, 0.14, 0.26],
                "exposure_density_n": [20, 60, 20],
                "exposure_density_fraction": [0.2, 0.6, 0.2],
            }
        ),
        INPUTS[2]: pd.DataFrame(
            {
                "axis": ["primary", "functional form"],
                "total_specs": [1, 1],
                "converged_specs": [1, 1],
                "range_low": [1.89, 1.24],
                "range_high": [2.03, 1.27],
            }
        ),
        INPUTS[3]: pd.DataFrame(
            {"concept": ["exposure"], "n_total": [100], "measured_one_n": [54]}
        ),
    }


def trajectory_selection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_clusters": [2, 3, 4],
            "bic": [300.0, 200.0, 100.0],
            "aic": [280.0, 180.0, 80.0],
            "final_log_likelihood": [-120.0, -70.0, -20.0],
            "parameter_count": [20, 20, 20],
            "selected": [False, False, True],
            "aic_minimum": [False, False, True],
            "upper_boundary": [False, False, True],
            "scientific_status": ["failed_closed"] * 3,
            "reason_code": ["NO_INTERIOR_OPTIMUM"] * 3,
            "reportable_result": [
                "no_interior_solution_in_prespecified_candidate_range"
            ]
            * 3,
        }
    )


def trajectory_selected() -> pd.DataFrame:
    frame = trajectory_selection()
    frame["selected"] = [False, True, False]
    frame["aic_minimum"] = [False, True, False]
    frame["upper_boundary"] = [False, False, True]
    frame["bic"] = [300.0, 100.0, 200.0]
    frame["aic"] = [280.0, 80.0, 180.0]
    frame["scientific_status"] = ["selected"] * 3
    frame["reason_code"] = ["INTERIOR_OPTIMUM"] * 3
    frame["reportable_result"] = ["interior_solution"] * 3
    return frame


def trajectory_availability() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["resp__h0_12", "resp__h12_24", "lact__h0_12", "lact__h12_24"],
            "observed_n": [80, 60, 90, 70],
            "missing_n": [20, 40, 10, 30],
            "missing_fraction": [0.2, 0.4, 0.1, 0.3],
        }
    )


def trajectory_profiles(*, empty: bool = False) -> pd.DataFrame:
    columns = [
        "cluster",
        "source_column",
        "window_start_hours",
        "window_end_hours",
        "summary_statistic",
        "value",
        "n_observed",
    ]
    if empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for cluster in (0, 1):
        for concept in ("resp", "lact"):
            for start in (0, 12):
                rows.append(
                    {
                        "cluster": cluster,
                        "source_column": f"{concept}__h{start}_{start + 12}",
                        "window_start_hours": start,
                        "window_end_hours": start + 12,
                        "summary_statistic": "mean",
                        "value": 0.5 * cluster + 0.1 * start,
                        "n_observed": 40,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def trajectory_cluster_sizes(*, empty: bool = False) -> pd.DataFrame:
    columns = ["cluster", "n"]
    if empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame({"cluster": [0, 1], "n": [55, 45]}, columns=columns)


def trajectory_cluster_stability(*, empty: bool = False) -> pd.DataFrame:
    if empty:
        return pd.DataFrame(columns=_STABILITY_OWNER_COLUMNS)
    return pd.DataFrame(
        {
            "resample_id": [f"stability_resample_{index:03}" for index in range(12)],
            "n_overlap": [80] * 12,
            "adjusted_rand_index": [0.7 + 0.01 * index for index in range(12)],
            "clustering_method": ["latent_class_diagonal_gaussian_mixture"] * 12,
            "refit_model_id": [f"refit_{index}" for index in range(12)],
            "seed": list(range(12)),
            "sampling_method": ["subsample_without_replacement"] * 12,
            "sample_n": [80] * 12,
            "sample_id_hash": [f"{index:064x}" for index in range(12)],
            "selected_n_clusters": [2] * 12,
            "converged": [True] * 12,
            "n_iter": [20] * 12,
            "final_log_likelihood": [-100.0 - index for index in range(12)],
            "parameter_sha256": [f"{index + 1:064x}" for index in range(12)],
        },
        columns=[*_STABILITY_OWNER_COLUMNS, *_STABILITY_REFIT_TRACE],
    )


def trajectory_signed_figure_step(step_id: str):
    from easyicu.research_agent.schema import AnalysisStep
    from easyicu.research_agent.trajectory.scientific_runtime_authority import (
        TRAJECTORY_CHARACTERIZATION_FIGURE_PANELS,
        TRAJECTORY_SELECTION_FIGURE_PANELS,
    )
    from easyicu.research_agent.execution.runners.trajectory_selection_figure_executor import (
        TRAJECTORY_SELECTION_FIGURE_INPUTS,
        TRAJECTORY_SELECTION_FIGURE_OUTPUTS,
    )

    return AnalysisStep(
        step_id=step_id,
        planned_analysis_role="auxiliary",
        intent="Render the signed candidate-grid decision and the class characterization.",
        inputs=list(TRAJECTORY_SELECTION_FIGURE_INPUTS),
        expected_outputs=list(TRAJECTORY_SELECTION_FIGURE_OUTPUTS),
        method="signed_trajectory_selection_diagnostic_figure",
        input_consumption_contracts=[
            {"input_key": key, "mode": "all_rows"}
            for key in TRAJECTORY_SELECTION_FIGURE_INPUTS
        ],
        figure_panels=[
            *(
                panel.bind(figure_output="figure:trajectory_selection_diagnostics")
                for panel in TRAJECTORY_SELECTION_FIGURE_PANELS
            ),
            *(
                panel.bind(figure_output="figure:trajectory_phenotype_characterization")
                for panel in TRAJECTORY_CHARACTERIZATION_FIGURE_PANELS
            ),
        ],
    )


_STABILITY_OWNER_COLUMNS = [
    "resample_id",
    "n_overlap",
    "adjusted_rand_index",
    "clustering_method",
    "refit_model_id",
    "seed",
    "sampling_method",
    "sample_n",
    "sample_id_hash",
    "selected_n_clusters",
]


_STABILITY_REFIT_TRACE = [
    "converged",
    "n_iter",
    "final_log_likelihood",
    "parameter_sha256",
]

"""Repackage frozen Dev9 outputs into article-level figure suites.

This benchmark-only renderer performs no fitting, imputation, outcome-driven
selection, or scientific recomputation.  It validates named frozen outputs and
uses declared row roles only to separate reader-facing panels.  It must never
be used to promote Dev9 evidence to paper-authorized status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import easyicu
from easyicu.research_agent.execution.runners.prediction_figure_executor import (
    run_prediction_figure,
)
from easyicu.research_agent.figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from easyicu.research_agent.reporting.article_display_package import (
    inspect_article_display_package,
)
from easyicu.research_agent.reporting.article_display_policy import (
    ArticleDisplayPolicyRequest,
    decide_article_display,
)


E3_RUN_RELATIVE = Path("e3/e3_kdigo_gradient/aware/run_20260825T024928_3b8fef")
M2_RUN_RELATIVE = Path("m2/m2_mortality_prediction/aware/run_20260825T025332_5c7922")
H2_FEASIBILITY_RELATIVE = Path(
    "steps/00_authority_compiled_source_feasibility/outputs/h2_source_feasibility.csv"
)
RUN_RELATIVES = {
    "e1": Path("e1/e1_sepsis3_prevalence_mortality/aware/run_20260825T024625_c92a2b"),
    "e2": Path("e2/e2_lactate_mortality/aware/run_20260825T024807_5b7759"),
    "e3": E3_RUN_RELATIVE,
    "m1": Path("m1/m1_hepatobiliary_missingness/aware/run_20260825T025150_5ca36b"),
    "m2": M2_RUN_RELATIVE,
    "m3": Path("m3/m3_sepsis_subphenotype/aware/run_20260825T025546_81b23c"),
    "h1": Path("h1/h1_ventilation_survival/aware/run_20260825T025904_fceb36"),
    "h2": Path("h2/h2_vasopressor_causal/aware/run_20260825T025958_39cbae"),
    "h3": Path("h3/h3_trajectory_clustering/aware/run_20260825T030126_bbb780"),
}

# Benchmark-specific display choices. These never enter shared prompts or execution
# contracts. Each tuple is (display id, placement, producing step, source basename,
# title). The source table is copied byte-for-byte from the frozen EvidenceStore.
ARTICLE_TABLE_SPECS = {
    "e1": (
        (
            "table_1_cohort_characteristics",
            "main",
            "baseline_context",
            "table_one.csv",
            "Cohort characteristics",
        ),
        (
            "table_2_adjusted_association",
            "main",
            "primary_adjusted_association",
            "adjusted_association_estimates.csv",
            "Adjusted association estimates",
        ),
        (
            "table_s1_component_completeness",
            "supplementary",
            "data_quality_audit",
            "exposure_component_completeness_audit.csv",
            "Component completeness audit",
        ),
        (
            "table_s2_definition_sensitivity",
            "supplementary",
            "scientific_sensitivity",
            "e1_scientific_sensitivity.csv",
            "Definition and cohort sensitivity analyses",
        ),
    ),
    "e2": (
        (
            "table_1_cohort_characteristics",
            "main",
            "table_one",
            "table_one.csv",
            "Cohort characteristics",
        ),
        (
            "table_2_lactate_contrasts",
            "main",
            "primary_adjusted_association",
            "e2_landmark_rcs_contrasts.csv",
            "Adjusted lactate contrasts",
        ),
        (
            "table_s1_measurement_process",
            "supplementary",
            "measurement_audit",
            "measurement_process_audit.csv",
            "Lactate measurement-process audit",
        ),
        (
            "table_s2_robustness",
            "supplementary",
            "robustness_replay",
            "robustness_summary.csv",
            "Robustness specifications",
        ),
    ),
    "e3": (
        (
            "table_1_cohort_characteristics",
            "main",
            "table_one_01",
            "table_one.csv",
            "Cohort characteristics",
        ),
        (
            "table_2_stage_outcomes",
            "main",
            "ordinal_trend_01",
            "ordered_stratified_outcomes.csv",
            "Outcomes across ordered KDIGO stages",
        ),
        (
            "table_s1_missingness",
            "supplementary",
            "measurement_audit_01",
            "missingness_measurement_audit.csv",
            "Kidney-measurement missingness audit",
        ),
        (
            "table_s2_stage_sensitivity",
            "supplementary",
            "host_association_model_grid_d3cfb9c38429",
            "e3_scientific_sensitivity.csv",
            "KDIGO sensitivity analyses",
        ),
    ),
    "m1": (
        (
            "table_1_cohort_characteristics",
            "main",
            "baseline_table",
            "table_one.csv",
            "Cohort characteristics",
        ),
        (
            "table_2_bilirubin_contrasts",
            "main",
            "primary_adjusted_estimate",
            "m1_landmark_bilirubin_contrasts.csv",
            "Adjusted bilirubin contrasts",
        ),
        (
            "table_s1_measurement_process",
            "supplementary",
            "measurement_process_audit",
            "measurement_process_audit.csv",
            "Bilirubin measurement-process audit",
        ),
        (
            "table_s2_robustness",
            "supplementary",
            "robustness_grid",
            "robustness_summary.csv",
            "Robustness specifications",
        ),
    ),
    "m2": (
        (
            "table_1_cohort_characteristics",
            "main",
            "baseline_context",
            "table_one.csv",
            "Cohort characteristics",
        ),
        (
            "table_2_model_performance",
            "main",
            "primary_discrimination",
            "prediction_performance.csv",
            "Model discrimination and overall performance",
        ),
        (
            "table_3_calibration",
            "main",
            "calibration_assessment",
            "calibration_assessment.csv",
            "Calibration assessment",
        ),
        (
            "table_s1_internal_validation",
            "supplementary",
            "test_validation",
            "internal_validation.csv",
            "Repeated patient-level internal validation",
        ),
        (
            "table_s2_clinical_utility",
            "supplementary",
            "clinical_utility",
            "clinical_utility.csv",
            "Exploratory decision-curve values",
        ),
    ),
    "m3": (
        (
            "table_1_cohort_characteristics",
            "main",
            "baseline_context",
            "table_one.csv",
            "Cohort characteristics",
        ),
        (
            "table_2_candidate_profiles",
            "main",
            "primary_phenotype_solution",
            "phenotype_profiles.csv",
            "Candidate cluster profiles",
        ),
        (
            "table_3_stability",
            "main",
            "cluster_stability",
            "cluster_stability_with_algorithm_agreement.csv",
            "Cluster stability and algorithm agreement",
        ),
        (
            "table_s1_complete_case",
            "supplementary",
            "primary_phenotype_solution",
            "phenotyping_complete_case_sensitivity.csv",
            "Complete-case sensitivity analysis",
        ),
        (
            "table_s2_measurement_process",
            "supplementary",
            "measurement_process_audit",
            "measurement_process_audit.csv",
            "Phenotyping measurement-process audit",
        ),
    ),
    "h1": (
        (
            "table_1_landmark_cohort",
            "main",
            "01_authority_compiled_survival_suite",
            "landmark_table_one.csv",
            "Landmark cohort characteristics",
        ),
        (
            "table_2_time_to_event_model",
            "main",
            "01_authority_compiled_survival_suite",
            "landmark_cox_summary.csv",
            "Landmark time-to-event model",
        ),
        (
            "table_3_rmst",
            "main",
            "01_authority_compiled_survival_suite",
            "landmark_rmst_summary.csv",
            "Restricted mean survival-time contrasts",
        ),
        (
            "table_s1_ph_diagnostics",
            "supplementary",
            "01_authority_compiled_survival_suite",
            "landmark_ph_diagnostics.csv",
            "Proportional-hazards diagnostics",
        ),
        (
            "table_s2_risk_set",
            "supplementary",
            "01_authority_compiled_survival_suite",
            "landmark_risk_set_flow.csv",
            "Landmark risk-set accounting",
        ),
    ),
    "h2": (
        (
            "table_1_causal_identifiability",
            "main",
            "00_authority_compiled_source_feasibility",
            "h2_source_feasibility.csv",
            "Causal-contrast identifiability assessment",
        ),
    ),
    "h3": (
        (
            "table_1_candidate_selection",
            "main",
            "01_authority_compiled_trajectory_candidates",
            "trajectory_candidate_selection.csv",
            "Candidate-grid selection diagnostics",
        ),
        (
            "table_s1_feature_availability",
            "supplementary",
            "00_authority_compiled_trajectory_representation",
            "feature_availability.csv",
            "Trajectory-feature availability",
        ),
    ),
}

ARTICLE_TABLE_ROLES = {
    ("e1", "table_1_cohort_characteristics"): "baseline_context",
    ("e1", "table_2_adjusted_association"): "primary_estimand",
    ("e1", "table_s1_component_completeness"): "data_quality",
    ("e1", "table_s2_definition_sensitivity"): "robustness",
    ("e2", "table_1_cohort_characteristics"): "baseline_context",
    ("e2", "table_2_lactate_contrasts"): "primary_estimand",
    ("e2", "table_s1_measurement_process"): "data_quality",
    ("e2", "table_s2_robustness"): "robustness",
    ("e3", "table_1_cohort_characteristics"): "baseline_context",
    ("e3", "table_2_stage_outcomes"): "descriptive_result",
    ("e3", "table_s1_missingness"): "data_quality",
    ("e3", "table_s2_stage_sensitivity"): "robustness",
    ("m1", "table_1_cohort_characteristics"): "baseline_context",
    ("m1", "table_2_bilirubin_contrasts"): "primary_estimand",
    ("m1", "table_s1_measurement_process"): "data_quality",
    ("m1", "table_s2_robustness"): "robustness",
    ("m2", "table_1_cohort_characteristics"): "baseline_context",
    ("m2", "table_2_model_performance"): "model_performance",
    ("m2", "table_3_calibration"): "calibration",
    ("m2", "table_s1_internal_validation"): "validation",
    ("m2", "table_s2_clinical_utility"): "clinical_utility",
    ("m3", "table_1_cohort_characteristics"): "baseline_context",
    ("m3", "table_2_candidate_profiles"): "phenotype_profile",
    ("m3", "table_3_stability"): "stability",
    ("m3", "table_s1_complete_case"): "robustness",
    ("m3", "table_s2_measurement_process"): "data_quality",
    ("h1", "table_1_landmark_cohort"): "baseline_context",
    ("h1", "table_2_time_to_event_model"): "survival_effect",
    ("h1", "table_3_rmst"): "survival_effect",
    ("h1", "table_s1_ph_diagnostics"): "diagnostics",
    ("h1", "table_s2_risk_set"): "cohort_accounting",
    ("h2", "table_1_causal_identifiability"): "diagnostics",
    ("h3", "table_1_candidate_selection"): "cluster_selection",
    ("h3", "table_s1_feature_availability"): "data_quality",
}


def _require_current_worktree_import() -> None:
    expected = Path(__file__).resolve().parents[2] / "src"
    imported = Path(easyicu.__file__).resolve()
    if not imported.is_relative_to(expected):
        raise RuntimeError(
            "Renderer imported EasyICU outside the current worktree; rerun with PYTHONPATH=src"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains non-finite values")


def _measurement_state_label(value: Any) -> str:
    token = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if token in {"observed", "measured", "source_present", "with_source"}:
        return "Measured"
    if token in {"no_source", "not_measured", "unmeasured", "source_absent"}:
        return "Not measured"
    return str(value or "").strip().replace("_", " ").title()


def _evidence_table(
    *, run_dir: Path, step_id: str, basename: str
) -> tuple[dict[str, Any], Path]:
    index_path = run_dir / "evidence/evidence_index.json"
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Evidence index must be a list: {index_path}")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("kind") == "table"
        and row.get("produced_by_step") == step_id
        and Path(str(row.get("relative_path") or "")).name.endswith(f"__{basename}")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one frozen {step_id}/{basename}; found {len(matches)}"
        )
    row = matches[0]
    source_path = run_dir / str(row["relative_path"])
    if _sha256(source_path) != row.get("sha256"):
        raise ValueError(f"Evidence digest mismatch: {source_path}")
    return row, source_path


def _copy_source(frame: pd.DataFrame, path: Path, output: Path) -> str:
    copied = frame.copy()
    for column in ("source_row_index", "source_file", "source_sha256"):
        if column in copied.columns:
            upstream = f"upstream_{column}"
            if upstream in copied.columns:
                raise ValueError(f"nested provenance collision for {column}")
            copied = copied.rename(columns={column: upstream})
    copied.insert(0, "source_sha256", _sha256(path))
    copied.insert(0, "source_file", path.name)
    copied.insert(0, "source_row_index", range(len(copied)))
    copied.to_csv(output, index=False)
    return output.name


def _package_article_tables(
    *, source_root: Path, output_root: Path
) -> dict[str, dict[str, Any]]:
    """Copy selected frozen evidence tables and bind manuscript placement."""

    summaries: dict[str, dict[str, Any]] = {}
    for task_id, specs in ARTICLE_TABLE_SPECS.items():
        run_dir = source_root / RUN_RELATIVES[task_id]
        task_out = output_root / task_id
        task_out.mkdir(parents=True, exist_ok=True)
        placements = {"main": 0, "supplementary": 0}
        contracts: list[str] = []
        for table_id, placement, step_id, basename, title in specs:
            row, source_path = _evidence_table(
                run_dir=run_dir, step_id=step_id, basename=basename
            )
            packaged_name = f"{table_id}{source_path.suffix.lower()}"
            packaged_path = task_out / packaged_name
            shutil.copy2(source_path, packaged_path)
            article_role = ARTICLE_TABLE_ROLES[(task_id, table_id)]
            terminal_diagnostic = task_id in {"h2", "h3"} and placement == "main"
            display_decision = decide_article_display(
                ArticleDisplayPolicyRequest(
                    article_role=article_role,
                    requested_placement=placement,
                    scientific_status=(
                        "failed_closed" if task_id in {"h2", "h3"} else "analysis_only"
                    ),
                    central_to_question=terminal_diagnostic,
                    terminal_diagnostic=terminal_diagnostic,
                )
            )
            contract = {
                "schema_version": "easyicu.article_table_contract/1",
                "table_id": f"table:{task_id}:{table_id}",
                "title": title,
                "article_role": article_role,
                "placement": display_decision.placement,
                "display_purpose": display_decision.display_purpose,
                "display_policy_reason_code": display_decision.reason_code,
                "authority_scope": "analysis_only",
                "paper_authorization_allowed": False,
                "source_path": packaged_name,
                "source_sha256": _sha256(packaged_path),
                "upstream_evidence_id": row.get("evidence_id"),
                "upstream_relative_path": row.get("relative_path"),
                "upstream_sha256": row.get("sha256"),
                "produced_by_step": step_id,
                "supports": (
                    f"The frozen EvidenceStore supports the values displayed in {title}."
                ),
                "cannot_prove": (
                    "This table does not establish publication readiness, external "
                    "validity, causality, or clinical utility beyond its declared analysis."
                ),
            }
            if display_decision.display_purpose == "diagnostic":
                contract["cannot_prove"] = (
                    "This diagnostic table does not provide an effect estimate, an "
                    "authorized causal contrast, or a selected trajectory-class solution."
                )
            contract_path = task_out / f"{table_id}.table_contract.json"
            contract_path.write_text(
                json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            contracts.append(contract_path.name)
            placements[display_decision.placement] += 1
        summaries[task_id] = {
            "main_table_count": placements["main"],
            "supplementary_table_count": placements["supplementary"],
            "table_contracts": contracts,
        }
    return summaries


def _load_frozen_tables(
    source_dir: Path, names: dict[str, str]
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    paths = {key: source_dir / name for key, name in names.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen display sources are incomplete: {missing}")
    return paths, {key: pd.read_csv(path) for key, path in paths.items()}


def _copy_frozen_tables(
    *,
    out_dir: Path,
    task_id: str,
    paths: dict[str, Path],
    frames: dict[str, pd.DataFrame],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        key: _copy_source(
            frames[key], paths[key], out_dir / f"{task_id}_{key}_source_data.csv"
        )
        for key in paths
    }


def _task_summary(
    *,
    source: Path,
    paths: dict[str, Path],
    figure_files: list[str],
    main_figure_count: int,
    supplementary_figure_count: int,
    scientific_status: str = "analysis_only",
    reason_code: str | None = None,
) -> dict[str, Any]:
    return {
        "source": str(source),
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "scientific_status": scientific_status,
        "reason_code": reason_code,
        "main_figure_count": main_figure_count,
        "supplementary_figure_count": supplementary_figure_count,
        "figure_files": sorted(set(figure_files)),
        "source_sha256": {name: _sha256(path) for name, path in paths.items()},
    }


def _save(
    fig: Any,
    *,
    out_dir: Path,
    product: str,
    width_mm: float = 183.0,
    height_mm: float,
    panels: list[dict[str, Any]],
    source_data: list[str],
    core_claim: str,
    statistics_note: str,
) -> list[str]:
    contract = make_figure_contract(
        figure_id=f"figure:{product}",
        core_claim=core_claim,
        archetype="asymmetric_mixed_modality",
        width_mm=width_mm,
        height_mm=height_mm,
        panels=panels,
        source_data=source_data,
        statistics_note=statistics_note,
    )
    outputs = save_publication_figure(
        fig,
        out_dir / product,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
        pad_inches=0.14,
    )
    plt.close(fig)
    return [path.name for path in outputs.values()]


def _forest(
    ax: Any, frame: pd.DataFrame, labels: list[str], title: str, color: str
) -> None:
    estimates = frame["estimate"].to_numpy(dtype=float)
    lows = frame["ci_low"].to_numpy(dtype=float)
    highs = frame["ci_high"].to_numpy(dtype=float)
    positions = np.arange(len(frame))
    ax.errorbar(
        estimates,
        positions,
        xerr=np.vstack((estimates - lows, highs - estimates)),
        fmt="o",
        color=color,
        capsize=2.5,
    )
    ax.axvline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    lower = min(1.0, float(np.min(lows)))
    upper = max(1.0, float(np.max(highs)))
    ax.set_xlim(max(lower / 1.15, np.finfo(float).tiny), upper * 1.15)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Odds ratio (95% CI)")
    ax.set_title(title, loc="left", pad=10)


def _render_e3(source_run: Path, out_dir: Path) -> dict[str, Any]:
    paths = {
        "ordered": source_run
        / "steps/ordinal_trend_01/outputs/ordered_stratified_outcomes.csv",
        "adjusted": source_run
        / "steps/adjusted_association_01/outputs/adjusted_association_estimates.csv",
        "sensitivity": source_run
        / "steps/host_association_model_grid_d3cfb9c38429/outputs/e3_scientific_sensitivity.csv",
        "missingness": source_run
        / "steps/measurement_audit_01/outputs/missingness_measurement_audit.csv",
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("E3 source run is incomplete")
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    ordered = frames["ordered"].sort_values("level_order").reset_index(drop=True)
    if not ordered["row_role"].astype(str).eq("exposure_level").all():
        raise ValueError("E3 ordered outcomes contain non-level rows")
    if ordered["level_order"].tolist() != list(range(len(ordered))):
        raise ValueError("E3 ordered levels are not consecutive")
    _finite(
        ordered,
        (
            "level_n",
            "binary_n",
            "binary_event_n",
            "binary_percentage",
            "binary_ci_low",
            "binary_ci_high",
            "continuous_n",
            "continuous_median",
            "continuous_q25",
            "continuous_q75",
        ),
    )
    expected_risk = 100.0 * ordered["binary_event_n"] / ordered["binary_n"]
    if not np.allclose(expected_risk, ordered["binary_percentage"], atol=1e-8):
        raise ValueError("E3 binary risks do not reconcile to counts")
    if not (
        (ordered["continuous_q25"] <= ordered["continuous_median"])
        & (ordered["continuous_median"] <= ordered["continuous_q75"])
    ).all():
        raise ValueError("E3 LOS quantiles are reversed")

    out_dir.mkdir(parents=True, exist_ok=True)
    source_files = {
        name: _copy_source(frame, paths[name], out_dir / f"e3_{name}_source_data.csv")
        for name, frame in frames.items()
    }
    palette = apply_publication_style(font_size=7.0)
    level_labels = [f"KDIGO stage {int(value)}" for value in ordered["level_value"]]

    fig, axes = plt.subplots(
        1, 2, figsize=(183 / 25.4, 82 / 25.4), constrained_layout=True
    )
    positions = np.arange(len(ordered))
    risk = ordered["binary_percentage"].to_numpy(dtype=float)
    risk_low = 100.0 * ordered["binary_ci_low"].to_numpy(dtype=float)
    risk_high = 100.0 * ordered["binary_ci_high"].to_numpy(dtype=float)
    axes[0].errorbar(
        positions,
        risk,
        yerr=np.vstack((risk - risk_low, risk_high - risk)),
        fmt="o-",
        color=palette["blue"],
        capsize=2.5,
    )
    axes[0].set_xticks(positions, level_labels, rotation=20, ha="right")
    axes[0].set_ylabel("Observed mortality risk (%)")
    axes[0].set_title("Mortality across ordered KDIGO stages", loc="left", pad=10)
    add_panel_label(axes[0], "a", x=-0.12, y=1.04, fontsize=8.0)

    median = ordered["continuous_median"].to_numpy(dtype=float)
    q25 = ordered["continuous_q25"].to_numpy(dtype=float)
    q75 = ordered["continuous_q75"].to_numpy(dtype=float)
    axes[1].errorbar(
        positions,
        median,
        yerr=np.vstack((median - q25, q75 - median)),
        fmt="o-",
        color=palette["orange"],
        capsize=2.5,
    )
    axes[1].set_xticks(positions, level_labels, rotation=20, ha="right")
    axes[1].set_ylabel("ICU length of stay (days), median (IQR)")
    axes[1].set_title("ICU length of stay across KDIGO stages", loc="left", pad=10)
    add_panel_label(axes[1], "b", x=-0.12, y=1.04, fontsize=8.0)
    figure_files = _save(
        fig,
        out_dir=out_dir,
        product="e3_main_figure_1_outcomes_by_kdigo_stage",
        height_mm=82.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Mortality across ordered KDIGO stages",
                "role": "descriptive_result",
                "article_role": "descriptive_result",
                "chart_type": "dot_interval_absolute_risk",
                "claim": "Observed mortality risk with Wilson 95% confidence intervals is shown for every ordered stage.",
                "evidence_ids": [_sha256(paths["ordered"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["ordered"]],
                },
            },
            {
                "panel_id": "b",
                "title": "ICU length of stay across KDIGO stages",
                "role": "descriptive_result",
                "article_role": "descriptive_result",
                "chart_type": "dot_interval",
                "claim": "Median ICU length of stay and interquartile range are shown for every ordered stage.",
                "evidence_ids": [_sha256(paths["ordered"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["ordered"]],
                },
            },
        ],
        source_data=[source_files["ordered"]],
        core_claim="Mortality risk and ICU length of stay are displayed separately across all ordered KDIGO stages.",
        statistics_note="All ordered-stratified rows are preserved. Mortality intervals are the source Wilson 95% confidence intervals; length of stay is the source median and IQR. No model is refit.",
    )

    adjusted = frames["adjusted"].copy()
    sensitivity = frames["sensitivity"].copy()
    _finite(adjusted, ("estimate", "ci_low", "ci_high"))
    _finite(sensitivity, ("estimate", "ci_low", "ci_high"))
    if not adjusted["fit_status"].astype(str).eq("fitted").all():
        raise ValueError("E3 adjusted association contains non-fitted rows")
    if not sensitivity["converged"].astype(str).str.casefold().eq("true").all():
        raise ValueError("E3 sensitivity table contains non-converged rows")
    fig, axes = plt.subplots(
        1, 2, figsize=(183 / 25.4, 100 / 25.4), constrained_layout=True
    )
    _forest(
        axes[0],
        adjusted,
        [str(value).replace(" vs ", " vs stage ") for value in adjusted["contrast"]],
        "Primary adjusted mortality association",
        palette["blue"],
    )
    add_panel_label(axes[0], "a", x=-0.12, y=1.04, fontsize=8.0)
    _forest(
        axes[1],
        sensitivity,
        [str(value).replace("_", " ") for value in sensitivity["analysis_id"]],
        "Scientific sensitivity analyses",
        palette["orange"],
    )
    add_panel_label(axes[1], "b", x=-0.12, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="e3_main_figure_2_adjusted_and_sensitivity",
        height_mm=100.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Primary adjusted mortality association",
                "role": "primary_estimand",
                "article_role": "primary_estimand",
                "chart_type": "forest",
                "claim": "All fitted stage contrasts from the registered primary model are shown with 95% confidence intervals.",
                "evidence_ids": [_sha256(paths["adjusted"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["adjusted"]],
                },
            },
            {
                "panel_id": "b",
                "title": "Scientific sensitivity analyses",
                "role": "robustness",
                "article_role": "robustness",
                "chart_type": "sensitivity_forest",
                "claim": "Every converged registered sensitivity analysis is shown with 95% confidence intervals.",
                "evidence_ids": [_sha256(paths["sensitivity"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["sensitivity"]],
                },
            },
        ],
        source_data=[source_files["adjusted"], source_files["sensitivity"]],
        core_claim="The registered adjusted association and all converged scientific sensitivity analyses are shown without refitting.",
        statistics_note="Odds ratios and 95% confidence intervals are copied from the registered primary and sensitivity tables. The renderer performs no row selection or model fitting.",
    )

    missingness = frames["missingness"].copy()
    _finite(missingness, ("missing_n", "n_total", "missing_pct"))
    expected_missing = 100.0 * missingness["missing_n"] / missingness["n_total"]
    if not np.allclose(expected_missing, missingness["missing_pct"], atol=1e-8):
        raise ValueError("E3 missingness percentages do not reconcile to counts")
    quality = missingness.sort_values("missing_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(183 / 25.4, 105 / 25.4), constrained_layout=True)
    positions = np.arange(len(quality))
    ax.barh(positions, quality["missing_pct"], color=palette["blue_soft"])
    ax.set_yticks(positions, quality["label"].astype(str).str.replace("_", " "))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing among eligible records (%)")
    ax.set_title("Measurement and missingness audit", loc="left", pad=10)
    add_panel_label(ax, "a", x=-0.12, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="e3_supplementary_figure_s1_missingness",
        height_mm=105.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Measurement and missingness audit",
                "role": "data_quality",
                "article_role": "data_quality",
                "chart_type": "availability_panel",
                "claim": "Routine measurement missingness is shown as supplementary audit evidence.",
                "evidence_ids": [_sha256(paths["missingness"])],
                "metadata": {
                    "placement": "supplementary",
                    "source_data": [source_files["missingness"]],
                },
            }
        ],
        source_data=[source_files["missingness"]],
        core_claim="Routine E3 missingness is retained as supplementary audit evidence rather than a primary-result panel.",
        statistics_note="All rows and source percentages are preserved and reconciled to counts. Missingness is not interpreted as a clinical effect.",
    )
    return {
        "source_run": str(source_run),
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "main_figure_count": 2,
        "supplementary_figure_count": 1,
        "figure_files": sorted(set(figure_files)),
        "source_sha256": {name: _sha256(path) for name, path in paths.items()},
    }


def _render_e1(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    paths, frames = _load_frozen_tables(
        source_dir,
        {
            "outcomes": "exposure_outcome_distribution_source_data.csv",
            "adjusted": "adjusted_association_estimates_source_data.csv",
            "sensitivity": "e1_scientific_sensitivity_source_data.csv",
            "completeness": "exposure_component_completeness_audit_source_data.csv",
        },
    )
    source_files = _copy_frozen_tables(
        out_dir=out_dir, task_id="e1", paths=paths, frames=frames
    )
    palette = apply_publication_style(font_size=7.0)
    figure_files: list[str] = []

    outcomes = frames["outcomes"].copy()
    levels = outcomes[outcomes["row_role"].astype(str).eq("exposure_level")].copy()
    overall = outcomes[outcomes["row_role"].astype(str).eq("overall")].copy()
    if levels["exposure_level"].astype(int).tolist() != [0, 1] or len(overall) != 1:
        raise ValueError(
            "E1 outcome table lacks the registered binary levels and overall row"
        )
    _finite(levels, ("exposure_pct", "outcome_rate_pct", "ci_low_pct", "ci_high_pct"))
    _finite(overall, ("outcome_rate_pct", "ci_low_pct", "ci_high_pct"))
    labels = ["No Sepsis-3", "Sepsis-3"]
    fig, axes = plt.subplots(
        1, 2, figsize=(183 / 25.4, 82 / 25.4), constrained_layout=True
    )
    positions = np.arange(2)
    axes[0].bar(
        positions, levels["exposure_pct"], color=[palette["neutral"], palette["blue"]]
    )
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Cohort share (%)")
    axes[0].set_ylim(0, 75)
    axes[0].set_title("Sepsis-3 denominator", loc="left", pad=10)
    add_panel_label(axes[0], "a", x=-0.12, y=1.04, fontsize=8.0)
    risk = levels["outcome_rate_pct"].to_numpy(dtype=float)
    low = levels["ci_low_pct"].to_numpy(dtype=float)
    high = levels["ci_high_pct"].to_numpy(dtype=float)
    axes[1].errorbar(
        positions,
        risk,
        yerr=np.vstack((risk - low, high - risk)),
        fmt="o",
        color=palette["blue"],
        capsize=2.5,
    )
    overall_risk = float(overall.iloc[0]["outcome_rate_pct"])
    axes[1].axhline(
        overall_risk, color=palette["neutral"], linestyle="--", linewidth=0.9
    )
    axes[1].text(
        1.02,
        overall_risk,
        "Overall",
        transform=axes[1].get_yaxis_transform(),
        va="center",
    )
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Observed mortality risk (%)")
    axes[1].set_title("Absolute mortality risk", loc="left", pad=10)
    add_panel_label(axes[1], "b", x=-0.12, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="e1_main_figure_1_denominator_and_absolute_risk",
        height_mm=82.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Sepsis-3 denominator",
                "role": "cohort_accounting",
                "article_role": "cohort_accounting",
                "chart_type": "denominator_bar",
                "claim": "The two registered exposure strata exhaust the frozen cohort denominator.",
                "evidence_ids": [_sha256(paths["outcomes"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["outcomes"]],
                },
            },
            {
                "panel_id": "b",
                "title": "Absolute mortality risk",
                "role": "descriptive_result",
                "article_role": "descriptive_result",
                "chart_type": "dot_interval_absolute_risk",
                "claim": "Observed mortality risk and 95% confidence intervals are shown by registered Sepsis-3 status.",
                "evidence_ids": [_sha256(paths["outcomes"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["outcomes"]],
                },
            },
        ],
        source_data=[source_files["outcomes"]],
        core_claim="The frozen cohort denominator and absolute mortality risk differ by registered Sepsis-3 status.",
        statistics_note="Cohort shares, risks and 95% confidence intervals are copied from the frozen exposure-outcome table; the overall risk is a source row, not a refit.",
    )

    adjusted = frames["adjusted"].copy()
    _finite(adjusted, ("estimate", "ci_low", "ci_high"))
    if len(adjusted) != 1 or not adjusted["fit_status"].astype(str).eq("fitted").all():
        raise ValueError("E1 adjusted table must contain one fitted primary contrast")
    fig, ax = plt.subplots(figsize=(89 / 25.4, 70 / 25.4), constrained_layout=True)
    _forest(
        ax,
        adjusted,
        ["Sepsis-3 vs no Sepsis-3"],
        "Primary adjusted association",
        palette["blue"],
    )
    add_panel_label(ax, "a", x=-0.17, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="e1_main_figure_2_adjusted_association",
        width_mm=89.0,
        height_mm=70.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Primary adjusted association",
                "role": "primary_estimand",
                "article_role": "primary_estimand",
                "chart_type": "forest",
                "claim": "The registered adjusted mortality odds ratio is shown with its 95% confidence interval.",
                "evidence_ids": [_sha256(paths["adjusted"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["adjusted"]],
                },
            }
        ],
        source_data=[source_files["adjusted"]],
        core_claim="The registered primary adjusted association is displayed separately from descriptive risk.",
        statistics_note="One fitted primary contrast is copied from the frozen adjusted-association table; no model is refit.",
    )

    sensitivity = frames["sensitivity"].copy()
    _finite(sensitivity, ("estimate", "ci_low", "ci_high"))
    if not sensitivity["converged"].astype(str).str.casefold().eq("true").all():
        raise ValueError("E1 sensitivity table contains a non-converged row")
    fig, ax = plt.subplots(figsize=(183 / 25.4, 86 / 25.4), constrained_layout=True)
    _forest(
        ax,
        sensitivity,
        [str(value).replace("_", " ") for value in sensitivity["analysis_id"]],
        "Registered scientific sensitivity analyses",
        palette["orange"],
    )
    add_panel_label(ax, "a", x=-0.12, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="e1_main_figure_3_definition_and_cohort_sensitivity",
        height_mm=86.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Registered scientific sensitivity analyses",
                "role": "robustness",
                "article_role": "robustness",
                "chart_type": "sensitivity_forest",
                "claim": "Every converged registered sensitivity analysis is shown without post-hoc filtering.",
                "evidence_ids": [_sha256(paths["sensitivity"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["sensitivity"]],
                },
            }
        ],
        source_data=[source_files["sensitivity"]],
        core_claim="The primary association is placed beside, but not conflated with, prespecified definition and cohort sensitivities.",
        statistics_note="All four converged source rows are displayed as odds ratios with 95% confidence intervals.",
    )

    completeness = frames["completeness"].copy()
    _finite(completeness, ("n_stratum", "measured_pct", "value_missing_pct"))
    matrix = completeness.pivot(
        index="variable", columns="exposure_category", values="measured_pct"
    )
    desired = [value for value in ["__all__", "0", "1"] if value in matrix.columns]
    matrix = matrix[desired]
    fig, ax = plt.subplots(figsize=(183 / 25.4, 112 / 25.4), constrained_layout=True)
    image = ax.imshow(
        matrix.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=100, cmap="Blues"
    )
    ax.set_yticks(
        np.arange(len(matrix)), [str(value).replace("_", " ") for value in matrix.index]
    )
    ax.set_xticks(
        np.arange(len(desired)), ["Overall", "No Sepsis-3", "Sepsis-3"][: len(desired)]
    )
    ax.set_title("Component measurement availability", loc="left", pad=10)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Measured (%)")
    add_panel_label(ax, "a", x=-0.12, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="e1_supplementary_figure_s1_component_availability",
        height_mm=112.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Component measurement availability",
                "role": "data_quality",
                "article_role": "data_quality",
                "chart_type": "availability_heatmap",
                "claim": "All registered component-availability rows are retained as supplementary audit evidence.",
                "evidence_ids": [_sha256(paths["completeness"])],
                "metadata": {
                    "placement": "supplementary",
                    "source_data": [source_files["completeness"]],
                },
            }
        ],
        source_data=[source_files["completeness"]],
        core_claim="Routine component completeness is visible without occupying the primary clinical-result figures.",
        statistics_note="Measured percentages are copied from every source row and arranged by variable and exposure stratum; missingness is not interpreted as a clinical effect.",
    )
    return _task_summary(
        source=source_dir,
        paths=paths,
        figure_files=figure_files,
        main_figure_count=3,
        supplementary_figure_count=1,
    )


def _render_landmark_association(
    *,
    task_id: str,
    source_dir: Path,
    out_dir: Path,
    exposure_label: str,
    curve_file: str,
    contrast_path: Path,
    measurement_file: str,
    measurement_is_main: bool,
) -> dict[str, Any]:
    paths, frames = _load_frozen_tables(
        source_dir,
        {
            "absolute_risk": "absolute_risk_context_source_data.csv",
            "curve": curve_file,
            "measurement": measurement_file,
        },
    )
    if not contrast_path.is_file():
        raise FileNotFoundError(f"Frozen contrast source is missing: {contrast_path}")
    paths["contrasts"] = contrast_path
    frames["contrasts"] = pd.read_csv(contrast_path)
    source_files = _copy_frozen_tables(
        out_dir=out_dir, task_id=task_id, paths=paths, frames=frames
    )
    palette = apply_publication_style(font_size=7.0)
    figure_files: list[str] = []

    context = frames["absolute_risk"].copy()
    prevalence = context[context["prevalence_pct"].notna()].copy()
    outcomes = context[context["outcome_risk_pct"].notna()].copy()
    if len(prevalence) != 2 or len(outcomes) != 2:
        raise ValueError(
            f"{task_id} absolute-risk context must contain two source states"
        )
    _finite(prevalence, ("prevalence_pct", "estimate", "ci_low", "ci_high"))
    _finite(outcomes, ("outcome_risk_pct", "estimate", "ci_low", "ci_high"))
    prevalence = prevalence.reset_index(drop=True)
    outcomes = outcomes.reset_index(drop=True)
    labels = [_measurement_state_label(value) for value in prevalence["group_value"]]
    colors = [palette["blue"], palette["neutral"]]
    fig, axes = plt.subplots(
        1, 2, figsize=(183 / 25.4, 82 / 25.4), constrained_layout=True
    )
    positions = np.arange(2)
    axes[0].bar(positions, prevalence["prevalence_pct"], color=colors)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Cohort share (%)")
    axes[0].set_title(f"{exposure_label} measurement status", loc="left", pad=10)
    add_panel_label(axes[0], "a", x=-0.12, y=1.04, fontsize=8.0)
    risk = outcomes["outcome_risk_pct"].to_numpy(dtype=float)
    low = 100.0 * outcomes["ci_low"].to_numpy(dtype=float)
    high = 100.0 * outcomes["ci_high"].to_numpy(dtype=float)
    axes[1].errorbar(
        positions,
        risk,
        yerr=np.vstack((risk - low, high - risk)),
        fmt="o",
        color=palette["blue"],
        capsize=2.5,
    )
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Observed mortality risk (%)")
    axes[1].set_title("Outcome risk by measurement status", loc="left", pad=10)
    add_panel_label(axes[1], "b", x=-0.12, y=1.04, fontsize=8.0)
    source_state_placement = "main" if measurement_is_main else "supplementary"
    source_state_product = (
        f"{task_id}_main_figure_1_source_state_and_absolute_risk"
        if measurement_is_main
        else f"{task_id}_supplementary_figure_s1_measurement_context"
    )
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product=source_state_product,
        height_mm=82.0,
        panels=[
            {
                "panel_id": "a",
                "title": f"{exposure_label} measurement status",
                "role": "cohort_accounting",
                "article_role": "cohort_accounting",
                "chart_type": "source_state_bar",
                "claim": "Measured and not-measured states exhaust the frozen denominator; this is measurement availability, not structural source absence.",
                "evidence_ids": [_sha256(paths["absolute_risk"])],
                "metadata": {
                    "placement": source_state_placement,
                    "source_data": [source_files["absolute_risk"]],
                },
            },
            {
                "panel_id": "b",
                "title": "Outcome risk by measurement status",
                "role": "descriptive_result",
                "article_role": "descriptive_result",
                "chart_type": "dot_interval_absolute_risk",
                "claim": "Observed mortality risk is separated from the continuous exposure association.",
                "evidence_ids": [_sha256(paths["absolute_risk"])],
                "metadata": {
                    "placement": source_state_placement,
                    "source_data": [source_files["absolute_risk"]],
                },
            },
        ],
        source_data=[source_files["absolute_risk"]],
        core_claim=f"{exposure_label} measurement availability and observed outcome risk are shown before the continuous association.",
        statistics_note="Source-state prevalence, mortality risk and 95% confidence intervals are copied from the frozen absolute-risk context table.",
    )

    curve = frames["curve"].copy().sort_values("exposure_value")
    contrasts = frames["contrasts"].copy().sort_values("exposure_value")
    _finite(curve, ("exposure_value", "adjusted_odds_ratio", "ci_low", "ci_high"))
    _finite(
        contrasts,
        (
            "exposure_value",
            "reference_exposure_value",
            "adjusted_odds_ratio",
            "ci_low",
            "ci_high",
        ),
    )
    if (
        not (curve["ci_low"] <= curve["adjusted_odds_ratio"]).all()
        or not (curve["adjusted_odds_ratio"] <= curve["ci_high"]).all()
    ):
        raise ValueError(f"{task_id} curve intervals are reversed")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 90 / 25.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    x = curve["exposure_value"].to_numpy(dtype=float)
    estimate = curve["adjusted_odds_ratio"].to_numpy(dtype=float)
    low = curve["ci_low"].to_numpy(dtype=float)
    high = curve["ci_high"].to_numpy(dtype=float)
    axes[0].plot(x, estimate, color=palette["blue"], linewidth=1.4)
    axes[0].fill_between(
        x, low, high, color=palette["blue_soft"], alpha=0.75, linewidth=0
    )
    axes[0].axhline(1.0, color=palette["neutral"], linestyle="--", linewidth=0.8)
    axes[0].axvline(
        float(curve.iloc[0]["reference_exposure_value"]),
        color=palette["neutral"],
        linestyle=":",
        linewidth=0.8,
    )
    axes[0].set_xlabel(exposure_label)
    axes[0].set_ylabel("Adjusted mortality odds ratio (95% CI)")
    axes[0].set_title("Continuous dose-response", loc="left", pad=10)
    add_panel_label(axes[0], "a", x=-0.12, y=1.04, fontsize=8.0)
    if contrasts["reference_exposure_value"].nunique() != 1:
        raise ValueError(f"{task_id} contrast rows do not share one reference")
    if (
        not (contrasts["ci_low"] <= contrasts["adjusted_odds_ratio"]).all()
        or not (contrasts["adjusted_odds_ratio"] <= contrasts["ci_high"]).all()
    ):
        raise ValueError(f"{task_id} contrast intervals are reversed")
    positions = np.arange(len(contrasts))
    contrast_estimates = contrasts["adjusted_odds_ratio"].to_numpy(dtype=float)
    contrast_low = contrasts["ci_low"].to_numpy(dtype=float)
    contrast_high = contrasts["ci_high"].to_numpy(dtype=float)
    axes[1].errorbar(
        contrast_estimates,
        positions,
        xerr=np.vstack(
            (contrast_estimates - contrast_low, contrast_high - contrast_estimates)
        ),
        fmt="o",
        color=palette["orange"],
        capsize=2.5,
    )
    axes[1].axvline(1.0, color=palette["neutral"], linestyle="--", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlim(
        max(0.85 * min(1.0, float(contrast_low.min())), np.finfo(float).tiny),
        1.12 * max(1.0, float(contrast_high.max())),
    )
    reference = float(contrasts["reference_exposure_value"].iloc[0])
    exposure_unit = (
        exposure_label.rsplit("(", 1)[1].rstrip(")")
        if "(" in exposure_label and exposure_label.endswith(")")
        else "exposure units"
    )
    axes[1].set_yticks(
        positions,
        [
            f"{float(value):g} vs {reference:g} {exposure_unit}"
            for value in contrasts["exposure_value"]
        ],
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Adjusted mortality odds ratio (95% CI)")
    axes[1].set_title("Selected contrasts vs reference", loc="left", pad=10)
    add_panel_label(axes[1], "b", x=-0.14, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product=f"{task_id}_main_figure_2_continuous_association_and_contrasts",
        height_mm=90.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Continuous dose-response",
                "role": "primary_estimand",
                "article_role": "primary_estimand",
                "chart_type": "dose_response_curve",
                "claim": "Every frozen grid point and its 95% confidence interval are shown.",
                "evidence_ids": [_sha256(paths["curve"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["curve"]],
                },
            },
            {
                "panel_id": "b",
                "title": "Selected contrasts vs reference",
                "role": "primary_estimand",
                "article_role": "primary_estimand",
                "chart_type": "contrast_forest",
                "claim": "All displayed contrasts use the same fitted curve, effect scale and frozen reference value.",
                "evidence_ids": [_sha256(paths["contrasts"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["contrasts"]],
                    "effect_comparison_authorized": True,
                    "contrast_reference": reference,
                },
            },
        ],
        source_data=[source_files["curve"], source_files["contrasts"]],
        core_claim=f"The continuous {exposure_label.lower()} association and two prespecified contrasts sharing one reference are displayed as primary evidence.",
        statistics_note="The curve and confidence band use every frozen grid row. Contrast points and 95% confidence intervals are copied from the frozen registered contrast table; no robustness-summary envelope is plotted as an effect interval.",
    )

    measurement = frames["measurement"].copy()
    _finite(measurement, ("n_total", "measured_one_n", "repeat_measured_n"))
    measurement["measured_pct_display"] = (
        100.0 * measurement["measured_one_n"] / measurement["n_total"]
    )
    measurement["repeat_pct_display"] = (
        100.0 * measurement["repeat_measured_n"] / measurement["n_total"]
    )
    measurement_source = _copy_source(
        measurement,
        paths["measurement"],
        out_dir / f"{task_id}_measurement_display_source_data.csv",
    )
    ordered = measurement.sort_values("measured_pct_display")
    measurement_label_column = (
        "variable" if "variable" in ordered.columns else "concept"
    )
    if measurement_label_column not in ordered.columns:
        raise ValueError(f"{task_id} measurement audit has no variable label column")
    fig, ax = plt.subplots(figsize=(183 / 25.4, 105 / 25.4), constrained_layout=True)
    positions = np.arange(len(ordered))
    ax.barh(
        positions,
        ordered["measured_pct_display"],
        color=palette["blue_soft"],
        label="Measured",
    )
    ax.barh(
        positions,
        ordered["repeat_pct_display"],
        color=palette["blue"],
        label="Repeated measurement",
    )
    ax.set_yticks(
        positions,
        [str(value).replace("_", " ") for value in ordered[measurement_label_column]],
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Eligible records (%)")
    ax.set_title("Measurement-process audit", loc="left", pad=10)
    ax.legend(loc="lower right")
    add_panel_label(ax, "a", x=-0.12, y=1.04, fontsize=8.0)
    placement = "main" if measurement_is_main else "supplementary"
    product = (
        f"{task_id}_main_figure_3_measurement_process"
        if measurement_is_main
        else f"{task_id}_supplementary_figure_s2_measurement_process"
    )
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product=product,
        height_mm=105.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Measurement-process audit",
                "role": "data_quality",
                "article_role": "data_quality",
                "chart_type": "measurement_frequency_bar",
                "claim": "Measured and repeated-measurement fractions are shown for every registered variable.",
                "evidence_ids": [_sha256(paths["measurement"])],
                "metadata": {
                    "placement": placement,
                    "central_to_question": measurement_is_main,
                    "source_data": [measurement_source],
                },
            }
        ],
        source_data=[measurement_source],
        core_claim=(
            "Measurement availability is a primary scientific result because source absence is the research question."
            if measurement_is_main
            else "Detailed measurement availability is retained as supplementary validity evidence."
        ),
        statistics_note="Percentages are deterministic display calculations from frozen total, measured and repeat-measured counts; all registered variables are shown.",
    )
    return _task_summary(
        source=source_dir,
        paths=paths,
        figure_files=figure_files,
        main_figure_count=3 if measurement_is_main else 1,
        supplementary_figure_count=0 if measurement_is_main else 2,
    )


def _render_m3(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    paths, frames = _load_frozen_tables(
        source_dir,
        {
            "profiles": "phenotype_profiles_source_data.csv",
            "assignments": "phenotype_assignments_source_data.csv",
            "stability": "cluster_stability_source_data.csv",
        },
    )
    source_files = _copy_frozen_tables(
        out_dir=out_dir, task_id="m3", paths=paths, frames=frames
    )
    palette = apply_publication_style(font_size=7.0)
    figure_files: list[str] = []

    profiles = frames["profiles"].copy()
    _finite(profiles, ("cluster", "standardised_centroid", "n", "missing_pct"))
    matrix = profiles.pivot(
        index="cluster", columns="variable", values="standardised_centroid"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(183 / 25.4, 72 / 25.4), constrained_layout=True)
    limit = max(1.0, float(np.nanmax(np.abs(matrix.to_numpy(dtype=float)))))
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_yticks(
        np.arange(len(matrix)),
        [f"Candidate cluster {int(value) + 1}" for value in matrix.index],
    )
    ax.set_xticks(
        np.arange(len(matrix.columns)),
        [str(value).replace("_max", "").replace("_", " ") for value in matrix.columns],
        rotation=35,
        ha="right",
    )
    ax.set_title("Standardised candidate-cluster profiles", loc="left", pad=10)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Standardised centroid")
    add_panel_label(ax, "a", x=-0.12, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="m3_main_figure_1_candidate_cluster_profiles",
        height_mm=72.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Standardised candidate-cluster profiles",
                "role": "phenotype_structure",
                "article_role": "phenotype_structure",
                "chart_type": "profile_heatmap",
                "claim": "Every frozen standardised centroid is shown without assigning phenotype names.",
                "evidence_ids": [_sha256(paths["profiles"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["profiles"]],
                },
            }
        ],
        source_data=[source_files["profiles"]],
        core_claim="The frozen candidate clustering has two distinct descriptive profiles, without establishing biological phenotypes.",
        statistics_note="Standardised centroids are copied from every frozen profile row. Candidate numbers are neutral labels and imply no clinical entity.",
    )

    assignments = frames["assignments"].copy()
    _finite(assignments, ("cluster",))
    sizes = assignments.groupby("cluster", sort=True).size().rename("n").reset_index()
    sizes["percentage"] = 100.0 * sizes["n"] / sizes["n"].sum()
    size_source = _copy_source(
        sizes, paths["assignments"], out_dir / "m3_cluster_size_source_data.csv"
    )
    fig, ax = plt.subplots(figsize=(89 / 25.4, 72 / 25.4), constrained_layout=True)
    positions = np.arange(len(sizes))
    ax.bar(
        positions,
        sizes["percentage"],
        color=[palette["blue"], palette["teal"]][: len(sizes)],
    )
    ax.set_xticks(
        positions, [f"Candidate cluster {int(value) + 1}" for value in sizes["cluster"]]
    )
    ax.set_ylabel("Assigned records (%)")
    ax.set_title("Candidate-cluster size", loc="left", pad=10)
    for position, row in zip(positions, sizes.itertuples(index=False), strict=True):
        ax.text(
            position,
            row.percentage + 1.0,
            f"n={int(row.n):,}",
            ha="center",
            va="bottom",
        )
    add_panel_label(ax, "a", x=-0.17, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="m3_main_figure_2_candidate_cluster_size",
        width_mm=89.0,
        height_mm=72.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Candidate-cluster size",
                "role": "phenotype_profile",
                "article_role": "phenotype_profile",
                "chart_type": "cluster_size_bar",
                "claim": "The complete frozen assignment table is summarized as candidate-cluster counts and shares.",
                "evidence_ids": [_sha256(paths["assignments"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [size_source, source_files["assignments"]],
                },
            }
        ],
        source_data=[size_source, source_files["assignments"]],
        core_claim="Candidate-cluster size is reported separately from profile structure and stability.",
        statistics_note="Counts are deterministic tabulations of all frozen assignments; no reassignment or clustering is performed.",
    )

    stability = frames["stability"].copy().sort_values("replicate")
    _finite(
        stability,
        (
            "replicate",
            "adjusted_rand_index",
            "mean_adjusted_rand_index",
            "algorithm_agreement_ari",
        ),
    )
    if (
        stability["mean_adjusted_rand_index"].nunique() != 1
        or stability["algorithm_agreement_ari"].nunique() != 1
    ):
        raise ValueError("M3 stability summary is inconsistent across replicates")
    fig, ax = plt.subplots(figsize=(183 / 25.4, 78 / 25.4), constrained_layout=True)
    ax.plot(
        stability["replicate"],
        stability["adjusted_rand_index"],
        "o-",
        color=palette["blue"],
        label="Subsample refit ARI",
    )
    ax.axhline(
        float(stability.iloc[0]["mean_adjusted_rand_index"]),
        color=palette["blue"],
        linestyle="--",
        label="Mean refit ARI",
    )
    ax.axhline(
        float(stability.iloc[0]["algorithm_agreement_ari"]),
        color=palette["orange"],
        linestyle=":",
        label="Alternative-algorithm ARI",
    )
    ax.set_ylim(0, 1)
    ax.set_xticks(stability["replicate"].astype(int))
    ax.set_xlabel("Fixed-seed stability replicate")
    ax.set_ylabel("Adjusted Rand index")
    ax.set_title(
        "Candidate-cluster stability and algorithm agreement", loc="left", pad=10
    )
    ax.legend(loc="upper right")
    add_panel_label(ax, "a", x=-0.12, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="m3_main_figure_3_stability_and_algorithm_agreement",
        height_mm=78.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Candidate-cluster stability and algorithm agreement",
                "role": "stability",
                "article_role": "stability",
                "chart_type": "stability_trace",
                "claim": "All fixed-seed refit ARIs and the frozen alternative-algorithm agreement are shown.",
                "evidence_ids": [_sha256(paths["stability"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["stability"]],
                },
            }
        ],
        source_data=[source_files["stability"]],
        core_claim="Low resampling and algorithm agreement limit interpretation of the candidate clustering.",
        statistics_note="ARI values are copied from all five registered replicates; the horizontal references are source-reported mean and algorithm-agreement values.",
    )
    return _task_summary(
        source=source_dir,
        paths=paths,
        figure_files=figure_files,
        main_figure_count=3,
        supplementary_figure_count=0,
    )


def _render_h1(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    table_names = {
        "km": "landmark_km_curve.csv",
        "cox": "landmark_cox_summary.csv",
        "risk_flow": "landmark_risk_set_flow.csv",
        "ph": "landmark_ph_diagnostics.csv",
        "rmst": "landmark_rmst_summary.csv",
    }
    time_varying_name = "landmark_time_varying_cox_summary.csv"
    if (source_dir / time_varying_name).is_file():
        table_names["time_varying"] = time_varying_name
    paths, frames = _load_frozen_tables(
        source_dir,
        table_names,
    )
    source_files = _copy_frozen_tables(
        out_dir=out_dir, task_id="h1", paths=paths, frames=frames
    )
    palette = apply_publication_style(font_size=7.0)
    figure_files: list[str] = []

    km = frames["km"].copy()
    _finite(
        km,
        (
            "exposure_group",
            "time_from_landmark_days",
            "survival_probability",
            "at_risk",
            "group_n",
            "group_events",
        ),
    )
    groups = sorted(km["exposure_group"].unique())
    if groups != [0, 1]:
        raise ValueError("H1 Kaplan-Meier table lacks both registered groups")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(183 / 25.4, 126 / 25.4),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )
    labels = {0: "No incident ventilation by 24 h", 1: "Incident ventilation by 24 h"}
    colors = {0: palette["neutral"], 1: palette["blue"]}
    for group in groups:
        subset = km[km["exposure_group"].eq(group)].sort_values(
            "time_from_landmark_days"
        )
        axes[0].step(
            subset["time_from_landmark_days"],
            subset["survival_probability"],
            where="post",
            color=colors[group],
            label=labels[group],
            linewidth=1.4,
        )
    axes[0].set_ylim(0.82, 1.005)
    axes[0].set_xlabel("Days from 24-h landmark")
    axes[0].set_ylabel("Survival probability")
    axes[0].set_title("Unadjusted post-landmark survival", loc="left", pad=10)
    axes[0].legend(loc="lower left")
    add_panel_label(axes[0], "a", x=-0.08, y=1.04, fontsize=8.0)
    risk_times = [0, 7, 14, 21, 27]
    risk_rows: list[list[str]] = []
    for group in groups:
        subset = km[km["exposure_group"].eq(group)].set_index("time_from_landmark_days")
        risk_rows.append(
            [f"{int(subset.loc[time, 'at_risk']):,}" for time in risk_times]
        )
    axes[1].axis("off")
    table = axes[1].table(
        cellText=risk_rows,
        rowLabels=[labels[group] for group in groups],
        colLabels=[str(time) for time in risk_times],
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    axes[1].set_title("Number at risk", loc="left", pad=4)
    add_panel_label(axes[1], "b", x=-0.08, y=1.04, fontsize=8.0)
    risk_source = out_dir / "h1_selected_risk_table_source_data.csv"
    risk_frame = pd.DataFrame(
        risk_rows, index=[labels[group] for group in groups], columns=risk_times
    ).reset_index(names="exposure_group")
    risk_frame.to_csv(risk_source, index=False)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="h1_main_figure_1_landmark_survival",
        height_mm=126.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Unadjusted post-landmark survival",
                "role": "temporal_absolute_risk",
                "article_role": "temporal_absolute_risk",
                "chart_type": "kaplan_meier",
                "claim": "The complete frozen post-landmark survival curves are shown for both exposure groups.",
                "evidence_ids": [_sha256(paths["km"])],
                "metadata": {"placement": "main", "source_data": [source_files["km"]]},
            },
            {
                "panel_id": "b",
                "title": "Number at risk",
                "role": "cohort_accounting",
                "article_role": "cohort_accounting",
                "chart_type": "risk_table",
                "claim": "The risk set is displayed at fixed reader-facing time points.",
                "evidence_ids": [_sha256(paths["km"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [risk_source.name, source_files["km"]],
                },
            },
        ],
        source_data=[source_files["km"], risk_source.name],
        core_claim="Post-landmark survival is shown with the denominator remaining under observation.",
        statistics_note="Kaplan-Meier estimates are unadjusted. The risk table selects prespecified display days from the same frozen source without interpolation.",
    )

    rmst = frames["rmst"].copy()
    _finite(
        rmst,
        (
            "tau_days_from_landmark",
            "exposed_rmst_days",
            "exposed_rmst_ci_low",
            "exposed_rmst_ci_high",
            "comparator_rmst_days",
            "comparator_rmst_ci_low",
            "comparator_rmst_ci_high",
            "rmst_difference_days",
            "ci_low",
            "ci_high",
        ),
    )
    if len(rmst) != 1:
        raise ValueError("H1 RMST summary must contain exactly one registered contrast")
    row = rmst.iloc[0]
    estimates = np.array(
        [row["comparator_rmst_days"], row["exposed_rmst_days"]], dtype=float
    )
    lows = np.array(
        [row["comparator_rmst_ci_low"], row["exposed_rmst_ci_low"]], dtype=float
    )
    highs = np.array(
        [row["comparator_rmst_ci_high"], row["exposed_rmst_ci_high"]], dtype=float
    )
    fig, ax = plt.subplots(figsize=(89 / 25.4, 76 / 25.4), constrained_layout=True)
    positions = np.arange(2)
    ax.errorbar(
        positions,
        estimates,
        yerr=np.vstack((estimates - lows, highs - estimates)),
        fmt="o",
        color=palette["blue"],
        capsize=2.5,
    )
    ax.set_xticks(positions, ["No incident\nventilation", "Incident\nventilation"])
    ax.set_ylabel(
        f"Restricted mean survival through {row['tau_days_from_landmark']:.0f} days"
    )
    ax.set_title("PH-free descriptive survival contrast", loc="left", pad=10)
    ax.text(
        0.02,
        0.04,
        f"Difference {row['rmst_difference_days']:.2f} days\n95% CI {row['ci_low']:.2f} to {row['ci_high']:.2f}",
        transform=ax.transAxes,
        va="bottom",
    )
    add_panel_label(ax, "a", x=-0.17, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="h1_main_figure_2_rmst_contrast",
        width_mm=89.0,
        height_mm=76.0,
        panels=[
            {
                "panel_id": "a",
                "title": "PH-free descriptive survival contrast",
                "role": "survival_effect",
                "article_role": "survival_effect",
                "chart_type": "rmst_interval",
                "claim": "The frozen unadjusted restricted-mean survival contrast is shown because a constant Cox effect is not reportable.",
                "evidence_ids": [_sha256(paths["rmst"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["rmst"]],
                },
            }
        ],
        source_data=[source_files["rmst"]],
        core_claim="A proportional-hazards-free descriptive contrast replaces the blocked constant hazard-ratio interpretation.",
        statistics_note="RMST estimates, 95% confidence intervals and the unadjusted difference are copied from the frozen RMST summary.",
    )

    if "time_varying" in frames:
        time_varying = frames["time_varying"].copy()
        exposure_rows = time_varying.loc[
            time_varying["is_exposure"].astype(bool)
        ].sort_values("interval_index")
        _finite(
            exposure_rows,
            (
                "interval_start_days",
                "interval_end_days",
                "hazard_ratio",
                "ci_low",
                "ci_high",
            ),
        )
        if len(exposure_rows) < 2:
            raise ValueError("H1 time-varying sensitivity lacks interval estimates")
        estimates = exposure_rows["hazard_ratio"].to_numpy(dtype=float)
        lows = exposure_rows["ci_low"].to_numpy(dtype=float)
        highs = exposure_rows["ci_high"].to_numpy(dtype=float)
        positions = np.arange(len(exposure_rows))
        labels = [
            f"{row.interval_start_days:g}–{row.interval_end_days:g} days"
            for row in exposure_rows.itertuples()
        ]
        fig, ax = plt.subplots(figsize=(89 / 25.4, 82 / 25.4), constrained_layout=True)
        ax.errorbar(
            estimates,
            positions,
            xerr=np.vstack((estimates - lows, highs - estimates)),
            fmt="o",
            color=palette["blue"],
            capsize=2.5,
        )
        ax.axvline(1.0, color=palette["neutral"], linestyle="--", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Adjusted interval-specific hazard ratio (95% CI)")
        ax.set_title("Time-varying adjusted association", loc="left", pad=10)
        add_panel_label(ax, "a", x=-0.19, y=1.04, fontsize=8.0)
        figure_files += _save(
            fig,
            out_dir=out_dir,
            product="h1_main_figure_3_time_varying_association",
            width_mm=89.0,
            height_mm=82.0,
            panels=[
                {
                    "panel_id": "a",
                    "title": "Time-varying adjusted association",
                    "role": "survival_effect",
                    "article_role": "survival_effect",
                    "chart_type": "time_varying_hazard_ratio_forest",
                    "claim": "The prespecified extended Cox sensitivity reports interval-specific adjusted associations instead of one invalid constant effect.",
                    "evidence_ids": [_sha256(paths["time_varying"])],
                    "metadata": {
                        "placement": "main",
                        "source_data": [source_files["time_varying"]],
                    },
                }
            ],
            source_data=[source_files["time_varying"]],
            core_claim="The adjusted association is allowed to vary over post-landmark follow-up after the constant-effect assumption was rejected.",
            statistics_note="Interval cut points and all covariate-by-interval interactions are prespecified in the signed runtime authority. Wald 95% confidence intervals remain observational and analysis-only.",
        )

    risk_flow = frames["risk_flow"].copy().sort_values("stage_order")
    ph = frames["ph"].copy()
    _finite(
        risk_flow, ("stage_order", "count", "source_denominator", "percent_of_source")
    )
    _finite(ph, ("p_value", "declared_alpha"))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 100 / 25.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    positions = np.arange(len(risk_flow))
    axes[0].barh(positions, risk_flow["percent_of_source"], color=palette["blue_soft"])
    axes[0].set_yticks(
        positions, [str(value).replace("_", " ") for value in risk_flow["stage"]]
    )
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 105)
    axes[0].set_xlabel("Source denominator retained (%)")
    axes[0].set_title("Risk-set accounting", loc="left", pad=10)
    add_panel_label(axes[0], "a", x=-0.12, y=1.04, fontsize=8.0)
    ph_values = -np.log10(ph["p_value"].to_numpy(dtype=float))
    positions = np.arange(len(ph))
    axes[1].barh(positions, ph_values, color=palette["red_soft"])
    axes[1].axvline(
        -np.log10(float(ph["declared_alpha"].iloc[0])),
        color=palette["red"],
        linestyle="--",
        linewidth=0.9,
    )
    axes[1].set_yticks(
        positions, [str(value).replace("_", " ") for value in ph["covariate"]]
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("−log10(P), Schoenfeld test")
    axes[1].set_title("Proportional-hazards diagnostics", loc="left", pad=10)
    add_panel_label(axes[1], "b", x=-0.14, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product=(
            "h1_main_figure_4_risk_set_and_ph_diagnostics"
            if "time_varying" in frames
            else "h1_main_figure_3_risk_set_and_ph_diagnostics"
        ),
        height_mm=100.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Risk-set accounting",
                "role": "cohort_accounting",
                "article_role": "cohort_accounting",
                "chart_type": "cohort_attrition_bar",
                "claim": "Every frozen denominator gate is shown.",
                "evidence_ids": [_sha256(paths["risk_flow"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["risk_flow"]],
                },
            },
            {
                "panel_id": "b",
                "title": "Proportional-hazards diagnostics",
                "role": "diagnostics",
                "article_role": "diagnostics",
                "chart_type": "ph_diagnostic_bar",
                "claim": "All Schoenfeld-residual tests disclose the assumption failure that blocks a constant Cox effect.",
                "evidence_ids": [_sha256(paths["ph"]), _sha256(paths["cox"])],
                "metadata": {
                    "placement": "main",
                    "source_data": [source_files["ph"], source_files["cox"]],
                },
            },
        ],
        source_data=[
            source_files["risk_flow"],
            source_files["ph"],
            source_files["cox"],
        ],
        core_claim="Risk-set loss and proportional-hazards failure materially limit the survival interpretation.",
        statistics_note="All risk-set stages and Schoenfeld-test rows are shown. The frozen Cox table is retained as audit source data but its constant exposure hazard ratio is not plotted as a reportable effect.",
    )
    return _task_summary(
        source=source_dir,
        paths=paths,
        figure_files=figure_files,
        main_figure_count=4 if "time_varying" in frames else 3,
        supplementary_figure_count=0,
        reason_code=(
            "H1_CONSTANT_HR_WITHHELD_TIME_VARYING_SENSITIVITY_AVAILABLE"
            if "time_varying" in frames
            else "H1_PROPORTIONAL_HAZARDS_REJECTED"
        ),
    )


def _render_h2(source_run: Path, out_dir: Path) -> dict[str, Any]:
    path = source_run / H2_FEASIBILITY_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"H2 feasibility receipt is missing: {path}")
    frame = pd.read_csv(path)
    required = [
        "verified_non_use_available",
        "binary_control_arm_authorized",
        "causal_contrast_authorized",
    ]
    if len(frame) != 1 or any(bool(frame.iloc[0][column]) for column in required):
        raise ValueError(
            "H2 source authority no longer supports the registered fail-closed result"
        )
    if str(frame.iloc[0]["decision"]) != "fail_closed" or pd.notna(
        frame.iloc[0]["effect_estimate"]
    ):
        raise ValueError(
            "H2 feasibility receipt contains an unauthorized causal result"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    source_file = _copy_source(
        frame, path, out_dir / "h2_source_feasibility_source_data.csv"
    )
    palette = apply_publication_style(font_size=7.0)
    labels = [
        "Verified non-use comparator",
        "Binary control arm",
        "Causal contrast",
        "Effect estimate",
    ]
    statuses = ["Unavailable", "Not authorized", "Not authorized", "Not estimated"]
    fig, ax = plt.subplots(figsize=(89 / 25.4, 75 / 25.4), constrained_layout=True)
    positions = np.arange(len(labels))[::-1]
    for position, label, status in zip(positions, labels, statuses, strict=True):
        ax.axhline(position, color=palette["neutral_light"], linewidth=0.6, zorder=0)
        ax.text(0.02, position, label, ha="left", va="center", color="#272727")
        ax.scatter(
            [0.66],
            [position],
            s=64,
            color=palette["red_soft"],
            edgecolor=palette["red"],
            zorder=2,
        )
        ax.text(
            0.66,
            position,
            "×",
            ha="center",
            va="center",
            color=palette["red"],
            fontsize=10,
            fontweight="bold",
            zorder=3,
        )
        ax.text(
            0.74,
            position,
            status,
            ha="left",
            va="center",
            color=palette["red"],
            fontweight="bold",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.9, 3.65)
    ax.axis("off")
    ax.set_title("Causal-contrast identifiability", loc="left", pad=10)
    ax.text(
        0.02,
        -0.62,
        "Analysis stops before estimation",
        color=palette["red"],
        fontweight="bold",
    )
    add_panel_label(ax, "a", x=-0.17, y=1.04, fontsize=8.0)
    figure_files = _save(
        fig,
        out_dir=out_dir,
        product="h2_main_figure_1_causal_identifiability_diagnostic",
        width_mm=89.0,
        height_mm=75.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Causal-contrast identifiability",
                "role": "causal_protocol",
                "article_role": "causal_protocol",
                "chart_type": "identifiability_status_matrix",
                "claim": "Verified non-use, a binary control arm and a causal contrast are all unauthorized; no effect estimate exists.",
                "evidence_ids": [_sha256(path)],
                "metadata": {
                    "placement": "main",
                    "display_purpose": "diagnostic",
                    "scientific_status": "failed_closed",
                    "terminal_diagnostic": True,
                    "central_to_question": True,
                    "source_data": [source_file],
                },
            }
        ],
        source_data=[source_file],
        core_claim="The available source cannot identify a verified non-use comparator, so the causal analysis stops before estimation.",
        statistics_note="This is a categorical source-authority receipt, not an effect figure. Crosses encode frozen false states; no patient-level analysis or effect calculation is performed.",
    )
    return _task_summary(
        source=source_run,
        paths={"feasibility": path},
        figure_files=figure_files,
        main_figure_count=1,
        supplementary_figure_count=0,
        scientific_status="failed_closed",
        reason_code=str(frame.iloc[0]["reason_code"]),
    )


def _render_h3(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    paths, frames = _load_frozen_tables(
        source_dir,
        {
            "selection": "trajectory_selection_bic_source_data.csv",
            "availability": "trajectory_selection_availability_source_data.csv",
        },
    )
    selection = frames["selection"].copy().sort_values("n_clusters")
    availability = frames["availability"].copy()
    _finite(selection, ("n_clusters", "bic", "aic"))
    _finite(availability, ("observed_n", "missing_n", "missing_fraction"))
    if not selection["scientific_status"].astype(str).eq("failed_closed").all():
        raise ValueError("H3 selection table is no longer fail closed")
    if not bool(selection.iloc[-1]["upper_boundary"]) or not bool(
        selection.iloc[-1]["selected"]
    ):
        raise ValueError("H3 source no longer records the upper-boundary minimum")
    source_files = _copy_frozen_tables(
        out_dir=out_dir, task_id="h3", paths=paths, frames=frames
    )
    palette = apply_publication_style(font_size=7.0)
    fig, ax = plt.subplots(figsize=(89 / 25.4, 70 / 25.4), constrained_layout=True)
    ax.plot(
        selection["n_clusters"],
        selection["bic"],
        "o-",
        color=palette["blue"],
        label="BIC",
    )
    ax.plot(
        selection["n_clusters"],
        selection["aic"],
        "o--",
        color=palette["neutral"],
        label="AIC diagnostic",
    )
    ax.scatter(
        [selection.iloc[-1]["n_clusters"]],
        [selection.iloc[-1]["bic"]],
        s=80,
        facecolors="none",
        edgecolors=palette["red"],
        linewidths=1.2,
    )
    ax.set_xlabel("Candidate number of trajectory classes")
    ax.set_ylabel("Information criterion")
    ax.set_title("Prespecified candidate-grid assessment", loc="left", pad=10)
    ax.legend()
    ax.text(
        0.98,
        0.95,
        "Minimum at upper boundary\nNo class solution authorized",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=palette["red"],
        fontsize=6.5,
    )
    add_panel_label(ax, "a", x=-0.17, y=1.04, fontsize=8.0)
    figure_files = _save(
        fig,
        out_dir=out_dir,
        product="h3_main_figure_1_candidate_selection_diagnostic",
        width_mm=89.0,
        height_mm=70.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Prespecified candidate-grid assessment",
                "role": "phenotype_structure",
                "article_role": "phenotype_structure",
                "chart_type": "information_criterion_trace",
                "claim": "The minimum BIC occurs at the upper candidate boundary, so no interior solution is authorized.",
                "evidence_ids": [_sha256(paths["selection"])],
                "metadata": {
                    "placement": "main",
                    "display_purpose": "diagnostic",
                    "scientific_status": "failed_closed",
                    "terminal_diagnostic": True,
                    "central_to_question": True,
                    "source_data": [source_files["selection"]],
                },
            }
        ],
        source_data=[source_files["selection"]],
        core_claim="The prespecified candidate grid did not establish an interior trajectory-class solution.",
        statistics_note="BIC and AIC values are copied from every candidate row. The upper-boundary minimum is reported as a failed selection diagnostic; no class is selected, named, or related to an outcome.",
    )
    parts = (
        availability["feature"]
        .astype(str)
        .str.extract(r"^(?P<component>.+)__h(?P<start>\d+)_(?P<end>\d+)$")
    )
    if parts.isna().any().any():
        raise ValueError(
            "H3 availability labels do not match the registered component-window schema"
        )
    availability = availability.assign(
        component=parts["component"], window=parts["start"] + "–" + parts["end"] + " h"
    )
    matrix = availability.pivot(
        index="component", columns="window", values="missing_fraction"
    )
    ordered_windows = sorted(
        matrix.columns, key=lambda value: int(str(value).split("–")[0])
    )
    matrix = matrix[ordered_windows]
    fig, ax = plt.subplots(figsize=(125 / 25.4, 92 / 25.4), constrained_layout=True)
    image = ax.imshow(
        matrix.to_numpy(dtype=float) * 100.0,
        aspect="auto",
        vmin=0,
        vmax=100,
        cmap="Reds",
    )
    ax.set_yticks(
        np.arange(len(matrix)),
        [str(value).replace("sofa2_", "") for value in matrix.index],
    )
    ax.set_xticks(
        np.arange(len(matrix.columns)), matrix.columns, rotation=35, ha="right"
    )
    ax.set_title("Trajectory-coordinate missingness", loc="left", pad=10)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Missing (%)")
    add_panel_label(ax, "a", x=-0.15, y=1.04, fontsize=8.0)
    figure_files += _save(
        fig,
        out_dir=out_dir,
        product="h3_supplementary_figure_s1_feature_availability",
        width_mm=125.0,
        height_mm=92.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Trajectory-coordinate missingness",
                "role": "data_quality",
                "article_role": "data_quality",
                "chart_type": "availability_heatmap",
                "claim": "Every prespecified component-window availability row is shown as diagnostic evidence.",
                "evidence_ids": [_sha256(paths["availability"])],
                "metadata": {
                    "placement": "supplementary",
                    "display_purpose": "audit",
                    "scientific_status": "failed_closed",
                    "source_data": [source_files["availability"]],
                },
            },
        ],
        source_data=[source_files["availability"]],
        core_claim="Trajectory-coordinate availability is disclosed as supplementary audit evidence.",
        statistics_note="Missingness percentages are deterministic displays of every frozen component-window row; no class is named or selected.",
    )
    return _task_summary(
        source=source_dir,
        paths=paths,
        figure_files=figure_files,
        main_figure_count=1,
        supplementary_figure_count=1,
        scientific_status="failed_closed",
        reason_code="H3_NO_INTERIOR_BIC_OPTIMUM",
    )


def main() -> int:
    _require_current_worktree_import()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--visual-source-root", type=Path, required=True)
    parser.add_argument("--h1-source-dir", type=Path)
    parser.add_argument("--h2-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    visual_source_root = args.visual_source_root.resolve()
    h1_source_dir = (
        args.h1_source_dir.resolve()
        if args.h1_source_dir is not None
        else visual_source_root / "h1"
    )
    h2_run = args.h2_run.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    e1 = _render_e1(visual_source_root / "e1", output_root / "e1")
    _, e2_contrast_path = _evidence_table(
        run_dir=source_root / RUN_RELATIVES["e2"],
        step_id="primary_adjusted_association",
        basename="e2_landmark_rcs_contrasts.csv",
    )
    e2 = _render_landmark_association(
        task_id="e2",
        source_dir=visual_source_root / "e2",
        out_dir=output_root / "e2",
        exposure_label="Maximum lactate (mmol/L)",
        curve_file="e2_landmark_rcs_curve_source_data.csv",
        contrast_path=e2_contrast_path,
        measurement_file="measurement_process_source_data.csv",
        measurement_is_main=False,
    )
    e3_run = source_root / E3_RUN_RELATIVE
    m2_run = source_root / M2_RUN_RELATIVE
    e3 = _render_e3(e3_run, output_root / "e3")
    _, m1_contrast_path = _evidence_table(
        run_dir=source_root / RUN_RELATIVES["m1"],
        step_id="primary_adjusted_estimate",
        basename="m1_landmark_bilirubin_contrasts.csv",
    )
    m1 = _render_landmark_association(
        task_id="m1",
        source_dir=visual_source_root / "m1",
        out_dir=output_root / "m1",
        exposure_label="Maximum bilirubin (mg/dL)",
        curve_file="m1_landmark_bilirubin_curve_source_data.csv",
        contrast_path=m1_contrast_path,
        measurement_file="measurement_process_audit_source_data.csv",
        measurement_is_main=True,
    )
    m2_summary = run_prediction_figure(
        out_dir=output_root / "m2",
        run_dir=m2_run,
        resolved_inputs=m2_run / "resolved_inputs/prediction_figure_suite.json",
        step_id="prediction_figure_suite",
        figure_product="m2_main_figure_prediction_performance",
    )
    m2_summary.update(
        {
            "authority_scope": "analysis_only",
            "paper_authorization_allowed": False,
            "main_figure_count": 2,
            "supplementary_figure_count": 1,
        }
    )
    m3 = _render_m3(visual_source_root / "m3", output_root / "m3")
    h1 = _render_h1(h1_source_dir, output_root / "h1")
    h2 = _render_h2(h2_run, output_root / "h2")
    h3 = _render_h3(visual_source_root / "h3", output_root / "h3")
    table_summaries = _package_article_tables(
        source_root=source_root, output_root=output_root
    )
    task_summaries = {
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "m1": m1,
        "m2": m2_summary,
        "m3": m3,
        "h1": h1,
        "h2": h2,
        "h3": h3,
    }
    for task_id, table_summary in table_summaries.items():
        task_summaries[task_id].update(table_summary)
    for task_id, summary in task_summaries.items():
        status_contract = {
            "schema_version": "easyicu.article_display_status/1",
            "task_id": task_id,
            "scientific_status": summary.get("scientific_status", "analysis_only"),
            "authority_scope": "analysis_only",
            "paper_authorization_allowed": False,
            "reason_code": summary.get("reason_code"),
        }
        (output_root / task_id / "article_display_status.json").write_text(
            json.dumps(
                status_contract,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    article_inventories = {
        task_id: inspect_article_display_package(output_root / task_id)
        for task_id in task_summaries
    }
    inventory_path = output_root / "article_display_inventory.json"
    inventory_path.write_text(
        json.dumps(
            article_inventories,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        code_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        code_head = "unknown"
    manifest = {
        "schema_version": "easyicu.dev9_article_display_remediation/3",
        "code_head": code_head,
        "source_root": str(source_root),
        "visual_source_root": str(visual_source_root),
        "h2_run": str(h2_run),
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "provider_calls": 0,
        "scientific_recomputation": False,
        "article_display_inventory": inventory_path.name,
        **task_summaries,
    }
    (output_root / "article_display_remediation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

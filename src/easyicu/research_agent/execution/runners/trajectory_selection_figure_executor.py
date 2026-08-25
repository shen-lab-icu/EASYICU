"""Deterministic diagnostic figure for signed trajectory-class selection.

This owner renders two already-governed tables.  It cannot choose K, refit a
model, recover candidate labels, or turn a failed-closed selection into a
phenotype claim.
"""

from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...figures.display_labels import display_label
from ...schema import AnalysisStep
from .typed_input_binding import load_typed_input

TRAJECTORY_SELECTION_TABLE = "table:trajectory_candidate_selection"
TRAJECTORY_AVAILABILITY_TABLE = "table:feature_availability"
TRAJECTORY_SELECTION_FIGURE = "figure:trajectory_selection_diagnostics"
TRAJECTORY_SELECTION_FIGURE_INPUTS = (
    TRAJECTORY_SELECTION_TABLE,
    TRAJECTORY_AVAILABILITY_TABLE,
)
TRAJECTORY_SELECTION_FIGURE_METHOD = "signed_trajectory_selection_diagnostic_figure"

_SELECTION_COLUMNS = (
    "n_clusters",
    "bic",
    "aic",
    "final_log_likelihood",
    "parameter_count",
    "selected",
    "aic_minimum",
    "upper_boundary",
    "scientific_status",
    "reason_code",
    "reportable_result",
)
_AVAILABILITY_COLUMNS = (
    "feature",
    "observed_n",
    "missing_n",
    "missing_fraction",
)


def _boolean_column(series: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalised = series.astype(str).str.strip().str.lower()
    if not set(normalised).issubset({"true", "false", "1", "0"}):
        raise ValueError(f"trajectory {label} column is not boolean")
    return normalised.map({"true": True, "false": False, "1": True, "0": False})


def trajectory_selection_figure_executor_owns_step(step: AnalysisStep) -> bool:
    return bool(
        step.planned_analysis_role == "auxiliary"
        and step.method == TRAJECTORY_SELECTION_FIGURE_METHOD
        and tuple(step.inputs) == TRAJECTORY_SELECTION_FIGURE_INPUTS
        and tuple(step.expected_outputs) == (TRAJECTORY_SELECTION_FIGURE,)
        and step.table_one_spec is None
        and step.cohort_definition_spec is None
        and step.measurement_audit_spec is None
        and step.robustness_replay_spec is None
        and step.trajectory_stability_spec is None
        and not step.model_requirements
    )


def trajectory_selection_figure_executor_code(step: AnalysisStep) -> str:
    if not trajectory_selection_figure_executor_owns_step(step):
        raise ValueError("step is not owned by the trajectory selection figure")
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.trajectory_selection_figure_executor import run_trajectory_selection_figure

        summary = run_trajectory_selection_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        """
    ).strip()


def _validated_selection(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    selected = frame.copy()
    selected["n_clusters"] = pd.to_numeric(selected["n_clusters"], errors="coerce")
    selected["bic"] = pd.to_numeric(selected["bic"], errors="coerce")
    for column in ("aic", "final_log_likelihood", "parameter_count"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if (
        selected.empty
        or selected[list(_SELECTION_COLUMNS)].isna().any().any()
        or not np.isfinite(selected["n_clusters"]).all()
        or not np.isfinite(selected["bic"]).all()
        or not np.isfinite(
            selected[["aic", "final_log_likelihood", "parameter_count"]]
        ).all().all()
        or selected["n_clusters"].duplicated().any()
    ):
        raise ValueError("trajectory candidate-selection table is incomplete")
    selected["selected"] = _boolean_column(selected["selected"], label="selected")
    selected["aic_minimum"] = _boolean_column(
        selected["aic_minimum"], label="aic_minimum"
    )
    selected["upper_boundary"] = _boolean_column(
        selected["upper_boundary"], label="upper_boundary"
    )
    if int(selected["selected"].sum()) != 1:
        raise ValueError("trajectory selection must identify exactly one candidate")
    if int(selected["aic_minimum"].sum()) != 1:
        raise ValueError("trajectory AIC diagnostic must identify exactly one minimum")
    winner = selected.loc[selected["selected"]].iloc[0]
    expected_k = int(
        selected.sort_values(["bic", "n_clusters"], ascending=[True, True]).iloc[0][
            "n_clusters"
        ]
    )
    if int(winner["n_clusters"]) != expected_k:
        raise ValueError("selected trajectory candidate is not the minimum-BIC row")
    statuses = {str(value).strip() for value in selected["scientific_status"]}
    reasons = {str(value).strip() for value in selected["reason_code"]}
    if len(statuses) != 1 or len(reasons) != 1:
        raise ValueError("trajectory scientific decision is inconsistent across rows")
    failed_closed = statuses == {"failed_closed"}
    reason_code = next(iter(reasons))
    if failed_closed and (not bool(winner["upper_boundary"]) or not reason_code):
        raise ValueError("failed-closed trajectory selection lacks its boundary reason")
    if not failed_closed and statuses != {"selected"}:
        raise ValueError("trajectory scientific status is unsupported")
    return selected.sort_values("n_clusters"), failed_closed, reason_code


def _validated_availability(frame: pd.DataFrame) -> pd.DataFrame:
    availability = frame.copy()
    for column in ("observed_n", "missing_n", "missing_fraction"):
        availability[column] = pd.to_numeric(availability[column], errors="coerce")
    if (
        availability.empty
        or availability[list(_AVAILABILITY_COLUMNS)].isna().any().any()
        or availability["feature"].astype(str).duplicated().any()
        or not np.isfinite(
            availability[["observed_n", "missing_n", "missing_fraction"]]
        )
        .all()
        .all()
    ):
        raise ValueError("trajectory feature-availability table is incomplete")
    denominators = availability["observed_n"] + availability["missing_n"]
    if (denominators <= 0).any() or denominators.nunique() != 1:
        raise ValueError("trajectory availability denominators are inconsistent")
    recomputed = availability["missing_n"] / denominators
    if not np.allclose(
        recomputed,
        availability["missing_fraction"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("trajectory missing fractions disagree with their counts")
    availability["available_pct"] = 100.0 * availability["observed_n"] / denominators
    return availability


def run_trajectory_selection_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
) -> dict[str, Any]:
    """Render BIC and availability from two exact, digest-bound source tables."""

    selection_bound = load_typed_input(
        input_key=TRAJECTORY_SELECTION_TABLE,
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        expected_columns=_SELECTION_COLUMNS,
        require_consumption_contract=True,
        minimum_row_count=2,
    )
    availability_bound = load_typed_input(
        input_key=TRAJECTORY_AVAILABILITY_TABLE,
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        expected_declared_kind="table",
        expected_evidence_kind="table",
        expected_columns=_AVAILABILITY_COLUMNS,
        require_consumption_contract=True,
        minimum_row_count=2,
    )
    selection, failed_closed, reason_code = _validated_selection(selection_bound.frame)
    availability = _validated_availability(availability_bound.frame)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_source = out_dir / "trajectory_selection_bic_source_data.csv"
    availability_source = out_dir / "trajectory_selection_availability_source_data.csv"
    selection_parent_step = str(
        selection_bound.binding.get("produced_by_step") or ""
    )
    availability_parent_step = str(
        availability_bound.binding.get("produced_by_step") or ""
    )
    if not selection_parent_step or not availability_parent_step:
        raise ValueError("trajectory figure parents lack producer-step lineage")
    selection_projection = selection.copy()
    selection_projection["source_row_index"] = range(len(selection_projection))
    selection_projection["source_table"] = selection_bound.path.name
    selection_projection["source_step_id"] = selection_parent_step
    selection_projection.to_csv(selection_source, index=False)
    # Keep the registered source-data bytes as a row/value projection of the
    # exact parent table. ``available_pct`` is a renderer-local derivation from
    # the two count columns and must not masquerade as an upstream value.
    availability_projection = availability.loc[:, list(_AVAILABILITY_COLUMNS)].copy()
    availability_projection["source_table"] = availability_bound.path.name
    availability_projection["source_step_id"] = availability_parent_step
    availability_projection.to_csv(availability_source, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    fig, (ax_bic, ax_availability) = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 88 / 25.4),
        gridspec_kw={"width_ratios": [0.94, 1.30]},
    )
    ax_bic.plot(
        selection["n_clusters"],
        selection["bic"],
        color=palette["blue"],
        marker="o",
        linewidth=1.4,
        markersize=4.0,
        label="BIC (selection)",
    )
    if "aic" in selection.columns:
        ax_bic.plot(
            selection["n_clusters"],
            selection["aic"],
            color=palette["orange"],
            marker="s",
            linewidth=1.1,
            markersize=3.2,
            label="AIC (diagnostic)",
        )
    winner = selection.loc[selection["selected"]].iloc[0]
    ax_bic.scatter(
        [winner["n_clusters"]],
        [winner["bic"]],
        color=palette["red"] if failed_closed else palette["blue"],
        edgecolor="white",
        linewidth=0.7,
        s=44,
        zorder=4,
    )
    ax_bic.set_xticks(selection["n_clusters"])
    ax_bic.set_xlabel("Candidate number of classes (K)")
    ax_bic.set_ylabel("Information criterion")
    ax_bic.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_bic.set_title("Prespecified candidate-grid assessment", loc="left", pad=5)
    if "aic" in selection.columns:
        ax_bic.legend(frameon=False, fontsize=5.8, loc="upper right")

    concepts: list[str] = []
    windows: list[str] = []
    values: dict[tuple[str, str], float] = {}
    for row in availability.itertuples(index=False):
        concept, separator, window = str(row.feature).partition("__h")
        if not separator:
            concept, window = str(row.feature), "overall"
        if concept not in concepts:
            concepts.append(concept)
        if window not in windows:
            windows.append(window)
        values[(concept, window)] = float(row.available_pct)
    if set(values) != {(concept, window) for concept in concepts for window in windows}:
        raise ValueError("trajectory availability does not form a complete grid")
    availability_matrix = np.asarray(
        [[values[(concept, window)] for window in windows] for concept in concepts]
    )
    heatmap = ax_availability.imshow(
        availability_matrix,
        aspect="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
    )
    ax_availability.set_xticks(range(len(windows)))
    ax_availability.set_xticklabels(
        [f"{window.replace('_', '–')} h" for window in windows]
    )
    ax_availability.set_yticks(range(len(concepts)))
    ax_availability.set_yticklabels([display_label(value) for value in concepts])
    ax_availability.set_xlabel("Prespecified ICU time window")
    ax_availability.set_title("Observed coordinate availability", loc="left", pad=5)
    for row_index in range(len(concepts)):
        for column_index in range(len(windows)):
            value = availability_matrix[row_index, column_index]
            ax_availability.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if value >= 58.0 else palette["blue"],
            )
    colourbar = fig.colorbar(heatmap, ax=ax_availability, fraction=0.045, pad=0.04)
    colourbar.set_label("Available (%)")
    add_panel_label(ax_bic, "a", x=-0.16, y=1.05, fontsize=8.0)
    add_panel_label(ax_availability, "b", x=-0.15, y=1.05, fontsize=8.0)
    status_text = (
        "Fail closed: the minimum occurred at the upper candidate boundary; no trajectory-class solution is authorised."
        if failed_closed
        else "Interior candidate selected; stability and external reproducibility remain separate requirements."
    )
    fig.text(
        0.10,
        0.035,
        status_text,
        fontsize=6.2,
        color=palette["red"] if failed_closed else palette["blue"],
    )
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.24, top=0.86, wspace=0.38)

    if failed_closed:
        core_claim = (
            "The prespecified candidate grid did not establish an authorised "
            "trajectory-class solution; the diagnostic is reportable only as a "
            "fail-closed selection result."
        )
        selection_claim = (
            "The minimum BIC occurred at the upper candidate boundary, so no "
            "interior solution was established; AIC is shown only as a "
            "prespecified diagnostic and cannot alter the selection."
        )
    else:
        core_claim = (
            "The prespecified candidate grid selected an interior candidate, "
            "while stability remains a separately governed requirement."
        )
        selection_claim = (
            "The minimum BIC occurred at an interior candidate; this panel alone "
            "does not establish cluster stability or clinical phenotypes."
        )
    contract = make_figure_contract(
        figure_id=TRAJECTORY_SELECTION_FIGURE,
        core_claim=core_claim,
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=88.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Prespecified candidate-grid assessment",
                "role": "phenotype_structure",
                "claim": selection_claim,
                "evidence_ids": [selection_source.name],
                "metadata": {
                    "article_role": "phenotype_structure",
                    "chart_type": "criterion_curve",
                    "source_data": [selection_source.name],
                },
            },
            {
                "panel_id": "b",
                "title": "Observed coordinate availability",
                "role": "data_quality",
                "claim": (
                    "Availability is shown for every prespecified coordinate and "
                    "time window among rows admitted to candidate fitting."
                ),
                "evidence_ids": [availability_source.name],
                "metadata": {
                    "article_role": "data_quality",
                    "chart_type": "availability_heatmap",
                    "source_data": [availability_source.name],
                },
            },
        ],
        source_data=[selection_source.name, availability_source.name],
        statistics_note=(
            "BIC values come from every model in the signed candidate grid and "
            "remain the only selection criterion. When present, AIC is a "
            "secondary diagnostic computed from the same fits. "
            "Availability is re-derived from the exact producer counts. "
            "Candidate labels are not displayed as validated phenotypes."
        ),
    )
    stem = out_dir / TRAJECTORY_SELECTION_FIGURE.split(":", 1)[1]
    outputs = save_publication_figure(
        fig,
        stem,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
    )
    plt.close(fig)
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": TRAJECTORY_SELECTION_FIGURE_METHOD,
        "analysis_family": "phenotyping",
        "deterministic_standard_analysis": "trajectory_selection_diagnostic_figure",
        "rendering_only": True,
        "scientific_status": "failed_closed" if failed_closed else "selected",
        "reason_code": reason_code if failed_closed else None,
        "candidate_count": int(len(selection)),
        "coordinate_count": int(len(availability)),
        "figure_path": f"{stem.name}.png",
        "figure_contract": f"{stem.name}.figure_contract.json",
        "figure_files": figure_files,
        "contract_files": [f"{stem.name}.figure_contract.json"],
        "source_data_files": [selection_source.name, availability_source.name],
        "output_files": {TRAJECTORY_SELECTION_FIGURE: f"{stem.name}.png"},
        "input_bindings": [
            {
                "input_key": bound.input_key,
                "evidence_id": bound.evidence_id,
                "sha256": bound.sha256,
                "loaded": True,
                "row_count": bound.row_count,
            }
            for bound in (selection_bound, availability_bound)
        ],
        "export_qa": [],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


__all__ = [
    "TRAJECTORY_SELECTION_FIGURE",
    "TRAJECTORY_SELECTION_FIGURE_INPUTS",
    "TRAJECTORY_SELECTION_FIGURE_METHOD",
    "run_trajectory_selection_figure",
    "trajectory_selection_figure_executor_code",
    "trajectory_selection_figure_executor_owns_step",
]

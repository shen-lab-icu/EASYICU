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
TRAJECTORY_PROFILE_TABLE = "table:trajectory_profiles"
TRAJECTORY_CLUSTER_SIZE_TABLE = "table:cluster_sizes"
TRAJECTORY_STABILITY_TABLE = "table:cluster_stability"
TRAJECTORY_SELECTION_FIGURE = "figure:trajectory_selection_diagnostics"
#: The second declared surface. One exported image is one surface, and the
#: end-of-execute join resolves a runtime contract per declared output, so the
#: phenotype characterization cannot be extra panels on the diagnostic figure.
TRAJECTORY_CHARACTERIZATION_FIGURE = "figure:trajectory_phenotype_characterization"
TRAJECTORY_SELECTION_FIGURE_INPUTS = (
    TRAJECTORY_SELECTION_TABLE,
    TRAJECTORY_AVAILABILITY_TABLE,
    TRAJECTORY_PROFILE_TABLE,
    TRAJECTORY_CLUSTER_SIZE_TABLE,
    TRAJECTORY_STABILITY_TABLE,
)
TRAJECTORY_SELECTION_FIGURE_OUTPUTS = (
    TRAJECTORY_SELECTION_FIGURE,
    TRAJECTORY_CHARACTERIZATION_FIGURE,
)
TRAJECTORY_SELECTION_FIGURE_METHOD = "signed_trajectory_selection_diagnostic_figure"
_PROFILE_COLUMNS = (
    "cluster",
    "source_column",
    "window_start_hours",
    "window_end_hours",
    "summary_statistic",
    "value",
    "n_observed",
)
_CLUSTER_SIZE_COLUMNS = ("cluster", "n")
_STABILITY_COLUMNS = (
    "resample_id",
    "n_overlap",
    "adjusted_rand_index",
    "selected_n_clusters",
)

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
        and tuple(step.expected_outputs) == TRAJECTORY_SELECTION_FIGURE_OUTPUTS
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


def _source_projection(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    bound: Any,
    path: Path,
) -> None:
    """Write one registered source-data projection of an exact parent table."""

    projection = frame.loc[:, list(columns)].copy()
    parent = str(bound.binding.get("produced_by_step") or "")
    if not parent:
        raise ValueError("trajectory figure parents lack producer-step lineage")
    projection["source_row_index"] = range(len(projection))
    projection["source_table"] = bound.path.name
    projection["source_step_id"] = parent
    projection.to_csv(path, index=False)


def _no_solution_axis(axis: Any, *, title: str, reason_code: str) -> None:
    """State the sealed decision instead of drawing a phenotype that has none."""

    axis.set_title(title, loc="left", pad=5)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(
        0.5,
        0.5,
        "No stable phenotype solution\n"
        + (reason_code or "prespecified reportability rule not met"),
        ha="center",
        va="center",
        fontsize=6.4,
        wrap=True,
    )


def _profile_matrix(profiles: pd.DataFrame) -> tuple[Any, list[str], list[str], list[int]]:
    """Coordinate x (cluster, window) means, ordered exactly as the table is."""

    frame = profiles.copy()
    frame["concept"] = frame["source_column"].astype(str).str.split("__h").str[0]
    frame["window_start_hours"] = pd.to_numeric(
        frame["window_start_hours"], errors="coerce"
    )
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    concepts = sorted(dict.fromkeys(frame["concept"]))
    clusters = sorted(dict.fromkeys(frame["cluster"].astype(str)))
    windows = sorted(dict.fromkeys(frame["window_start_hours"].dropna().tolist()))
    matrix = np.full((len(concepts), len(clusters) * len(windows)), np.nan)
    for _, row in frame.iterrows():
        if pd.isna(row["window_start_hours"]):
            continue
        r = concepts.index(str(row["concept"]))
        c = clusters.index(str(row["cluster"])) * len(windows) + windows.index(
            row["window_start_hours"]
        )
        matrix[r, c] = row["value"]
    return matrix, concepts, clusters, [int(value) for value in windows]


def _render_characterization_figure(
    *,
    out_dir: Path,
    profiles: pd.DataFrame,
    sizes: pd.DataFrame,
    stability: pd.DataFrame,
    bounds: tuple[Any, Any, Any],
    failed_closed: bool,
    reason_code: str,
) -> dict[str, Any]:
    """Render the phenotype characterization surface from three sealed tables."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profile_bound, size_bound, stability_bound = bounds
    profile_source = out_dir / "trajectory_profiles_source_data.csv"
    size_source = out_dir / "trajectory_cluster_sizes_source_data.csv"
    stability_source = out_dir / "trajectory_cluster_stability_source_data.csv"
    _source_projection(
        profiles, columns=_PROFILE_COLUMNS, bound=profile_bound, path=profile_source
    )
    _source_projection(
        sizes, columns=_CLUSTER_SIZE_COLUMNS, bound=size_bound, path=size_source
    )
    _source_projection(
        stability,
        columns=_STABILITY_COLUMNS,
        bound=stability_bound,
        path=stability_source,
    )

    palette = apply_publication_style(font_size=7.0)
    fig, (ax_profile, ax_structure, ax_stability) = plt.subplots(
        1, 3, figsize=(183 / 25.4, 66 / 25.4), gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]}
    )
    solution = not failed_closed and not profiles.empty and not sizes.empty
    if solution:
        matrix, concepts, clusters, windows = _profile_matrix(profiles)
        image = ax_profile.imshow(matrix, aspect="auto", cmap="viridis")
        ax_profile.set_yticks(range(len(concepts)))
        ax_profile.set_yticklabels(
            [display_label(concept) for concept in concepts], fontsize=5.6
        )
        ax_profile.set_xticks(
            [
                index * len(windows) + len(windows) / 2 - 0.5
                for index in range(len(clusters))
            ]
        )
        ax_profile.set_xticklabels([f"Class {name}" for name in clusters], fontsize=5.8)
        for index in range(1, len(clusters)):
            ax_profile.axvline(index * len(windows) - 0.5, color="white", linewidth=1.1)
        ax_profile.set_xlabel(
            f"Prespecified {windows[1] - windows[0] if len(windows) > 1 else 0}-hour "
            "windows within each class"
        )
        bar = fig.colorbar(image, ax=ax_profile, fraction=0.045, pad=0.02)
        bar.ax.tick_params(labelsize=5.4)
        bar.set_label("Pooled z-scored coordinate mean", fontsize=5.6)
        ax_profile.set_title("Class coordinate profiles", loc="left", pad=5)

        collapsed = np.full((len(clusters), len(concepts)), np.nan)
        for row_index in range(len(concepts)):
            for column_index in range(len(clusters)):
                block = matrix[
                    row_index,
                    column_index * len(windows) : (column_index + 1) * len(windows),
                ]
                collapsed[column_index, row_index] = np.nanmean(block)
        structure = ax_structure.imshow(collapsed, aspect="auto", cmap="viridis")
        size_by_cluster = {
            str(row["cluster"]): int(row["n"]) for _, row in sizes.iterrows()
        }
        ax_structure.set_yticks(range(len(clusters)))
        ax_structure.set_yticklabels(
            [
                f"Class {name} (n={size_by_cluster.get(str(name), 0)})"
                for name in clusters
            ],
            fontsize=5.6,
        )
        ax_structure.set_xticks(range(len(concepts)))
        ax_structure.set_xticklabels(
            [display_label(concept) for concept in concepts],
            rotation=45,
            ha="right",
            fontsize=5.4,
        )
        bar = fig.colorbar(structure, ax=ax_structure, fraction=0.045, pad=0.02)
        bar.ax.tick_params(labelsize=5.4)
        ax_structure.set_title("Between-class separation", loc="left", pad=5)
    else:
        _no_solution_axis(
            ax_profile, title="Class coordinate profiles", reason_code=reason_code
        )
        _no_solution_axis(
            ax_structure, title="Between-class separation", reason_code=reason_code
        )

    ari = pd.to_numeric(
        stability.get("adjusted_rand_index", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    if len(ari):
        ax_stability.hist(
            ari, bins=min(20, max(5, len(ari) // 5)), color=palette["blue"], alpha=0.85
        )
        mean_ari = float(ari.mean())
        ax_stability.axvline(
            mean_ari,
            color=palette["red"],
            linewidth=1.3,
            label=f"Mean ARI = {mean_ari:.3f}",
        )
        ax_stability.legend(frameon=False, fontsize=5.8, loc="upper left")
        ax_stability.set_xlabel("Adjusted Rand index per prespecified resample")
        ax_stability.set_ylabel("Resamples")
    else:
        _no_solution_axis(
            ax_stability,
            title="Resampling stability",
            reason_code=reason_code or "no completed resample",
        )
    ax_stability.set_title("Resampling stability", loc="left", pad=5)
    for index, axis in enumerate((ax_profile, ax_structure, ax_stability)):
        add_panel_label(axis, "abc"[index])
    fig.tight_layout()

    solution_claim = (
        "Class profiles are the sealed representation's own coordinate means; "
        "they describe the selected partition and do not establish that the "
        "classes are distinct clinical entities."
        if solution
        else "The prespecified reportability rule returned no stable phenotype "
        "solution, so no class profile is drawn."
    )
    contract = make_figure_contract(
        figure_id=TRAJECTORY_CHARACTERIZATION_FIGURE,
        core_claim=(
            "Class profiles, between-class separation, and resampling stability "
            "of the prespecified trajectory solution."
            if solution
            else "No stable phenotype solution was reportable under the "
            "prespecified rule."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=66.0,
        panels=[
            {
                "panel_id": "a",
                "title": "Class coordinate profiles",
                "role": "phenotype_profile",
                "claim": solution_claim,
                "evidence_ids": [profile_source.name],
                "metadata": {
                    "article_role": "phenotype_profile",
                    "chart_type": "profile_heatmap",
                    "source_products": [TRAJECTORY_PROFILE_TABLE],
                    "source_data": [profile_source.name],
                },
            },
            {
                "panel_id": "b",
                "title": "Between-class separation",
                "role": "phenotype_structure",
                "claim": (
                    "Window-averaged coordinate means per class show how far "
                    "apart the selected classes sit; separation is not evidence "
                    "of clinical validity."
                    if solution
                    else solution_claim
                ),
                "evidence_ids": [profile_source.name, size_source.name],
                "metadata": {
                    "article_role": "phenotype_structure",
                    "chart_type": "cluster_heatmap",
                    "source_products": [
                        TRAJECTORY_PROFILE_TABLE,
                        TRAJECTORY_CLUSTER_SIZE_TABLE,
                    ],
                    "source_data": [profile_source.name, size_source.name],
                },
            },
            {
                "panel_id": "c",
                "title": "Resampling stability",
                "role": "stability",
                "claim": (
                    "Adjusted Rand indices come from the sealed subsampling "
                    "design; the threshold decision itself is made by the "
                    "stability owner, not by this figure."
                ),
                "evidence_ids": [stability_source.name],
                "metadata": {
                    "article_role": "stability",
                    "chart_type": "subsampling_ari",
                    "source_products": [TRAJECTORY_STABILITY_TABLE],
                    "source_data": [stability_source.name],
                },
            },
        ],
        source_data=[profile_source.name, size_source.name, stability_source.name],
        reader_caption=(
            "(a) Coordinate means per prespecified window within each class. "
            "(b) Window-averaged coordinate means per class with class sizes. "
            "(c) Distribution of adjusted Rand indices across the prespecified "
            "resamples. No panel establishes clinical validity of the classes."
        ),
        statistics_note=(
            "Every value is copied from the sealed characterization and "
            "stability tables; this renderer refits nothing, relabels nothing, "
            "and applies no threshold of its own."
        ),
    )
    stem = out_dir / TRAJECTORY_CHARACTERIZATION_FIGURE.split(":", 1)[1]
    outputs = save_publication_figure(
        fig, stem, contract=contract, formats=("png", "svg", "pdf", "tiff"), dpi=300
    )
    plt.close(fig)
    return {
        "stem": stem,
        "figure_files": [
            path.name for key, path in outputs.items() if key != "contract"
        ],
        "contract_file": f"{stem.name}.figure_contract.json",
        "source_data_files": [
            profile_source.name,
            size_source.name,
            stability_source.name,
        ],
        "panel_ids": ["a", "b", "c"],
        "reportable_solution": bool(solution),
    }


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
    # The characterization tables are empty by design when the sealed
    # reportability rule returns no stable solution, so they bind with no row
    # floor; the renderer states that outcome instead of drawing a phenotype.
    profile_bound, size_bound, stability_bound = (
        load_typed_input(
            input_key=key,
            run_dir=Path(run_dir),
            resolved_inputs=resolved_inputs,
            step_id=step_id,
            expected_declared_kind="table",
            expected_evidence_kind="table",
            expected_columns=columns,
            require_consumption_contract=True,
            minimum_row_count=0,
        )
        for key, columns in (
            (TRAJECTORY_PROFILE_TABLE, _PROFILE_COLUMNS),
            (TRAJECTORY_CLUSTER_SIZE_TABLE, _CLUSTER_SIZE_COLUMNS),
            (TRAJECTORY_STABILITY_TABLE, _STABILITY_COLUMNS),
        )
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
        label="No interior optimum" if failed_closed else "Selected candidate",
    )
    ax_bic.set_xticks(selection["n_clusters"])
    ax_bic.set_xlabel("Candidate number of classes (K)")
    ax_bic.set_ylabel("Information criterion")
    ax_bic.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_bic.set_title("Prespecified candidate-grid assessment", loc="left", pad=5)
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
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.18, top=0.86, wspace=0.38)

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
                # ``PanelRole`` is the rendered-surface vocabulary; the typed
                # article role below is what the plan promises.
                "role": "diagnostics",
                "claim": selection_claim,
                "evidence_ids": [selection_source.name],
                "metadata": {
                    "article_role": "cluster_selection",
                    "chart_type": "criterion_curve",
                    "source_products": [TRAJECTORY_SELECTION_TABLE],
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
                    "source_products": [TRAJECTORY_AVAILABILITY_TABLE],
                    "source_data": [availability_source.name],
                },
            },
        ],
        source_data=[selection_source.name, availability_source.name],
        reader_caption=(
            "(a) BIC across the prespecified candidate grid; AIC, when shown, "
            "is diagnostic only. (b) Observed coordinate availability by "
            f"prespecified ICU time window. {status_text} "
            "Availability percentages reproduce the producer's counts, not "
            "independent repeated measurements."
        ),
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
    characterization = _render_characterization_figure(
        out_dir=out_dir,
        profiles=profile_bound.frame,
        sizes=size_bound.frame,
        stability=stability_bound.frame,
        bounds=(profile_bound, size_bound, stability_bound),
        failed_closed=failed_closed,
        reason_code=reason_code,
    )
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
        "figure_files": [*figure_files, *characterization["figure_files"]],
        "contract_files": [
            f"{stem.name}.figure_contract.json",
            characterization["contract_file"],
        ],
        "source_data_files": [
            selection_source.name,
            availability_source.name,
            *characterization["source_data_files"],
        ],
        "reportable_phenotype_solution": characterization["reportable_solution"],
        "output_files": {
            TRAJECTORY_SELECTION_FIGURE: f"{stem.name}.png",
            TRAJECTORY_CHARACTERIZATION_FIGURE: f"{characterization['stem'].name}.png",
        },
        # Two declared outputs, so each surface names the panels that answer it
        # rather than letting the join guess.
        "planner_product_slot_bindings": {
            TRAJECTORY_SELECTION_FIGURE: {
                "slot": "selection_diagnostics",
                "panel_ids": ["a", "b"],
            },
            TRAJECTORY_CHARACTERIZATION_FIGURE: {
                "slot": "phenotype_characterization",
                "panel_ids": list(characterization["panel_ids"]),
            },
        },
        "input_bindings": [
            {
                "input_key": bound.input_key,
                "evidence_id": bound.evidence_id,
                "sha256": bound.sha256,
                "loaded": True,
                "row_count": bound.row_count,
            }
            for bound in (
                selection_bound,
                availability_bound,
                profile_bound,
                size_bound,
                stability_bound,
            )
        ],
        "export_qa": [],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


__all__ = [
    "TRAJECTORY_CHARACTERIZATION_FIGURE",
    "TRAJECTORY_SELECTION_FIGURE",
    "TRAJECTORY_SELECTION_FIGURE_OUTPUTS",
    "TRAJECTORY_SELECTION_FIGURE_INPUTS",
    "TRAJECTORY_SELECTION_FIGURE_METHOD",
    "run_trajectory_selection_figure",
    "trajectory_selection_figure_executor_code",
    "trajectory_selection_figure_executor_owns_step",
]

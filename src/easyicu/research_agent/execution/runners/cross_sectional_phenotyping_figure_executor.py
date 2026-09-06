"""Source-bound publication renderer for cross-sectional phenotyping."""

from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ...contracts.figure_plan import (
    CROSS_SECTIONAL_PHENOTYPING_FIGURE_INPUTS,
    CROSS_SECTIONAL_PHENOTYPING_FIGURE_PANELS,
)
from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...figures.display_labels import display_label
from ...schema import AnalysisStep
from .cross_sectional_phenotyping_executor import (
    CLUSTER_STABILITY_PRODUCT,
    PHENOTYPE_ASSIGNMENTS_PRODUCT,
    PHENOTYPE_PROFILES_PRODUCT,
)
from .figure_input_capability import TypedInputCapability
from .typed_input_binding import BoundTypedInput, load_typed_input, sha256_file

PHENOTYPING_FIGURE_INPUTS = CROSS_SECTIONAL_PHENOTYPING_FIGURE_INPUTS
PHENOTYPING_FIGURE_ANALYSIS_KIND = "cross_sectional_phenotyping_figure"
_CAPABILITY = TypedInputCapability(required=frozenset(PHENOTYPING_FIGURE_INPUTS))
_REQUIRED_COLUMNS = {
    PHENOTYPE_PROFILES_PRODUCT: frozenset(
        {"cluster", "variable", "standardised_centroid", "n"}
    ),
    PHENOTYPE_ASSIGNMENTS_PRODUCT: frozenset({"unit_id", "cluster"}),
    CLUSTER_STABILITY_PRODUCT: frozenset(
        {
            "replicate",
            "adjusted_rand_index",
            "mean_adjusted_rand_index",
            "algorithm_agreement_ari",
        }
    ),
}


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if (
        kind != "figure"
        or not separator
        or not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", product)
    ):
        return None
    return product


def _binding_has_columns(binding: Any, key: str) -> bool:
    if not isinstance(binding, Mapping):
        return False
    contract = binding.get("product_contract")
    columns = contract.get("columns") if isinstance(contract, Mapping) else None
    return bool(isinstance(columns, list) and _REQUIRED_COLUMNS[key] <= set(columns))


def cross_sectional_phenotyping_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    products = [_figure_product(value) for value in step.expected_outputs]
    return bool(
        step.planned_analysis_role == "auxiliary"
        and str(step.method or "").strip().casefold().split(" with ", 1)[0]
        == "visualization"
        and len(step.inputs) == len(PHENOTYPING_FIGURE_INPUTS)
        and set(step.inputs) == set(PHENOTYPING_FIGURE_INPUTS)
        and _CAPABILITY.admits_step(step)
        and len(products) == 1
        and products[0] is not None
        and isinstance(resolved_bindings, Mapping)
        and set(resolved_bindings) == set(PHENOTYPING_FIGURE_INPUTS)
        and all(
            _binding_has_columns(resolved_bindings.get(key), key)
            for key in PHENOTYPING_FIGURE_INPUTS
        )
    )


def cross_sectional_phenotyping_figure_executor_code(step: AnalysisStep) -> str:
    product = (
        _figure_product(step.expected_outputs[0]) if step.expected_outputs else None
    )
    if product is None:
        raise ValueError("phenotyping figure has no safe figure product")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.cross_sectional_phenotyping_figure_executor import run_cross_sectional_phenotyping_figure

        run_cross_sectional_phenotyping_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
        )
        """
    ).strip()


def _load_inputs(
    *, run_dir: Path, resolved_inputs: Path | Mapping[str, Any], step_id: str
) -> dict[str, BoundTypedInput]:
    return {
        key: load_typed_input(
            input_key=key,
            run_dir=run_dir,
            resolved_inputs=resolved_inputs,
            step_id=step_id,
            expected_declared_kind="table",
            expected_evidence_kind="table",
            require_consumption_contract=True,
            minimum_row_count=1,
        )
        for key in PHENOTYPING_FIGURE_INPUTS
    }


def _project_assignments(assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A reproducible display transform of all sealed rows, never a cluster fit."""

    columns = [column for column in assignments if column.startswith("feature__")]
    if len(columns) < 2 or len(assignments) < 2:
        raise RuntimeError("phenotype projection requires the sealed feature matrix")
    matrix = assignments[columns].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise RuntimeError("phenotype projection feature matrix is not finite")
    if assignments[["unit_id", "cluster"]].isna().any().any() or assignments["unit_id"].duplicated().any():
        raise RuntimeError("phenotype assignments lack unique units or complete cluster labels")
    projection = PCA(n_components=2, svd_solver="full", whiten=False).fit(matrix)
    if not np.isfinite(projection.explained_variance_ratio_).all():
        raise RuntimeError("phenotype projection has no finite explained variance")
    coordinates = projection.transform(matrix)
    frame = pd.DataFrame({
        "source_row_index": np.arange(len(assignments)),
        "cluster": assignments["cluster"].to_numpy(),
        "pc1": coordinates[:, 0], "pc2": coordinates[:, 1],
    })
    return frame, {
        "schema_version": "easyicu.phenotype_display_projection/1",
        "method": "PCA", "svd_solver": "full", "whiten": False,
        "n_components": 2, "feature_columns": columns,
        "mean": projection.mean_.tolist(), "components": projection.components_.tolist(),
        "explained_variance": projection.explained_variance_.tolist(),
        "explained_variance_ratio": projection.explained_variance_ratio_.tolist(),
        "n_rows": len(assignments), "row_policy": "all_rows",
        "input_representation": "sealed_standardized_primary_matrix",
        "refit_clustering": False, "outcome_used": False,
        "interpretation": "display_only_not_evidence_of_valid_subtypes",
    }


def run_cross_sectional_phenotyping_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
) -> dict[str, Any]:
    if _figure_product(f"figure:{figure_product}") is None:
        raise ValueError("unsafe phenotyping figure product")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bound = _load_inputs(
        run_dir=Path(run_dir), resolved_inputs=resolved_inputs, step_id=step_id
    )
    for key, item in bound.items():
        missing = _REQUIRED_COLUMNS[key] - set(item.frame.columns)
        if missing:
            raise RuntimeError(f"{key} is missing columns {sorted(missing)!r}")
    profiles = bound[PHENOTYPE_PROFILES_PRODUCT].frame.copy()
    assignments = bound[PHENOTYPE_ASSIGNMENTS_PRODUCT].frame.copy()
    stability = bound[CLUSTER_STABILITY_PRODUCT].frame.copy()
    profiles["standardised_centroid"] = pd.to_numeric(
        profiles["standardised_centroid"], errors="coerce"
    )
    wide = profiles.pivot(
        index="cluster",
        columns="variable",
        values="standardised_centroid",
    ).sort_index()
    if wide.shape[0] < 2 or wide.shape[1] < 2 or not np.isfinite(wide.to_numpy()).all():
        raise RuntimeError("phenotyping profile table is not a finite cluster matrix")
    sizes = assignments.groupby("cluster", sort=True).size()
    projection, transform = _project_assignments(assignments)
    features = [column.removeprefix("feature__") for column in transform["feature_columns"]]
    if set(sizes.index) != set(wide.index) or set(wide.columns) != set(features):
        raise RuntimeError("phenotype profiles disagree with sealed cluster or feature labels")
    counts = pd.to_numeric(profiles["n"], errors="coerce")
    if not np.array_equal(counts.to_numpy(), profiles["cluster"].map(sizes).to_numpy()):
        raise RuntimeError("phenotype profile counts disagree with sealed assignments")
    sealed_means = assignments.groupby("cluster")[transform["feature_columns"]].mean()
    sealed_means.columns = features
    if not np.allclose(wide, sealed_means.loc[wide.index, wide.columns], rtol=1e-10, atol=1e-12):
        raise RuntimeError("phenotype centroids disagree with the sealed feature matrix")
    stability_values = pd.to_numeric(stability["adjusted_rand_index"], errors="coerce")
    if not stability_values.between(-1, 1).all() or stability["replicate"].isna().any() or stability["replicate"].duplicated().any():
        raise RuntimeError("phenotyping stability table has invalid agreement or replicate values")
    mean_values = pd.to_numeric(stability["mean_adjusted_rand_index"], errors="coerce")
    if not np.allclose(mean_values, stability_values.mean(), rtol=1e-12, atol=1e-12):
        raise RuntimeError("phenotyping stability mean disagrees with sealed replicates")
    algorithm_values = pd.to_numeric(
        stability.get("algorithm_agreement_ari"), errors="coerce"
    )
    if (
        not algorithm_values.between(-1, 1).all()
        or algorithm_values.nunique() != 1
    ):
        raise RuntimeError("phenotyping algorithm-agreement value is not sealed")
    algorithm_agreement = float(algorithm_values.iloc[0])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_files = []
    for key, item in bound.items():
        filename = f"{key.partition(':')[2]}_source_data.csv"
        source = item.frame.copy()
        parent_name = item.path.name.split("__", 1)[-1]
        source.insert(0, "source_step_id", item.binding.get("produced_by_step"))
        source.insert(0, "source_table", parent_name)
        source.insert(0, "source_row_index", range(len(source)))
        source.to_csv(out_dir / filename, index=False)
        source_files.append(filename)
    projection_filename = "phenotype_projection_source_data.csv"
    projection.to_csv(out_dir / projection_filename, index=False)
    source_files.append(projection_filename)
    transform.update({
        "source_product": PHENOTYPE_ASSIGNMENTS_PRODUCT,
        "source_sha256": bound[PHENOTYPE_ASSIGNMENTS_PRODUCT].sha256,
        "projection_source_data": projection_filename,
        "projection_sha256": sha256_file(out_dir / projection_filename),
    })
    transform_filename = "phenotype_projection_transform.json"
    (out_dir / transform_filename).write_text(
        json.dumps(transform, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    palette = apply_publication_style(font_size=7.0)
    profile_labels = [textwrap.fill(display_label(str(value)), width=25) for value in wide.columns]
    profile_text_rows = len(profile_labels) * max(label.count("\n") + 1 for label in profile_labels)
    height_mm = max(118.0, 30.0 + 3.4 * profile_text_rows)
    fig = plt.figure(figsize=(183 / 25.4, height_mm / 25.4), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.15, 1.0),
        height_ratios=(1.65, 1.0),
    )
    ax_projection = fig.add_subplot(grid[0, 0])
    ax_profiles = fig.add_subplot(grid[:, 1])
    ax_stability = fig.add_subplot(grid[1, 0])
    colours = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    markers = ("o", "^", "s", "D", "v", "P")
    for index, cluster in enumerate(sizes.index):
        points = projection.loc[projection["cluster"] == cluster]
        ax_projection.scatter(
            points["pc1"], points["pc2"], s=5, alpha=0.45,
            color=colours[index % len(colours)], marker=markers[index % len(markers)],
            linewidths=0, rasterized=True, label=f"C{cluster} (n = {sizes[cluster]:,})",
        )
    variance = transform["explained_variance_ratio"]
    ax_projection.set_xlabel(f"PC1 ({variance[0]:.1%} variance)")
    ax_projection.set_ylabel(f"PC2 ({variance[1]:.1%} variance)")
    ax_projection.set_aspect("equal", adjustable="datalim")
    ax_projection.set_title("Candidate-cluster PCA display", loc="left", pad=7)
    ax_projection.legend(frameon=False, fontsize=6, markerscale=1.8, loc="best", ncols=2 if len(sizes) > 3 else 1)
    add_panel_label(ax_projection, "a", x=-0.14, y=1.04, fontsize=8.0)

    centroid_limit = max(1.0, float(np.abs(wide.to_numpy()).max()))
    image = ax_profiles.imshow(
        wide.to_numpy().T, aspect="auto", cmap="RdBu_r",
        vmin=-centroid_limit, vmax=centroid_limit,
    )
    ax_profiles.set_xticks(
        range(len(wide.index)), [f"C{x}" for x in wide.index],
    )
    ax_profiles.set_yticks(
        range(len(wide.columns)), profile_labels,
    )
    ax_profiles.set_title("Clinical feature profiles", loc="left", pad=7)
    fig.colorbar(image, ax=ax_profiles, fraction=0.046, pad=0.03, label="Standardised centroid")
    add_panel_label(ax_profiles, "b", x=-0.18, y=1.04, fontsize=8.0)

    ax_stability.bar(
        range(len(stability)), stability_values, color=palette["orange"]
    )
    ax_stability.set_xticks(range(len(stability)), stability["replicate"].astype(str))
    ax_stability.axhline(0, color="#777777", linewidth=0.5)
    ax_stability.axhline(
        float(stability_values.mean()), color="#333333", linestyle="--", linewidth=0.9,
        label="Subsample mean",
    )
    ax_stability.axhline(
        algorithm_agreement,
        color=palette["blue"],
        linestyle=":",
        linewidth=1.1,
        label="GMM agreement",
    )
    ax_stability.set_ylim(min(0.0, float(stability_values.min()), algorithm_agreement) - 0.08, 1.08)
    ax_stability.set_xlabel("Subsample (fixed preprocessing and K)")
    ax_stability.set_ylabel("Adjusted Rand index")
    ax_stability.set_title("Conditional stability", loc="left", pad=7)
    ax_stability.legend(frameon=False, fontsize=6, loc="best")
    add_panel_label(ax_stability, "c", x=-0.18, y=1.04, fontsize=8.0)

    evidence = {key: item.evidence_id for key, item in bound.items()}
    titles = {
        "a": "Candidate-cluster PCA display", "b": "Clinical feature profiles",
        "c": "Conditional stability and algorithm agreement",
    }
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The analysis-only candidate clustering solution is displayed with "
            "all-row PCA projection, exact standardised profiles, cluster sizes "
            "and conditional subsample agreement; "
            "no phenotype name or biological entity is authorized."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=height_mm,
        panels=[
            {
                "panel_id": panel.panel_id,
                "title": titles[panel.panel_id],
                "role": panel.article_role,
                "claim": "This panel is descriptive and does not establish a biological ground truth or clinical utility.",
                "evidence_ids": [evidence[source] for source in panel.source_products],
                "metadata": {
                    "chart_type": panel.chart_type,
                    "source_products": list(panel.source_products),
                    "source_data": [
                        f"{source.partition(':')[2]}_source_data.csv"
                        for source in panel.source_products
                    ],
                    **({"display_projection": transform, "projection_transform_file": transform_filename}
                       if panel.panel_id == "a" else {}),
                },
            }
            for panel in CROSS_SECTIONAL_PHENOTYPING_FIGURE_PANELS
        ],
        source_data=source_files,
        statistics_note=(
            "Full-SVD PCA centers the sealed standardized matrix without additional "
            "scaling or whitening; all rows and primary cluster labels are retained. "
            "This display does not refit clustering or validate distinct subtypes. "
            "Adjusted Rand indices compare fixed-seed subsample refits with primary "
            "preprocessing and K held fixed; bars are replicates, not confidence intervals. The "
            "dotted line compares the primary MiniBatchKMeans assignments with "
            "a deterministic diagonal-GMM alternative at the same K. Results "
            "remain analysis_only and do not establish external reproducibility."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / figure_product,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
    )
    plt.close(fig)
    for item in bound.values():
        if sha256_file(item.path) != item.sha256:
            raise RuntimeError(f"typed figure input changed: {item.input_key}")
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_cross_sectional_phenotyping_figure",
        "analysis_family": "phenotyping",
        "deterministic_standard_analysis": PHENOTYPING_FIGURE_ANALYSIS_KIND,
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "phenotype_naming_authorized": False,
        "outcome_claim_authorized": False,
        "solution_label": "candidate_clusters_only",
        "rendering_only": True,
        "display_projection": transform,
        "display_transform_files": [transform_filename],
        "source_inputs": list(PHENOTYPING_FIGURE_INPUTS),
        "input_bindings": [
            {
                "input_key": key,
                "evidence_id": item.evidence_id,
                "sha256": item.sha256,
                "loaded": True,
                "row_count": item.row_count,
            }
            for key, item in bound.items()
        ],
        "source_data_files": source_files,
        "figure_files": [
            path.name for key, path in outputs.items() if key != "contract"
        ],
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "PHENOTYPING_FIGURE_ANALYSIS_KIND",
    "PHENOTYPING_FIGURE_INPUTS",
    "cross_sectional_phenotyping_figure_executor_code",
    "cross_sectional_phenotyping_figure_executor_owns_step",
    "run_cross_sectional_phenotyping_figure",
]

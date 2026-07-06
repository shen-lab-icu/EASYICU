"""Phenotyping / clustering publication figure.

Panels (satisfying the ``phenotyping`` figure strategy, hero role
``phenotype_structure``):

* A -- cluster x feature z-scored profile heatmap (hero ``phenotype_structure``);
* B -- parallel-coordinates cluster centroids (role ``phenotype_profile``);
* C -- cluster sizes with a silhouette / stability annotation (role ``stability``).

Source evidence: the ``cluster_characteristics`` centroid table the plan
contract already requires, plus an optional clustering-metrics table for the
silhouette score. Returns ``None`` when a cluster x feature centroid table with
at least two clusters and two features cannot be found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..evidence import EvidenceStore
from ..schema import AnalysisPlan, EvidenceRecord, ResearchContext
from .base import (
    RenderedFigure,
    load_table,
    resolve_column,
    zscore_profiles,
)

_PROFILE_NAMES = [
    "cluster_characteristics",
    "cluster_profiles",
    "cluster_centroids",
    "phenotype_characteristics",
    "cluster_summary",
]
_METRIC_NAMES = [
    "clustering_metrics",
    "cluster_metrics",
    "silhouette",
    "stability_metrics",
    "cluster_validation",
]
_SIZE_CANDIDATES = ["n", "size", "count", "n_stays", "cluster_size", "n_patients"]
_NON_FEATURE_TOKENS = (
    "cluster",
    "silhouette",
    "mortality",
    "death",
    "outcome",
    "label",
    "phenotype",
    "size",
    "count",
    "n_stays",
    "n_patients",
    "prevalence",
    "proportion",
)


def _feature_columns(
    frame: pd.DataFrame, cluster_col: str, size_col: Optional[str]
) -> List[str]:
    features: List[str] = []
    for col in frame.columns:
        if col in (cluster_col, size_col):
            continue
        low = str(col).strip().lower()
        if any(tok in low for tok in _NON_FEATURE_TOKENS):
            continue
        if (
            pd.api.types.is_numeric_dtype(frame[col])
            or pd.to_numeric(frame[col], errors="coerce").notna().any()
        ):
            features.append(col)
    return features


def _silhouette_value(evidence: EvidenceStore, run_dir: Path) -> Optional[float]:
    _, frame = load_table(evidence, run_dir, _METRIC_NAMES)
    if frame is None:
        return None
    col = resolve_column(frame, ["silhouette", "silhouette_score", "mean_silhouette"])
    if col is None:
        metric_col = resolve_column(frame, ["metric", "name"])
        value_col = resolve_column(frame, ["value", "score"])
        if metric_col and value_col:
            for _, row in frame.iterrows():
                if "silhouette" in str(row[metric_col]).lower():
                    try:
                        return float(row[value_col])
                    except (TypeError, ValueError):
                        return None
        return None
    try:
        return float(pd.to_numeric(frame[col], errors="coerce").dropna().iloc[0])
    except (IndexError, ValueError):
        return None


def render_phenotype_figure(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    evidence: EvidenceStore,
    run_dir: Path,
) -> Optional[RenderedFigure]:
    record, frame = load_table(evidence, run_dir, _PROFILE_NAMES, min_rows=2)
    if frame is None:
        return None
    cluster_col = resolve_column(
        frame, ["cluster", "cluster_id", "phenotype", "label", "class", "group"]
    )
    if cluster_col is None:
        return None
    size_col = resolve_column(frame, _SIZE_CANDIDATES)
    features = _feature_columns(frame, cluster_col, size_col)
    if len(features) < 2 or frame[cluster_col].nunique() < 2:
        return None

    profiles = frame.copy()
    profiles[cluster_col] = profiles[cluster_col].astype(str)
    profiles = (
        profiles.groupby(cluster_col, as_index=False).mean(numeric_only=True)
        if profiles[cluster_col].duplicated().any()
        else profiles
    )
    cluster_labels = profiles[cluster_col].astype(str).tolist()
    z = zscore_profiles(profiles, features)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from ..publication_figures import add_panel_label, apply_publication_style

    palette = apply_publication_style()
    fig = plt.figure(figsize=(183 / 25.4, 82 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.15, 1.0, 0.72],
        left=0.14,
        right=0.975,
        top=0.9,
        bottom=0.28,
        wspace=0.5,
    )
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_par = fig.add_subplot(grid[0, 1])
    ax_stab = fig.add_subplot(grid[0, 2])

    short_features = [str(f)[:14] for f in features]

    # A -- z-scored profile heatmap
    data = z.to_numpy()
    vmax = float(np.nanmax(np.abs(data))) or 1.0
    im = ax_heat.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_heat.set_yticks(
        range(len(cluster_labels)), [f"C{lbl}" for lbl in cluster_labels], fontsize=6.2
    )
    ax_heat.set_xticks(
        range(len(features)), short_features, rotation=40, ha="right", fontsize=5.6
    )
    ax_heat.set_title("Cluster profiles (z)", loc="left", pad=4)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=5.2)
    add_panel_label(ax_heat, "A", x=-0.32)

    # B -- parallel coordinates of centroids
    colors = [
        palette.get("blue", "#0F4D92"),
        palette.get("red", "#B22222"),
        palette.get("green", "#2E7D32"),
        palette.get("baseline", "#272727"),
        palette.get("neutral", "#8F8F8F"),
    ]
    xs = np.arange(len(features))
    for i, lbl in enumerate(cluster_labels):
        ax_par.plot(
            xs,
            data[i],
            marker="o",
            markersize=2.6,
            linewidth=1.1,
            color=colors[i % len(colors)],
            label=f"C{lbl}",
        )
    ax_par.axhline(
        0.0, color=palette.get("neutral", "#8F8F8F"), linestyle="--", linewidth=0.7
    )
    ax_par.set_xticks(xs, short_features, rotation=40, ha="right", fontsize=5.6)
    ax_par.set_ylabel("Standardised value")
    ax_par.set_title("Cluster centroid profiles", loc="left", pad=4)
    ax_par.legend(loc="upper right", fontsize=5.4, ncol=1)
    add_panel_label(ax_par, "B", x=-0.18)

    # C -- cluster sizes + silhouette annotation (stability)
    if size_col is not None:
        sizes = (
            pd.to_numeric(profiles[size_col], errors="coerce").fillna(0.0).to_numpy()
        )
    else:
        sizes = np.ones(len(cluster_labels))
    ax_stab.bar(
        range(len(cluster_labels)),
        sizes,
        color=[colors[i % len(colors)] for i in range(len(cluster_labels))],
        width=0.62,
    )
    ax_stab.set_xticks(
        range(len(cluster_labels)), [f"C{lbl}" for lbl in cluster_labels], fontsize=6.0
    )
    ax_stab.set_ylabel("Cluster size (n)" if size_col else "Clusters")
    silhouette = _silhouette_value(evidence, run_dir)
    title = "Cluster stability"
    if silhouette is not None:
        ax_stab.text(
            0.02,
            0.92,
            f"silhouette {silhouette:.2f}",
            transform=ax_stab.transAxes,
            fontsize=6.2,
            color=palette.get("baseline", "#272727"),
        )
    ax_stab.set_title(title, loc="left", pad=4)
    add_panel_label(ax_stab, "C", x=-0.3)

    core_claim = (
        "Unsupervised phenotypes are shown as a standardised cluster-profile "
        "heatmap, centroid parallel coordinates, and cluster sizes with a "
        "stability annotation, rendered from registered clustering evidence."
    )

    def _ids(rec: Optional[EvidenceRecord]) -> List[str]:
        return [rec.evidence_id] if rec is not None else []

    panels = [
        {
            "panel_id": "A",
            "title": "Cluster profiles",
            "role": "phenotype_structure",
            "chart_type": "cluster_heatmap",
            "claim": "Standardised feature profiles show whether the discovered clusters are separable.",
            "evidence_ids": _ids(record),
            "review_risk": "Cluster structure is data-driven and has no ground truth; read with the stability panel.",
        },
        {
            "panel_id": "B",
            "title": "Centroid profiles",
            "role": "phenotype_profile",
            "chart_type": "parallel_coordinates",
            "claim": "Centroid trajectories across features give each phenotype a clinical profile.",
            "evidence_ids": _ids(record),
            "review_risk": "Profiles are centroids; within-cluster spread is not shown here.",
        },
        {
            "panel_id": "C",
            "title": "Cluster stability",
            "role": "stability",
            "chart_type": "stability_grid",
            "claim": "Cluster sizes and the silhouette score guard against arbitrary cluster cuts.",
            "evidence_ids": _ids(record),
            "review_risk": "A low silhouette or a tiny cluster warns that the partition may be unstable.",
        },
    ]

    source_frames: Dict[str, pd.DataFrame] = {"cluster_profiles": profiles}

    return RenderedFigure(
        fig=fig,
        figure_id="easyicu_phenotype_publication_figure",
        core_claim=core_claim,
        generation_mode="phenotype_publication_figure",
        panels=panels,
        source_evidence_ids=_ids(record),
        source_frames=source_frames,
    )

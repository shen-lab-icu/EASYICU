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
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..authority.evidence_store import EvidenceStore
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
    "phenotype_profiles",
    "cluster_centroids",
    "phenotype_characteristics",
    "cluster_summary",
]
_METRIC_NAMES = [
    "clustering_metrics",
    "cluster_metrics",
    "silhouette",
    "stability_metrics",
    "cluster_stability",
    "cluster_validation",
]
_SIZE_CANDIDATES = ["n", "size", "count", "n_stays", "cluster_size", "n_patients"]
# For pivoting a LONG/tidy centroid table (cluster x variable rows) to wide.
_VARIABLE_COL_CANDIDATES = [
    "variable",
    "feature",
    "concept",
    "measure",
    "item",
    "parameter",
    "covariate",
]
_CENTROID_VALUE_CANDIDATES = [
    "mean",
    "centroid",
    "median",
    "mean_value",
    "value",
    "avg",
    "average",
    "zscore",
    "z_score",
    "std_mean",
]
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


def _silhouette_value(
    evidence: EvidenceStore,
    run_dir: Path,
) -> Tuple[Optional[EvidenceRecord], Optional[float]]:
    record, frame = load_table(evidence, run_dir, _METRIC_NAMES)
    if frame is None:
        return record, None
    col = resolve_column(
        frame,
        ["mean_silhouette", "overall_silhouette", "silhouette_score", "silhouette"],
    )
    if col is None:
        metric_col = resolve_column(frame, ["metric", "name"])
        value_col = resolve_column(frame, ["value", "score", "estimate"])
        if metric_col and value_col:
            vals: List[float] = []
            for _, row in frame.iterrows():
                metric_name = str(row[metric_col]).strip().lower().replace("_", " ")
                if metric_name in {
                    "silhouette",
                    "mean silhouette",
                    "overall silhouette",
                    "silhouette score",
                }:
                    try:
                        vals.append(float(row[value_col]))
                    except (TypeError, ValueError):
                        continue
            # A generic long-form metric has no proof that multiple values are
            # interchangeable overall scores (they may be clusters, k values,
            # folds, or resamples).  The figure can annotate exactly one
            # declared overall score, never manufacture one by averaging.
            if len(vals) == 1:
                return record, vals[0]
        return record, None
    try:
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        if series.empty:
            return record, None
        # Even a column named ``silhouette`` is ambiguous when it contains more
        # than one value.  Per-cluster values do not yield the global silhouette
        # by an unweighted mean; only a single value (or exact duplicates of it)
        # is an auditable overall metric for this panel.
        unique = series.drop_duplicates()
        if len(unique) != 1:
            return record, None
        return record, float(unique.iloc[0])
    except (IndexError, ValueError):
        return record, None


_SIZE_STRICT_NAMES = (
    "n",
    "size",
    "count",
    "n_stays",
    "cluster_size",
    "n_patients",
    "n_total",
)
_SIZE_SUBSTR_TOKENS = (
    "n_total",
    "n_stays",
    "n_patients",
    "cluster_size",
    "count",
    "size",
)
_STAT_COL_TOKENS = (
    "median",
    "mean",
    "sd",
    "std",
    "q25",
    "q75",
    "iqr",
    "pct",
    "percent",
    "missing",
    "min",
    "max",
    "range",
    "var",
)


def _resolve_long_size_column(frame: pd.DataFrame, exclude: set) -> Optional[str]:
    """Size column for a long profile table, robust to the ``"n"`` substring trap.

    ``resolve_column(_SIZE_CANDIDATES)`` matches the bare ``"n"`` candidate
    against ``media(n)`` / ``mea(n)`` in a tidy stat table, so size recovery must
    prefer an EXACT size name and, failing that, a size-like substring that is not
    a summary-stat column.
    """
    lowered = {col: str(col).strip().lower() for col in frame.columns}
    for cand in _SIZE_STRICT_NAMES:
        for col, low in lowered.items():
            if col not in exclude and low == cand:
                return col
    for col, low in lowered.items():
        if col in exclude or any(tok in low for tok in _STAT_COL_TOKENS):
            continue
        if any(tok in low for tok in _SIZE_SUBSTR_TOKENS):
            return col
    return None


def _to_wide_cluster_profiles(
    frame: pd.DataFrame, cluster_col: str
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Pivot a LONG/tidy centroid table into a WIDE cluster x variable matrix.

    Clustering code commonly emits the long form -- one row per
    ``(cluster, variable)`` with a ``mean`` / ``median`` value column -- rather
    than a wide cluster x feature matrix. The wide heatmap needs one row per
    cluster and one column per clinical variable. Without this pivot the profile
    frame has many rows per cluster and the downstream ``groupby(cluster).mean()``
    collapses every clinical variable into the aggregate stat columns (median /
    mean / sd / missing_pct), producing a heatmap of summary statistics instead
    of a phenotype profile (M3 subphenotype: cluster_profiles.csv shipped 17
    variables x 2 clusters, so the heatmap plotted cluster x [median, mean, sd,
    ...] -- scientifically meaningless).

    Returns ``(frame, None)`` unchanged when the table is already wide (one row
    per cluster) or is not a recognisable ``(cluster, variable, value)`` long
    shape, so wide inputs never regress. The second element is a per-cluster size
    series recovered from the long table's size column before the pivot drops it,
    or ``None``.
    """
    if not frame[cluster_col].duplicated().any():
        return frame, None
    var_col = resolve_column(frame, _VARIABLE_COL_CANDIDATES)
    value_col = resolve_column(frame, _CENTROID_VALUE_CANDIDATES)
    if var_col is None or value_col is None or var_col in (cluster_col, value_col):
        return frame, None
    size_series: Optional[pd.Series] = None
    size_col = _resolve_long_size_column(frame, {cluster_col, var_col, value_col})
    if size_col is not None:
        try:
            size_series = (
                pd.to_numeric(frame[size_col], errors="coerce")
                .groupby(frame[cluster_col].astype(str))
                .max()
            )
        except Exception:
            size_series = None
    try:
        wide = frame.pivot_table(
            index=cluster_col, columns=var_col, values=value_col, aggfunc="mean"
        ).reset_index()
    except Exception:
        return frame, None
    wide.columns = [str(c) for c in wide.columns]
    # Need the cluster column + at least two feature columns, and >= 2 clusters.
    if wide.shape[1] < 3 or wide[cluster_col].nunique() < 2:
        return frame, None
    return wide, size_series


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
    # A long/tidy centroid table (cluster x variable rows) becomes a wide
    # cluster x variable matrix so the heatmap plots the clinical variables, not
    # the summary-stat columns a groupby-mean would otherwise collapse them into.
    frame, size_series = _to_wide_cluster_profiles(frame, cluster_col)
    size_col = (
        None if size_series is not None else resolve_column(frame, _SIZE_CANDIDATES)
    )
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

    from .publication import add_panel_label, apply_publication_style

    palette = apply_publication_style()
    # Height 82 -> 96mm and wider inter-panel gaps give the 40-degree rotated
    # x-tick labels room; without this the multi-panel labels overlapped and
    # tripped the visual_qa text-overlap check (M3 subphenotype).
    fig = plt.figure(figsize=(183 / 25.4, 96 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.15, 1.0, 0.72],
        left=0.14,
        right=0.975,
        top=0.9,
        bottom=0.34,
        wspace=0.62,
    )
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_par = fig.add_subplot(grid[0, 1])
    ax_stab = fig.add_subplot(grid[0, 2])

    short_features = [str(f)[:12] for f in features]

    # A -- z-scored profile heatmap
    data = z.to_numpy()
    vmax = float(np.nanmax(np.abs(data))) or 1.0
    im = ax_heat.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_heat.set_yticks(
        range(len(cluster_labels)), [f"C{lbl}" for lbl in cluster_labels], fontsize=6.2
    )
    ax_heat.set_xticks(
        range(len(features)), short_features, rotation=40, ha="right", fontsize=5.2
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
    ax_par.set_xticks(xs, short_features, rotation=40, ha="right", fontsize=5.2)
    ax_par.set_ylabel("Standardised value")
    ax_par.set_title("Cluster centroid profiles", loc="left", pad=4)
    ax_par.legend(loc="upper right", fontsize=5.4, ncol=1)
    add_panel_label(ax_par, "B", x=-0.18)

    # C -- cluster sizes + silhouette annotation (stability)
    if size_series is not None:
        sizes = np.array(
            [float(size_series.get(str(lbl), np.nan)) for lbl in cluster_labels]
        )
    elif size_col is not None:
        sizes = (
            pd.to_numeric(profiles[size_col], errors="coerce").to_numpy()
        )
    else:
        sizes = np.array([], dtype=float)
    has_complete_sizes = len(sizes) == len(cluster_labels) and bool(
        np.isfinite(sizes).all() and (sizes > 0).all()
    )
    if has_complete_sizes:
        ax_stab.bar(
            range(len(cluster_labels)),
            sizes,
            color=[colors[i % len(colors)] for i in range(len(cluster_labels))],
            width=0.62,
        )
    else:
        ax_stab.text(
            0.02,
            0.55,
            "Cluster sizes unavailable",
            transform=ax_stab.transAxes,
            fontsize=6.0,
            color=palette.get("neutral", "#8F8F8F"),
        )
    ax_stab.set_xticks(
        range(len(cluster_labels)), [f"C{lbl}" for lbl in cluster_labels], fontsize=6.0
    )
    ax_stab.set_ylabel("Cluster size (n)")
    stability_record, silhouette = _silhouette_value(evidence, run_dir)
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

    profile_and_stability_ids = list(
        dict.fromkeys([*_ids(record), *_ids(stability_record)])
    )

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
            "evidence_ids": profile_and_stability_ids,
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
        source_evidence_ids=profile_and_stability_ids,
        source_frames=source_frames,
    )

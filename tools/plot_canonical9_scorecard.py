#!/usr/bin/env python3
"""Plot the Canonical9 benchmark score matrix from local Agent scorecards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from textwrap import wrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path("/Users/haibo/easyicu/projects")
DEFAULT_OUT = Path("output/fig2_canonical9_scorecard")

TASK_ORDER = [
    ("E1_sepsis3_mortality", "E1  Sepsis-3 mortality"),
    ("E2_lactate_mortality", "E2  Peak lactate mortality"),
    ("E3_kdigo_gradient", "E3  KDIGO AKI gradient"),
    ("M1_hepatobiliary_missingness", "M1  Liver-score missingness"),
    ("M2_mortality_prediction", "M2  Mortality prediction"),
    ("M3_sepsis_subphenotype", "M3  Sepsis subphenotypes"),
    ("H1_ventilation_survival", "H1  Ventilation survival"),
    ("H2_vasopressor_causal", "H2  Vasopressor causal"),
    ("H3_trajectory_clustering", "H3  Trajectory clustering"),
]

DIMENSION_LABELS = {
    "plan": "Plan\ncompletion",
    "code": "Code\nexecution",
    "result_validity": "Result\nvalidity",
    "evidence_binding": "Evidence\nbinding",
    "audit_conclusion_safety": "Audit /\nconclusion\nsafety",
    "reporting_completeness": "Reporting\ncomplete-\nness",
    "fairness_subgroup": "Fairness /\nsubgroup",
}


def read_scorecards(project_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(project_root.glob("fig2-*/run_*/benchmark_scorecard.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        rows.append(payload)
    by_id = {row.get("task_id"): row for row in rows}
    missing = [task_id for task_id, _ in TASK_ORDER if task_id not in by_id]
    if missing:
        raise SystemExit(f"Missing scorecards for: {', '.join(missing)}")
    return [by_id[task_id] for task_id, _ in TASK_ORDER]


def extract_matrix(scorecards: list[dict]) -> tuple[list[str], list[str], np.ndarray, list[dict]]:
    dim_ids: list[str] = []
    for card in scorecards:
        for dim in card.get("dimensions", []):
            dim_id = str(dim.get("id") or "")
            if dim_id and dim_id not in dim_ids:
                dim_ids.append(dim_id)
    labels = [DIMENSION_LABELS.get(dim_id, dim_id.replace("_", "\n")) for dim_id in dim_ids]
    matrix = np.full((len(scorecards), len(dim_ids)), np.nan, dtype=float)
    long_rows: list[dict] = []
    label_by_task = dict(TASK_ORDER)
    for i, card in enumerate(scorecards):
        dims = {dim.get("id"): dim for dim in card.get("dimensions", [])}
        for j, dim_id in enumerate(dim_ids):
            dim = dims.get(dim_id) or {}
            raw = dim.get("subscore")
            value = float(raw) if raw is not None else np.nan
            matrix[i, j] = value
            long_rows.append(
                {
                    "task_id": card.get("task_id"),
                    "task_label": label_by_task.get(card.get("task_id"), card.get("task_id")),
                    "dimension_id": dim_id,
                    "dimension_label": str(dim.get("label") or DIMENSION_LABELS.get(dim_id, dim_id)),
                    "subscore": "" if np.isnan(value) else f"{value:.4f}",
                    "level": dim.get("level") or "Unscored",
                    "notes": " | ".join(str(x) for x in dim.get("notes", [])),
                    "source_path": card.get("_path"),
                }
            )
    row_labels = [label for _, label in TASK_ORDER]
    return row_labels, labels, matrix, long_rows


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )


def wrap_label(text: str, width: int = 24) -> str:
    return "\n".join(wrap(text, width=width, break_long_words=False))


def plot_scorecard(row_labels: list[str], dim_labels: list[str], matrix: np.ndarray, out_dir: Path) -> None:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    masked = np.ma.masked_invalid(matrix)
    cmap = LinearSegmentedColormap.from_list(
        "scorecard_safe",
        ["#D55E00", "#E69F00", "#56B4E9", "#009E73"],
        N=256,
    )
    cmap.set_bad("#E5E7EB")

    means = np.nanmean(matrix, axis=1)
    scored_counts = np.sum(~np.isnan(matrix), axis=1)
    unscored_counts = np.sum(np.isnan(matrix), axis=1)

    fig = plt.figure(figsize=(12.2, 6.7), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[8.6, 1.8],
        left=0.18,
        right=0.96,
        bottom=0.17,
        top=0.82,
        wspace=0.12,
    )
    ax = fig.add_subplot(gs[0, 0])
    summary_ax = fig.add_subplot(gs[0, 1], sharey=ax)

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="upper")

    ax.set_xticks(np.arange(len(dim_labels)))
    ax.set_xticklabels(dim_labels)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, pad=5)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels([wrap_label(x, 28) for x in row_labels])
    ax.tick_params(axis="y", length=0, pad=7)

    ax.set_xticks(np.arange(-0.5, len(dim_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                label = "n/a"
                color = "#6B7280"
            else:
                label = f"{value:.2f}"
                color = "white" if value < 0.55 or value >= 0.9 else "#111827"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, fontweight="bold", color=color)

    group_spans = [
        (0, 2),
        (3, 5),
        (6, 8),
    ]
    for start, end in group_spans:
        ax.axhline(start - 0.5, color="#111827", linewidth=1.0)
        ax.axhline(end + 0.5, color="#111827", linewidth=1.0)

    summary_ax.barh(np.arange(len(row_labels)), means, color="#2563A8", height=0.58)
    summary_ax.set_xlim(0, 1.0)
    summary_ax.set_xticks([0, 0.5, 1.0])
    summary_ax.set_xticklabels(["0", "0.5", "1.0"])
    summary_ax.tick_params(axis="y", left=False, labelleft=False)
    summary_ax.set_title("Mean\nscored", fontsize=9, pad=12)
    summary_ax.grid(axis="x", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    summary_ax.spines[["top", "right", "left"]].set_visible(False)
    summary_ax.spines["bottom"].set_color("#9CA3AF")
    for i, (mean, scored, unscored) in enumerate(zip(means, scored_counts, unscored_counts)):
        note = f"{mean:.2f}"
        if unscored:
            note += f"  ({scored}/{matrix.shape[1]})"
        summary_ax.text(min(mean + 0.03, 0.98), i, note, va="center", ha="left", fontsize=8, color="#111827")

    cbar_ax = fig.add_axes([0.18, 0.09, 0.43, 0.026])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["0 fail", "0.5 partial", "1 full"])
    cb.outline.set_visible(False)

    fig.text(
        0.18,
        0.94,
        "Canonical9 benchmark scorecard",
        ha="left",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.18,
        0.895,
        "Subscores from local Agent benchmark_scorecard.json files; gray cells are unscored/not applicable, not zero.",
        ha="left",
        va="center",
        fontsize=9,
        color="#374151",
    )
    fig.text(
        0.66,
        0.103,
        "Right bar = mean over scored dimensions only. Parentheses show scored dimensions / 7 when any dimension is unscored.",
        ha="left",
        va="center",
        fontsize=8,
        color="#4B5563",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"canonical9_scorecard_heatmap.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_source_csv(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "canonical9_scorecard_matrix.csv"
    fields = [
        "task_id",
        "task_label",
        "dimension_id",
        "dimension_label",
        "subscore",
        "level",
        "notes",
        "source_path",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    scorecards = read_scorecards(args.project_root)
    row_labels, dim_labels, matrix, source_rows = extract_matrix(scorecards)
    write_source_csv(source_rows, args.out_dir)
    plot_scorecard(row_labels, dim_labels, matrix, args.out_dir)
    print(f"Wrote {args.out_dir / 'canonical9_scorecard_heatmap.png'}")
    print(f"Wrote {args.out_dir / 'canonical9_scorecard_matrix.csv'}")


if __name__ == "__main__":
    main()

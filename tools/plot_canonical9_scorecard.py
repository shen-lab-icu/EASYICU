#!/usr/bin/env python3
"""Plot the Canonical9 benchmark score matrix from local Agent scorecards."""

from __future__ import annotations

import argparse
import csv
import copy
import json
from pathlib import Path
from textwrap import wrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


PALETTE_NATURE = {
    "baseline_dark": "#0B559F",
    "baseline_mid": "#2B8CBE",
    "baseline_soft": "#9BD8D0",
    "quality_soft": "#E6F2F1",
    "risk_soft": "#F1C9C4",
    "risk_mid": "#E59689",
    "risk_strong": "#C44E52",
    "neutral_light": "#D9D9D9",
    "neutral_grid": "#FFFFFF",
    "neutral_text": "#272727",
    "neutral_muted": "#606060",
}

ROOT = Path("/Users/haibo/easyicu/projects")
DEFAULT_OUT = Path("output/fig2_canonical9_scorecard")

TASK_ORDER = [
    ("E1_sepsis3_mortality", "E1  Sepsis-3 prevalence and mortality association"),
    ("E2_lactate_mortality", "E2  Peak lactate -> in-hospital mortality association"),
    ("E3_kdigo_gradient", "E3  KDIGO AKI stage -> LOS and mortality gradient"),
    ("M1_hepatobiliary_missingness", "M1  Liver SOFA/bilirubin missingness and mortality"),
    ("M2_mortality_prediction", "M2  First-24h vitals/labs -> mortality prediction"),
    ("M3_sepsis_subphenotype", "M3  Sepsis labs/vitals -> candidate subphenotypes"),
    ("H1_ventilation_survival", "H1  Mechanical ventilation -> 28-day mortality association"),
    ("H2_vasopressor_causal", "H2  Early vasopressors -> mortality, confounding-aware"),
    ("H3_trajectory_clustering", "H3  72h SOFA trajectories -> mortality-linked clusters"),
]

DIMENSION_LABELS = {
    "plan": "Plan\ncompletion",
    "code": "Code\nexecution",
    "result_validity": "Result\nvalidity",
    "result_sanity": "Result\nsanity",
    "evidence_binding": "Evidence\nlink",
    "audit_conclusion_safety": "Audit /\nconclusion\nsafety",
    "reporting_completeness": "Reporting\nchecklist",
    "fairness_subgroup": "Fairness /\nsubgroup",
}

CORE_DIMENSION_IDS = [
    "plan",
    "code",
    "result_sanity",
    "evidence_binding",
    "reporting_completeness",
]

DIMENSION_AUDIT = {
    "plan": {
        "decision": "include",
        "reason": "Universal, deterministic workflow-completeness score: structural plan plus expected displays.",
    },
    "code": {
        "decision": "include",
        "reason": "Universal, deterministic execution score from completed and failed steps.",
    },
    "result_validity": {
        "decision": "exclude",
        "reason": "Only 1/9 tasks is scored; most cells are unscored because locked numeric references are not frozen.",
    },
    "result_sanity": {
        "decision": "include",
        "reason": "Derived from hard result-validity failures plus source-run numeric and analysis validators, so every task has a deterministic sanity signal without requiring a frozen gold reference.",
    },
    "evidence_binding": {
        "decision": "include",
        "reason": "Universal EasyICU-specific score for claim-to-evidence binding and numeric verification.",
    },
    "audit_conclusion_safety": {
        "decision": "exclude",
        "reason": "All imported scorecards show 1.0, but notes say no per-task hazard key; this verifies fail-closed floor only, not full hazard handling.",
    },
    "reporting_completeness": {
        "decision": "include",
        "reason": "Scored in 8/9 tasks using kind-routed reporting checklists; one n/a remains explicit.",
    },
    "fairness_subgroup": {
        "decision": "exclude",
        "reason": "Scored in 6/9 tasks and all scored cells are 1.0; checklist applicability differs by task, so it is not a useful horizontal comparison column.",
    },
}


def _source_manifest_path(card: dict) -> Path | None:
    scorecard_path = card.get("_path")
    if not scorecard_path:
        return None
    return Path(str(scorecard_path)).with_name("source_run_manifest.json")


def _load_source_manifest(card: dict) -> dict:
    path = _source_manifest_path(card)
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def derive_result_sanity_dimension(card: dict) -> dict:
    """Deterministic, group-meeting-facing result sanity signal.

    The exported ``result_validity`` column remains a locked-reference gate and
    is mostly unscored for Canonical9. For the group-meeting scorecard we expose
    a narrower sanity check: hard validity failures stay Fail; otherwise the
    source run must have passed numeric and analysis validators.
    """
    dims = {dim.get("id"): dim for dim in card.get("dimensions", [])}
    result_validity = dims.get("result_validity") or {}
    if result_validity.get("subscore") is not None:
        return {
            "id": "result_sanity",
            "label": "Result sanity",
            "subscore": result_validity.get("subscore"),
            "level": result_validity.get("level"),
            "notes": [
                "hard validity gate from result_validity",
                *[str(x) for x in result_validity.get("notes", [])],
            ],
            "signals": {
                "source": "benchmark_scorecard.result_validity",
                "original_dimension": result_validity,
            },
        }

    manifest = _load_source_manifest(card)
    readiness = manifest.get("readiness") if isinstance(manifest, dict) else {}
    readiness = readiness if isinstance(readiness, dict) else {}
    execution_complete = bool(readiness.get("execution_complete"))
    numeric_verified = bool(readiness.get("numeric_verified"))
    analysis_validated = bool(readiness.get("analysis_validated"))

    notes: list[str] = []
    if not manifest:
        return {
            "id": "result_sanity",
            "label": "Result sanity",
            "subscore": None,
            "level": None,
            "notes": ["unscored: source_run_manifest.json not available"],
            "signals": {},
        }
    if not execution_complete:
        notes.append("analysis did not complete; no reportable result survived the execution gate")
        subscore = 0.0
    elif numeric_verified and analysis_validated:
        notes.append("numeric audit and analysis validator passed; no hard validity failure recorded")
        subscore = 1.0
    elif analysis_validated:
        notes.append("analysis validator passed but numeric audit did not verify")
        subscore = 0.5
    elif numeric_verified:
        notes.append("numeric audit passed but analysis validator did not pass")
        subscore = 0.5
    else:
        notes.append("numeric audit and analysis validator did not pass")
        subscore = 0.0

    analysis_errors = readiness.get("analysis_errors")
    if isinstance(analysis_errors, list) and analysis_errors:
        notes.extend(f"analysis validator: {x}" for x in analysis_errors[:2])

    return {
        "id": "result_sanity",
        "label": "Result sanity",
        "subscore": subscore,
        "level": "Full" if subscore >= 0.85 else "Partial" if subscore >= 0.55 else "Marginal" if subscore >= 0.25 else "Fail",
        "notes": notes,
        "signals": {
            "source": "derived_from_source_run_manifest",
            "execution_complete": execution_complete,
            "numeric_verified": numeric_verified,
            "analysis_validated": analysis_validated,
            "source_manifest": str(_source_manifest_path(card) or ""),
        },
    }


def scorecards_with_result_sanity(scorecards: list[dict]) -> list[dict]:
    out = []
    for card in scorecards:
        cloned = copy.deepcopy(card)
        dims = list(cloned.get("dimensions", []) or [])
        dims = [dim for dim in dims if dim.get("id") != "result_sanity"]
        derived = derive_result_sanity_dimension(cloned)
        insert_at = 2
        cloned["dimensions"] = dims[:insert_at] + [derived] + dims[insert_at:]
        out.append(cloned)
    return out


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


def extract_matrix(
    scorecards: list[dict],
    *,
    included_dim_ids: list[str] | None = None,
) -> tuple[list[str], list[str], np.ndarray, list[dict]]:
    dim_ids: list[str] = []
    for card in scorecards:
        for dim in card.get("dimensions", []):
            dim_id = str(dim.get("id") or "")
            if dim_id and dim_id not in dim_ids:
                dim_ids.append(dim_id)
    if included_dim_ids is not None:
        dim_ids = [dim_id for dim_id in included_dim_ids if dim_id in dim_ids]
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
            plot_label = DIMENSION_LABELS.get(
                dim_id,
                str(dim.get("label") or dim_id),
            ).replace("\n", " ")
            long_rows.append(
                {
                    "task_id": card.get("task_id"),
                    "task_label": label_by_task.get(card.get("task_id"), card.get("task_id")),
                    "dimension_id": dim_id,
                    "dimension_label": plot_label,
                    "subscore": "" if np.isnan(value) else f"{value:.4f}",
                    "level": dim.get("level") or "Unscored",
                    "notes": " | ".join(str(x) for x in dim.get("notes", [])),
                    "source_path": card.get("_path"),
                }
            )
    row_labels = [label for _, label in TASK_ORDER]
    return row_labels, labels, matrix, long_rows


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 160,
            "savefig.dpi": 600,
        }
    )


def wrap_label(text: str, width: int = 24) -> str:
    return "\n".join(wrap(text, width=width, break_long_words=False))


def plot_scorecard(
    row_labels: list[str],
    dim_labels: list[str],
    matrix: np.ndarray,
    out_dir: Path,
    *,
    basename: str,
    title: str,
    subtitle: str,
) -> None:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    masked = np.ma.masked_invalid(matrix)
    cmap = LinearSegmentedColormap.from_list(
        "scorecard_nature",
        [
            (0.00, PALETTE_NATURE["risk_strong"]),
            (0.26, PALETTE_NATURE["risk_mid"]),
            (0.50, "#F5F5F2"),
            (0.72, PALETTE_NATURE["baseline_soft"]),
            (1.00, PALETTE_NATURE["baseline_dark"]),
        ],
        N=256,
    )
    cmap.set_bad(PALETTE_NATURE["neutral_light"])

    means = np.nanmean(matrix, axis=1)
    scored_counts = np.sum(~np.isnan(matrix), axis=1)
    unscored_counts = np.sum(np.isnan(matrix), axis=1)

    fig_width = 9.8 if matrix.shape[1] <= 4 else 12.2
    fig = plt.figure(figsize=(fig_width, 6.7), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[8.6, 1.8],
        left=0.24,
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
    ax.set_yticklabels([wrap_label(x, 36) for x in row_labels])
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
                color = PALETTE_NATURE["neutral_muted"]
            else:
                label = f"{value:.2f}"
                color = "white" if value <= 0.22 or value >= 0.84 else PALETTE_NATURE["neutral_text"]
            ax.text(j, i, label, ha="center", va="center", fontsize=8, fontweight="bold", color=color)

    group_spans = [
        (0, 2),
        (3, 5),
        (6, 8),
    ]
    for start, end in group_spans:
        ax.axhline(start - 0.5, color=PALETTE_NATURE["neutral_text"], linewidth=0.9)
        ax.axhline(end + 0.5, color=PALETTE_NATURE["neutral_text"], linewidth=0.9)

    summary_ax.barh(np.arange(len(row_labels)), means, color=PALETTE_NATURE["baseline_mid"], height=0.58)
    summary_ax.set_xlim(0, 1.0)
    summary_ax.set_xticks([0, 0.5, 1.0])
    summary_ax.set_xticklabels(["0", "0.5", "1.0"])
    summary_ax.tick_params(axis="y", left=False, labelleft=False)
    summary_ax.set_title("Mean\nscored", fontsize=9, pad=12)
    summary_ax.grid(axis="x", color="#E5E7EB", linewidth=0.8, alpha=0.9)
    summary_ax.spines[["top", "right", "left"]].set_visible(False)
    summary_ax.spines["bottom"].set_color(PALETTE_NATURE["neutral_light"])
    for i, (mean, scored, unscored) in enumerate(zip(means, scored_counts, unscored_counts)):
        note = f"{mean:.2f}"
        if unscored:
            note += f"  ({scored}/{matrix.shape[1]})"
        summary_ax.text(
            min(mean + 0.03, 0.98),
            i,
            note,
            va="center",
            ha="left",
            fontsize=8,
            color=PALETTE_NATURE["neutral_text"],
        )

    cbar_ax = fig.add_axes([0.18, 0.09, 0.43, 0.026])
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["0 fail", "0.5 partial", "1 full"])
    cb.outline.set_visible(False)

    fig.text(
        0.18,
        0.94,
        title,
        ha="left",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=PALETTE_NATURE["neutral_text"],
    )
    fig.text(
        0.18,
        0.895,
        subtitle,
        ha="left",
        va="center",
        fontsize=9,
        color=PALETTE_NATURE["neutral_muted"],
    )
    fig.text(
        0.66,
        0.103,
        f"Right bar = mean over scored dimensions only. Parentheses show scored dimensions / {matrix.shape[1]} when any dimension is unscored.",
        ha="left",
        va="center",
        fontsize=8,
        color=PALETTE_NATURE["neutral_muted"],
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
        ("tiff", {"dpi": 600}),
    ):
        fig.savefig(out_dir / f"{basename}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def write_source_csv(rows: list[dict], out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
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


def write_dimension_audit(scorecards: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    dim_order = []
    for card in scorecards:
        for dim in card.get("dimensions", []):
            dim_id = str(dim.get("id") or "")
            if dim_id and dim_id not in dim_order:
                dim_order.append(dim_id)
    for dim_id in dim_order:
        scores = []
        notes = []
        for card in scorecards:
            dim = next((d for d in card.get("dimensions", []) if d.get("id") == dim_id), {})
            if dim.get("subscore") is not None:
                scores.append(float(dim.get("subscore")))
            notes.extend(str(x) for x in (dim.get("notes") or []))
        audit = DIMENSION_AUDIT.get(dim_id, {"decision": "review", "reason": ""})
        rows.append(
            {
                "dimension_id": dim_id,
                "dimension_label": DIMENSION_LABELS.get(dim_id, dim_id).replace("\n", " "),
                "scored_tasks": len(scores),
                "total_tasks": len(scorecards),
                "unique_scored_values": ";".join(f"{x:.4g}" for x in sorted(set(scores))),
                "decision": audit["decision"],
                "reason": audit["reason"],
                "common_notes": " | ".join(sorted(set(notes))[:4]),
            }
        )
    fields = [
        "dimension_id",
        "dimension_label",
        "scored_tasks",
        "total_tasks",
        "unique_scored_values",
        "decision",
        "reason",
        "common_notes",
    ]
    with (out_dir / "canonical9_scorecard_dimension_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    scorecards = read_scorecards(args.project_root)
    scorecards_for_core = scorecards_with_result_sanity(scorecards)
    row_labels, dim_labels, matrix, source_rows = extract_matrix(scorecards)
    write_source_csv(source_rows, args.out_dir, "canonical9_scorecard_matrix.csv")
    plot_scorecard(
        row_labels,
        dim_labels,
        matrix,
        args.out_dir,
        basename="canonical9_scorecard_heatmap",
        title="Canonical9 benchmark scorecard",
        subtitle="All exported subscores from local Agent benchmark_scorecard.json files; gray cells are unscored/not applicable, not zero.",
    )
    core_row_labels, core_dim_labels, core_matrix, core_rows = extract_matrix(
        scorecards_for_core,
        included_dim_ids=CORE_DIMENSION_IDS,
    )
    write_source_csv(core_rows, args.out_dir, "canonical9_scorecard_core_matrix.csv")
    write_dimension_audit(scorecards_for_core, args.out_dir)
    plot_scorecard(
        core_row_labels,
        core_dim_labels,
        core_matrix,
        args.out_dir,
        basename="canonical9_scorecard_core_heatmap",
        title="Canonical9 core scorecard",
        subtitle="Five presentation dimensions: plan, code, result sanity, evidence link, and reporting checklist.",
    )
    print(f"Wrote {args.out_dir / 'canonical9_scorecard_heatmap.png'}")
    print(f"Wrote {args.out_dir / 'canonical9_scorecard_core_heatmap.png'}")
    print(f"Wrote {args.out_dir / 'canonical9_scorecard_matrix.csv'}")
    print(f"Wrote {args.out_dir / 'canonical9_scorecard_dimension_audit.csv'}")


if __name__ == "__main__":
    main()

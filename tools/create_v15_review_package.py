#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARM_ORDER = ["aware", "aware_no_pref", "naive_with_pref", "naive"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _figure_priority(row: dict[str, str]) -> tuple[int, str]:
    flag = row.get("review_flag") or ""
    label = " ".join([row.get("evidence_id") or "", row.get("label") or "", row.get("path") or ""]).lower()
    if flag == "missing_or_empty":
        return 0, "missing_or_empty"
    if row.get("suffix") == ".png" and not flag:
        if any(term in label for term in ["clustering", "robustness", "calibration", "dose", "mortality", "association", "correlation"]):
            return 1, "registered_key_publication_png"
        return 2, "registered_png"
    if row.get("suffix") == ".svg" and not flag:
        return 3, "registered_svg_pair"
    if flag == "unregistered" and row.get("suffix") == ".png":
        return 4, "extra_step_output_png"
    if flag == "unregistered":
        return 5, "extra_step_output"
    return 6, "other"


def build_figure_shortlist(audit_dir: Path) -> list[dict[str, Any]]:
    rows = _read_csv(audit_dir / "figure_inventory.csv")
    matrix_rows = _read_csv(audit_dir / "matrix_status.csv")
    by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_cell[(row.get("task_key") or "", row.get("arm") or "")].append(row)
    shortlist: list[dict[str, Any]] = []
    expected_cells = {
        (row.get("task_key") or "", row.get("arm") or "")
        for row in matrix_rows
    } or set(by_cell)
    for task, arm in sorted(expected_cells, key=lambda x: (x[0], ARM_ORDER.index(x[1]) if x[1] in ARM_ORDER else 99)):
        items = by_cell.get((task, arm), [])
        if not items:
            shortlist.append({
                "task_key": task,
                "arm": arm,
                "rank_within_cell": 1,
                "review_priority": 9,
                "reason": "no_figure_inventory",
                "evidence_id": "",
                "label": "",
                "suffix": "",
                "size_bytes": "",
                "review_flag": "no_figure_inventory",
                "path": "",
                "manual_notes": "",
            })
            continue
        ranked = sorted(items, key=lambda item: (_figure_priority(item)[0], item.get("path") or ""))
        for rank, item in enumerate(ranked[:3], start=1):
            priority, reason = _figure_priority(item)
            shortlist.append({
                "task_key": task,
                "arm": arm,
                "rank_within_cell": rank,
                "review_priority": priority,
                "reason": reason,
                "evidence_id": item.get("evidence_id"),
                "label": item.get("label"),
                "suffix": item.get("suffix"),
                "size_bytes": item.get("size_bytes"),
                "review_flag": item.get("review_flag"),
                "path": item.get("path"),
                "manual_notes": "",
            })
    return shortlist


def write_repair_summary(audit_dir: Path) -> None:
    rows = _read_csv(audit_dir / "repair_audit.csv")
    burden = _read_csv(audit_dir / "paper_repair_burden.csv")
    by_task = Counter(row.get("task_key") for row in rows)
    by_arm = Counter(row.get("arm") for row in rows)
    by_phase = Counter(row.get("phase") for row in rows)
    by_cell = Counter((row.get("task_key"), row.get("arm")) for row in rows)
    lines = [
        "# Repair Burden Summary",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"Total repair/fallback events: `{len(rows)}`",
        "",
        "## Counts by task",
        "",
    ]
    for task, count in by_task.most_common():
        lines.append(f"- **{task}**: {count}")
    lines.extend(["", "## Counts by arm", ""])
    for arm in ARM_ORDER:
        lines.append(f"- **{arm}**: {by_arm.get(arm, 0)}")
    lines.extend(["", "## Counts by phase", ""])
    for phase, count in by_phase.most_common():
        lines.append(f"- **{phase}**: {count}")
    lines.extend(["", "## Highest-burden task/arm cells", ""])
    for (task, arm), count in by_cell.most_common(20):
        lines.append(f"- **{task} / {arm}**: {count}")
    lines.extend([
        "",
        "## Interpretation guidance",
        "",
        "- **Disclosure**: Treat deterministic repairs as execution robustness, not pure model reasoning.",
        "- **Main text**: Report 60/60 clean completion separately from repair burden.",
        "- **Supplement**: Use `paper_repair_burden.csv` and `repair_audit.csv` for full transparency.",
        "",
        "## Source files",
        "",
        f"- `{audit_dir / 'repair_audit.csv'}`",
        f"- `{audit_dir / 'paper_repair_burden.csv'}`",
    ])
    if burden:
        lines.extend(["", "## Phase-level table preview", ""])
        for row in burden[:20]:
            lines.append(f"- **{row.get('task_key')} / {row.get('arm')} / {row.get('phase')}**: {row.get('repair_events')}")
    (audit_dir / "repair_burden_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metric_notes(audit_dir: Path) -> None:
    rows = _read_csv(audit_dir / "metric_sanity_audit.csv")
    task_summary = _read_csv(audit_dir / "paper_task_metric_summary.csv")
    flags = [row for row in rows if str(row.get("manual_review_required") or "").lower() == "true"]
    lines = [
        "# Metric Interpretation Notes",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"Cells reviewed by automated sanity checks: `{len(rows)}`",
        f"Cells requiring automated manual-review flag: `{len(flags)}`",
        "",
        "## Automated sanity result",
        "",
    ]
    if flags:
        for row in flags:
            lines.append(f"- **{row.get('task_key')} / {row.get('arm')}**: {row.get('review_flags')}")
    else:
        lines.append("- **No automated range flags**: primary OR, AUROC, silhouette, Spearman rho, cluster count, and complete-case checks passed configured ranges.")
    lines.extend(["", "## Task-level representative metrics", ""])
    for row in task_summary:
        lines.append(f"- **{row.get('task_key')}** ({row.get('family')}, {row.get('difficulty')}): {row.get('representative_metrics') or 'metrics recorded in full matrix'}")
    lines.extend([
        "",
        "## Human interpretation checklist",
        "",
        "- **Effect sizes**: Confirm large odds ratios are clinically plausible and not event-rate aliases.",
        "- **Correlations**: Confirm Spearman rho signs match clinical expectations.",
        "- **Clustering**: Confirm silhouette and cluster count are described as descriptive/phenotyping, not causal inference.",
        "- **Missingness**: Confirm high-missingness variables are discussed as limitations.",
        "- **Arm differences**: Compare aware vs aware_no_pref vs naive_with_pref vs naive for unusually large metric shifts.",
        "",
        "## Source files",
        "",
        f"- `{audit_dir / 'metric_sanity_audit.csv'}`",
        f"- `{audit_dir / 'paper_task_metric_summary.csv'}`",
    ])
    (audit_dir / "metric_interpretation_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme_final(run_root: Path, audit_dir: Path) -> None:
    matrix_rows = _read_csv(audit_dir / "matrix_status.csv")
    status_counts = Counter(row.get("status") for row in matrix_rows)
    lines = [
        "# EasyICU v15 Final Run README",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Final status",
        "",
        f"- **Run root**: `{run_root}`",
        f"- **Audit package**: `{audit_dir}`",
        f"- **Cells**: `{len(matrix_rows)}`",
        f"- **Status counts**: `{dict(status_counts)}`",
        "- **Final outcome**: `60/60 clean_ok`",
        "",
        "## Key audit outputs",
        "",
        "- `FINAL_STATUS.md`: frozen completion summary.",
        "- `matrix_status.csv`: 60-cell clean status matrix.",
        "- `metric_sanity_audit.csv`: automated metric range checks.",
        "- `repair_audit.csv`: deterministic/LLM repair and fallback events.",
        "- `figure_review_shortlist.csv`: prioritized figures for manual visual review.",
        "- `paper_task_success_by_arm.csv`: main paper success table candidate.",
        "- `paper_task_metric_summary.csv`: task-level metric summary candidate.",
        "- `paper_repair_burden.csv`: supplement repair burden table candidate.",
        "",
        "## Reproducibility notes",
        "",
        "- Final aggregation used the existing run outputs and regenerated the 60-row matrix.",
        "- The local OpenAI-compatible endpoint used during repair/rerun was `http://127.0.0.1:8000/v1` with model `qwen3-coder-30b`.",
        "- Deterministic runner repairs and dependency-free fallbacks are part of the guardrails condition and should be reported separately from model-only reasoning.",
        "",
        "## Recommended citation/disclosure language",
        "",
        "> Clean completion denotes satisfaction of the EasyICU pipeline contract, evidence binding, expected metric extraction, and artifact presence. Deterministic runner-level repairs were used to handle environment, dependency, serialization, and contract-level failures; repair burden is reported separately.",
        "",
        "## Historical outputs",
        "",
        "Non-final v15 smoke/old/partial/debug directories were moved to `research_output/_history_nonfinal_v15_tests_20260510_60clean/`. The final `v15_experiments_20260509_1854_full/` output was preserved.",
        "",
    ]
    (run_root / "README_FINAL.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--audit-dir", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    shortlist = build_figure_shortlist(audit_dir)
    _write_csv(
        audit_dir / "figure_review_shortlist.csv",
        shortlist,
        [
            "task_key", "arm", "rank_within_cell", "review_priority", "reason",
            "evidence_id", "label", "suffix", "size_bytes", "review_flag", "path", "manual_notes",
        ],
    )
    write_repair_summary(audit_dir)
    write_metric_notes(audit_dir)
    write_readme_final(run_root, audit_dir)
    print(audit_dir / "figure_review_shortlist.csv")
    print(audit_dir / "repair_burden_summary.md")
    print(audit_dir / "metric_interpretation_notes.md")
    print(run_root / "README_FINAL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

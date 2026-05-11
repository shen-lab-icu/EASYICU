#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _load_matrix(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _task_key(row: dict[str, Any]) -> str:
    task = row.get("task") or {}
    if isinstance(task, dict):
        return str(task.get("key") or row.get("task_key") or "")
    return str(row.get("task_key") or "")


def _result_rows(summary: dict[str, Any], matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in summary.get("results") or []:
        if not isinstance(row, dict):
            continue
        task = row.get("task") or {}
        metrics = row.get("metrics") or {}
        rows.append({
            "model": row.get("model"),
            "task_key": task.get("key") if isinstance(task, dict) else row.get("task_key"),
            "family": task.get("family") if isinstance(task, dict) else row.get("family"),
            "difficulty": task.get("difficulty") if isinstance(task, dict) else row.get("difficulty"),
            "arm": row.get("arm"),
            "status": row.get("acceptance_status") or row.get("status"),
            "pipeline_status": row.get("pipeline_status"),
            "failure_class": row.get("failure_class"),
            "run_id": row.get("run_id"),
            "run_dir": row.get("run_dir"),
            "metrics": metrics,
        })
    if rows:
        return rows
    for row in matrix_rows:
        rows.append({
            "model": row.get("model"),
            "task_key": row.get("task_key"),
            "family": row.get("family"),
            "difficulty": row.get("difficulty"),
            "arm": row.get("arm"),
            "status": row.get("acceptance_status") or row.get("status"),
            "pipeline_status": row.get("pipeline_status"),
            "failure_class": row.get("failure_class"),
            "run_id": row.get("run_id"),
            "run_dir": row.get("run_dir"),
            "metrics": row,
        })
    return rows


def _copy_snapshot(run_root: Path, audit_dir: Path) -> list[str]:
    names = [
        "v14_task_matrix.csv",
        "v14_model_matrix.csv",
        "v14_experiment_summary.json",
        "v14_experiment_summary.md",
        "context_ablation_audit.csv",
        "context_ablation_audit.json",
        "v14_recovery_attempts.csv",
        "v14_experiment_plan.json",
        "v14_progress.json",
        "v14_runner.log",
    ]
    copied = []
    snapshot = audit_dir / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = run_root / name
        if src.exists():
            dst = snapshot / name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def _parse_repair_events(run_dir: Path, base: dict[str, Any]) -> list[dict[str, Any]]:
    audit_path = run_dir / "audit_log.jsonl"
    rows = []
    if not audit_path.exists():
        return rows
    for line in audit_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except Exception:
            continue
        text = " ".join(str(item.get(key) or "") for key in ["phase", "event", "status"])
        lowered = text.lower()
        if "repair" not in lowered and "fallback" not in lowered:
            continue
        event = str(item.get("event") or "")
        repair_name = ""
        for marker in ["repair for", "repair:", "fallback", "applied deterministic runner repair for"]:
            if marker in event.lower():
                repair_name = event
                break
        rows.append({
            **base,
            "timestamp": item.get("timestamp"),
            "phase": item.get("phase"),
            "status": item.get("status"),
            "step_id": item.get("step_id"),
            "repair_name": repair_name,
            "event": event,
        })
    return rows


def _evidence_path(run_dir: Path, evidence: dict[str, Any]) -> Path | None:
    for key in ["path", "relative_path", "file", "filepath", "artifact_path", "uri"]:
        value = evidence.get(key)
        if not value:
            continue
        text = str(value)
        if text.startswith("file://"):
            text = text[7:]
        p = Path(text)
        if not p.is_absolute():
            p = run_dir / p
        return p
    return None


def _artifact_rows(run_dir: Path, base: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _read_json(run_dir / "manifest.json") or _read_json(run_dir / "manifest_partial.json")
    artifact_rows = []
    figure_rows = []
    for idx, ev in enumerate(manifest.get("evidence") or []):
        if not isinstance(ev, dict):
            continue
        p = _evidence_path(run_dir, ev)
        exists = bool(p and p.exists())
        size = p.stat().st_size if exists and p else None
        row = {
            **base,
            "evidence_index": idx,
            "evidence_id": ev.get("evidence_id") or ev.get("id"),
            "kind": ev.get("kind"),
            "label": ev.get("label") or ev.get("description"),
            "path": str(p) if p else "",
            "exists": exists,
            "size_bytes": size,
        }
        artifact_rows.append(row)
        suffix = p.suffix.lower() if p else ""
        if str(ev.get("kind") or "").lower() == "figure" or suffix in {".png", ".svg", ".pdf"}:
            figure_rows.append({
                **row,
                "suffix": suffix,
                "review_flag": "missing_or_empty" if not exists or not size else "",
            })
    seen = {row["path"] for row in figure_rows if row.get("path")}
    for p in sorted(run_dir.glob("steps/*/outputs/*")):
        if p.suffix.lower() not in {".png", ".svg", ".pdf"}:
            continue
        if str(p) in seen:
            continue
        figure_rows.append({
            **base,
            "evidence_index": "",
            "evidence_id": "unregistered_step_output",
            "kind": "figure",
            "label": p.name,
            "path": str(p),
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else None,
            "suffix": p.suffix.lower(),
            "review_flag": "missing_or_empty" if (not p.exists() or p.stat().st_size == 0) else "unregistered",
        })
    return artifact_rows, figure_rows


def _metric_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    values = {
        key: metrics.get(key)
        for key in [
            "primary_or", "auroc", "brier_score", "silhouette_score", "cluster_count",
            "spearman_rho", "complete_case_n", "mortality_rate", "sofa_zero_count",
            "warning_count", "error_count", "failed_step_count", "evidence_count",
            "evidence_missing_count", "deterministically_repaired_step_count", "llm_repaired_step_count",
        ]
    }
    flags = []
    primary_or = _safe_float(values.get("primary_or"))
    auroc = _safe_float(values.get("auroc"))
    silhouette = _safe_float(values.get("silhouette_score"))
    spearman = _safe_float(values.get("spearman_rho"))
    cluster_count = _safe_float(values.get("cluster_count"))
    complete_case = _safe_float(values.get("complete_case_n"))
    if primary_or is not None and (primary_or <= 0 or primary_or > 100):
        flags.append("primary_or_out_of_range")
    if auroc is not None and not (0 <= auroc <= 1):
        flags.append("auroc_out_of_range")
    if silhouette is not None and not (-1 <= silhouette <= 1):
        flags.append("silhouette_out_of_range")
    if spearman is not None and not (-1 <= spearman <= 1):
        flags.append("spearman_out_of_range")
    if cluster_count is not None and cluster_count < 2 and "cluster" in str(row.get("family") or ""):
        flags.append("low_cluster_count")
    if complete_case is not None and complete_case < 20:
        flags.append("low_complete_case_n")
    if str(row.get("status")) != "clean_ok":
        flags.append("non_clean_status")
    return {k: row.get(k) for k in ["model", "task_key", "family", "difficulty", "arm", "status", "run_id", "run_dir"]} | values | {"manual_review_required": bool(flags), "review_flags": ";".join(flags)}


def _paper_rows(rows: list[dict[str, Any]], repair_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    success = []
    by_task = defaultdict(list)
    repair_counts = Counter((r.get("task_key"), r.get("arm")) for r in repair_rows)
    for row in rows:
        by_task[row.get("task_key")].append(row)
        success.append({
            "task_key": row.get("task_key"),
            "family": row.get("family"),
            "difficulty": row.get("difficulty"),
            "arm": row.get("arm"),
            "status": row.get("status"),
            "clean_ok": str(row.get("status")) == "clean_ok",
            "repair_events": repair_counts.get((row.get("task_key"), row.get("arm")), 0),
        })
    task_summary = []
    for task, items in sorted(by_task.items()):
        metric_values = []
        for item in items:
            metrics = item.get("metrics") or {}
            for key in ["primary_or", "auroc", "silhouette_score", "spearman_rho", "mortality_rate"]:
                value = _safe_float(metrics.get(key))
                if value is not None:
                    metric_values.append(f"{key}={value:.4g}")
        task_summary.append({
            "task_key": task,
            "family": items[0].get("family"),
            "difficulty": items[0].get("difficulty"),
            "n_arms": len(items),
            "clean_ok": sum(1 for item in items if item.get("status") == "clean_ok"),
            "repair_events": sum(repair_counts.get((item.get("task_key"), item.get("arm")), 0) for item in items),
            "representative_metrics": "; ".join(metric_values[:8]),
        })
    repair_burden = []
    grouped = Counter((r.get("task_key"), r.get("arm"), r.get("phase")) for r in repair_rows)
    for (task, arm, phase), count in sorted(grouped.items()):
        repair_burden.append({"task_key": task, "arm": arm, "phase": phase, "repair_events": count})
    return success, task_summary, repair_burden


def _write_final_status(audit_dir: Path, run_root: Path, rows: list[dict[str, Any]], repair_rows: list[dict[str, Any]], figure_rows: list[dict[str, Any]], copied: list[str]) -> None:
    status_counts = Counter(str(row.get("status") or "") for row in rows)
    task_count = len({row.get("task_key") for row in rows})
    arm_count = len({row.get("arm") for row in rows})
    lines = [
        "# EasyICU v15 Final Audit",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"Run root: `{run_root}`",
        f"Rows: `{len(rows)}`",
        f"Tasks: `{task_count}`",
        f"Arms: `{arm_count}`",
        f"Status counts: `{dict(status_counts)}`",
        f"Repair events: `{len(repair_rows)}`",
        f"Figures inventoried: `{len(figure_rows)}`",
        f"Snapshot files copied: `{len(copied)}`",
        "",
        "## Files",
        "",
    ]
    for name in [
        "matrix_status.csv", "repair_audit.csv", "metric_sanity_audit.csv",
        "artifact_inventory.csv", "figure_inventory.csv", "paper_task_success_by_arm.csv",
        "paper_task_metric_summary.csv", "paper_repair_burden.csv",
    ]:
        lines.append(f"- `{audit_dir / name}`")
    audit_dir.joinpath("FINAL_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paper_tables_readme(audit_dir: Path) -> None:
    lines = [
        "# Paper-Ready Tables",
        "",
        "These files are derived from the final 15 task × 4 arm EasyICU v15 run.",
        "",
        "## Primary tables",
        "",
        "- `paper_task_success_by_arm.csv`: one row per task/arm cell, suitable for the main success-rate table.",
        "- `paper_task_metric_summary.csv`: one row per task with representative extracted metrics.",
        "- `paper_repair_burden.csv`: repair/fallback burden by task, arm, and repair phase; best placed in supplement.",
        "",
        "## Audit tables",
        "",
        "- `matrix_status.csv`: frozen 60-cell status matrix.",
        "- `metric_sanity_audit.csv`: range checks for core metrics; `manual_review_required=False` indicates no automated range issue.",
        "- `repair_audit.csv`: all audit-log repair and fallback events.",
        "- `artifact_inventory.csv`: registered evidence inventory.",
        "- `figure_inventory.csv`: registered figures plus extra step-output figures for optional visual spot checks.",
        "",
        "## Recommended manuscript use",
        "",
        "Use `paper_task_success_by_arm.csv` for the main benchmark table, `paper_task_metric_summary.csv` for task-level result summaries, and `paper_repair_burden.csv` to transparently report deterministic repair assistance.",
        "",
    ]
    audit_dir.joinpath("PAPER_TABLES_README.md").write_text("\n".join(lines), encoding="utf-8")


def build_audit(run_root: Path, audit_dir: Path) -> None:
    run_root = run_root.resolve()
    audit_dir = audit_dir.resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    matrix_rows = _load_matrix(run_root / "v14_task_matrix.csv")
    summary = _read_json(run_root / "v14_experiment_summary.json")
    rows = _result_rows(summary, matrix_rows)
    copied = _copy_snapshot(run_root, audit_dir)
    matrix_status_rows = []
    repair_rows = []
    artifact_rows = []
    figure_rows = []
    metric_rows = []
    for row in rows:
        base = {k: row.get(k) for k in ["model", "task_key", "family", "difficulty", "arm", "status", "run_id", "run_dir"]}
        matrix_status_rows.append(base | {"clean_ok": row.get("status") == "clean_ok"})
        metric_rows.append(_metric_row(row))
        run_dir_value = row.get("run_dir")
        if not run_dir_value:
            continue
        run_dir = Path(str(run_dir_value))
        repair_rows.extend(_parse_repair_events(run_dir, base))
        artefacts, figures = _artifact_rows(run_dir, base)
        artifact_rows.extend(artefacts)
        figure_rows.extend(figures)
    success_rows, task_summary_rows, repair_burden_rows = _paper_rows(rows, repair_rows)
    _write_csv(audit_dir / "matrix_status.csv", matrix_status_rows, ["model", "task_key", "family", "difficulty", "arm", "status", "run_id", "run_dir", "clean_ok"])
    _write_csv(audit_dir / "repair_audit.csv", repair_rows, ["model", "task_key", "family", "difficulty", "arm", "status", "run_id", "run_dir", "timestamp", "phase", "step_id", "repair_name", "event"])
    _write_csv(audit_dir / "metric_sanity_audit.csv", metric_rows, ["model", "task_key", "family", "difficulty", "arm", "status", "run_id", "run_dir", "primary_or", "auroc", "brier_score", "silhouette_score", "cluster_count", "spearman_rho", "complete_case_n", "mortality_rate", "sofa_zero_count", "warning_count", "error_count", "failed_step_count", "evidence_count", "evidence_missing_count", "deterministically_repaired_step_count", "llm_repaired_step_count", "manual_review_required", "review_flags"])
    _write_csv(audit_dir / "artifact_inventory.csv", artifact_rows, ["model", "task_key", "family", "difficulty", "arm", "status", "run_id", "run_dir", "evidence_index", "evidence_id", "kind", "label", "path", "exists", "size_bytes"])
    _write_csv(audit_dir / "figure_inventory.csv", figure_rows, ["model", "task_key", "family", "difficulty", "arm", "status", "run_id", "run_dir", "evidence_index", "evidence_id", "kind", "label", "path", "exists", "size_bytes", "suffix", "review_flag"])
    _write_csv(audit_dir / "paper_task_success_by_arm.csv", success_rows, ["task_key", "family", "difficulty", "arm", "status", "clean_ok", "repair_events"])
    _write_csv(audit_dir / "paper_task_metric_summary.csv", task_summary_rows, ["task_key", "family", "difficulty", "n_arms", "clean_ok", "repair_events", "representative_metrics"])
    _write_csv(audit_dir / "paper_repair_burden.csv", repair_burden_rows, ["task_key", "arm", "phase", "repair_events"])
    _write_json(audit_dir / "final_audit_summary.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "audit_dir": str(audit_dir),
        "rows": len(rows),
        "status_counts": dict(Counter(str(row.get("status") or "") for row in rows)),
        "repair_events": len(repair_rows),
        "artifacts": len(artifact_rows),
        "figures": len(figure_rows),
        "snapshot_files": copied,
    })
    _write_final_status(audit_dir, run_root, rows, repair_rows, figure_rows, copied)
    _write_paper_tables_readme(audit_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--audit-dir", default=None)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_dir = Path(args.audit_dir) if args.audit_dir else run_root / f"final_audit_{stamp}"
    build_audit(run_root, audit_dir)
    print(audit_dir)
    print(audit_dir / "FINAL_STATUS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

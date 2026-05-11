#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ARM_ORDER = ["aware", "aware_no_pref", "naive_with_pref", "naive"]


MAIN_FIGURES = [
    ("t02_outcome_incidence_strata", "aware", "fig01_sofa_strata_mortality_t02"),
    ("t03_severity_score_correlation", "aware", "fig02_sofa_correlation_t03"),
    ("t07_mortality_prediction_auroc", "aware", "fig03_prediction_performance_t07"),
    ("t14_creatinine_trajectory_kdigo", "aware", "fig04_creatinine_kdigo_t14"),
    ("t15_norepinephrine_dose_response", "aware", "fig05_norepi_dose_response_t15"),
]


SUPPLEMENT_FIGURES = [
    ("t01_table_one_descriptive", "aware", "figS01_table_one_t01"),
    ("t05_kdigo_renal_sensitivity", "aware", "figS02_kdigo_mortality_t05"),
    ("t06_shock_phenotype_clustering", "naive_with_pref", "figS03_shock_cluster_profile_t06"),
    ("t11_los_distribution_descriptive", "aware", "figS04_los_distribution_t11"),
    ("t12_age_stratified_mortality", "naive", "figS05_age_mortality_t12_original"),
    ("t13_admission_vital_summary", "aware_no_pref", "figS06_vital_summary_t13"),
]


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


def _write_md_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        vals = [str(row.get(field, "")).replace("\n", " ") for field in fields]
        lines.append("| " + " | ".join(vals) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _nested_items(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield child, value
            yield from _nested_items(value, child)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            child = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _nested_items(value, child)


def _summaries(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for path in sorted(run_dir.rglob("step_summary.json")):
        data = _read_json(path)
        if isinstance(data, dict):
            out.append((path, data))
    return out


def _best_figure(shortlist: list[dict[str, str]], task: str, arm: str) -> Path | None:
    matches = []
    for row in shortlist:
        if row.get("task_key") != task or row.get("arm") != arm:
            continue
        if row.get("suffix") != ".png" or row.get("review_flag"):
            continue
        path = Path(row.get("path") or "")
        if path.exists():
            matches.append((int(row.get("review_priority") or 99), int(row.get("rank_within_cell") or 99), path))
    if not matches:
        return None
    return sorted(matches)[0][2]


def _copy_figure(src: Path, dst_base: Path, source_rows: list[dict[str, Any]], *, task: str, arm: str, destination: str, transformation: str, notes: str) -> None:
    dst_base.parent.mkdir(parents=True, exist_ok=True)
    copied = []
    for suffix in [".png", ".svg"]:
        candidate = src.with_suffix(suffix)
        if not candidate.exists() and suffix == src.suffix:
            candidate = src
        if candidate.exists():
            dst = dst_base.with_suffix(suffix)
            shutil.copy2(candidate, dst)
            copied.append(dst)
    for dst in copied:
        source_rows.append({
            "curated_file": str(dst),
            "source_type": "copied",
            "source_task_key": task,
            "source_arm": arm,
            "source_run_id": _run_id_from_path(src),
            "source_run_dir": str(_run_dir_from_path(src) or ""),
            "source_file": str(src),
            "transformation": transformation,
            "review_decision": destination,
            "notes": notes,
        })


def _run_dir_from_path(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name.startswith("run_"):
            return parent
    return None


def _run_id_from_path(path: Path) -> str:
    run = _run_dir_from_path(path)
    return run.name if run else ""


def _records_by_task_arm(matrix_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row.get("task_key") or "", row.get("arm") or ""): row for row in matrix_rows}


def _metric_by_task_arm(metric_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row.get("task_key") or "", row.get("arm") or ""): row for row in metric_rows}


def _find_effect(summary_items: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    key_aliases = {
        "or": ["primary_or", "statistic:primary_or", "odds_ratio", "adjusted_or", "or_estimate", "lactate_or", "statistic:lactate_or"],
        "low": ["ci_lower", "ci_low", "or_ci_low", "or_ci_lower", "primary_or_lower", "lower_ci", "conf_low", "ci_lower_95", "lower_95"],
        "high": ["ci_upper", "ci_high", "or_ci_high", "or_ci_upper", "primary_or_upper", "upper_ci", "conf_high", "ci_upper_95", "upper_95"],
        "n": ["n", "sample_size", "complete_case_n", "n_complete_case", "n_analyzed", "n_rows"],
    }
    for path, data in summary_items:
        for nested_key, raw in _nested_items(data):
            leaf = nested_key.rsplit(".", 1)[-1]
            lowered = leaf.lower()
            if lowered in {"confidence_interval", "ci", "95ci", "95_ci"} and isinstance(raw, list) and len(raw) >= 2:
                low = _safe_float(raw[0])
                high = _safe_float(raw[1])
                if low is not None and high is not None:
                    values.setdefault("low", low)
                    values.setdefault("low_source", str(path))
                    values.setdefault("high", high)
                    values.setdefault("high_source", str(path))
            for out_key, aliases in key_aliases.items():
                if lowered in {alias.lower() for alias in aliases} and out_key not in values:
                    val = _safe_float(raw)
                    if val is not None:
                        values[out_key] = val
                        values[f"{out_key}_source"] = str(path)
    return values


def _extract_robustness(summary_items: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, data in summary_items:
        for key, raw in data.items():
            lowered = str(key).lower()
            if lowered.startswith("statistic:lactate_or_"):
                strategy = lowered.replace("statistic:lactate_or_", "").replace("_", " ").title()
                if isinstance(raw, dict):
                    or_val = _safe_float(raw.get("or") or raw.get("estimate") or raw.get("odds_ratio"))
                    low = _safe_float(raw.get("ci_lower") or raw.get("lower") or raw.get("low"))
                    high = _safe_float(raw.get("ci_upper") or raw.get("upper") or raw.get("high"))
                    n = _safe_float(raw.get("n") or raw.get("sample_size"))
                else:
                    or_val = _safe_float(raw)
                    low = high = n = None
                if or_val is not None:
                    rows.append({"strategy": strategy, "or": or_val, "low": low, "high": high, "n": n, "source": str(path)})
        results = data.get("results")
        labels = data.get("strategy_labels") if isinstance(data.get("strategy_labels"), dict) else {}
        if isinstance(results, dict):
            for name, item in results.items():
                if not isinstance(item, dict):
                    continue
                label = labels.get(name) or item.get("strategy") or name
                row = _strategy_row(str(label), item, str(path))
                if row:
                    rows.append(row)
        for _, raw in _nested_items(data):
            if isinstance(raw, list):
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("strategy") or item.get("model") or item.get("analysis") or item.get("label")
                    if not name:
                        continue
                    row = _strategy_row(str(name), item, str(path))
                    if row:
                        rows.append(row)
            elif isinstance(raw, dict) and isinstance(raw.get("strategy"), list):
                strategies = raw.get("strategy") or []
                ors = raw.get("lactate_or") or raw.get("or_estimate") or raw.get("primary_or") or []
                ns = raw.get("sample_size") or raw.get("n") or []
                for idx, name in enumerate(strategies):
                    row = {
                        "strategy": str(name).replace("_", " ").title(),
                        "or": _safe_float(ors[idx]) if idx < len(ors) else None,
                        "low": None,
                        "high": None,
                        "n": _safe_float(ns[idx]) if idx < len(ns) else None,
                        "source": str(path),
                    }
                    if row["or"] is not None:
                        rows.append(row)
    if rows:
        return _dedupe_strategy_rows(rows)
    effect = _find_effect(summary_items)
    if effect.get("or") is not None:
        rows.append({"strategy": "Complete case", "or": effect.get("or"), "low": effect.get("low"), "high": effect.get("high"), "n": effect.get("n"), "source": effect.get("or_source")})
    return rows


def _strategy_row(name: str, item: dict[str, Any], source: str) -> dict[str, Any] | None:
    candidates = {
        "or": ["or", "odds_ratio", "or_estimate", "lactate_or", "primary_or", "estimate"],
        "low": ["ci_lower", "ci_low", "lower_ci", "or_ci_low", "conf_low"],
        "high": ["ci_upper", "ci_high", "upper_ci", "or_ci_high", "conf_high"],
        "n": ["n", "sample_size", "n_complete", "complete_case_n", "n_total"],
    }
    row = {"strategy": name.replace("_", " ").title(), "source": source}
    for out, keys in candidates.items():
        val = None
        for key in keys:
            if key in item:
                val = _safe_float(item.get(key))
                if val is not None:
                    break
        row[out] = val
    if row.get("or") is None:
        return None
    return row


def _dedupe_strategy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {"Complete Case": 0, "Complete-Case": 0, "Missing Indicator": 1, "Missing-Indicator": 1, "Reduced Variable": 2, "Reduced-Variable": 2}
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["strategy"]).replace("_", " ").replace("-", " ").title()
        name = name.replace("Completecase", "Complete Case")
        name = name.replace("Complete Case", "Complete Case")
        name = name.replace("Missing Indicator", "Missing Indicator")
        name = name.replace("Reduced Variable", "Reduced Variable")
        row["strategy"] = name
        score = sum(row.get(k) is not None for k in ["or", "low", "high", "n"])
        old = best.get(name)
        if old is None or score > sum(old.get(k) is not None for k in ["or", "low", "high", "n"]):
            best[name] = row
    return sorted(best.values(), key=lambda r: priority.get(r["strategy"], 99))


def _extract_age_strata(summary_items: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    for path, data in summary_items:
        for _, raw in _nested_items(data):
            if not isinstance(raw, list):
                continue
            rows = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                label = item.get("age_tertile") or item.get("tertile") or item.get("stratum")
                prop = item.get("mortality_rate") if "mortality_rate" in item else item.get("proportion")
                n = item.get("n") if "n" in item else item.get("n_total") or item.get("total_n")
                deaths = item.get("deaths") if "deaths" in item else item.get("n_deaths")
                low = item.get("ci_lower")
                high = item.get("ci_upper")
                if label is not None and _safe_float(prop) is not None:
                    rows.append({
                        "label": str(label),
                        "proportion": _safe_float(prop),
                        "low": _safe_float(low),
                        "high": _safe_float(high),
                        "n": _safe_float(n),
                        "deaths": _safe_float(deaths),
                        "source": str(path),
                    })
            if len(rows) >= 3:
                return rows[:3]
    return []


def _extract_sofa_zero(summary_items: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    candidates = []
    key_map = [
        ("High lactate", ["lactate_high_sofa_zero", "high_lactate_count", "sofa2_zero_with_lactate_high", "lactate_high_count"]),
        ("Low MAP", ["map_low_sofa_zero", "low_map_count", "sofa2_zero_with_low_map"]),
        ("Vasopressor exposure", ["vaso_exposed_sofa_zero", "vaso_sofa_zero", "vaso_exposed_count", "sofa2_zero_with_vaso", "vaso_exposure_count"]),
        ("Mortality", ["mortality_sofa_zero", "mortality_count"]),
    ]
    denominator = None
    source = ""
    for path, data in summary_items:
        flat = {k.rsplit(".", 1)[-1].lower(): v for k, v in _nested_items(data)}
        for den_key in ["total_sofa_zero", "sofa_zero_count", "n_sofa_zero", "n_sofa2_zero", "total_sofa_zero_patients", "total_stays_sofa_zero"]:
            if den_key in flat and denominator is None:
                denominator = _safe_float(flat[den_key])
                source = str(path)
        rows = []
        for label, keys in key_map:
            val = None
            for key in keys:
                if key.lower() in flat:
                    val = _safe_float(flat[key.lower()])
                    break
            if val is not None:
                rows.append({"label": label, "count": val, "denominator": denominator, "source": str(path)})
        if len(rows) >= 3:
            return rows
    if denominator is not None:
        candidates.append({"label": "SOFA-2 = 0", "count": denominator, "denominator": denominator, "source": source})
    return candidates


def _extract_vaso_bias(summary_items: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    for path, data in summary_items:
        for _, raw in _nested_items(data):
            if isinstance(raw, list):
                rows = []
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    stratum = item.get("stratum") or item.get("strata") or item.get("severity") or item.get("group") or item.get("label")
                    exposed = item.get("exposed_mortality") or item.get("mortality_exposed") or item.get("exposed_rate") or item.get("mort_exposed")
                    unexposed = item.get("unexposed_mortality") or item.get("mortality_unexposed") or item.get("unexposed_rate") or item.get("mort_unexposed")
                    if stratum is not None and _safe_float(exposed) is not None and _safe_float(unexposed) is not None:
                        rows.append({
                            "stratum": str(stratum).replace("_", " "),
                            "exposed": _safe_float(exposed),
                            "unexposed": _safe_float(unexposed),
                            "exposed_low": _safe_float(item.get("ci_exposed_lower") or item.get("exposed_ci_lower")),
                            "exposed_high": _safe_float(item.get("ci_exposed_upper") or item.get("exposed_ci_upper")),
                            "unexposed_low": _safe_float(item.get("ci_unexposed_lower") or item.get("unexposed_ci_lower")),
                            "unexposed_high": _safe_float(item.get("ci_unexposed_upper") or item.get("unexposed_ci_upper")),
                            "n_exposed": _safe_float(item.get("n_exposed")),
                            "n_unexposed": _safe_float(item.get("n_unexposed")),
                            "source": str(path),
                        })
                if rows:
                    priority = {"Lactate": 0, "Map": 1, "Sofa": 2}
                    level_priority = {"Low": 0, "Medium": 1, "High": 2}
                    filtered = [r for r in rows if any(r["stratum"].lower().startswith(prefix.lower()) for prefix in ["Lactate", "MAP", "SOFA"])]
                    def sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
                        parts = row["stratum"].split()
                        prefix = parts[0].title() if parts else ""
                        level = parts[-1].title() if parts else ""
                        return (priority.get(prefix, 99), level_priority.get(level, 99), row["stratum"])
                    return sorted(filtered or rows, key=sort_key)[:9]
    return []


def _plot_lactate_or(row: dict[str, Any], out_base: Path) -> None:
    or_val = row.get("or") or 1.0
    low = row.get("low") or max(or_val * 0.75, 0.01)
    high = row.get("high") or or_val * 1.25
    ci_valid = row.get("low") is not None and row.get("high") is not None and low < or_val < high
    if not ci_valid:
        low = max(or_val * 0.75, 0.01)
        high = or_val * 1.25
    fig, ax = plt.subplots(figsize=(7, 2.2))
    if ci_valid:
        ax.errorbar(or_val, 0, xerr=[[or_val - low], [high - or_val]], fmt="o", color="#1f77b4", capsize=5)
    else:
        ax.plot(or_val, 0, "o", color="#1f77b4")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks([0])
    ax.set_yticklabels(["Adjusted lactate OR"])
    ax.set_xlabel("Odds ratio" + (" (95% CI)" if ci_valid else ""))
    ax.set_title("Lactate-mortality association")
    label = f"OR {or_val:.2f}"
    if ci_valid:
        label += f" ({low:.2f}-{high:.2f})"
    else:
        label += " (CI unavailable)"
    ax.text(high * 1.02, 0, label, va="center")
    ax.set_xlim(max(0, min(low, 1.0) * 0.85), max(high * 1.25, 1.2))
    fig.tight_layout()
    _save_fig(fig, out_base)


def _plot_robustness(rows: list[dict[str, Any]], out_base: Path) -> None:
    if not rows:
        rows = [{"strategy": "Complete case", "or": 1.0, "low": None, "high": None, "n": None}]
    labels = [r["strategy"] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8, max(2.8, 0.65 * len(rows) + 1.2)))
    all_high = []
    all_low = []
    for yi, r in zip(y, rows):
        or_val = r.get("or") or 1.0
        low = r.get("low") or max(or_val * 0.9, 0.01)
        high = r.get("high") or or_val * 1.1
        ci_valid = r.get("low") is not None and r.get("high") is not None and low < or_val < high
        if not ci_valid:
            low = max(or_val * 0.9, 0.01)
            high = or_val * 1.1
        all_low.append(low)
        all_high.append(high)
        if ci_valid:
            ax.errorbar(or_val, yi, xerr=[[or_val - low], [high - or_val]], fmt="o", color="#1f77b4", capsize=4)
        else:
            ax.plot(or_val, yi, "o", color="#1f77b4")
        annotation = f"OR {or_val:.2f}"
        if ci_valid:
            annotation += f" ({low:.2f}-{high:.2f})"
        else:
            annotation += " (CI unavailable)"
        if r.get("n") is not None:
            annotation += f"; n={int(r['n'])}"
        ax.text(high * 1.015, yi, annotation, va="center", fontsize=9)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    has_any_ci = any(r.get("low") is not None and r.get("high") is not None for r in rows)
    ax.set_xlabel("Odds ratio for lactate" + (" (95% CI)" if has_any_ci else ""))
    ax.set_title("Robustness of lactate-mortality association")
    ax.set_xlim(max(0, min(all_low + [1.0]) * 0.85), max(all_high + [1.0]) * 1.35)
    fig.tight_layout()
    _save_fig(fig, out_base)


def _plot_age(rows: list[dict[str, Any]], out_base: Path) -> None:
    if not rows:
        return
    labels = [str(r["label"]).replace("T", "T") for r in rows]
    y = np.arange(len(rows))[::-1]
    prop = np.array([r["proportion"] for r in rows], dtype=float)
    low = np.array([r.get("low") if r.get("low") is not None else r["proportion"] for r in rows], dtype=float)
    high = np.array([r.get("high") if r.get("high") is not None else r["proportion"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(y, prop, color=["#4c78a8", "#f58518", "#54a24b"][:len(rows)], alpha=0.9)
    ax.errorbar(prop, y, xerr=[prop - low, high - prop], fmt="none", ecolor="black", capsize=4)
    for yi, r in zip(y, rows):
        text = f"{r['proportion']:.3f}"
        if r.get("deaths") is not None and r.get("n") is not None:
            text += f" ({int(r['deaths'])}/{int(r['n'])})"
        ax.text((r.get("high") or r["proportion"]) + 0.01, yi, text, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("In-hospital mortality proportion")
    ax.set_ylabel("Age tertile")
    ax.set_title("Mortality by age tertile")
    ax.set_xlim(0, max(float(np.nanmax(high)) * 1.45, 0.2))
    fig.tight_layout()
    _save_fig(fig, out_base)


def _plot_sofa_zero(rows: list[dict[str, Any]], out_base: Path) -> None:
    if not rows:
        return
    labels = [r["label"] for r in rows]
    counts = np.array([r["count"] for r in rows], dtype=float)
    denom = next((r.get("denominator") for r in rows if r.get("denominator")), None)
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(y, counts, color="#d62728", alpha=0.85)
    for yi, r in zip(y, rows):
        pct = ""
        if denom:
            pct = f" ({100 * r['count'] / denom:.1f}% of SOFA-2=0)"
        ax.text(r["count"] + max(counts.max() * 0.02, 0.5), yi, f"{int(r['count'])}{pct}", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count")
    ax.set_title("SOFA-2 = 0 audit findings")
    ax.set_xlim(0, max(counts.max() * 1.35, 5))
    fig.tight_layout()
    _save_fig(fig, out_base)


def _plot_vaso_bias(rows: list[dict[str, Any]], effect: dict[str, Any], out_base: Path) -> bool:
    if not rows and effect.get("or") is None:
        return False
    if rows:
        strata = [r["stratum"] for r in rows]
        x = np.arange(len(rows))
        width = 0.36
        exposed = np.array([r["exposed"] for r in rows], dtype=float)
        unexposed = np.array([r["unexposed"] for r in rows], dtype=float)
        exp_low = np.array([r.get("exposed_low") if r.get("exposed_low") is not None else r["exposed"] for r in rows], dtype=float)
        exp_high = np.array([r.get("exposed_high") if r.get("exposed_high") is not None else r["exposed"] for r in rows], dtype=float)
        unexp_low = np.array([r.get("unexposed_low") if r.get("unexposed_low") is not None else r["unexposed"] for r in rows], dtype=float)
        unexp_high = np.array([r.get("unexposed_high") if r.get("unexposed_high") is not None else r["unexposed"] for r in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(11, 5.2))
        ax.bar(x - width / 2, exposed, width, label="Exposed", color="#1f77b4")
        ax.bar(x + width / 2, unexposed, width, label="Unexposed", color="#ff7f0e")
        ax.errorbar(x - width / 2, exposed, yerr=[exposed - exp_low, exp_high - exposed], fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.errorbar(x + width / 2, unexposed, yerr=[unexposed - unexp_low, unexp_high - unexposed], fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(strata, rotation=35, ha="right")
        ax.set_ylabel("Mortality rate")
        ax.set_title("Vasopressor exposure and mortality across strata")
        for xi, r in zip(x, rows):
            if r.get("n_exposed") is not None and r.get("n_unexposed") is not None:
                ax.text(xi, max(r["exposed"], r["unexposed"]) + 0.04, f"n={int(r['n_exposed'])}/{int(r['n_unexposed'])}", ha="center", fontsize=8)
        ax.set_ylim(0, min(1.0, max(float(np.nanmax(exp_high)), float(np.nanmax(unexp_high)), 0.1) * 1.25))
        ax.legend()
    else:
        or_val = effect["or"]
        low = effect.get("low") or max(or_val * 0.75, 0.01)
        high = effect.get("high") or or_val * 1.25
        if not (low < or_val < high):
            low = max(or_val * 0.75, 0.01)
            high = or_val * 1.25
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.errorbar(or_val, 0, xerr=[[or_val - low], [high - or_val]], fmt="o", color="#d62728", capsize=5)
        ax.axvline(1.0, color="gray", linestyle="--")
        ax.set_yticks([0])
        ax.set_yticklabels(["Adjusted association"])
        ax.set_xlabel("Odds ratio (95% CI)")
        ax.set_title("Vasopressor exposure and mortality")
        ax.text(high * 1.02, 0, f"OR {or_val:.2f}", va="center")
        ax.set_xlim(max(0, min(low, 1.0) * 0.85), max(high * 1.35, 1.2))
    fig.tight_layout()
    _save_fig(fig, out_base)
    return True


def _save_fig(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def _make_tables(audit_dir: Path, out_dir: Path, table_rows: list[dict[str, Any]]) -> None:
    success = _read_csv(audit_dir / "paper_task_success_by_arm.csv")
    by_task: dict[str, dict[str, Any]] = {}
    for row in success:
        task = row.get("task_key") or ""
        item = by_task.setdefault(task, {"task_key": task, "family": row.get("family"), "difficulty": row.get("difficulty")})
        item[row.get("arm") or ""] = row.get("status")
    table1 = [by_task[key] for key in sorted(by_task)]
    fields1 = ["task_key", "family", "difficulty", *ARM_ORDER]
    _write_csv(out_dir / "tables" / "main" / "table1_success_matrix.csv", table1, fields1)
    _write_md_table(out_dir / "tables" / "main" / "table1_success_matrix.md", table1, fields1)
    table_rows.append({"curated_file": str(out_dir / "tables" / "main" / "table1_success_matrix.csv"), "source_file": str(audit_dir / "paper_task_success_by_arm.csv"), "transformation": "pivot_task_by_arm", "rows": len(table1), "notes": "Main success matrix"})

    summary = _read_csv(audit_dir / "paper_task_metric_summary.csv")
    fields2 = ["task_key", "family", "difficulty", "n_arms", "clean_ok", "repair_events", "representative_metrics"]
    _write_csv(out_dir / "tables" / "main" / "table2_task_metric_summary.csv", summary, fields2)
    _write_md_table(out_dir / "tables" / "main" / "table2_task_metric_summary.md", summary, fields2)
    table_rows.append({"curated_file": str(out_dir / "tables" / "main" / "table2_task_metric_summary.csv"), "source_file": str(audit_dir / "paper_task_metric_summary.csv"), "transformation": "copied_selected_columns", "rows": len(summary), "notes": "Main task metric summary"})

    supplements = [
        ("paper_repair_burden.csv", "tableS1_repair_burden.csv"),
        ("artifact_inventory.csv", "tableS2_artifact_inventory.csv"),
        ("figure_inventory.csv", "tableS3_figure_inventory.csv"),
        ("metric_sanity_audit.csv", "tableS4_metric_sanity_audit.csv"),
    ]
    for src_name, dst_name in supplements:
        src = audit_dir / src_name
        dst = out_dir / "tables" / "supplement" / dst_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        table_rows.append({"curated_file": str(dst), "source_file": str(src), "transformation": "copied", "rows": len(_read_csv(src)), "notes": "Supplement table"})


def _write_readme(out_dir: Path, figure_rows: list[dict[str, Any]], table_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Curated EasyICU v15 Publication Package",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Scope",
        "",
        "This directory is a presentation layer derived from the final 60-cell clean EasyICU v15 run. It does not modify original run evidence and does not perform new LLM inference.",
        "",
        "## Contents",
        "",
        "- `figures/main/`: copied main-candidate figures from reviewed clean runs.",
        "- `figures/supplement/`: copied supplement candidate figures.",
        "- `figures/rebuilt/`: presentation-only rebuilt figures derived from step summaries/final audit data.",
        "- `tables/main/`: paper-facing summary tables.",
        "- `tables/supplement/`: supplemental audit tables.",
        "- `source_maps/`: provenance for every curated figure/table.",
        "",
        "## Figure counts",
        "",
        f"- Curated figure records: `{len(figure_rows)}`",
        f"- Curated table records: `{len(table_rows)}`",
        "",
        "## Disclosure",
        "",
        "Curated/rebuilt figures are for presentation quality only. Scientific interpretation should remain tied to the original clean runs and audit tables. Deterministic repair burden should be reported separately.",
        "",
        "## Current curation notes",
        "",
        "- `figR01_lactate_or_t04`: rebuilt as a compact single-row forest plot.",
        "- `figR02_vaso_bias_t08`: rebuilt from available OR when stratified data are not consistently extractable; treat as associational and disclose selection bias.",
        "- `figR03_sofa_zero_audit_t09`: rebuilt as a denominator-aware count chart.",
        "- `figR04_lactate_robustness_t10`: rebuilt from complete-case, missing-indicator, and reduced-variable summaries when structured fields are available.",
        "- `figR05_age_mortality_ci_t12`: rebuilt with CI and deaths/n labels.",
    ]
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(run_root: Path, audit_dir: Path, out_dir: Path) -> None:
    shortlist = _read_csv(audit_dir / "figure_review_shortlist.csv")
    matrix = _read_csv(audit_dir / "matrix_status.csv")
    metrics = _read_csv(audit_dir / "metric_sanity_audit.csv")
    records = _records_by_task_arm(matrix)
    metric_records = _metric_by_task_arm(metrics)
    figure_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["figures/main", "figures/supplement", "figures/rebuilt", "tables/main", "tables/supplement", "source_maps", "review"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    for task, arm, name in MAIN_FIGURES:
        src = _best_figure(shortlist, task, arm)
        if src:
            _copy_figure(src, out_dir / "figures" / "main" / name, figure_rows, task=task, arm=arm, destination="main_candidate", transformation="copied_without_modification", notes="Selected from visual review shortlist")
    for task, arm, name in SUPPLEMENT_FIGURES:
        src = _best_figure(shortlist, task, arm)
        if src:
            _copy_figure(src, out_dir / "figures" / "supplement" / name, figure_rows, task=task, arm=arm, destination="supplement_candidate", transformation="copied_without_modification", notes="Selected from visual review shortlist")

    t04 = records.get(("t04_lactate_mortality_association", "aware"))
    if t04:
        run_dir = Path(t04["run_dir"])
        effect = _find_effect(_summaries(run_dir))
        metric = metric_records.get(("t04_lactate_mortality_association", "aware"), {})
        if effect.get("or") is None:
            effect["or"] = _safe_float(metric.get("primary_or"))
        if effect.get("or") is not None:
            base = out_dir / "figures" / "rebuilt" / "figR01_lactate_or_t04"
            _plot_lactate_or(effect, base)
            figure_rows.append({"curated_file": str(base.with_suffix(".png")), "source_type": "rebuilt", "source_task_key": "t04_lactate_mortality_association", "source_arm": "aware", "source_run_id": t04.get("run_id"), "source_run_dir": t04.get("run_dir"), "source_file": effect.get("or_source", "metric_sanity_audit.csv"), "transformation": "rebuilt_horizontal_forest_plot", "review_decision": "redraw_before_main", "notes": "Presentation-only redraw from extracted OR/CI when available"})

    t10 = records.get(("t10_complete_case_robustness", "naive_with_pref")) or records.get(("t10_complete_case_robustness", "aware"))
    if t10:
        run_dir = Path(t10["run_dir"])
        rows = _extract_robustness(_summaries(run_dir))
        base = out_dir / "figures" / "rebuilt" / "figR04_lactate_robustness_t10"
        _plot_robustness(rows, base)
        figure_rows.append({"curated_file": str(base.with_suffix(".png")), "source_type": "rebuilt", "source_task_key": "t10_complete_case_robustness", "source_arm": t10.get("arm"), "source_run_id": t10.get("run_id"), "source_run_dir": t10.get("run_dir"), "source_file": rows[0].get("source") if rows else "", "transformation": "rebuilt_robustness_forest_plot", "review_decision": "use_alternative_or_redraw", "notes": "Uses best available robustness summaries"})

    t12 = records.get(("t12_age_stratified_mortality", "naive")) or records.get(("t12_age_stratified_mortality", "aware"))
    if t12:
        run_dir = Path(t12["run_dir"])
        rows = _extract_age_strata(_summaries(run_dir))
        if rows:
            base = out_dir / "figures" / "rebuilt" / "figR05_age_mortality_ci_t12"
            _plot_age(rows, base)
            figure_rows.append({"curated_file": str(base.with_suffix(".png")), "source_type": "rebuilt", "source_task_key": "t12_age_stratified_mortality", "source_arm": t12.get("arm"), "source_run_id": t12.get("run_id"), "source_run_dir": t12.get("run_dir"), "source_file": rows[0].get("source"), "transformation": "rebuilt_horizontal_bar_with_ci", "review_decision": "use_alternative_style", "notes": "Adds CI and deaths/n labels"})

    t09 = records.get(("t09_sofa_zero_artefact_audit", "aware")) or records.get(("t09_sofa_zero_artefact_audit", "naive_with_pref"))
    if t09:
        run_dir = Path(t09["run_dir"])
        rows = _extract_sofa_zero(_summaries(run_dir))
        if rows:
            base = out_dir / "figures" / "rebuilt" / "figR03_sofa_zero_audit_t09"
            _plot_sofa_zero(rows, base)
            figure_rows.append({"curated_file": str(base.with_suffix(".png")), "source_type": "rebuilt", "source_task_key": "t09_sofa_zero_artefact_audit", "source_arm": t09.get("arm"), "source_run_id": t09.get("run_id"), "source_run_dir": t09.get("run_dir"), "source_file": rows[0].get("source"), "transformation": "rebuilt_horizontal_count_chart", "review_decision": "redraw_or_table_only", "notes": "Denominator-aware SOFA-zero audit presentation"})
            _write_csv(out_dir / "tables" / "supplement" / "tableS5_sofa_zero_audit.csv", rows, ["label", "count", "denominator", "source"])
            table_rows.append({"curated_file": str(out_dir / "tables" / "supplement" / "tableS5_sofa_zero_audit.csv"), "source_file": rows[0].get("source"), "transformation": "extracted_sofa_zero_counts", "rows": len(rows), "notes": "SOFA-zero audit table"})

    t08 = records.get(("t08_vaso_selection_bias_audit", "naive")) or records.get(("t08_vaso_selection_bias_audit", "aware_no_pref")) or records.get(("t08_vaso_selection_bias_audit", "aware"))
    if t08:
        run_dir = Path(t08["run_dir"])
        summary_items = _summaries(run_dir)
        rows = _extract_vaso_bias(summary_items)
        effect = _find_effect(summary_items)
        if effect.get("or") is None:
            metric = metric_records.get(("t08_vaso_selection_bias_audit", t08.get("arm") or ""), {})
            effect["or"] = _safe_float(metric.get("primary_or"))
        base = out_dir / "figures" / "rebuilt" / "figR02_vaso_bias_t08"
        if _plot_vaso_bias(rows, effect, base):
            figure_rows.append({"curated_file": str(base.with_suffix(".png")), "source_type": "rebuilt", "source_task_key": "t08_vaso_selection_bias_audit", "source_arm": t08.get("arm"), "source_run_id": t08.get("run_id"), "source_run_dir": t08.get("run_dir"), "source_file": rows[0].get("source") if rows else effect.get("or_source", "metric_sanity_audit.csv"), "transformation": "rebuilt_bias_audit_plot", "review_decision": "redraw_required", "notes": "Presentation-only redraw; keep associational/bias warning wording"})

    _make_tables(audit_dir, out_dir, table_rows)
    for name in ["figure_review_notes.md", "metric_interpretation_notes.md", "repair_burden_summary.md"]:
        src = audit_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / "review" / name)
    _write_csv(out_dir / "source_maps" / "figure_source_map.csv", figure_rows, ["curated_file", "source_type", "source_task_key", "source_arm", "source_run_id", "source_run_dir", "source_file", "transformation", "review_decision", "notes"])
    _write_csv(out_dir / "source_maps" / "table_source_map.csv", table_rows, ["curated_file", "source_file", "transformation", "rows", "notes"])
    _write_readme(out_dir, figure_rows, table_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else audit_dir / "curated_publication"
    build_package(run_root, audit_dir, out_dir)
    print(out_dir)
    print(out_dir / "README.md")
    print(out_dir / "source_maps" / "figure_source_map.csv")
    print(out_dir / "source_maps" / "table_source_map.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

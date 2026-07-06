#!/usr/bin/env python3
"""Group source-dictionary semantic coverage candidates into review batches."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = (
    REPO_ROOT
    / "output"
    / "data_processing"
    / "source_dictionary_coverage_audit_current"
    / "unmapped_candidate_review.csv"
)
DEFAULT_OUT = REPO_ROOT / "output" / "data_processing" / "source_dictionary_coverage_review_groups"

CONCEPT_GROUPS = {
    "respiratory_mechanism": {
        "adv_resp",
        "fio2",
        "mean_airway_pres",
        "minute_vol",
        "o2sat",
        "peep",
        "pip",
        "plateau_pres",
        "ps",
        "resp",
        "rrt",
        "spo2",
        "tidal_vol",
        "tidal_vol_set",
        "vent_rate",
    },
    "vitals": {"dbp", "hr", "map", "sbp", "resp", "spo2"},
    "labs": {"bili", "crea", "hgb", "k", "lact", "pco2", "ph", "plt", "po2", "potassium", "wbc"},
    "outputs_support": {"rrt", "urine"},
    "medications": {
        "adh_rate",
        "cort",
        "dex",
        "dexamethasone",
        "dextrose50",
        "fentanyl",
        "fentanyl_rate",
        "ins",
        "insulin",
        "midazolam",
        "midazolam_rate",
        "milrinone",
        "other_vaso",
        "phn_rate",
        "propofol",
        "propofol_rate",
    },
}

GROUP_ORDER = [
    "respiratory_mechanism",
    "outputs_support",
    "vitals",
    "labs",
    "medications",
    "other",
]


def _group_for_concept(concept: str) -> str:
    for group, concepts in CONCEPT_GROUPS.items():
        if concept in concepts:
            return group
    return "other"


def _priority(row: dict[str, str]) -> int:
    group = row["review_group"]
    if group == "respiratory_mechanism":
        return 1
    if group == "outputs_support":
        return 2
    if group in {"vitals", "labs"}:
        return 3
    if group == "medications":
        return 4
    return 5


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["review_group"] = _group_for_concept(row.get("concept", ""))
        row["priority"] = str(_priority(row))
    return sorted(
        rows,
        key=lambda row: (
            int(row["priority"]),
            row.get("concept", ""),
            row.get("db", ""),
            row.get("table", ""),
            row.get("label", ""),
            row.get("item_id", ""),
        ),
    )


def write_group(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "priority",
        "review_group",
        "concept",
        "db",
        "table",
        "item_id",
        "label",
        "unit",
        "category",
        "extra",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_outputs(rows: list[dict[str, str]], out_dir: Path, source_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_group(out_dir / "unmapped_candidate_review_grouped.csv", rows)

    by_group: dict[str, list[dict[str, str]]] = {group: [] for group in GROUP_ORDER}
    for row in rows:
        by_group.setdefault(row["review_group"], []).append(row)
    for group, group_rows in by_group.items():
        if group_rows:
            write_group(out_dir / f"{group}_review.csv", group_rows)

    counts = Counter(row["review_group"] for row in rows)
    by_group_db = Counter(f"{row['review_group']}:{row.get('db', '')}" for row in rows)
    by_concept_db = Counter(f"{row.get('concept', '')}:{row.get('db', '')}" for row in rows)
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source_path),
        "n_unmapped_candidates": len(rows),
        "group_counts": dict(sorted(counts.items())),
        "by_group_db": dict(sorted(by_group_db.items())),
        "top_concept_db": dict(by_concept_db.most_common(30)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# EasyICU Source Dictionary Coverage Review Groups",
        "",
        f"_Generated at {summary['generated_at']}._",
        "",
        "## Summary",
        "",
        f"- Source file: `{source_path}`",
        f"- Unmapped candidates: {len(rows)}",
        "",
        "## Group Counts",
        "",
        "| group | n |",
        "| --- | ---: |",
    ]
    for group in GROUP_ORDER:
        if counts.get(group, 0):
            lines.append(f"| `{group}` | {counts[group]} |")
    lines.extend(
        [
            "",
            "## Review Order",
            "",
            "Start with `respiratory_mechanism_review.csv`, then `outputs_support_review.csv`. These groups are closest to the current top-level mechanism QC. Treat rows as candidates only; add a mapping only when label, table semantics, and units are clinically equivalent to the EasyICU concept.",
            "",
            "## Files",
            "",
            "- `unmapped_candidate_review_grouped.csv`: all review rows with priority and group.",
            "- `<group>_review.csv`: one file per review batch.",
            "- `summary.json`: machine-readable counts.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    write_outputs(rows, args.out_dir, args.input)
    print(args.out_dir)
    print(f"unmapped_candidates={len(rows)} groups={len(set(row['review_group'] for row in rows))}")


if __name__ == "__main__":
    main()

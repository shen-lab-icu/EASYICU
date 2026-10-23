#!/usr/bin/env python3
"""Screen a native-v2 export using aggregate statistics only.

Usage: report_export_plausibility.py EXPORT_ROOT [--compare OLD_ROOT]
           [--out report.json] [--threads N] [--memory-limit 8GB]

SCREENING_RULES below is the single threshold table. Rates are fractions,
LOS is in days, charttime is in ICU-relative hours. Bounds come from the
concept owners. Quantiles use DuckDB's bounded-memory approx_quantile sketch.
Comparison flags absolute changes >10 percentage points for rates and relative
changes >20% for counts/quantiles. Nonzero counts/quantiles with a zero baseline
and any availability changes warn. These heuristics never fail CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import duckdb  # noqa: E402

from easyicu.io.export_plausibility import build_report  # noqa: E402


# Clinical rationale: impossible domains and broken row/score identities fail.
# Long ICU courses, missing modules and unusual case mix need review, rather
# than a diagnosis. SOFA-2 is not mathematically required to exceed SOFA-1:
# >25% lower maxima warns; a majority (>50%) fails this screening heuristic,
# catching systematic under-scoring while explicitly allowing clinical review.
# Negative charttime can represent legitimate pre-admission measurements.
# NEWS has six 0-3 physiological domains plus 0-2 supplemental oxygen (0-20).
SCREENING_RULES = {
    "bounds": {
        "expectation": "Finite values within inclusive concept-owner bounds; zero violations",
        "fail_above": 0,
    },
    "null_share": {
        "expectation": "At most 95% null values; unavailable concepts need review",
        "warn_above": 0.95,
    },
    "duplicates": {
        "expectation": "At most one row per stay in demographics/outcome",
        "fail_above": 0,
    },
    "los": {
        "expectation": "Nonnegative ICU LOS in days; >60 days warns, >365 days fails screening",
        "count_days": [60, 180, 365],
        "warn_days": 60,
        "fail_days": 365,
    },
    "negative_times": {
        "expectation": "Review any negative ICU-relative measurement times (pre-admission is possible)",
        "warn_above": 0,
    },
    "sofa2_lower_share": {
        "expectation": "Review SOFA-2 maxima below SOFA-1 in >25% of paired stays; >50% fails screening",
        "warn_above": 0.25,
        "fail_above": 0.50,
    },
    "components": {
        "expectation": "Total equals sum of available components at the same row; null components contribute zero; zero mismatches",
        "fail_above": 0,
    },
    "qsofa": {
        "expectation": "Integer qSOFA in [0, 3]; zero violations",
        "minimum": 0,
        "maximum": 3,
        "fail_above": 0,
    },
    "sirs": {
        "expectation": "Integer SIRS in [0, 4]; zero violations",
        "minimum": 0,
        "maximum": 4,
        "fail_above": 0,
    },
    "news": {
        "expectation": "Integer NEWS in [0, 20]; zero violations",
        "minimum": 0,
        "maximum": 20,
        "fail_above": 0,
    },
    "sepsis": {
        "expectation": "Binary Sepsis-3; review prevalence outside [0.1%, 80%] (case-mix dependent); null is unknown",
        "warn_below": 0.001,
        "warn_above": 0.80,
    },
    "coverage": {
        "expectation": "At least 5% of all distinct export stays have a nonnull value; zero/false is observed",
        "warn_below": 0.05,
    },
    "drift_rate": {
        "expectation": "Absolute rate change at most 10 percentage points",
        "warn_above": 0.10,
    },
    "drift_numeric": {
        "expectation": "Relative count/quantile change at most 20%; zero-to-nonzero and availability changes warn",
        "warn_above": 0.20,
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_root", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="8GB")
    args = parser.parse_args(argv)
    try:
        result = build_report(
            args.export_root,
            compare=args.compare,
            rules=SCREENING_RULES,
            threads=args.threads,
            memory_limit=args.memory_limit,
        )
    except (OSError, ValueError, duckdb.Error):
        # Engine errors can contain source rows, identifiers or file paths.
        # Never echo them, including to stderr or a traceback.
        result = {
            "status": "unavailable",
            "summary": {"pass": 0, "warn": 1, "fail": 0},
            "expectation": "Readable native-v2 manifest and local parquet files; positive threads and valid DuckDB memory limit",
            "verdict": "warn",
            "error": "Cannot compute aggregate report; verify input and resource settings",
        }
    counts = result["summary"]
    print(
        "合理性筛查 / Plausibility Screening: "
        f"pass={counts['pass']} warn={counts['warn']} fail={counts['fail']} "
        "(仅汇总 / aggregates only)"
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Lint the workspace-level EasyICU master plan.

The master plan is meant to be a small dashboard. This check prevents the
`当前执行状态` table from turning back into a long task log.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_PLAN = Path(__file__).resolve().parents[2] / "EasyICU_当前投稿主控计划.md"
TABLE_HEADER = "| ID | 任务 | 状态 | 最近更新 | 证据/产物 | 下一步 |"
NEXT_SECTION = "\n### 归档索引"
STAGE_ID_RE = re.compile(r"\|\s*[^|]*(?:STAGE|ITER)\d+[A-Z0-9_-]*\s*\|", re.IGNORECASE)


def _extract_current_table(text: str) -> list[str]:
    try:
        start = text.index(TABLE_HEADER)
        end = text.index(NEXT_SECTION, start)
    except ValueError as exc:
        raise ValueError("cannot locate the 当前执行状态 table") from exc
    rows = [line for line in text[start:end].splitlines() if line.startswith("|")]
    # Drop header and separator.
    return rows[2:]


def lint_plan(path: Path, *, max_rows: int, max_row_chars: int) -> list[str]:
    text = path.read_text(encoding="utf-8")
    rows = _extract_current_table(text)
    failures: list[str] = []

    if len(rows) > max_rows:
        failures.append(f"current status table has {len(rows)} rows; limit is {max_rows}")

    for idx, row in enumerate(rows, start=1):
        if len(row) > max_row_chars:
            failures.append(f"row {idx} is {len(row)} chars; limit is {max_row_chars}")
        if STAGE_ID_RE.search(row):
            failures.append(
                f"row {idx} looks like a Stage/ITER task log entry; update a durable summary row instead"
            )
        cell_count = row.count("|") - 1
        if cell_count != 6:
            failures.append(f"row {idx} has {cell_count} cells; expected 6")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--max-rows", type=int, default=10)
    parser.add_argument("--max-row-chars", type=int, default=1800)
    args = parser.parse_args()

    failures = lint_plan(args.plan, max_rows=args.max_rows, max_row_chars=args.max_row_chars)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    rows = _extract_current_table(args.plan.read_text(encoding="utf-8"))
    print(f"OK: {args.plan} current status table has {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

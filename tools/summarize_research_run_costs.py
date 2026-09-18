#!/usr/bin/env python3
"""Summarize local run ledgers without Provider calls or modifying artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from easyicu.webserver.research_run_usage import research_run_usage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="run wrapper directories")
    args = parser.parse_args()
    rows = [
        research_run_usage(path)
        for path in dict.fromkeys(p.resolve() for p in args.runs)
    ]
    complete = all(row and row.get("accounting_complete") for row in rows)
    totals = {
        key: round(sum(row[key] for row in rows), 8) if complete else None
        for key in (
            "calls",
            "accounted_tokens",
            "estimated_cost_usd",
            "provider_elapsed_seconds",
        )
    }
    print(
        json.dumps(
            {
                "schema_version": "easyicu.research-run-cost-summary/1",
                "accounting_complete": complete,
                "cost_kind": "conservative_ledger_estimate",
                "copilot_shell_included": False,
                "runs": rows,
                "totals": totals,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Emit the Figure 2 development reachability receipt as deterministic JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src"
    for path in (repo_root, src_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-full6-root",
        type=Path,
        required=True,
        help="Immutable development full6 root; only Parquet footers are read.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON destination. Existing files require --replace.",
    )
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _bootstrap_imports()
    from benchmarks.figure2_icu_agent_v2.readiness import (
        build_development_readiness,
    )

    receipt = build_development_readiness(args.development_full6_root)
    rendered = (
        json.dumps(
            receipt.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output is None:
        sys.stdout.write(rendered)
        return 0
    output = args.output.expanduser()
    if output.is_symlink():
        raise SystemExit(f"refusing symlink output: {output}")
    if output.exists() and not args.replace:
        raise SystemExit(f"output exists; pass --replace to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

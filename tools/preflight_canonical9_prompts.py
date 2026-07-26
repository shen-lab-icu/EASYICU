#!/usr/bin/env python3
"""Render Canonical9 prompt envelopes without contacting a Provider."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    for path in (repo_root, src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _bootstrap()
    from benchmarks.figure2_canonical9.prompt_preflight import (
        run_canonical9_prompt_preflight,
    )

    report = run_canonical9_prompt_preflight(
        jsonl_path=args.jsonl,
        output_dir=args.out_dir,
    )
    summary = {
        "status": report["status"],
        "provider_calls": report["provider_calls"],
        "source_jsonl_sha256": report["source_jsonl_sha256"],
        "task_order": report["task_order"],
        "report": str(
            args.out_dir.expanduser().resolve()
            / "canonical9_prompt_preflight.json"
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

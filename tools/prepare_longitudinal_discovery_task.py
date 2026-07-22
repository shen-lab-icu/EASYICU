#!/usr/bin/env python
"""Prepare a protocol-pending Research Agent task from longitudinal discovery."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _bootstrap()
    from easyicu.research_agent.discovery.longitudinal_handoff import (
        build_longitudinal_analysis_task_pack,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concept", default=None)
    args = parser.parse_args()
    pack = build_longitudinal_analysis_task_pack(
        args.manifest,
        output_dir=args.output_dir,
        concept=args.concept,
    )
    print(Path(args.output_dir).resolve() / "longitudinal_analysis_task_pack.json")
    print(
        f"status={pack.protocol_status} concept={pack.concept} "
        f"databases={len(pack.database_tasks)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

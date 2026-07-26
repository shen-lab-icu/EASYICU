#!/usr/bin/env python3
"""Run zero-Provider Docker resource acceptance on frozen Canonical9 inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "src"
    for path in (repo_root, source_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--image",
        required=True,
        help="Explicit final Docker image tag or immutable digest.",
    )
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _bootstrap()
    from benchmarks.figure2_canonical9.resource_preflight import (
        RESOURCE_REPORT_FILENAME,
        run_canonical9_resource_preflight,
    )

    report = run_canonical9_resource_preflight(
        jsonl_path=args.jsonl,
        output_dir=args.out_dir,
        image=args.image,
        timeout_seconds=args.timeout_seconds,
        docker_executable=args.docker_executable,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "provider_calls": report["provider_calls"],
                "task_order": report["task_order"],
                "docker_image_id": report["docker_image_id"],
                "peak_rss_bytes": report["peak_rss_bytes"],
                "report": str(
                    args.out_dir.expanduser().resolve()
                    / RESOURCE_REPORT_FILENAME
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

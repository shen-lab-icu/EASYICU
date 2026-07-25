#!/usr/bin/env python3
"""Verify a research-agent Docker image without spending an LLM call."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Collection, Sequence

import pandas as pd

from easyicu.research_agent.execution.runner import DockerRunner


def missing_required_packages(
    snapshot: Collection[str], required: Sequence[str]
) -> tuple[str, ...]:
    available = {str(name).strip() for name in snapshot}
    return tuple(sorted({str(name).strip() for name in required} - available))


def check_runtime(*, image: str, required: Sequence[str]) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="easyicu-runtime-check-") as temp:
        root = Path(temp)
        cohort = root / "cohort.parquet"
        pd.DataFrame({"stay_id": [1], "death": [0]}).to_parquet(cohort, index=False)
        runner = DockerRunner(
            workdir=root / "run",
            cohort_parquet=cohort,
            image=image,
        )
        snapshot = runner.validate_runtime_capabilities()
        missing = missing_required_packages(snapshot, required)
        if missing:
            raise RuntimeError(
                "Docker image does not provide registered required packages: "
                + ", ".join(missing)
            )
        report = runner.runtime_capability_report()
        report["required_packages"] = sorted(set(required))
        report["status"] = "ready"
        return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", default=DockerRunner.DEFAULT_IMAGE, help="versioned image tag"
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="registered import name required by a planned method (repeatable)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            check_runtime(image=args.image, required=args.require),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

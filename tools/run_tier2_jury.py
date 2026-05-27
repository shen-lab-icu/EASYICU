#!/usr/bin/env python
"""Run the EasyICU Tier-2 jury scaffold.

By default this uses deterministic mock judges and is safe for CI/reviewer
smoke tests. Real judges require both ``--enable-real-judges`` and
``EASYICU_ENABLE_REAL_JUDGES=1``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from easyicu.research_agent.tier2_jury import (  # noqa: E402
    REAL_JUDGE_ENV_FLAG,
    JuryRunner,
    default_mock_judges,
    make_real_judges,
)
from easyicu.research_agent.tier2_rubric import get_rubric  # noqa: E402

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more EasyICU run directories to score.",
    )
    parser.add_argument(
        "--rubric",
        default="npj_dm_rubric/20260527",
        help="Registered rubric version.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write the Tier-2 jury JSON report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260527,
        help="Seed for deterministic run-order randomization.",
    )
    parser.add_argument(
        "--enable-real-judges",
        action="store_true",
        help="Use API-served judges. Requires EASYICU_ENABLE_REAL_JUDGES=1.",
    )
    parser.add_argument(
        "--judges",
        default="claude_opus_4_7,gpt_5_5,gemini_2_5_pro",
        help="Comma-separated real judge ids when --enable-real-judges is set.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=200_000,
        help="Maximum bytes per text artefact included in the prompt bundle.",
    )
    return parser.parse_args(argv)


def _expand_run_dirs(patterns: Iterable[str]) -> List[Path]:
    run_dirs: List[Path] = []
    for pattern in patterns:
        matches = (
            [Path(match) for match in sorted(glob.glob(pattern))]
            if any(ch in pattern for ch in "*?[]")
            else [Path(pattern)]
        )
        for match in matches:
            if match.is_dir():
                run_dirs.append(match)
    if not run_dirs:
        raise SystemExit("no run directories matched --run-dirs")
    return run_dirs


def _read_text_file(path: Path, *, max_file_bytes: int) -> str:
    data = path.read_bytes()
    truncated = len(data) > max_file_bytes
    if truncated:
        data = data[:max_file_bytes]
    text = data.decode("utf-8", errors="replace")
    if truncated:
        text += f"\n\n[TRUNCATED at {max_file_bytes} bytes]\n"
    return text


def load_run_bundle(run_dir: Path, *, max_file_bytes: int = 200_000) -> Dict[str, str]:
    bundle: Dict[str, str] = {"__run_id__": run_dir.name}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(run_dir).as_posix()
        if rel.startswith("."):
            continue
        bundle[rel] = _read_text_file(path, max_file_bytes=max_file_bytes)
    if len(bundle) == 1:
        raise SystemExit(f"run directory has no readable text artefacts: {run_dir}")
    return bundle


def _build_judges(args: argparse.Namespace):
    if not args.enable_real_judges:
        return default_mock_judges()
    if os.environ.get(REAL_JUDGE_ENV_FLAG) != "1":
        raise SystemExit(
            f"--enable-real-judges requires {REAL_JUDGE_ENV_FLAG}=1; "
            "mock judges are the default"
        )
    judge_ids = [item.strip() for item in args.judges.split(",") if item.strip()]
    judges = make_real_judges(judge_ids)
    missing = [
        judge.api_key_env
        for judge in judges
        if not (judge.api_key or os.environ.get(judge.api_key_env))
    ]
    if missing:
        raise SystemExit("missing real-judge API key env vars: " + ", ".join(missing))
    return judges


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rubric = get_rubric(args.rubric)
    run_dirs = _expand_run_dirs(args.run_dirs)
    bundles = [
        load_run_bundle(run_dir, max_file_bytes=args.max_file_bytes)
        for run_dir in run_dirs
    ]
    runner = JuryRunner(judges=_build_judges(args), rubric=rubric, seed=args.seed)
    report = runner.score_runs(bundles)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    print(
        f"wrote Tier-2 jury report for {len(bundles)} run(s) "
        f"with {len(report.judges)} judge(s): {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

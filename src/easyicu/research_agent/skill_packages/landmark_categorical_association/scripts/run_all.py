"""The four-call standard workflow in one function, plus a small CLI.

    python -m easyicu.research_agent.skill_packages.landmark_categorical_association \
        --cohort cohort.parquet --spec spec.json --out results/

    python -m easyicu.research_agent.skill_packages.landmark_categorical_association \
        --example --out results_example/

The CLI is a convenience over ``run_all``; it adds no scientific behaviour.
Running it does not register anything with the research-agent pipeline, does
not touch Provider credentials and does not grant reporting authority: the
output directory is analysis-only material until the host's gates say otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from ..spec import LandmarkCategoricalSpec
from .example_data import example_spec, make_example_cohort
from .export_all import ExportReceipt, export_all
from .generate_all_plots import FigureArtifact, generate_all_plots
from .load_cohort import LoadedCohort, load_cohort
from .run_analysis import AnalysisResult, run_analysis


@dataclass(frozen=True)
class SkillRun:
    cohort: LoadedCohort
    result: AnalysisResult
    figures: dict[str, FigureArtifact]
    receipt: ExportReceipt

    @property
    def key_metrics(self) -> dict[str, Any]:
        return dict(self.result.key_metrics)


def run_all(
    cohort: str | Path | pd.DataFrame,
    spec: LandmarkCategoricalSpec | Mapping[str, Any],
    out_dir: str | Path,
    *,
    provenance: Optional[str] = None,
    verbose: bool = True,
) -> SkillRun:
    """load → analyse → plot → export, in the fixed order, into ``out_dir``."""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    loaded = load_cohort(cohort, spec, provenance=provenance, verbose=verbose)
    result = run_analysis(loaded, work_dir=target, verbose=verbose)
    figures = generate_all_plots(result, target, verbose=verbose)
    receipt = export_all(result, target, figures=figures, verbose=verbose)
    return SkillRun(cohort=loaded, result=result, figures=figures, receipt=receipt)


def _load_spec(path: Path) -> LandmarkCategoricalSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LandmarkCategoricalSpec.model_validate(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m easyicu.research_agent.skill_packages.landmark_categorical_association",
        description="Fixed-landmark categorical association: load → analyse → plot → export.",
    )
    parser.add_argument("--cohort", type=Path, help="cohort .parquet/.csv/.tsv (one row per ICU stay)")
    parser.add_argument("--spec", type=Path, help="specification JSON (LandmarkCategoricalSpec)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--provenance",
        default=None,
        help=(
            "where the cohort rows come from (e.g. 'official_demo:eicu_demo_v2_0_1'); "
            "recorded in manifest.json and report.md, 'undeclared_by_caller' when omitted"
        ),
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="run on the built-in synthetic example instead of --cohort/--spec",
    )
    parser.add_argument("--example-size", type=int, default=3000, help="synthetic example rows")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument(
        "--write-example-spec",
        type=Path,
        default=None,
        help="also write the example specification JSON to this path",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    verbose = not args.quiet
    if args.example:
        spec = example_spec()
        cohort: Any = make_example_cohort(args.example_size)
        if args.write_example_spec is not None:
            args.write_example_spec.write_text(
                json.dumps(spec.to_json_dict(), indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
    else:
        if args.cohort is None or args.spec is None:
            parser.error("--cohort and --spec are required unless --example is given")
        spec = _load_spec(args.spec)
        cohort = args.cohort
    run = run_all(cohort, spec, args.out, provenance=args.provenance, verbose=verbose)
    if verbose:
        metrics = run.key_metrics
        print(
            "\nHeadline (copied from key_metrics.csv): OR "
            f"{metrics['primary_contrast_level']} vs {metrics['reference_level']} = "
            f"{metrics['primary_or']:.3f} ({metrics['primary_or_ci_low']:.3f}–{metrics['primary_or_ci_high']:.3f}), "
            f"n={metrics['n_fit']}, events={metrics['n_events_fit']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through __main__.py
    sys.exit(main())


__all__ = ["SkillRun", "build_parser", "main", "run_all"]

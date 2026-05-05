"""Command-line entry point: ``easyicu-research-agent``.

Minimal CLI so users can run a pipeline from a parquet file without
writing Python::

    easyicu-research-agent \\
        --question "Is admission SOFA associated with ICU mortality?" \\
        --cohort path/to/cohort.parquet \\
        --database miiv \\
        --target-outcome death \\
        --workdir ./research_output

By default the CLI uses :class:`MockLLMClient`, so the command can run
offline. Pass ``--llm openai`` (and an ``OPENAI_API_KEY`` env var) to
use a real model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="easyicu-research-agent",
        description="Run an ICU-aware analysis agent over an EasyICU cohort parquet.",
    )
    p.add_argument("--question", required=True, help="Plain-language research question.")
    p.add_argument("--cohort", required=True, help="Path to cohort parquet (or CSV).")
    p.add_argument("--workdir", default="./research_output",
                   help="Output directory (default: ./research_output).")
    p.add_argument("--cohort-name", default="cohort")
    p.add_argument("--database", default="miiv",
                   help="Source database tag (miiv, eicu, hirid, aumc, sic, custom).")
    p.add_argument("--target-outcome", default=None,
                   help="Name of the primary outcome column.")
    p.add_argument("--cross-database", default=None,
                   help="Comma-separated list of databases for replication "
                        "(e.g. 'eicu,hirid').")
    p.add_argument("--inclusion", action="append", default=[],
                   help="Inclusion criterion (repeatable).")
    p.add_argument("--exclusion", action="append", default=[],
                   help="Exclusion criterion (repeatable).")
    p.add_argument("--llm", choices=["mock", "openai"], default="mock",
                   help="LLM backend (default: mock — runs offline).")
    p.add_argument("--openai-model", default="gpt-4o-mini",
                   help="Model name when --llm openai (default: gpt-4o-mini).")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="Per-step subprocess timeout in seconds (default: 300).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Lazy imports so --help works without pandas / openai installed.
    from .llm import MockLLMClient, OpenAIClient
    from .pipeline import ResearchAgentPipeline

    if args.llm == "openai":
        llm = OpenAIClient(model=args.openai_model)
    else:
        llm = MockLLMClient()

    pipeline = ResearchAgentPipeline(
        workdir=args.workdir,
        llm=llm,
        timeout_seconds=args.timeout,
    )

    cross_db: List[str] = (
        [s.strip() for s in args.cross_database.split(",") if s.strip()]
        if args.cross_database else []
    )

    result = pipeline.run(
        question=args.question,
        cohort=args.cohort,
        cohort_name=args.cohort_name,
        database=args.database,
        target_outcome=args.target_outcome,
        cross_database_validation=cross_db,
        inclusion_criteria=args.inclusion,
        exclusion_criteria=args.exclusion,
    )

    print(f"run_id:       {result.run_id}")
    print(f"workdir:      {result.workdir}")
    print(f"context:      {result.context_path}")
    print(f"plan:         {result.plan_path}")
    print(f"manifest:     {result.manifest_path}")
    print(f"report:       {result.report_path}")
    print(f"manuscript:   {result.manuscript_path}")
    print(f"evidence:     {result.evidence_count} artefacts")
    print(f"findings:     {result.findings_count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

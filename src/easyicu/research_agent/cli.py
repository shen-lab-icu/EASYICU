"""Command-line entry point: ``easyicu-research-agent``.

Minimal CLI so users can run a pipeline from a parquet file without
writing Python::

    easyicu-research-agent \\
        --question "Is admission SOFA associated with ICU mortality?" \\
        --cohort path/to/cohort.parquet \\
        --database miiv \\
        --target-outcome death \\
        --workdir ./research_output

The CLI requires an explicit ``--llm`` choice so main-path runs never
silently fall back to :class:`MockLLMClient`. Use ``--llm mock`` only
for tests or deterministic demos; use ``--llm openai`` (and an
``OPENAI_API_KEY`` env var) for real runs.
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
    p.add_argument("--question", required=False, help="Plain-language research question.")
    p.add_argument("--cohort", required=False, help="Path to cohort parquet (or CSV).")
    p.add_argument("--spec", default=None,
                   help="Optional YAML/JSON experiment spec. When set, it can provide question/cohort/runtime settings.")
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
    p.add_argument("--llm", choices=["mock", "openai"], default=None,
                   help="LLM backend. Required: choose mock for offline tests or openai for real runs.")
    p.add_argument("--openai-model", default="gpt-4o-mini",
                   help="Model name when --llm openai (default: gpt-4o-mini).")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="Per-step subprocess timeout in seconds (default: 300).")
    p.add_argument("--manuscript-language", choices=["en", "zh"], default="en",
                   help="Manuscript scaffold language (default: en).")
    p.add_argument("--context-top-k", type=int, default=None,
                   help="Optional top-K concept retrieval for long-context prompts.")
    p.add_argument("--latex-venue-template", default="article",
                   choices=["article", "nature", "npj", "lancet"],
                   help="LaTeX scaffold template (default: article).")
    p.add_argument("--enable-pubmed", action="store_true",
                   help="Augment curated citations with live PubMed E-utilities hits.")
    p.add_argument("--pubmed-email", default=None,
                   help="Optional NCBI E-utilities email for --enable-pubmed.")
    p.add_argument("--enable-tavily", action="store_true",
                   help="Augment citations with Tavily web/preprint/guideline search.")
    p.add_argument("--tavily-retmax", type=int, default=5,
                   help="Maximum Tavily results when --enable-tavily (default: 5).")
    p.add_argument("--enable-vlm-visual-qa", action="store_true",
                   help="Run optional VLM figure review using the configured LLM.")
    p.add_argument("--enable-llm-concept-audit", action="store_true",
                   help="Run optional LLM semantic concept-use audit after static checks.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Lazy imports so --help works without pandas / openai installed.
    from .llm import MockLLMClient, OpenAIClient
    from .pipeline import ResearchAgentPipeline
    from .experiment_spec import load_experiment_spec

    if args.llm is None:
        raise SystemExit("Choose an explicit --llm backend (`mock` or `openai`).")
    if args.llm == "openai":
        llm = OpenAIClient(model=args.openai_model)
    else:
        llm = MockLLMClient()

    if args.spec:
        spec = load_experiment_spec(args.spec)
        runtime_kwargs = spec.pipeline_kwargs()
        runtime_kwargs.update(
            {
                "llm": llm,
                "latex_venue_template": args.latex_venue_template,
                "enable_pubmed": args.enable_pubmed,
                "pubmed_email": args.pubmed_email,
                "enable_tavily": args.enable_tavily,
                "tavily_retmax": args.tavily_retmax,
                "enable_vlm_visual_qa": args.enable_vlm_visual_qa,
                "enable_llm_concept_audit": args.enable_llm_concept_audit,
            }
        )
        pipeline = ResearchAgentPipeline(**runtime_kwargs)
        result = pipeline.run_from_spec(spec)
    else:
        if not args.question or not args.cohort:
            raise SystemExit("--question and --cohort are required unless --spec is provided.")

        pipeline = ResearchAgentPipeline(
            workdir=args.workdir,
            llm=llm,
            timeout_seconds=args.timeout,
            manuscript_language=args.manuscript_language,
            context_top_k=args.context_top_k,
            latex_venue_template=args.latex_venue_template,
            enable_pubmed=args.enable_pubmed,
            pubmed_email=args.pubmed_email,
            enable_tavily=args.enable_tavily,
            tavily_retmax=args.tavily_retmax,
            enable_vlm_visual_qa=args.enable_vlm_visual_qa,
            enable_llm_concept_audit=args.enable_llm_concept_audit,
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

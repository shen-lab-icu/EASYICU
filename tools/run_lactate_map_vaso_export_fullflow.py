#!/usr/bin/env python3
"""Run the EasyICU research agent from an EasyICU export package.

This helper intentionally skips raw-database extraction/conversion and
starts from an existing EasyICU concept export directory. It builds the
deterministic shock-physiology case cohort, writes a cohort parquet, and
then runs ``ResearchAgentPipeline`` end-to-end with the built-in
``lactate_map_vaso_shock_mortality`` skill.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import easyicu.research_agent as ra


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run EasyICU research-agent full flow from an export package."
    )
    p.add_argument("--export-dir", required=True, help="Path to EasyICU export package.")
    p.add_argument("--workdir", required=True, help="Directory for the research-agent run.")
    p.add_argument(
        "--llm",
        choices=["mock", "openai"],
        required=True,
        help="LLM backend for the research-agent runtime.",
    )
    p.add_argument("--openai-model", default="openrouter/free")
    p.add_argument("--openai-base-url", default=None)
    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--openai-timeout", type=float, default=180.0)
    p.add_argument("--agent-timeout", type=float, default=900.0)
    p.add_argument("--database", default="miiv")
    p.add_argument("--cohort-name", default="miiv_lactate_map_vaso_24h")
    p.add_argument(
        "--manuscript-language",
        choices=["en", "zh"],
        default="en",
    )
    p.add_argument(
        "--enable-vlm-visual-qa",
        action="store_true",
        help="Enable optional VLM figure QA if the selected client supports it.",
    )
    p.add_argument(
        "--enable-llm-concept-audit",
        action="store_true",
        help="Enable optional LLM semantic concept audit.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    export_dir = Path(args.export_dir).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    package = ra.build_lactate_map_vaso_cohort_from_export(export_dir)
    built_dir = workdir / "built_case"
    written = package.write(built_dir, stem=args.cohort_name)

    if args.llm == "mock":
        llm = ra.MockLLMClient()
    else:
        llm = ra.OpenAIClient(
            model=args.openai_model,
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
            request_timeout=args.openai_timeout,
            extra_headers={
                "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
                "X-Title": "EasyICU research-agent",
            },
        )

    pipeline = ra.ResearchAgentPipeline(
        workdir=workdir,
        llm=llm,
        timeout_seconds=args.agent_timeout,
        manuscript_language=args.manuscript_language,
        enable_literature=True,
        enable_latex=True,
        enable_vlm_visual_qa=args.enable_vlm_visual_qa,
        enable_llm_concept_audit=args.enable_llm_concept_audit,
        enable_deterministic_code_fallback=False,
        enable_deterministic_planner_fallback=False,
    )

    result = pipeline.run(
        cohort=written["parquet"],
        cohort_name=args.cohort_name,
        database=args.database,
        target_outcome="death",
        skill="lactate_map_vaso_shock_mortality",
        notes=(
            "Started from an EasyICU export package rather than raw database tables. "
            f"Source manifest: {written['manifest']}"
        ),
    )

    print(f"run_id: {result.run_id}")
    print(f"workdir: {result.workdir}")
    print(f"context: {result.context_path}")
    print(f"plan: {result.plan_path}")
    print(f"manifest: {result.manifest_path}")
    print(f"report: {result.report_path}")
    print(f"manuscript: {result.manuscript_path}")
    print(f"evidence_count: {result.evidence_count}")
    print(f"findings_count: {result.findings_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

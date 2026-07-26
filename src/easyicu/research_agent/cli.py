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
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


HUMAN_REVIEW_PENDING_EXIT_CODE = 75
HUMAN_REVIEW_REJECTED_EXIT_CODE = 77


def _is_interactive_terminal() -> bool:
    """Return whether a same-process review conversation is possible."""

    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _pending_payload(pending: Any, *, entrypoint: str) -> Dict[str, Any]:
    payload = pending.model_dump(mode="json")
    payload.update(
        {
            "status": "human_review_pending",
            "terminal": False,
            "entrypoint": entrypoint,
            "external_resume_supported": False,
            f"resumable_via_{entrypoint}": False,
            "message": (
                "This pause supports same-process resume only. The current "
                f"{entrypoint} response does not retain a resume channel after "
                "it returns."
            ),
        }
    )
    return payload


def _emit_noninteractive_pending(pending: Any) -> int:
    """Emit one machine-readable pause and return the dedicated exit code."""

    print(
        json.dumps(
            _pending_payload(pending, entrypoint="cli"),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return HUMAN_REVIEW_PENDING_EXIT_CODE


def _prompt_review_decision(request: Any) -> tuple[str, str, str]:
    print(
        json.dumps(
            request.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    while True:
        token = input("Decision [approve/reject]: ").strip().casefold()
        if token in {"approve", "approved", "a"}:
            decision = "approved"
            break
        if token in {"reject", "rejected", "r"}:
            decision = "rejected"
            break
        print("Enter 'approve' or 'reject'.", file=sys.stderr)
    reviewer = ""
    while not reviewer:
        reviewer = input("Reviewer name: ").strip()
        if not reviewer:
            print("Reviewer name cannot be empty.", file=sys.stderr)
    note = input("Review note (optional): ").strip()
    return decision, reviewer, note


def _resume_interactive_review(pipeline: Any, pending: Any) -> Any:
    """Collect exact digest-bound decisions and resume before this process exits."""

    from .orchestration.workflow import HumanReviewDecision

    print(
        f"Run {pending.run_id} requires {len(pending.requests)} human review "
        f"decision(s). Resume scope: {pending.resume_scope}."
    )
    decisions = []
    for request in pending.requests:
        decision, reviewer, note = _prompt_review_decision(request)
        decisions.append(
            HumanReviewDecision(
                review_id=request.review_id,
                authority_sha256=request.authority_sha256,
                decision=decision,
                reviewer=reviewer,
                decided_at=datetime.now(timezone.utc).isoformat(),
                note=note,
            )
        )
    return pipeline.resume_human_review(decisions, run_id=pending.run_id)


def _print_pipeline_result(result: Any) -> None:
    print(f"run_id:       {result.run_id}")
    print(f"workdir:      {result.workdir}")
    print(f"context:      {result.context_path}")
    print(f"plan:         {result.plan_path}")
    print(f"manifest:     {result.manifest_path}")
    print(f"report:       {result.report_path}")
    print(f"manuscript:   {result.manuscript_path}")
    print(f"evidence:     {result.evidence_count} artefacts")
    print(f"findings:     {result.findings_count}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="easyicu-research-agent",
        description="Run an ICU-aware analysis agent over an EasyICU cohort parquet.",
    )
    p.add_argument(
        "--question", required=False, help="Plain-language research question."
    )
    p.add_argument("--cohort", required=False, help="Path to cohort parquet (or CSV).")
    p.add_argument(
        "--cohort-map",
        action="append",
        default=[],
        metavar="DB=PATH",
        help=(
            "Cross-database mode: repeatable database-to-cohort mapping, "
            "e.g. miiv=/path/a.parquet --cohort-map eicu=/path/b.parquet."
        ),
    )
    p.add_argument(
        "--spec",
        default=None,
        help="Optional YAML/JSON experiment spec. When set, it can provide question/cohort/runtime settings.",
    )
    p.add_argument(
        "--workdir",
        default="./research_output",
        help="Output directory (default: ./research_output).",
    )
    p.add_argument("--cohort-name", default="cohort")
    p.add_argument(
        "--database",
        default="miiv",
        help="Source database tag (miiv, eicu, hirid, aumc, sic, custom).",
    )
    p.add_argument(
        "--target-outcome", default=None, help="Name of the primary outcome column."
    )
    p.add_argument(
        "--cross-database",
        default=None,
        help="Comma-separated list of databases for replication (e.g. 'eicu,hirid').",
    )
    p.add_argument(
        "--inclusion",
        action="append",
        default=[],
        help="Inclusion criterion (repeatable).",
    )
    p.add_argument(
        "--exclusion",
        action="append",
        default=[],
        help="Exclusion criterion (repeatable).",
    )
    p.add_argument(
        "--llm",
        choices=["mock", "openai"],
        default=None,
        help="LLM backend. Required: choose mock for offline tests or openai for real runs.",
    )
    p.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="Model name when --llm openai (default: gpt-4o-mini).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-step subprocess timeout in seconds (default: 300).",
    )
    p.add_argument(
        "--manuscript-language",
        choices=["en", "zh"],
        default="en",
        help="Manuscript scaffold language (default: en).",
    )
    p.add_argument(
        "--context-top-k",
        type=int,
        default=None,
        help="Optional top-K concept retrieval for long-context prompts.",
    )
    p.add_argument(
        "--latex-venue-template",
        default="article",
        choices=["article", "nature", "npj", "lancet"],
        help="LaTeX scaffold template (default: article).",
    )
    p.add_argument(
        "--enable-pubmed",
        action="store_true",
        help="Augment curated citations with live PubMed E-utilities hits.",
    )
    p.add_argument(
        "--pubmed-email",
        default=None,
        help="Optional NCBI E-utilities email for --enable-pubmed.",
    )
    p.add_argument(
        "--enable-tavily",
        action="store_true",
        help="Augment citations with Tavily web/preprint/guideline search.",
    )
    p.add_argument(
        "--tavily-retmax",
        type=int,
        default=5,
        help="Maximum Tavily results when --enable-tavily (default: 5).",
    )
    vlm_group = p.add_mutually_exclusive_group()
    vlm_group.add_argument(
        "--enable-vlm-visual-qa",
        action="store_true",
        dest="enable_vlm_visual_qa",
        help="Force-enable model-based figure review.",
    )
    vlm_group.add_argument(
        "--disable-vlm-visual-qa",
        action="store_false",
        dest="enable_vlm_visual_qa",
        help="Force-disable model-based figure review even if the model looks vision-capable.",
    )
    p.set_defaults(enable_vlm_visual_qa=None)

    concept_group = p.add_mutually_exclusive_group()
    concept_group.add_argument(
        "--enable-llm-concept-audit",
        action="store_true",
        dest="enable_llm_concept_audit",
        help="Force-enable semantic concept-use audit after static checks.",
    )
    concept_group.add_argument(
        "--disable-llm-concept-audit",
        action="store_false",
        dest="enable_llm_concept_audit",
        help="Force-disable semantic concept-use audit.",
    )
    p.set_defaults(enable_llm_concept_audit=None)
    return p


def _parse_cohort_map(raw_items: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw in raw_items:
        if "=" not in raw:
            raise SystemExit(f"--cohort-map must be DB=PATH, got: {raw!r}")
        database, path = raw.split("=", 1)
        database = database.strip()
        path = path.strip()
        if not database or not path:
            raise SystemExit(f"--cohort-map must be DB=PATH, got: {raw!r}")
        mapping[database] = str(Path(path).resolve())
    return mapping


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Lazy imports so --help works without pandas / openai installed.
    from .providers.factory import build_provider_client
    from .providers.llm import OpenAIClient
    from .providers.mocks import MockLLMClient
    from .pipeline import ResearchAgentPipeline
    from .orchestration.experiment_spec import load_experiment_spec
    from .orchestration.workflow import HumanReviewPending, HumanReviewRejected

    if args.llm is None:
        raise SystemExit("Choose an explicit --llm backend (`mock` or `openai`).")
    if args.llm == "openai":
        llm = build_provider_client(
            provider="openai",
            model=args.openai_model,
            request_timeout=120.0,
            title="EasyICU research-agent CLI",
            client_cls=OpenAIClient,
        )
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
        cross_db_cohorts = _parse_cohort_map(args.cohort_map)
        if not args.question or (not args.cohort and not cross_db_cohorts):
            raise SystemExit(
                "--question and either --cohort or --cohort-map are required unless --spec is provided."
            )

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
            if args.cross_database
            else []
        )

        if cross_db_cohorts:
            cohorts = dict(cross_db_cohorts)
            if args.cohort:
                cohorts = {args.database: str(Path(args.cohort).resolve()), **cohorts}
            result = pipeline.replicate(
                question=args.question,
                cohorts=cohorts,
                target_outcome=args.target_outcome,
                cohort_name_prefix=args.cohort_name,
                inclusion_criteria=args.inclusion,
                exclusion_criteria=args.exclusion,
                manuscript_language=args.manuscript_language,
                stop_after_analysis=False,
            )
            print(f"replication_id: {result['replication_id']}")
            print(f"replication_dir: {result['replication_dir']}")
            print(f"comparison_csv: {result['comparison_csv']}")
            print(f"comparison_md: {result['comparison_md']}")
            if "summary_csv" in result:
                print(f"summary_csv: {result['summary_csv']}")
            if "summary_md" in result:
                print(f"summary_md: {result['summary_md']}")
            if "validation_report" in result:
                print(f"validation_report: {result['validation_report']}")
            return 0

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

    if isinstance(result, HumanReviewPending):
        if not _is_interactive_terminal():
            return _emit_noninteractive_pending(result)
        try:
            result = _resume_interactive_review(pipeline, result)
        except HumanReviewRejected as exc:
            print(
                json.dumps(
                    {
                        "status": "human_review_rejected",
                        "terminal": True,
                        "run_id": result.run_id,
                        "rejected_review_ids": list(exc.review_ids),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return HUMAN_REVIEW_REJECTED_EXIT_CODE
        except (EOFError, KeyboardInterrupt):
            print("", file=sys.stderr)
            return _emit_noninteractive_pending(result)

    _print_pipeline_result(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

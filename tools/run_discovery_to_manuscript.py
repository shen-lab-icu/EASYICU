#!/usr/bin/env python
"""Run the discovery-to-manuscript handoff.

This is the EasyICU analogue of AI-Scientist's ``idea.json -> experiment ->
figures -> writeup`` launcher. It starts from an idea-mining
``candidate_triage_report.json`` and writes a frozen handoff packet. With
``--run-analysis`` it also materialises a question-specific universe, launches
the aware research-agent workflow, and validates the final article package.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _bootstrap_imports() -> Path:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _bootstrap_imports()

from easyicu.research_agent.data_foundation import acquire_universe_for_question  # noqa: E402
from easyicu.research_agent.discovery_handoff import (  # noqa: E402
    build_handoff_from_row,
    load_discovery_ledger,
    select_discovery_row,
    write_handoff_packet,
)
from easyicu.research_agent.discovery_package import (  # noqa: E402
    validate_discovery_manuscript_package,
    write_discovery_package_assessment,
)
from easyicu.research_agent.discovery_story_figure import (  # noqa: E402
    render_discovery_story_figure,
)
from easyicu.research_agent.llm import OpenAIClient  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze an idea-mining handoff and optionally run analysis/writing."
    )
    parser.add_argument("--triage-report", required=True)
    parser.add_argument("--idea-index", type=int, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["agent_selected", "human_curated", "manual_scaffold"],
        default="agent_selected",
    )
    parser.add_argument("--selection-rationale", default=None)
    parser.add_argument("--research-question", default=None)
    parser.add_argument("--target-outcome", default="death")
    parser.add_argument(
        "--outcome-concepts",
        default=None,
        help=(
            "Comma-separated concept ids to materialise DETERMINISTICALLY as "
            "binary outcomes (each emits a bare <c> 0/1 column plus <c>_time "
            "onset), independent of the data-foundation agent's feature "
            "selection. Use when --target-outcome is a non-death outcome whose "
            "presence must not depend on the LLM picking it as a feature "
            "(e.g. --outcome-concepts aki --target-outcome aki). Defaults to "
            "the target outcome when it is not a per-stay summary suffix, else "
            "to death."
        ),
    )
    parser.add_argument("--database", default="miiv")
    parser.add_argument(
        "--out-root",
        default=str(
            REPO_ROOT
            / "research_output"
            / "discovery_to_manuscript"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        ),
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Materialise universe and launch the aware research-agent workflow.",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Prepared EasyICU export directory required when --run-analysis is set.",
    )
    parser.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument(
        "--model",
        default=os.environ.get("EASYICU_HOSTED_DEFAULT_MODEL", "gpt-5.4"),
    )
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--runner", choices=["subprocess", "docker"], default="subprocess")
    parser.add_argument("--llm-seed", type=int, default=None)
    parser.add_argument("--max-total-steps", type=int, default=None)
    parser.add_argument("--disable-replanning", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    triage_report = Path(args.triage_report).resolve()

    rows = load_discovery_ledger(triage_report)
    selected = select_discovery_row(rows, index=args.idea_index)
    handoff = build_handoff_from_row(
        selected,
        triage_report_path=triage_report,
        selection_mode=args.selection_mode,
        selection_rationale=args.selection_rationale,
        target_outcome=args.target_outcome,
        database=args.database,
        research_question=args.research_question,
    )
    handoff_path = write_handoff_packet(handoff, out_root / "discovery_handoff.json")
    print(f"[discovery] handoff: {handoff_path}")
    print(f"[discovery] selected topic: {handoff.candidate_topic}")

    if not args.run_analysis:
        print("[discovery] --run-analysis not set; stopping after frozen handoff.")
        return 0

    if not args.export_dir:
        raise SystemExit("--export-dir is required with --run-analysis")
    llm = OpenAIClient(
        model=args.model,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        request_timeout=args.request_timeout,
    )
    universe_dir = out_root / "universe"
    # Deterministic outcome materialisation: a non-death target outcome only
    # appears in the universe if the data-foundation agent happens to pick its
    # concept as a feature (brittle — the LLM may select a sibling like
    # aki_stage instead of aki, leaving the target column missing and the run
    # at 0 steps). When --outcome-concepts is given we pass them as outcome
    # concepts so the materialiser emits a bare 0/1 column (+ <c>_time onset)
    # regardless of feature selection.
    if args.outcome_concepts:
        outcome_concepts = tuple(
            c.strip() for c in args.outcome_concepts.split(",") if c.strip()
        )
    elif args.target_outcome and args.target_outcome != "death":
        outcome_concepts = (args.target_outcome,)
    else:
        outcome_concepts = ("death",)
    acquisition = acquire_universe_for_question(
        export_dir=Path(args.export_dir).resolve(),
        question=handoff.research_question,
        llm=llm,
        output_dir=universe_dir,
        stem="discovery_universe",
        target_outcome=handoff.target_outcome,
        outcome_concepts=outcome_concepts,
        database=handoff.database,
    )
    acquisition_path = out_root / "data_foundation_acquisition.json"
    acquisition_path.write_text(
        json.dumps(acquisition.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if acquisition.blocked or acquisition.universe_path is None:
        raise SystemExit(f"data foundation blocked: {acquisition.note}")

    # Make the long-format trajectory reachable inside the analysis sandbox.
    # The bench loads the cohort parquet into a DataFrame (the original universe
    # dir path is lost), so the runner's sibling auto-discovery cannot find the
    # trajectory. Export it in the environment instead: the bench subprocess
    # inherits it, and the runner's os.environ.copy() carries it through to the
    # sandbox, where the coder reads os.environ["TRAJECTORY_PARQUET"]. Keyed by
    # stay_id, so it stays valid through any cohort 纳排 re-pointing.
    trajectory_path = Path(acquisition.universe_path).with_name(
        f"{Path(acquisition.universe_path).stem}_trajectory.parquet"
    )
    if trajectory_path.exists():
        for traj_alias in (
            "TRAJECTORY_PARQUET",
            "EASYICU_TRAJECTORY_PARQUET",
            "COHORT_TRAJECTORY_PARQUET",
        ):
            os.environ[traj_alias] = str(trajectory_path)
        print(f"[discovery] trajectory: {trajectory_path}")

    jsonl_path = _write_ehrflowbench_row(
        out_root=out_root,
        handoff=handoff,
        cohort_path=acquisition.universe_path,
    )
    bench_root = out_root / "bench"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_research_agent_bench.py"),
        "--ehrflowbench-jsonl",
        str(jsonl_path),
        "--arms",
        "aware",
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--out-root",
        str(bench_root),
        "--runner",
        args.runner,
        "--request-timeout",
        str(args.request_timeout),
    ]
    if args.llm_seed is not None:
        cmd.extend(["--llm-seed", str(args.llm_seed)])
    if args.max_total_steps is not None:
        cmd.extend(["--max-total-steps", str(args.max_total_steps)])
    if args.disable_replanning:
        cmd.append("--disable-replanning")
    if args.reuse_existing:
        cmd.append("--reuse-existing")

    print("[discovery] running:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if completed.returncode != 0:
        return completed.returncode

    run_dir = _latest_aware_run_dir(bench_root)
    if run_dir is None:
        raise SystemExit(f"could not locate aware run under {bench_root}")
    write_handoff_packet(handoff, run_dir / "discovery_handoff.json")
    render_discovery_story_figure(run_dir=run_dir, handoff=handoff)
    assessment = validate_discovery_manuscript_package(run_dir=run_dir)
    assessment_path = write_discovery_package_assessment(
        assessment, run_dir / "discovery_package_assessment.json"
    )
    print(f"[discovery] package assessment: {assessment_path}")
    print(f"[discovery] package status: {assessment.status}")
    return 0 if assessment.package_ready else 3


def _write_ehrflowbench_row(
    *,
    out_root: Path,
    handoff,
    cohort_path: Path,
) -> Path:
    row: Dict[str, Any] = {
        "key": f"discovery_{handoff.literature_idea_id}",
        "name": handoff.candidate_topic[:120],
        "question": handoff.research_question,
        "cohort_path": str(cohort_path.resolve()),
        "target_outcome": handoff.target_outcome,
        "primary_predictor": handoff.resolved_predictor_concept or "agent_mined_idea",
        "expected_or_direction": 0,
        "kind": "descriptive_association",
        "inclusion_criteria": list(handoff.inclusion_criteria),
    }
    path = out_root / "discovery_ehrflowbench.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _latest_aware_run_dir(bench_root: Path) -> Optional[Path]:
    candidates = sorted(
        bench_root.glob("*/aware/run_*"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

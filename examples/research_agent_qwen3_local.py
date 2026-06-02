#!/usr/bin/env python
"""Drive the research agent with a local Qwen3-8B via vLLM.

Usage:
    python examples/research_agent_qwen3_local.py \\
        --base-url http://localhost:8000/v1 \\
        --model qwen3-8b

This exercises every deterministic safety / rigor layer we built
(baseline registry, reviewer round, reporting checklist, hypothesis
generator, causal audit, survival analysis, reproducibility envelope,
PRISMA flow, multiple-testing control, E-value, fairness, missing-data
sensitivity, notebook/lockfile, raw-EHR provenance) plus the actual
planner/coder/analyzer/writer chain running against a hosted LLM.

Outputs go to ``research_output/qwen3_local/run_*``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _build_cohort():
    """Synthetic SOFA-2 cohort with the usual missingness artefact."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 800
    age = rng.normal(65, 15, n).clip(18, 95)
    base = rng.integers(1, 14, size=n, endpoint=False)
    miss = rng.random(n) < 0.10
    truly_low = rng.random(n) < 0.05
    sofa2 = np.where(miss, 0, np.where(truly_low, 0, base))
    logit = -3.5 + 0.18 * sofa2 + 0.012 * (age - 65) + np.where(miss, 1.5, 0.0)
    p = 1.0 / (1.0 + np.exp(-logit))
    death = (rng.random(n) < p).astype(int)
    los = rng.gamma(2.0, 1.5 + 0.15 * sofa2, size=n).clip(0.1, 60)
    lact = rng.lognormal(0.4 + 0.08 * sofa2, 0.6, size=n).clip(0.5, 25)
    creat = rng.lognormal(0.05 + 0.04 * sofa2, 0.4, size=n).clip(0.1, 12)
    map_v = rng.normal(85 - 1.6 * sofa2, 12, size=n).clip(40, 130)
    vaso = (rng.random(n) < 1.0 / (1.0 + np.exp(-(-1.5 + 0.20 * sofa2)))).astype(int)
    return pd.DataFrame({
        "stay_id": np.arange(1, n + 1),
        "age": age,
        "sex": rng.choice(["M", "F"], size=n),
        "sofa2": sofa2,
        "lact": lact,
        "creat": creat,
        "map": map_v,
        "vaso": vaso,
        "los_icu": los,
        "death": death,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--workdir", default="research_output/qwen3_local")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skill", default="sofa_mortality")
    parser.add_argument(
        "--no-live-llm",
        action="store_true",
        help="Use MockLLMClient instead (diagnostic).",
    )
    args = parser.parse_args()

    # Make ``easyicu.research_agent`` importable from the source tree
    # without requiring ``pip install -e``.
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from easyicu.research_agent import (
        MockLLMClient,
        OpenAIClient,
        ResearchAgentPipeline,
    )

    cohort = _build_cohort()
    # The pipeline expects a parquet / CSV path or a DataFrame; we hand
    # it the DataFrame so it can hash + register it as provenance.
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    if args.no_live_llm:
        llm = MockLLMClient()
        print("[qwen3-local] Running with MockLLMClient (diagnostic mode).")
    else:
        # vLLM's OpenAI-compatible server accepts any non-empty bearer;
        # we still honour OPENAI_API_KEY if set.
        api_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or "local-vllm"
        )
        llm = OpenAIClient(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            request_timeout=600.0,
        )
        print(f"[qwen3-local] Using {args.model} @ {args.base_url}.")

    pipeline = ResearchAgentPipeline(
        workdir=workdir,
        llm=llm,
        timeout_seconds=600.0,
        enable_literature=True,
        enable_visual_qa=True,
        enable_memory=True,
        enable_latex=True,
        enable_reproducibility_envelope=True,
        llm_seed=args.seed,
        envelope_include_previews=False,
        enable_multiple_testing_correction=True,
        enable_causal_audit=True,
        enable_reporting_checklist=True,
        enable_reviewer_round=True,
        enable_fairness_subgroups=True,
        enable_hypothesis_generator=True,
        # Hosted models sometimes return malformed JSON / partial code;
        # allow the deterministic fallbacks so the run still finishes.
        enable_deterministic_planner_fallback=True,
        enable_deterministic_code_fallback=True,
        enable_deterministic_runner_repair=True,
        max_code_repair_attempts=2,
    )
    print("[qwen3-local] Running pipeline ...")
    result = pipeline.run(
        skill=args.skill,
        cohort=cohort,
        cohort_name="Qwen3 local smoke cohort",
        database="miiv",
        manuscript_authors=["A. Researcher", "B. Clinician"],
    )
    print(f"[qwen3-local] Done. run_id={result.run_id}")
    print(f"[qwen3-local] manifest    : {result.manifest_path}")
    print(f"[qwen3-local] report      : {result.report_path}")
    print(f"[qwen3-local] manuscript  : {result.manuscript_path}")
    print(f"[qwen3-local] evidence    : {result.evidence_count} records")
    print(f"[qwen3-local] findings    : {result.findings_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

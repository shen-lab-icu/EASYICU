#!/usr/bin/env python
"""Full paper via OpenRouter free model (DeepSeek V4 Flash).

Usage:
    export OPENROUTER_API_KEY='sk-or-v1-...'
    python examples/research_agent_openrouter_paper.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _build_cohort():
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
        "age": age, "sex": rng.choice(["M", "F"], size=n),
        "sofa2": sofa2, "lact": lact, "creat": creat,
        "map": map_v, "vaso": vaso, "los_icu": los, "death": death,
    })


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from easyicu.research_agent import OpenAIClient, ResearchAgentPipeline

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY first.")
        return

    llm = OpenAIClient(
        model="nousresearch/hermes-3-llama-3.1-405b:free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        request_timeout=120.0,
        extra_headers={
            "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
            "X-Title": "EasyICU research-agent",
        },
    )
    print("[openrouter] Using nousresearch/hermes-3-llama-3.1-405b:free")

    pipeline = ResearchAgentPipeline(
        workdir="research_output/openrouter_paper",
        llm=llm,
        timeout_seconds=120.0,
        enable_reproducibility_envelope=True,
        llm_seed=2026,
        enable_multiple_testing_correction=True,
        enable_causal_audit=True,
        enable_reporting_checklist=True,
        enable_reviewer_round=True,
        enable_fairness_subgroups=True,
        enable_pdf_render=True,
        enable_deterministic_planner_fallback=True,
        enable_deterministic_code_fallback=True,
        enable_deterministic_runner_repair=True,
        max_code_repair_attempts=2,
    )

    cohort = _build_cohort()
    result = pipeline.run(
        skill="sofa_mortality",
        cohort=cohort,
        cohort_name="MIMIC-IV-like synthetic ICU cohort",
        database="miiv",
        manuscript_title=(
            "Association Between Admission SOFA-2 Score and ICU Mortality: "
            "A Traceable Agent-Assisted Analysis"
        ),
        manuscript_authors=["A. Researcher", "B. Clinician", "C. Data Scientist"],
    )
    run_dir = Path(result.manifest_path).parent
    print(f"[openrouter] Done. run_id={result.run_id}")
    print(f"[openrouter] evidence={result.evidence_count} findings={result.findings_count}")
    pdf = run_dir / "manuscript_scaffold.pdf"
    if pdf.exists():
        print(f"[openrouter] PDF: {pdf} ({pdf.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"[openrouter] No PDF. Check {run_dir / 'manuscript_scaffold.pdfrender.log'}")
    print(f"[openrouter] Manuscript: {run_dir / 'manuscript_scaffold_bound.md'}")


if __name__ == "__main__":
    main()

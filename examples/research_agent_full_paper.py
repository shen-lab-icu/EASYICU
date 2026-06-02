#!/usr/bin/env python
"""Full paper run: WriterAgent produces complete manuscript → PDF.

Usage:
    python examples/research_agent_full_paper.py
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

    api_key = os.environ.get("OPENAI_API_KEY") or "local-vllm"
    base_url = os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1"
    model = os.environ.get("EASYICU_MODEL") or "qwen3-coder-30b"

    llm = OpenAIClient(
        model=model, api_key=api_key, base_url=base_url, request_timeout=600.0,
    )
    print(f"[full-paper] Using {model} @ {base_url}")

    pipeline = ResearchAgentPipeline(
        workdir="research_output/full_paper",
        llm=llm,
        timeout_seconds=600.0,
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
    print(f"[full-paper] Done. run_id={result.run_id}")
    print(f"[full-paper] evidence={result.evidence_count} findings={result.findings_count}")
    print(f"[full-paper] manuscript: {run_dir / 'manuscript_scaffold_bound.md'}")
    pdf = run_dir / "manuscript_scaffold.pdf"
    if pdf.exists():
        print(f"[full-paper] PDF: {pdf} ({pdf.stat().st_size / 1024:.1f} KB)")
    else:
        print("[full-paper] PDF not generated (check pdfrender log)")


if __name__ == "__main__":
    main()

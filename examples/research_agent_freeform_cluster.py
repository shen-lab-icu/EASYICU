#!/usr/bin/env python
"""Free-form clustering test: no skill, agent decides everything.

Proves that the ICU safety layer (ConceptUsageAuditor +
AnalysisPatternAuditor) fires on a clustering task without any
hardcoded skill.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _build_cohort():
    rng = np.random.default_rng(7)
    n = 600
    age = rng.normal(65, 15, n).clip(18, 95)
    sofa2 = rng.integers(0, 15, size=n)
    lact = rng.lognormal(0.4 + 0.08 * sofa2, 0.6, size=n).clip(0.5, 25)
    creat = rng.lognormal(0.05 + 0.04 * sofa2, 0.4, size=n).clip(0.1, 12)
    map_v = rng.normal(85 - 1.6 * sofa2, 12, size=n).clip(40, 130)
    vaso = (rng.random(n) < 1.0 / (1.0 + np.exp(-(-1.5 + 0.20 * sofa2)))).astype(int)
    logit = -3.5 + 0.18 * sofa2 + 0.012 * (age - 65)
    p = 1.0 / (1.0 + np.exp(-logit))
    death = (rng.random(n) < p).astype(int)
    return pd.DataFrame({
        "stay_id": np.arange(1, n + 1),
        "age": age,
        "sex": rng.choice(["M", "F"], size=n),
        "sofa2": sofa2,
        "lact": lact,
        "creat": creat,
        "map": map_v,
        "vaso": vaso,
        "los_icu": rng.gamma(2.0, 1.5 + 0.15 * sofa2, size=n).clip(0.1, 60),
        "death": death,
    })


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from easyicu.research_agent import OpenAIClient, ResearchAgentPipeline

    api_key = os.environ.get("OPENAI_API_KEY") or "local-vllm"
    base_url = os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1"
    model = os.environ.get("EASYICU_MODEL") or "qwen3-coder-30b"

    llm = OpenAIClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        request_timeout=600.0,
    )
    print(f"[cluster-test] Using {model} @ {base_url}")

    pipeline = ResearchAgentPipeline(
        workdir="research_output/freeform_cluster",
        llm=llm,
        timeout_seconds=600.0,
        enable_reproducibility_envelope=True,
        llm_seed=2026,
        enable_multiple_testing_correction=True,
        enable_causal_audit=True,
        enable_reporting_checklist=True,
        enable_reviewer_round=True,
        enable_fairness_subgroups=True,
        enable_deterministic_planner_fallback=True,
        enable_deterministic_code_fallback=True,
        enable_deterministic_runner_repair=True,
        max_code_repair_attempts=2,
    )

    cohort = _build_cohort()
    result = pipeline.run(
        question=(
            "Can ICU patients be clustered into distinct phenotypes "
            "based on admission SOFA-2, lactate, creatinine, MAP and "
            "vasopressor use? Do these clusters differ in ICU mortality?"
        ),
        cohort=cohort,
        cohort_name="Synthetic ICU cohort for clustering",
        database="miiv",
        target_outcome="death",
    )
    print(f"[cluster-test] Done. run_id={result.run_id}")
    print(f"[cluster-test] manifest : {result.manifest_path}")
    print(f"[cluster-test] report   : {result.report_path}")
    print(f"[cluster-test] manuscript: {result.manuscript_path}")
    print(f"[cluster-test] evidence : {result.evidence_count}")
    print(f"[cluster-test] findings : {result.findings_count}")

    # Check if pattern auditor fired.
    import json
    manifest = json.loads(Path(result.manifest_path).read_text())
    pattern_findings = [
        f for f in manifest["findings"]
        if f.get("validator") == "analysis_pattern_auditor"
    ]
    print(f"[cluster-test] pattern_auditor findings: {len(pattern_findings)}")
    for f in pattern_findings[:5]:
        print(f"  [{f['severity']}] {f['message'][:120]}")


if __name__ == "__main__":
    main()

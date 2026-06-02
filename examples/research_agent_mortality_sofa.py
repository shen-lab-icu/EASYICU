"""End-to-end demo for ``easyicu.research_agent``.

This demo is deliberately self-contained: it generates a small
synthetic ICU cohort *that reproduces the SOFA2==0 missingness
anomaly* and runs the full agent pipeline against it using the
:class:`MockLLMClient` (no API key required).

Run it with::

    python examples/research_agent_mortality_sofa.py

Outputs land in ``./research_output/<run_id>/`` and include:

* ``research_context.json`` — ICU-aware context built from the cohort
* ``analysis_plan.json`` — multi-step plan emitted by the planner
* ``steps/<step_id>/analysis.py`` — generated analysis scripts
* ``steps/<step_id>/outputs/`` — produced tables and figures
* ``evidence/evidence_index.json`` — hashed provenance index
* ``manuscript_scaffold_bound.md`` — manuscript with placeholders resolved
* ``results_report.md`` — human-readable run summary
* ``manifest.json`` — top-level provenance manifest

The synthetic cohort is intentionally rigged so that:

1. SOFA2 score increases mortality monotonically EXCEPT at score==0,
   where the cohort contains many patients with missing component
   inputs that defaulted to zero. The :class:`StatisticalValidator`
   should flag the non-monotonic stratum, demonstrating the
   ICU-specific rule the agent layer enforces.
2. Lactate is right-skewed, so the script chooses median (IQR) over
   mean (SD) — exercising the ``ConceptUsageAuditor``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build_synthetic_cohort(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Build a small synthetic ICU cohort with the SOFA2==0 anomaly baked in."""
    rng = np.random.default_rng(seed)

    age = rng.normal(loc=65, scale=15, size=n).clip(18, 95)
    sex = rng.choice(["M", "F"], size=n, p=[0.55, 0.45])

    # Most patients get a SOFA2 in 0..15 with exposure-driven distribution.
    base_sofa2 = rng.integers(low=1, high=15, size=n, endpoint=False)

    # Designate ~10% of patients as having component-level missingness.
    # Their components default to 0 → composite SOFA2 reads as 0 even
    # though they are sicker than average. This is the canonical
    # MNAR-like artefact we want the validator to flag.
    missing_components = rng.random(n) < 0.10
    sofa2 = np.where(missing_components, 0, base_sofa2)
    # the truly-zero (low severity) tail
    truly_low = rng.random(n) < 0.05
    sofa2 = np.where(truly_low & ~missing_components, 0, sofa2)

    # Mortality rises with SOFA2 (logit), but the missing-component
    # patients are sicker than their score suggests.
    logit = -3.5 + 0.18 * sofa2 + 0.012 * (age - 65)
    logit += np.where(missing_components, 1.5, 0.0)  # hidden severity
    p_death = 1.0 / (1.0 + np.exp(-logit))
    death = (rng.random(n) < p_death).astype(int)

    # ICU LoS — right-skewed; rises with SOFA2 and age.
    los_icu = rng.gamma(shape=2.0, scale=1.5 + 0.15 * sofa2, size=n).clip(0.1, 60)

    # Lactate — right-skewed.
    lact = rng.lognormal(mean=0.4 + 0.08 * sofa2, sigma=0.6, size=n).clip(0.5, 25)

    # Creatinine — right-skewed; correlates weakly with SOFA2.
    creat = rng.lognormal(mean=0.05 + 0.04 * sofa2, sigma=0.4, size=n).clip(0.1, 12)

    # MAP — continuous, lower with higher SOFA2.
    map_mmhg = rng.normal(loc=85 - 1.6 * sofa2, scale=12, size=n).clip(40, 130)

    # Vasopressor — binary, more likely with higher SOFA2.
    vaso_p = 1.0 / (1.0 + np.exp(-(-1.5 + 0.20 * sofa2)))
    vaso = (rng.random(n) < vaso_p).astype(int)

    return pd.DataFrame({
        "stay_id": np.arange(1, n + 1),
        "age": age,
        "sex": sex,
        "sofa2": sofa2,
        "lact": lact,
        "creat": creat,
        "map": map_mmhg,
        "vaso": vaso,
        "los_icu": los_icu,
        "death": death,
    })


def main() -> int:
    # Make sure the local source tree is importable when run from a checkout.
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from easyicu.research_agent import ResearchAgentPipeline
    from easyicu.research_agent.llm import MockLLMClient

    cohort = build_synthetic_cohort()
    workdir = repo_root / "research_output"
    workdir.mkdir(parents=True, exist_ok=True)

    pipeline = ResearchAgentPipeline(workdir=workdir, llm=MockLLMClient())
    result = pipeline.run(
        question="Is admission SOFA-2 score associated with ICU mortality?",
        cohort=cohort,
        cohort_name="synthetic_demo_cohort",
        database="synthetic",
        target_outcome="death",
        cross_database_validation=["miiv", "eicu"],
        inclusion_criteria=[
            "First ICU admission",
            "Age ≥ 18 years",
            "ICU length of stay ≥ 6 hours",
        ],
        exclusion_criteria=[
            "Discharged within first 6 hours",
        ],
        notes=(
            "Synthetic cohort generated by examples/research_agent_mortality_sofa.py "
            "with the SOFA2==0 missingness anomaly baked in to exercise the validator."
        ),
    )

    print()
    print("=== Pipeline finished ===")
    for k, v in result.model_dump().items():
        print(f"{k:>16}: {v}")
    print()
    print("Open the following files to inspect the run:")
    print(f"  Report:     {result.report_path}")
    print(f"  Manuscript: {result.manuscript_path}")
    print(f"  Manifest:   {result.manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

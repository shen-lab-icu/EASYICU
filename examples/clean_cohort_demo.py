"""Clean synthetic ICU cohort demo — no SOFA=0 missingness anomaly.

This script generates a realistic synthetic ICU cohort where:
- Mortality increases *monotonically* with SOFA-2 score (no artefact)
- No component-missingness trick: SOFA=0 truly means low severity
- All variables follow clinically plausible distributions

Then it runs the full ResearchAgentPipeline with a real LLM so you can
evaluate the quality of the generated manuscript end-to-end.

Usage::

    export OPENROUTER_API_KEY='sk-or-v1-...'
    export OPENROUTER_BASE_URL='https://openrouter.ai/api/v1'
    export EASYICU_HOSTED_DEFAULT_MODEL='openai/gpt-oss-120b:free'
    python examples/clean_cohort_demo.py

Outputs land in  research_output/clean_demo/<run_id>/
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cohort generation
# ---------------------------------------------------------------------------

def build_clean_cohort(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Clean ICU cohort: mortality rises monotonically with SOFA-2.

    Design choices:
    - SOFA-2 range 0–14, distributed roughly log-normal (most patients 2–8)
    - Mortality logit = -4.0 + 0.28 * sofa2 + 0.010 * (age - 65)
      → ~4% at SOFA=0, ~45% at SOFA=14
    - No component-missingness artefact: every SOFA=0 is genuinely low acuity
    - Lactate, creatinine, MAP correlated with SOFA in realistic ranges
    - Vasopressor use rises with SOFA (sigmoid)
    """
    rng = np.random.default_rng(seed)

    # --- Demographics ---
    age = rng.normal(loc=63, scale=14, size=n).clip(18, 95)
    sex = rng.choice(["M", "F"], size=n, p=[0.56, 0.44])

    # --- SOFA-2: log-normal shaped, clipped 0–14 ---
    raw = rng.lognormal(mean=1.5, sigma=0.65, size=n)
    sofa2 = np.round(raw).clip(0, 14).astype(int)

    # --- Mortality: clean monotonic relationship ---
    logit = -4.0 + 0.28 * sofa2 + 0.010 * (age - 65)
    p_death = 1.0 / (1.0 + np.exp(-logit))
    death = (rng.random(n) < p_death).astype(int)

    # --- ICU length of stay (days) ---
    los_icu = rng.gamma(shape=2.0, scale=1.2 + 0.14 * sofa2, size=n).clip(0.1, 60)

    # --- Lactate (mmol/L) — right-skewed ---
    lact = rng.lognormal(mean=0.3 + 0.07 * sofa2, sigma=0.55, size=n).clip(0.4, 20)

    # --- Creatinine (mg/dL) ---
    creat = rng.lognormal(mean=0.04 + 0.035 * sofa2, sigma=0.38, size=n).clip(0.1, 10)

    # --- MAP (mmHg) ---
    map_mmhg = rng.normal(loc=86 - 1.5 * sofa2, scale=11, size=n).clip(40, 130)

    # --- Vasopressor (binary) ---
    vaso_p = 1.0 / (1.0 + np.exp(-(-1.8 + 0.22 * sofa2)))
    vaso = (rng.random(n) < vaso_p).astype(int)

    return pd.DataFrame({
        "stay_id":  np.arange(1, n + 1),
        "age":      age,
        "sex":      sex,
        "sofa2":    sofa2,
        "lact":     lact,
        "creat":    creat,
        "map":      map_mmhg,
        "vaso":     vaso,
        "los_icu":  los_icu,
        "death":    death,
    })


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _bootstrap():
    here = Path(__file__).resolve().parent
    src = here.parent / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def _make_client(model: str, api_key: str, base_url: str):
    """Return an OpenAIClient wrapped with exponential-backoff retry on 429/5xx."""
    import random
    from easyicu.research_agent import OpenAIClient
    from easyicu.research_agent.llm import openrouter_reasoning_extra_body

    kwargs = dict(
        model=model,
        api_key=api_key,
        base_url=base_url,
        request_timeout=180.0,
        extra_headers={
            "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
            "X-Title": "EasyICU clean-cohort demo",
        },
    )
    extra_body = openrouter_reasoning_extra_body(model)
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    inner = OpenAIClient(**kwargs)

    # Wrap complete() with retry logic so 429 / transient 5xx don't abort the run.
    _orig_complete = inner.complete

    def _complete_with_retry(messages, *, max_tokens=2048, temperature=0.2):
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                return _orig_complete(messages, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:
                code = getattr(getattr(exc, "response", None), "status_code", None)
                is_rate = "429" in str(exc) or code == 429
                is_server = "5" in str(code) if code else False
                if (is_rate or is_server) and attempt < max_attempts - 1:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"   ⏳ Rate-limited (attempt {attempt+1}/{max_attempts}), "
                          f"retrying in {wait:.1f}s …")
                    time.sleep(wait)
                else:
                    raise

    inner.complete = _complete_with_retry  # type: ignore[method-assign]
    return inner


def main() -> int:
    _bootstrap()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
        help="Override the LLM model (e.g. 'meta-llama/llama-3.1-8b-instruct:free')")
    args = parser.parse_args()

    api_key  = os.environ.get("OPENROUTER_API_KEY", "")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model    = args.model or (
        os.environ.get("EASYICU_HOSTED_DEFAULT_MODEL")
        or os.environ.get("EASYICU_SMOKE_MODEL")
        or "openai/gpt-oss-120b:free"
    )

    if not api_key:
        print("❌  OPENROUTER_API_KEY not set. Export it and rerun.")
        return 1

    print(f"Model : {model}")
    print(f"Endpoint: {base_url}")

    from easyicu.research_agent import ResearchAgentPipeline

    cohort   = build_clean_cohort(n=800)
    workdir  = Path(__file__).resolve().parent.parent / "research_output" / "clean_demo"
    workdir.mkdir(parents=True, exist_ok=True)

    # Quick sanity-check on the cohort
    n_total  = len(cohort)
    n_deaths = cohort["death"].sum()
    print(f"\nCohort: {n_total} patients, {n_deaths} deaths "
          f"({100*n_deaths/n_total:.1f}%)")
    by_sofa = cohort.groupby("sofa2")["death"].agg(["sum","count"])
    by_sofa["rate"] = by_sofa["sum"] / by_sofa["count"]
    print("\nMortality by SOFA-2 (should be monotonically increasing):")
    print(by_sofa[["sum","count","rate"]].rename(
        columns={"sum":"deaths","count":"n","rate":"mort_rate"}).to_string())
    print()

    client   = _make_client(model, api_key, base_url)
    pipeline = ResearchAgentPipeline(
        workdir=workdir,
        llm=client,
        timeout_seconds=360.0,
    )

    print("Running pipeline … (this takes ~3–5 min on free-tier models)\n")
    started = time.monotonic()
    try:
        result = pipeline.run(
            question="Is admission SOFA-2 score independently associated with ICU mortality?",
            cohort=cohort,
            cohort_name="clean_synthetic_icu_cohort",
            database="synthetic",
            target_outcome="death",
            cross_database_validation=["miiv", "eicu"],
            inclusion_criteria=[
                "First ICU admission only",
                "Age ≥ 18 years",
                "ICU length of stay ≥ 4 hours",
            ],
            exclusion_criteria=[
                "Missing SOFA-2 components at admission",
                "Readmissions",
            ],
            notes=(
                "Clean synthetic cohort — no component-missingness artefact. "
                "Mortality is monotonically increasing with SOFA-2 by design. "
                "Goal: evaluate end-to-end manuscript quality."
            ),
        )
    except Exception as exc:
        import traceback
        print(f"❌ Pipeline raised: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    elapsed = time.monotonic() - started
    rd = Path(result.workdir).resolve()

    print(f"\n✅ Pipeline finished in {elapsed:.0f}s")
    print(f"   Evidence items : {result.evidence_count}")
    print(f"   Findings       : {result.findings_count}")
    print("\n--- Deliverables ---")
    deliverables = {
        "manifest"          : rd / "manifest.json",
        "report"            : rd / "results_report.md",
        "manuscript (md)"   : rd / "manuscript_scaffold.md",
        "manuscript (bound)": rd / "manuscript_scaffold_bound.md",
        "manuscript (tex)"  : rd / "manuscript_scaffold.tex",
        "bibliography"      : rd / "manuscript_scaffold.bib",
    }
    for label, path in deliverables.items():
        marker = "✅" if path.exists() else "❌"
        print(f"   {marker}  {label:<22} {path}")

    # Count unresolved placeholders
    bound = rd / "manuscript_scaffold_bound.md"
    if bound.exists():
        text = bound.read_text(encoding="utf-8")
        missing = text.count("[evidence missing:")
        if missing == 0:
            print("\n🎉 Manuscript bound cleanly — zero [evidence missing] placeholders!")
        else:
            print(f"\n⚠️  {missing} [evidence missing] placeholder(s) remain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

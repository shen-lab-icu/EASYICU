"""Real-LLM end-to-end test for the EasyICU research agent.

This script runs the synthetic SOFA cohort through the **full**
pipeline (planner → coder → runner → analyzer → writer → bibtex/latex
export) using a real LLM, not the deterministic ``MockLLMClient``.
It is the first end-to-end verification that planner / coder /
writer prompts survive contact with a chat model whose output is,
by definition, not under our direct control.

The script is opinionated about provider but provider-agnostic about
host: anything with an OpenAI-compatible chat-completions endpoint
works. Defaults are tuned for the **OpenRouter free tier**.

Default model is ``openai/gpt-oss-120b:free`` because it is currently a
stronger free OpenRouter option for structured writing and evidence-bound
manuscript output. Override with ``EASYICU_SMOKE_MODEL`` or
``--model``.

Usage::

    # One-line setup using the user's tested config:
    export OPENROUTER_API_KEY='sk-or-v1-...'
    export OPENROUTER_BASE_URL='https://openrouter.ai/api/v1'
    export EASYICU_SMOKE_MODEL='openai/gpt-oss-120b:free'
    python examples/research_agent_real_llm_smoke.py

    # Single-temperature mode (cheaper, recommended for the first run):
    python examples/research_agent_real_llm_smoke.py --temperature 0.1

    # OpenAI proper:
    OPENAI_API_KEY=sk-... EASYICU_SMOKE_PROVIDER=openai \\
        EASYICU_SMOKE_MODEL=gpt-4o-mini \\
        python examples/research_agent_real_llm_smoke.py

The exit code is 0 on success, 1 if any acceptance check fails. CI
should NOT run this script (it spends real tokens); it is intended
for local pre-submission verification.

When the run finishes the script prints absolute paths to:

* ``manifest.json``            — full provenance (what was registered)
* ``results_report.md``        — one-page narrative report
* ``manuscript_scaffold.md``   — manuscript draft from WriterAgent
* ``manuscript_scaffold_bound.md`` — same draft with ``{evidence:...}``
                                     markers replaced by registered ids
* ``manuscript_scaffold.tex``  — LaTeX export with \\cite{} / \\nocite{}
* ``manuscript_scaffold.bib``  — auto-generated BibTeX bibliography
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional


def _bootstrap_imports():
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def _make_client(provider: str, model: str, api_key: Optional[str]):
    """Construct an OpenAIClient configured for the chosen provider."""
    from easyicu.research_agent import OpenAIClient
    from easyicu.research_agent.llm import openrouter_reasoning_extra_body

    if provider == "openrouter":
        if not api_key:
            raise SystemExit(
                "OPENROUTER_API_KEY is required for provider=openrouter. "
                "Either set the env var or pass --api-key."
            )
        base_url = (
            os.environ.get("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        kwargs = dict(
            model=model,
            api_key=api_key,
            base_url=base_url,
            request_timeout=180.0,
            extra_headers={
                "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
                "X-Title": "EasyICU research-agent smoke test",
            },
        )
        extra_body = openrouter_reasoning_extra_body(model)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        return OpenAIClient(**kwargs)
    if provider == "openai":
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY is required for provider=openai. "
                "Either set the env var or pass --api-key."
            )
        return OpenAIClient(model=model, api_key=api_key, request_timeout=180.0)
    raise SystemExit(f"Unknown provider {provider!r}; expected 'openrouter' or 'openai'.")


def _check_acceptance(result, run_dir: Path) -> bool:
    """Return True iff the run satisfies the acceptance criteria.

    The criteria mirror those in :mod:`tests/research_agent/test_pipeline`
    so a real-LLM run is held to the same bar as the mock pipeline.
    """
    ok = True

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ no manifest.json at {manifest_path}")
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1) Evidence kinds — should span code/log/table/figure/statistic.
    kinds = {e.get("kind") for e in manifest.get("evidence", [])}
    expected_kinds = {"code", "log", "table", "figure", "statistic"}
    missing_kinds = expected_kinds - kinds
    if missing_kinds:
        print(f"⚠️  missing evidence kinds: {missing_kinds} (got {kinds})")
        # not a hard failure — small models sometimes skip the figure step
    else:
        print(f"✅ evidence kinds cover all expected: {sorted(kinds)}")

    # 2) Manuscript should bind cleanly.
    bound_path = run_dir / "manuscript_scaffold_bound.md"
    if not bound_path.exists():
        print(f"❌ bound manuscript missing at {bound_path}")
        ok = False
    else:
        bound = bound_path.read_text(encoding="utf-8")
        n_missing = bound.count("[evidence missing:")
        if n_missing > 0:
            print(f"❌ {n_missing} unresolved [evidence missing: …] line(s) "
                  "in bound manuscript")
            ok = False
        else:
            print("✅ bound manuscript has zero [evidence missing] lines")

    # 3) At least one validator finding should reference the SOFA-zero anomaly
    #    (the synthetic cohort is rigged to produce one).
    manifest_findings = manifest.get("findings", [])
    error_findings = [
        f for f in manifest_findings
        if str(f.get("severity", "")).lower() == "error"
    ]
    if error_findings:
        print(f"❌ manifest contains {len(error_findings)} error finding(s):")
        for f in error_findings[:5]:
            print(f"   - {f.get('validator', '?')}: {f.get('message', '')}")
        ok = False
    else:
        print("✅ manifest has no error-severity findings")

    finding_msgs = " ".join(f.get("message", "") for f in manifest_findings)
    if "non-monotonic" in finding_msgs.lower() or "exceeds" in finding_msgs.lower():
        print("✅ validator surfaced the SOFA-zero anomaly")
    else:
        print("❌ no SOFA-zero anomaly finding; the audit step may not have run "
              "or sofa_strata.csv may not have been produced")
        ok = False

    # 4) Plan should have at least three steps.
    plan_path = run_dir / "analysis_plan.json"
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            n_steps = len(plan.get("steps", []))
            if n_steps < 3:
                print(f"❌ plan has only {n_steps} step(s); expected ≥3")
                ok = False
            else:
                print(f"✅ plan has {n_steps} step(s)")
        except Exception as exc:
            print(f"❌ could not parse analysis_plan.json: {exc}")
            ok = False

    return ok


def _load_concept_dictionary(path: Optional[Path]) -> Optional[dict]:
    """Best-effort load of an EasyICU concept_dictionary.csv into the
    ``concept_descriptions`` dict that ResearchContext accepts.

    The official export uses columns ``column,group,unit_or_type,
    nonmissing_n,missing_n,missing_pct,mapping_note``. We compose a
    one-line human-readable description per column from those fields.
    """
    if path is None or not path.exists():
        return None
    import pandas as pd
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "column" not in df.columns:
        return None
    out: dict = {}
    for _, row in df.iterrows():
        col = str(row["column"])
        bits = []
        if "group" in df.columns and pd.notna(row.get("group")):
            bits.append(str(row["group"]))
        if "unit_or_type" in df.columns and pd.notna(row.get("unit_or_type")):
            bits.append(str(row["unit_or_type"]))
        if "missing_pct" in df.columns and pd.notna(row.get("missing_pct")):
            try:
                bits.append(f"missing={float(row['missing_pct']):.1f}%")
            except (TypeError, ValueError):
                pass
        if "mapping_note" in df.columns and pd.notna(row.get("mapping_note")):
            note = str(row["mapping_note"]).strip()
            if note:
                bits.append(note)
        if bits:
            out[col] = " | ".join(bits)
    return out or None


def _resolve_cohort(args):
    """Pick the cohort: real parquet/csv if --cohort-parquet, else synthetic.

    Returns ``(cohort_df, cohort_name, database, target_outcome,
    question, concept_descriptions)``.
    """
    if args.cohort_parquet:
        path = Path(args.cohort_parquet).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"--cohort-parquet not found: {path}")
        import pandas as pd
        if path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        elif path.suffix.lower() in {".csv", ".tsv"}:
            df = pd.read_csv(path, sep=("\t" if path.suffix.lower() == ".tsv" else ","))
        else:
            raise SystemExit(
                f"unsupported cohort file extension {path.suffix}; "
                "expected .parquet, .csv, or .tsv"
            )
        cohort_name = args.cohort_name or path.stem
        database = args.database or "miiv"
        # Auto-detect the outcome column. The user's MIMIC-IV
        # export uses ``in_hospital_mortality``; the synthetic cohort
        # uses ``death``. Honour --target-outcome when set.
        if args.target_outcome:
            target = args.target_outcome
        elif "in_hospital_mortality" in df.columns:
            target = "in_hospital_mortality"
        elif "death" in df.columns:
            target = "death"
        elif "mortality" in df.columns:
            target = "mortality"
        else:
            raise SystemExit(
                "Could not auto-detect outcome column; pass --target-outcome. "
                f"Columns present: {list(df.columns)[:20]}…"
            )
        question = (
            args.question
            or f"Is admission SOFA-2 score associated with {target} in {database}?"
        )
        # Auto-pick concept_dictionary.csv from the same export package
        # directory if the user didn't pass one explicitly.
        cd_path = (
            Path(args.concept_dictionary).expanduser().resolve()
            if args.concept_dictionary
            else (path.parent / "concept_dictionary.csv")
        )
        concept_desc = _load_concept_dictionary(cd_path)
        print(
            f"   cohort: {path}  ({len(df):,} rows, {df.shape[1]} cols, "
            f"outcome={target}, db={database})"
        )
        if concept_desc:
            print(
                f"   concept_dictionary: {cd_path} "
                f"({len(concept_desc)} variables described)"
            )
        return df, cohort_name, database, target, question, concept_desc

    # Fall back to the synthetic cohort.
    from research_agent_mortality_sofa import build_synthetic_cohort  # type: ignore
    df = build_synthetic_cohort(n=args.synthetic_n)
    print(
        f"   cohort: synthetic_smoke_cohort ({len(df):,} rows; "
        "use --cohort-parquet PATH for real data)"
    )
    return (
        df,
        args.cohort_name or "synthetic_smoke_cohort",
        args.database or "synthetic",
        args.target_outcome or "death",
        args.question or "Is admission SOFA-2 score associated with ICU mortality?",
        None,
    )


def _run_one(
    *,
    args,
    temperature: float,
    workdir: Path,
    label: str,
) -> bool:
    print(f"\n=== Smoke run [{label}] — {args.provider}:{args.model} (T={temperature}) ===")
    api_key = args.api_key or (
        os.environ.get("OPENROUTER_API_KEY") if args.provider == "openrouter"
        else os.environ.get("OPENAI_API_KEY")
    )
    client = _make_client(args.provider, args.model, api_key)
    if args.router:
        from easyicu.research_agent import LLMRouter
        client = LLMRouter(default=client)

    base_complete = client.complete

    class _OverrideClient:
        name = client.name

        def complete(self, messages, *, max_tokens=2048, temperature=None):
            return base_complete(messages, max_tokens=max_tokens, temperature=temperature)

    proxy = _OverrideClient.__new__(_OverrideClient)
    def _complete(messages, *, max_tokens=2048, temperature=temperature):
        return base_complete(messages, max_tokens=max_tokens, temperature=temperature)
    proxy.complete = _complete  # type: ignore[attr-defined]
    proxy.name = client.name  # type: ignore[attr-defined]

    from easyicu.research_agent import ResearchAgentPipeline

    cohort, cohort_name, database, target_outcome, question, concept_desc = _resolve_cohort(args)
    pipeline = ResearchAgentPipeline(workdir=workdir, llm=proxy, timeout_seconds=300.0)

    started = time.monotonic()
    try:
        kwargs = dict(
            question=question,
            cohort=cohort,
            cohort_name=cohort_name,
            database=database,
            target_outcome=target_outcome,
        )
        if concept_desc:
            kwargs["concept_descriptions"] = concept_desc
        if args.cross_db:
            kwargs["cross_database_validation"] = list(args.cross_db)
        result = pipeline.run(**kwargs)
    except Exception as exc:
        print(f"❌ pipeline raised: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return False
    elapsed = time.monotonic() - started

    print(f"   run_id: {result.run_id}")
    print(f"   workdir: {result.workdir}")
    print(f"   evidence: {result.evidence_count}, findings: {result.findings_count}")
    print(f"   elapsed: {elapsed:.1f}s")

    # Surface the manuscript paths so the user can open them
    # immediately. Absolute paths because the agent output is the
    # primary deliverable for this smoke test.
    rd = Path(result.workdir).resolve()
    deliverables = {
        "manifest": rd / "manifest.json",
        "report": rd / "results_report.md",
        "manuscript_md": rd / "manuscript_scaffold.md",
        "manuscript_bound_md": rd / "manuscript_scaffold_bound.md",
        "manuscript_tex": rd / "manuscript_scaffold.tex",
        "manuscript_bib": rd / "manuscript_scaffold.bib",
    }
    print()
    print("   --- deliverables ---")
    for label, path in deliverables.items():
        marker = "✅" if path.exists() else "❌"
        print(f"   {marker} {label:<22} {path}")
    return _check_acceptance(result, rd)


def main() -> int:
    _bootstrap_imports()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", default=os.environ.get("EASYICU_SMOKE_PROVIDER", "openrouter"),
        choices=["openrouter", "openai"],
        help="OpenAI-compatible provider (default: openrouter).",
    )
    parser.add_argument(
        "--model",
        default=(
            os.environ.get("EASYICU_SMOKE_MODEL")
            or os.environ.get("EASYICU_HOSTED_DEFAULT_MODEL")
            or "openai/gpt-oss-120b:free"
        ),
        help="Model name. Default reads EASYICU_SMOKE_MODEL or "
             "EASYICU_HOSTED_DEFAULT_MODEL env var, falling back to "
             "openai/gpt-oss-120b:free (free OpenRouter tier).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key. Defaults to OPENROUTER_API_KEY (provider=openrouter) "
             "or OPENAI_API_KEY (provider=openai).",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="LLM temperature for a single run. Overrides --double.",
    )
    parser.add_argument(
        "--double", action="store_true",
        help="Acceptance mode: run twice, at T=0.0 and T=0.3. By "
             "default the script does a single run at T=0.1 to save "
             "tokens on free-tier models.",
    )
    parser.add_argument(
        "--router", action="store_true",
        help="Wrap the resolved client in an LLMRouter (T2.3) — handy for "
             "demonstrating per-agent model selection. Without per-role "
             "overrides this is functionally equivalent to a plain client.",
    )
    parser.add_argument(
        "--workdir-root", default=str((Path.cwd() / "research_output" / "smoke").resolve()),
        help="Where to land run artefacts.",
    )
    parser.add_argument(
        "--cohort-parquet", default=os.environ.get("EASYICU_SMOKE_COHORT"),
        help="Path to a real cohort file (.parquet/.csv/.tsv). When set, "
             "the harness skips the synthetic cohort and runs against "
             "the real EasyICU export package.",
    )
    parser.add_argument(
        "--cohort-name", default=None,
        help="Human-readable cohort name (default: synthetic_smoke_cohort, "
             "or the parquet file's basename when --cohort-parquet is set).",
    )
    parser.add_argument(
        "--database", default=os.environ.get("EASYICU_SMOKE_DATABASE"),
        help="Database identifier passed to ResearchContext (e.g. miiv, "
             "eicu, hirid). Defaults to 'miiv' when --cohort-parquet is "
             "set, 'synthetic' otherwise.",
    )
    parser.add_argument(
        "--target-outcome", default=os.environ.get("EASYICU_SMOKE_OUTCOME"),
        help="Outcome column name. Auto-detected as in_hospital_mortality "
             "/ death / mortality when not specified.",
    )
    parser.add_argument(
        "--question", default=os.environ.get("EASYICU_SMOKE_QUESTION"),
        help="Research question (free text). Auto-composed from the "
             "outcome column + database when not specified.",
    )
    parser.add_argument(
        "--cross-db", nargs="*", default=None,
        help="Cross-database replication targets (zero or more, e.g. "
             "'--cross-db miiv eicu'). Pass an empty list with no values "
             "to disable cross-DB; default is no cross-DB step (saves "
             "tokens — the synthetic cohort can't actually replicate).",
    )
    parser.add_argument(
        "--synthetic-n", type=int, default=600,
        help="Synthetic cohort size when --cohort-parquet is not set.",
    )
    parser.add_argument(
        "--concept-dictionary", default=None,
        help="Path to a concept_dictionary.csv (auto-detected next to "
             "--cohort-parquet when not specified). Variables documented "
             "here are surfaced to the planner via concept_descriptions.",
    )
    args = parser.parse_args()

    workdir_root = Path(args.workdir_root).resolve()
    workdir_root.mkdir(parents=True, exist_ok=True)

    # Default: single run at T=0.1 (cheap, friendly to free-tier models).
    # ``--temperature`` overrides; ``--double`` brings back the
    # acceptance pair (T=0.0, T=0.3).
    if args.temperature is not None:
        temps = [args.temperature]
    elif args.double:
        temps = [0.0, 0.3]
    else:
        temps = [0.1]

    all_ok = True
    for t in temps:
        wd = workdir_root / f"T{t:.1f}".replace(".", "_")
        wd.mkdir(parents=True, exist_ok=True)
        ok = _run_one(
            args=args,
            temperature=t,
            workdir=wd,
            label=f"router temp={t}" if args.router else f"temp={t}",
        )
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("🎉 Smoke test PASSED — pipeline survived a real LLM at "
              + " and ".join(f"T={t}" for t in temps) + ".")
        return 0
    print("❌ Smoke test FAILED — see warnings above. Tighten the relevant "
          "agent prompt in src/easyicu/research_agent/agents.py and rerun.")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

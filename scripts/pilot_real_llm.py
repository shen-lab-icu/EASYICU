"""Phase A Step 1 — Real-LLM × real-data pilot run.

See ``AGENT_PLAN.md`` for the surrounding plan. Purpose: surface
failure modes that the 582 mock-LLM tests do not exercise. The output
goes under ``pilot_runs/`` (gitignored).

Usage (from the EASYICU repo root)::

    python scripts/pilot_real_llm.py

Credentials are read from ``.env.local`` (also gitignored). The
default model is whichever free OpenRouter tier is configured in
``EASYICU_HOSTED_DEFAULT_MODEL``; falls back to a short list of free
models if the first is rate-limited.

After the run finishes, inspect three artefacts (the triage targets
described in AGENT_PLAN.md Step 1):

1. ``manuscript_scaffold_bound.md`` — readability + value-trace footnotes
2. ``manifest.json`` findings — validator signal-to-noise
3. ``evidence/numeric_claims.json`` — numeric capture completeness

Write the resulting triage as ``pilot_runs/<run_id>/TRIAGE.md`` so the
next agent / human picks up what to fix in Step 2.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env.local"
MIIV_PREPARED_EXPORT = Path("/Users/haibo/Documents/GitHub/其他文件/miiv_20260420")
PILOT_OUT = REPO_ROOT / "pilot_runs"

FREE_MODEL_FALLBACK = [
    "z-ai/glm-4.5-air:free",
    "deepseek/deepseek-chat-v3.1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-235b-a22b:free",
]

# Raw-data roots for cross-DB pilots. The user's databases live on a
# macFUSE-mounted disk. Per-DB paths are absolute so the orchestrator
# (`pilot_cross_db.py`) can hand each per-DB pilot a single string.
DB_RAW_ROOTS = {
    "miiv":  Path("/Users/haibo/.mounty/新加卷/databases/mimic-iv-3.1"),
    "mimic": Path("/Users/haibo/.mounty/新加卷/databases/mimiciii"),
    "eicu":  Path("/Users/haibo/.mounty/新加卷/databases/eicu"),
    "hirid": Path("/Users/haibo/.mounty/新加卷/databases/hirid-a-high-time-resolution-icu-dataset-1.1.1"),
    "aumc":  Path("/Users/haibo/.mounty/新加卷/databases/aumc"),
    "sic":   Path("/Users/haibo/.mounty/新加卷/databases/sic"),
}

# Per-DB primary stay/admission id column. Used to detect which column
# carries the cohort key after a `load_concepts` call.
DB_ID_COLS = {
    "miiv": "stay_id",
    "mimic": "icustay_id",
    "eicu": "patientunitstayid",
    "hirid": "patientid",
    "aumc": "admissionid",
    "sic": "CaseID",
}


def _load_env_local() -> None:
    """Parse ``.env.local`` (gitignored) into ``os.environ``.

    Existing env vars win — ``setdefault`` is intentional so an
    explicit ``OPENROUTER_API_KEY=... python scripts/pilot_real_llm.py``
    still overrides the file.
    """
    if not ENV_PATH.exists():
        print(f"[warn] {ENV_PATH} not found; relying on existing env vars",
              file=sys.stderr)
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _load_cohort_miiv_prepared(n: int = 500) -> pd.DataFrame:
    """Build a cohort from the miiv_20260420 prepared parquet export.

    This is the fast path used by the original pilots (1/2/3): just join
    three pre-extracted parquet files. Only works for miiv when the
    prepared export is present locally.
    """
    demo = pd.read_parquet(
        MIIV_PREPARED_EXPORT / "demographics_adm_age_bmi_height_etc6.parquet"
    )
    outc = pd.read_parquet(
        MIIV_PREPARED_EXPORT / "outcome_death_los_hosp_los_icu.parquet"
    )
    sofa = pd.read_parquet(
        MIIV_PREPARED_EXPORT / "sofa2_score_sofa2_sofa2_cardio_sofa2_cns_sofa2_coag_etc7.parquet"
    )

    sofa = sofa[sofa["charttime"] >= 0].sort_values(["stay_id", "charttime"])
    sofa_adm = (
        sofa.groupby("stay_id")
        .first()[["sofa2"]]
        .reset_index()
        .rename(columns={"sofa2": "sofa2_admission"})
    )

    df = (
        demo.merge(outc, on="stay_id", how="inner")
        .merge(sofa_adm, on="stay_id", how="left")
    )
    df["death"] = df["death"].fillna(0).astype(int)
    df = df.dropna(subset=["sofa2_admission", "age"]).reset_index(drop=True)

    if n is not None and len(df) > n:
        df = df.head(n).reset_index(drop=True)
    return df


def _load_cohort_via_api(
    database: str,
    data_path: Path,
    n: int = 500,
) -> pd.DataFrame:
    """Build a cohort from any of the 6 supported databases via easyicu API.

    Uses ``load_concepts`` for cohort metadata (age / sex / death / los_icu)
    and ``load_sofa2`` for the SOFA-2 time series, then joins on the
    database's primary stay id. Admission SOFA-2 is the first observation
    at or after ICU admission (``charttime >= 0``).

    This is the cross-DB code path (Phase A Step 3 in AGENT_PLAN.md).
    Note: SOFA-2 extraction on full cohorts is heavyweight on hirid
    (observations sweep) — keep cohort sizes modest during pilot runs.
    """
    from easyicu.api import load_concepts, load_sofa2

    id_col = DB_ID_COLS.get(database, "stay_id")

    # Try the full demographics + outcome bundle first; some databases
    # (notably mimic-iii) have age/sex sourced from a patients table that
    # the loader cannot auto-merge with the primary stay id, so fall
    # back progressively if the rich set fails.
    cohort = None
    for concept_set in (
        ["age", "sex", "death", "los_icu"],
        ["death", "los_icu"],
        ["death"],
    ):
        try:
            cohort = load_concepts(
                concept_set,
                database=database,
                data_path=data_path,
            )
            print(f"  [_load_cohort_via_api] {database}: loaded {concept_set} ({len(cohort)} rows)")
            break
        except Exception as exc:
            print(f"  [_load_cohort_via_api] {database}: concept_set={concept_set} "
                  f"failed ({type(exc).__name__}), falling back")
    if cohort is None:
        raise RuntimeError(
            f"Failed to build cohort on {database} — even 'death' alone "
            f"did not load. Check that the database has been converted."
        )

    if id_col not in cohort.columns:
        for cand in DB_ID_COLS.values():
            if cand in cohort.columns:
                id_col = cand
                break
        else:
            raise RuntimeError(
                f"Cannot determine cohort id column on {database}. "
                f"Got columns: {list(cohort.columns)[:10]}"
            )

    cohort = cohort.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    if "death" in cohort.columns:
        cohort["death"] = cohort["death"].fillna(0).astype(int)

    sofa = load_sofa2(database=database, data_path=data_path)
    if "charttime" in sofa.columns:
        sofa_first = (
            sofa[sofa["charttime"] >= 0]
            .sort_values([id_col, "charttime"])
            .groupby(id_col)
            .first()[["sofa2"]]
            .reset_index()
            .rename(columns={"sofa2": "sofa2_admission"})
        )
    else:
        # Some databases return only one row per stay already (no time series).
        sofa_first = sofa[[id_col, "sofa2"]].rename(
            columns={"sofa2": "sofa2_admission"}
        )

    df = cohort.merge(sofa_first, on=id_col, how="left")
    df = df.dropna(subset=["sofa2_admission"]).reset_index(drop=True)
    if "age" in df.columns:
        df = df.dropna(subset=["age"]).reset_index(drop=True)

    if n is not None and len(df) > n:
        df = df.head(n).reset_index(drop=True)
    return df


def _load_cohort(
    n: int = 500,
    *,
    database: str = "miiv",
    data_path: Path | None = None,
) -> pd.DataFrame:
    """Dispatch to the prepared-export path (miiv) or the API path."""
    if database == "miiv" and data_path is None and MIIV_PREPARED_EXPORT.exists():
        return _load_cohort_miiv_prepared(n=n)
    if data_path is None:
        data_path = DB_RAW_ROOTS.get(database)
    if data_path is None or not Path(data_path).exists():
        raise SystemExit(
            f"No data_path resolvable for database '{database}'. "
            f"Pass --data-path or mount the volume."
        )
    return _load_cohort_via_api(database, Path(data_path), n=n)


def _build_llm(model: str | None = None):
    from easyicu.research_agent import OpenAIClient

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY missing. Populate .env.local "
            "or `export OPENROUTER_API_KEY=...`."
        )
    base_url = os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    chosen = model or os.environ.get(
        "EASYICU_HOSTED_DEFAULT_MODEL", FREE_MODEL_FALLBACK[0]
    )
    return OpenAIClient(
        model=chosen,
        api_key=api_key,
        base_url=base_url,
        request_timeout=180.0,
    ), chosen


def _write_triage_template(run_dir: Path, *, run_id: str, model: str,
                           cohort_size: int, result_obj) -> Path:
    """Drop an empty TRIAGE.md template alongside the run output.

    Phase A Step 1's definition of done says a triage is mandatory.
    The template forces the reviewer (human or AI) to fill in
    findings rather than leave the pilot under-inspected.
    """
    path = run_dir / "TRIAGE.md"
    template = f"""# Pilot triage — {run_id}

> Generated by `scripts/pilot_real_llm.py`. **Fill this in before
> proceeding to AGENT_PLAN.md Step 2.**

## Run summary

- Run ID: `{run_id}`
- Model: `{model}`
- Cohort size: {cohort_size} rows
- Workdir: `{run_dir}`
- Evidence count: {getattr(result_obj, "evidence_count", "?")}
- Findings count: {getattr(result_obj, "findings_count", "?")}
- Manuscript: `{getattr(result_obj, "manuscript_path", "?")}`

## 1. Manuscript readability check

Read `manuscript_scaffold_bound.md`. Answer:

- [ ] Does the prose read like a coherent results paragraph?
- [ ] Are value-trace footnotes attached to ORs / p-values / counts?
- [ ] Any sentence stripped because evidence binding failed?
- [ ] Any `<!-- UNTRACED:... -->` markers? List them:

  -

## 2. Validator signal-to-noise

Open `manifest.json`, look at `findings`:

- Total warnings: ?
- Total errors: ?
- Findings that look like real problems (cite validator name): ?
- Findings that look like noise / false positives: ?

## 3. Numeric capture completeness

Open `evidence/numeric_claims.json`:

- Total claims registered: ?
- Are key result quantities (primary OR, CI, p, AUC) captured? ?
- Anything obviously missing? ?

## 4. Issues found (prioritised)

### Critical (blocks experiment quality)

-

### Engineering (nice to fix)

-

### Cosmetic (defer)

-

## 5. Next-step recommendation

What changes (if any) feed back into AGENT_PLAN.md Step 2?

-
"""
    path.write_text(template, encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=None,
        help=(
            "Model id. In legacy cohort-pilot mode this is an OpenRouter model. "
            "In --case mode it is forwarded to the benchmark runner."
        ),
    )
    parser.add_argument(
        "--cohort-size", type=int, default=500,
        help="Max rows to keep in the pilot cohort (default 500).",
    )
    parser.add_argument(
        "--question",
        default="Is admission SOFA-2 score associated with ICU mortality?",
    )
    parser.add_argument(
        "--enforcement", choices=["soft", "strict"], default="soft",
        help="Evidence enforcement mode. SOFT during initial pilot to "
             "make untraced numerics visible without aborting the run.",
    )
    parser.add_argument(
        "--database", default="miiv",
        choices=sorted(DB_RAW_ROOTS.keys()),
        help="Source database. Default miiv (uses prepared parquet "
             "export when available). Other databases use the easyicu "
             "API on raw data.",
    )
    parser.add_argument(
        "--data-path", default=None,
        help="Override the raw-data root path. Defaults to the per-DB "
             "entry in DB_RAW_ROOTS.",
    )
    parser.add_argument(
        "--case",
        default=None,
        help=(
            "Optional case directory under benchmark/cases. When set, this "
            "script delegates to tools/run_research_agent_bench.py after "
            "case bootstrap instead of running the legacy single-cohort pilot."
        ),
    )
    parser.add_argument(
        "--bench-kind",
        choices=["rule", "analysis"],
        default="rule",
        help="Benchmark fixture family forwarded in --case mode.",
    )
    parser.add_argument(
        "--bench-items",
        nargs="+",
        default=None,
        help="Benchmark item keys forwarded in --case mode.",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "openrouter", "openai"],
        default="mock",
        help="LLM provider forwarded in --case mode.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["aware"],
        help="Benchmark arm(s) forwarded in --case mode.",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output root forwarded in --case mode. Defaults to pilot_runs/bench_case.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="Per-request timeout forwarded in --case mode.",
    )
    parser.add_argument(
        "--max-total-steps",
        type=int,
        default=None,
        help="Optional ResearchAgentPipeline max_total_steps forwarded in --case mode.",
    )
    parser.add_argument(
        "--submission-profile",
        action="store_true",
        help="Forward the paper-facing submission profile to the benchmark runner.",
    )
    parser.add_argument(
        "--profile",
        default="npj_dm/20260527",
        help="Versioned submission profile ref forwarded in --case mode.",
    )
    return parser.parse_args()


def _run_case_benchmark_delegate(args: argparse.Namespace) -> int:
    """Delegate case-aware pilots to the benchmark runner.

    ``scripts/pilot_real_llm.py`` remains the canonical pilot entrypoint; this
    branch prevents Case B from growing a second, parallel pilot path while
    preserving the legacy prepared-cohort pilot above.
    """

    bench_items = list(args.bench_items or [])
    if not bench_items:
        raise SystemExit("--bench-items is required when --case is supplied")
    out_root = Path(args.out_root) if args.out_root else PILOT_OUT / "bench_case"
    model = args.model or os.environ.get("EASYICU_PILOT_MODEL", "gpt-5.4")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_research_agent_bench.py"),
        "--case",
        str(args.case),
        "--bench-kind",
        str(args.bench_kind),
        "--items",
        *bench_items,
        "--arms",
        *list(args.arms or ["aware"]),
        "--provider",
        str(args.provider),
        "--model",
        str(model),
        "--out-root",
        str(out_root.resolve()),
        "--request-timeout",
        str(float(args.request_timeout)),
    ]
    if args.max_total_steps is not None:
        cmd.extend(["--max-total-steps", str(int(args.max_total_steps))])
    if args.submission_profile:
        cmd.extend(["--submission-profile", "--profile", str(args.profile)])
    print("[delegate] " + " ".join(shlex.quote(part) for part in cmd))
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def main() -> int:
    args = _parse_args()

    _load_env_local()
    if args.case:
        return _run_case_benchmark_delegate(args)

    print(f"[env] OPENROUTER_API_KEY set: {'OPENROUTER_API_KEY' in os.environ}")
    print(f"[env] Model: {args.model or os.environ.get('EASYICU_HOSTED_DEFAULT_MODEL', FREE_MODEL_FALLBACK[0])}")

    data_path = Path(args.data_path) if args.data_path else None
    src_label = (
        "miiv_20260420 prepared parquet"
        if args.database == "miiv" and data_path is None and MIIV_PREPARED_EXPORT.exists()
        else f"easyicu API on {args.database} ({data_path or DB_RAW_ROOTS.get(args.database)})"
    )
    print(f"[cohort] Building via {src_label} ...")
    cohort = _load_cohort(
        n=args.cohort_size,
        database=args.database,
        data_path=data_path,
    )
    print(f"[cohort] {len(cohort)} stays | "
          f"death rate {cohort['death'].mean():.1%} | "
          f"SOFA-2 admission mean {cohort['sofa2_admission'].mean():.2f}")

    llm, chosen_model = _build_llm(args.model)

    PILOT_OUT.mkdir(parents=True, exist_ok=True)

    from easyicu.research_agent import ResearchAgentPipeline

    pipeline = ResearchAgentPipeline(
        workdir=PILOT_OUT,
        llm=llm,
        timeout_seconds=600.0,
        evidence_enforcement_mode=args.enforcement,
        enable_cache=False,
        max_code_repair_attempts=2,
    )

    started = datetime.now(timezone.utc)
    print(f"[run] Starting pilot at {started.isoformat()}")
    try:
        result = pipeline.run(
            question=args.question,
            cohort=cohort,
            cohort_name=f"{args.database}_pilot_{len(cohort)}",
            database=args.database,
            target_outcome="death",
        )
    except Exception:
        print("[run] Pipeline raised — see traceback. Pilot driver still "
              "exits 1 so CI / wrapping scripts can detect failure.",
              file=sys.stderr)
        traceback.print_exc()
        return 1
    finished = datetime.now(timezone.utc)
    elapsed = (finished - started).total_seconds()

    print(f"[run] Finished in {elapsed:.0f}s | run_id={result.run_id}")
    print(f"[run] workdir   : {result.workdir}")
    print(f"[run] manifest  : {result.manifest_path}")
    print(f"[run] manuscript: {result.manuscript_path}")
    print(f"[run] evidence  : {result.evidence_count}")
    print(f"[run] findings  : {result.findings_count}")

    run_dir = Path(result.workdir)
    triage_path = _write_triage_template(
        run_dir,
        run_id=result.run_id,
        model=chosen_model,
        cohort_size=len(cohort),
        result_obj=result,
    )
    print(f"[run] TRIAGE template at {triage_path}")
    # Also persist a small machine-readable summary so other agents can
    # pick up the run programmatically.
    summary_path = run_dir / "pilot_summary.json"
    summary_path.write_text(
        json.dumps({
            "run_id": result.run_id,
            "model": chosen_model,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "elapsed_seconds": elapsed,
            "cohort_size": int(len(cohort)),
            "evidence_count": int(result.evidence_count),
            "findings_count": int(result.findings_count),
            "manuscript_path": str(result.manuscript_path),
            "manifest_path": str(result.manifest_path),
            "enforcement_mode": args.enforcement,
            "database": args.database,
            "question": args.question,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[run] pilot_summary.json at {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

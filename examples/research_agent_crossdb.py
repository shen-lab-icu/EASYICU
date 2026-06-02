"""Cross-database replication for the research agent.

Run the same ``sofa_mortality`` ClinicalSkill across MIMIC-IV
(``miiv``), eICU (``eicu``) and HiRID (``hirid``) and emit a single
``crossdb_summary.md``. The script's central claim — that a single
``ResearchContext`` schema works across heterogeneous ICU databases —
is demonstrated end to end.

Two execution modes:

* **Real mode** (default when ``EASYICU_DATA_PATH`` is set or
  ``--data-root`` is passed): uses :func:`easyicu.filter_patients`
  with the standard inclusion criteria (first ICU admission, age ≥
  18, LoS ≥ 6 h) plus a thin per-stay aggregator
  (:func:`_assemble_cohort`) that produces a one-row-per-stay
  DataFrame with the columns the ``sofa_mortality`` skill expects.

* **Dry-run / sandbox mode** (``--dry-run`` or no data root): uses a
  per-database synthetic cohort with database-specific
  characteristics (different N, SOFA distribution, sex ratio) so the
  comparison harness can be validated without local ICU data. This
  is not a substitute for the real run — it just exercises the
  scaffolding.

Usage::

    # Real (requires the database paths under EASYICU_DATA_PATH)
    EASYICU_DATA_PATH=/data/icu \\
        python examples/research_agent_crossdb.py

    # Subset of databases (e.g. only miiv)
    python examples/research_agent_crossdb.py --databases miiv

    # Quick smoke run on a synthetic stand-in
    python examples/research_agent_crossdb.py --dry-run

The script also runs the same database **twice with the same seed**
to confirm the EvidenceStore's sha256 hashes are stable — a
prerequisite for cross-database reproducibility claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _bootstrap_imports():
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


# ---------------------------------------------------------------------------
# Cohort assembly — real mode
# ---------------------------------------------------------------------------


_REAL_REQUIRED_COLUMNS = ["stay_id", "age", "sex", "sofa2", "death"]
_REAL_OPTIONAL_COLUMNS = [
    "lact", "creat", "map", "vaso", "los_icu",
    "sofa2_resp", "sofa2_coag", "sofa2_liver", "sofa2_cardio", "sofa2_cns", "sofa2_renal",
    "hr", "sbp", "dbp", "temp", "spo2", "resp",
]


def _per_stay_agg(df, value_col: str, id_col: str, agg: str):
    """Aggregate a long-form (id, time, value) frame to one row per stay."""
    import pandas as pd

    if df is None or df.empty or value_col not in df.columns:
        return None
    sub = df[[id_col, value_col]].dropna()
    if sub.empty:
        return None
    if agg == "max":
        return sub.groupby(id_col, sort=False)[value_col].max()
    if agg == "median":
        return sub.groupby(id_col, sort=False)[value_col].median()
    if agg == "any":
        return (sub.groupby(id_col, sort=False)[value_col]
                  .apply(lambda s: int(bool((s.fillna(0) > 0).any()))))
    raise ValueError(f"unknown agg {agg!r}")


def _detect_id_col(df) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in ("stay_id", "icustay_id", "patient_id", "subject_id"):
        if c in df.columns:
            return c
    return None


def _assemble_cohort(*, database: str, data_path: str, max_patients: Optional[int] = None,
                     los_min_hours: float = 6.0, batch_size: Optional[int] = 20000,
                     verbose: bool = False):
    """Build a per-stay cohort matching the ``sofa_mortality`` skill schema.

    Strategy:
    1. ``filter_patients(... return_dataframe=True)`` for demographics +
       inclusion (first ICU admit, age ≥ 18, configurable minimum LoS).
    2. ``extract_database(modules=[...])`` for the small set of
       time-series concepts the skill cares about.
    3. Aggregate each concept to one row per stay using ICU-aware
       defaults (max for SOFA, median for labs / vitals, any for
       vasopressor exposure).
    4. Inner-join on the id column.
    """
    import pandas as pd
    from easyicu import filter_patients, extract_database  # type: ignore

    if verbose:
        print(f"[{database}] filter_patients → cohort frame")
    cohort_df = filter_patients(
        database=database, data_path=data_path,
        age_min=18.0, first_icu_stay=True, los_min=los_min_hours,
        return_dataframe=True, verbose=verbose,
    )
    if cohort_df is None or len(cohort_df) == 0:
        raise RuntimeError(f"no patients matched inclusion in {database}")
    id_col = _detect_id_col(cohort_df)
    if id_col is None:
        raise RuntimeError(f"could not detect id column on {database} demographics")

    # ``filter_patients`` already provides age, sex, los_icu, death.
    keep = [id_col]
    # tolerate alternative column names from filter_patients
    rename: Dict[str, str] = {}
    for src, tgt in (("age", "age"), ("gender", "sex"), ("sex", "sex"),
                     ("los_icu", "los_icu"), ("los_icu_h", "los_icu"),
                     ("los", "los_icu"), ("death", "death"),
                     ("death_icu", "death"), ("mortality", "death")):
        if src in cohort_df.columns and tgt not in rename.values():
            rename[src] = tgt
    if rename:
        cohort_df = cohort_df.rename(columns=rename)
    if "death" not in cohort_df.columns and "survived" in cohort_df.columns:
        survived_num = pd.to_numeric(cohort_df["survived"], errors="coerce")
        cohort_df["death"] = 1 - survived_num.astype("Int64")
    for c in ("age", "sex", "los_icu", "death"):
        if c in cohort_df.columns and c not in keep:
            keep.append(c)
    if max_patients:
        cohort_df = cohort_df.head(int(max_patients))
    cohort_df = cohort_df[keep].copy()

    # Pull a small set of relevant concepts via extract_database.
    if verbose:
        print(f"[{database}] extract_database → sofa2_score / chemistry / vitals / vasopressors")
    pid_filter = {id_col: cohort_df[id_col].tolist()}
    ext = extract_database(
        database=database, data_path=data_path,
        modules=["sofa2_score", "chemistry", "vitals", "vasopressors", "outcome"],
        patient_ids=pid_filter,
        batch_size=batch_size,
        verbose=verbose,
    )

    def _concept(module: str, name: str):
        try:
            return ext["modules"][module]["concepts"][name]
        except Exception:
            return None

    # Best-effort per-stay summaries; missing concepts are tolerated and the
    # downstream skill's ``validate_against`` will warn when relevant.
    aggregated: Dict[str, Any] = {}
    for name in ("sofa2", "sofa2_resp", "sofa2_coag", "sofa2_liver", "sofa2_cardio", "sofa2_cns", "sofa2_renal"):
        df = _concept("sofa2_score", name)
        if df is None:
            continue
        s = _per_stay_agg(df, name, id_col, "max")
        if s is not None:
            aggregated[name] = s
    for name, agg in (("lact", "median"), ("creat", "median"), ("bili", "median")):
        df = _concept("chemistry", name)
        if df is not None:
            s = _per_stay_agg(df, name, id_col, agg)
            if s is not None:
                aggregated[name] = s
    for name in ("hr", "map", "sbp", "dbp", "temp", "spo2", "resp"):
        df = _concept("vitals", name)
        if df is None:
            continue
        s = _per_stay_agg(df, name, id_col, "median")
        if s is not None:
            aggregated[name] = s
    vaso_df = _concept("vasopressors", "vaso_ind")
    if vaso_df is None:
        vaso_df = _concept("vasopressors", "norepi")
    if vaso_df is not None:
        # try a few likely value columns
        for value_col in ("vaso_ind", "norepi", "value", "rate"):
            if value_col in vaso_df.columns:
                s = _per_stay_agg(vaso_df, value_col, id_col, "any")
                if s is not None:
                    aggregated["vaso"] = s
                    break

    # Stitch onto cohort frame.
    out = cohort_df.set_index(id_col)
    for col, series in aggregated.items():
        out[col] = series
    out = out.reset_index().rename(columns={id_col: "stay_id"})

    # Coerce expected numeric types so the agent's auditors are happy.
    for c in ("sofa2",):
        if c in out.columns:
            out[c] = out[c].astype("Int64")
    return out


# ---------------------------------------------------------------------------
# Cohort assembly — dry-run / synthetic mode
# ---------------------------------------------------------------------------


def _build_synth_cohort(database: str, n: int, seed: int):
    """Per-database synthetic cohort.

    Each database gets a slightly different distribution so the
    cross-DB diff has signal:

    * ``miiv`` — large cohort, moderate SOFA, balanced sex.
    * ``eicu`` — medium, lower SOFA mean, more females.
    * ``hirid`` — small, higher SOFA mean (sicker referral centre).
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed + sum(ord(c) for c in database))
    profile = {
        "miiv": dict(age=(65, 15), sofa_lo=1, sofa_hi=14, miss=0.08, female=0.45),
        "eicu": dict(age=(63, 14), sofa_lo=0, sofa_hi=12, miss=0.10, female=0.52),
        "hirid": dict(age=(67, 13), sofa_lo=2, sofa_hi=16, miss=0.12, female=0.40),
    }
    cfg = profile.get(database, profile["miiv"])
    age = rng.normal(cfg["age"][0], cfg["age"][1], n).clip(18, 95)
    sex = rng.choice(["F", "M"], size=n, p=[cfg["female"], 1 - cfg["female"]])
    base = rng.integers(cfg["sofa_lo"], cfg["sofa_hi"], size=n, endpoint=False)
    miss = rng.random(n) < cfg["miss"]
    truly_low = rng.random(n) < 0.04
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
        "age": age, "sex": sex,
        "sofa2": sofa2, "lact": lact, "creat": creat,
        "map": map_v, "vaso": vaso, "los_icu": los, "death": death,
    })


# ---------------------------------------------------------------------------
# Per-database run + run-level summary
# ---------------------------------------------------------------------------


def _run_database(
    *,
    database: str,
    cohort,
    workdir: Path,
    llm,
    timeout_seconds: float,
    question: str,
    skill_key: Optional[str],
    los_min_hours: float,
    stop_after_analysis: bool,
    verbose: bool = True,
) -> Dict[str, Any]:
    from easyicu.research_agent import ResearchAgentPipeline  # type: ignore
    workdir.mkdir(parents=True, exist_ok=True)
    pipeline = ResearchAgentPipeline(
        workdir=workdir,
        llm=llm,
        timeout_seconds=timeout_seconds,
    )
    if verbose:
        skill_label = skill_key if skill_key else "free-form planner"
        print(f"[{database}] running ResearchAgentPipeline.run({skill_label}) …")
    started = time.monotonic()
    result = pipeline.run(
        question=question,
        cohort=cohort,
        cohort_name=f"{database}_crossdb_cohort",
        database=database,
        target_outcome="death",
        skill=skill_key,
        inclusion_criteria=[
            "First ICU admission",
            "Age >= 18 years",
            f"ICU length of stay >= {los_min_hours:g} hours",
        ],
        exclusion_criteria=[
            f"Discharged within first {los_min_hours:g} hours",
        ],
        stop_after_analysis=stop_after_analysis,
    )
    elapsed = time.monotonic() - started
    return _summarise_run(database=database, cohort=cohort, result=result, elapsed=elapsed)


def _summarise_run(*, database, cohort, result, elapsed) -> Dict[str, Any]:
    run_dir = Path(result.workdir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    # Cohort-level
    n_stays = int(len(cohort))
    sofa2_zero_rate = float((cohort["sofa2"] == 0).mean()) if "sofa2" in cohort.columns else None
    death_rate = float(cohort["death"].mean()) if "death" in cohort.columns else None
    miss_profile = {
        col: float(cohort[col].isna().mean())
        for col in cohort.columns if col != "stay_id"
    }

    # SOFA-zero anomaly + stratum table
    sofa_strata_csv = next(run_dir.rglob("sofa_strata.csv"), None)
    sofa_strata_rows: List[Dict[str, Any]] = []
    sofa_zero_anomaly = False
    if sofa_strata_csv is not None and sofa_strata_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(sofa_strata_csv)
            cols = list(df.columns)
            score_col = next((c for c in cols if c not in ("n", "outcome_rate")), None)
            if score_col:
                for _, row in df.iterrows():
                    sofa_strata_rows.append({
                        score_col: int(row[score_col]),
                        "n": int(row["n"]),
                        "outcome_rate": float(row["outcome_rate"]),
                    })
                try:
                    r0 = next(r["outcome_rate"] for r in sofa_strata_rows if r[score_col] == 0)
                    r1 = next(r["outcome_rate"] for r in sofa_strata_rows if r[score_col] == 1)
                    sofa_zero_anomaly = r0 > r1
                except StopIteration:
                    pass
        except Exception:
            pass

    # Primary OR
    primary_or = None
    primary_or_ci = None
    primary_backend = None
    primary_summary_path = next(run_dir.rglob("step_summary.json"), None)
    # find the primary_association step's summary specifically
    for ssj in run_dir.rglob("step_summary.json"):
        try:
            data = json.loads(ssj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("method") == "logistic_regression" and data.get("primary_or") is not None:
            primary_or = data["primary_or"]
            primary_or_ci = data.get("primary_or_ci")
            primary_backend = data.get("backend")
            break

    # Hash-stability sentinel: pick the table_one.csv evidence sha for replay test.
    table_one_sha = None
    for rec in manifest.get("evidence", []):
        if (rec.get("kind") == "table"
                and Path(rec.get("relative_path", "")).name.endswith("table_one.csv")):
            table_one_sha = rec.get("sha256")
            break

    return {
        "database": database,
        "run_id": result.run_id,
        "workdir": str(run_dir),
        "elapsed_seconds": round(float(elapsed), 2),
        "evidence_count": int(result.evidence_count),
        "findings_count": int(result.findings_count),
        "n_findings_warning": sum(1 for f in manifest.get("findings", [])
                                  if f.get("severity") == "warning"),
        "n_findings_error": sum(1 for f in manifest.get("findings", [])
                                if f.get("severity") == "error"),
        "cohort": {
            "n_stays": n_stays,
            "death_rate": death_rate,
            "sofa2_zero_rate": sofa2_zero_rate,
            "missingness": miss_profile,
        },
        "sofa_zero_anomaly": sofa_zero_anomaly,
        "sofa_strata": sofa_strata_rows,
        "primary_association": {
            "odds_ratio": primary_or,
            "ci": primary_or_ci,
            "backend": primary_backend,
        },
        "table_one_sha256": table_one_sha,
    }


# ---------------------------------------------------------------------------
# Hash-stability check
# ---------------------------------------------------------------------------


def _hash_stability_check(
    *,
    database: str,
    cohort,
    workdir: Path,
    llm,
    timeout_seconds: float,
    question: str,
    skill_key: Optional[str],
    los_min_hours: float,
    stop_after_analysis: bool,
) -> Dict[str, Any]:
    """Re-run the same database with the same seed and confirm hashes."""
    a = _run_database(
        database=database, cohort=cohort, workdir=workdir / "replay_a", llm=llm,
        timeout_seconds=timeout_seconds,
        question=question, skill_key=skill_key, los_min_hours=los_min_hours,
        stop_after_analysis=stop_after_analysis, verbose=False,
    )
    b = _run_database(
        database=database, cohort=cohort, workdir=workdir / "replay_b", llm=llm,
        timeout_seconds=timeout_seconds,
        question=question, skill_key=skill_key, los_min_hours=los_min_hours,
        stop_after_analysis=stop_after_analysis, verbose=False,
    )
    stable = a["table_one_sha256"] is not None and a["table_one_sha256"] == b["table_one_sha256"]
    return {
        "database": database,
        "stable": bool(stable),
        "table_one_sha256_a": a["table_one_sha256"],
        "table_one_sha256_b": b["table_one_sha256"],
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_or(or_block: Dict[str, Any]) -> str:
    if or_block.get("odds_ratio") is None:
        return "—"
    or_ = or_block["odds_ratio"]
    ci = or_block.get("ci")
    if isinstance(ci, list) and len(ci) == 2:
        return f"{or_:.2f} (95% CI {ci[0]:.2f}–{ci[1]:.2f})"
    return f"{or_:.2f}"


def _render_markdown(*, summaries: List[Dict[str, Any]], stability: List[Dict[str, Any]],
                     dry_run: bool, los_min_hours: float, skill_label: str) -> str:
    lines: List[str] = [
        "# Cross-database replication summary",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()} ({'dry-run synthetic' if dry_run else 'real EasyICU extraction'} mode)._",
        "",
        "Inclusion criteria across all databases: first ICU admission, age ≥ 18 years, "
        f"ICU length-of-stay ≥ {los_min_hours:g} hours. Skill/planner: `{skill_label}`. Same "
        "`ResearchContext` schema, same `EvidenceStore`, same validators.",
        "",
        "## Cohort sizes & illness severity",
        "",
        "| Database | Stays | Mortality | SOFA-2==0 fraction | Anomaly flagged | Primary OR (sofa2) | Run time |",
        "|---|---:|---:|---:|:-:|---|---:|",
    ]
    for s in summaries:
        coh = s["cohort"]
        lines.append(
            "| `{db}` | {n} | {dr} | {sz} | {flag} | {or_} | {t}s |".format(
                db=s["database"], n=coh["n_stays"],
                dr=("?" if coh["death_rate"] is None else f"{coh['death_rate']:.1%}"),
                sz=("?" if coh["sofa2_zero_rate"] is None else f"{coh['sofa2_zero_rate']:.1%}"),
                flag=("⚠️ yes" if s["sofa_zero_anomaly"] else "no"),
                or_=_fmt_or(s["primary_association"]),
                t=s["elapsed_seconds"],
            )
        )
    lines.append("")

    # Per-DB missingness table
    lines.append("## Missingness profile (fraction missing)")
    lines.append("")
    cols = sorted({c for s in summaries for c in s["cohort"]["missingness"]})
    header = "| Database | " + " | ".join(f"`{c}`" for c in cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for s in summaries:
        ms = s["cohort"]["missingness"]
        row = "| `{db}` | ".format(db=s["database"]) + " | ".join(
            f"{ms.get(c, 0.0):.1%}" if c in ms else "—" for c in cols
        ) + " |"
        lines.append(row)
    lines.append("")

    # Validator findings
    lines.append("## Validator findings")
    lines.append("")
    lines.append("| Database | warnings | errors |")
    lines.append("|---|---:|---:|")
    for s in summaries:
        lines.append(f"| `{s['database']}` | {s['n_findings_warning']} | {s['n_findings_error']} |")
    lines.append("")

    # Hash stability
    lines.append("## Hash-stability sentinel (replay test)")
    lines.append("")
    lines.append("Re-running the same database with the same cohort hashes should yield "
                 "the same `table_one.csv` sha256.")
    lines.append("")
    lines.append("| Database | Stable | sha256 (run A) | sha256 (run B) |")
    lines.append("|---|:-:|---|---|")
    for st in stability:
        a = st.get("table_one_sha256_a") or ""
        b = st.get("table_one_sha256_b") or ""
        lines.append(
            f"| `{st['database']}` | {'✅' if st['stable'] else '❌'} | "
            f"`{a[:12]}…` | `{b[:12]}…` |"
        )
    lines.append("")

    lines.append("## Provenance")
    lines.append("")
    for s in summaries:
        lines.append(f"- `{s['database']}` → `{s['workdir']}` (run_id: `{s['run_id']}`)")
    lines.append("")
    return "\n".join(lines)


def _paper_figure_handoff(*, summaries: List[Dict[str, Any]], out_root: Path) -> Dict[str, Any]:
    """Create the default paper-display handoff after analysis.

    This is deliberately not a Nature-figure renderer. It freezes the
    post-analysis boundary that manuscript work should start from:
    evidence QA, model/validator risks, and the tables/figures that are
    suitable inputs for a publication redraw.
    """
    handoff: Dict[str, Any] = {
        "schema_version": "easyicu.paper_figure_handoff/1",
        "status": "ready_for_figure_qa",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
        "next_step": (
            "Use the registered EasyICU evidence tables/figures as source data "
            "for a Nature-ready figure workflow; do not use unregistered or "
            "recycled artefacts."
        ),
    }

    for s in summaries:
        run_dir = Path(s["workdir"])
        manifest_path = run_dir / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        figures = []
        tables = []
        statistics = []
        for rec in manifest.get("evidence", []) or []:
            rel = rec.get("relative_path")
            if not rel:
                continue
            item = {
                "evidence_id": rec.get("evidence_id"),
                "description": rec.get("description"),
                "path": str(run_dir / rel),
                "sha256": rec.get("sha256"),
            }
            if rec.get("kind") == "figure":
                figures.append(item)
            elif rec.get("kind") == "table":
                tables.append(item)
            elif rec.get("kind") == "statistic":
                statistics.append(item)

        step_records = manifest.get("per_step_records", []) or []
        step_status = {
            str(rec.get("step_id")): rec.get("status")
            for rec in step_records
            if isinstance(rec, dict) and rec.get("step_id")
        }
        findings = manifest.get("findings", []) or []
        risk_notes: List[str] = []
        if any(f.get("severity") == "error" for f in findings if isinstance(f, dict)):
            risk_notes.append("At least one validator error is present; do not use this run for paper figures without repair.")
        if "05_primary_association" in step_status and step_status.get("05_primary_association") != "ok":
            risk_notes.append("Primary association step did not complete; association figures are not ready.")

        primary_step_summary = run_dir / "steps" / "05_primary_association" / "outputs" / "step_summary.json"
        if primary_step_summary.exists():
            try:
                primary_summary = json.loads(primary_step_summary.read_text(encoding="utf-8"))
            except Exception:
                primary_summary = {}
            adjusted_fig = (primary_summary.get("outputs") or {}).get("adjusted_effect_summary_png")
            adjusted_status = ((primary_summary.get("figure") or {}).get("status"))
            if not adjusted_fig or adjusted_status == "skipped":
                risk_notes.append(
                    "Adjusted association figure is not ready; use unadjusted association table only "
                    "or refit a more stable adjusted model before drawing adjusted effects."
                )

        handoff["runs"].append({
            "database": s["database"],
            "run_id": s["run_id"],
            "workdir": str(run_dir),
            "cohort": s.get("cohort"),
            "step_status": step_status,
            "validator_findings": findings,
            "risk_notes": risk_notes,
            "figures": figures,
            "tables": tables,
            "statistics": statistics,
            "recommended_figure_inputs": {
                "cohort_and_outcome": [
                    t for t in tables
                    if t.get("evidence_id") and (
                        "table_one" in str(t.get("evidence_id"))
                        or "outcome_incidence" in str(t.get("evidence_id"))
                    )
                ],
                "missingness": [
                    x for x in [*tables, *figures]
                    if x.get("evidence_id") and "missing" in str(x.get("evidence_id"))
                ],
                "sofa2_strata": [
                    x for x in [*tables, *figures]
                    if x.get("evidence_id") and "sofa2" in str(x.get("evidence_id"))
                ],
                "association": [
                    t for t in tables
                    if t.get("evidence_id") and (
                        "unadjusted_associations" in str(t.get("evidence_id"))
                        or "primary_association" in str(t.get("evidence_id"))
                    )
                ],
            },
        })
    return handoff


def _render_paper_handoff_markdown(handoff: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Paper figure handoff",
        "",
        f"_Generated {handoff.get('generated_at')}._",
        "",
        "This file marks the default EasyICU paper-display boundary: analysis is complete, evidence is registered, and figure source artefacts are ready for manual/Nature-style figure QA.",
        "",
    ]
    for run in handoff.get("runs", []) or []:
        lines.extend([
            f"## {run.get('database')} — {run.get('run_id')}",
            "",
            f"- Workdir: `{run.get('workdir')}`",
        ])
        cohort = run.get("cohort") or {}
        if cohort:
            death_rate = cohort.get("death_rate")
            lines.append(f"- Cohort: {cohort.get('n_stays')} stays; death rate {death_rate:.1%}" if isinstance(death_rate, (int, float)) else f"- Cohort: {cohort.get('n_stays')} stays")
        step_status = run.get("step_status") or {}
        if step_status:
            ok = sum(1 for v in step_status.values() if v == "ok")
            lines.append(f"- Step status: {ok}/{len(step_status)} steps ok")
        risk_notes = run.get("risk_notes") or []
        lines.append("")
        lines.append("### Risk notes")
        lines.append("")
        if risk_notes:
            for note in risk_notes:
                lines.append(f"- {note}")
        else:
            lines.append("- No blocking paper-figure risk was detected by this handoff audit.")
        lines.append("")
        lines.append("### Registered figure inputs")
        lines.append("")
        for fig in run.get("figures", []) or []:
            lines.append(f"- `{fig.get('evidence_id')}` → `{fig.get('path')}`")
        lines.append("")
        lines.append("### Key registered table inputs")
        lines.append("")
        for table in run.get("tables", []) or []:
            eid = str(table.get("evidence_id") or "")
            if any(k in eid for k in ("table_one", "outcome_incidence", "missing", "sofa2", "association")):
                lines.append(f"- `{eid}` → `{table.get('path')}`")
        lines.append("")
    lines.append("Next: choose Python or R for the Nature-ready redraw, then generate publication SVG/PDF/TIFF from these registered inputs.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    _bootstrap_imports()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--databases", nargs="+",
                        default=["miiv", "eicu", "hirid"],
                        help="Databases to replicate across.")
    parser.add_argument("--data-root", default=os.environ.get("EASYICU_DATA_PATH", ""),
                        help="Root directory holding miiv/eicu/hirid subfolders. "
                             "When empty AND --dry-run is not set, the script "
                             "falls back to dry-run automatically.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use per-database synthetic cohorts instead of "
                             "real EasyICU extractions.")
    parser.add_argument("--max-patients", type=int, default=2000,
                        help="Cap per-database patient count (real mode only).")
    parser.add_argument("--los-min-hours", type=float, default=6.0,
                        help="Minimum ICU length of stay in hours for real-mode cohort filtering.")
    parser.add_argument("--batch-size", type=int, default=20000,
                        help="Patient batch size for real-mode EasyICU extraction. "
                             "Use 10000 for tighter memory, 20000+ for faster runs "
                             "when memory is comfortable.")
    parser.add_argument("--cohort-parquet", default="",
                        help="Optional already-materialised cohort parquet for agent-only reruns.")
    parser.add_argument("--agent-timeout", type=float, default=300.0,
                        help="Per-step analysis subprocess timeout in seconds.")
    parser.add_argument("--llm", choices=["mock", "openai"], default="mock",
                        help="LLM backend for ResearchAgentPipeline.")
    parser.add_argument("--skill", default="sofa_mortality",
                        help="ClinicalSkill key, or 'none' for free-form planning.")
    parser.add_argument("--question", default=(
                        "In an adult first-ICU-stay cohort with ICU length of stay at least 24 hours, "
                        "describe whether early SOFA-2 component patterns and selected physiologic "
                        "variables are associated with in-hospital mortality."
                        ),
                        help="Research question when --skill none, or an override for the skill question.")
    parser.add_argument("--stop-after-analysis", dest="stop_after_analysis",
                        action="store_true", default=True,
                        help="Stop after registered analysis tables/figures and the paper-figure handoff are produced. This is the default.")
    parser.add_argument("--continue-to-manuscript", dest="stop_after_analysis",
                        action="store_false",
                        help="Continue past the analysis/figure-QA boundary into literature retrieval and manuscript generation.")
    parser.add_argument("--openai-model", default="gpt-4o-mini",
                        help="Model name when --llm openai.")
    parser.add_argument("--openai-base-url", default=None,
                        help="OpenAI-compatible base URL when --llm openai.")
    parser.add_argument("--openai-api-key", default=None,
                        help="API key when --llm openai.")
    parser.add_argument("--openai-timeout", type=float, default=120.0,
                        help="OpenAI-compatible chat completion request timeout in seconds.")
    parser.add_argument("--seed", type=int, default=7, help="Synthetic-cohort seed.")
    parser.add_argument("--n-per-db", type=int, default=600,
                        help="Stays per synthetic cohort (dry-run only).")
    parser.add_argument("--out-root",
                        default=str((Path.cwd() / "research_output" / "crossdb").resolve()))
    parser.add_argument("--no-replay", action="store_true",
                        help="Skip the per-database replay hash-stability check.")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # If no data root supplied and the user didn't pass --dry-run, fall back
    # to dry-run with a notice — better than failing silently.
    has_cohort_parquet = bool(str(args.cohort_parquet).strip())
    dry_run = bool(args.dry_run or (not args.data_root and not has_cohort_parquet))
    if dry_run and not args.dry_run and not has_cohort_parquet:
        print("ℹ️  no --data-root / EASYICU_DATA_PATH; falling back to --dry-run.")

    from easyicu.research_agent import MockLLMClient, OpenAIClient  # type: ignore
    if args.llm == "openai":
        llm = OpenAIClient(
            model=args.openai_model,
            base_url=args.openai_base_url,
            api_key=args.openai_api_key,
            request_timeout=float(args.openai_timeout),
        )
    else:
        llm = MockLLMClient()

    summaries: List[Dict[str, Any]] = []
    stability: List[Dict[str, Any]] = []
    for db in args.databases:
        wd = out_root / db
        wd.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {db.upper()} ===")
        if args.cohort_parquet:
            import pandas as pd
            cohort = pd.read_parquet(args.cohort_parquet)
        elif dry_run:
            cohort = _build_synth_cohort(db, n=int(args.n_per_db), seed=int(args.seed))
        else:
            data_path = os.path.join(args.data_root, db)
            cohort = _assemble_cohort(database=db, data_path=data_path,
                                      max_patients=args.max_patients,
                                      los_min_hours=float(args.los_min_hours),
                                      batch_size=int(args.batch_size) if args.batch_size else None,
                                      verbose=True)
        skill_key = None if str(args.skill).lower() in {"none", "null", ""} else str(args.skill)
        summaries.append(_run_database(
            database=db,
            cohort=cohort,
            workdir=wd,
            llm=llm,
            timeout_seconds=float(args.agent_timeout),
            question=str(args.question),
            skill_key=skill_key,
            los_min_hours=float(args.los_min_hours),
            stop_after_analysis=bool(args.stop_after_analysis),
        ))

        if not args.no_replay:
            print(f"[{db}] hash-stability replay …")
            stability.append(_hash_stability_check(
                database=db, cohort=cohort.copy(), workdir=wd, llm=llm,
                timeout_seconds=float(args.agent_timeout),
                question=str(args.question), skill_key=skill_key,
                los_min_hours=float(args.los_min_hours),
                stop_after_analysis=bool(args.stop_after_analysis),
            ))

    summary_obj = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run_synthetic" if dry_run else "real",
        "databases": args.databases,
        "summaries": summaries,
        "hash_stability": stability,
    }
    (out_root / "crossdb_summary.json").write_text(
        json.dumps(summary_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    skill_label = "free-form planner" if str(args.skill).lower() in {"none", "null", ""} else str(args.skill)
    md = _render_markdown(
        summaries=summaries,
        stability=stability,
        dry_run=dry_run,
        los_min_hours=float(args.los_min_hours),
        skill_label=skill_label,
    )
    (out_root / "crossdb_summary.md").write_text(md, encoding="utf-8")

    if args.stop_after_analysis:
        handoff = _paper_figure_handoff(summaries=summaries, out_root=out_root)
        (out_root / "paper_figure_handoff.json").write_text(
            json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_root / "paper_figure_handoff.md").write_text(
            _render_paper_handoff_markdown(handoff), encoding="utf-8")

    print("\n=== Cross-database summary ===")
    print(f"  -> {out_root / 'crossdb_summary.json'}")
    print(f"  -> {out_root / 'crossdb_summary.md'}")
    if args.stop_after_analysis:
        print(f"  -> {out_root / 'paper_figure_handoff.json'}")
        print(f"  -> {out_root / 'paper_figure_handoff.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

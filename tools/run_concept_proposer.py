#!/usr/bin/env python3
"""T2 concept-proposer driver: turn a not-in-dictionary concept into a
human-review-ready, evidence-backed proposal — without letting the LLM write
extraction code.

Flow (selection-only LLM + deterministic gatekeeping):
  1. gather candidate catalog rows for the concept (frozen d_labitems/d_items)
  2. real proxy LLM SELECTS itemids + declarative metadata (no code)
  3. real-data distribution probe (duckdb over the source shards)
  4. validate_concept_proposal() runs the deterministic gate battery
  5. write proposal_<name>.{json,md} — the human-confirm handoff

A proposal can only reach ``needs_human_review``; nothing is auto-accepted and
nothing is written to the shared concept-dict. API keys come from the
environment; none are written to disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT / "src"), str(REPO_ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from easyicu.research_agent.discovery.concept_proposal import (  # noqa: E402
    DistributionStat,
    gather_candidate_rows,
    propose_concept_selection,
    validate_concept_proposal,
)
from easyicu.research_agent.discovery.idea_mining_feasibility_tier import (  # noqa: E402
    SourceItemIndex,
)
from easyicu.research_agent.providers.protocol import LLMMessage  # noqa: E402

DEFAULT_DB = Path("/Volumes/外置硬盘/databases/mimiciv")
DEFAULT_CATALOG = REPO_ROOT / "benchmark" / "source_item_catalog_miiv.json"

# table -> (shard glob, linking-id column, denominator table glob)
_TABLE_SOURCES = {
    "hosp/labevents": (
        "hosp/labevents_bucket/**/*.parquet",
        "hadm_id",
        "hosp/admissions*.parquet",
    ),
    "icu/chartevents": (
        "icu/chartevents_bucket/**/*.parquet",
        "stay_id",
        "icu/icustays*.parquet",
    ),
}


def make_duckdb_probe(db: Path):
    """Return a distribution_probe(itemids, table) backed by real shards."""
    import duckdb

    con = duckdb.connect()
    denom_cache: dict[str, int] = {}

    def _denominator(table: str, link_col: str, denom_glob: str) -> int:
        if table in denom_cache:
            return denom_cache[table]
        files = list(db.glob(denom_glob.replace("*", "")[:-1] + "*"))  # noqa
        # admissions/icustays are small single files; fall back to a glob.
        path = (db / denom_glob).as_posix()
        try:
            n = con.execute(
                f"SELECT count(DISTINCT {link_col}) FROM read_parquet('{path}')"
            ).fetchone()[0]
        except Exception:  # noqa: BLE001
            n = 0
        denom_cache[table] = int(n or 0)
        return denom_cache[table]

    def probe(itemids, table):
        if table not in _TABLE_SOURCES or not itemids:
            return {}
        glob, link_col, denom_glob = _TABLE_SOURCES[table]
        path = (db / glob).as_posix()
        denom = _denominator(table, link_col, denom_glob) or 1
        ids = ",".join(str(int(i)) for i in itemids)
        rows = con.execute(
            f"""
            SELECT itemid,
                   count(*) AS n_rows,
                   count(DISTINCT {link_col}) AS n_link,
                   approx_quantile(valuenum, 0.01) AS p01,
                   approx_quantile(valuenum, 0.50) AS p50,
                   approx_quantile(valuenum, 0.99) AS p99,
                   list(DISTINCT lower(trim(valueuom))) AS units
            FROM read_parquet('{path}')
            WHERE itemid IN ({ids}) AND valuenum IS NOT NULL
            GROUP BY itemid
            """
        ).fetchall()
        out: dict[int, DistributionStat] = {}
        for r in rows:
            iid, n_rows, n_link, p01, p50, p99, units = r
            units_t = tuple(u for u in (units or []) if u)
            out[int(iid)] = DistributionStat(
                itemid=int(iid),
                n_rows=int(n_rows),
                n_stays=int(n_link),
                coverage_fraction=round(float(n_link) / denom, 6),
                p01=None if p01 is None else float(p01),
                p50=None if p50 is None else float(p50),
                p99=None if p99 is None else float(p99),
                units=units_t,
            )
        return out

    return probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "concept", help="concept name to propose, e.g. 'lactate dehydrogenase'"
    )
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--model", default="gpt5.4")
    ap.add_argument("--candidate-limit", type=int, default=15)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "research_output" / "_concept_proposals",
    )
    ap.add_argument(
        "--no-probe",
        action="store_true",
        help="skip the real-data distribution gate (proposal stays unvalidated).",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index = SourceItemIndex.from_json(args.catalog)
    rows = gather_candidate_rows(index, args.concept, limit=args.candidate_limit)
    print(f"[1/4] candidate catalog rows: {len(rows)}", flush=True)
    if not rows:
        raise SystemExit(
            f"no source-catalog items match {args.concept!r} — this is T3 "
            "(not measured in this database); no proposal possible."
        )

    from run_research_agent_bench import _make_llm  # noqa: E402

    client = _make_llm(provider="openai", model=args.model, request_timeout=600.0)

    from easyicu.research_agent.providers.factory import authorized_complete

    def complete(system: str, user: str) -> str:
        return authorized_complete(
            client,
            [
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
            max_tokens=1024,
            temperature=0.0,
        )

    print("[2/4] LLM selection (itemids + declarative metadata only) ...", flush=True)
    draft = propose_concept_selection(args.concept, rows, complete=complete)
    print(
        f"      selected itemids={list(draft.candidate_itemids)} role={draft.role} "
        f"unit={draft.unit} bounds=[{draft.min_value},{draft.max_value}] "
        f"fluid={draft.target_fluid}",
        flush=True,
    )

    probe = None
    if not args.no_probe and args.db.exists():
        print("[3/4] real-data distribution probe (duckdb) ...", flush=True)
        probe = make_duckdb_probe(args.db)
    else:
        print("[3/4] distribution probe skipped", flush=True)

    print("[4/4] deterministic gate battery ...", flush=True)
    result = validate_concept_proposal(
        draft, source_index=index, distribution_probe=probe
    )

    catalog_rows = {int(r["itemid"]): r for r in rows}
    payload = {
        "concept": args.concept,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": result.status,
        "quarantine": draft.quarantine,
        "draft": {
            "candidate_itemids": list(draft.candidate_itemids),
            "role": draft.role,
            "unit": draft.unit,
            "min_value": draft.min_value,
            "max_value": draft.max_value,
            "target_fluid": draft.target_fluid,
            "rationale": draft.rationale,
        },
        "resolved_itemids": list(result.resolved_itemids),
        "dropped_itemids": list(result.dropped_itemids),
        "gate_findings": [
            {"gate": f.gate, "severity": f.severity, "message": f.message}
            for f in result.findings
        ],
        "distribution": [
            {
                "itemid": s.itemid,
                "label": catalog_rows.get(s.itemid, {}).get("label"),
                "n_rows": s.n_rows,
                "n_stays": s.n_stays,
                "coverage_fraction": s.coverage_fraction,
                "p01": s.p01,
                "p50": s.p50,
                "p99": s.p99,
                "units": list(s.units),
            }
            for s in result.distribution
        ],
    }
    slug = "".join(c if c.isalnum() else "_" for c in args.concept.lower())[:40]
    json_path = args.out_dir / f"proposal_{slug}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        f"# Concept proposal (T2): {args.concept}",
        "",
        f"**Status:** `{result.status}` (never auto-accepted; quarantine="
        f"{draft.quarantine}, run-local + provenance-tagged)",
        "",
        f"LLM selection: role={draft.role}, unit={draft.unit}, "
        f"bounds=[{draft.min_value}, {draft.max_value}], fluid={draft.target_fluid}",
        f"Rationale: {draft.rationale}",
        "",
        "## Resolved source items (after deterministic gates)",
        "| itemid | label | table | n_stays | coverage | p01/p50/p99 | units |",
        "|---|---|---|---|---|---|---|",
    ]
    dist_by = {s.itemid: s for s in result.distribution}
    for iid in result.resolved_itemids:
        r = catalog_rows.get(iid, {})
        s = dist_by.get(iid)
        cov = f"{s.coverage_fraction:.4f}" if s else "-"
        q = f"{s.p01}/{s.p50}/{s.p99}" if s else "-"
        u = ", ".join(s.units) if s else "-"
        ns = s.n_stays if s else "-"
        md.append(
            f"| {iid} | {r.get('label')} | {r.get('table')} | {ns} | {cov} | {q} | {u} |"
        )
    md += ["", "## Gate findings"]
    for f in result.findings:
        md.append(f"- **[{f.severity}] {f.gate}** — {f.message}")
    md += [
        "",
        "## Human review",
        "Confirm each resolved itemid truly represents the concept (label, "
        "specimen, unit, plausible distribution). On approval the concept is "
        "used run-local only and every result using it is tagged "
        "'AI-proposed concept (unvalidated extension)'. Promotion into the "
        "shared dictionary is a separate maintainer action.",
    ]
    md_path = args.out_dir / f"proposal_{slug}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nstatus={result.status}")
    print(
        f"resolved={list(result.resolved_itemids)} dropped={list(result.dropped_itemids)}"
    )
    print(f"proposal: {md_path}")


if __name__ == "__main__":
    main()

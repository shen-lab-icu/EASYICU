"""EHRFlowBench-style benchmark runner for the research agent (T2.1).

For every :class:`tests.bench.items.BenchItem` we run the same cohort
through ``ResearchAgentPipeline`` *twice* — once with the ICU-aware
context (this work) and once with the naive context (T1.4 baseline) —
and score each run on four orthogonal axes:

1. **Direction match.** The fitted primary OR sign matches the
   item's `expected_or_direction`.
2. **ICU-rule findings.** Each substring in
   `expected_finding_substrings` appears in at least one validator
   finding's message — i.e. the agent surfaced the ICU pitfall the
   item was designed to expose.
3. **Evidence completeness.** Every kind in the standard set
   {code, log, table, figure, statistic} is registered for the run.
4. **Manuscript bindability.** Count of unresolved
   ``[evidence missing: …]`` markers in the bound scaffold (lower
   is better; 0 is the goal).

The bench then writes ``bench_results.json`` (machine-readable) and
``bench_results.md`` (paper-ready) under ``--out-root``. The
Markdown table is the figure caption for the EHRFlowBench-style
panel in the paper.

Usage::

    python tools/run_research_agent_bench.py
    python tools/run_research_agent_bench.py --items sofa2_mortality gcs_mortality
    python tools/run_research_agent_bench.py --seed 42 --out-root ./bench_runs

The bench is mock-LLM only by design: the goal is to isolate the
*context layer's* contribution from the LLM's, so a deterministic
mock that follows the context is exactly what we want here. Use
``examples/research_agent_real_llm_smoke.py`` (T1.3) for end-to-end
LLM verification.
"""

from __future__ import annotations

import argparse
import json
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
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


# ---------------------------------------------------------------------------
# Per-arm scoring
# ---------------------------------------------------------------------------


_REQUIRED_KINDS = {"code", "log", "table", "figure", "statistic"}


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _evidence_missing_count(run_dir: Path) -> int:
    bound = run_dir / "manuscript_scaffold_bound.md"
    if not bound.exists():
        return -1
    return bound.read_text(encoding="utf-8").count("[evidence missing:")


def _findings_join(manifest: Dict[str, Any]) -> str:
    return " || ".join(f.get("message", "") for f in manifest.get("findings", []))


def _primary_or(run_dir: Path) -> Optional[float]:
    """Return the primary OR from the primary_association step's summary."""
    for ssj in run_dir.rglob("step_summary.json"):
        try:
            data = json.loads(ssj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("method") == "logistic_regression" and data.get("primary_or") is not None:
            return float(data["primary_or"])
    return None


def _direction_match(or_value: Optional[float], expected: int) -> Optional[bool]:
    """+1 → OR > 1 ; -1 → OR < 1. Returns None if no OR was produced."""
    if or_value is None:
        return None
    if expected == +1:
        return or_value > 1.0
    if expected == -1:
        return or_value < 1.0
    return None


def _findings_substring_hits(manifest: Dict[str, Any], needles: List[str]) -> Dict[str, bool]:
    blob = _findings_join(manifest).lower()
    return {n: (n.lower() in blob) for n in needles}


def _kinds_complete(manifest: Dict[str, Any]) -> Dict[str, Any]:
    kinds = {e.get("kind") for e in manifest.get("evidence", [])}
    return {
        "kinds_seen": sorted(k for k in kinds if k),
        "kinds_missing": sorted(_REQUIRED_KINDS - kinds),
        "complete": _REQUIRED_KINDS <= kinds,
    }


def _score_arm(*, run_dir: Path, item, label: str) -> Dict[str, Any]:
    manifest = _load_manifest(run_dir)
    or_value = _primary_or(run_dir)
    return {
        "arm": label,
        "run_id": manifest.get("run_id"),
        "workdir": str(run_dir),
        "primary_or": or_value,
        "direction_match": _direction_match(or_value, item.expected_or_direction),
        "expected_direction": item.expected_or_direction,
        "icu_findings": _findings_substring_hits(manifest, item.expected_finding_substrings),
        "n_findings": len(manifest.get("findings", [])),
        "n_warnings": sum(1 for f in manifest.get("findings", []) if f.get("severity") == "warning"),
        "n_errors": sum(1 for f in manifest.get("findings", []) if f.get("severity") == "error"),
        "evidence_count": len(manifest.get("evidence", [])),
        "evidence_kinds": _kinds_complete(manifest),
        "evidence_missing_in_manuscript": _evidence_missing_count(run_dir),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run_one_arm(*, item, cohort, workdir: Path, disable_icu_context: bool, label: str) -> Dict[str, Any]:
    from easyicu.research_agent import ResearchAgentPipeline, MockLLMClient  # type: ignore

    workdir.mkdir(parents=True, exist_ok=True)
    pipeline = ResearchAgentPipeline(
        workdir=workdir, llm=MockLLMClient(),
        disable_icu_context=disable_icu_context,
    )
    started = time.monotonic()
    result = pipeline.run(
        question=item.research_question,
        cohort=cohort,
        cohort_name=f"bench_{item.key}",
        database="bench",
        target_outcome=item.target_outcome,
        inclusion_criteria=item.inclusion_criteria,
    )
    elapsed = time.monotonic() - started
    score = _score_arm(run_dir=Path(result.workdir), item=item, label=label)
    score["elapsed_seconds"] = round(elapsed, 2)
    return score


def _run_one_item(*, item, seed: int, out_root: Path, verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print(f"\n=== {item.key} — {item.name} ===")
    cohort = item.cohort_factory(seed)
    item_root = out_root / item.key
    naive = _run_one_arm(
        item=item, cohort=cohort.copy(),
        workdir=item_root / "naive",
        disable_icu_context=True, label="naive",
    )
    aware = _run_one_arm(
        item=item, cohort=cohort.copy(),
        workdir=item_root / "aware",
        disable_icu_context=False, label="aware",
    )
    return {
        "item_key": item.key,
        "name": item.name,
        "research_question": item.research_question,
        "expected_predictor": item.primary_predictor,
        "expected_or_direction": item.expected_or_direction,
        "cohort_size": int(len(cohort)),
        "naive": naive,
        "aware": aware,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-arm aggregate metrics across all bench items."""
    totals: Dict[str, Dict[str, int]] = {"naive": {}, "aware": {}}
    for arm in ("naive", "aware"):
        n_total = len(scores)
        n_dir_correct = sum(1 for s in scores if s[arm]["direction_match"] is True)
        n_dir_wrong = sum(1 for s in scores if s[arm]["direction_match"] is False)
        n_dir_missing = sum(1 for s in scores if s[arm]["direction_match"] is None)
        n_findings_full_hit = 0
        n_findings_partial = 0
        for s in scores:
            hits = s[arm]["icu_findings"]
            if not hits:
                continue
            if all(hits.values()):
                n_findings_full_hit += 1
            elif any(hits.values()):
                n_findings_partial += 1
        n_kinds_complete = sum(1 for s in scores if s[arm]["evidence_kinds"]["complete"])
        evidence_missing = sum(max(0, s[arm]["evidence_missing_in_manuscript"]) for s in scores)
        totals[arm] = {
            "n_items": n_total,
            "direction_correct": n_dir_correct,
            "direction_wrong": n_dir_wrong,
            "direction_missing": n_dir_missing,
            "icu_findings_full_hit": n_findings_full_hit,
            "icu_findings_partial_hit": n_findings_partial,
            "evidence_kinds_complete": n_kinds_complete,
            "evidence_missing_in_manuscripts": evidence_missing,
        }
    return totals


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_or(or_value: Optional[float]) -> str:
    if or_value is None:
        return "—"
    return f"{or_value:.2f}"


def _direction_marker(direction_match: Optional[bool]) -> str:
    if direction_match is True:
        return "✅"
    if direction_match is False:
        return "❌"
    return "—"


def _findings_marker(hits: Dict[str, bool]) -> str:
    if not hits:
        return "—"
    n_hit = sum(1 for v in hits.values() if v)
    return f"{n_hit}/{len(hits)}"


def _render_markdown(*, scores: List[Dict[str, Any]], totals: Dict[str, Any],
                     seed: int) -> str:
    lines: List[str] = [
        "# EHRFlowBench-style benchmark — research agent (T2.1)",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()} (seed={seed})._",
        "",
        "Each item runs the *same* cohort through `ResearchAgentPipeline` "
        "twice — once with the ICU-aware context (`aware`) and once with "
        "the naive context (`naive`, T1.4 ablation arm) — and scores both "
        "on direction match, ICU-rule findings, evidence completeness and "
        "manuscript bindability. Mock LLM (deterministic) is used for "
        "both arms so the comparison isolates the *context layer*.",
        "",
        "## Per-item results",
        "",
        "| Item | Direction (naive) | Direction (aware) | OR (naive) | OR (aware) | ICU findings (naive) | ICU findings (aware) | `[evidence missing]` (naive → aware) |",
        "|---|:-:|:-:|---:|---:|:-:|:-:|---:|",
    ]
    for s in scores:
        n = s["naive"]; a = s["aware"]
        lines.append(
            f"| `{s['item_key']}` "
            f"| {_direction_marker(n['direction_match'])} "
            f"| {_direction_marker(a['direction_match'])} "
            f"| {_fmt_or(n['primary_or'])} "
            f"| {_fmt_or(a['primary_or'])} "
            f"| {_findings_marker(n['icu_findings'])} "
            f"| {_findings_marker(a['icu_findings'])} "
            f"| {n['evidence_missing_in_manuscript']} → {a['evidence_missing_in_manuscript']} |"
        )
    lines.append("")

    lines.append("## Aggregate (across all items)")
    lines.append("")
    lines.append("| Metric | Naive | ICU-aware |")
    lines.append("|---|---:|---:|")
    rows = [
        ("Number of items", totals["naive"]["n_items"], totals["aware"]["n_items"]),
        ("Direction correct", totals["naive"]["direction_correct"],
         totals["aware"]["direction_correct"]),
        ("Direction wrong", totals["naive"]["direction_wrong"],
         totals["aware"]["direction_wrong"]),
        ("Direction missing (no OR produced)",
         totals["naive"]["direction_missing"], totals["aware"]["direction_missing"]),
        ("Items with all ICU findings hit",
         totals["naive"]["icu_findings_full_hit"],
         totals["aware"]["icu_findings_full_hit"]),
        ("Items with partial ICU findings",
         totals["naive"]["icu_findings_partial_hit"],
         totals["aware"]["icu_findings_partial_hit"]),
        ("Items with all 5 evidence kinds",
         totals["naive"]["evidence_kinds_complete"],
         totals["aware"]["evidence_kinds_complete"]),
        ("Total `[evidence missing]` lines (lower is better)",
         totals["naive"]["evidence_missing_in_manuscripts"],
         totals["aware"]["evidence_missing_in_manuscripts"]),
    ]
    for name, n, a in rows:
        lines.append(f"| {name} | {n} | {a} |")
    lines.append("")

    lines.append("## Per-item provenance")
    lines.append("")
    for s in scores:
        lines.append(f"- **{s['item_key']}** — `{s['naive']['workdir']}` "
                     f"(naive) ; `{s['aware']['workdir']}` (aware)")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    _bootstrap_imports()

    from tests.bench import BENCH_ITEMS  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", nargs="+", default=None,
                        help="Subset of bench item keys to run (default: all).")
    parser.add_argument("--seed", type=int, default=7,
                        help="Synthetic-cohort seed (deterministic).")
    parser.add_argument("--out-root",
                        default=str((Path.cwd() / "research_output" / "bench").resolve()))
    args = parser.parse_args()

    if args.items:
        items = [it for it in BENCH_ITEMS if it.key in set(args.items)]
        unknown = set(args.items) - {it.key for it in BENCH_ITEMS}
        if unknown:
            print(f"Unknown bench keys: {sorted(unknown)}; "
                  f"available: {[it.key for it in BENCH_ITEMS]}")
            return 2
    else:
        items = list(BENCH_ITEMS)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    scores: List[Dict[str, Any]] = []
    for item in items:
        scores.append(_run_one_item(item=item, seed=args.seed, out_root=out_root))

    totals = _aggregate(scores)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "items": [it.key for it in items],
        "scores": scores,
        "totals": totals,
    }
    (out_root / "bench_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md = _render_markdown(scores=scores, totals=totals, seed=args.seed)
    (out_root / "bench_results.md").write_text(md, encoding="utf-8")

    print()
    print("=== Bench complete ===")
    print(f"  -> {out_root / 'bench_results.json'}")
    print(f"  -> {out_root / 'bench_results.md'}")
    print()
    print(f"  Direction correct  — naive: {totals['naive']['direction_correct']}/"
          f"{totals['naive']['n_items']} ; "
          f"aware: {totals['aware']['direction_correct']}/{totals['aware']['n_items']}")
    print(f"  ICU findings full  — naive: {totals['naive']['icu_findings_full_hit']} ; "
          f"aware: {totals['aware']['icu_findings_full_hit']}")
    print(f"  Evidence missing   — naive: {totals['naive']['evidence_missing_in_manuscripts']} ; "
          f"aware: {totals['aware']['evidence_missing_in_manuscripts']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

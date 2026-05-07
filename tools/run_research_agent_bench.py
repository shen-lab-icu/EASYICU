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
    python tools/run_research_agent_bench.py --provider openrouter --model z-ai/glm-4.5-air:free
    python tools/run_research_agent_bench.py --provider openrouter --models z-ai/glm-4.5-air:free deepseek/deepseek-chat-v3-0324:free

The original bench was mock-LLM only so the comparison isolated the
*context layer's* contribution from the LLM's. This script now also
supports real OpenAI-compatible providers (notably OpenRouter free
tier models) to make paper-facing context-ablation and model-comparison
runs reproducible from one entrypoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


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


def _step_records(run_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for ssj in run_dir.rglob("step_summary.json"):
        try:
            records.append(json.loads(ssj.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records


def _step_substring_hits(run_dir: Path, needles: List[str]) -> Dict[str, bool]:
    if not needles:
        return {}
    tokens: List[str] = []
    for record in _step_records(run_dir):
        for key in ("step_name", "title", "method", "step_key", "description"):
            value = record.get(key)
            if value:
                tokens.append(str(value))
    blob = " || ".join(tokens).lower()
    return {n: (n.lower() in blob) for n in needles}


def _artifact_substring_hits(manifest: Dict[str, Any], needles: List[str]) -> Dict[str, bool]:
    if not needles:
        return {}
    tokens: List[str] = []
    for evidence in manifest.get("evidence", []):
        for key in ("artifact_id", "label", "kind", "path", "summary"):
            value = evidence.get(key)
            if value:
                tokens.append(str(value))
    blob = " || ".join(tokens).lower()
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
        "workflow_hits": _step_substring_hits(run_dir, getattr(item, "expected_step_substrings", [])),
        "artifact_hits": _artifact_substring_hits(manifest, getattr(item, "expected_artifact_substrings", [])),
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


def _run_one_arm(*, item, cohort, workdir: Path, disable_icu_context: bool, label: str, llm) -> Dict[str, Any]:
    from easyicu.research_agent import ResearchAgentPipeline  # type: ignore

    workdir.mkdir(parents=True, exist_ok=True)
    pipeline = ResearchAgentPipeline(
        workdir=workdir, llm=llm,
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


def _run_one_item(*, item, seed: int, out_root: Path, llm, verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print(f"\n=== {item.key} — {item.name} ===")
    cohort = item.cohort_factory(seed)
    item_root = out_root / item.key
    naive = _run_one_arm(
        item=item, cohort=cohort.copy(),
        workdir=item_root / "naive",
        disable_icu_context=True, label="naive", llm=llm,
    )
    aware = _run_one_arm(
        item=item, cohort=cohort.copy(),
        workdir=item_root / "aware",
        disable_icu_context=False, label="aware", llm=llm,
    )
    return {
        "item_key": item.key,
        "name": item.name,
        "research_question": item.research_question,
        "expected_predictor": item.primary_predictor,
        "expected_or_direction": item.expected_or_direction,
        "benchmark_family": getattr(item, "benchmark_family", "rule"),
        "difficulty": getattr(item, "difficulty", "basic"),
        "evidence_basis": getattr(item, "evidence_basis", "internal_synthetic"),
        "claim_scope": getattr(item, "claim_scope", "internal_benchmark_only"),
        "interpretation_note": getattr(item, "interpretation_note", None),
        "cohort_size": int(len(cohort)),
        "naive": naive,
        "aware": aware,
    }


def _reuse_arm_if_complete(*, arm_dir: Path, item, label: str) -> Optional[Dict[str, Any]]:
    if not arm_dir.exists():
        return None
    runs = sorted(
        (p for p in arm_dir.glob("run_*") if (p / "manifest.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not runs:
        return None
    return _score_arm(run_dir=runs[0], item=item, label=label)


def _run_one_item_with_reuse(
    *,
    item,
    seed: int,
    out_root: Path,
    llm,
    reuse_existing: bool,
    verbose: bool = True,
) -> Dict[str, Any]:
    if verbose:
        print(f"\n=== {item.key} — {item.name} ===")
    cohort = item.cohort_factory(seed)
    item_root = out_root / item.key

    naive = None
    aware = None
    if reuse_existing:
        naive = _reuse_arm_if_complete(arm_dir=item_root / "naive", item=item, label="naive")
        aware = _reuse_arm_if_complete(arm_dir=item_root / "aware", item=item, label="aware")

    if naive is None:
        naive = _run_one_arm(
            item=item, cohort=cohort.copy(),
            workdir=item_root / "naive",
            disable_icu_context=True, label="naive", llm=llm,
        )
    if aware is None:
        aware = _run_one_arm(
            item=item, cohort=cohort.copy(),
            workdir=item_root / "aware",
            disable_icu_context=False, label="aware", llm=llm,
        )

    return {
        "item_key": item.key,
        "name": item.name,
        "research_question": item.research_question,
        "expected_predictor": item.primary_predictor,
        "expected_or_direction": item.expected_or_direction,
        "benchmark_family": getattr(item, "benchmark_family", "rule"),
        "difficulty": getattr(item, "difficulty", "basic"),
        "evidence_basis": getattr(item, "evidence_basis", "internal_synthetic"),
        "claim_scope": getattr(item, "claim_scope", "internal_benchmark_only"),
        "interpretation_note": getattr(item, "interpretation_note", None),
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
        n_workflow_full_hit = 0
        n_workflow_partial = 0
        n_artifact_full_hit = 0
        n_artifact_partial = 0
        for s in scores:
            workflow_hits = s[arm].get("workflow_hits", {})
            if workflow_hits:
                if all(workflow_hits.values()):
                    n_workflow_full_hit += 1
                elif any(workflow_hits.values()):
                    n_workflow_partial += 1
            artifact_hits = s[arm].get("artifact_hits", {})
            if artifact_hits:
                if all(artifact_hits.values()):
                    n_artifact_full_hit += 1
                elif any(artifact_hits.values()):
                    n_artifact_partial += 1
        n_kinds_complete = sum(1 for s in scores if s[arm]["evidence_kinds"]["complete"])
        evidence_missing = sum(max(0, s[arm]["evidence_missing_in_manuscript"]) for s in scores)
        totals[arm] = {
            "n_items": n_total,
            "direction_correct": n_dir_correct,
            "direction_wrong": n_dir_wrong,
            "direction_missing": n_dir_missing,
            "icu_findings_full_hit": n_findings_full_hit,
            "icu_findings_partial_hit": n_findings_partial,
            "workflow_full_hit": n_workflow_full_hit,
            "workflow_partial_hit": n_workflow_partial,
            "artifact_full_hit": n_artifact_full_hit,
            "artifact_partial_hit": n_artifact_partial,
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


def _bench_label(scores: List[Dict[str, Any]]) -> str:
    families = sorted({str(s.get("benchmark_family") or "rule") for s in scores})
    if families == ["analysis"]:
        return "AnalysisBench"
    if families == ["rule"]:
        return "RuleBench"
    return "MixedBench"


def _render_markdown(*, scores: List[Dict[str, Any]], totals: Dict[str, Any],
                     seed: int, bench_kind: str) -> str:
    label = _bench_label(scores)
    lines: List[str] = [
        f"# {label} — research agent benchmark",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()} (seed={seed}, bench_kind={bench_kind})._",
        "",
        "Each item runs the *same* cohort through `ResearchAgentPipeline` "
        "twice — once with the ICU-aware context (`aware`) and once with "
        "the naive context (`naive`, T1.4 ablation arm) — and scores both "
        "on direction match, ICU-rule findings, evidence completeness and "
        "manuscript bindability. Within each benchmark run, both arms use "
        "the same LLM backend so the comparison still isolates the "
        "*context layer* as cleanly as possible.",
        "",
        "**Interpretation boundary.** All analysis-bench tasks use synthetic cohorts. "
        "`evidence_basis` describes how a task was designed (for example, literature-inspired, "
        "consensus-inspired, or internal stress-test synthetic); it does **not** mean the benchmark "
        "finding itself is externally validated. Substring-matched ICU findings are benchmark-rule hits, "
        "not stand-alone publishable clinical claims.",
        "",
        "## Per-item results",
        "",
        "| Item | Family | Difficulty | Evidence basis | Direction (naive) | Direction (aware) | OR (naive) | OR (aware) | Predefined rule hits (naive) | Predefined rule hits (aware) | Workflow hits (naive) | Workflow hits (aware) | Artifact hits (naive) | Artifact hits (aware) | `[evidence missing]` (naive → aware) |",
        "|---|---|---|---|:-:|:-:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|---:|",
    ]
    for s in scores:
        n = s["naive"]; a = s["aware"]
        lines.append(
            f"| `{s['item_key']}` "
            f"| `{s.get('benchmark_family', 'rule')}` "
            f"| `{s.get('difficulty', 'basic')}` "
            f"| `{s.get('evidence_basis', 'internal_synthetic')}` "
            f"| {_direction_marker(n['direction_match'])} "
            f"| {_direction_marker(a['direction_match'])} "
            f"| {_fmt_or(n['primary_or'])} "
            f"| {_fmt_or(a['primary_or'])} "
            f"| {_findings_marker(n['icu_findings'])} "
            f"| {_findings_marker(a['icu_findings'])} "
            f"| {_findings_marker(n.get('workflow_hits', {}))} "
            f"| {_findings_marker(a.get('workflow_hits', {}))} "
            f"| {_findings_marker(n.get('artifact_hits', {}))} "
            f"| {_findings_marker(a.get('artifact_hits', {}))} "
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
        ("Items with all predefined rule hits",
         totals["naive"]["icu_findings_full_hit"],
         totals["aware"]["icu_findings_full_hit"]),
        ("Items with partial predefined rule hits",
         totals["naive"]["icu_findings_partial_hit"],
         totals["aware"]["icu_findings_partial_hit"]),
        ("Items with all workflow expectations hit",
         totals["naive"]["workflow_full_hit"],
         totals["aware"]["workflow_full_hit"]),
        ("Items with partial workflow expectations",
         totals["naive"]["workflow_partial_hit"],
         totals["aware"]["workflow_partial_hit"]),
        ("Items with all artifact expectations hit",
         totals["naive"]["artifact_full_hit"],
         totals["aware"]["artifact_full_hit"]),
        ("Items with partial artifact expectations",
         totals["naive"]["artifact_partial_hit"],
         totals["aware"]["artifact_partial_hit"]),
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

    lines.append("## Interpretation Notes")
    lines.append("")
    for s in scores:
        note = s.get("interpretation_note") or "Interpret only as an internal benchmark result."
        lines.append(f"- **{s['item_key']}** — `{s.get('claim_scope', 'internal_benchmark_only')}`. {note}")
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


def _slugify_model(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip())
    return slug.strip("._-") or "model"


def _make_llm(*, provider: str, model: str, request_timeout: float):
    from easyicu.research_agent import MockLLMClient, OpenAIClient  # type: ignore

    if provider == "mock":
        return MockLLMClient()
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is required for --provider openrouter")
        return OpenAIClient(
            model=model,
            api_key=key,
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            request_timeout=float(request_timeout),
            extra_headers={
                "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
                "X-Title": "EasyICU research-agent benchmark",
            },
            extra_body={"reasoning": {"effort": "none", "exclude": True}},
        )
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY is required for --provider openai")
        return OpenAIClient(
            model=model,
            api_key=key,
            request_timeout=float(request_timeout),
        )
    raise SystemExit(f"Unsupported provider: {provider}")


def _run_suite(
    *,
    items: Sequence[Any],
    out_root: Path,
    seed: int,
    bench_kind: str,
    provider: str,
    model: str,
    verbose: bool = True,
    request_timeout: float = 180.0,
    reuse_existing: bool = False,
) -> Dict[str, Any]:
    llm = _make_llm(provider=provider, model=model, request_timeout=request_timeout)
    scores: List[Dict[str, Any]] = []
    for item in items:
        scores.append(
            _run_one_item_with_reuse(
                item=item,
                seed=seed,
                out_root=out_root,
                llm=llm,
                reuse_existing=reuse_existing,
                verbose=verbose,
            )
        )

    totals = _aggregate(scores)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "bench_kind": bench_kind,
        "provider": provider,
        "model": model,
        "items": [it.key for it in items],
        "scores": scores,
        "totals": totals,
    }
    (out_root / "bench_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md = _render_markdown(scores=scores, totals=totals, seed=seed, bench_kind=bench_kind)
    header = [
        f"_Provider: `{provider}`_",
        f"_Model: `{model}`_",
        "",
    ]
    (out_root / "bench_results.md").write_text("\n".join(header) + md, encoding="utf-8")
    return payload


def _render_model_matrix(runs: List[Dict[str, Any]]) -> str:
    lines = [
        "# Benchmark model matrix",
        "",
        "| Model | Provider | Bench kind | Direction correct (aware) | Direction correct (naive) | ICU findings full-hit (aware) | ICU findings full-hit (naive) | Workflow full-hit (aware) | Workflow full-hit (naive) | Artifact full-hit (aware) | Artifact full-hit (naive) | Evidence missing (aware) | Evidence missing (naive) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        totals = run["totals"]
        lines.append(
            f"| `{run['model']}` | `{run['provider']}` | `{run.get('bench_kind', 'rule')}` | "
            f"{totals['aware']['direction_correct']}/{totals['aware']['n_items']} | "
            f"{totals['naive']['direction_correct']}/{totals['naive']['n_items']} | "
            f"{totals['aware']['icu_findings_full_hit']} | "
            f"{totals['naive']['icu_findings_full_hit']} | "
            f"{totals['aware'].get('workflow_full_hit', 0)} | "
            f"{totals['naive'].get('workflow_full_hit', 0)} | "
            f"{totals['aware'].get('artifact_full_hit', 0)} | "
            f"{totals['naive'].get('artifact_full_hit', 0)} | "
            f"{totals['aware']['evidence_missing_in_manuscripts']} | "
            f"{totals['naive']['evidence_missing_in_manuscripts']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    _bootstrap_imports()

    from tests.bench import ANALYSIS_BENCH_ITEMS, RULE_BENCH_ITEMS  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-kind", choices=["rule", "analysis"], default="rule",
                        help="Which benchmark fixture family to run.")
    parser.add_argument("--items", nargs="+", default=None,
                        help="Subset of bench item keys to run (default: all).")
    parser.add_argument("--seed", type=int, default=7,
                        help="Synthetic-cohort seed (deterministic).")
    parser.add_argument("--out-root",
                        default=str((Path.cwd() / "research_output" / "bench").resolve()))
    parser.add_argument("--provider", choices=["mock", "openrouter", "openai"], default="mock",
                        help="LLM backend for the benchmark arms.")
    parser.add_argument("--model",
                        default=os.environ.get("EASYICU_HOSTED_DEFAULT_MODEL", "z-ai/glm-4.5-air:free"),
                        help="Single model name for real-provider runs.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Optional multiple model names. When set, the benchmark runs once per model.")
    parser.add_argument("--request-timeout", type=float, default=180.0,
                        help="Per-request timeout for real LLM providers.")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Reuse completed item/arm runs already present under --out-root.")
    parser.add_argument("--ehrflowbench-jsonl", default=None,
                        help="Optional EHRFlowBench-style JSONL export. Each row may include "
                             "key, question, cohort_path, target_outcome, expected_or_direction.")
    args = parser.parse_args()

    if args.ehrflowbench_jsonl:
        return _run_ehrflowbench_jsonl(
            jsonl_path=Path(args.ehrflowbench_jsonl).resolve(),
            out_root=Path(args.out_root).resolve(),
            seed=args.seed,
        )

    all_items = list(RULE_BENCH_ITEMS if args.bench_kind == "rule" else ANALYSIS_BENCH_ITEMS)
    if args.items:
        items = [it for it in all_items if it.key in set(args.items)]
        unknown = set(args.items) - {it.key for it in all_items}
        if unknown:
            print(f"Unknown bench keys: {sorted(unknown)}; "
                  f"available: {[it.key for it in all_items]}")
            return 2
    else:
        items = all_items

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.provider == "mock":
        models = ["mock"]
    else:
        models = list(args.models or [args.model])
    all_runs: List[Dict[str, Any]] = []
    for idx, model in enumerate(models):
        model_root = out_root if len(models) == 1 else (out_root / _slugify_model(model))
        model_root.mkdir(parents=True, exist_ok=True)
        if len(models) > 1:
            print(f"\n=== Model {idx + 1}/{len(models)} — {model} ===")
        payload = _run_suite(
            items=items,
            out_root=model_root,
            seed=args.seed,
            bench_kind=args.bench_kind,
            provider=args.provider,
            model=model,
            request_timeout=float(args.request_timeout),
            reuse_existing=bool(args.reuse_existing),
        )
        all_runs.append(payload)
        totals = payload["totals"]
        print()
        print(f"=== Bench complete — {model} ===")
        print(f"  -> {model_root / 'bench_results.json'}")
        print(f"  -> {model_root / 'bench_results.md'}")
        print(f"  Direction correct  — naive: {totals['naive']['direction_correct']}/"
              f"{totals['naive']['n_items']} ; "
              f"aware: {totals['aware']['direction_correct']}/{totals['aware']['n_items']}")
        print(f"  ICU findings full  — naive: {totals['naive']['icu_findings_full_hit']} ; "
              f"aware: {totals['aware']['icu_findings_full_hit']}")
        print(f"  Evidence missing   — naive: {totals['naive']['evidence_missing_in_manuscripts']} ; "
              f"aware: {totals['aware']['evidence_missing_in_manuscripts']}")

    if len(all_runs) > 1:
        matrix_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "bench_kind": args.bench_kind,
            "provider": args.provider,
            "items": [it.key for it in items],
            "runs": all_runs,
        }
        (out_root / "bench_model_matrix.json").write_text(
            json.dumps(matrix_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (out_root / "bench_model_matrix.md").write_text(
            _render_model_matrix(all_runs),
            encoding="utf-8",
        )
        print(f"  -> {out_root / 'bench_model_matrix.json'}")
        print(f"  -> {out_root / 'bench_model_matrix.md'}")
    return 0


def _run_ehrflowbench_jsonl(*, jsonl_path: Path, out_root: Path, seed: int) -> int:
    """Run an external EHRFlowBench-style JSONL export when available."""
    from types import SimpleNamespace
    import pandas as pd

    if not jsonl_path.exists():
        print(f"EHRFlowBench JSONL not found: {jsonl_path}")
        return 2
    out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            rows.append({"status": "invalid_json", "error": str(exc), "raw": line[:200]})

    scores: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        key = str(row.get("key") or row.get("id") or f"ehrflowbench_{idx:03d}")
        cohort_path = row.get("cohort_path") or row.get("cohort")
        question = row.get("question") or row.get("research_question")
        target = row.get("target_outcome") or row.get("outcome")
        if not cohort_path or not question or not target:
            pending.append({
                "key": key,
                "status": "pending_missing_fields",
                "required": ["question", "cohort_path", "target_outcome"],
            })
            continue
        path = Path(str(cohort_path)).expanduser().resolve()
        if not path.exists():
            pending.append({"key": key, "status": "pending_missing_cohort", "cohort_path": str(path)})
            continue
        if path.suffix.lower() in {".parquet", ".pq"}:
            cohort = pd.read_parquet(path)
        elif path.suffix.lower() in {".csv", ".tsv"}:
            cohort = pd.read_csv(path, sep=("\t" if path.suffix.lower() == ".tsv" else ","))
        else:
            pending.append({"key": key, "status": "unsupported_cohort_format", "cohort_path": str(path)})
            continue
        item = SimpleNamespace(
            key=key,
            name=str(row.get("name") or key),
            research_question=str(question),
            target_outcome=str(target),
            primary_predictor=str(row.get("primary_predictor") or ""),
            expected_or_direction=int(row.get("expected_or_direction") or 0),
            expected_finding_substrings=list(row.get("expected_finding_substrings") or []),
            inclusion_criteria=list(row.get("inclusion_criteria") or []),
        )
        scores.append(_run_one_item_from_cohort(item=item, cohort=cohort, out_root=out_root))

    totals = _aggregate(scores) if scores else {"naive": {}, "aware": {}}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(jsonl_path),
        "seed": seed,
        "scores": scores,
        "pending": pending,
        "totals": totals,
    }
    (out_root / "ehrflowbench_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    md = [
        "# EHRFlowBench external import",
        "",
        f"Source: `{jsonl_path}`",
        f"Runnable items: {len(scores)}",
        f"Pending items: {len(pending)}",
        "",
    ]
    if scores:
        md.append(_render_markdown(scores=scores, totals=totals, seed=seed, bench_kind="external"))
    if pending:
        md.extend(["", "## Pending", ""])
        for p in pending:
            md.append(f"- `{p['key']}` — {p['status']}")
    (out_root / "ehrflowbench_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  -> {out_root / 'ehrflowbench_results.json'}")
    print(f"  -> {out_root / 'ehrflowbench_results.md'}")
    return 0


def _run_one_item_from_cohort(*, item, cohort, out_root: Path) -> Dict[str, Any]:
    llm = _make_llm(provider="mock", model="mock", request_timeout=180.0)
    item_root = out_root / item.key
    naive = _run_one_arm(
        item=item, cohort=cohort.copy(),
        workdir=item_root / "naive",
        disable_icu_context=True, label="naive", llm=llm,
    )
    aware = _run_one_arm(
        item=item, cohort=cohort.copy(),
        workdir=item_root / "aware",
        disable_icu_context=False, label="aware", llm=llm,
    )
    return {
        "item_key": item.key,
        "name": item.name,
        "research_question": item.research_question,
        "expected_predictor": item.primary_predictor,
        "expected_or_direction": item.expected_or_direction,
        "benchmark_family": getattr(item, "benchmark_family", "external"),
        "difficulty": getattr(item, "difficulty", "external"),
        "evidence_basis": getattr(item, "evidence_basis", "external_import"),
        "claim_scope": getattr(item, "claim_scope", "external_import_only"),
        "interpretation_note": getattr(item, "interpretation_note", None),
        "cohort_size": int(len(cohort)),
        "naive": naive,
        "aware": aware,
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

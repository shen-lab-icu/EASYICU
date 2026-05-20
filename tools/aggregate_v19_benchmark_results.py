"""Aggregate v19 EasyICU benchmark runs into Figure 4 source data.

This is intentionally a lightweight source-data builder, not a statistical
analysis. It reads one or more ``bench_results.json`` files emitted by
``run_research_agent_bench.py`` and joins them to the frozen Figure 4 rubric.

Example:

    python tools/aggregate_v19_benchmark_results.py \
      --rubric ../easyicu写作/00_当前投稿_20260516/v19_benchmark_runs/figure4_benchmark_rubric_20260519.csv \
      --runs ../easyicu写作/00_当前投稿_20260516/v19_benchmark_runs/gpt54_dryrun_20260519/analysis_hepatobiliary_missingness_aware_only \
      --item-map analysis_hepatobiliary_missingness=M1 \
      --out-csv /tmp/figure4_results.csv \
      --out-md /tmp/figure4_results.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RESULT_COLUMNS = [
    "question_id",
    "difficulty",
    "short_name",
    "item_key",
    "model",
    "provider",
    "run_id",
    "run_dir",
    "arms",
    "pipeline_options",
    "status",
    "elapsed_seconds",
    "request_count",
    "plan_completion",
    "code_execution",
    "result_validity",
    "evidence_binding",
    "audit_conclusion_safety",
    "plan_completion_rank",
    "code_execution_rank",
    "result_validity_rank",
    "evidence_binding_rank",
    "audit_conclusion_safety_rank",
    "overall_status",
    "primary_failure_category",
    "evidence_missing_count",
    "n_warnings",
    "n_errors",
    "manuscript_ready",
    "figure_ready",
    "notes",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rubric(path: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["question_id"]] = row
            rows[row["short_name"]] = row
    return rows


def _parse_item_map(items: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--item-map must use item_key=question_id, got {item!r}")
        key, question_id = item.split("=", 1)
        mapping[key.strip()] = question_id.strip()
    return mapping


def _bench_paths(paths: Iterable[Path]) -> List[Path]:
    found: List[Path] = []
    for path in paths:
        p = path.resolve()
        if p.is_file() and p.name == "bench_results.json":
            found.append(p)
        elif p.is_dir() and (p / "bench_results.json").exists():
            found.append(p / "bench_results.json")
        elif p.is_dir():
            found.extend(sorted(p.rglob("bench_results.json")))
        else:
            raise SystemExit(f"Run path not found or unsupported: {path}")
    unique: List[Path] = []
    seen = set()
    for path in found:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _run_status_paths(paths: Iterable[Path]) -> List[Path]:
    found: List[Path] = []
    for path in paths:
        p = path.resolve()
        if p.is_file() and p.name == "run_status.json":
            found.append(p)
        elif p.is_dir() and (p / "run_status.json").exists():
            found.append(p / "run_status.json")
        elif p.is_dir():
            found.extend(sorted(p.rglob("run_status.json")))
        else:
            raise SystemExit(f"Run-status path not found or unsupported: {path}")
    unique: List[Path] = []
    seen = set()
    for path in found:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _load_run_status(run_dir: Optional[str]) -> Dict[str, Any]:
    if not run_dir:
        return {}
    path = Path(run_dir) / "run_status.json"
    if not path.exists():
        return {}
    try:
        return _read_json(path)
    except Exception:
        return {}


_RANK_TO_LABEL = {2: "pass", 1: "partial", 0: "fail"}


def _label(rank: int) -> str:
    return _RANK_TO_LABEL.get(rank, "fail")


def _score_plan(gates: Dict[str, Any], arm: Dict[str, Any]) -> int:
    required = gates.get("required_step_count")
    completed = gates.get("completed_step_count")
    failed = gates.get("failed_steps") or []
    if required and completed == required and not failed:
        return 2
    if completed or arm.get("evidence_count", 0) > 0:
        return 1
    return 0


def _score_code(gates: Dict[str, Any], arm: Dict[str, Any]) -> int:
    failed = gates.get("failed_steps") or []
    if gates and gates.get("execution_complete") is True and not failed:
        return 2
    if arm.get("evidence_count", 0) > 0 or gates.get("completed_step_count", 0) > 0:
        return 1
    return 0


def _score_result(gates: Dict[str, Any], arm: Dict[str, Any]) -> int:
    if arm.get("_from_run_status"):
        if gates.get("numeric_verified") and gates.get("analysis_validated"):
            return 2
        if gates.get("analysis_validated") or gates.get("completed_step_count", 0) > 0:
            return 1
        return 0
    if arm.get("primary_or") is not None and arm.get("direction_match") is not None:
        return 2 if not gates.get("analysis_errors") else 1
    if arm.get("primary_or") is not None:
        return 1
    return 0


def _score_evidence(arm: Dict[str, Any]) -> int:
    missing = arm.get("evidence_missing_in_manuscript")
    kinds = arm.get("evidence_kinds") or {}
    if kinds.get("complete") and missing == 0:
        return 2
    if arm.get("evidence_count", 0) > 0:
        return 1
    return 0


def _score_audit(gates: Dict[str, Any], arm: Dict[str, Any]) -> int:
    if gates.get("manuscript_ready") or gates.get("publication_ready"):
        return 2
    failed = gates.get("failed_steps") or []
    if failed and gates.get("manuscript_generated") is False:
        return 2
    if arm.get("n_warnings", 0) or arm.get("n_errors", 0):
        return 1
    return 1


def _failure_category(gates: Dict[str, Any], arm: Dict[str, Any]) -> str:
    combined_errors = " ".join(
        str(e)
        for key in ("analysis_errors", "numeric_errors", "evidence_errors")
        for e in (gates.get(key) or [])
    )
    if "usage_limit_reached" in combined_errors or "502" in combined_errors:
        return "provider_or_runtime_failure"
    if gates.get("numeric_error_count", 0) or gates.get("evidence_error_count", 0):
        return "evidence_binding_failure"
    if (arm.get("evidence_missing_in_manuscript") or 0) > 0:
        return "evidence_binding_failure"
    if gates.get("failed_steps"):
        return "audit_block_fail_closed"
    if arm.get("primary_or") is None:
        return "statistical_validity_failure"
    if arm.get("n_errors", 0) > 0:
        return "code_execution_failure"
    return ""


def _overall_status(ranks: Iterable[int], failure: str) -> str:
    values = list(ranks)
    if failure == "audit_block_fail_closed":
        return "diagnostic_blocked"
    if any(v == 0 for v in values):
        return "fail"
    if any(v == 1 for v in values):
        return "partial"
    return "pass"


def _request_count(bench_path: Path) -> str:
    terminal = bench_path.parent / "terminal_run.log"
    if not terminal.exists():
        return ""
    text = terminal.read_text(encoding="utf-8", errors="replace")
    return str(text.count("HTTP Request:"))


def _row_for_score(
    *,
    bench_path: Path,
    bench: Dict[str, Any],
    score: Dict[str, Any],
    arm_name: str,
    arm: Dict[str, Any],
    rubric: Dict[str, Dict[str, str]],
    item_map: Dict[str, str],
) -> Dict[str, Any]:
    item_key = score["item_key"]
    question_id = item_map.get(item_key, item_key)
    rubric_row = rubric.get(question_id, {})
    run_status = _load_run_status(arm.get("workdir"))
    gates = run_status.get("gates") or {}

    plan = _score_plan(gates, arm)
    code = _score_code(gates, arm)
    result = _score_result(gates, arm)
    evidence = _score_evidence(arm)
    audit = _score_audit(gates, arm)
    failure = _failure_category(gates, arm)
    ranks = [plan, code, result, evidence, audit]
    status = run_status.get("status") or ("complete" if arm.get("run_id") else "missing")

    notes: List[str] = []
    if gates.get("analysis_errors"):
        notes.append("; ".join(str(e) for e in gates["analysis_errors"][:2]))
    if arm.get("direction_match") is None:
        notes.append("direction missing")

    return {
        "question_id": rubric_row.get("question_id", question_id),
        "difficulty": rubric_row.get("difficulty", score.get("difficulty", "")),
        "short_name": rubric_row.get("short_name", ""),
        "item_key": item_key,
        "model": bench.get("model", ""),
        "provider": bench.get("provider", ""),
        "run_id": arm.get("run_id", ""),
        "run_dir": arm.get("workdir", ""),
        "arms": arm_name,
        "pipeline_options": json.dumps(bench.get("pipeline_options", {}), sort_keys=True),
        "status": status,
        "elapsed_seconds": arm.get("elapsed_seconds", ""),
        "request_count": _request_count(bench_path),
        "plan_completion": _label(plan),
        "code_execution": _label(code),
        "result_validity": _label(result),
        "evidence_binding": _label(evidence),
        "audit_conclusion_safety": _label(audit),
        "plan_completion_rank": plan,
        "code_execution_rank": code,
        "result_validity_rank": result,
        "evidence_binding_rank": evidence,
        "audit_conclusion_safety_rank": audit,
        "overall_status": _overall_status(ranks, failure),
        "primary_failure_category": failure,
        "evidence_missing_count": arm.get("evidence_missing_in_manuscript", ""),
        "n_warnings": arm.get("n_warnings", ""),
        "n_errors": arm.get("n_errors", ""),
        "manuscript_ready": gates.get("manuscript_ready", ""),
        "figure_ready": gates.get("publication_figure_bundle_ready", ""),
        "notes": " | ".join(n for n in notes if n),
    }


def _item_key_from_run_status_path(path: Path) -> str:
    # Expected layout: <root>/<item_key>/<arm>/run_<timestamp>/run_status.json
    run_dir = path.parent
    if run_dir.parent.name in {"aware", "naive"}:
        return run_dir.parent.parent.name
    return run_dir.parent.name


def _row_for_run_status(
    *,
    status_path: Path,
    run_status: Dict[str, Any],
    rubric: Dict[str, Dict[str, str]],
    item_map: Dict[str, str],
    model: str,
    provider: str,
) -> Dict[str, Any]:
    run_dir = status_path.parent
    item_key = _item_key_from_run_status_path(status_path)
    question_id = item_map.get(item_key, item_key)
    rubric_row = rubric.get(question_id, {})
    gates = run_status.get("gates") or {}
    evidence_dir = run_dir / "evidence"
    evidence_count = len(list(evidence_dir.iterdir())) if evidence_dir.exists() else 0
    arm = {
        "_from_run_status": True,
        "evidence_count": evidence_count,
        "evidence_missing_in_manuscript": gates.get("missing_evidence_count", ""),
        "run_id": run_dir.name,
        "workdir": str(run_dir),
        "n_warnings": gates.get("evidence_error_count", 0),
        "n_errors": (
            int(gates.get("numeric_error_count") or 0)
            + int(gates.get("evidence_error_count") or 0)
            + int(gates.get("analysis_error_count") or 0)
        ),
    }

    plan = _score_plan(gates, arm)
    code = _score_code(gates, arm)
    result = _score_result(gates, arm)
    evidence = _score_evidence(arm)
    audit = _score_audit(gates, arm)
    failure = _failure_category(gates, arm)
    ranks = [plan, code, result, evidence, audit]

    notes: List[str] = []
    for key in ("analysis_errors", "numeric_errors", "evidence_errors"):
        values = gates.get(key) or []
        if values:
            notes.append("; ".join(str(e) for e in values[:2]))

    return {
        "question_id": rubric_row.get("question_id", question_id),
        "difficulty": rubric_row.get("difficulty", ""),
        "short_name": rubric_row.get("short_name", ""),
        "item_key": item_key,
        "model": model,
        "provider": provider,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "arms": run_dir.parent.name if run_dir.parent.name in {"aware", "naive"} else "",
        "pipeline_options": "",
        "status": run_status.get("status", ""),
        "elapsed_seconds": "",
        "request_count": "",
        "plan_completion": _label(plan),
        "code_execution": _label(code),
        "result_validity": _label(result),
        "evidence_binding": _label(evidence),
        "audit_conclusion_safety": _label(audit),
        "plan_completion_rank": plan,
        "code_execution_rank": code,
        "result_validity_rank": result,
        "evidence_binding_rank": evidence,
        "audit_conclusion_safety_rank": audit,
        "overall_status": _overall_status(ranks, failure),
        "primary_failure_category": failure,
        "evidence_missing_count": gates.get("missing_evidence_count", ""),
        "n_warnings": arm["n_warnings"],
        "n_errors": arm["n_errors"],
        "manuscript_ready": gates.get("manuscript_ready", ""),
        "figure_ready": gates.get("publication_figure_bundle_ready", ""),
        "notes": " | ".join(n for n in notes if n),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Figure 4 Benchmark Results Source Data",
        "",
        "| Question | Difficulty | Model | Arm | Overall | Plan | Code | Result | Evidence | Audit | Failure |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {question_id} | {difficulty} | {model} | {arms} | "
            "{overall_status} | {plan_completion} | {code_execution} | "
            "{result_validity} | {evidence_binding} | "
            "{audit_conclusion_safety} | {primary_failure_category} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric", required=True, type=Path)
    parser.add_argument("--runs", nargs="*", default=[], type=Path)
    parser.add_argument("--status-runs", nargs="*", default=[], type=Path)
    parser.add_argument("--status-model", default="", help="Model label for --status-runs rows.")
    parser.add_argument("--status-provider", default="", help="Provider label for --status-runs rows.")
    parser.add_argument("--item-map", action="append", default=[])
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", default=None, type=Path)
    args = parser.parse_args()

    rubric = _load_rubric(args.rubric.resolve())
    item_map = _parse_item_map(args.item_map)

    rows: List[Dict[str, Any]] = []
    for bench_path in _bench_paths(args.runs):
        bench = _read_json(bench_path)
        arms = bench.get("arms") or ["naive", "aware"]
        for score in bench.get("scores", []):
            for arm_name in arms:
                arm = score.get(arm_name)
                if not arm or arm.get("status") == "skipped":
                    continue
                rows.append(
                    _row_for_score(
                        bench_path=bench_path,
                        bench=bench,
                        score=score,
                        arm_name=arm_name,
                        arm=arm,
                        rubric=rubric,
                        item_map=item_map,
                    )
                )

    for status_path in _run_status_paths(args.status_runs):
        rows.append(
            _row_for_run_status(
                status_path=status_path,
                run_status=_read_json(status_path),
                rubric=rubric,
                item_map=item_map,
                model=args.status_model,
                provider=args.status_provider,
            )
        )

    _write_csv(args.out_csv.resolve(), rows)
    if args.out_md:
        _write_md(args.out_md.resolve(), rows)
    print(f"Wrote {len(rows)} row(s) to {args.out_csv.resolve()}")
    if args.out_md:
        print(f"Wrote markdown to {args.out_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

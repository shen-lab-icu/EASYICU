"""Sequential full-flow validation runner for real OpenRouter free models.

This tool is intentionally pragmatic: it validates a curated set of
synthetic ICU tasks that are common in critical-care research and that
have a realistic chance of reaching publication-figure/manuscript
artifacts under a free hosted model.

The suite prefers:
* association studies
* prediction-model workflows
* trajectory-clustering workflows

More ambitious families (survival, causal inference, RL, multimodal)
remain protocol-first in the current runtime and should be validated in
separate suites rather than being mixed into this overnight runtime
validation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _bootstrap_imports() -> Path:
    import sys

    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


@dataclass(frozen=True)
class ValidationTask:
    key: str
    title: str
    family: str
    difficulty: str
    question: str
    cohort_factory: Callable[[int], Any]
    target_outcome: str = "death"
    notes: str = ""
    preferred_methods: str = ""
    evaluation_focus: str = ""
    must_have_outputs: str = ""


@dataclass
class AttemptRecord:
    task_key: str
    attempt: int
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"
    elapsed_seconds: Optional[float] = None
    run_id: Optional[str] = None
    run_dir: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ProgressState:
    generated_at: str
    provider: str
    model: str
    out_root: str
    tasks: List[str]
    max_retries: int
    sleep_seconds: int
    request_timeout: float
    attempts: List[AttemptRecord] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _progress_payload(state: ProgressState) -> Dict[str, Any]:
    return {
        "generated_at": state.generated_at,
        "provider": state.provider,
        "model": state.model,
        "out_root": state.out_root,
        "tasks": state.tasks,
        "max_retries": state.max_retries,
        "sleep_seconds": state.sleep_seconds,
        "request_timeout": state.request_timeout,
        "attempts": [asdict(x) for x in state.attempts],
    }


def _publication_figure_candidates(run_dir: Path) -> List[Path]:
    candidates = sorted(
        p
        for p in run_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".png", ".svg", ".pdf", ".tiff", ".tif"}
        and (
            "publication_figure" in p.name
            or "figure_contract" in p.name
            or "claim_first" in p.name
        )
    )
    return candidates


def _has_publication_figure_step(summary: Dict[str, Any]) -> bool:
    statuses = summary.get("per_step_statuses_detailed", []) or []
    return any(
        "publication_figure" in str(record.get("step_id", ""))
        and str(record.get("status", "")).lower() == "ok"
        for record in statuses
    )


def _manifest_summary(manifest_path: Path) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_step_records = manifest.get("per_step_records", []) or []
    statuses = [
        str(rec.get("status") or "") for rec in per_step_records if rec.get("step_id")
    ]
    evidence = manifest.get("evidence", []) or []
    generation_modes = sorted(
        {
            str(rec.get("generation_mode"))
            for rec in evidence
            if rec.get("generation_mode")
        }
    )
    findings = manifest.get("findings", []) or []
    warning_count = sum(
        1 for finding in findings if finding.get("severity") == "warning"
    )
    error_count = sum(1 for finding in findings if finding.get("severity") == "error")
    return {
        "run_id": manifest.get("run_id"),
        "used_mock_llm": bool(manifest.get("used_mock_llm")),
        "llm_signature": manifest.get("llm_signature"),
        "prompt_pack_version": manifest.get("prompt_pack_version"),
        "step_statuses": statuses,
        "per_step_statuses_detailed": [
            {"step_id": rec.get("step_id"), "status": rec.get("status")}
            for rec in per_step_records
            if rec.get("step_id")
        ],
        "evidence_count": len(evidence),
        "generation_modes": generation_modes,
        "warning_count": warning_count,
        "error_count": error_count,
        "manuscript_path": manifest.get("manuscript_path"),
        "report_path": manifest.get("report_path"),
    }


def _task_success(run_dir: Path) -> tuple[bool, Dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return False, {"reason": "manifest_missing"}
    summary = _manifest_summary(manifest_path)
    figure_candidates = _publication_figure_candidates(run_dir)
    manuscript_bound = run_dir / "manuscript_scaffold_bound.md"
    runtime_success = (
        summary["error_count"] == 0
        and manuscript_bound.exists()
        and "ok" in summary["step_statuses"]
    )
    publication_ready = any(
        "publication_figure_generation" in p.as_posix()
        or "publication_figure" in p.name
        or "claim_first" in p.name
        for p in figure_candidates
    ) or _has_publication_figure_step(summary)
    strict_success = (
        runtime_success
        and publication_ready
        and manuscript_bound.exists()
        and not summary["used_mock_llm"]
        and "fallback" not in summary["generation_modes"]
    )
    summary["publication_figures"] = [str(p) for p in figure_candidates]
    summary["bound_manuscript_exists"] = manuscript_bound.exists()
    summary["runtime_success"] = runtime_success
    summary["strict_success"] = strict_success
    summary["publication_ready"] = publication_ready
    return strict_success, summary


def _default_tasks() -> List[ValidationTask]:
    from tests.support.benchmark_cases.analysis_items import (  # type: ignore
        _hepatorenal_missingness_cohort,
        _rich_multisystem_cohort,
        _shock_discordance_cohort,
        _zero_artifact_cohort,
    )

    return [
        ValidationTask(
            key="association_basic_sofa_mortality",
            title="Basic SOFA-2 mortality association",
            family="association_study",
            difficulty="basic",
            question=(
                "Analyze whether admission SOFA-2 is associated with ICU mortality; "
                "include cohort summary, outcome incidence, explicit missingness audit, "
                "multivariable logistic regression, odds ratios with 95% confidence intervals, "
                "and a publication-ready multi-panel figure."
            ),
            cohort_factory=_rich_multisystem_cohort,
            notes="Simplest end-to-end association study smoke test.",
            preferred_methods=(
                "Multivariable logistic regression with odds ratios, confidence intervals, "
                "and explicit missingness reporting."
            ),
            evaluation_focus="effect estimate completeness, missingness handling, publication figure",
            must_have_outputs="primary_association table, missingness audit, publication-ready figure, bound manuscript",
        ),
        ValidationTask(
            key="association_missingness_hepatobiliary",
            title="Association under hepatobiliary missingness",
            family="association_study",
            difficulty="intermediate",
            question=(
                "Before claiming a mortality association, determine whether liver-related "
                "and vasopressor missingness materially limit interpretation; then fit the "
                "most defensible adjusted association model and generate a publication-ready figure."
            ),
            cohort_factory=_hepatorenal_missingness_cohort,
            notes="Association task with structured missingness and clinical caveats.",
            preferred_methods=(
                "Adjusted logistic regression with explicit missingness audit and sensitivity-minded interpretation."
            ),
            evaluation_focus="clinical caveats, missingness robustness, figure generation",
            must_have_outputs="missingness summary, primary_association table, publication-ready figure, bound manuscript",
        ),
        ValidationTask(
            key="prediction_basic_mortality",
            title="Basic ICU mortality prediction",
            family="prediction_model",
            difficulty="basic",
            question=(
                "Build an ICU mortality prediction model using age, sex, SOFA-2, lactate, "
                "mean arterial pressure, vasopressor use, and creatinine; use an explicit "
                "train/test split, report AUROC, Brier score, calibration, key coefficients, "
                "and generate a publication-ready multi-panel figure."
            ),
            cohort_factory=_rich_multisystem_cohort,
            notes="Current best-path end-to-end prediction smoke test.",
            preferred_methods=(
                "Logistic regression with explicit train/test split, AUROC, Brier score, and calibration assessment."
            ),
            evaluation_focus="split integrity, held-out discrimination, calibration, publication figure",
            must_have_outputs="split definition, held-out metrics, publication-ready figure, bound manuscript",
        ),
        ValidationTask(
            key="prediction_missingness_advanced",
            title="Prediction under structured missingness",
            family="prediction_model",
            difficulty="advanced",
            question=(
                "Build an ICU mortality prediction model using age, sex, SOFA-2, lactate, "
                "bilirubin, vasopressor use, creatinine, and mean arterial pressure; make "
                "missingness handling explicit, report AUROC, Brier score, calibration, and "
                "generate a publication-ready multi-panel figure."
            ),
            cohort_factory=_hepatorenal_missingness_cohort,
            notes="Harder prediction task with liver/vasopressor missingness.",
            preferred_methods=(
                "Prediction model with explicit missingness handling, held-out evaluation, and calibration outputs."
            ),
            evaluation_focus="missingness strategy, held-out metrics, publication figure",
            must_have_outputs="split definition, held-out metrics, publication-ready figure, bound manuscript",
        ),
        ValidationTask(
            key="clustering_basic_phenotypes",
            title="Basic physiologic phenotype clustering",
            family="trajectory_clustering",
            difficulty="basic",
            question=(
                "Cluster ICU patients using SOFA-2, lactate, mean arterial pressure, heart rate, "
                "creatinine, and vasopressor use; summarise cluster size, physiologic profiles, "
                "mortality differences, and generate a publication-ready clustering figure."
            ),
            cohort_factory=_rich_multisystem_cohort,
            notes="Current best-path end-to-end clustering smoke test.",
            preferred_methods=(
                "K-means or hierarchical clustering with explicit cluster profiles, size summaries, and mortality comparisons."
            ),
            evaluation_focus="cluster interpretability, cluster mortality summaries, publication figure",
            must_have_outputs="cluster summaries, phenotype figure, bound manuscript",
        ),
        ValidationTask(
            key="prediction_zero_artifact_intermediate",
            title="Prediction with SOFA zero-artifact stress",
            family="prediction_model",
            difficulty="intermediate",
            question=(
                "Build an ICU mortality prediction model using age, sex, SOFA-2, lactate, "
                "mean arterial pressure, vasopressor use, and bilirubin; inspect whether "
                "SOFA-2 zero strata behave unusually, report train/test performance and "
                "calibration, and generate a publication-ready multi-panel figure."
            ),
            cohort_factory=_zero_artifact_cohort,
            notes="Prediction stress test with score-zero anomaly structure.",
            preferred_methods=(
                "Prediction model with explicit audit of zero-score artefacts, held-out AUROC, and calibration."
            ),
            evaluation_focus="zero-score audit, held-out metrics, publication figure",
            must_have_outputs="zero-stratum audit, held-out metrics, publication-ready figure, bound manuscript",
        ),
        ValidationTask(
            key="clustering_shock_advanced",
            title="Advanced shock-phenotype clustering",
            family="trajectory_clustering",
            difficulty="advanced",
            question=(
                "Cluster ICU patients using SOFA-2, lactate, bilirubin, mean arterial pressure, "
                "heart rate, creatinine, and vasopressor use to identify shock-related physiologic "
                "phenotypes; summarise cluster stability, cluster mortality differences, and generate "
                "a publication-ready clustering figure."
            ),
            cohort_factory=_shock_discordance_cohort,
            notes="Harder clustering task with treatment and missingness structure.",
            preferred_methods=(
                "Clustering workflow with explicit cluster profiles, mortality comparison, and publication figure."
            ),
            evaluation_focus="cluster stability narrative, cluster mortality differences, publication figure",
            must_have_outputs="cluster summaries, phenotype figure, bound manuscript",
        ),
    ]


def _make_llm(*, provider: str, model: str, request_timeout: float):
    from easyicu.research_agent.providers.factory import build_provider_client
    from easyicu.research_agent.providers.llm import OpenAIClient

    if provider != "openrouter":
        raise SystemExit("This runner currently supports only --provider openrouter.")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is required for the OpenRouter validation run."
        )
    return build_provider_client(
        provider="openrouter",
        model=model,
        request_timeout=float(request_timeout),
        title="EasyICU full-flow validation",
        client_cls=OpenAIClient,
    )


def _run_task(
    *,
    task: ValidationTask,
    seed: int,
    task_root: Path,
    provider: str,
    model: str,
    request_timeout: float,
) -> Dict[str, Any]:
    import pandas as pd
    from easyicu.research_agent import ResearchAgentPipeline  # type: ignore

    llm = _make_llm(provider=provider, model=model, request_timeout=request_timeout)
    task_root.mkdir(parents=True, exist_ok=True)
    cohort = task.cohort_factory(seed)
    if not isinstance(cohort, pd.DataFrame):
        cohort = pd.DataFrame(cohort)

    pipeline = ResearchAgentPipeline(
        workdir=task_root,
        llm=llm,
        timeout_seconds=float(request_timeout),
        enable_literature=False,
        enable_visual_qa=True,
        enable_memory=False,
        enable_latex=True,
        enable_probe_step=True,
        enable_replanning=False,
        max_code_repair_attempts=1,
        enable_deterministic_code_fallback=False,
        enable_deterministic_planner_fallback=False,
    )
    result = pipeline.run(
        question=task.question,
        cohort=cohort,
        cohort_name=f"validation_{task.key}",
        database="synthetic",
        target_outcome=task.target_outcome,
        notes=(
            "Full-flow validation task for current real-model runtime. "
            "This is a synthetic-cohort execution test, not an external clinical claim."
        ),
        user_preferences={
            "inferred_analysis_family": task.family,
            "preferred_methods": task.preferred_methods,
            "evaluation_focus": task.evaluation_focus,
            "must_have_outputs": task.must_have_outputs
            or "publication-ready figure, bound manuscript, manifest",
            "extra_notes": task.notes,
        },
    )
    run_dir = Path(result.workdir)
    success, summary = _task_success(run_dir)
    return {
        "task_key": task.key,
        "title": task.title,
        "family": task.family,
        "difficulty": task.difficulty,
        "question": task.question,
        "run_id": result.run_id,
        "run_dir": str(run_dir),
        "success": success,
        "summary": summary,
    }


def _render_markdown(
    results: List[Dict[str, Any]], *, provider: str, model: str
) -> str:
    lines = [
        "# OpenRouter full-flow validation",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        f"_Provider: `{provider}`_",
        f"_Model: `{model}`_",
        "",
        "This suite validates the **currently executable** EASYICU full-flow task families",
        "(prediction and clustering) using a real OpenRouter free model. It is a runtime",
        "stability check on synthetic ICU-like cohorts, not an external scientific benchmark.",
        "",
        "| Task | Family | Difficulty | Success | Errors | Warnings | Publication figure | Generation modes | Run dir |",
        "|---|---|---|:-:|---:|---:|---|---|---|",
    ]
    for item in results:
        summary = item["summary"]
        pub_fig = summary.get("publication_figures", [])
        pub_fig_label = Path(pub_fig[0]).name if pub_fig else "—"
        lines.append(
            f"| `{item['task_key']}` | `{item['family']}` | `{item['difficulty']}` | "
            f"{'✅' if summary.get('strict_success') else '❌'} | "
            f"{summary.get('error_count', 0)} | "
            f"{summary.get('warning_count', 0)} | "
            f"{pub_fig_label} | "
            f"`{', '.join(summary.get('generation_modes', []))}` | "
            f"`{item['run_dir']}` |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append(
        "- Validation scope here is deliberately limited to task families that can currently reach publication figures end to end."
    )
    lines.append(
        "- Survival, causal inference, RL, multimodal, and related families remain protocol-first and should be reported separately."
    )
    lines.append(
        "- A strict success requires a real hosted model run, a bound manuscript, and publication-figure evidence with no `fallback` generation mode."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    _bootstrap_imports()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openrouter"], default="openrouter")
    parser.add_argument("--model", default="openrouter/free")
    parser.add_argument(
        "--items", nargs="+", default=None, help="Subset of validation task keys."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=int, default=45)
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()

    all_tasks = _default_tasks()
    selected = all_tasks
    if args.items:
        wanted = set(args.items)
        selected = [task for task in all_tasks if task.key in wanted]
        unknown = sorted(wanted - {task.key for task in all_tasks})
        if unknown:
            raise SystemExit(f"Unknown validation task keys: {unknown}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_root = Path(
        args.out_root or f"./research_output/validation/openrouter_fullflow_{timestamp}"
    ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "runner.log"
    progress_path = out_root / "progress.json"
    plan_path = out_root / "plan.json"

    state = ProgressState(
        generated_at=_utc_now(),
        provider=args.provider,
        model=args.model,
        out_root=str(out_root),
        tasks=[task.key for task in selected],
        max_retries=int(args.max_retries),
        sleep_seconds=int(args.sleep_seconds),
        request_timeout=float(args.request_timeout),
    )
    _write_json(
        plan_path,
        {
            "provider": args.provider,
            "model": args.model,
            "seed": args.seed,
            "request_timeout": args.request_timeout,
            "max_retries": args.max_retries,
            "sleep_seconds": args.sleep_seconds,
            "tasks": [
                task.__dict__ | {"cohort_factory": task.cohort_factory.__name__}
                for task in selected
            ],
        },
    )

    results: List[Dict[str, Any]] = []
    for task in selected:
        success = False
        last_result: Optional[Dict[str, Any]] = None
        for attempt in range(1, int(args.max_retries) + 1):
            record = AttemptRecord(
                task_key=task.key, attempt=attempt, started_at=_utc_now()
            )
            state.attempts.append(record)
            _write_json(progress_path, _progress_payload(state))
            _append_log(
                log_path, f"[{record.started_at}] START {task.key} attempt={attempt}"
            )
            started = time.monotonic()
            try:
                task_root = out_root / task.key
                last_result = _run_task(
                    task=task,
                    seed=int(args.seed),
                    task_root=task_root,
                    provider=args.provider,
                    model=args.model,
                    request_timeout=float(args.request_timeout),
                )
                success = bool(last_result["success"])
                record.run_id = last_result.get("run_id")
                record.run_dir = last_result.get("run_dir")
                if success:
                    record.status = "ok"
                elif last_result.get("summary", {}).get("runtime_success"):
                    record.status = "runtime_complete_but_not_strict"
                else:
                    record.status = "completed_with_findings"
            except Exception as exc:
                last_result = {
                    "task_key": task.key,
                    "title": task.title,
                    "family": task.family,
                    "difficulty": task.difficulty,
                    "success": False,
                    "summary": {"reason": "exception"},
                    "error": f"{type(exc).__name__}: {exc}",
                }
                record.error = last_result["error"]
                record.status = "error"
            record.completed_at = _utc_now()
            record.elapsed_seconds = round(time.monotonic() - started, 2)
            _write_json(progress_path, _progress_payload(state))
            _append_log(
                log_path,
                f"[{record.completed_at}] END {task.key} attempt={attempt} "
                f"status={record.status} elapsed={record.elapsed_seconds}s "
                f"run_id={record.run_id or '—'} error={record.error or '—'}",
            )
            if success or (
                last_result is not None
                and last_result.get("summary", {}).get("runtime_success")
            ):
                break
            if attempt < int(args.max_retries):
                time.sleep(int(args.sleep_seconds))
        if last_result is not None:
            results.append(last_result)

    summary = {
        "generated_at": _utc_now(),
        "provider": args.provider,
        "model": args.model,
        "results": results,
        "n_tasks": len(results),
        "n_success": sum(1 for result in results if result.get("success")),
        "n_failed": sum(1 for result in results if not result.get("success")),
    }
    _write_json(out_root / "validation_results.json", summary)
    (out_root / "validation_results.md").write_text(
        _render_markdown(results, provider=args.provider, model=args.model),
        encoding="utf-8",
    )
    _append_log(
        log_path,
        f"[{_utc_now()}] COMPLETE success={summary['n_success']}/{summary['n_tasks']}",
    )
    print(out_root / "validation_results.json")
    print(out_root / "validation_results.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

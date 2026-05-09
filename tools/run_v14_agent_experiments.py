#!/usr/bin/env python3
"""Run the EasyICU v14 real-cohort research-agent experiments.

This runner is intentionally separate from the older synthetic benchmark
runner. It consumes the real task cohorts produced by
``tools/build_v14_task_cohorts.py`` and records every task/arm/model attempt
in a restartable, paper-facing output bundle.

Typical usage::

    python tools/run_v14_agent_experiments.py \
        --export-dir /Users/haibo/Documents/GitHub/miiv_20260420 \
        --model z-ai/glm-4.5-air:free \
        --items t04_lactate_mortality_association \
        --arms aware naive

The OpenRouter API key must be supplied through the environment; this script
never writes provider credentials into output files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import queue as queue_module
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REQUIRED_EVIDENCE_KINDS = {"code", "log", "table", "figure", "statistic"}
AGGREGATION_VERSION = "v14_a_lite_20260509"
DEFAULT_FREE_MODELS = [
    "z-ai/glm-4.5-air:free",
    "openai/gpt-oss-120b:free",
    "deepseek/deepseek-chat-v3-0324:free",
]


def _bootstrap_imports() -> Path:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    src_path = repo_root / "src"
    for candidate in (src_path, repo_root):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)
    return repo_root


@dataclass(frozen=True)
class V14Task:
    key: str
    title: str
    family: str
    difficulty: str
    cohort_file: str
    question: str
    target_outcome: Optional[str] = "death"
    primary_predictor: Optional[str] = None
    expected_outputs: List[str] = field(default_factory=list)
    expected_metrics: List[str] = field(default_factory=list)
    required_artifacts: List[str] = field(default_factory=list)
    allowed_warnings: List[str] = field(default_factory=list)
    fatal_conditions: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttemptRecord:
    task_key: str
    arm: str
    model: str
    attempt: int
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"
    pipeline_status: str = "running"
    acceptance_status: str = "running"
    elapsed_seconds: Optional[float] = None
    run_id: Optional[str] = None
    run_dir: Optional[str] = None
    failure_class: Optional[str] = None
    error: Optional[str] = None
    heartbeat_path: Optional[str] = None
    last_heartbeat_at: Optional[str] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    safe = []
    for ch in value.strip():
        safe.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(safe).strip("._-") or "value"


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _git_status(repo_root: Path) -> List[str]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [line for line in output.splitlines() if line.strip()]


def _git_diff_hash(repo_root: Path) -> str:
    try:
        diff = subprocess.check_output(
            ["git", "-C", str(repo_root), "diff", "--no-ext-diff"],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "unknown"
    return hashlib.sha256(diff).hexdigest()[:16]


def _provider_base_url(provider: str) -> Optional[str]:
    if provider == "openrouter":
        return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if provider == "openai":
        return os.environ.get("OPENAI_BASE_URL")
    return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_task_specs_from_builder() -> List[V14Task]:
    from tools.build_v14_task_cohorts import task_specs  # type: ignore

    metadata = {
        "t01_table_one_descriptive": {
            "family": "descriptive",
            "difficulty": "basic",
            "question": (
                "Create a descriptive Table 1 for adult first ICU stays using age, sex, "
                "ICU length of stay, early SOFA-2, MAP, lactate, vasopressor exposure, "
                "and mortality. Report medians/IQRs, counts/percentages, missingness, "
                "and a concise publication-ready figure."
            ),
            "primary_predictor": None,
            "expected_metrics": ["n_rows", "mortality_rate", "missingness"],
            "required_artifacts": ["manifest", "bound_manuscript", "step_summary", "table", "statistic", "figure"],
        },
        "t02_outcome_incidence_strata": {
            "family": "descriptive_stratified",
            "difficulty": "basic",
            "question": (
                "Estimate mortality incidence across SOFA-2 severity strata. Report the "
                "denominator, deaths, mortality percentage, Wilson-style confidence "
                "intervals when feasible, and a publication-ready mortality-by-stratum "
                "figure. Do not fit a causal model."
            ),
            "primary_predictor": "sofa2_max_24h",
            "expected_metrics": ["mortality_rate", "stratum_table"],
            "required_artifacts": ["manifest", "bound_manuscript", "step_summary", "table", "statistic", "figure"],
        },
        "t03_severity_score_correlation": {
            "family": "correlation",
            "difficulty": "intermediate",
            "question": (
                "Quantify Spearman correlations between total SOFA-2 and each available "
                "SOFA-2 component. Identify component-total collinearity, missing "
                "components, and produce a heatmap or equivalent publication figure."
            ),
            "target_outcome": None,
            "primary_predictor": "sofa2_max_24h",
            "expected_metrics": ["spearman_rho", "missingness"],
            "required_artifacts": ["manifest", "bound_manuscript", "step_summary", "table", "statistic", "figure"],
        },
        "t04_lactate_mortality_association": {
            "family": "association_study",
            "difficulty": "intermediate",
            "question": (
                "Analyze whether first-24h maximum lactate is associated with in-hospital "
                "mortality. Include a cohort summary, missingness audit, adjusted logistic "
                "regression with odds ratio and confidence interval, and sensitivity notes "
                "for lactate not measured. Produce a publication-ready figure showing the "
                "lactate association or sensitivity results."
            ),
            "primary_predictor": "lactate_max_24h",
            "expected_metrics": ["primary_or", "missingness"],
            "required_artifacts": ["manifest", "bound_manuscript", "step_summary", "table", "statistic", "figure"],
        },
        "t05_kdigo_renal_sensitivity": {
            "family": "association_sensitivity",
            "difficulty": "intermediate",
            "question": (
                "Evaluate whether first-24h KDIGO stage is associated with mortality, "
                "adjusting for age, sex, renal SOFA component, and vasopressor exposure "
                "when available. Compare complete-case and reduced-variable sensitivity "
                "models, and produce a figure summarizing effect estimates across "
                "sensitivity strategies."
            ),
            "primary_predictor": "kdigo_stage_max_24h",
            "expected_metrics": ["primary_or", "complete_case_n"],
            "required_artifacts": ["manifest", "bound_manuscript", "step_summary", "table", "statistic", "figure"],
        },
        "t06_shock_phenotype_clustering": {
            "family": "clustering",
            "difficulty": "advanced",
            "question": (
                "Cluster ICU stays using first-24h shock physiology: lactate, MAP, "
                "vasopressor exposure, heart rate, and systolic blood pressure. Report "
                "cluster size, physiologic profiles, silhouette or stability metric when "
                "available, post-hoc mortality by cluster, and a clustering figure. Keep "
                "cluster-label generation, cluster characteristics, mortality by cluster, "
                "silhouette, and figure generation in one self-contained executable "
                "clustering step; do not create later steps that need to read cluster labels "
                "from prior step outputs."
            ),
            "primary_predictor": None,
            "expected_metrics": [
                "silhouette_or_cluster_count",
                "cluster_characteristics",
                "cluster_mortality",
            ],
            "required_artifacts": [
                "manifest",
                "bound_manuscript",
                "step_summary",
                "table",
                "statistic",
                "figure",
            ],
        },
        "t07_mortality_prediction_auroc": {
            "family": "prediction_model",
            "difficulty": "advanced",
            "question": (
                "Build a mortality prediction workflow using age, sex, SOFA-2 components, "
                "lactate, MAP, and vasopressor exposure. Use an explicit split or 5-fold "
                "cross-validation, report AUROC, calibration or Brier score, baseline "
                "prevalence comparison, and a publication-ready discrimination/calibration "
                "figure. Keep model training, validation metrics, baseline prevalence, "
                "and the figure in one self-contained executable analysis step; do not "
                "create later steps that need to read prior step outputs."
            ),
            "primary_predictor": None,
            "expected_metrics": [
                "auroc",
                "brier_score",
                "calibration",
                "baseline_prevalence",
                "split_or_cv",
            ],
            "required_artifacts": [
                "manifest",
                "bound_manuscript",
                "step_summary",
                "table",
                "statistic",
                "figure",
            ],
        },
        "t08_vaso_selection_bias_audit": {
            "family": "bias_audit",
            "difficulty": "advanced",
            "question": (
                "Estimate the association between first-24h vasopressor exposure and "
                "mortality while explicitly auditing treatment-selection bias, severity "
                "confounding, and missingness. Avoid causal language and record any "
                "clinical-constraint warning."
            ),
            "primary_predictor": "vaso_any_24h",
            "expected_metrics": ["primary_or", "selection_bias_warning"],
            "required_artifacts": ["manifest", "bound_manuscript", "step_summary", "table", "statistic"],
        },
        "t09_sofa_zero_artefact_audit": {
            "family": "data_quality_audit",
            "difficulty": "advanced",
            "question": (
                "Audit whether SOFA-2 equal to zero co-occurs with high lactate, low MAP, "
                "vasopressor exposure, or mortality. Treat this as a data-quality and "
                "missing-component problem before any modelling."
            ),
            "primary_predictor": "sofa2_max_24h",
            "expected_metrics": ["sofa_zero_count", "guardrail_warning"],
            "required_artifacts": ["manifest", "bound_manuscript", "step_summary", "table", "statistic"],
        },
        "t10_complete_case_robustness": {
            "family": "robustness",
            "difficulty": "advanced",
            "question": (
                "Fit mortality association models for first-24h maximum lactate and "
                "compare complete-case, missing-indicator, and reduced-variable "
                "strategies. Report the sample size, event rate, lactate odds ratio "
                "stability, missingness profile, and interpretation limits for each "
                "missing-data strategy, with a publication-ready robustness figure. "
                "Keep complete-case, missing-indicator, reduced-variable modelling, "
                "effect extraction, summary table, and robustness figure generation in "
                "one self-contained executable robustness step; do not create later "
                "steps that depend on model objects or intermediate files from prior "
                "steps."
            ),
            "primary_predictor": "lactate_max_24h",
            "expected_metrics": ["primary_or", "complete_case_n", "missingness"],
            "required_artifacts": ["manifest", "bound_manuscript", "step_summary", "table", "statistic", "figure"],
        },
    }

    tasks: List[V14Task] = []
    for spec in task_specs():
        meta = metadata.get(spec.key)
        if meta is None:
            raise RuntimeError(f"No v14 metadata registered for task {spec.key!r}")
        tasks.append(
            V14Task(
                key=spec.key,
                title=spec.description,
                family=str(meta["family"]),
                difficulty=str(meta["difficulty"]),
                cohort_file=f"{spec.key}.parquet",
                question=str(meta["question"]),
                target_outcome=meta.get("target_outcome", "death"),
                primary_predictor=meta.get("primary_predictor"),
                expected_outputs=[
                    "manifest",
                    "bound_manuscript",
                    "step_summary",
                    "table",
                    "statistic",
                    "figure",
                ],
                expected_metrics=list(meta.get("expected_metrics", [])),
                required_artifacts=list(meta.get("required_artifacts", [])),
                allowed_warnings=list(meta.get("allowed_warnings", [])),
                fatal_conditions=list(meta.get("fatal_conditions", [])),
                user_preferences={
                    "inferred_analysis_family": str(meta["family"]),
                    "preferred_methods": str(meta["question"]),
                    "evaluation_focus": ", ".join(list(meta.get("expected_metrics", []))),
                    "must_have_outputs": (
                        "table, log, statistic, figure, manifest, bound manuscript"
                    ),
                    "data_constraints": (
                        "Use the supplied real EasyICU export cohort only; do not "
                        "invent external rows or outcomes."
                    ),
                    "extra_notes": (
                        "This is a v14 EasyICU agent experiment on a real exported "
                        "cohort. Preserve evidence provenance and avoid unsupported "
                        "clinical causal claims."
                    ),
                },
            )
        )
    return tasks


def _build_cohorts_if_needed(
    *, repo_root: Path, export_dir: Path, cohort_dir: Path, force: bool, log_path: Path
) -> None:
    summary_path = cohort_dir / "v14_task_cohorts_summary.json"
    if summary_path.exists() and not force:
        return
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "build_v14_task_cohorts.py"),
        "--export-dir",
        str(export_dir),
        "--out-dir",
        str(cohort_dir),
    ]
    _append_log(log_path, f"$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    _append_log(log_path, proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"v14 cohort build failed with exit code {proc.returncode}")


def _make_llm(*, provider: str, model: str, request_timeout: float):
    from easyicu.research_agent import MockLLMClient, OpenAIClient  # type: ignore

    if provider == "mock":
        return MockLLMClient()
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENROUTER_API_KEY is required for --provider openrouter."
            )
        extra_body = {"reasoning": {"effort": "none", "exclude": True}}
        if "gpt-oss" in model.lower():
            # OpenRouter's GPT-OSS endpoints require reasoning to remain enabled.
            extra_body = {"reasoning": {"effort": "low"}}
        return OpenAIClient(
            model=model,
            api_key=api_key,
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            request_timeout=float(request_timeout),
            extra_headers={
                "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
                "X-Title": "EasyICU v14 agent experiments",
            },
            extra_body=extra_body,
        )
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required for --provider openai.")
        return OpenAIClient(
            model=model,
            api_key=api_key,
            request_timeout=float(request_timeout),
        )
    raise SystemExit(f"Unsupported provider: {provider}")


def _validate_provider_env(provider: str) -> None:
    if provider == "openrouter" and not (
        os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    ):
        raise SystemExit("OPENROUTER_API_KEY is required for --provider openrouter.")
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for --provider openai.")


def _latest_run_dir(arm_root: Path) -> Optional[Path]:
    runs = sorted(
        (path for path in arm_root.glob("run_*") if (path / "manifest.json").exists()),
        key=lambda path: path.name,
        reverse=True,
    )
    return runs[0] if runs else None


def _rank_reusable_record(record: Dict[str, Any]) -> Tuple[int, int, int, int, int, str]:
    metrics = record.get("metrics") or {}
    acceptance_order = {
        "clean_ok": 0,
        "partial": 1,
        "failed": 2,
        "stalled": 3,
        "missing": 4,
    }
    return (
        acceptance_order.get(str(record.get("acceptance_status")), 5),
        int(metrics.get("failed_step_count") or 0),
        int(metrics.get("evidence_missing_count") or 0),
        int(metrics.get("error_count") or 0),
        len(_missing_expected_metrics(metrics)) + len(_missing_required_artifacts(metrics)),
        str(record.get("run_id") or ""),
    )


def _best_reusable_run_record(
    *, task: V14Task, arm: str, model: str, cohort_path: Path, out_root: Path
) -> Optional[Dict[str, Any]]:
    arm_root = out_root / _slugify(model) / task.key / arm
    runs = sorted(
        (path for path in arm_root.glob("run_*") if (path / "manifest.json").exists()),
        key=lambda path: path.name,
    )
    candidates: List[Dict[str, Any]] = []
    for run_dir in runs:
        metrics = _extract_metrics(run_dir, task)
        failure_class = _classify_failure(None, metrics, task)
        candidates.append(
            _with_status_fields({
                "task": asdict(task),
                "arm": arm,
                "provider": "reused",
                "model": model,
                "cohort_path": str(cohort_path),
                "cohort_sha256": _sha256_file(cohort_path),
                "run_dir": str(run_dir),
                "run_id": metrics.get("run_id"),
                "failure_class": failure_class,
                "metrics": metrics,
            })
        )
    if not candidates:
        return None
    return sorted(candidates, key=_rank_reusable_record)[0]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _step_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    summaries = []
    for path in sorted(run_dir.rglob("step_summary.json")):
        data = _read_json(path)
        if data:
            data["_path"] = str(path)
            summaries.append(data)
    return summaries


def _probe_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    summaries = []
    for path in sorted(run_dir.rglob("probe_summary.json")):
        data = _read_json(path)
        if data:
            data["_path"] = str(path)
            summaries.append(data)
    return summaries


def _nested_values(obj: Any, *, keys: Iterable[str]) -> List[Any]:
    wanted = {key.lower() for key in keys}
    found: List[Any] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in wanted:
                found.append(value)
            found.extend(_nested_values(value, keys=wanted))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_nested_values(item, keys=wanted))
    return found


def _first_float(values: Iterable[Any]) -> Optional[float]:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, (list, tuple)):
            nested = _first_float(value)
            if nested is not None:
                return nested
        if isinstance(value, dict):
            nested = _first_float(value.values())
            if nested is not None:
                return nested
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _nested_key_values(obj: Any, *, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield child, value
            yield from _nested_key_values(value, prefix=child)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            child = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _nested_key_values(value, prefix=child)


def _candidate_key_values(obj: Any, *, keys: Sequence[str]) -> List[Tuple[str, Any]]:
    wanted = {str(key).lower() for key in keys}
    found: List[Tuple[str, Any]] = []
    for path, value in _nested_key_values(obj):
        leaf = path.rsplit(".", 1)[-1].lower()
        lowered = path.lower()
        if leaf in wanted or lowered in wanted:
            found.append((path, value))
    return found


def _first_float_with_source(candidates: Iterable[Tuple[str, Any]]) -> Tuple[Optional[float], Optional[str]]:
    for source, raw in candidates:
        if isinstance(raw, (list, tuple)):
            nested = _first_float(raw)
        elif isinstance(raw, dict):
            nested = _first_float(raw.values())
        else:
            nested = _first_float([raw])
        if nested is not None:
            return nested, source
    return None, None


def _normalized_source_path(source: Optional[str]) -> Optional[str]:
    if source is None:
        return None
    return re.sub(r"^\[\d+\]\.", "", source)


def _first_float_by_priority(obj: Any, *, keys: Sequence[str]) -> Tuple[Optional[float], Optional[str]]:
    for key in keys:
        value, source = _first_float_with_source(_candidate_key_values(obj, keys=[key]))
        if value is not None:
            return value, _normalized_source_path(source)
    return None, None


def _rate_like_values(summaries: Sequence[Dict[str, Any]]) -> List[float]:
    values = _nested_values(
        summaries,
        keys=[
            "event_rate",
            "statistic:event_rate",
            "outcome_rate",
            "statistic:outcome_rate",
            "mortality_rate",
            "statistic:mortality_rate",
            "death_rate",
            "statistic:death_rate",
            "baseline_prevalence",
            "statistic:baseline_prevalence",
        ],
    )
    rates: List[float] = []
    for value in values:
        numeric = _first_float([value])
        if numeric is not None:
            rates.append(numeric)
    return rates


def _is_rate_alias(value: float, summaries: Sequence[Dict[str, Any]]) -> bool:
    for rate in _rate_like_values(summaries):
        if abs(value - rate) < 1e-9:
            return True
        if abs(value - rate * 100.0) < 1e-9:
            return True
        if abs(value * 100.0 - rate) < 1e-9:
            return True
    return False


def _first_text_effect_estimate_with_source(
    summaries: Sequence[Dict[str, Any]]
) -> Tuple[Optional[float], Optional[str]]:
    blob = json.dumps(list(summaries), ensure_ascii=False, default=str)
    patterns = (
        r"\b(?:OR|odds\s+ratio)\b\s*(?:=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
        r"\b(?:adjusted\s+OR|adjusted\s+odds\s+ratio)\b\s*(?:=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, blob, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value, "text_pattern"
    return None, None


def _normalize_strategy_name(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _extract_robustness_effect_estimate(
    summaries: Sequence[Dict[str, Any]]
) -> Tuple[Optional[float], Optional[str]]:
    preferred_keys = [
        "statistic:lactate_or_complete_case",
        "lactate_or_complete_case",
        "statistic:primary_or_complete_case",
        "primary_or_complete_case",
        "statistic:cc_or_lactate",
        "cc_or_lactate",
        "complete_case_or",
        "statistic:complete_case_or",
        "cc_or",
        "statistic:cc_or",
        "lactate_or",
        "statistic:lactate_or",
    ]
    value, source = _first_float_by_priority(summaries, keys=preferred_keys)
    if value is not None:
        return value, source

    for summary in summaries:
        for path, raw in _candidate_key_values(summary, keys=["log:robustness_model_fits", "robustness_model_fits"]):
            if isinstance(raw, list):
                for idx, item in enumerate(raw):
                    if not isinstance(item, dict):
                        continue
                    if _normalize_strategy_name(item.get("strategy")) not in {"complete_case", "completecase"}:
                        continue
                    value, source = _first_float_with_source(
                        _candidate_key_values(
                            item,
                            keys=[
                                "or_estimate",
                                "statistic:or_estimate",
                                "lactate_or",
                                "statistic:lactate_or",
                                "estimate",
                                "statistic:estimate",
                            ],
                        )
                    )
                    if value is not None:
                        return value, f"{path}[{idx}].{source}"
            elif isinstance(raw, dict):
                strategies = raw.get("strategy")
                if isinstance(strategies, list):
                    or_values = None
                    for key in ("lactate_or", "or_estimate", "estimate", "primary_or"):
                        candidate = raw.get(key)
                        if isinstance(candidate, list):
                            or_values = candidate
                            break
                    if or_values is None:
                        continue
                    for idx, strategy in enumerate(strategies):
                        if _normalize_strategy_name(strategy) not in {"complete_case", "completecase"}:
                            continue
                        if idx >= len(or_values):
                            continue
                        value = _first_float([or_values[idx]])
                        if value is not None:
                            return value, f"{path}.{key}[{idx}]"

    fallback_keys = [
        "primary_or",
        "statistic:primary_or",
        "or",
        "odds_ratio",
        "statistic:odds_ratio",
        "or_estimate",
        "statistic:or_estimate",
        "estimate",
        "statistic:estimate",
    ]
    for key in fallback_keys:
        for source, raw in _candidate_key_values(summaries, keys=[key]):
            value = _first_float([raw])
            if value is None or _is_rate_alias(value, summaries):
                continue
            return value, _normalized_source_path(source)

    return _first_text_effect_estimate_with_source(summaries)


def _extract_bias_audit_effect_estimate(
    summaries: Sequence[Dict[str, Any]]
) -> Tuple[Optional[float], Optional[str]]:
    preferred_keys = [
        "adjusted_or",
        "statistic:adjusted_or",
        "or_estimate",
        "statistic:or_estimate",
        "estimate",
        "statistic:estimate",
        "primary_or",
        "statistic:primary_or",
        "or",
        "odds_ratio",
        "statistic:odds_ratio",
    ]
    value, source = _first_float_by_priority(summaries, keys=preferred_keys)
    if value is not None:
        return value, source
    return _first_text_effect_estimate_with_source(summaries)


def _first_effect_estimate(
    summaries: Sequence[Dict[str, Any]], *, task_family: Optional[str] = None
) -> Tuple[Optional[float], Optional[str]]:
    if task_family == "robustness":
        return _extract_robustness_effect_estimate(summaries)
    if task_family == "bias_audit":
        return _extract_bias_audit_effect_estimate(summaries)

    value, source = _first_float_by_priority(
        summaries,
        keys=[
            "primary_or",
            "statistic:primary_or",
            "or",
            "odds_ratio",
            "statistic:odds_ratio",
            "lactate_or",
            "lactate_max_24h_or",
            "adjusted_or",
            "statistic:adjusted_or",
            "or_estimate",
            "statistic:or_estimate",
            "estimate",
            "statistic:estimate",
        ],
    )
    if value is not None:
        return value, source
    for summary in summaries:
        for key, raw in _nested_key_values(summary):
            lowered = key.lower()
            if (
                lowered.endswith("_or")
                or lowered.endswith("_odds_ratio")
                or "_or_" in lowered
                or "or_lactate" in lowered
                or "odds_ratio" in lowered
            ):
                value = _first_float([raw])
                if value is not None:
                    return value, _normalized_source_path(key)
    return _first_text_effect_estimate_with_source(summaries)


def _extract_selection_bias_warning(
    manifest: Dict[str, Any],
    summaries: Sequence[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    explicit_warning_keys = [
        "selection_bias_warning",
        "statistic:selection_bias_warning",
        "warning:selection_bias",
        "clinical_constraint_warning",
    ]
    targeted_phrases = [
        "confounded by indication",
        "confounding by indication",
        "avoid causal treatment-effect language",
        "avoid causal treatment effect language",
        "avoid causal language",
        "avoid treatment-effect language",
        "avoid treatment effect language",
        "treatment-effect language",
        "treatment effect language",
    ]
    for source, raw in _candidate_key_values(summaries, keys=explicit_warning_keys):
        if isinstance(raw, bool):
            if raw:
                return True, _normalized_source_path(source)
            continue
        if raw is None:
            continue
        text = str(raw).lower()
        if any(phrase in text for phrase in targeted_phrases):
            return True, _normalized_source_path(source)

    findings = manifest.get("findings") or []
    for idx, finding in enumerate(findings):
        text = json.dumps(finding, ensure_ascii=False, default=str).lower()
        if any(phrase in text for phrase in targeted_phrases):
            return True, f"findings[{idx}]"

    for idx, summary in enumerate(summaries):
        text = json.dumps(summary, ensure_ascii=False, default=str).lower()
        if any(phrase in text for phrase in targeted_phrases):
            return True, f"summaries[{idx}]"
    return False, None


def _contains_text(manifest: Dict[str, Any], summaries: List[Dict[str, Any]], needle: str) -> bool:
    blob = json.dumps(
        {"manifest": manifest, "summaries": summaries},
        ensure_ascii=False,
        default=str,
    ).lower()
    return needle.lower() in blob


def _evidence_name_contains(evidence: Sequence[Dict[str, Any]], *needles: str) -> bool:
    haystack = json.dumps(list(evidence), ensure_ascii=False, default=str).lower()
    return all(needle.lower() in haystack for needle in needles)


def _extract_metrics(run_dir: Path, task: V14Task) -> Dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    summaries = _step_summaries(run_dir)
    metric_summaries = summaries + _probe_summaries(run_dir)
    bound = run_dir / "manuscript_scaffold_bound.md"
    evidence = manifest.get("evidence", []) or []
    findings = manifest.get("findings", []) or []
    kinds = {str(item.get("kind")) for item in evidence if item.get("kind")}
    per_step_records = manifest.get("per_step_records", []) or []
    step_statuses = [
        {"step_id": item.get("step_id"), "status": item.get("status")}
        for item in per_step_records
        if item.get("step_id")
    ]
    failed_steps = [
        item for item in step_statuses if str(item.get("status", "")).lower() not in {"ok", "complete", "completed"}
    ]
    llm_repaired_steps = [
        item
        for item in per_step_records
        if int(item.get("code_repair_attempts") or 0) > 0
    ]
    deterministically_repaired_steps = [
        item for item in per_step_records if item.get("runner_repair")
    ]
    manuscript_errors = [
        item
        for item in findings
        if item.get("severity") == "error"
        and str(item.get("validator", "")).lower() in {"critic_agent", "evidence_bound_writer"}
    ]
    non_manuscript_errors = [
        item
        for item in findings
        if item.get("severity") == "error"
        and str(item.get("validator", "")).lower() not in {"critic_agent", "evidence_bound_writer"}
    ]
    evidence_missing_count = (
        bound.read_text(encoding="utf-8").count("[evidence missing:")
        if bound.exists()
        else None
    )

    metrics: Dict[str, Any] = {
        "run_id": manifest.get("run_id"),
        "aggregation_version": AGGREGATION_VERSION,
        "used_mock_llm": bool(manifest.get("used_mock_llm")),
        "llm_signature": manifest.get("llm_signature"),
        "prompt_pack_version": manifest.get("prompt_pack_version"),
        "evidence_count": len(evidence),
        "evidence_kinds_seen": sorted(kinds),
        "evidence_kinds_missing": sorted(REQUIRED_EVIDENCE_KINDS - kinds),
        "evidence_kinds_complete": REQUIRED_EVIDENCE_KINDS <= kinds,
        "findings_count": len(findings),
        "warning_count": sum(1 for f in findings if f.get("severity") == "warning"),
        "error_count": sum(1 for f in findings if f.get("severity") == "error"),
        "step_summary_count": len(summaries),
        "step_statuses": step_statuses,
        "failed_step_count": len(failed_steps),
        "failed_steps": failed_steps,
        "llm_repaired_step_count": len(llm_repaired_steps),
        "deterministically_repaired_step_count": len(deterministically_repaired_steps),
        "deterministic_repairs": [
            {
                "step_id": item.get("step_id"),
                "repair": item.get("runner_repair"),
            }
            for item in deterministically_repaired_steps
        ],
        "bound_manuscript_exists": bound.exists(),
        "evidence_missing_count": evidence_missing_count,
        "manuscript_error_count": len(manuscript_errors),
        "non_manuscript_error_count": len(non_manuscript_errors),
    }

    primary_or, primary_metric_source = _first_effect_estimate(
        metric_summaries,
        task_family=task.family,
    )
    metrics["primary_or"] = primary_or
    metrics["primary_metric_source"] = primary_metric_source
    metrics["auroc"] = _first_float(
        _nested_values(
            metric_summaries,
            keys=[
                "auroc",
                "statistic:auroc",
                "auc",
                "statistic:auc",
                "held_out_auroc",
                "statistic:held_out_auroc",
                "cv_auroc",
                "statistic:cv_auroc",
                "cv_auroc_mean",
                "statistic:cv_auroc_mean",
                "mean_auroc",
                "auroc_mean",
            ],
        )
    )
    metrics["brier_score"] = _first_float(
        _nested_values(
            metric_summaries,
            keys=[
                "brier_score",
                "statistic:brier_score",
                "held_out_brier",
                "statistic:held_out_brier",
                "cv_brier_mean",
                "statistic:cv_brier_mean",
                "brier_mean",
            ],
        )
    )
    metrics["silhouette_score"] = _first_float(
        _nested_values(
            metric_summaries,
            keys=[
                "silhouette_score",
                "statistic:silhouette_score",
                "silhouette",
                "statistic:silhouette",
            ],
        )
    )
    metrics["cluster_count"] = _first_float(
        _nested_values(
            metric_summaries,
            keys=[
                "cluster_count",
                "statistic:cluster_count",
                "n_clusters",
                "statistic:n_clusters",
            ],
        )
    )
    metrics["spearman_rho"] = _first_float(
        _nested_values(
            metric_summaries,
            keys=[
                "spearman_rho",
                "rho",
                "spearman",
                "spearman_correlations",
                "correlations",
            ],
        )
    )
    metrics["n_rows"] = _first_float(
        _nested_values(
            metric_summaries,
            keys=["n_rows", "n_stays", "sample_size", "n", "cohort_n"],
        )
    )
    metrics["mortality_rate"] = _first_float(
        _nested_values(
            metric_summaries,
            keys=[
                "mortality_rate",
                "death_rate",
                "outcome_rate",
                "event_rate",
                "baseline_prevalence",
            ],
        )
    )
    metrics["complete_case_n"] = _first_float(
        _nested_values(
            metric_summaries,
            keys=[
                "n_complete_cases",
                "n_complete_case",
                "complete_case_n",
                "statistic:complete_case_n",
                "sample_size_complete_case",
                "cc_n",
                "statistic:cc_n",
                "cc_sample_size",
                "statistic:cc_sample_size",
                "complete_case_sample_size",
                "complete_case_sample_size",
                "sample_size:complete_case",
            ],
        )
    )
    if metrics["complete_case_n"] is None and task.family == "robustness":
        metrics["complete_case_n"] = _first_float(
            _nested_values(metric_summaries, keys=["sample_size", "n"])
    )
    metrics["sofa_zero_count"] = _first_float(
        _nested_values(
            metric_summaries,
            keys=[
                "n_sofa2_zero",
                "sofa_zero_count",
                "sofa2_max_24h_zero_count",
            ],
        )
    )
    selection_bias_warning, warning_source = _extract_selection_bias_warning(
        manifest,
        metric_summaries,
    )
    metrics["selection_bias_warning"] = selection_bias_warning
    metrics["warning_source"] = warning_source
    metrics["guardrail_warning"] = _contains_text(manifest, metric_summaries, "sofa") and (
        _contains_text(manifest, metric_summaries, "zero")
        or _contains_text(manifest, metric_summaries, "missing")
        or _contains_text(manifest, metric_summaries, "artifact")
        or _contains_text(manifest, metric_summaries, "artefact")
    )
    metrics["baseline_prevalence"] = _first_float(
        _nested_values(
            metric_summaries,
            keys=[
                "baseline_prevalence",
                "statistic:baseline_prevalence",
                "prevalence_baseline",
                "statistic:prevalence_baseline",
                "event_rate",
                "statistic:event_rate",
                "outcome_rate",
                "statistic:outcome_rate",
                "mortality_rate",
                "statistic:mortality_rate",
            ],
        )
    )
    metrics["split_or_cv"] = (
        _contains_text(manifest, metric_summaries, "cross-validation")
        or _contains_text(manifest, metric_summaries, "cross validation")
        or _contains_text(manifest, metric_summaries, "5-fold")
        or _contains_text(manifest, metric_summaries, "fold")
        or _contains_text(manifest, metric_summaries, "held-out")
        or _contains_text(manifest, metric_summaries, "train_test_split")
        or _contains_text(manifest, metric_summaries, "train/test")
        or _contains_text(manifest, metric_summaries, "cv_")
    )
    metrics["calibration"] = (
        _contains_text(manifest, metric_summaries, "calibration")
        or metrics.get("brier_score") is not None
    )
    metrics["missingness"] = (
        _contains_text(manifest, metric_summaries, "missingness")
        or _contains_text(manifest, metric_summaries, "missing")
        or _evidence_name_contains(evidence, "missing")
    )
    metrics["stratum_table"] = (
        _evidence_name_contains(evidence, "strata")
        or _evidence_name_contains(evidence, "stratum")
        or _contains_text(manifest, metric_summaries, "strata")
        or _contains_text(manifest, metric_summaries, "stratum")
    )
    metrics["cluster_characteristics"] = (
        _evidence_name_contains(evidence, "cluster", "character")
        or _evidence_name_contains(evidence, "cluster", "profile")
        or _contains_text(manifest, metric_summaries, "cluster characteristics")
        or _contains_text(manifest, metric_summaries, "cluster profile")
        or _contains_text(manifest, metric_summaries, "physiologic profiles")
    )
    metrics["cluster_mortality"] = (
        _evidence_name_contains(evidence, "cluster", "mortality")
        or _contains_text(manifest, metric_summaries, "mortality by cluster")
        or (_contains_text(manifest, metric_summaries, "cluster") and _contains_text(manifest, metric_summaries, "mortality"))
    )
    metrics["silhouette_or_cluster_count"] = (
        metrics.get("silhouette_score") is not None
        or metrics.get("cluster_count") is not None
    )
    metrics["workflow_hit"] = metrics["step_summary_count"] > 0
    metrics["artifact_hit"] = metrics["evidence_count"] > 0
    metrics["pipeline_completed"] = (
        (run_dir / "manifest.json").exists()
        and metrics["bound_manuscript_exists"]
        and metrics["step_summary_count"] > 0
    )
    metrics["analysis_complete"] = (
        metrics["pipeline_completed"]
        and metrics["failed_step_count"] == 0
    )
    metrics["manuscript_binding_complete"] = (
        metrics["pipeline_completed"]
        and int(metrics.get("evidence_missing_count") or 0) == 0
        and metrics["manuscript_error_count"] == 0
    )
    metrics["execution_success"] = (
        metrics["bound_manuscript_exists"]
        and metrics["step_summary_count"] > 0
    )

    expected_hit = {}
    for metric in task.expected_metrics:
        value = metrics.get(metric)
        if isinstance(value, bool):
            expected_hit[metric] = value
        else:
            expected_hit[metric] = value is not None
    metrics["expected_metric_hits"] = expected_hit

    required_artifact_hits: Dict[str, bool] = {}
    for artifact in task.required_artifacts:
        key = str(artifact).strip().lower()
        if key == "manifest":
            required_artifact_hits[artifact] = (run_dir / "manifest.json").exists()
        elif key == "bound_manuscript":
            required_artifact_hits[artifact] = bool(metrics["bound_manuscript_exists"])
        elif key == "step_summary":
            required_artifact_hits[artifact] = int(metrics["step_summary_count"] or 0) > 0
        elif key in {"code", "log", "table", "figure", "statistic"}:
            required_artifact_hits[artifact] = key in kinds
        elif key == "required_evidence_kinds":
            required_artifact_hits[artifact] = bool(metrics["evidence_kinds_complete"])
        else:
            required_artifact_hits[artifact] = _evidence_name_contains(evidence, key)
    metrics["required_artifact_hits"] = required_artifact_hits
    return metrics


def _missing_expected_metrics(metrics: Dict[str, Any]) -> List[str]:
    hits = metrics.get("expected_metric_hits") or {}
    return [str(key) for key, value in hits.items() if not bool(value)]


def _missing_required_artifacts(metrics: Dict[str, Any]) -> List[str]:
    hits = metrics.get("required_artifact_hits") or {}
    return [str(key) for key, value in hits.items() if not bool(value)]


def _classify_failure(
    exc: Optional[BaseException],
    metrics: Optional[Dict[str, Any]],
    task: Optional[V14Task] = None,
) -> Optional[str]:
    if exc is not None:
        text = f"{type(exc).__name__}: {exc}".lower()
        if "timeout" in text or "timed out" in text:
            return "model_timeout"
        if "stalled" in text or "heartbeat" in text:
            return "runtime_stalled"
        if "planner" in text or "analysis_plan" in text:
            return "planner_failure"
        if "coder" in text or "code" in text or "syntaxerror" in text:
            return "coder_failure"
        if "forbidden" in text or "aggregation" in text:
            return "forbidden_aggregation"
        return "runtime_failure"
    if not metrics:
        return "runtime_failure"
    if not metrics.get("pipeline_completed"):
        return "runtime_failure"
    if metrics.get("failed_step_count", 0) > 0:
        return "step_execution_failed"
    if not metrics.get("bound_manuscript_exists"):
        return "evidence_binding_failure"
    if metrics.get("step_summary_count", 0) <= 0:
        return "coder_failure"
    if int(metrics.get("evidence_missing_count") or 0) > 0:
        return "evidence_binding_issue"
    if metrics.get("manuscript_error_count", 0) > 0:
        return "manuscript_binding_issue"
    if metrics.get("non_manuscript_error_count", 0) > 0:
        return "runtime_finding"
    if task is not None and _missing_expected_metrics(metrics):
        return "metric_contract_failure"
    if task is not None and _missing_required_artifacts(metrics):
        return "artifact_contract_failure"
    return None


def _pipeline_status_from_record(record: Dict[str, Any]) -> str:
    failure = str(record.get("failure_class") or "")
    if failure == "runtime_stalled":
        return "stalled"
    if record.get("run_dir") is None:
        return "failed"
    metrics = record.get("metrics") or {}
    if metrics.get("pipeline_completed"):
        return "completed"
    if metrics.get("execution_success") or metrics.get("artifact_hit"):
        return "partial"
    return "failed"


def _acceptance_status_from_record(record: Dict[str, Any]) -> str:
    failure = str(record.get("failure_class") or "")
    if failure == "runtime_stalled":
        return "stalled"
    if record.get("status") == "missing_reusable_run":
        return "missing"
    if record.get("run_dir") is None:
        return "failed"
    metrics = record.get("metrics") or {}
    clean = (
        record.get("failure_class") is None
        and metrics.get("pipeline_completed")
        and metrics.get("analysis_complete")
        and metrics.get("manuscript_binding_complete")
        and not _missing_expected_metrics(metrics)
        and not _missing_required_artifacts(metrics)
    )
    return "clean_ok" if clean else "partial"


def _with_status_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    pipeline_status = _pipeline_status_from_record(record)
    acceptance_status = _acceptance_status_from_record(record)
    record["pipeline_status"] = pipeline_status
    record["acceptance_status"] = acceptance_status
    record["status"] = acceptance_status
    return record


def _run_task_arm(
    *,
    task: V14Task,
    arm: str,
    model: str,
    provider: str,
    cohort_path: Path,
    out_root: Path,
    request_timeout: float,
    experiment_mode: str,
    repo_root: Path,
    log_path: Path,
) -> Tuple[Dict[str, Any], bool]:
    import pandas as pd
    from easyicu.research_agent import ResearchAgentPipeline  # type: ignore

    cohort = pd.read_parquet(cohort_path)
    arm_root = out_root / _slugify(model) / task.key / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    llm = _make_llm(provider=provider, model=model, request_timeout=request_timeout)
    if experiment_mode == "native":
        max_code_repair_attempts = 0
        enable_deterministic_runner_repair = False
        enable_deterministic_code_fallback = False
    elif experiment_mode == "self_repair":
        max_code_repair_attempts = 3
        enable_deterministic_runner_repair = False
        enable_deterministic_code_fallback = False
    elif experiment_mode == "guardrails":
        max_code_repair_attempts = 3
        enable_deterministic_runner_repair = True
        enable_deterministic_code_fallback = False
    else:
        raise ValueError(f"Unknown experiment_mode: {experiment_mode}")
    pipeline = ResearchAgentPipeline(
        workdir=arm_root,
        llm=llm,
        timeout_seconds=float(request_timeout),
        enable_literature=False,
        enable_visual_qa=True,
        enable_memory=False,
        enable_latex=True,
        enable_probe_step=True,
        enable_replanning=False,
        max_code_repair_attempts=max_code_repair_attempts,
        enable_deterministic_code_fallback=enable_deterministic_code_fallback,
        enable_deterministic_planner_fallback=False,
        enable_deterministic_runner_repair=enable_deterministic_runner_repair,
        disable_icu_context=(arm == "naive"),
    )
    _append_log(
        log_path,
        f"[{_utc_now()}] RUN model={model} task={task.key} arm={arm} mode={experiment_mode}",
    )
    result = pipeline.run(
        question=task.question,
        cohort=cohort,
        cohort_name=f"v14_{task.key}",
        database="miiv",
        target_outcome=task.target_outcome,
        notes=(
            "EasyICU v14 real-cohort agent experiment. "
            f"task_key={task.key}; family={task.family}; difficulty={task.difficulty}; "
            f"arm={arm}; experiment_mode={experiment_mode}; "
            f"disable_icu_context={arm == 'naive'}; "
            f"cohort_sha256={_sha256_file(cohort_path)}; git_commit={_git_commit(repo_root)}. "
            "The task is designed for agent evaluation and manuscript evidence provenance."
        ),
        user_preferences=None if arm == "naive" else task.user_preferences,
    )
    run_dir = Path(result.workdir)
    _write_json(
        run_dir / "v14_task_contract.json",
        {
            "task_key": task.key,
            "arm": arm,
            "expected_metrics": task.expected_metrics,
            "required_artifacts": task.required_artifacts,
            "allowed_warnings": task.allowed_warnings,
            "fatal_conditions": task.fatal_conditions,
            "naive_user_preferences_stripped": arm == "naive",
        },
    )
    metrics = _extract_metrics(run_dir, task)
    failure_class = _classify_failure(None, metrics, task)
    record = {
        "task": asdict(task),
        "arm": arm,
        "provider": provider,
        "model": model,
        "experiment_mode": experiment_mode,
        "cohort_path": str(cohort_path),
        "cohort_sha256": _sha256_file(cohort_path),
        "run_dir": str(run_dir),
        "run_id": result.run_id,
        "failure_class": failure_class,
        "metrics": metrics,
    }
    record = _with_status_fields(record)
    success = record["acceptance_status"] == "clean_ok"
    return record, success


def _heartbeat_writer(
    *,
    heartbeat_path: Path,
    stop_event: threading.Event,
    payload: Dict[str, Any],
    interval_seconds: int,
) -> None:
    while not stop_event.is_set():
        heartbeat = dict(payload)
        heartbeat["updated_at"] = _utc_now()
        heartbeat["pid"] = os.getpid()
        _write_json(heartbeat_path, heartbeat)
        stop_event.wait(max(1, int(interval_seconds)))


def _run_task_arm_child(queue, kwargs: Dict[str, Any], heartbeat_path: str, heartbeat_interval: int) -> None:
    heartbeat = Path(heartbeat_path)
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    payload = {
        "status": "running",
        "model": kwargs.get("model"),
        "task_key": getattr(kwargs.get("task"), "key", None),
        "arm": kwargs.get("arm"),
        "provider": kwargs.get("provider"),
        "experiment_mode": kwargs.get("experiment_mode"),
    }
    thread = threading.Thread(
        target=_heartbeat_writer,
        kwargs={
            "heartbeat_path": heartbeat,
            "stop_event": stop_event,
            "payload": payload,
            "interval_seconds": heartbeat_interval,
        },
        daemon=True,
    )
    thread.start()
    try:
        record, success = _run_task_arm(**kwargs)
        _write_json(
            heartbeat,
            {
                **payload,
                "status": record.get("acceptance_status") or ("clean_ok" if success else "partial"),
                "pipeline_status": record.get("pipeline_status"),
                "acceptance_status": record.get("acceptance_status"),
                "updated_at": _utc_now(),
                "pid": os.getpid(),
                "run_dir": record.get("run_dir"),
                "run_id": record.get("run_id"),
                "failure_class": record.get("failure_class"),
            },
        )
        queue.put({"record": record, "success": success})
    except BaseException as exc:
        failure_class = _classify_failure(exc, None)
        message = f"{type(exc).__name__}: {exc}"
        _write_json(
            heartbeat,
            {
                **payload,
                "status": "error",
                "pipeline_status": "stalled" if failure_class == "runtime_stalled" else "failed",
                "acceptance_status": "stalled" if failure_class == "runtime_stalled" else "failed",
                "updated_at": _utc_now(),
                "pid": os.getpid(),
                "failure_class": failure_class,
                "error": message,
            },
        )
        queue.put({"error": message, "failure_class": failure_class})
    finally:
        stop_event.set()
        thread.join(timeout=2)


def _run_task_arm_with_watchdog(
    *,
    task: V14Task,
    arm: str,
    model: str,
    provider: str,
    cohort_path: Path,
    out_root: Path,
    request_timeout: float,
    experiment_mode: str,
    repo_root: Path,
    log_path: Path,
    task_timeout: float,
    heartbeat_interval: int,
) -> Tuple[Dict[str, Any], bool]:
    heartbeat_path = (
        out_root
        / "_heartbeats"
        / _slugify(model)
        / task.key
        / arm
        / "heartbeat.json"
    )
    kwargs = {
        "task": task,
        "arm": arm,
        "model": model,
        "provider": provider,
        "cohort_path": cohort_path,
        "out_root": out_root,
        "request_timeout": request_timeout,
        "experiment_mode": experiment_mode,
        "repo_root": repo_root,
        "log_path": log_path,
    }
    timeout = float(task_timeout or 0)
    if timeout <= 0:
        return _run_task_arm(**kwargs)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_run_task_arm_child,
        args=(queue, kwargs, str(heartbeat_path), heartbeat_interval),
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        record = {
            "task": asdict(task),
            "arm": arm,
            "provider": provider,
            "model": model,
            "experiment_mode": experiment_mode,
            "cohort_path": str(cohort_path),
            "cohort_sha256": _sha256_file(cohort_path),
            "run_dir": None,
            "run_id": None,
            "status": "error",
            "failure_class": "runtime_stalled",
            "error": f"Task exceeded watchdog timeout of {timeout:.0f}s.",
            "metrics": {},
            "heartbeat_path": str(heartbeat_path),
        }
        record = _with_status_fields(record)
        _write_json(
            heartbeat_path,
            {
                "status": "runtime_stalled",
                "pipeline_status": "stalled",
                "acceptance_status": "stalled",
                "updated_at": _utc_now(),
                "model": model,
                "task_key": task.key,
                "arm": arm,
                "provider": provider,
                "experiment_mode": experiment_mode,
                "timeout_seconds": timeout,
            },
        )
        return record, False

    try:
        payload = queue.get(timeout=5)
    except queue_module.Empty:
        payload = {}
    if payload.get("record") is not None:
        record = payload["record"]
        record["heartbeat_path"] = str(heartbeat_path)
        return record, bool(payload.get("success"))
    message = payload.get("error") or "Child process ended without returning a record."
    record = {
        "task": asdict(task),
        "arm": arm,
        "provider": provider,
        "model": model,
        "experiment_mode": experiment_mode,
        "cohort_path": str(cohort_path),
        "cohort_sha256": _sha256_file(cohort_path),
        "run_dir": None,
        "run_id": None,
        "status": "error",
        "failure_class": payload.get("failure_class") or "runtime_failure",
        "error": message,
        "metrics": {},
        "heartbeat_path": str(heartbeat_path),
    }
    return _with_status_fields(record), False


def _reuse_task_arm(
    *, task: V14Task, arm: str, model: str, cohort_path: Path, out_root: Path
) -> Optional[Dict[str, Any]]:
    return _best_reusable_run_record(
        task=task,
        arm=arm,
        model=model,
        cohort_path=cohort_path,
        out_root=out_root,
    )


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _pid_exists(pid: Any) -> bool:
    try:
        value = int(pid)
    except Exception:
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _stale_heartbeat_record(
    *,
    task: V14Task,
    arm: str,
    model: str,
    provider: str,
    experiment_mode: str,
    cohort_path: Path,
    out_root: Path,
    task_timeout: float,
) -> Optional[Dict[str, Any]]:
    heartbeat_path = (
        out_root
        / "_heartbeats"
        / _slugify(model)
        / task.key
        / arm
        / "heartbeat.json"
    )
    if not heartbeat_path.exists():
        return None
    heartbeat = _read_json(heartbeat_path)
    status = str(heartbeat.get("status") or "").lower()
    if status not in {"running", "starting"}:
        return None
    updated_at = _parse_iso_datetime(heartbeat.get("updated_at"))
    age_seconds = None
    if updated_at is not None:
        age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
    is_stale = (
        not _pid_exists(heartbeat.get("pid"))
        or (age_seconds is not None and age_seconds > max(float(task_timeout or 0), 60.0))
    )
    if not is_stale:
        return None
    record = {
        "task": asdict(task),
        "arm": arm,
        "provider": provider,
        "model": model,
        "experiment_mode": experiment_mode,
        "cohort_path": str(cohort_path),
        "cohort_sha256": _sha256_file(cohort_path),
        "run_dir": heartbeat.get("run_dir"),
        "run_id": heartbeat.get("run_id"),
        "status": "runtime_stalled",
        "failure_class": "runtime_stalled",
        "error": "Stale heartbeat found during aggregate-only recovery.",
        "metrics": {},
        "heartbeat_path": str(heartbeat_path),
    }
    _write_json(
        heartbeat_path,
        {
            **heartbeat,
            "status": "runtime_stalled",
            "pipeline_status": "stalled",
            "acceptance_status": "stalled",
            "recovered_at": _utc_now(),
            "stale_age_seconds": age_seconds,
        },
    )
    return _with_status_fields(record)


def _terminal_heartbeat_record(
    *,
    task: V14Task,
    arm: str,
    model: str,
    provider: str,
    experiment_mode: str,
    cohort_path: Path,
    out_root: Path,
) -> Optional[Dict[str, Any]]:
    heartbeat_path = (
        out_root
        / "_heartbeats"
        / _slugify(model)
        / task.key
        / arm
        / "heartbeat.json"
    )
    if not heartbeat_path.exists():
        return None
    heartbeat = _read_json(heartbeat_path)
    status = str(heartbeat.get("status") or "").lower()
    if status in {"running", "starting", "", "clean_ok", "partial"}:
        return None
    acceptance_status = "stalled" if status in {"runtime_stalled", "stalled"} else "failed"
    record = {
        "task": asdict(task),
        "arm": arm,
        "provider": provider,
        "model": model,
        "experiment_mode": experiment_mode,
        "cohort_path": str(cohort_path),
        "cohort_sha256": _sha256_file(cohort_path),
        "run_dir": heartbeat.get("run_dir"),
        "run_id": heartbeat.get("run_id"),
        "status": acceptance_status,
        "failure_class": heartbeat.get("failure_class") or ("runtime_stalled" if acceptance_status == "stalled" else "runtime_failure"),
        "error": heartbeat.get("error"),
        "metrics": {},
        "heartbeat_path": str(heartbeat_path),
    }
    return _with_status_fields(record)


def _progress_payload(
    *,
    repo_root: Path,
    out_root: Path,
    provider: str,
    models: Sequence[str],
    tasks: Sequence[V14Task],
    arms: Sequence[str],
    attempts: Sequence[AttemptRecord],
) -> Dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "aggregation_version": AGGREGATION_VERSION,
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "git_diff_hash": _git_diff_hash(repo_root),
        "git_status_at_write": _git_status(repo_root),
        "out_root": str(out_root),
        "provider": provider,
        "provider_base_url": _provider_base_url(provider),
        "models": list(models),
        "tasks": [task.key for task in tasks],
        "arms": list(arms),
        "attempts": [asdict(item) for item in attempts],
    }


def _render_markdown(
    results: Sequence[Dict[str, Any]], *, provider: str, models: Sequence[str]
) -> str:
    lines = [
        "# EasyICU v14 Agent Experiments",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        f"_Aggregation version: `{AGGREGATION_VERSION}`_",
        f"_Provider: `{provider}`_",
        f"_Models: `{', '.join(models)}`_",
        "",
        "This report summarizes v14 real-cohort agent runs. Each task/arm keeps "
        "its original run directory and manifest; failures are retained as "
        "experimental outcomes.",
        "",
        "| Model | Task | Arm | Mode | Pipeline status | Acceptance status | Failure class | Pipeline | Analysis | Manuscript | Failed steps | LLM repairs | Deterministic repairs | Evidence | Missing evidence | Warnings | Errors | Primary OR | AUROC | Silhouette | Spearman | Run dir |",
        "|---|---|---|---|---|---|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        metrics = row.get("metrics", {})
        lines.append(
            f"| `{row.get('model')}` "
            f"| `{row.get('task', {}).get('key')}` "
            f"| `{row.get('arm')}` "
            f"| `{row.get('experiment_mode') or ''}` "
            f"| `{row.get('pipeline_status') or ''}` "
            f"| `{row.get('acceptance_status') or row.get('status')}` "
            f"| `{row.get('failure_class') or ''}` "
            f"| {'Y' if metrics.get('pipeline_completed') else 'N'} "
            f"| {'Y' if metrics.get('analysis_complete') else 'N'} "
            f"| {'Y' if metrics.get('manuscript_binding_complete') else 'N'} "
            f"| {metrics.get('failed_step_count', '')} "
            f"| {metrics.get('llm_repaired_step_count', '')} "
            f"| {metrics.get('deterministically_repaired_step_count', '')} "
            f"| {metrics.get('evidence_count', '')} "
            f"| {metrics.get('evidence_missing_count', '')} "
            f"| {metrics.get('warning_count', '')} "
            f"| {metrics.get('error_count', '')} "
            f"| {'' if metrics.get('primary_or') is None else round(float(metrics.get('primary_or')), 4)} "
            f"| {'' if metrics.get('auroc') is None else round(float(metrics.get('auroc')), 4)} "
            f"| {'' if metrics.get('silhouette_score') is None else round(float(metrics.get('silhouette_score')), 4)} "
            f"| {'' if metrics.get('spearman_rho') is None else round(float(metrics.get('spearman_rho')), 4)} "
            f"| `{row.get('run_dir')}` |"
        )
    lines.extend(["", "## Aggregate", ""])
    by_model_arm: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in results:
        by_model_arm.setdefault((str(row.get("model")), str(row.get("arm"))), []).append(row)
    lines.append("| Model | Arm | Runs | Pipeline completed | Analysis complete | Manuscript binding complete | Evidence complete | Total missing evidence | Warnings | Errors |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (model, arm), rows in sorted(by_model_arm.items()):
        metrics_rows = [row.get("metrics", {}) for row in rows]
        lines.append(
            f"| `{model}` | `{arm}` | {len(rows)} | "
            f"{sum(1 for m in metrics_rows if m.get('pipeline_completed'))} | "
            f"{sum(1 for m in metrics_rows if m.get('analysis_complete'))} | "
            f"{sum(1 for m in metrics_rows if m.get('manuscript_binding_complete'))} | "
            f"{sum(1 for m in metrics_rows if m.get('evidence_kinds_complete'))} | "
            f"{sum(int(m.get('evidence_missing_count') or 0) for m in metrics_rows)} | "
            f"{sum(int(m.get('warning_count') or 0) for m in metrics_rows)} | "
            f"{sum(int(m.get('error_count') or 0) for m in metrics_rows)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_csvs(out_root: Path, results: Sequence[Dict[str, Any]]) -> None:
    task_matrix = out_root / "v14_task_matrix.csv"
    fields = [
        "model",
        "task_key",
        "family",
        "difficulty",
        "arm",
        "experiment_mode",
        "status",
        "pipeline_status",
        "acceptance_status",
        "failure_class",
        "run_id",
        "run_dir",
        "cohort_sha256",
        "execution_success",
        "pipeline_completed",
        "analysis_complete",
        "manuscript_binding_complete",
        "workflow_hit",
        "artifact_hit",
        "evidence_count",
        "evidence_kinds_complete",
        "evidence_missing_count",
        "findings_count",
        "warning_count",
        "error_count",
        "manuscript_error_count",
        "non_manuscript_error_count",
        "failed_step_count",
        "llm_repaired_step_count",
        "deterministically_repaired_step_count",
        "aggregation_version",
        "primary_or",
        "primary_metric_source",
        "auroc",
        "brier_score",
        "silhouette_score",
        "cluster_count",
        "spearman_rho",
        "complete_case_n",
        "sofa_zero_count",
        "selection_bias_warning",
        "warning_source",
        "guardrail_warning",
        "calibration",
        "baseline_prevalence",
        "split_or_cv",
        "missingness",
        "stratum_table",
        "cluster_characteristics",
        "cluster_mortality",
        "silhouette_or_cluster_count",
        "missing_expected_metrics",
        "missing_required_artifacts",
    ]
    with task_matrix.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in results:
            task = row.get("task", {})
            metrics = row.get("metrics", {})
            writer.writerow(
                {
                    "model": row.get("model"),
                    "task_key": task.get("key"),
                    "family": task.get("family"),
                    "difficulty": task.get("difficulty"),
                    "arm": row.get("arm"),
                    "experiment_mode": row.get("experiment_mode"),
                    "status": row.get("status"),
                    "pipeline_status": row.get("pipeline_status"),
                    "acceptance_status": row.get("acceptance_status"),
                    "failure_class": row.get("failure_class"),
                    "run_id": row.get("run_id"),
                    "run_dir": row.get("run_dir"),
                    "cohort_sha256": row.get("cohort_sha256"),
                    **{field: metrics.get(field) for field in fields if field in metrics},
                    "missing_expected_metrics": ";".join(_missing_expected_metrics(metrics)),
                    "missing_required_artifacts": ";".join(_missing_required_artifacts(metrics)),
                }
            )

    matrix_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in results:
        metrics = row.get("metrics", {})
        key = (str(row.get("model")), str(row.get("arm")))
        agg = matrix_rows.setdefault(
            key,
            {
                "model": row.get("model"),
                "arm": row.get("arm"),
                "aggregation_version": AGGREGATION_VERSION,
                "n_runs": 0,
                "execution_success": 0,
                "pipeline_completed": 0,
                "analysis_complete": 0,
                "manuscript_binding_complete": 0,
                "evidence_kinds_complete": 0,
                "evidence_missing_count": 0,
                "warning_count": 0,
                "error_count": 0,
                "clean_ok": 0,
                "partial": 0,
                "failed": 0,
                "stalled": 0,
                "missing": 0,
            },
        )
        agg["n_runs"] += 1
        status = str(row.get("acceptance_status") or row.get("status") or "")
        if status in {"clean_ok", "partial", "failed", "stalled", "missing"}:
            agg[status] += 1
        agg["execution_success"] += int(bool(metrics.get("execution_success")))
        agg["pipeline_completed"] += int(bool(metrics.get("pipeline_completed")))
        agg["analysis_complete"] += int(bool(metrics.get("analysis_complete")))
        agg["manuscript_binding_complete"] += int(bool(metrics.get("manuscript_binding_complete")))
        agg["evidence_kinds_complete"] += int(bool(metrics.get("evidence_kinds_complete")))
        agg["evidence_missing_count"] += int(metrics.get("evidence_missing_count") or 0)
        agg["warning_count"] += int(metrics.get("warning_count") or 0)
        agg["error_count"] += int(metrics.get("error_count") or 0)

    model_matrix = out_root / "v14_model_matrix.csv"
    with model_matrix.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model",
                "arm",
                "n_runs",
                "aggregation_version",
                "execution_success",
                "pipeline_completed",
                "analysis_complete",
                "manuscript_binding_complete",
                "evidence_kinds_complete",
                "evidence_missing_count",
                "warning_count",
                "error_count",
                "clean_ok",
                "partial",
                "failed",
                "stalled",
                "missing",
            ],
        )
        writer.writeheader()
        writer.writerows(matrix_rows.values())


def _context_metadata_counts(context_path: Path) -> Dict[str, Any]:
    if not context_path.exists():
        return {"context_exists": False}
    data = _read_json(context_path)
    variables = data.get("variables") or []
    clinical_roles = {
        "lab",
        "vital",
        "intervention",
        "composite_score",
        "organ_support",
        "score_component",
        "demographic",
    }
    return {
        "context_exists": True,
        "time_windows": len(data.get("time_windows") or []),
        "temporal_constraints": len(data.get("temporal_constraints") or []),
        "variables": len(variables),
        "clinical_role_variables": sum(1 for v in variables if str(v.get("role") or "") in clinical_roles),
        "variables_with_missingness": sum(1 for v in variables if v.get("missingness") is not None),
        "variables_with_pitfalls": sum(1 for v in variables if v.get("pitfalls")),
        "variables_with_clinical_caveats": sum(1 for v in variables if v.get("clinical_caveats")),
        "variables_with_source_concept": sum(1 for v in variables if v.get("source_concept")),
        "variables_with_non_any_aggregation": sum(
            1
            for v in variables
            if any(str(item).lower() != "any" for item in (v.get("allowed_aggregations") or []))
        ),
        "user_preferences_present": data.get("user_preferences") is not None,
    }


def _write_context_ablation_audit(out_root: Path, results: Sequence[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for row in results:
        run_dir = row.get("run_dir")
        task = row.get("task") or {}
        context_path = Path(run_dir) / "research_context.json" if run_dir else Path("__missing__")
        counts = _context_metadata_counts(context_path)
        rows.append(
            {
                "model": row.get("model"),
                "task_key": task.get("key"),
                "arm": row.get("arm"),
                "run_id": row.get("run_id"),
                "run_dir": run_dir,
                **counts,
            }
        )
    _write_json(out_root / "context_ablation_audit.json", {"rows": rows})
    fields = [
        "model",
        "task_key",
        "arm",
        "run_id",
        "run_dir",
        "context_exists",
        "time_windows",
        "temporal_constraints",
        "variables",
        "clinical_role_variables",
        "variables_with_missingness",
        "variables_with_pitfalls",
        "variables_with_clinical_caveats",
        "variables_with_source_concept",
        "variables_with_non_any_aggregation",
        "user_preferences_present",
    ]
    with (out_root / "context_ablation_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_recovery_attempts_csv(out_root: Path) -> None:
    recovery_root = out_root / "_recovery_attempts"
    rows: List[Dict[str, Any]] = []
    if recovery_root.exists():
        for matrix in sorted(recovery_root.rglob("v14_task_matrix.csv")):
            try:
                with matrix.open(newline="", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        row = dict(row)
                        row["source_matrix"] = str(matrix)
                        rows.append(row)
            except Exception:
                continue
    fields = sorted({key for row in rows for key in row}) or ["source_matrix"]
    with (out_root / "v14_recovery_attempts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _archive_pre_v14_outputs(repo_root: Path, out_root: Path, log_path: Path) -> None:
    archive_root = repo_root / "research_output" / "archive" / f"pre_v14_agent_runs_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    archive_root.mkdir(parents=True, exist_ok=True)
    for candidate in [repo_root / "research_output" / "bench", repo_root / "research_output" / "validation"]:
        if not candidate.exists():
            continue
        target = archive_root / candidate.name
        if target.exists():
            continue
        shutil.copytree(candidate, target)
        _append_log(log_path, f"Archived {candidate} -> {target}")
    _write_json(out_root / "pre_v14_archive_manifest.json", {"archive_root": str(archive_root)})


def _select_tasks(all_tasks: Sequence[V14Task], wanted: Optional[Sequence[str]]) -> List[V14Task]:
    if not wanted:
        return list(all_tasks)
    by_key = {task.key: task for task in all_tasks}
    unknown = sorted(set(wanted) - set(by_key))
    if unknown:
        raise SystemExit(
            f"Unknown v14 task keys: {unknown}; available: {sorted(by_key)}"
        )
    return [by_key[key] for key in wanted]


def main() -> int:
    repo_root = _bootstrap_imports()
    all_tasks = _load_task_specs_from_builder()

    default_out = repo_root / "research_output" / f"v14_experiments_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", default="/Users/haibo/Documents/GitHub/miiv_20260420")
    parser.add_argument("--cohort-dir", default=None)
    parser.add_argument("--out-root", default=str(default_out))
    parser.add_argument("--provider", choices=["openrouter", "openai", "mock"], default="openrouter")
    parser.add_argument("--model", default=os.environ.get("EASYICU_HOSTED_DEFAULT_MODEL", DEFAULT_FREE_MODELS[0]))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--items", nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", choices=["aware", "naive"], default=["aware", "naive"])
    parser.add_argument(
        "--experiment-mode",
        choices=["native", "self_repair", "guardrails"],
        default="self_repair",
        help=(
            "native disables code repair; self_repair allows LLM coder repair only; "
            "guardrails also enables deterministic runner repairs."
        ),
    )
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=None,
        help=(
            "Wall-clock watchdog per task/arm in seconds. Defaults to max(900, "
            "4 * request-timeout). Use 0 to disable the child-process watchdog."
        ),
    )
    parser.add_argument("--heartbeat-interval", type=int, default=15)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=int, default=30)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--force-build-cohorts", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--archive-pre-v14", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "v14_runner.log"
    progress_path = out_root / "v14_progress.json"
    cohort_dir = Path(args.cohort_dir).expanduser().resolve() if args.cohort_dir else out_root / "cohorts"
    export_dir = Path(args.export_dir).expanduser().resolve()
    tasks = _select_tasks(all_tasks, args.items)
    models = list(args.models or [args.model])
    arms = list(dict.fromkeys(args.arms))
    task_timeout = (
        float(args.task_timeout)
        if args.task_timeout is not None
        else max(900.0, float(args.request_timeout) * 4.0)
    )

    plan = {
        "generated_at": _utc_now(),
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "git_diff_hash": _git_diff_hash(repo_root),
        "git_status_at_start": _git_status(repo_root),
        "export_dir": str(export_dir),
        "cohort_dir": str(cohort_dir),
        "out_root": str(out_root),
        "provider": args.provider,
        "provider_base_url": _provider_base_url(args.provider),
        "experiment_mode": args.experiment_mode,
        "models": models,
        "arms": arms,
        "request_timeout": float(args.request_timeout),
        "task_timeout": task_timeout,
        "heartbeat_interval": int(args.heartbeat_interval),
        "tasks": [asdict(task) for task in tasks],
    }
    _write_json(out_root / "v14_experiment_plan.json", plan)

    if not args.aggregate_only:
        _validate_provider_env(args.provider)

    if args.archive_pre_v14:
        _archive_pre_v14_outputs(repo_root, out_root, log_path)

    _build_cohorts_if_needed(
        repo_root=repo_root,
        export_dir=export_dir,
        cohort_dir=cohort_dir,
        force=bool(args.force_build_cohorts),
        log_path=log_path,
    )

    attempts: List[AttemptRecord] = []
    results: List[Dict[str, Any]] = []
    for model in models:
        for task in tasks:
            cohort_path = cohort_dir / task.cohort_file
            if not cohort_path.exists():
                raise SystemExit(f"Missing cohort file for {task.key}: {cohort_path}")
            for arm in arms:
                if args.reuse_existing or args.aggregate_only:
                    reused = _reuse_task_arm(
                        task=task,
                        arm=arm,
                        model=model,
                        cohort_path=cohort_path,
                        out_root=out_root,
                    )
                    if reused is not None:
                        results.append(reused)
                        continue
                    if args.aggregate_only:
                        terminal = _terminal_heartbeat_record(
                            task=task,
                            arm=arm,
                            model=model,
                            provider=args.provider,
                            experiment_mode=args.experiment_mode,
                            cohort_path=cohort_path,
                            out_root=out_root,
                        )
                        if terminal is not None:
                            results.append(terminal)
                            continue
                        stale = _stale_heartbeat_record(
                            task=task,
                            arm=arm,
                            model=model,
                            provider=args.provider,
                            experiment_mode=args.experiment_mode,
                            cohort_path=cohort_path,
                            out_root=out_root,
                            task_timeout=task_timeout,
                        )
                        if stale is not None:
                            results.append(stale)
                            continue
                        results.append(
                            _with_status_fields({
                                "task": asdict(task),
                                "arm": arm,
                                "provider": args.provider,
                                "model": model,
                                "experiment_mode": args.experiment_mode,
                                "cohort_path": str(cohort_path),
                                "cohort_sha256": _sha256_file(cohort_path),
                                "run_dir": None,
                                "run_id": None,
                                "status": "missing_reusable_run",
                                "failure_class": "runtime_failure",
                                "metrics": {},
                            })
                        )
                        continue

                last_record: Optional[Dict[str, Any]] = None
                for attempt_i in range(1, int(args.max_retries) + 1):
                    attempt = AttemptRecord(
                        task_key=task.key,
                        arm=arm,
                        model=model,
                        attempt=attempt_i,
                        started_at=_utc_now(),
                    )
                    attempts.append(attempt)
                    _write_json(
                        progress_path,
                        _progress_payload(
                            repo_root=repo_root,
                            out_root=out_root,
                            provider=args.provider,
                            models=models,
                            tasks=tasks,
                            arms=arms,
                            attempts=attempts,
                        ),
                    )
                    started = time.monotonic()
                    exc: Optional[BaseException] = None
                    success = False
                    try:
                        last_record, success = _run_task_arm_with_watchdog(
                            task=task,
                            arm=arm,
                            model=model,
                            provider=args.provider,
                            cohort_path=cohort_path,
                            out_root=out_root,
                            request_timeout=float(args.request_timeout),
                            experiment_mode=args.experiment_mode,
                            repo_root=repo_root,
                            log_path=log_path,
                            task_timeout=task_timeout,
                            heartbeat_interval=int(args.heartbeat_interval),
                        )
                        attempt.run_id = last_record.get("run_id")
                        attempt.run_dir = last_record.get("run_dir")
                        attempt.heartbeat_path = last_record.get("heartbeat_path")
                        if attempt.heartbeat_path:
                            heartbeat = _read_json(Path(attempt.heartbeat_path))
                            attempt.last_heartbeat_at = heartbeat.get("updated_at")
                        attempt.pipeline_status = last_record.get("pipeline_status") or "completed"
                        attempt.acceptance_status = last_record.get("acceptance_status") or ("clean_ok" if success else "partial")
                        attempt.status = attempt.acceptance_status
                        attempt.failure_class = last_record.get("failure_class")
                    except BaseException as err:
                        exc = err
                        attempt.status = "error"
                        attempt.error = f"{type(err).__name__}: {err}"
                        attempt.failure_class = _classify_failure(err, None)
                        last_record = {
                            "task": asdict(task),
                            "arm": arm,
                            "provider": args.provider,
                            "model": model,
                            "experiment_mode": args.experiment_mode,
                            "cohort_path": str(cohort_path),
                            "cohort_sha256": _sha256_file(cohort_path),
                            "run_dir": None,
                            "run_id": None,
                            "status": "error",
                            "failure_class": attempt.failure_class,
                            "error": attempt.error,
                            "metrics": {},
                        }
                        last_record = _with_status_fields(last_record)
                        attempt.pipeline_status = last_record["pipeline_status"]
                        attempt.acceptance_status = last_record["acceptance_status"]
                    attempt.completed_at = _utc_now()
                    attempt.elapsed_seconds = round(time.monotonic() - started, 2)
                    _write_json(
                        progress_path,
                        _progress_payload(
                            repo_root=repo_root,
                            out_root=out_root,
                            provider=args.provider,
                            models=models,
                            tasks=tasks,
                            arms=arms,
                            attempts=attempts,
                        ),
                    )
                    _append_log(
                        log_path,
                        (
                            f"[{attempt.completed_at}] END model={model} task={task.key} "
                            f"arm={arm} status={attempt.status} failure={attempt.failure_class or ''} "
                            f"elapsed={attempt.elapsed_seconds}s error={attempt.error or ''}"
                        ),
                    )
                    if success:
                        break
                    if attempt_i < int(args.max_retries):
                        time.sleep(int(args.sleep_seconds))
                if last_record is not None:
                    results.append(last_record)

    summary = {
        "generated_at": _utc_now(),
        "aggregation_version": AGGREGATION_VERSION,
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "git_diff_hash": _git_diff_hash(repo_root),
        "git_status_at_finish": _git_status(repo_root),
        "provider": args.provider,
        "provider_base_url": _provider_base_url(args.provider),
        "models": models,
        "arms": arms,
        "cohort_dir": str(cohort_dir),
        "results": results,
        "n_results": len(results),
        "n_execution_success": sum(
            1 for row in results if row.get("metrics", {}).get("execution_success")
        ),
        "n_clean_ok": sum(1 for row in results if row.get("acceptance_status") == "clean_ok"),
        "n_partial": sum(1 for row in results if row.get("acceptance_status") == "partial"),
        "n_failed": sum(1 for row in results if row.get("acceptance_status") == "failed"),
        "n_stalled": sum(1 for row in results if row.get("acceptance_status") == "stalled"),
        "n_missing": sum(1 for row in results if row.get("acceptance_status") == "missing"),
    }
    _write_json(out_root / "v14_experiment_summary.json", summary)
    (out_root / "v14_experiment_summary.md").write_text(
        _render_markdown(results, provider=args.provider, models=models),
        encoding="utf-8",
    )
    _write_csvs(out_root, results)
    _write_context_ablation_audit(out_root, results)
    _write_recovery_attempts_csv(out_root)
    print(out_root / "v14_experiment_summary.json")
    print(out_root / "v14_experiment_summary.md")
    print(out_root / "v14_task_matrix.csv")
    print(out_root / "v14_model_matrix.csv")
    print(out_root / "context_ablation_audit.csv")
    print(out_root / "v14_recovery_attempts.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

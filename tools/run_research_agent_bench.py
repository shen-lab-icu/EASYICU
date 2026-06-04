"""EHRFlowBench-style benchmark runner for the research agent (T2.1).

For every :class:`tests.bench.items.BenchItem` we run the same cohort
through ``ResearchAgentPipeline`` using the requested benchmark arm(s).
By default this preserves the historical two-arm context ablation —
once with the ICU-aware context (this work) and once with the naive
context (T1.4 baseline) — but paper-facing platform runs can use
``--arms aware`` to avoid running the naive ablation.

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

    # Offline plumbing smoke test (mock LLM). The 'aware' arm on mock returns
    # canned responses, so offline runs use the naive arm:
    python tools/run_research_agent_bench.py --arms naive
    python tools/run_research_agent_bench.py --items sofa2_mortality gcs_mortality --arms naive
    python tools/run_research_agent_bench.py --seed 42 --out-root ./bench_runs --arms naive
    # Paper-facing ICU-aware runs require a real provider:
    python tools/run_research_agent_bench.py --bench-kind analysis --arms aware --provider openrouter --model openai/gpt-oss-120b:free
    python tools/run_research_agent_bench.py --provider openrouter --models openai/gpt-oss-120b:free deepseek/deepseek-chat-v3-0324:free
    # Offline 'aware' plumbing check only (non-substantive, canned results):
    python tools/run_research_agent_bench.py --arms aware --allow-mock-aware

The original bench was mock-LLM only so the comparison isolated the
*context layer's* contribution from the LLM's. This script now also
supports real OpenAI-compatible providers (notably OpenRouter free
tier models) to make paper-facing context-ablation and model-comparison
runs reproducible from one entrypoint.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from easyicu.research_agent.pipeline_profiles import SubmissionProfile


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
_ARM_ORDER = ("naive", "aware")
_ARM_LABELS = {"naive": "Naive", "aware": "ICU-aware"}
_DEFAULT_SUBMISSION_PROFILE_REF = "npj_dm/20260527"


def _local_openai_base_url(base_url: Optional[str]) -> bool:
    lowered = (base_url or "").strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in ("localhost", "127.0.0.1", "0.0.0.0"))


def _normalize_arms(arms: Optional[Sequence[str]]) -> List[str]:
    selected = list(arms or _ARM_ORDER)
    unknown = [arm for arm in selected if arm not in _ARM_ORDER]
    if unknown:
        raise SystemExit(f"Unsupported arm(s): {unknown}; choose from {_ARM_ORDER}")
    ordered = [arm for arm in _ARM_ORDER if arm in selected]
    if not ordered:
        raise SystemExit("At least one benchmark arm is required.")
    return ordered


def _resolve_submission_profile(profile_ref: Optional[str]):
    _bootstrap_imports()
    from easyicu.research_agent.pipeline_profiles import get_submission_profile

    try:
        return get_submission_profile(profile_ref or _DEFAULT_SUBMISSION_PROFILE_REF)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _register_case_patterns(case_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not case_name:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_]+", case_name):
        raise SystemExit(
            "--case must be a case directory name containing only letters, "
            "numbers, and underscores"
        )
    _bootstrap_imports()
    from easyicu.research_agent.cohort_schema import default_pattern_registry

    module_name = f"benchmark.cases.{case_name}.register_patterns"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Unknown case {case_name!r}: {module_name} not found") from exc
    register = getattr(module, "register_patterns", None)
    if register is None:
        raise SystemExit(
            f"Case {case_name!r} must expose register_patterns()"
        )
    register(default_pattern_registry())
    patterns_path = getattr(module, "COHORT_PATTERNS_PATH", None)
    config_path = getattr(module, "CASE_CONFIG_PATH", None)
    return {
        "case": case_name,
        "patterns_path": str(patterns_path) if patterns_path is not None else None,
        "case_config_path": str(config_path) if config_path is not None else None,
    }


def _skipped_arm(label: str) -> Dict[str, Any]:
    return {
        "arm": label,
        "status": "skipped",
        "run_id": None,
        "workdir": None,
        "primary_or": None,
        "direction_match": None,
        "expected_direction": None,
        "icu_findings": {},
        "workflow_hits": {},
        "artifact_hits": {},
        "n_findings": 0,
        "n_warnings": 0,
        "n_errors": 0,
        "evidence_count": 0,
        "evidence_kinds": {
            "kinds_seen": [],
            "kinds_missing": sorted(_REQUIRED_KINDS),
            "complete": False,
        },
        "evidence_missing_in_manuscript": None,
        "elapsed_seconds": 0.0,
    }


def _arm_was_run(score: Optional[Dict[str, Any]]) -> bool:
    return bool(score) and score.get("status") != "skipped"


def _run_arms_in_scores(scores: List[Dict[str, Any]]) -> List[str]:
    arms = [
        arm
        for arm in _ARM_ORDER
        if any(_arm_was_run(s.get(arm)) for s in scores)
    ]
    return arms or list(_ARM_ORDER)


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _evidence_missing_count(run_dir: Path) -> int:
    bound = run_dir / "manuscript_scaffold_bound.md"
    if not bound.exists():
        return -1
    return bound.read_text(encoding="utf-8").count("[evidence missing:")


def _findings_join(manifest: Dict[str, Any]) -> str:
    return " || ".join(f.get("message", "") for f in manifest.get("findings", []))


_LOGISTIC_METHODS = {
    "logistic",
    "logistic_regression",
    "logit",
    "glm_binomial",
}

_BINARY_CONTRAST_PREDICTORS = {
    "sex",
    "gender",
    "vaso",
    "vasopressor",
    "mech_vent",
    "mechanical_ventilation",
    "vent",
}

_NON_OR_BENCHMARK_TOKENS = {
    "auroc",
    "brier",
    "cox",
    "hazard",
    "hr",
    "linear",
    "los",
    "prediction",
    "survival",
    "time_to",
}


def _finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _non_or_benchmark(*, item_key: str = "", research_question: str = "") -> bool:
    blob = f"{item_key} {research_question}".lower()
    return any(token in blob for token in _NON_OR_BENCHMARK_TOKENS)


def _primary_or_from_robustness_panel(run_dir: Path) -> Optional[float]:
    panel_path = run_dir / "robustness_panel.json"
    if not panel_path.exists():
        return None
    try:
        panel = json.loads(panel_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for row in panel.get("rows", []) or []:
        if row.get("spec_id") == panel.get("primary_spec_id", "primary"):
            value = _finite_float(row.get("point_estimate"))
            if value is not None:
                return value
    return _finite_float(panel.get("primary_point_estimate"))


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _primary_or_from_predictor_trend(
    records: Sequence[Dict[str, Any]], expected_predictor: str
) -> Optional[float]:
    predictor = expected_predictor.strip().lower()
    if not predictor:
        return None
    keys = {
        f"{predictor}_or_per_point",
        f"{predictor}_odds_ratio_per_point",
        f"{predictor}_or_per_unit",
        f"{predictor}_odds_ratio_per_unit",
    }
    for record in records:
        for node in _iter_dicts(record):
            lowered = {str(k).lower(): v for k, v in node.items()}
            for key in keys:
                value = _finite_float(lowered.get(key))
                if value is not None:
                    return value
    return None


def _term_is_single_level_contrast(term: Any, predictor: str) -> bool:
    if not predictor:
        return False
    lowered = str(term or "").strip().lower().replace(" ", "")
    pred = predictor.strip().lower()
    return lowered.startswith(f"{pred}==") or lowered.startswith(f"c({pred})[")


def _primary_or_from_logistic_summary(
    records: Sequence[Dict[str, Any]], expected_predictor: str
) -> Optional[float]:
    predictor = expected_predictor.strip().lower()
    allow_level_contrast = predictor in _BINARY_CONTRAST_PREDICTORS
    for data in records:
        primary_model = data.get("primary_model") or {}
        method = str(
            data.get("method")
            or primary_model.get("model_type")
            or data.get("model_type")
            or ""
        ).strip().lower()
        if method in _LOGISTIC_METHODS and data.get("primary_or") is not None:
            return _finite_float(data.get("primary_or"))
        term = data.get("primary_association_term") or data.get("primary_term")
        value = _finite_float(data.get("primary_association_estimate"))
        if value is not None and (
            allow_level_contrast
            or not _term_is_single_level_contrast(term, predictor)
        ):
            return value
        for node in _iter_dicts(data):
            node_primary_model = node.get("primary_model") or {}
            method = str(
                node.get("method")
                or node.get("model_type")
                or node.get("fit_method")
                or node_primary_model.get("model_type")
                or ""
            ).strip().lower()
            if method and method not in _LOGISTIC_METHODS and "logit" not in method:
                continue
            term = (
                node.get("primary_association_term")
                or node.get("primary_term")
                or data.get("primary_association_term")
            )
            value = _finite_float(node.get("primary_or"))
            if value is not None and (
                allow_level_contrast
                or not _term_is_single_level_contrast(term, predictor)
            ):
                return value
    return None


def _primary_or(
    run_dir: Path,
    *,
    expected_predictor: str = "",
    item_key: str = "",
    research_question: str = "",
) -> Optional[float]:
    """Return the manuscript-facing primary odds ratio for OR benchmarks.

    The benchmark scores the effect the agent actually surfaced in the
    manuscript. Prefer the robustness-panel primary row because writer-facing
    claims are registered from that canonical panel. Fall back to explicit
    per-point predictor trends, then to unambiguous logistic summaries.
    """
    if _non_or_benchmark(item_key=item_key, research_question=research_question):
        return None
    panel_value = _primary_or_from_robustness_panel(run_dir)
    if panel_value is not None:
        return panel_value
    records = _step_records(run_dir)
    trend_value = _primary_or_from_predictor_trend(records, expected_predictor)
    if trend_value is not None:
        return trend_value
    return _primary_or_from_logistic_summary(records, expected_predictor)


def _direction_match(or_value: Optional[float], expected: int) -> Optional[bool]:
    """+1 → OR > 1 ; -1 → OR < 1. Returns None if no OR was produced."""
    if or_value is None:
        return None
    if expected == +1:
        return or_value > 1.0
    if expected == -1:
        return or_value < 1.0
    return None


def _findings_substring_hits(
    manifest: Dict[str, Any], needles: List[str]
) -> Dict[str, bool]:
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


def _artifact_substring_hits(
    manifest: Dict[str, Any], needles: List[str]
) -> Dict[str, bool]:
    if not needles:
        return {}
    tokens: List[str] = []
    for evidence in manifest.get("evidence", []):
        for key in (
            "evidence_id",
            "kind",
            "description",
            "relative_path",
            "artifact_id",
            "label",
            "path",
            "summary",
        ):
            value = evidence.get(key)
            if value:
                tokens.append(str(value))
        metadata = evidence.get("metadata")
        if isinstance(metadata, dict):
            for value in metadata.values():
                if isinstance(value, (str, int, float, bool)):
                    tokens.append(str(value))
                elif isinstance(value, list):
                    tokens.extend(str(item) for item in value if item)
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
    or_value = _primary_or(
        run_dir,
        expected_predictor=getattr(item, "primary_predictor", ""),
        item_key=getattr(item, "key", ""),
        research_question=getattr(item, "research_question", ""),
    )
    return {
        "arm": label,
        "run_id": manifest.get("run_id"),
        "workdir": str(run_dir),
        "primary_or": or_value,
        "direction_match": _direction_match(or_value, item.expected_or_direction),
        "expected_direction": item.expected_or_direction,
        "icu_findings": _findings_substring_hits(
            manifest, item.expected_finding_substrings
        ),
        "workflow_hits": _step_substring_hits(
            run_dir, getattr(item, "expected_step_substrings", [])
        ),
        "artifact_hits": _artifact_substring_hits(
            manifest, getattr(item, "expected_artifact_substrings", [])
        ),
        "n_findings": len(manifest.get("findings", [])),
        "n_warnings": sum(
            1 for f in manifest.get("findings", []) if f.get("severity") == "warning"
        ),
        "n_errors": sum(
            1 for f in manifest.get("findings", []) if f.get("severity") == "error"
        ),
        "evidence_count": len(manifest.get("evidence", [])),
        "evidence_kinds": _kinds_complete(manifest),
        "evidence_missing_in_manuscript": _evidence_missing_count(run_dir),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _find_resumable_run(workdir: Path) -> Optional[str]:
    """Return the run_id of an interrupted run that can be resumed.

    A run is resumable when it wrote a ``manifest_partial.json`` checkpoint but
    did not reach ``execution_complete`` — e.g. a provider 502 / quota outage
    killed it mid-step. ResearchAgentPipeline.run(resume_run_id=...) then skips
    the already-completed steps instead of redoing the whole analysis.
    """
    candidates = []
    for partial in workdir.glob("run_*/manifest_partial.json"):
        run_dir = partial.parent
        rs = run_dir / "run_status.json"
        if rs.exists():
            try:
                gates = json.loads(rs.read_text(encoding="utf-8")).get(
                    "gates", {})
                if gates.get("execution_complete"):
                    continue  # already finished — nothing to resume
            except (json.JSONDecodeError, OSError):
                pass
        candidates.append(run_dir.name)
    return sorted(candidates)[-1] if candidates else None


def _run_one_arm(
    *,
    item,
    cohort,
    workdir: Path,
    disable_icu_context: bool,
    label: str,
    llm,
    pipeline_options: Optional[Dict[str, Any]] = None,
    reuse_existing: bool = False,
    force_writer_probe: bool = False,
) -> Dict[str, Any]:
    from easyicu.research_agent import ResearchAgentPipeline  # type: ignore

    workdir.mkdir(parents=True, exist_ok=True)
    pipeline = ResearchAgentPipeline(
        workdir=workdir,
        llm=llm,
        disable_icu_context=disable_icu_context,
        **dict(pipeline_options or {}),
    )
    resume_run_id = _find_resumable_run(workdir) if reuse_existing else None
    if resume_run_id:
        print(f"[research_agent] resuming interrupted run {resume_run_id} "
              f"(step-level checkpoint) for {item.key}/{label}")
    started = time.monotonic()
    result = pipeline.run(
        question=item.research_question,
        cohort=cohort,
        cohort_name=f"bench_{item.key}",
        database="bench",
        target_outcome=item.target_outcome,
        inclusion_criteria=item.inclusion_criteria,
        resume_run_id=resume_run_id,
        force_writer_probe=bool(force_writer_probe),
    )
    elapsed = time.monotonic() - started
    score = _score_arm(run_dir=Path(result.workdir), item=item, label=label)
    score["elapsed_seconds"] = round(elapsed, 2)
    return score


def _run_one_item(
    *,
    item,
    seed: int,
    out_root: Path,
    llm,
    arms: Sequence[str],
    pipeline_options: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    force_writer_probe: bool = False,
) -> Dict[str, Any]:
    if verbose:
        print(f"\n=== {item.key} — {item.name} ===")
    cohort = item.cohort_factory(seed)
    item_root = out_root / item.key
    selected = set(_normalize_arms(arms))
    naive = _skipped_arm("naive")
    aware = _skipped_arm("aware")
    if "naive" in selected:
        naive = _run_one_arm(
            item=item,
            cohort=cohort.copy(),
            workdir=item_root / "naive",
            disable_icu_context=True,
            label="naive",
            llm=llm,
            pipeline_options=pipeline_options,
            force_writer_probe=force_writer_probe,
        )
    if "aware" in selected:
        aware = _run_one_arm(
            item=item,
            cohort=cohort.copy(),
            workdir=item_root / "aware",
            disable_icu_context=False,
            label="aware",
            llm=llm,
            pipeline_options=pipeline_options,
            force_writer_probe=force_writer_probe,
        )
    payload = {
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
    }
    if "naive" in selected:
        payload["naive"] = naive
    if "aware" in selected:
        payload["aware"] = aware
    return payload


def _reuse_arm_if_complete(
    *, arm_dir: Path, item, label: str
) -> Optional[Dict[str, Any]]:
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
    arms: Sequence[str],
    pipeline_options: Optional[Dict[str, Any]],
    reuse_existing: bool,
    verbose: bool = True,
    force_writer_probe: bool = False,
) -> Dict[str, Any]:
    if verbose:
        print(f"\n=== {item.key} — {item.name} ===")
    cohort = item.cohort_factory(seed)
    item_root = out_root / item.key
    selected = set(_normalize_arms(arms))

    naive = _skipped_arm("naive")
    aware = _skipped_arm("aware")
    if reuse_existing:
        if "naive" in selected:
            naive = _reuse_arm_if_complete(
                arm_dir=item_root / "naive", item=item, label="naive"
            ) or _skipped_arm("naive")
        if "aware" in selected:
            aware = _reuse_arm_if_complete(
                arm_dir=item_root / "aware", item=item, label="aware"
            ) or _skipped_arm("aware")

    if "naive" in selected and not _arm_was_run(naive):
        naive = _run_one_arm(
            item=item,
            cohort=cohort.copy(),
            workdir=item_root / "naive",
            disable_icu_context=True,
            label="naive",
            llm=llm,
            pipeline_options=pipeline_options,
            force_writer_probe=force_writer_probe,
        )
    if "aware" in selected and not _arm_was_run(aware):
        aware = _run_one_arm(
            item=item,
            cohort=cohort.copy(),
            workdir=item_root / "aware",
            disable_icu_context=False,
            label="aware",
            llm=llm,
            pipeline_options=pipeline_options,
            force_writer_probe=force_writer_probe,
        )

    payload = {
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
    }
    if "naive" in selected:
        payload["naive"] = naive
    if "aware" in selected:
        payload["aware"] = aware
    return payload


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-arm aggregate metrics across all bench items."""
    aggregate_arms = _run_arms_in_scores(scores) if scores else list(_ARM_ORDER)
    totals: Dict[str, Dict[str, int]] = {arm: {} for arm in aggregate_arms}
    for arm in aggregate_arms:
        arm_scores = [s for s in scores if _arm_was_run(s.get(arm))]
        n_total = len(arm_scores)
        n_dir_correct = sum(
            1 for s in arm_scores if s[arm]["direction_match"] is True
        )
        n_dir_wrong = sum(
            1 for s in arm_scores if s[arm]["direction_match"] is False
        )
        n_dir_missing = sum(
            1 for s in arm_scores if s[arm]["direction_match"] is None
        )
        n_findings_full_hit = 0
        n_findings_partial = 0
        for s in arm_scores:
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
        for s in arm_scores:
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
        n_kinds_complete = sum(
            1 for s in arm_scores if s[arm]["evidence_kinds"]["complete"]
        )
        evidence_missing = sum(
            max(0, s[arm]["evidence_missing_in_manuscript"])
            for s in arm_scores
            if s[arm]["evidence_missing_in_manuscript"] is not None
        )
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


def _fmt_missing(value: Any) -> str:
    if value is None:
        return "—"
    return str(value)


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


def _render_markdown(
    *, scores: List[Dict[str, Any]], totals: Dict[str, Any], seed: int, bench_kind: str
) -> str:
    label = _bench_label(scores)
    ran_arms = _run_arms_in_scores(scores)
    lines: List[str] = [
        f"# {label} — research agent benchmark",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()} (seed={seed}, bench_kind={bench_kind})._",
        "",
        "Each item runs the *same* cohort through `ResearchAgentPipeline` "
        "using the requested arm(s): "
        + ", ".join(f"`{arm}` ({_ARM_LABELS[arm]})" for arm in ran_arms)
        + ". Scores cover direction match, ICU-rule findings, evidence "
        "completeness and manuscript bindability.",
        "",
        "**Interpretation boundary.** All analysis-bench tasks use synthetic cohorts. "
        "`evidence_basis` describes how a task was designed (for example, literature-inspired, "
        "consensus-inspired, or internal stress-test synthetic); it does **not** mean the benchmark "
        "finding itself is externally validated. Substring-matched ICU findings are benchmark-rule hits, "
        "not stand-alone publishable clinical claims.",
        "",
        "## Per-item results",
        "",
    ]
    if ran_arms == list(_ARM_ORDER):
        lines.extend(
            [
                "| Item | Family | Difficulty | Evidence basis | Direction (naive) | Direction (aware) | OR (naive) | OR (aware) | Predefined rule hits (naive) | Predefined rule hits (aware) | Workflow hits (naive) | Workflow hits (aware) | Artifact hits (naive) | Artifact hits (aware) | `[evidence missing]` (naive → aware) |",
                "|---|---|---|---|:-:|:-:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|---:|",
            ]
        )
        for s in scores:
            n = s["naive"]
            a = s["aware"]
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
                f"| {_fmt_missing(n['evidence_missing_in_manuscript'])} → {_fmt_missing(a['evidence_missing_in_manuscript'])} |"
            )
    else:
        arm = ran_arms[0]
        label_name = _ARM_LABELS[arm].lower()
        lines.extend(
            [
                f"| Item | Family | Difficulty | Evidence basis | Direction ({arm}) | OR ({arm}) | Predefined rule hits ({arm}) | Workflow hits ({arm}) | Artifact hits ({arm}) | `[evidence missing]` ({arm}) |",
                "|---|---|---|---|:-:|---:|:-:|:-:|:-:|---:|",
            ]
        )
        for s in scores:
            arm_score = s[arm]
            lines.append(
                f"| `{s['item_key']}` "
                f"| `{s.get('benchmark_family', 'rule')}` "
                f"| `{s.get('difficulty', 'basic')}` "
                f"| `{s.get('evidence_basis', 'internal_synthetic')}` "
                f"| {_direction_marker(arm_score['direction_match'])} "
                f"| {_fmt_or(arm_score['primary_or'])} "
                f"| {_findings_marker(arm_score['icu_findings'])} "
                f"| {_findings_marker(arm_score.get('workflow_hits', {}))} "
                f"| {_findings_marker(arm_score.get('artifact_hits', {}))} "
                f"| {_fmt_missing(arm_score['evidence_missing_in_manuscript'])} |"
            )
        lines.append("")
        lines.append(
            f"_Only the {label_name} arm was executed; skipped arms were not called._"
        )
    lines.append("")

    lines.append("## Aggregate (across all items)")
    lines.append("")
    lines.append("| Metric | " + " | ".join(_ARM_LABELS[arm] for arm in ran_arms) + " |")
    lines.append("|---" + "|---:" * len(ran_arms) + "|")
    rows = [
        ("Number of items", "n_items"),
        ("Direction correct", "direction_correct"),
        ("Direction wrong", "direction_wrong"),
        ("Direction missing (no OR produced)", "direction_missing"),
        ("Items with all predefined rule hits", "icu_findings_full_hit"),
        ("Items with partial predefined rule hits", "icu_findings_partial_hit"),
        ("Items with all workflow expectations hit", "workflow_full_hit"),
        ("Items with partial workflow expectations", "workflow_partial_hit"),
        ("Items with all artifact expectations hit", "artifact_full_hit"),
        ("Items with partial artifact expectations", "artifact_partial_hit"),
        ("Items with all 5 evidence kinds", "evidence_kinds_complete"),
        ("Total `[evidence missing]` lines (lower is better)", "evidence_missing_in_manuscripts"),
    ]
    for name, key in rows:
        values = [str(totals[arm][key]) for arm in ran_arms]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines.append("")

    lines.append("## Interpretation Notes")
    lines.append("")
    for s in scores:
        note = (
            s.get("interpretation_note")
            or "Interpret only as an internal benchmark result."
        )
        lines.append(
            f"- **{s['item_key']}** — `{s.get('claim_scope', 'internal_benchmark_only')}`. {note}"
        )
    lines.append("")

    lines.append("## Per-item provenance")
    lines.append("")
    for s in scores:
        parts = [
            f"`{s[arm]['workdir']}` ({arm})"
            for arm in ran_arms
            if _arm_was_run(s.get(arm))
        ]
        lines.append(f"- **{s['item_key']}** — " + " ; ".join(parts))
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
    from easyicu.research_agent.llm import openrouter_reasoning_extra_body  # type: ignore

    if provider == "mock":
        return MockLLMClient()
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is required for --provider openrouter")
        kwargs = dict(
            model=model,
            api_key=key,
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            request_timeout=float(request_timeout),
            extra_headers={
                "HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
                "X-Title": "EasyICU research-agent benchmark",
            },
        )
        extra_body = openrouter_reasoning_extra_body(model)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        return OpenAIClient(**kwargs)
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not key and not _local_openai_base_url(base_url):
            raise SystemExit("OPENAI_API_KEY is required for --provider openai")
        kwargs: Dict[str, Any] = {
            "model": model,
            "request_timeout": float(request_timeout),
        }
        if key:
            kwargs["api_key"] = key
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIClient(**kwargs)
    raise SystemExit(f"Unsupported provider: {provider}")


def _benchmark_pipeline_options(
    *,
    max_total_steps: Optional[int],
    disable_replanning: bool,
    max_code_repair_attempts: Optional[int],
    enable_repro_envelope: bool = True,
    llm_seed: Optional[int] = None,
    writer_digest_widened: bool = False,
    strict_evidence: bool = False,
    submission_profile: Optional["SubmissionProfile"] = None,
    runner_kind: Optional[str] = None,
) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    if submission_profile:
        options.update(submission_profile.pipeline_options())
    if runner_kind:
        # Code-execution backend. ``docker`` swaps in DockerRunner
        # (``--network none`` + read-only cohort mount); recorded here so
        # the run manifest / bench_results.json document the execution
        # isolation the manuscript Methods section cites.
        options["runner_kind"] = runner_kind
    if max_total_steps is not None:
        options["max_total_steps"] = int(max_total_steps)
    if disable_replanning:
        options["enable_replanning"] = False
    if max_code_repair_attempts is not None:
        options["max_code_repair_attempts"] = int(max_code_repair_attempts)
    if strict_evidence:
        options["evidence_enforcement_mode"] = "strict"
    if enable_repro_envelope:
        # Default ON for bench runs so the per-call envelope
        # (temperature / requested_top_p / seed / model / prompt+response
        # SHA256) lands as reproducibility_envelope.json next to each
        # arm's run_status.json.
        options["enable_reproducibility_envelope"] = True
    if writer_digest_widened:
        options["writer_digest_widened"] = True
    if llm_seed is not None:
        options["llm_seed"] = int(llm_seed)
    return options


def _enforce_submission_profile_arms(
    arms: Sequence[str],
    *,
    profile: Optional["SubmissionProfile"],
) -> List[str]:
    selected = _normalize_arms(arms)
    if profile is not None and selected != [profile.requires_arm]:
        raise SystemExit(
            "Submission profile is paper-facing and must run the full "
            f"EasyICU workflow only: pass '--arms {profile.requires_arm}'. Use "
            "the historical naive arm only for an explicit ablation or "
            "reviewer-response run."
        )
    return selected


def _enforce_submission_profile_runner(
    runner: Optional[str],
    *,
    profile: Optional["SubmissionProfile"],
    allow_host_runner: bool = False,
) -> str:
    """Resolve and gate the code-execution backend for a benchmark run.

    Paper-facing submission profiles require a containerised runner so
    agent-generated code executes under ``docker run --network none``
    with a read-only cohort mount, never on the host subprocess. With no
    profile, the host ``subprocess`` runner stays the default. The
    ``--allow-host-runner`` escape hatch exists for offline development
    and is never valid for an archival/canonical batch.
    """
    if profile is None:
        return (runner or "subprocess").lower()
    required = (profile.requires_runner or "docker").lower()
    resolved = (runner or required).lower()
    if resolved != required and not allow_host_runner:
        raise SystemExit(
            f"Submission profile '{profile.ref}' is paper-facing and must "
            f"execute agent-generated code in a network-isolated sandbox: "
            f"pass '--runner {required}'. Build the image first with "
            "`docker build -t easyicu-research-agent:latest -f "
            "src/easyicu/research_agent/runner_image/Dockerfile .`. For a "
            "non-archival development run only, pass '--allow-host-runner'."
        )
    return resolved


def _enforce_mock_aware_provider(
    arms: Sequence[str],
    *,
    provider: str,
    allow_mock_aware: bool = False,
) -> None:
    """Reject mock-provider aware runs unless they are explicit smoke tests."""
    selected_arms = _normalize_arms(arms)
    # The MockLLMClient returns canned responses, so an "aware" arm run on
    # the mock provider reports fixture output rather than a genuine
    # ICU-aware analysis. Paper-facing results must use a real provider.
    # Offline plumbing smoke tests can opt in explicitly with
    # --allow-mock-aware.
    if "aware" in selected_arms and provider == "mock" and not allow_mock_aware:
        raise SystemExit(
            "The 'aware' arm on the mock provider returns pre-written, "
            "fixture responses, so its results are not real. Use "
            "--provider openrouter/openai for paper-facing runs, restrict to "
            "--arms naive, or pass --allow-mock-aware for an offline plumbing "
            "smoke test (results are non-substantive)."
        )


def _run_suite(
    *,
    items: Sequence[Any],
    out_root: Path,
    seed: int,
    bench_kind: str,
    provider: str,
    model: str,
    arms: Sequence[str],
    pipeline_options: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    request_timeout: float = 180.0,
    reuse_existing: bool = False,
    case_registration: Optional[Dict[str, Any]] = None,
    force_writer_probe: bool = False,
    allow_mock_aware: bool = False,
) -> Dict[str, Any]:
    selected_arms = _normalize_arms(arms)
    _enforce_mock_aware_provider(
        selected_arms,
        provider=provider,
        allow_mock_aware=allow_mock_aware,
    )
    llm = _make_llm(provider=provider, model=model, request_timeout=request_timeout)
    from easyicu.research_agent import (  # type: ignore
        default_icu_agent_bench_suite,
        icu_agent_bench_markdown,
    )

    scores: List[Dict[str, Any]] = []
    for item in items:
        scores.append(
            _run_one_item_with_reuse(
                item=item,
                seed=seed,
                out_root=out_root,
                llm=llm,
                arms=selected_arms,
                pipeline_options=pipeline_options,
                reuse_existing=reuse_existing,
                force_writer_probe=force_writer_probe,
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
        "arms": selected_arms,
        "case_registration": case_registration,
        "force_writer_probe": bool(force_writer_probe),
        "pipeline_options": dict(pipeline_options or {}),
        "items": [it.key for it in items],
        "scores": scores,
        "totals": totals,
        "icu_agent_bench_suite": default_icu_agent_bench_suite().model_dump(
            mode="json"
        ),
    }
    (out_root / "bench_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    md = _render_markdown(
        scores=scores, totals=totals, seed=seed, bench_kind=bench_kind
    )
    header = [
        f"_Provider: `{provider}`_",
        f"_Model: `{model}`_",
        "",
    ]
    (out_root / "bench_results.md").write_text("\n".join(header) + md, encoding="utf-8")
    suite = default_icu_agent_bench_suite()
    (out_root / "icu_agent_bench_suite.json").write_text(
        suite.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (out_root / "icu_agent_bench_suite.md").write_text(
        icu_agent_bench_markdown(suite),
        encoding="utf-8",
    )
    return payload


def _render_model_matrix(runs: List[Dict[str, Any]]) -> str:
    ran_arms = [
        arm
        for arm in _ARM_ORDER
        if any(run.get("totals", {}).get(arm, {}).get("n_items", 0) > 0 for run in runs)
    ] or list(_ARM_ORDER)
    metric_columns: List[tuple[str, str, str]] = []
    for arm in ran_arms:
        suffix = _ARM_LABELS[arm]
        metric_columns.extend(
            [
                (f"Direction correct ({suffix})", arm, "direction_correct"),
                (f"ICU findings full-hit ({suffix})", arm, "icu_findings_full_hit"),
                (f"Workflow full-hit ({suffix})", arm, "workflow_full_hit"),
                (f"Artifact full-hit ({suffix})", arm, "artifact_full_hit"),
                (f"Evidence missing ({suffix})", arm, "evidence_missing_in_manuscripts"),
            ]
        )
    lines = [
        "# Benchmark model matrix",
        "",
        "| Model | Provider | Bench kind | "
        + " | ".join(name for name, _, _ in metric_columns)
        + " |",
        "|---|---|---" + "|---:" * len(metric_columns) + "|",
    ]
    for run in runs:
        totals = run["totals"]
        values: List[str] = []
        for _, arm, key in metric_columns:
            if key == "direction_correct":
                values.append(
                    f"{totals[arm]['direction_correct']}/{totals[arm]['n_items']}"
                )
            else:
                values.append(str(totals[arm].get(key, 0)))
        lines.append(
            f"| `{run['model']}` | `{run['provider']}` | `{run.get('bench_kind', 'rule')}` | "
            + " | ".join(values)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    _bootstrap_imports()

    from tests.bench import ANALYSIS_BENCH_ITEMS, RULE_BENCH_ITEMS  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-kind",
        choices=["rule", "analysis"],
        default="rule",
        help="Which benchmark fixture family to run.",
    )
    parser.add_argument(
        "--items",
        nargs="+",
        default=None,
        help="Subset of bench item keys to run (default: all).",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Synthetic-cohort seed (deterministic)."
    )
    parser.add_argument(
        "--out-root", default=str((Path.cwd() / "research_output" / "bench").resolve())
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "openrouter", "openai"],
        default="mock",
        help="LLM backend for the benchmark arms.",
    )
    parser.add_argument(
        "--allow-mock-aware",
        action="store_true",
        help=(
            "Permit the 'aware' arm to run on the mock provider for an offline "
            "plumbing smoke test. Results are non-substantive (canned responses) "
            "and must not be used for paper-facing benchmark results."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EASYICU_HOSTED_DEFAULT_MODEL", "openai/gpt-oss-120b:free"),
        help="Single model name for real-provider runs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional multiple model names. When set, the benchmark runs once per model.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="Per-request timeout for real LLM providers.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=list(_ARM_ORDER),
        default=list(_ARM_ORDER),
        help="Benchmark arm(s) to run. Use '--arms aware' for platform-only runs.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse completed item/arm runs already present under --out-root.",
    )
    parser.add_argument(
        "--max-total-steps",
        type=int,
        default=None,
        help=(
            "Override ResearchAgentPipeline max_total_steps. For cheap "
            "development dry runs, use a small value such as 4."
        ),
    )
    parser.add_argument(
        "--disable-replanning",
        action="store_true",
        help=(
            "Disable LLM replanning inside ResearchAgentPipeline. Useful for "
            "low-cost dry runs before protocol freeze."
        ),
    )
    parser.add_argument(
        "--max-code-repair-attempts",
        type=int,
        default=None,
        help="Override the per-step generated-code repair attempt budget.",
    )
    parser.add_argument(
        "--no-repro-envelope",
        action="store_true",
        help=(
            "Disable the LLM reproducibility envelope (O20). The envelope "
            "is ON by default for bench runs because the manuscript cites "
            "per-call temperature / seed / top_p / model and SHA256 hashes."
        ),
    )
    parser.add_argument(
        "--writer-digest-widened",
        action="store_true",
        help=(
            "Expose primary, secondary, and derived numeric claims to the "
            "writer. Kept opt-in for compatibility; submission profile "
            "enables it automatically."
        ),
    )
    parser.add_argument(
        "--strict-evidence",
        action="store_true",
        help=(
            "Fail the run if manuscript evidence placeholders or numeric "
            "claims cannot be bound. Submission profile enables this "
            "automatically."
        ),
    )
    parser.add_argument(
        "--submission-profile",
        action="store_true",
        help=(
            "Use paper-facing canonical options: require '--arms aware', "
            "strict evidence, reproducibility envelope, and widened writer "
            "digest."
        ),
    )
    parser.add_argument(
        "--profile",
        default=_DEFAULT_SUBMISSION_PROFILE_REF,
        help=(
            "Versioned submission profile ref used with --submission-profile "
            f"(default: {_DEFAULT_SUBMISSION_PROFILE_REF})."
        ),
    )
    parser.add_argument(
        "--runner",
        choices=["subprocess", "docker"],
        default=None,
        help=(
            "Code-execution backend for agent-generated scripts. "
            "'subprocess' runs on the host (default for dev). 'docker' uses "
            "the network-isolated container runner. A submission profile "
            "requires 'docker'; omit this flag under --submission-profile to "
            "default to it."
        ),
    )
    parser.add_argument(
        "--allow-host-runner",
        action="store_true",
        help=(
            "Development escape hatch: permit the host subprocess runner "
            "under --submission-profile. Never valid for an archival/"
            "canonical batch."
        ),
    )
    parser.add_argument(
        "--case",
        default=None,
        help=(
            "Optional case protocol directory name under benchmark/cases. "
            "When set, case-owned cohort patterns are registered before "
            "planning. Example: case_b_sofa2_sepsis."
        ),
    )
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=None,
        help=(
            "Optional integer seed forwarded to OpenAI-compatible providers "
            "via the chat-completions `seed` field. Recorded in the "
            "reproducibility envelope regardless of provider honoring it."
        ),
    )
    parser.add_argument(
        "--ehrflowbench-jsonl",
        default=None,
        help="Optional EHRFlowBench-style JSONL export. Each row may include "
        "key, question, cohort_path, target_outcome, expected_or_direction.",
    )
    parser.add_argument(
        "--force-writer-probe",
        action="store_true",
        help=(
            "Diagnostic engineering use only: force writer output even when "
            "the execution gate fails. Do NOT use for archival benchmarks."
        ),
    )
    args = parser.parse_args()
    case_registration = _register_case_patterns(args.case)
    submission_profile = (
        _resolve_submission_profile(args.profile)
        if bool(args.submission_profile)
        else None
    )
    args.arms = _enforce_submission_profile_arms(
        args.arms,
        profile=submission_profile,
    )
    runner_kind = _enforce_submission_profile_runner(
        getattr(args, "runner", None),
        profile=submission_profile,
        allow_host_runner=bool(getattr(args, "allow_host_runner", False)),
    )
    if (
        submission_profile is not None
        and runner_kind != submission_profile.requires_runner
    ):
        print(
            f"[research_agent] WARNING: submission profile "
            f"'{submission_profile.ref}' run on host '{runner_kind}' runner via "
            "--allow-host-runner; this batch is NOT archival/canonical."
        )
    pipeline_options = _benchmark_pipeline_options(
        max_total_steps=args.max_total_steps,
        disable_replanning=bool(args.disable_replanning),
        max_code_repair_attempts=args.max_code_repair_attempts,
        enable_repro_envelope=not bool(getattr(args, "no_repro_envelope", False)),
        llm_seed=getattr(args, "llm_seed", None),
        writer_digest_widened=bool(args.writer_digest_widened),
        strict_evidence=bool(args.strict_evidence),
        submission_profile=submission_profile,
        runner_kind=runner_kind,
    )

    if args.ehrflowbench_jsonl:
        ehrflow_model = args.model if args.provider != "mock" else "mock"
        if args.models:
            ehrflow_model = args.models[0]
        return _run_ehrflowbench_jsonl(
            jsonl_path=Path(args.ehrflowbench_jsonl).resolve(),
            out_root=Path(args.out_root).resolve(),
            seed=args.seed,
            arms=args.arms,
            pipeline_options=pipeline_options,
            provider=args.provider,
            model=ehrflow_model,
            request_timeout=float(args.request_timeout),
            reuse_existing=bool(args.reuse_existing),
            force_writer_probe=bool(args.force_writer_probe),
            allow_mock_aware=bool(args.allow_mock_aware),
        )

    all_items = list(
        RULE_BENCH_ITEMS if args.bench_kind == "rule" else ANALYSIS_BENCH_ITEMS
    )
    if args.items:
        items = [it for it in all_items if it.key in set(args.items)]
        unknown = set(args.items) - {it.key for it in all_items}
        if unknown:
            print(
                f"Unknown bench keys: {sorted(unknown)}; "
                f"available: {[it.key for it in all_items]}"
            )
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
        model_root = (
            out_root if len(models) == 1 else (out_root / _slugify_model(model))
        )
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
            arms=args.arms,
            pipeline_options=pipeline_options,
            request_timeout=float(args.request_timeout),
            reuse_existing=bool(args.reuse_existing),
            case_registration=case_registration,
            force_writer_probe=bool(args.force_writer_probe),
            allow_mock_aware=bool(args.allow_mock_aware),
        )
        all_runs.append(payload)
        totals = payload["totals"]
        print()
        print(f"=== Bench complete — {model} ===")
        print(f"  -> {model_root / 'bench_results.json'}")
        print(f"  -> {model_root / 'bench_results.md'}")
        ran_arms = _run_arms_in_scores(payload["scores"])
        for arm in ran_arms:
            print(
                f"  Direction correct  — {arm}: {totals[arm]['direction_correct']}/"
                f"{totals[arm]['n_items']}"
            )
            print(
                f"  ICU findings full  — {arm}: {totals[arm]['icu_findings_full_hit']}"
            )
            print(
                f"  Evidence missing   — {arm}: "
                f"{totals[arm]['evidence_missing_in_manuscripts']}"
            )

    if len(all_runs) > 1:
        matrix_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "bench_kind": args.bench_kind,
            "provider": args.provider,
            "arms": _normalize_arms(args.arms),
            "case_registration": case_registration,
            "pipeline_options": pipeline_options,
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


def _ehrflow_item_done(item_root: Path) -> bool:
    """True if this item already has a run that reached execution_complete.

    Used for resume: a quota 502 mid-batch should not force re-running items
    that already finished cleanly. Quota-disrupted (incomplete) runs return
    False so they are redone.
    """
    for rs in item_root.glob("*/run_*/run_status.json"):
        try:
            gates = json.loads(rs.read_text(encoding="utf-8")).get("gates", {})
        except (json.JSONDecodeError, OSError):
            continue
        if gates.get("execution_complete"):
            return True
    return False


def _run_ehrflowbench_jsonl(
    *,
    jsonl_path: Path,
    out_root: Path,
    seed: int,
    arms: Sequence[str],
    pipeline_options: Optional[Dict[str, Any]] = None,
    provider: str = "mock",
    model: str = "mock",
    request_timeout: float = 180.0,
    reuse_existing: bool = False,
    force_writer_probe: bool = False,
    allow_mock_aware: bool = False,
) -> int:
    """Run an external EHRFlowBench-style JSONL export when available."""
    from types import SimpleNamespace
    import pandas as pd

    _enforce_mock_aware_provider(
        arms,
        provider=provider,
        allow_mock_aware=allow_mock_aware,
    )
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
            rows.append(
                {"status": "invalid_json", "error": str(exc), "raw": line[:200]}
            )

    scores: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        key = str(row.get("key") or row.get("id") or f"ehrflowbench_{idx:03d}")
        cohort_path = row.get("cohort_path") or row.get("cohort")
        question = row.get("question") or row.get("research_question")
        target = row.get("target_outcome") or row.get("outcome")
        if not cohort_path or not question or not target:
            pending.append(
                {
                    "key": key,
                    "status": "pending_missing_fields",
                    "required": ["question", "cohort_path", "target_outcome"],
                }
            )
            continue
        path = Path(str(cohort_path)).expanduser().resolve()
        if not path.exists():
            pending.append(
                {
                    "key": key,
                    "status": "pending_missing_cohort",
                    "cohort_path": str(path),
                }
            )
            continue
        if path.suffix.lower() in {".parquet", ".pq"}:
            cohort = pd.read_parquet(path)
        elif path.suffix.lower() in {".csv", ".tsv"}:
            cohort = pd.read_csv(
                path, sep=("\t" if path.suffix.lower() == ".tsv" else ",")
            )
        else:
            pending.append(
                {
                    "key": key,
                    "status": "unsupported_cohort_format",
                    "cohort_path": str(path),
                }
            )
            continue
        item = SimpleNamespace(
            key=key,
            name=str(row.get("name") or key),
            research_question=str(question),
            target_outcome=str(target),
            primary_predictor=str(row.get("primary_predictor") or ""),
            expected_or_direction=int(row.get("expected_or_direction") or 0),
            expected_finding_substrings=list(
                row.get("expected_finding_substrings") or []
            ),
            inclusion_criteria=list(row.get("inclusion_criteria") or []),
        )
        # Resume support: skip items that already finished cleanly so a quota
        # 502 mid-batch never forces a full redo. An item counts as "done" only
        # if its latest run reached execution_complete — quota-disrupted
        # diagnostic_only runs are redone.
        if reuse_existing and _ehrflow_item_done(out_root / key):
            print(f"\n=== {key} — reuse existing complete run ===")
            pending.append({"key": key, "status": "reused_complete"})
            continue
        # Per-item isolation: a provider 502 / crash on one item must not abort
        # the remaining items. Record the failure and continue.
        try:
            score = _run_one_item_from_cohort(
                item=item,
                cohort=cohort,
                out_root=out_root,
                arms=arms,
                pipeline_options=pipeline_options,
                provider=provider,
                model=model,
                request_timeout=request_timeout,
                reuse_existing=reuse_existing,
                force_writer_probe=force_writer_probe,
            )
            scores.append(score)
        except Exception as exc:  # noqa: BLE001 — keep batch alive on 502/etc.
            print(f"[ehrflowbench] item {key} FAILED: {type(exc).__name__}: "
                  f"{str(exc)[:200]}")
            pending.append({
                "key": key,
                "status": "item_exception",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            })
            continue

    totals = _aggregate(scores) if scores else {"naive": {}, "aware": {}}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(jsonl_path),
        "seed": seed,
        "arms": _normalize_arms(arms),
        "pipeline_options": dict(pipeline_options or {}),
        "force_writer_probe": bool(force_writer_probe),
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
        md.append(
            _render_markdown(
                scores=scores, totals=totals, seed=seed, bench_kind="external"
            )
        )
    if pending:
        md.extend(["", "## Pending", ""])
        for p in pending:
            md.append(f"- `{p['key']}` — {p['status']}")
    (out_root / "ehrflowbench_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  -> {out_root / 'ehrflowbench_results.json'}")
    print(f"  -> {out_root / 'ehrflowbench_results.md'}")
    return 0


def _run_one_item_from_cohort(
    *,
    item,
    cohort,
    out_root: Path,
    arms: Sequence[str],
    pipeline_options: Optional[Dict[str, Any]] = None,
    provider: str = "mock",
    model: str = "mock",
    request_timeout: float = 180.0,
    reuse_existing: bool = False,
    force_writer_probe: bool = False,
) -> Dict[str, Any]:
    llm = _make_llm(
        provider=provider, model=model, request_timeout=request_timeout
    )
    item_root = out_root / item.key
    selected = set(_normalize_arms(arms))
    naive = _skipped_arm("naive")
    aware = _skipped_arm("aware")
    if "naive" in selected:
        naive = _run_one_arm(
            item=item,
            cohort=cohort.copy(),
            workdir=item_root / "naive",
            disable_icu_context=True,
            label="naive",
            llm=llm,
            pipeline_options=pipeline_options,
            reuse_existing=reuse_existing,
            force_writer_probe=force_writer_probe,
        )
    if "aware" in selected:
        aware = _run_one_arm(
            item=item,
            cohort=cohort.copy(),
            workdir=item_root / "aware",
            disable_icu_context=False,
            label="aware",
            llm=llm,
            pipeline_options=pipeline_options,
            reuse_existing=reuse_existing,
            force_writer_probe=force_writer_probe,
        )
    payload = {
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
    }
    if "naive" in selected:
        payload["naive"] = naive
    if "aware" in selected:
        payload["aware"] = aware
    return payload


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

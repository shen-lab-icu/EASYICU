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
import hashlib
import importlib
import json
import math
import os
import re
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

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


class _JSONLObjectDecodeError(ValueError):
    """Raised when one benchmark JSONL row is not a strict JSON object."""


_FIGURE2_PAPER_ACCEPTANCE_EXIT_CODE = 3


def _is_figure2_task_id(value: object) -> bool:
    """Return True only for an exact frozen Canonical9 identifier."""

    from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS

    return type(value) is str and value in FIGURE2_TASK_IDS


def _operational_exposure_for_item(item: object) -> object:
    """Resolve the execution exposure once without laundering falsey values."""

    declared = getattr(item, "operational_exposure", None)
    if declared is not None:
        return declared
    return getattr(item, "primary_predictor", None)


def _reject_jsonl_duplicate_pairs(
    pairs: Sequence[tuple[str, object]],
) -> Dict[str, object]:
    decoded: Dict[str, object] = {}
    for key, value in pairs:
        if key in decoded:
            raise _JSONLObjectDecodeError(f"duplicate JSON key: {key!r}")
        decoded[key] = value
    return decoded


def _reject_jsonl_nonfinite_constant(value: str) -> object:
    raise _JSONLObjectDecodeError(f"non-finite JSON constant is forbidden: {value}")


def _decode_jsonl_object(line: str) -> Dict[str, Any]:
    """Decode one handoff row without JSON's duplicate/non-finite extensions."""

    try:
        decoded = json.loads(
            line,
            object_pairs_hook=_reject_jsonl_duplicate_pairs,
            parse_constant=_reject_jsonl_nonfinite_constant,
        )
    except _JSONLObjectDecodeError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise _JSONLObjectDecodeError(str(exc)) from exc
    if not isinstance(decoded, dict):
        raise _JSONLObjectDecodeError("benchmark JSONL row must be an object")
    return decoded


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
        return get_submission_profile(profile_ref)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _default_submission_profile_ref() -> str:
    """Resolve the benchmark CLI default from the canonical registry."""

    _bootstrap_imports()
    from easyicu.research_agent.pipeline_profiles import (
        DEFAULT_SUBMISSION_PROFILE_REF,
    )

    return DEFAULT_SUBMISSION_PROFILE_REF


def _register_case_patterns(case_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not case_name:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_]+", case_name):
        raise SystemExit(
            "--case must be a case directory name containing only letters, "
            "numbers, and underscores"
        )
    _bootstrap_imports()
    from easyicu.research_agent.cohort.schema import default_pattern_registry

    module_name = f"benchmark.cases.{case_name}.register_patterns"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Unknown case {case_name!r}: {module_name} not found"
        ) from exc
    register = getattr(module, "register_patterns", None)
    if register is None:
        raise SystemExit(f"Case {case_name!r} must expose register_patterns()")
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
    arms = [arm for arm in _ARM_ORDER if any(_arm_was_run(s.get(arm)) for s in scores)]
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


def _manifest_for_scoring(
    run_dir: Path, manifest: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    if manifest is not None:
        return manifest
    path = run_dir / "manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _latest_active_step_records(
    manifest: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Return the latest successful outer record for each modern-run step.

    ``None`` means the manifest predates ``per_step_records`` and may use the
    legacy filesystem fallback. An existing but empty/malformed field is a
    modern manifest with no active records, so it deliberately returns ``[]``.
    """
    if "per_step_records" not in manifest:
        return None
    raw_records = manifest.get("per_step_records")
    if not isinstance(raw_records, list):
        return []

    latest_by_step: Dict[str, tuple[int, Dict[str, Any]]] = {}
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            continue
        step_id = str(record.get("step_id") or "").strip()
        if step_id:
            latest_by_step[step_id] = (index, record)

    latest = [
        item
        for item in sorted(latest_by_step.values(), key=lambda item: item[0])
        if str(item[1].get("status") or "").strip().lower() == "ok"
    ]
    return [record for _, record in latest]


def _active_evidence_records(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = manifest.get("evidence")
    if not isinstance(evidence, list):
        return []
    records = [item for item in evidence if isinstance(item, dict)]
    active_steps = _latest_active_step_records(manifest)
    if active_steps is None:
        return records

    active_ids = {
        str(evidence_id)
        for record in active_steps
        for evidence_id in (
            record.get("evidence_ids")
            if isinstance(record.get("evidence_ids"), list)
            else []
        )
        if str(evidence_id).strip()
    }
    return [
        record
        for record in records
        if str(record.get("evidence_id") or "") in active_ids
    ]


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
        method = (
            str(
                data.get("method")
                or primary_model.get("model_type")
                or data.get("model_type")
                or ""
            )
            .strip()
            .lower()
        )
        if method in _LOGISTIC_METHODS and data.get("primary_or") is not None:
            return _finite_float(data.get("primary_or"))
        term = data.get("primary_association_term") or data.get("primary_term")
        value = _finite_float(data.get("primary_association_estimate"))
        if value is not None and (
            allow_level_contrast or not _term_is_single_level_contrast(term, predictor)
        ):
            return value
        for node in _iter_dicts(data):
            node_primary_model = node.get("primary_model") or {}
            method = (
                str(
                    node.get("method")
                    or node.get("model_type")
                    or node.get("fit_method")
                    or node_primary_model.get("model_type")
                    or ""
                )
                .strip()
                .lower()
            )
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
    manifest: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Return the manuscript-facing primary odds ratio for OR benchmarks.

    For modern manifests, score only summaries attached to the latest
    successful outer record for each step. Legacy runs without
    ``per_step_records`` retain the historical robustness-panel/filesystem
    fallback. Within the selected summaries, prefer explicit per-point trends
    and then unambiguous logistic summaries.
    """
    if _non_or_benchmark(item_key=item_key, research_question=research_question):
        return None
    scoring_manifest = _manifest_for_scoring(run_dir, manifest)
    modern_manifest = (
        scoring_manifest is not None and "per_step_records" in scoring_manifest
    )
    if not modern_manifest:
        panel_value = _primary_or_from_robustness_panel(run_dir)
        if panel_value is not None:
            return panel_value
    records = _step_records(run_dir, manifest=scoring_manifest)
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


def _step_records(
    run_dir: Path, manifest: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    scoring_manifest = _manifest_for_scoring(run_dir, manifest)
    if scoring_manifest is not None and "per_step_records" in scoring_manifest:
        active_steps = _latest_active_step_records(scoring_manifest) or []
        return [
            summary
            for record in active_steps
            if isinstance((summary := record.get("step_summary")), dict)
        ]

    records: List[Dict[str, Any]] = []
    for ssj in run_dir.rglob("step_summary.json"):
        try:
            records.append(json.loads(ssj.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records


def _step_substring_hits(
    run_dir: Path,
    needles: List[str],
    manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    if not needles:
        return {}
    tokens: List[str] = []
    for record in _step_records(run_dir, manifest=manifest):
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
    for evidence in _active_evidence_records(manifest):
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
    kinds = {e.get("kind") for e in _active_evidence_records(manifest)}
    return {
        "kinds_seen": sorted(k for k in kinds if k),
        "kinds_missing": sorted(_REQUIRED_KINDS - kinds),
        "complete": _REQUIRED_KINDS <= kinds,
    }


def _bench_item_to_task(item):
    """Adapt a ``tests.bench`` BenchItem to a minimal ``ICUAgentBenchTask``.

    This lets the §M1 five-dimension Tier-1 scorecard be computed from a run's
    readiness artifacts for legacy and external bench items too. Legacy items
    without an explicit frozen gold contract remain unscored for numeric result
    validity; external protocol rows may carry that contract structurally. The
    item's ``expected_finding_substrings`` remain a backward-compatible
    audit-hazard answer key. The scorecard is additive and does not replace the
    legacy OR/substring diagnostics.
    """
    from easyicu.research_agent.icu_agent_bench import (  # type: ignore
        ICUAgentBenchGoldAnswer,
        ICUAgentBenchTask,
    )

    def _items(value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return []
        return [str(entry).strip() for entry in value if str(entry).strip()]

    warns = _items(getattr(item, "expected_finding_substrings", []))
    required_warnings = _items(getattr(item, "required_warnings", []))
    explicit_gold = getattr(item, "gold_answer", None)
    gold = None
    if explicit_gold is not None:
        gold = (
            explicit_gold
            if isinstance(explicit_gold, ICUAgentBenchGoldAnswer)
            else ICUAgentBenchGoldAnswer.model_validate(explicit_gold)
        )
    elif any(
        (
            warns,
            required_warnings,
            getattr(item, "numeric_targets", None),
            getattr(item, "forbidden_outputs", None),
        )
    ):
        gold = ICUAgentBenchGoldAnswer(
            numeric_targets=dict(getattr(item, "numeric_targets", {}) or {}),
            required_warnings=list(dict.fromkeys([*warns, *required_warnings])),
            forbidden_outputs=_items(getattr(item, "forbidden_outputs", [])),
            derivation=str(getattr(item, "gold_derivation", "") or ""),
            data_fixture=(getattr(item, "data_fixture", None) or None),
        )
    if gold is not None and (warns or required_warnings):
        merged_warnings = list(
            dict.fromkeys([*gold.required_warnings, *warns, *required_warnings])
        )
        if merged_warnings != gold.required_warnings:
            gold = gold.model_copy(update={"required_warnings": merged_warnings})

    explicit_outputs = _items(getattr(item, "expected_outputs", []))
    legacy_outputs = _items(getattr(item, "expected_artifact_substrings", []))
    semantic_guardrails = _items(getattr(item, "semantic_guardrails", []))
    if not semantic_guardrails:
        semantic_guardrails = list(dict.fromkeys([*warns, *required_warnings]))
    gold_status = str(getattr(item, "gold_answer_status", "") or "").strip().lower()
    if gold_status not in {"planned", "frozen"}:
        gold_status = "frozen" if gold is not None else "planned"
    difficulty = str(getattr(item, "difficulty", "") or "").strip().lower()
    if difficulty not in {"basic", "intermediate", "advanced"}:
        difficulty = "intermediate"
    category = str(getattr(item, "category", "") or "").strip().lower()
    if category not in {"evaluation", "self_check"}:
        category = "evaluation"
    return ICUAgentBenchTask(
        task_id=getattr(item, "key", "bench_item"),
        kind=getattr(item, "kind", "descriptive_association")
        or "descriptive_association",
        title=getattr(item, "name", getattr(item, "key", "bench item")),
        objective=getattr(item, "research_question", ""),
        expected_outputs=explicit_outputs or legacy_outputs,
        semantic_guardrails=semantic_guardrails,
        evaluation_notes=_items(getattr(item, "evaluation_notes", [])),
        target_databases=_items(getattr(item, "target_databases", [])),
        gold_answer=gold,
        gold_answer_status=gold_status,
        difficulty=difficulty,
        category=category,
    )


def _five_dim_scorecard(*, run_dir: Path, item, or_value, manifest) -> Dict[str, Any]:
    """Compute the additive five-dimension Tier-1 scorecard for an arm run.

    Wrapped so a scorecard failure can never break a (possibly expensive) real
    bench run — it is reported as a diagnostic field, not raised.
    """
    try:
        from easyicu.research_agent.evaluation_scorecard import (  # type: ignore
            score_run_from_dir,
        )

        observed_warnings = [
            str(f.get("message", "")) for f in manifest.get("findings", [])
        ]
        task = _bench_item_to_task(item)
        card = score_run_from_dir(
            task,
            run_dir,
            observed_metrics=(
                {"primary_or": or_value} if or_value is not None else None
            ),
            observed_warnings=observed_warnings,
            # The bench item declares its primary predictor + outcome; pass them
            # so the gold-free overadjustment / treatment-mediator / outcome-
            # leakage checks run in the runner path too (declared, never inferred).
            # Scoring stays keyed to the declared benchmark concept.  The
            # separately declared operational column is execution-only and is
            # passed to ``pipeline.run(primary_exposure=...)`` below.
            exposure_concept=(getattr(item, "primary_predictor", "") or None),
            outcome_concept=(getattr(item, "target_outcome", "") or None),
            locked_reference_frozen=bool(
                task.gold_answer_status == "frozen" and task.gold_answer is not None
            ),
        )
        return card.model_dump()
    except Exception as exc:  # pragma: no cover - additive diagnostic only
        return {"error": f"five_dim_scorecard failed: {exc}"}


def _load_cost_summary(run_dir: Path) -> Dict[str, Any]:
    """Read the machine-readable per-run cost aggregate, if present.

    Returns the token totals + estimated USD (``cost_summary.json``, written
    when cost tracking is enabled). Empty dict when the run predates cost
    tracking or it was disabled — token counts are the durable truth; the
    USD figure is ``None`` for models absent from the price table.
    """
    path = run_dir / "cost_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _active_error_count(manifest: Dict[str, Any]) -> Optional[int]:
    readiness = manifest.get("readiness")
    if not isinstance(readiness, dict):
        return None
    keys = ("numeric_error_count", "evidence_error_count", "analysis_error_count")
    if not all(key in readiness for key in keys):
        return None
    total = 0
    for key in keys:
        try:
            total += int(readiness.get(key) or 0)
        except (TypeError, ValueError):
            return None
    return total


def _gate_ladder(run_dir: Path, readiness: Dict[str, Any]) -> Optional[str]:
    """Final gate-ladder status for the run (publication_ready > ... ).

    The selected manifest readiness is the authority supplied by the caller;
    mutable run-root summaries are never allowed to override it.
    """
    del run_dir
    if not isinstance(readiness, dict) or not readiness:
        return None
    for key in ("publication_ready", "manuscript_ready", "analysis_validated"):
        if readiness.get(key):
            return (
                key.replace("_validated", "_only")
                if key == "analysis_validated"
                else key
            )
    return "analysis_only" if readiness.get("execution_complete") else "incomplete"


def _writer_attempts(run_dir: Path, readiness: Dict[str, Any]) -> Optional[int]:
    """Writer drafting passes for the run (attempts-to-ready fragility proxy).

    Prefer the manifest gate ``writer_attempt_count``; fall back to counting
    ``"Drafting manuscript scaffold."`` events in audit_log.jsonl so runs
    whose manifests predate the gate still report a value.
    """
    if (
        isinstance(readiness, dict)
        and readiness.get("writer_attempt_count") is not None
    ):
        try:
            return int(readiness["writer_attempt_count"])
        except (TypeError, ValueError):
            pass
    audit_path = run_dir / "audit_log.jsonl"
    if not audit_path.exists():
        return None
    count = 0
    try:
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if str(event.get("event", "")).startswith("Drafting manuscript scaffold"):
                count += 1
    except Exception:
        return None
    return count


def _figure2_evaluation_attempt(*, run_dir: Path, item) -> Dict[str, Any]:
    """Seal and score one exact Canonical9 run without touching Agent calls."""

    from benchmarks.figure2_canonical9.evaluator.scoring import (
        FIGURE2_EVALUATION_ATTEMPT_SCHEMA,
        Figure2EvaluationAttempt,
        evaluate_figure2_run_from_receipt_path,
    )
    from benchmarks.figure2_canonical9.evaluator.scoring_inputs import (
        seal_figure2_run_task_authority,
    )

    task_id = str(getattr(item, "key", "") or "")
    try:
        seal_figure2_run_task_authority(
            run_dir,
            task_id=task_id,
            research_question=str(getattr(item, "research_question", "") or ""),
            exposure_concept=getattr(item, "primary_predictor", None),
            outcome_concept=getattr(item, "target_outcome", None),
            operational_exposure=_operational_exposure_for_item(item),
        )
    except Exception as exc:  # evaluator metadata must never abort a bench run
        return Figure2EvaluationAttempt(
            schema_version=FIGURE2_EVALUATION_ATTEMPT_SCHEMA,
            status="invalid",
            task_id=task_id,
            run_id=run_dir.name or None,
            invalid_reason_codes=("SCORING_INPUT_AUTHORITY_INVALID",),
            invalid_details=(f"posthoc task-authority seal failed: {exc}",),
        ).model_dump(mode="json")
    try:
        return evaluate_figure2_run_from_receipt_path(
            run_dir,
            task_id=task_id,
        ).model_dump(mode="json")
    except Exception as exc:  # scorer failures are evaluator results, not run failures
        return Figure2EvaluationAttempt(
            schema_version=FIGURE2_EVALUATION_ATTEMPT_SCHEMA,
            status="invalid",
            task_id=task_id,
            run_id=run_dir.name or None,
            invalid_reason_codes=("SCORER_ERROR",),
            invalid_details=(f"posthoc Figure 2 scorer failed: {exc}",),
        ).model_dump(mode="json")


def _score_arm(*, run_dir: Path, item, label: str) -> Dict[str, Any]:
    manifest = _load_manifest(run_dir)
    historical_error_count = sum(
        1 for f in manifest.get("findings", []) if f.get("severity") == "error"
    )
    active_error_count = _active_error_count(manifest)
    readiness = manifest.get("readiness") or {}
    or_value = _primary_or(
        run_dir,
        expected_predictor=getattr(item, "primary_predictor", ""),
        item_key=getattr(item, "key", ""),
        research_question=getattr(item, "research_question", ""),
        manifest=manifest,
    )
    result = {
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
            run_dir,
            getattr(item, "expected_step_substrings", []),
            manifest=manifest,
        ),
        "artifact_hits": _artifact_substring_hits(
            manifest, getattr(item, "expected_artifact_substrings", [])
        ),
        "n_findings": len(manifest.get("findings", [])),
        "n_warnings": sum(
            1 for f in manifest.get("findings", []) if f.get("severity") == "warning"
        ),
        "n_errors": (
            historical_error_count if active_error_count is None else active_error_count
        ),
        "n_historical_errors": historical_error_count,
        # Gate-ladder outcome + active/superseded split (assessment fix:
        # the gate story should be quantitative in the bench report, not
        # only visible inside each run's run_status.json).
        "gate_status": _gate_ladder(run_dir, readiness),
        "manuscript_ready": bool(readiness.get("manuscript_ready")),
        "publication_ready": bool(readiness.get("publication_ready")),
        "writer_attempts": _writer_attempts(run_dir, readiness),
        "superseded_error_count": (
            int(readiness.get("superseded_error_count") or 0)
            if isinstance(readiness, dict) and "superseded_error_count" in readiness
            else None
        ),
        "evidence_count": len(_active_evidence_records(manifest)),
        "evidence_kinds": _kinds_complete(manifest),
        "evidence_missing_in_manuscript": _evidence_missing_count(run_dir),
        # Additive §M1 Tier-1 five-dimension scorecard (does not yet replace the
        # legacy OR/substring scoring above — see _bench_item_to_task).
        "five_dim_scorecard": _five_dim_scorecard(
            run_dir=run_dir, item=item, or_value=or_value, manifest=manifest
        ),
        # Per-run LLM token totals + estimated USD (cost_summary.json). Feeds
        # the manuscript cost table; {} when cost tracking was off.
        "cost_summary": _load_cost_summary(run_dir),
    }
    if _is_figure2_task_id(getattr(item, "key", None)):
        result["figure2_evaluation_attempt"] = _figure2_evaluation_attempt(
            run_dir=run_dir,
            item=item,
        )
    return result


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
        if (run_dir / "manifest.json").exists() and _run_reached_execution_complete(
            run_dir
        ):
            continue
        candidates.append(run_dir.name)
    return sorted(candidates)[-1] if candidates else None


def _normalize_resume_run_id(value: Optional[str]) -> Optional[str]:
    """Validate a CLI-provided run id without accepting path traversal."""
    run_id = str(value or "").strip()
    if not run_id:
        return None
    if Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise SystemExit(
            "--resume-run-id must be a run directory name such as "
            "'run_20260701T085813_abcdef', not a path."
        )
    if not run_id.startswith("run_"):
        raise SystemExit("--resume-run-id must start with 'run_'.")
    return run_id


def _resolve_resume_run_id(
    *,
    workdir: Path,
    reuse_existing: bool,
    resume_run_id: Optional[str] = None,
) -> Optional[str]:
    """Choose the run_id to pass into ResearchAgentPipeline.run.

    ``--resume-run-id`` is an explicit user selection and therefore wins over
    ``--reuse-existing`` auto-discovery. The explicit path must already contain
    both the locked plan and partial manifest; otherwise a typo would silently
    create a fresh run directory instead of continuing the selected plan.
    """
    explicit = _normalize_resume_run_id(resume_run_id)
    if explicit:
        run_dir = workdir / explicit
        missing = [
            name
            for name in ("analysis_plan.json", "manifest_partial.json")
            if not (run_dir / name).exists()
        ]
        if missing:
            raise SystemExit(
                f"Cannot resume {explicit!r} under {workdir}: missing "
                f"{', '.join(missing)}. Choose a run that has already "
                "produced a locked plan and checkpoint."
            )
        return explicit
    return _find_resumable_run(workdir) if reuse_existing else None


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
    resume_run_id: Optional[str] = None,
    resume_from_step_id: Optional[str] = None,
    stop_after_step_id: Optional[str] = None,
    force_writer_probe: bool = False,
) -> Dict[str, Any]:
    from easyicu.research_agent import ResearchAgentPipeline  # type: ignore
    from easyicu.research_agent.cohort.schema import register_cohort_concept_ids
    from easyicu.research_agent.reporting.reporting_checklist import (
        checklist_names_for_kind,
    )

    # The provided cohort is already materialised; let the planner reference any
    # of its columns in a CTAS predicate without tripping the static dictionary
    # check ("unknown concept_id: <derived column>").
    register_cohort_concept_ids(
        list(getattr(item, "cohort_columns", None) or getattr(cohort, "columns", []))
    )

    workdir.mkdir(parents=True, exist_ok=True)
    # Force the kind-matched reporting checklist(s) so the EMITTED file matches
    # what the scorecard READS by task kind (single source of truth:
    # ``checklist_names_for_kind``). Without this the pipeline falls back to
    # free-text analysis-family inference, which emitted STROBE for the
    # mortality_prediction task while the scorecard expected TRIPOD+AI — so
    # reporting_completeness was silently NA on a run that did reach the write
    # phase (detector/emitter contract mismatch, G-2). ``setdefault`` lets an
    # explicit pipeline_options override win.
    opts = dict(pipeline_options or {})
    opts.setdefault(
        "reporting_checklist_names",
        list(checklist_names_for_kind(getattr(item, "kind", None))),
    )
    # The authoritative task kind lets the internal phenotype checklist decide
    # trajectory-item applicability by kind (cross-sectional clustering vs
    # longitudinal) instead of fragile manuscript wording (M3 false-open).
    opts.setdefault("task_kind", getattr(item, "kind", None))
    pipeline = ResearchAgentPipeline(
        workdir=workdir,
        llm=llm,
        disable_icu_context=disable_icu_context,
        **opts,
    )
    resolved_resume_run_id = _resolve_resume_run_id(
        workdir=workdir,
        reuse_existing=reuse_existing,
        resume_run_id=resume_run_id,
    )
    if resolved_resume_run_id:
        mode = "selected" if resume_run_id else "interrupted"
        print(
            f"[research_agent] resuming {mode} run {resolved_resume_run_id} "
            f"(step-level checkpoint) for {item.key}/{label}"
        )
    started = time.monotonic()
    database = str(getattr(item, "database", "") or "bench").strip() or "bench"
    operational_exposure = _operational_exposure_for_item(item)
    exposure_display_name = getattr(item, "primary_predictor", None) or None
    normalized_question = re.sub(
        r"[^a-z0-9]+", "_", str(item.research_question or "").lower()
    ).strip("_")
    normalized_display_name = re.sub(
        r"[^a-z0-9]+", "_", str(exposure_display_name or "").lower()
    ).strip("_")
    display_name_is_question_exposed = bool(
        normalized_display_name
        and re.search(
            rf"(?:^|_){re.escape(normalized_display_name)}(?:_|$)",
            normalized_question,
        )
    )
    concept_descriptions = (
        {str(operational_exposure): str(exposure_display_name)}
        if operational_exposure
        and exposure_display_name
        and display_name_is_question_exposed
        else None
    )
    result = pipeline.run(
        question=item.research_question,
        cohort=cohort,
        cohort_authority_path=getattr(item, "cohort_authority_path", None),
        cohort_authority_ref=getattr(item, "cohort_authority_ref", None),
        trajectory_path=getattr(item, "trajectory_path", None),
        trajectory_authority_path=getattr(item, "trajectory_authority_path", None),
        trajectory_authority_ref=getattr(item, "trajectory_authority_ref", None),
        cohort_name=f"bench_{item.key}",
        database=database,
        target_outcome=item.target_outcome,
        primary_exposure=operational_exposure,
        concept_descriptions=concept_descriptions,
        inclusion_criteria=item.inclusion_criteria,
        resume_run_id=resolved_resume_run_id,
        resume_from_step_id=resume_from_step_id,
        stop_after_step_id=stop_after_step_id,
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
    resume_run_id: Optional[str] = None,
    resume_from_step_id: Optional[str] = None,
    stop_after_step_id: Optional[str] = None,
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
            resume_run_id=resume_run_id,
            resume_from_step_id=resume_from_step_id,
            stop_after_step_id=stop_after_step_id,
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
            resume_run_id=resume_run_id,
            resume_from_step_id=resume_from_step_id,
            stop_after_step_id=stop_after_step_id,
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


def _run_reached_execution_complete(run_dir: Path) -> bool:
    """Return True only when current checkpoint and EvidenceStore status agree."""

    from easyicu.research_agent.authority.run_lock import (
        RunExecutionLockError,
        acquire_run_execution_lock,
    )
    from easyicu.research_agent.authority.runtime_artifacts import (
        RunArtifactAuthorityError,
        current_evidence_records,
        load_run_artifact_authority,
        verified_run_evidence_path,
    )
    from easyicu.research_agent.authority.evidence_snapshot import (
        EvidenceAuthorityIntegrityError,
        load_current_evidence_snapshot,
    )

    try:
        with acquire_run_execution_lock(workdir=run_dir.parent, run_id=run_dir.name):
            selected = load_run_artifact_authority(run_dir)
            if selected is None:
                return False

            def read_regular_object(
                path: Path, *, expected_sha256: str | None = None
            ) -> Dict[str, object]:
                descriptor: int | None = None
                try:
                    flags = (
                        os.O_RDONLY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NONBLOCK", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    descriptor = os.open(path, flags)
                    metadata = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_size > 1024 * 1024
                    ):
                        raise OSError("run authority document is not a small file")
                    chunks: list[bytes] = []
                    total = 0
                    while True:
                        chunk = os.read(descriptor, 64 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > 1024 * 1024:
                            raise OSError("run authority document exceeds 1 MiB")
                        chunks.append(chunk)
                    payload = b"".join(chunks)
                    if expected_sha256 is not None and (
                        hashlib.sha256(payload).hexdigest() != expected_sha256
                    ):
                        raise OSError("run authority document digest changed")
                    return _decode_jsonl_object(payload.decode("utf-8"))
                finally:
                    if descriptor is not None:
                        os.close(descriptor)

            final_manifest = read_regular_object(run_dir / "manifest.json")
            if selected != final_manifest:
                return False
            readiness = selected.get("readiness")
            raw_steps = selected.get("per_step_records")
            raw_records = selected.get("evidence")
            if (
                not isinstance(readiness, dict)
                or readiness.get("execution_complete") is not True
                or not isinstance(raw_steps, list)
                or any(not isinstance(item, dict) for item in raw_steps)
                or not isinstance(raw_records, list)
                or any(not isinstance(item, dict) for item in raw_records)
            ):
                return False
            snapshot = load_current_evidence_snapshot(run_dir)
            if snapshot.source != "authority":
                return False
            coordinate_fields = (
                "evidence_id",
                "relative_path",
                "sha256",
                "kind",
                "producer",
                "generation_mode",
                "produced_by_step",
            )
            from collections import Counter

            checkpoint_coordinates = Counter(
                tuple(record.get(field) for field in coordinate_fields)
                for record in raw_records
            )
            evidence_coordinates = Counter(
                tuple(record.get(field) for field in coordinate_fields)
                for record in snapshot.records
            )
            if checkpoint_coordinates != evidence_coordinates:
                return False
            current_records = [
                dict(record)
                for record in current_evidence_records(snapshot.records, raw_steps)
                if isinstance(record, Mapping)
            ]
            status_records = [
                record
                for record in current_records
                if record.get("evidence_id") == "run_status"
            ]
            if len(status_records) != 1:
                return False
            status_record = status_records[0]
            status_path = verified_run_evidence_path(run_dir, status_record)
            if status_path is None:
                return False
            status_payload = read_regular_object(
                status_path,
                expected_sha256=str(status_record.get("sha256") or ""),
            )
            gates = status_payload.get("gates")
            return (
                isinstance(gates, dict)
                and gates.get("execution_complete") is True
                and gates == readiness
            )
    except (
        OSError,
        UnicodeDecodeError,
        _JSONLObjectDecodeError,
        EvidenceAuthorityIntegrityError,
        RunArtifactAuthorityError,
        RunExecutionLockError,
        ValueError,
    ):
        return False


def _figure2_run_is_reusable(run_dir: Path, item: object) -> bool:
    """Require the exact paper sidecar authority before suppressing a new run."""

    if not _run_reached_execution_complete(run_dir):
        return False
    from benchmarks.figure2_canonical9.evaluator.scoring_inputs import (
        load_figure2_scoring_inputs,
        seal_figure2_run_task_authority,
    )

    try:
        seal_figure2_run_task_authority(
            run_dir,
            task_id=str(getattr(item, "key", "") or ""),
            research_question=str(getattr(item, "research_question", "") or ""),
            exposure_concept=getattr(item, "primary_predictor", None),
            outcome_concept=getattr(item, "target_outcome", None),
            operational_exposure=_operational_exposure_for_item(item),
        )
        load_figure2_scoring_inputs(
            run_dir,
            expected_task_id=str(getattr(item, "key", "") or ""),
        )
    except Exception:
        return False
    return True


def _reuse_arm_if_complete(
    *, arm_dir: Path, item, label: str
) -> Optional[Dict[str, Any]]:
    if not arm_dir.exists():
        return None
    runs = sorted(
        (
            p
            for p in arm_dir.glob("run_*")
            if (p / "manifest.json").exists()
            and (
                _figure2_run_is_reusable(p, item)
                if _is_figure2_task_id(getattr(item, "key", None))
                else _run_reached_execution_complete(p)
            )
        ),
        key=lambda p: p.name,
        reverse=True,
    )
    if not runs:
        return None
    result = _score_arm(run_dir=runs[0], item=item, label=label)
    attempt = result.get("figure2_evaluation_attempt")
    if (
        _is_figure2_task_id(getattr(item, "key", None))
        and isinstance(attempt, dict)
        and attempt.get("status") == "invalid"
        and "SCORING_INPUT_AUTHORITY_INVALID"
        in set(attempt.get("invalid_reason_codes") or ())
    ):
        return None
    return result


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
    resume_run_id: Optional[str] = None,
    resume_from_step_id: Optional[str] = None,
    stop_after_step_id: Optional[str] = None,
    force_writer_probe: bool = False,
) -> Dict[str, Any]:
    if verbose:
        print(f"\n=== {item.key} — {item.name} ===")
    cohort = item.cohort_factory(seed)
    item_root = out_root / item.key
    selected = set(_normalize_arms(arms))

    naive = _skipped_arm("naive")
    aware = _skipped_arm("aware")
    if reuse_existing and not resume_run_id:
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
            resume_run_id=resume_run_id,
            resume_from_step_id=resume_from_step_id,
            stop_after_step_id=stop_after_step_id,
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
            resume_run_id=resume_run_id,
            resume_from_step_id=resume_from_step_id,
            stop_after_step_id=stop_after_step_id,
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
        n_dir_correct = sum(1 for s in arm_scores if s[arm]["direction_match"] is True)
        n_dir_wrong = sum(1 for s in arm_scores if s[arm]["direction_match"] is False)
        n_dir_missing = sum(1 for s in arm_scores if s[arm]["direction_match"] is None)
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
            "manuscript_ready": sum(
                1 for s in arm_scores if s[arm].get("manuscript_ready")
            ),
            "publication_ready": sum(
                1 for s in arm_scores if s[arm].get("publication_ready")
            ),
            "superseded_errors_total": sum(
                s[arm].get("superseded_error_count") or 0 for s in arm_scores
            ),
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
                f"| Item | Family | Difficulty | Evidence basis | Direction ({arm}) | OR ({arm}) | Gate status ({arm}) | Writer passes ({arm}) | Active errs ({arm}) | Superseded errs ({arm}) | Predefined rule hits ({arm}) | Workflow hits ({arm}) | Artifact hits ({arm}) | `[evidence missing]` ({arm}) |",
                "|---|---|---|---|:-:|---:|---|---:|---:|---:|:-:|:-:|:-:|---:|",
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
                f"| `{arm_score.get('gate_status') or '—'}` "
                f"| {_fmt_missing(arm_score.get('writer_attempts'))} "
                f"| {_fmt_missing(arm_score.get('n_errors'))} "
                f"| {_fmt_missing(arm_score.get('superseded_error_count'))} "
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
    lines.append(
        "| Metric | " + " | ".join(_ARM_LABELS[arm] for arm in ran_arms) + " |"
    )
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
        (
            "Total `[evidence missing]` lines (lower is better)",
            "evidence_missing_in_manuscripts",
        ),
        ("Items reaching manuscript_ready", "manuscript_ready"),
        ("Items reaching publication_ready", "publication_ready"),
        (
            "Total superseded (historically blocked, later resolved) errors",
            "superseded_errors_total",
        ),
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


def _git_short_sha() -> str:
    """Best-effort 7-char git SHA of the EasyICU checkout (``unknown`` if N/A)."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = out.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def _render_run_registry(payload: Dict[str, Any]) -> str:
    """Human-readable provenance index for a batch (one row per item/arm).

    Records the model + git SHA the run was produced under and, per item,
    the status / primary OR / token+USD cost / run directory — so a frozen
    (定稿) batch is traceable at a glance without opening each run folder.
    """
    lines = [
        "# Run registry",
        "",
        f"- generated: `{payload.get('generated_at')}`",
        f"- provider/model: `{payload.get('provider')}` / `{payload.get('model')}`",
        f"- backend: `{payload.get('backend_base_url', 'unknown')}`",
        f"- git: `{payload.get('git_sha', 'unknown')}`",
        f"- seed: `{payload.get('seed')}`  ·  arms: `{payload.get('arms')}`",
        "",
        "| item | arm | status | primary_OR | total_tokens | est_USD | run_dir |",
        "|------|-----|--------|-----------|--------------|---------|---------|",
    ]
    for item_payload in payload.get("scores", []):
        item_key = item_payload.get("item_key", "")
        for arm in ("aware", "naive"):
            arm_score = item_payload.get(arm)
            if not isinstance(arm_score, dict):
                continue
            cost = arm_score.get("cost_summary") or {}
            tokens = cost.get("total_tokens", "")
            usd = cost.get("total_cost_usd", None)
            usd_str = f"{usd:.4f}" if isinstance(usd, (int, float)) else ""
            sc = arm_score.get("five_dim_scorecard") or {}
            status = sc.get("tristate") or sc.get("overall_status") or ""
            run_dir = arm_score.get("workdir", "")
            lines.append(
                f"| {item_key} | {arm} | {status} "
                f"| {arm_score.get('primary_or', '')} | {tokens} | {usd_str} "
                f"| `{run_dir}` |"
            )
    lines.append("")
    return "\n".join(lines)


def _make_llm(*, provider: str, model: str, request_timeout: float):
    _bootstrap_imports()
    from easyicu.research_agent import MockLLMClient, OpenAIClient  # type: ignore
    from easyicu.research_agent.providers import (  # type: ignore
        ProviderConfigurationError,
        build_provider_client,
    )

    if provider == "mock":
        return MockLLMClient()
    try:
        return build_provider_client(
            provider=provider,
            model=model,
            request_timeout=request_timeout,
            title="EasyICU research-agent benchmark",
            client_cls=OpenAIClient,
        )
    except ProviderConfigurationError as exc:
        raise SystemExit(str(exc)) from exc


def _resolve_backend_base_url(provider: str) -> str:
    """Resolve the serving backend URL a run will actually hit.

    Recorded in the batch provenance so a frozen (定稿) run is unambiguous
    about the *call path*, not just the model string: ``--provider openai
    --model gpt-5.5`` can route to the local Codex Tools proxy
    (``http://127.0.0.1:8787/v1``) or to ``api.openai.com`` depending on
    ``OPENAI_BASE_URL``, and those are different serving paths with
    different latency / concurrency / rate-limit behaviour. No credential
    is included — a base URL carries no secret.
    """
    if provider == "mock":
        return "mock://deterministic"
    _bootstrap_imports()
    from easyicu.research_agent.providers import resolve_provider_base_url

    return resolve_provider_base_url(provider)


def _benchmark_pipeline_options(
    *,
    max_total_steps: Optional[int],
    disable_replanning: bool,
    max_code_repair_attempts: Optional[int],
    max_step_llm_repair_attempts: Optional[int] = None,
    timeout_seconds: float = 300.0,
    standard_executor_timeout_seconds: float = 3_600.0,
    enable_repro_envelope: bool = True,
    enable_cost_tracking: bool = True,
    llm_seed: Optional[int] = None,
    writer_digest_widened: bool = False,
    strict_evidence: bool = False,
    enable_cross_run_memory: bool = False,
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
    # These are independent execution budgets.  The ordinary timeout bounds
    # model-generated scripts; registered standards use their own longer
    # planner-owned workload budget.
    options["timeout_seconds"] = float(timeout_seconds)
    options["standard_executor_timeout_seconds"] = float(
        standard_executor_timeout_seconds
    )
    if max_total_steps is not None:
        options["max_total_steps"] = int(max_total_steps)
    if disable_replanning:
        options["enable_replanning"] = False
    if max_code_repair_attempts is not None:
        options["max_code_repair_attempts"] = int(max_code_repair_attempts)
    if max_step_llm_repair_attempts is not None:
        options["max_step_llm_repair_attempts"] = int(max_step_llm_repair_attempts)
    if strict_evidence:
        options["evidence_enforcement_mode"] = "strict"
    if enable_repro_envelope:
        # Default ON for bench runs so the per-call envelope
        # (temperature / requested_top_p / seed / model / prompt+response
        # SHA256) lands as reproducibility_envelope.json next to each
        # arm's run_status.json.
        options["enable_reproducibility_envelope"] = True
    if enable_cost_tracking:
        # Default ON for bench runs so each arm writes cost_records.json /
        # cost_summary.{md,json} and ``manifest.cost_records`` — the token
        # totals + estimated USD that become Fig.3 / cost-table source data.
        options["enable_cost_tracking"] = True
    if writer_digest_widened:
        options["writer_digest_widened"] = True
    if llm_seed is not None:
        options["llm_seed"] = int(llm_seed)
    # Cross-run RunMemory (StrategyCard) injection is OFF by default for
    # benchmark/canonical runs. Every resume reuses the same workdir, so a prior
    # run's distilled StrategyCards would be re-injected into the planner on the
    # next run — unvalidated procedural memory that undermines reproducibility of
    # a fresh/resumed canonical run. Disabling it also stops the run from writing
    # new cards. Within-run authority (StepAuthorityCapsule / checkpoints /
    # evidence store) does not use RunMemory and is unaffected. ExperienceBank is
    # already opt-in (engine default False; the harness never enables it). No
    # submission profile sets ``enable_memory``, so this is deterministic.
    # ``--enable-cross-run-memory`` opts back in for non-canonical runs ONLY.
    # A submission profile already pins both flags off as submission-defining
    # options (see SubmissionProfile.as_pipeline_options), and the profile must
    # win: fail closed rather than let a flag silently re-open cross-run memory
    # on a paper-facing run.
    if submission_profile is not None and enable_cross_run_memory:
        raise SystemExit(
            "--enable-cross-run-memory is incompatible with a submission "
            "profile: a paper-facing canonical run must not inject cross-run "
            "StrategyCards into the planner. Drop --submission-profile for an "
            "exploratory run instead."
        )
    # ``setdefault`` so a profile's pinned values (applied above) always win.
    options.setdefault("enable_memory", bool(enable_cross_run_memory))
    options.setdefault("enable_experience_bank", False)
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


def _enforce_development_resume_repair_budget(
    value: Optional[int],
    *,
    resume_run_id: Optional[str],
    resume_from_step_id: Optional[str],
    profile: Optional["SubmissionProfile"],
) -> Optional[int]:
    """Gate an explicit cross-resume logical-repair budget override.

    The pipeline already persists logical repair attempts monotonically.  This
    development-only switch may raise the configured ceiling for one selected
    failed step without deleting receipts or replaying completed work.  Paper-
    facing profiles keep their frozen budget and cannot use the override.
    """

    if value is None:
        return None
    normalized = int(value)
    if normalized != 3:
        raise SystemExit(
            "--max-step-llm-repair-attempts must be exactly 3 for an explicit "
            "one-call development resume; repeated budget ratcheting is not "
            "allowed."
        )
    if not resume_run_id or not resume_from_step_id:
        raise SystemExit(
            "--max-step-llm-repair-attempts is development-only and requires "
            "both --resume-run-id and --resume-from-step-id."
        )
    if profile is not None:
        raise SystemExit(
            "--max-step-llm-repair-attempts cannot override a submission "
            "profile's frozen repair budget."
        )
    return normalized


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
    profile, ``auto`` probes Docker first and may use macOS sandbox-exec. The
    ``--allow-host-runner`` escape hatch exists for offline development
    and is never valid for an archival/canonical batch.
    """
    requested = (runner or "auto").lower()
    if profile is None:
        return requested
    required = (profile.requires_runner or "docker").lower()
    resolved = required if requested == "auto" else requested
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
    resume_run_id: Optional[str] = None,
    resume_from_step_id: Optional[str] = None,
    stop_after_step_id: Optional[str] = None,
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
                resume_run_id=resume_run_id,
                resume_from_step_id=resume_from_step_id,
                stop_after_step_id=stop_after_step_id,
                force_writer_probe=force_writer_probe,
                verbose=verbose,
            )
        )

    totals = _aggregate(scores)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_short_sha(),
        "seed": seed,
        "bench_kind": bench_kind,
        "provider": provider,
        "model": model,
        "backend_base_url": _resolve_backend_base_url(provider),
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
    (out_root / "RUN_REGISTRY.md").write_text(
        _render_run_registry(payload), encoding="utf-8"
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
                (
                    f"Evidence missing ({suffix})",
                    arm,
                    "evidence_missing_in_manuscripts",
                ),
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

    default_submission_profile_ref = _default_submission_profile_ref()
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
        default=os.environ.get(
            "EASYICU_HOSTED_DEFAULT_MODEL", "openai/gpt-oss-120b:free"
        ),
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
        "--timeout",
        type=float,
        default=300.0,
        help=(
            "Per-attempt timeout in seconds for ordinary model-generated "
            "analysis code. This does not change the LLM request timeout or "
            "the registered standard-executor timeout."
        ),
    )
    parser.add_argument(
        "--standard-executor-timeout",
        type=float,
        default=3_600.0,
        help=(
            "Independent timeout in seconds for a registered deterministic "
            "standard executor running the Planner-owned workload."
        ),
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
        help=(
            "Reuse completed item/arm runs already present under --out-root "
            "and auto-resume the latest interrupted run in an item/arm."
        ),
    )
    parser.add_argument(
        "--resume-run-id",
        default=None,
        help=(
            "Explicit run_id to continue under the selected item/arm directory. "
            "Requires exactly one item and one arm; completed steps are skipped "
            "and the interrupted step is rerun from that step."
        ),
    )
    parser.add_argument(
        "--resume-from-step-id",
        default=None,
        help=(
            "With --resume-run-id, ignore completed checkpoint records from this "
            "plan step onward so the selected step is rerun."
        ),
    )
    parser.add_argument(
        "--stop-after-step-id",
        default=None,
        help=(
            "Stop execution after the named plan step. Useful for reviewing one "
            "resumed step at a time. Use '@first' to stop after the Agent's "
            "first planned step without depending on its generated step id."
        ),
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
        "--enable-cross-run-memory",
        action="store_true",
        help=(
            "Opt back into cross-run RunMemory (StrategyCard) injection. OFF by "
            "default for benchmark/canonical runs: distilling and re-injecting "
            "StrategyCards from a prior run of the same workdir (every resume "
            "reuses the workdir) feeds unvalidated procedural cards into the "
            "planner and hurts reproducibility. Within-run authority "
            "(StepAuthorityCapsule / checkpoints / evidence) is unaffected. Use "
            "only for non-canonical exploratory runs."
        ),
    )
    parser.add_argument(
        "--max-code-repair-attempts",
        type=int,
        default=None,
        help="Override the per-step generated-code repair attempt budget.",
    )
    parser.add_argument(
        "--max-step-llm-repair-attempts",
        type=int,
        default=None,
        help=(
            "Development resume only: set the durable total LLM-repair "
            "ceiling for the selected failed step. Requires --resume-run-id "
            "and --resume-from-step-id and is forbidden under a submission "
            "profile. Prior attempts remain counted."
        ),
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
        "--no-cost-tracking",
        action="store_true",
        help=(
            "Disable per-run LLM cost tracking (T3.2). Cost tracking is ON "
            "by default for bench runs so each arm writes cost_records.json / "
            "cost_summary.{md,json} and manifest.cost_records — the token "
            "totals + estimated USD that feed the Fig.3 / cost-table source "
            "data. Token counts are recorded exactly even when a model's "
            "price is unknown."
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
        default=default_submission_profile_ref,
        help=(
            "Versioned submission profile ref used with --submission-profile "
            f"(default: {default_submission_profile_ref})."
        ),
    )
    parser.add_argument(
        "--runner",
        choices=["auto", "subprocess", "docker"],
        default=None,
        help=(
            "Code-execution backend for agent-generated scripts. "
            "'auto' (default) probes Docker first and otherwise permits only "
            "macOS sandbox-exec. 'subprocess' is an explicit development "
            "choice; 'docker' uses the network-isolated container runner. A submission profile "
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
        "--require-figure2-paper-acceptance",
        action="store_true",
        help=(
            "After every EHRFlowBench row has run and results are written, "
            "require deterministic replay verification of one valid aware-arm "
            "attempt for each exact Canonical9 task. Invalid attempts remain "
            "nonfatal per item, but the completed batch exits with status 3."
        ),
    )
    parser.add_argument(
        "--force-writer-probe",
        action="store_true",
        help=(
            "Diagnostic engineering use only: force writer output even when "
            "the execution gate fails. Do NOT use for archival benchmarks."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run the EHRFlowBench batch N times into repeat_NN/ subdirs and "
            "write stability_report.md aggregating the primary estimate, "
            "gate outcome, writer passes and adjustment covariates across "
            "repeats. Measures run-to-run design variation (reviewer "
            "nondeterminism question). N=1 (default) is a normal single run."
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
    explicit_resume_run_id = _normalize_resume_run_id(
        getattr(args, "resume_run_id", None)
    )
    if explicit_resume_run_id and len(_normalize_arms(args.arms)) != 1:
        raise SystemExit(
            "--resume-run-id requires exactly one arm. Pass for example "
            "'--arms aware' so the selected run_id maps to one item/arm folder."
        )
    resume_from_step_id = (
        str(getattr(args, "resume_from_step_id", "") or "").strip() or None
    )
    stop_after_step_id = (
        str(getattr(args, "stop_after_step_id", "") or "").strip() or None
    )
    if resume_from_step_id and not explicit_resume_run_id:
        raise SystemExit("--resume-from-step-id requires --resume-run-id.")
    max_step_llm_repair_attempts = _enforce_development_resume_repair_budget(
        getattr(args, "max_step_llm_repair_attempts", None),
        resume_run_id=explicit_resume_run_id,
        resume_from_step_id=resume_from_step_id,
        profile=submission_profile,
    )
    if explicit_resume_run_id and args.models and len(args.models) != 1:
        raise SystemExit("--resume-run-id cannot be combined with multiple models.")
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
        max_step_llm_repair_attempts=max_step_llm_repair_attempts,
        timeout_seconds=float(args.timeout),
        standard_executor_timeout_seconds=float(args.standard_executor_timeout),
        enable_repro_envelope=not bool(getattr(args, "no_repro_envelope", False)),
        enable_cost_tracking=not bool(getattr(args, "no_cost_tracking", False)),
        llm_seed=getattr(args, "llm_seed", None),
        writer_digest_widened=bool(args.writer_digest_widened),
        strict_evidence=bool(args.strict_evidence),
        enable_cross_run_memory=bool(getattr(args, "enable_cross_run_memory", False)),
        submission_profile=submission_profile,
        runner_kind=runner_kind,
    )

    if args.ehrflowbench_jsonl:
        if bool(args.require_figure2_paper_acceptance) and _normalize_arms(
            args.arms
        ) != ["aware"]:
            raise SystemExit(
                "--require-figure2-paper-acceptance requires exactly '--arms aware'."
            )
        ehrflow_model = args.model if args.provider != "mock" else "mock"
        if args.models:
            ehrflow_model = args.models[0]
        n_repeat = max(1, int(args.repeat))
        base_out_root = Path(args.out_root).resolve()
        if n_repeat > 1 and explicit_resume_run_id:
            raise SystemExit(
                "--repeat cannot be combined with --resume-run-id: repeats "
                "start fresh runs, resume continues one existing run."
            )

        def _run_ehrflow_into(target_out_root: Path) -> int:
            return _run_ehrflowbench_jsonl(
                jsonl_path=Path(args.ehrflowbench_jsonl).resolve(),
                out_root=target_out_root,
                seed=args.seed,
                arms=args.arms,
                pipeline_options=pipeline_options,
                provider=args.provider,
                model=ehrflow_model,
                request_timeout=float(args.request_timeout),
                reuse_existing=bool(args.reuse_existing),
                resume_run_id=explicit_resume_run_id,
                resume_from_step_id=resume_from_step_id,
                stop_after_step_id=stop_after_step_id,
                force_writer_probe=bool(args.force_writer_probe),
                allow_mock_aware=bool(args.allow_mock_aware),
                require_figure2_paper_acceptance=bool(
                    args.require_figure2_paper_acceptance
                ),
            )

        if n_repeat == 1:
            return _run_ehrflow_into(base_out_root)

        rc = 0
        repeat_roots: List[Path] = []
        for i in range(1, n_repeat + 1):
            repeat_root = base_out_root / f"repeat_{i:02d}"
            print(f"\n########## STABILITY REPEAT {i}/{n_repeat} ##########")
            rc_i = _run_ehrflow_into(repeat_root)
            repeat_roots.append(repeat_root)
            rc = rc or rc_i
        _write_stability_report(base_out_root, repeat_roots, arms=args.arms)
        return rc

    if bool(args.require_figure2_paper_acceptance):
        raise SystemExit(
            "--require-figure2-paper-acceptance requires --ehrflowbench-jsonl."
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
    if explicit_resume_run_id and len(items) != 1:
        raise SystemExit(
            "--resume-run-id requires exactly one benchmark item. Pass "
            "'--items <key>' for the plan you want to continue."
        )

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
            resume_run_id=explicit_resume_run_id,
            resume_from_step_id=resume_from_step_id,
            stop_after_step_id=stop_after_step_id,
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
    for run_dir in item_root.glob("*/run_*"):
        if _run_reached_execution_complete(run_dir):
            return True
    return False


def _stability_adjustment_covariates(workdir: Optional[str]) -> Optional[List[str]]:
    """Best-effort adjustment covariate set from a run's model tables.

    The reviewer-facing "design variation" story is partly which covariates
    the agent chose (e.g. E1 adjusting for lactate vs not). Read the first
    model/coefficient table that carries a term column and return its
    non-intercept terms. Returns None when nothing parseable is found —
    the estimate-level columns carry the report either way.
    """
    if not workdir:
        return None
    run_dir = Path(workdir)
    if not run_dir.exists():
        return None
    import pandas as pd

    patterns = [
        "evidence/*model_coefficients*.csv",
        "steps/*/outputs/model_coefficients.csv",
        "evidence/*model_specification*.csv",
        "steps/*/outputs/model_specification.csv",
        "evidence/*adjusted_association*.csv",
    ]
    term_cols = {"term", "variable", "covariate", "feature", "parameter", "predictor"}
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern)):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            lower = {str(c).lower(): c for c in df.columns}
            hit = next((lower[c] for c in term_cols if c in lower), None)
            if hit is None:
                continue
            terms = [str(t).strip() for t in df[hit].tolist()]
            terms = [
                t
                for t in terms
                if t.lower() not in {"intercept", "const", "constant", ""}
            ]
            if terms:
                return sorted(dict.fromkeys(terms))
    return None


def _write_stability_report(
    base_out_root: Path,
    repeat_roots: Sequence[Path],
    *,
    arms: Sequence[str],
) -> None:
    """Aggregate per-repeat estimates into a run-to-run stability report.

    Answers the reviewer nondeterminism question quantitatively: for each
    item, the primary estimate, gate outcome, writer passes and adjustment
    covariates across N repeats, plus the OR spread (min/median/max).
    """
    import statistics

    ran_arms = _normalize_arms(arms)
    # item_key -> list of {repeat, arm, or, gate, attempts, covariates}
    per_item: Dict[str, List[Dict[str, Any]]] = {}
    for idx, root in enumerate(repeat_roots, start=1):
        results_path = root / "ehrflowbench_results.json"
        if not results_path.exists():
            continue
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for score in payload.get("scores", []):
            key = score.get("item_key") or score.get("key")
            if not key:
                continue
            for arm in ran_arms:
                arm_score = score.get(arm)
                if not isinstance(arm_score, dict):
                    continue
                per_item.setdefault(str(key), []).append(
                    {
                        "repeat": idx,
                        "arm": arm,
                        "or": arm_score.get("primary_or"),
                        "gate": arm_score.get("gate_status"),
                        "attempts": arm_score.get("writer_attempts"),
                        "covariates": _stability_adjustment_covariates(
                            arm_score.get("workdir")
                        ),
                    }
                )

    report: Dict[str, Any] = {
        "n_repeats": len(repeat_roots),
        "arms": ran_arms,
        "items": {},
    }
    md = [
        "# Bench stability report",
        "",
        f"Repeats: {len(repeat_roots)} · arms: {', '.join(ran_arms)}",
        "",
        "Run-to-run variation of the primary estimate, gate outcome and "
        "adjustment covariates. A wide OR spread or shifting covariate set "
        "means the result depends on design choices the agent made "
        "autonomously, not only on the data.",
        "",
    ]
    for key, rows in sorted(per_item.items()):
        ors = [float(r["or"]) for r in rows if isinstance(r["or"], (int, float))]
        or_min = min(ors) if ors else None
        or_max = max(ors) if ors else None
        or_med = statistics.median(ors) if ors else None
        gates = [r["gate"] for r in rows if r["gate"]]
        gate_dist: Dict[str, int] = {}
        for g in gates:
            gate_dist[g] = gate_dist.get(g, 0) + 1
        # covariate-set stability: how many DISTINCT adjustment sets appeared
        cov_sets = [tuple(r["covariates"]) for r in rows if r["covariates"]]
        distinct_cov_sets = len(set(cov_sets))
        report["items"][key] = {
            "n_runs": len(rows),
            "or_values": ors,
            "or_min": or_min,
            "or_median": or_med,
            "or_max": or_max,
            "or_spread": (or_max - or_min) if (ors and or_max is not None) else None,
            "gate_distribution": gate_dist,
            "distinct_covariate_sets": distinct_cov_sets,
            "runs": rows,
        }
        or_range = f"{or_min:.3f}–{or_max:.3f} (median {or_med:.3f})" if ors else "—"
        gate_summary = (
            ", ".join(f"{g}×{n}" for g, n in sorted(gate_dist.items())) or "—"
        )
        md.extend(
            [
                f"## `{key}`",
                "",
                f"- Primary OR across {len(rows)} run(s): **{or_range}**",
                f"- Gate outcomes: {gate_summary}",
                f"- Distinct adjustment-covariate sets: {distinct_cov_sets}",
                "",
                "| Repeat | Arm | OR | Gate | Writer passes | # covariates |",
                "|---:|---|---:|---|---:|---:|",
            ]
        )
        for r in rows:
            or_txt = (
                f"{float(r['or']):.3f}" if isinstance(r["or"], (int, float)) else "—"
            )
            n_cov = len(r["covariates"]) if r["covariates"] else "—"
            md.append(
                f"| {r['repeat']} | {r['arm']} | {or_txt} | "
                f"`{r['gate'] or '—'}` | {r['attempts'] if r['attempts'] is not None else '—'} "
                f"| {n_cov} |"
            )
        md.append("")

    base_out_root.mkdir(parents=True, exist_ok=True)
    (base_out_root / "stability_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (base_out_root / "stability_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  -> {base_out_root / 'stability_report.md'}")
    print(f"  -> {base_out_root / 'stability_report.json'}")


_EXTERNAL_DIFFICULTY_ALIASES = {
    "easy": "basic",
    "medium": "intermediate",
    "hard": "advanced",
}


def _external_string_list(
    row: Mapping[str, Any],
    field: str,
    diagnostics: List[Dict[str, Any]],
) -> List[str]:
    """Read one declared list field without mining values from prose."""

    value = row.get(field)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        diagnostics.append(
            {
                "field": field,
                "status": "coerced_scalar_to_list",
                "source_type": "str",
            }
        )
        return [stripped]
    if isinstance(value, (list, tuple)):
        return [str(entry).strip() for entry in value if str(entry).strip()]
    diagnostics.append(
        {
            "field": field,
            "status": "invalid_type_defaulted",
            "source_type": type(value).__name__,
            "default": [],
        }
    )
    return []


def _external_item_from_row(
    *,
    row: Mapping[str, Any],
    key: str,
    question: str,
    target: str,
    cohort_columns: Sequence[Any],
    cohort_size: int,
    cohort_authority_path: Optional[Path] = None,
    cohort_authority_ref: Optional[Mapping[str, object]] = None,
    trajectory_path: Optional[Path] = None,
    trajectory_authority_path: Optional[Path] = None,
    trajectory_authority_ref: Optional[Mapping[str, object]] = None,
) -> SimpleNamespace:
    """Build one external item from structured protocol fields.

    The adapter never derives database, exposure, or rubric decisions from the
    natural-language question.  Older JSONL exports remain runnable via explicit
    defaults recorded in ``protocol_adapter`` so a development run cannot
    silently masquerade as a fully declared protocol.
    """

    diagnostics: List[Dict[str, Any]] = []

    database_source = "database" if str(row.get("database") or "").strip() else None
    database = str(row.get("database") or "").strip()
    if not database:
        database = "bench"
        diagnostics.append(
            {
                "field": "database",
                "status": "missing_defaulted",
                "default": database,
            }
        )

    scoring_source = next(
        (
            field
            for field in ("primary_predictor", "scoring_predictor")
            if str(row.get(field) or "").strip()
        ),
        None,
    )
    scoring_predictor = (
        str(row.get(scoring_source) or "").strip() if scoring_source else ""
    )

    operational_source = next(
        (
            field
            for field in (
                "operational_exposure",
                "operational_exposure_column",
                "primary_exposure",
                "exposure_column",
            )
            if str(row.get(field) or "").strip()
        ),
        None,
    )
    operational_exposure = (
        str(row.get(operational_source) or "").strip() if operational_source else ""
    )
    if not scoring_predictor and operational_exposure:
        scoring_predictor = operational_exposure
        diagnostics.append(
            {
                "field": "primary_predictor",
                "status": "missing_defaulted",
                "default_from": operational_source,
                "default": scoring_predictor,
            }
        )
    if not operational_exposure and scoring_predictor:
        operational_exposure = scoring_predictor
        diagnostics.append(
            {
                "field": "operational_exposure",
                "status": "missing_defaulted",
                "default_from": scoring_source,
                "default": operational_exposure,
            }
        )
    if not operational_exposure:
        diagnostics.append(
            {
                "field": "operational_exposure",
                "status": "missing_no_default_available",
                "default": None,
            }
        )
    cohort_column_names = {str(column) for column in cohort_columns}
    operational_column_present = (
        operational_exposure in cohort_column_names if operational_exposure else None
    )
    if operational_exposure and not operational_column_present:
        diagnostics.append(
            {
                "field": "operational_exposure",
                "status": "column_not_present",
                "value": operational_exposure,
                "source_field": operational_source,
            }
        )

    try:
        expected_direction = int(row.get("expected_or_direction") or 0)
    except (TypeError, ValueError):
        expected_direction = 0
        diagnostics.append(
            {
                "field": "expected_or_direction",
                "status": "invalid_value_defaulted",
                "default": 0,
            }
        )

    difficulty_raw = str(row.get("difficulty") or "").strip().lower()
    difficulty = _EXTERNAL_DIFFICULTY_ALIASES.get(difficulty_raw, difficulty_raw)
    if difficulty not in {"basic", "intermediate", "advanced"}:
        difficulty = "intermediate"
        diagnostics.append(
            {
                "field": "difficulty",
                "status": "missing_or_invalid_defaulted",
                "default": difficulty,
            }
        )

    category = str(row.get("category") or "").strip().lower()
    if category not in {"evaluation", "self_check"}:
        category = "evaluation"
        if row.get("category") not in (None, ""):
            diagnostics.append(
                {
                    "field": "category",
                    "status": "invalid_value_defaulted",
                    "default": category,
                }
            )

    gold_answer = row.get("gold_answer")
    if gold_answer is not None and not isinstance(gold_answer, Mapping):
        diagnostics.append(
            {
                "field": "gold_answer",
                "status": "invalid_type_defaulted",
                "source_type": type(gold_answer).__name__,
                "default": None,
            }
        )
        gold_answer = None

    gold_status = str(row.get("gold_answer_status") or "").strip().lower()
    if gold_status and gold_status not in {"planned", "frozen"}:
        diagnostics.append(
            {
                "field": "gold_answer_status",
                "status": "invalid_value_defaulted",
                "default": "frozen" if gold_answer is not None else "planned",
            }
        )
        gold_status = ""

    expected_findings = _external_string_list(
        row, "expected_finding_substrings", diagnostics
    )
    expected_steps = _external_string_list(row, "expected_step_substrings", diagnostics)
    expected_artifacts = _external_string_list(
        row, "expected_artifact_substrings", diagnostics
    )

    protocol_adapter = {
        "schema_version": (
            "easyicu.external_benchmark_adapter/3"
            if trajectory_authority_ref is not None
            else (
                "easyicu.external_benchmark_adapter/2"
                if cohort_authority_ref is not None
                else "easyicu.external_benchmark_adapter/1"
            )
        ),
        "database": {
            "value": database,
            "source_field": database_source,
            "defaulted": database_source is None,
        },
        "scoring_predictor": {
            "value": scoring_predictor or None,
            "source_field": scoring_source,
            "defaulted": scoring_source is None,
        },
        "operational_exposure": {
            "value": operational_exposure or None,
            "source_field": operational_source,
            "defaulted": operational_source is None,
            "declared_column_present": (
                operational_column_present if operational_source is not None else None
            ),
            "resolved_column_present": operational_column_present,
        },
        "diagnostics": diagnostics,
    }

    return SimpleNamespace(
        key=key,
        name=str(row.get("name") or key),
        research_question=str(question),
        target_outcome=str(target),
        database=database,
        primary_predictor=scoring_predictor,
        operational_exposure=operational_exposure or None,
        expected_or_direction=expected_direction,
        expected_finding_substrings=expected_findings,
        expected_step_substrings=expected_steps,
        expected_artifact_substrings=expected_artifacts,
        expected_outputs=_external_string_list(row, "expected_outputs", diagnostics),
        semantic_guardrails=_external_string_list(
            row, "semantic_guardrails", diagnostics
        ),
        evaluation_notes=_external_string_list(row, "evaluation_notes", diagnostics),
        target_databases=_external_string_list(row, "target_databases", diagnostics),
        required_warnings=_external_string_list(row, "required_warnings", diagnostics),
        forbidden_outputs=_external_string_list(row, "forbidden_outputs", diagnostics),
        numeric_targets=(
            dict(row.get("numeric_targets") or {})
            if isinstance(row.get("numeric_targets"), Mapping)
            else {}
        ),
        gold_answer=dict(gold_answer) if isinstance(gold_answer, Mapping) else None,
        gold_answer_status=gold_status,
        gold_derivation=str(row.get("gold_derivation") or ""),
        data_fixture=(str(row.get("data_fixture") or "").strip() or None),
        inclusion_criteria=_external_string_list(
            row, "inclusion_criteria", diagnostics
        ),
        candidate_variables=_external_string_list(
            row, "candidate_variables", diagnostics
        ),
        kind=str(row.get("kind") or "descriptive_association"),
        difficulty=difficulty,
        category=category,
        benchmark_family=str(row.get("benchmark_family") or "external"),
        evidence_basis=str(row.get("evidence_basis") or "external_import"),
        claim_scope=str(row.get("claim_scope") or "external_import_only"),
        notes=(str(row.get("notes") or "").strip() or None),
        interpretation_note=(str(row.get("interpretation_note") or "").strip() or None),
        protocol_version=(str(row.get("protocol_version") or "").strip() or None),
        rubric_version=(str(row.get("rubric_version") or "").strip() or None),
        protocol_adapter=protocol_adapter,
        cohort_size=int(cohort_size),
        cohort_columns=[str(column) for column in cohort_columns],
        cohort_authority_path=cohort_authority_path,
        cohort_authority_ref=(
            dict(cohort_authority_ref) if cohort_authority_ref is not None else None
        ),
        trajectory_path=trajectory_path,
        trajectory_authority_path=trajectory_authority_path,
        trajectory_authority_ref=(
            dict(trajectory_authority_ref)
            if trajectory_authority_ref is not None
            else None
        ),
    )


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
    resume_run_id: Optional[str] = None,
    resume_from_step_id: Optional[str] = None,
    stop_after_step_id: Optional[str] = None,
    force_writer_probe: bool = False,
    allow_mock_aware: bool = False,
    require_figure2_paper_acceptance: bool = False,
) -> int:
    """Run an external EHRFlowBench-style JSONL export when available."""
    import pandas as pd
    from easyicu.research_agent.intake.materialized_metadata import (
        MaterializedCohortAuthorityRef,
        MaterializedMetadataError,
        load_verified_materialized_cohort_authority,
    )
    from easyicu.research_agent.intake.materialized_trajectory import (
        MaterializedTrajectoryAuthorityRef,
        MaterializedTrajectoryError,
        load_verified_materialized_trajectory_authority,
    )

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
    invalid_row_indices: set[int] = set()
    for line_number, line in enumerate(
        jsonl_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(_decode_jsonl_object(line))
        except _JSONLObjectDecodeError as exc:
            invalid_row_indices.add(len(rows))
            rows.append(
                {
                    "status": "invalid_json",
                    "error": str(exc),
                    "raw": line[:200],
                    "line": line_number,
                }
            )
    if resume_run_id and len(rows) != 1:
        raise SystemExit(
            "--resume-run-id requires a one-row EHRFlowBench JSONL file so the "
            "selected run_id maps to one item/arm folder."
        )

    scores: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    input_task_ids = [
        str(row.get("key") or row.get("id") or f"ehrflowbench_{idx:03d}")
        for idx, row in enumerate(rows)
    ]
    for idx, row in enumerate(rows):
        key = str(row.get("key") or row.get("id") or f"ehrflowbench_{idx:03d}")
        if idx in invalid_row_indices:
            pending.append({"key": key, **row})
            continue
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
        raw_authority_path = row.get("cohort_authority_path")
        raw_authority_ref = row.get("cohort_authority_ref")
        authority_required = row.get("cohort_authority_required")
        authority_declared = (
            raw_authority_path is not None or raw_authority_ref is not None
        )
        if authority_required is not None and not isinstance(authority_required, bool):
            pending.append({"key": key, "status": "invalid_cohort_authority_marker"})
            continue
        if (raw_authority_path is None) != (raw_authority_ref is None) or (
            authority_required is True and not authority_declared
        ):
            pending.append(
                {"key": key, "status": "incomplete_cohort_authority_declaration"}
            )
            continue
        cohort_authority_path: Optional[Path] = None
        cohort_authority_ref: Optional[MaterializedCohortAuthorityRef] = None
        if authority_declared:
            if not isinstance(raw_authority_ref, Mapping):
                pending.append(
                    {"key": key, "status": "invalid_cohort_authority_reference"}
                )
                continue
            try:
                cohort_authority_ref = MaterializedCohortAuthorityRef.from_dict(
                    raw_authority_ref
                )
                cohort_authority_path = Path(str(raw_authority_path)).expanduser()
                expected_path = path.parent / cohort_authority_ref.file
                if (
                    cohort_authority_path.is_symlink()
                    or cohort_authority_path.resolve() != expected_path.resolve()
                ):
                    raise MaterializedMetadataError(
                        "authority path does not match the declared reference"
                    )
                verified = load_verified_materialized_cohort_authority(
                    path,
                    expected_authority=cohort_authority_ref,
                )
                if verified is None:  # pragma: no cover - exact ref forbids legacy
                    raise MaterializedMetadataError(
                        "declared typed cohort lost its authority"
                    )
            except (OSError, MaterializedMetadataError, ValueError, TypeError) as exc:
                pending.append(
                    {
                        "key": key,
                        "status": "invalid_cohort_authority",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
        elif path.suffix.lower() in {".parquet", ".pq"}:
            try:
                discovered_authority = load_verified_materialized_cohort_authority(path)
            except MaterializedMetadataError as exc:
                pending.append(
                    {
                        "key": key,
                        "status": "invalid_cohort_authority",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if discovered_authority is not None:
                pending.append(
                    {
                        "key": key,
                        "status": "typed_cohort_authority_not_declared",
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
        raw_trajectory_path = row.get("trajectory_path")
        raw_trajectory_authority_path = row.get("trajectory_authority_path")
        raw_trajectory_authority_ref = row.get("trajectory_authority_ref")
        trajectory_authority_required = row.get("trajectory_authority_required")
        trajectory_authority_declared = (
            raw_trajectory_authority_path is not None
            or raw_trajectory_authority_ref is not None
        )
        if trajectory_authority_required is not None and not isinstance(
            trajectory_authority_required, bool
        ):
            pending.append(
                {"key": key, "status": "invalid_trajectory_authority_marker"}
            )
            continue
        if (raw_trajectory_authority_path is None) != (
            raw_trajectory_authority_ref is None
        ) or (
            trajectory_authority_required is True and not trajectory_authority_declared
        ):
            pending.append(
                {
                    "key": key,
                    "status": "incomplete_trajectory_authority_declaration",
                }
            )
            continue
        trajectory_path: Optional[Path] = None
        trajectory_authority_path: Optional[Path] = None
        trajectory_authority_ref: Optional[MaterializedTrajectoryAuthorityRef] = None
        if raw_trajectory_path:
            trajectory_candidate = Path(str(raw_trajectory_path)).expanduser()
            if trajectory_candidate.is_symlink() or not trajectory_candidate.is_file():
                pending.append(
                    {
                        "key": key,
                        "status": "pending_missing_trajectory",
                        "trajectory_path": str(trajectory_candidate.absolute()),
                    }
                )
                continue
            if trajectory_candidate.suffix.lower() not in {".parquet", ".pq"}:
                pending.append(
                    {
                        "key": key,
                        "status": "unsupported_trajectory_format",
                        "trajectory_path": str(trajectory_candidate.absolute()),
                    }
                )
                continue
            trajectory_path = trajectory_candidate.resolve(strict=True)
            if trajectory_authority_declared:
                if cohort_authority_ref is None or not isinstance(
                    raw_trajectory_authority_ref, Mapping
                ):
                    pending.append(
                        {"key": key, "status": "invalid_trajectory_authority_reference"}
                    )
                    continue
                try:
                    trajectory_authority_ref = (
                        MaterializedTrajectoryAuthorityRef.from_dict(
                            raw_trajectory_authority_ref
                        )
                    )
                    trajectory_authority_path = Path(
                        str(raw_trajectory_authority_path)
                    ).expanduser()
                    expected_path = trajectory_path.parent / (
                        trajectory_authority_ref.file
                    )
                    if (
                        trajectory_authority_path.is_symlink()
                        or trajectory_authority_path.resolve()
                        != expected_path.resolve()
                    ):
                        raise MaterializedTrajectoryError(
                            "trajectory authority path does not match its reference"
                        )
                    verified_trajectory = (
                        load_verified_materialized_trajectory_authority(
                            trajectory_path,
                            expected_authority=trajectory_authority_ref,
                            expected_universe_authority=cohort_authority_ref,
                        )
                    )
                    if verified_trajectory is None:
                        raise MaterializedTrajectoryError(
                            "declared typed trajectory lost its authority"
                        )
                except (
                    OSError,
                    MaterializedTrajectoryError,
                    ValueError,
                    TypeError,
                ) as exc:
                    pending.append(
                        {
                            "key": key,
                            "status": "invalid_trajectory_authority",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
            elif cohort_authority_ref is not None:
                pending.append(
                    {
                        "key": key,
                        "status": "typed_trajectory_authority_required",
                        "trajectory_path": str(trajectory_path),
                    }
                )
                continue
        elif trajectory_authority_declared:
            pending.append(
                {"key": key, "status": "trajectory_authority_without_artifact"}
            )
            continue
        item = _external_item_from_row(
            row=row,
            key=key,
            question=str(question),
            target=str(target),
            cohort_size=int(len(cohort)),
            cohort_columns=list(cohort.columns),
            cohort_authority_path=cohort_authority_path,
            cohort_authority_ref=(
                cohort_authority_ref.to_dict()
                if cohort_authority_ref is not None
                else None
            ),
            trajectory_path=trajectory_path,
            trajectory_authority_path=trajectory_authority_path,
            trajectory_authority_ref=(
                trajectory_authority_ref.to_dict()
                if trajectory_authority_ref is not None
                else None
            ),
        )
        # Resume support: skip items that already finished cleanly so a quota
        # 502 mid-batch never forces a full redo. An item counts as "done" only
        # if its latest run reached execution_complete — quota-disrupted
        # diagnostic_only runs are redone.
        if (
            reuse_existing
            and not resume_run_id
            and not _is_figure2_task_id(key)
            and _ehrflow_item_done(out_root / key)
        ):
            print(f"\n=== {key} — reuse existing complete run ===")
            pending.append({"key": key, "status": "reused_complete"})
            continue
        # Per-item isolation: a provider 502 / crash on one item must not abort
        # the remaining items. Record the failure and continue.
        try:
            score = _run_one_item_from_cohort(
                item=item,
                cohort=(path if cohort_authority_ref is not None else cohort),
                out_root=out_root,
                arms=arms,
                pipeline_options=dict(pipeline_options or {}),
                provider=provider,
                model=model,
                request_timeout=request_timeout,
                reuse_existing=reuse_existing,
                resume_run_id=resume_run_id,
                resume_from_step_id=resume_from_step_id,
                stop_after_step_id=stop_after_step_id,
                force_writer_probe=force_writer_probe,
            )
            scores.append(score)
        except Exception as exc:  # noqa: BLE001 — keep batch alive on 502/etc.
            import traceback as _tb

            tb = _tb.format_exc()
            print(
                f"[ehrflowbench] item {key} FAILED: {type(exc).__name__}: "
                f"{str(exc)[:200]}\n{tb}"
            )
            try:
                (out_root / key).mkdir(parents=True, exist_ok=True)
                (out_root / key / "item_exception_traceback.txt").write_text(
                    tb, encoding="utf-8"
                )
            except Exception:
                pass
            pending.append(
                {
                    "key": key,
                    "status": "item_exception",
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            )
            continue

    totals = _aggregate(scores) if scores else {"naive": {}, "aware": {}}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(jsonl_path),
        "seed": seed,
        "arms": _normalize_arms(arms),
        "pipeline_options": dict(pipeline_options or {}),
        "force_writer_probe": bool(force_writer_probe),
        "items": input_task_ids,
        "scores": scores,
        "pending": pending,
        "totals": totals,
    }
    results_path = out_root / "ehrflowbench_results.json"
    results_path.write_text(
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
    acceptance_status: str | None = None
    if require_figure2_paper_acceptance or any(
        _is_figure2_task_id(task_id) for task_id in input_task_ids
    ):
        from benchmarks.figure2_canonical9.evaluator.acceptance import (
            FIGURE2_PAPER_ACCEPTANCE_SCHEMA,
            Figure2AcceptanceIssue,
            Figure2PaperAcceptance,
            evaluate_figure2_paper_acceptance,
        )
        from benchmarks.figure2_canonical9.evaluator.rubric_v1 import (
            FIGURE2_TASK_IDS,
        )

        try:
            acceptance = evaluate_figure2_paper_acceptance(results_path)
        except Exception as exc:  # paper gate must not truncate item outputs
            acceptance = Figure2PaperAcceptance(
                schema_version=FIGURE2_PAPER_ACCEPTANCE_SCHEMA,
                status="invalid",
                results_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
                expected_task_ids=tuple(FIGURE2_TASK_IDS),
                observed_task_ids=tuple(input_task_ids),
                issues=(
                    Figure2AcceptanceIssue(
                        code="ACCEPTANCE_EVALUATOR_ERROR",
                        detail=f"{type(exc).__name__}: {exc}"[:2048],
                    ),
                ),
            )
        acceptance_path = out_root / "figure2_paper_acceptance.json"
        acceptance_path.write_text(
            json.dumps(
                acceptance.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        acceptance_status = acceptance.status
        print(f"  -> {acceptance_path}")
    print(f"  -> {results_path}")
    print(f"  -> {out_root / 'ehrflowbench_results.md'}")
    if require_figure2_paper_acceptance and acceptance_status != "accepted":
        return _FIGURE2_PAPER_ACCEPTANCE_EXIT_CODE
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
    resume_run_id: Optional[str] = None,
    resume_from_step_id: Optional[str] = None,
    stop_after_step_id: Optional[str] = None,
    force_writer_probe: bool = False,
) -> Dict[str, Any]:
    item_root = out_root / item.key
    selected = set(_normalize_arms(arms))
    naive = _skipped_arm("naive")
    aware = _skipped_arm("aware")
    if reuse_existing and not resume_run_id:
        if "naive" in selected:
            naive = _reuse_arm_if_complete(
                arm_dir=item_root / "naive",
                item=item,
                label="naive",
            ) or _skipped_arm("naive")
        if "aware" in selected:
            aware = _reuse_arm_if_complete(
                arm_dir=item_root / "aware",
                item=item,
                label="aware",
            ) or _skipped_arm("aware")
    run_naive = "naive" in selected and not _arm_was_run(naive)
    run_aware = "aware" in selected and not _arm_was_run(aware)
    llm = (
        _make_llm(provider=provider, model=model, request_timeout=request_timeout)
        if run_naive or run_aware
        else None
    )
    if run_naive:
        naive = _run_one_arm(
            item=item,
            cohort=(cohort if isinstance(cohort, (str, Path)) else cohort.copy()),
            workdir=item_root / "naive",
            disable_icu_context=True,
            label="naive",
            llm=llm,
            pipeline_options=pipeline_options,
            reuse_existing=reuse_existing,
            resume_run_id=resume_run_id,
            resume_from_step_id=resume_from_step_id,
            stop_after_step_id=stop_after_step_id,
            force_writer_probe=force_writer_probe,
        )
    if run_aware:
        aware = _run_one_arm(
            item=item,
            cohort=(cohort if isinstance(cohort, (str, Path)) else cohort.copy()),
            workdir=item_root / "aware",
            disable_icu_context=False,
            label="aware",
            llm=llm,
            pipeline_options=pipeline_options,
            reuse_existing=reuse_existing,
            resume_run_id=resume_run_id,
            resume_from_step_id=resume_from_step_id,
            stop_after_step_id=stop_after_step_id,
            force_writer_probe=force_writer_probe,
        )
    cohort_size = getattr(item, "cohort_size", None)
    if cohort_size is None:
        cohort_size = len(cohort)
    payload = {
        "item_key": item.key,
        "name": item.name,
        "research_question": item.research_question,
        "expected_predictor": item.primary_predictor,
        "operational_exposure": getattr(item, "operational_exposure", None),
        "database": getattr(item, "database", "bench"),
        "expected_or_direction": item.expected_or_direction,
        "benchmark_family": getattr(item, "benchmark_family", "external"),
        "difficulty": getattr(item, "difficulty", "external"),
        "evidence_basis": getattr(item, "evidence_basis", "external_import"),
        "claim_scope": getattr(item, "claim_scope", "external_import_only"),
        "interpretation_note": getattr(item, "interpretation_note", None),
        "protocol_version": getattr(item, "protocol_version", None),
        "rubric_version": getattr(item, "rubric_version", None),
        "protocol_adapter": getattr(item, "protocol_adapter", None),
        "cohort_size": int(cohort_size),
    }
    if "naive" in selected:
        payload["naive"] = naive
    if "aware" in selected:
        payload["aware"] = aware
    return payload


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

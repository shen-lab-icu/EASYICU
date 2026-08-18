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
    from easyicu.research_agent.orchestration.profiles import SubmissionProfile


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


class _CohortMetadataError(ValueError):
    """Raised when bounded cohort metadata inspection cannot prove its shape."""


_FIGURE2_PAPER_ACCEPTANCE_EXIT_CODE = 3
# A run that did not finish its plan. Distinct from paper-acceptance (3) so a
# development diagnostic, which is never paper-accepted, still reports the
# difference between "ran and was not authorized" and "did not finish running".
_EXECUTION_INCOMPLETE_EXIT_CODE = 4
# An item the bench never started. Distinct from 4 on purpose: 4 says a run
# began and stopped short, 5 says the item never entered the pipeline at all
# (missing fields, missing cohort, an authority marker that would not load).
# Both used to exit 0, so a JSONL whose every row was rejected at intake read
# as a clean pass -- `scores=0, pending=1, exit=0`.
_PENDING_ITEMS_EXIT_CODE = 5

_CODEX_USER_SESSION_BINDING_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CODEX_APP_SERVER_REASONING_EFFORT = "low"

_SCIENTIFIC_RUNTIME_AUTHORITY_OPTIONS = (
    "trajectory_scientific_runtime_authority",
    "current_case_scientific_runtime_authority",
)


def _bind_runtime_scientific_projection_options(
    pipeline_options: Optional[Mapping[str, Any]],
    runtime_projection: object,
) -> Dict[str, Any]:
    """Bind a reviewed runtime projection without permitting an override."""

    options = dict(pipeline_options or {})
    if not isinstance(runtime_projection, Mapping):
        return options
    execution_contract = runtime_projection.get("deterministic_execution_contract")
    if not isinstance(execution_contract, Mapping):
        return options
    projection_digest = str(
        runtime_projection.get("runtime_projection_sha256") or ""
    ).strip()
    schema_version = str(execution_contract.get("schema_version") or "")
    authority_option_by_schema = {
        "easyicu.trajectory_scientific_runtime_authority/1": (
            "trajectory_scientific_runtime_authority"
        ),
        "easyicu.landmark_spline_runtime_authority/1": (
            "current_case_scientific_runtime_authority"
        ),
        "easyicu.source_feasibility_runtime_authority/1": (
            "current_case_scientific_runtime_authority"
        ),
        "easyicu.association_model_grid_runtime_authority/1": (
            "current_case_scientific_runtime_authority"
        ),
    }
    try:
        authority_option = authority_option_by_schema[schema_version]
    except KeyError as exc:
        raise ValueError(
            "SCIENTIFIC_RUNTIME_AUTHORITY_SCHEMA_UNSUPPORTED: "
            f"{schema_version or '<missing>'}"
        ) from exc

    expected_contract = dict(execution_contract)
    for option_name in _SCIENTIFIC_RUNTIME_AUTHORITY_OPTIONS:
        if option_name not in options:
            continue
        supplied = options[option_name]
        if option_name != authority_option or supplied != expected_contract:
            raise ValueError(
                f"SCIENTIFIC_RUNTIME_AUTHORITY_OVERRIDE_FORBIDDEN: {option_name}"
            )
    supplied_digest = options.get("scientific_runtime_projection_sha256")
    if supplied_digest is not None and str(supplied_digest) != projection_digest:
        raise ValueError("SCIENTIFIC_RUNTIME_PROJECTION_OVERRIDE_FORBIDDEN")
    options[authority_option] = expected_contract
    options["scientific_runtime_projection_sha256"] = projection_digest
    return options


def _is_figure2_task_id(value: object) -> bool:
    """Return True only for an exact frozen Canonical9 identifier."""

    from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS

    return type(value) is str and value in FIGURE2_TASK_IDS


def _operational_exposure_for_item(item: object) -> object:
    """Resolve the execution exposure once without laundering invalid values."""

    declared = getattr(item, "operational_exposure", None)
    if declared is not None:
        return declared
    legacy_predictor = getattr(item, "primary_predictor", None)
    # Historical multi-input rows encode an absent predictor as ``""``.  The
    # run-input capsule and posthoc evaluator require that absence to be an
    # explicit JSON null.  Preserve every other non-null value so whitespace,
    # false booleans, and other malformed coordinates still fail closed.
    return None if legacy_predictor == "" else legacy_predictor


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
    from easyicu.research_agent.orchestration.profiles import get_submission_profile

    try:
        return get_submission_profile(profile_ref)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _default_submission_profile_ref() -> str:
    """Resolve the benchmark CLI default from the canonical registry."""

    _bootstrap_imports()
    from easyicu.research_agent.orchestration.profiles import (
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

    module_name = f"benchmarks.cases.{case_name}.register_patterns"
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


def _figure2_canary_passed(score: Mapping[str, Any]) -> bool:
    """Require the first formal Canonical9 item to clear paper-facing gates."""

    aware = score.get("aware")
    if not isinstance(aware, Mapping):
        return False
    requires_scientific_closure = "e1_scientific_closure" in str(
        score.get("protocol_version") or ""
    )
    scientific_acceptance = aware.get("scientific_acceptance")
    scientific_acceptance_ok = (
        isinstance(scientific_acceptance, Mapping)
        and scientific_acceptance.get("status") == "accepted"
    )
    evaluation = aware.get("figure2_evaluation_attempt")
    paper_tristate: Optional[str] = None
    if isinstance(evaluation, Mapping):
        envelope = evaluation.get("envelope")
        paper_scorecard = (
            envelope.get("scorecard") if isinstance(envelope, Mapping) else None
        )
        canonical_scorecard = (
            paper_scorecard.get("scorecard_canonical_json")
            if isinstance(paper_scorecard, Mapping)
            else None
        )
        if isinstance(canonical_scorecard, str):
            try:
                parsed_scorecard = json.loads(canonical_scorecard)
            except (TypeError, ValueError):
                parsed_scorecard = None
            if isinstance(parsed_scorecard, Mapping):
                paper_tristate = str(parsed_scorecard.get("tristate") or "")
    return bool(
        aware.get("publication_artifacts_ready")
        and aware.get("execution_paper_eligible")
        and aware.get("paper_authorized")
        and aware.get("publication_ready")
        and aware.get("manuscript_ready")
        and int(aware.get("n_errors") or 0) == 0
        and isinstance(evaluation, Mapping)
        and evaluation.get("status") == "valid"
        and paper_tristate == "gate_reportable"
        and (not requires_scientific_closure or scientific_acceptance_ok)
    )


def _write_figure2_canary_gate(
    *,
    out_root: Path,
    task_id: str,
    score: Optional[Mapping[str, Any]],
    status: str,
    reason: str,
) -> Path:
    """Persist why a formal batch did or did not advance beyond its canary."""

    score_payload = dict(score or {})
    payload = {
        "schema_version": "easyicu.figure2_canary_gate/1",
        "task_id": str(task_id),
        "status": str(status),
        "reason": str(reason),
        "score_sha256": (
            hashlib.sha256(
                json.dumps(
                    score_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            if score_payload
            else None
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(out_root) / "figure2_canary_gate.json"
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


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


def _ensure_formal_figure2_safety_and_rescore(
    *,
    score: Dict[str, Any],
    item: Any,
    provider_environment: Optional[Mapping[str, str]],
    request_timeout: float,
) -> None:
    """Close the evaluator-only safety chain for one formal aware-arm run.

    Development runs intentionally keep the missing-receipt diagnostic.  The
    authorized batch path calls this function after Agent execution and before
    the E1 canary decision, so the Agent never sees the rubric and the paper
    scorer never runs against an implicitly trusted model response.
    """

    aware = score.get("aware")
    if not isinstance(aware, dict):
        return
    attempt = aware.get("figure2_evaluation_attempt")
    if not isinstance(attempt, dict):
        return
    if attempt.get("status") == "valid":
        return
    reason_codes = set(attempt.get("invalid_reason_codes") or ())
    if "SAFETY_ADJUDICATION_MISSING" not in reason_codes:
        return
    workdir = aware.get("workdir")
    task_id = str(getattr(item, "key", "") or "")
    if not isinstance(workdir, str) or not workdir or not task_id:
        return

    from benchmarks.figure2_canonical9.evaluator.safety_runner import (
        Figure2SafetyAdjudicationError,
        LocalOpenAICompatibleSafetyTransport,
        ensure_figure2_safety_receipt,
    )

    environment = dict(provider_environment or {})
    api_key = str(environment.get("OPENAI_API_KEY") or "")
    diagnostic: Dict[str, str] | None = None
    try:
        transport = LocalOpenAICompatibleSafetyTransport(
            api_key=api_key,
            timeout_seconds=float(request_timeout),
        )
        ensure_figure2_safety_receipt(
            Path(workdir),
            task_id=task_id,
            transport=transport,
        )
    except Figure2SafetyAdjudicationError as exc:
        diagnostic = {
            "code": exc.code,
            "stage": exc.stage,
            "detail": str(exc)[:1800],
        }
    except Exception as exc:
        diagnostic = {
            "code": "SAFETY_TRANSPORT_CONFIG_INVALID",
            "stage": "transport_config",
            "detail": f"{type(exc).__name__}: {exc}"[:1800],
        }
    if diagnostic is not None:
        aware["figure2_safety_adjudication_error"] = diagnostic
        print(
            "[figure2-safety] "
            f"{task_id} blocked at {diagnostic['stage']}: {diagnostic['code']}"
        )
    aware["figure2_evaluation_attempt"] = _figure2_evaluation_attempt(
        run_dir=Path(workdir),
        item=item,
    )


def _failed_step_ids(readiness: Mapping[str, Any]) -> List[str]:
    """Return the step ids the run itself recorded as failed."""

    failed: List[str] = []
    for entry in readiness.get("failed_steps") or []:
        step_id = entry.get("step_id") if isinstance(entry, Mapping) else entry
        if step_id:
            failed.append(str(step_id))
    return failed


def _arm_execution_succeeded(arm: Any) -> bool:
    """Return whether one arm actually executed its plan to completion.

    This is deliberately NOT a paper-authority check. A development diagnostic
    ends ``status='diagnostic_only'`` and ``paper_authorized=false`` by design,
    and that is a legitimate completed execution. What is never a completed
    execution is a run whose own ``run_status.json`` reports unfinished or
    failed steps -- exactly the state that previously reported
    ``completed_tasks=1, failed_or_blocked_tasks=0`` and exit 0.
    """

    if not isinstance(arm, Mapping):
        return False
    scientific_acceptance = arm.get("scientific_acceptance")
    scientific_acceptance_ok = (
        not isinstance(scientific_acceptance, Mapping)
        or scientific_acceptance.get("status") == "accepted"
    )
    return bool(
        arm.get("execution_complete")
        and arm.get("step_scientific_requirements_complete")
        and not arm.get("failed_step_ids")
        and not arm.get("missing_step_ids")
        and scientific_acceptance_ok
    )


def _score_execution_failures(score: Any) -> List[str]:
    """Return a reason per arm that did not finish executing."""

    if not isinstance(score, Mapping):
        return ["benchmark item produced no score payload"]
    arms = [
        (label, score.get(label))
        for label in ("aware", "naive")
        if isinstance(score.get(label), Mapping)
    ]
    if not arms:
        return ["benchmark item produced no scored arm"]
    reasons: List[str] = []
    for label, arm in arms:
        if _arm_execution_succeeded(arm):
            continue
        assert isinstance(arm, Mapping)
        failed = list(arm.get("failed_step_ids") or [])
        missing = list(arm.get("missing_step_ids") or [])
        detail = []
        if failed:
            detail.append(f"failed steps {failed}")
        if missing:
            detail.append(f"missing steps {missing}")
        if not arm.get("step_scientific_requirements_complete") and not detail:
            detail.append("step scientific requirements incomplete")
        scientific_acceptance = arm.get("scientific_acceptance")
        if (
            isinstance(scientific_acceptance, Mapping)
            and scientific_acceptance.get("status") != "accepted"
        ):
            reason_codes = [
                str(issue.get("reason_code"))
                for issue in scientific_acceptance.get("issues") or []
                if isinstance(issue, Mapping) and issue.get("reason_code")
            ]
            detail.append(
                "scientific acceptance rejected"
                + (f" ({', '.join(reason_codes[:5])})" if reason_codes else "")
            )
        if not detail:
            detail.append("execution_complete is false")
        completed = arm.get("completed_step_count")
        required = arm.get("required_step_count")
        position = (
            f" ({completed}/{required} steps)"
            if isinstance(completed, int) and isinstance(required, int) and required
            else ""
        )
        reasons.append(
            f"{label} arm did not complete execution{position}: " + "; ".join(detail)
        )
    return reasons


def _finish_task_on_execution_outcome(task_hard_stop: Any, score: Any) -> None:
    """Close the ledger task on what the run did, not on the call returning."""

    failures = _score_execution_failures(score)
    if failures:
        task_hard_stop.finish(score=score, error="; ".join(failures)[:1800])
        return
    task_hard_stop.finish(score=score)


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
        # The execution axis, surfaced explicitly at the benchmark boundary.
        # A development-diagnostic run is expected to end paper_authorized=false,
        # but it must still have RUN: without these fields the only signal the
        # outer driver had was "the call returned", which reports a run that
        # failed two steps as a completed task.
        "execution_complete": bool(readiness.get("execution_complete")),
        "step_scientific_requirements_complete": bool(
            readiness.get("step_scientific_requirements_complete")
        ),
        "required_step_count": int(readiness.get("required_step_count") or 0),
        "completed_step_count": int(readiness.get("completed_step_count") or 0),
        "failed_step_ids": _failed_step_ids(readiness),
        "missing_step_ids": [
            str(step_id) for step_id in (readiness.get("missing_steps") or [])
        ],
        "manuscript_ready": bool(readiness.get("manuscript_ready")),
        "publication_ready": bool(readiness.get("publication_ready")),
        # Keep the three independent completion axes visible at the benchmark
        # boundary.  ``publication_ready`` describes artifact completeness; it
        # must never be interpreted as paper authority on its own.
        "publication_artifacts_ready": bool(
            readiness.get("publication_artifacts_ready")
        ),
        "execution_paper_eligible": bool(readiness.get("execution_paper_eligible")),
        "paper_authorized": bool(readiness.get("paper_authorized")),
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
        result["execution_identity"] = manifest.get("execution_identity")
        result["figure2_evaluation_attempt"] = _figure2_evaluation_attempt(
            run_dir=run_dir,
            item=item,
        )
    scientific_contract = getattr(item, "scientific_acceptance_contract", None)
    if isinstance(scientific_contract, Mapping):
        from benchmarks.figure2_canonical9.e1_scientific_acceptance import (
            write_e1_scientific_acceptance_receipt,
        )

        scientific_receipt, scientific_receipt_path = (
            write_e1_scientific_acceptance_receipt(
                run_dir=run_dir,
                contract=scientific_contract,
            )
        )
        result["scientific_acceptance"] = scientific_receipt
        result["scientific_acceptance_receipt"] = str(scientific_receipt_path)
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
    provider_hard_stop: Optional[Any] = None,
) -> Dict[str, Any]:
    from easyicu.research_agent import ResearchAgentPipeline  # type: ignore
    from easyicu.research_agent.cohort.schema import cohort_concept_id_scope
    from easyicu.research_agent.reporting.reporting_checklist import (
        checklist_names_for_kind,
    )
    from benchmarks.figure2_canonical9.protocol_prompt import (
        task_protocol_note_for_item,
        task_protocol_preferences_for_item,
    )

    # The provided cohort is already materialised; let the planner reference any
    # of its columns in a CTAS predicate without tripping the static dictionary
    # check ("unknown concept_id: <derived column>").
    #
    # SCOPED, not registered. This function runs once PER CASE and the whole
    # batch shares one process, so a permanent registration accumulated every
    # earlier case's cohort columns: case N's planner could name a column only
    # case N-1 materialised, pass validation against the leaked registry, and
    # fail at execution -- after the provider calls were already paid for. The
    # scope restores the exact prior set on exit, including on failure, so each
    # case validates against its own cohort and nothing else.
    with cohort_concept_id_scope(
        list(getattr(item, "cohort_columns", None) or getattr(cohort, "columns", []))
    ):
        workdir.mkdir(parents=True, exist_ok=True)
        # Force the kind-matched reporting checklist(s) so the EMITTED file matches
        # what the scorecard READS by task kind (single source of truth:
        # ``checklist_names_for_kind``). Without this the pipeline falls back to
        # free-text analysis-family inference, which emitted STROBE for the
        # mortality_prediction task while the scorecard expected TRIPOD+AI — so
        # reporting_completeness was silently NA on a run that did reach the write
        # phase (detector/emitter contract mismatch, G-2).
        scientific_contract = getattr(item, "scientific_acceptance_contract", None)
        runtime_projection = getattr(item, "runtime_scientific_projection", None)
        if (
            runtime_projection is None
            and isinstance(scientific_contract, Mapping)
            and scientific_contract.get("schema_version")
            == "easyicu.e1_scientific_acceptance_contract/1"
        ):
            from benchmarks.figure2_canonical9.e1_scientific_acceptance import (
                build_e1_model_grid_runtime_projection,
            )

            runtime_projection = build_e1_model_grid_runtime_projection(
                scientific_contract
            )
        opts = _bind_runtime_scientific_projection_options(
            pipeline_options,
            runtime_projection,
        )
        opts.setdefault(
            "reporting_checklist_names",
            list(checklist_names_for_kind(getattr(item, "kind", None))),
        )
        # The authoritative task kind lets the internal phenotype checklist decide
        # trajectory-item applicability by kind (cross-sectional clustering vs
        # longitudinal) instead of fragile manuscript wording (M3 false-open).
        opts.setdefault("task_kind", getattr(item, "kind", None))
        if isinstance(scientific_contract, Mapping):
            required_cohort_mode = str(
                scientific_contract.get("primary_cohort_selection_mode") or ""
            ).strip()
            if required_cohort_mode:
                opts.setdefault(
                    "required_primary_cohort_selection_mode",
                    required_cohort_mode,
                )
        pipeline = ResearchAgentPipeline(
            workdir=workdir,
            llm=llm,
            provider_hard_stop=provider_hard_stop,
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
            id_columns=(getattr(item, "id_columns", None) or None),
            user_preferences=task_protocol_preferences_for_item(item),
            notes=task_protocol_note_for_item(item),
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
    provider_hard_stop: Optional[Any] = None,
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
            provider_hard_stop=provider_hard_stop,
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
            provider_hard_stop=provider_hard_stop,
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
    *, arm_dir: Path, item, label: str, expected_execution_identity_sha256: str
) -> Optional[Dict[str, Any]]:
    if not arm_dir.exists():
        return None
    runs = sorted(
        (
            p
            for p in arm_dir.glob("run_*")
            if (p / "manifest.json").exists()
            and _manifest_execution_identity_matches(
                p,
                expected_execution_identity_sha256,
            )
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


def _manifest_execution_identity_matches(
    run_dir: Path,
    expected_sha256: str,
) -> bool:
    """Require an exact validated ExecutionIdentity before reusing a run."""

    from easyicu.research_agent.authority.execution_identity import ExecutionIdentity

    try:
        manifest = _load_manifest(run_dir)
        identity = ExecutionIdentity.model_validate(
            manifest.get("execution_identity"),
            strict=True,
        )
    except Exception:
        return False
    return identity.identity_sha256 == expected_sha256


def _benchmark_execution_identity(
    pipeline_options: Optional[Dict[str, Any]],
    llm: Any = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    reasoning_effort_profile: str = "provider_default",
    request_timeout: float = 120.0,
    transport_max_attempts: int = 1,
    stream_enabled: bool = False,
    planner_strict_json_schema: bool = False,
    provider_environment: Optional[Mapping[str, str]] = None,
):
    from easyicu.research_agent.authority.execution_identity import ExecutionIdentity
    from easyicu.research_agent.providers.factory import (
        provider_authorization_for_configuration,
    )

    options = dict(pipeline_options or {})
    provider_authorization = None
    if llm is None:
        if provider is None or model is None:
            raise ValueError("benchmark execution identity requires provider/model")
        provider_authorization = provider_authorization_for_configuration(
            provider=provider,
            model=model,
            reasoning_effort_profile=reasoning_effort_profile,
            request_timeout=request_timeout,
            transport_max_attempts=transport_max_attempts,
            stream_enabled=stream_enabled,
            supports_strict_json_schema=bool(planner_strict_json_schema),
            environment=provider_environment,
        )
    return ExecutionIdentity.create(
        submission_profile_name=options.get("submission_profile_name"),
        submission_profile_version=options.get("submission_profile_version"),
        runner=str(options.get("runner_kind") or "auto"),
        runner_image_digest=options.get("expected_runner_image_digest"),
        network_policy=str(options.get("runner_network") or "none"),
        provider_client=llm,
        provider_authorization=provider_authorization,
        llm_seed=options.get("llm_seed"),
        data_seed=options.get("execution_data_seed"),
        input_authority_sha256=options.get("execution_input_authority_sha256"),
        host_runner_authorized=bool(options.get("host_runner_authorized", False)),
    )


def _benchmark_input_authority_sha256(cohort: Any) -> str:
    """Hash the exact benchmark input without exposing its values."""

    def _canonical_bytes(payload: object) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        ).encode("utf-8")

    if isinstance(cohort, (str, Path)):
        from benchmarks.figure2_canonical9.realrun_authority import (
            production_cohort_input_sha256,
        )

        # ONE algorithm, shared with the real-run freeze gate's per-task input
        # digest, so the frozen ``input_sha256`` and this runtime-bound digest can
        # never silently diverge.
        return production_cohort_input_sha256(cohort)

    try:
        import pandas as pd

        if isinstance(cohort, pd.DataFrame):
            digest = hashlib.sha256()
            digest.update(
                _canonical_bytes(
                    {
                        "kind": "dataframe",
                        "columns": [str(column) for column in cohort.columns],
                        "dtypes": [str(dtype) for dtype in cohort.dtypes],
                        "row_count": int(len(cohort)),
                    }
                )
            )
            try:
                hashed = pd.util.hash_pandas_object(
                    cohort,
                    index=True,
                    categorize=True,
                )
                digest.update(hashed.to_numpy(dtype="uint64", copy=False).tobytes())
            except (TypeError, ValueError):
                digest.update(
                    cohort.to_json(
                        orient="split",
                        date_format="iso",
                        date_unit="ns",
                        default_handler=str,
                    ).encode("utf-8")
                )
            return digest.hexdigest()
    except ImportError:  # pragma: no cover - benchmark installs pandas
        pass

    return hashlib.sha256(
        _canonical_bytes({"kind": "jsonable", "value": cohort})
    ).hexdigest()


def _bind_benchmark_execution_input(
    pipeline_options: Optional[Mapping[str, Any]],
    *,
    cohort: Any,
    data_seed: int | None,
) -> Dict[str, Any]:
    """Bind reuse and the persisted run identity to the current input."""

    options = dict(pipeline_options or {})
    input_digest = _benchmark_input_authority_sha256(cohort)
    declared_digest = options.get("execution_input_authority_sha256")
    if declared_digest is not None and declared_digest != input_digest:
        raise ValueError("benchmark input authority override does not match cohort")
    declared_seed = options.get("execution_data_seed")
    if declared_seed is not None and declared_seed != data_seed:
        raise ValueError("benchmark data seed override does not match invocation")
    options["execution_input_authority_sha256"] = input_digest
    options["execution_data_seed"] = data_seed
    return options


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
    provider_hard_stop: Optional[Any] = None,
) -> Dict[str, Any]:
    if verbose:
        print(f"\n=== {item.key} — {item.name} ===")
    cohort = item.cohort_factory(seed)
    bound_pipeline_options = _bind_benchmark_execution_input(
        pipeline_options,
        cohort=cohort,
        data_seed=seed,
    )
    item_root = out_root / item.key
    selected = set(_normalize_arms(arms))
    expected_identity = _benchmark_execution_identity(bound_pipeline_options, llm)

    naive = _skipped_arm("naive")
    aware = _skipped_arm("aware")
    if reuse_existing and not resume_run_id:
        if "naive" in selected:
            naive = _reuse_arm_if_complete(
                arm_dir=item_root / "naive",
                item=item,
                label="naive",
                expected_execution_identity_sha256=expected_identity.identity_sha256,
            ) or _skipped_arm("naive")
        if "aware" in selected:
            aware = _reuse_arm_if_complete(
                arm_dir=item_root / "aware",
                item=item,
                label="aware",
                expected_execution_identity_sha256=expected_identity.identity_sha256,
            ) or _skipped_arm("aware")

    if "naive" in selected and not _arm_was_run(naive):
        naive = _run_one_arm(
            item=item,
            cohort=cohort.copy(),
            workdir=item_root / "naive",
            disable_icu_context=True,
            label="naive",
            llm=llm,
            pipeline_options=bound_pipeline_options,
            resume_run_id=resume_run_id,
            resume_from_step_id=resume_from_step_id,
            stop_after_step_id=stop_after_step_id,
            force_writer_probe=force_writer_probe,
            provider_hard_stop=provider_hard_stop,
        )
    if "aware" in selected and not _arm_was_run(aware):
        aware = _run_one_arm(
            item=item,
            cohort=cohort.copy(),
            workdir=item_root / "aware",
            disable_icu_context=False,
            label="aware",
            llm=llm,
            pipeline_options=bound_pipeline_options,
            resume_run_id=resume_run_id,
            resume_from_step_id=resume_from_step_id,
            stop_after_step_id=stop_after_step_id,
            force_writer_probe=force_writer_probe,
            provider_hard_stop=provider_hard_stop,
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


def _make_llm(
    *,
    provider: str,
    model: str,
    request_timeout: float,
    reasoning_effort_profile: str = "provider_default",
    transport_max_attempts: int = 1,
    stream_enabled: bool = False,
    planner_strict_json_schema: bool = False,
    provider_environment: Optional[Mapping[str, str]] = None,
):
    if not isinstance(transport_max_attempts, int) or isinstance(
        transport_max_attempts, bool
    ):
        raise ValueError("transport_max_attempts must be a positive integer")
    total_transport_attempts = transport_max_attempts
    if total_transport_attempts <= 0:
        raise ValueError("transport_max_attempts must be a positive integer")
    _bootstrap_imports()
    from easyicu.research_agent import (  # type: ignore
        LLMRouter,
        MockLLMClient,
    )
    from easyicu.research_agent.providers import (  # type: ignore
        ANTHROPIC_MESSAGES,
        ProviderConfigurationError,
        SUPPORTED_CLI_ACCOUNT_NAMES,
        build_provider_client,
        cli_account_profile,
        provider_profile,
        user_account_profile,
    )
    from easyicu.research_agent.providers.factory import authorize_provider_client
    from easyicu.research_agent.providers.llm import (
        CodexAppServerLLMClient,
        build_llm_client,
        reasoning_effort_by_role,
    )

    if provider == "mock":
        return MockLLMClient()
    if provider in SUPPORTED_CLI_ACCOUNT_NAMES:
        cli_profile = cli_account_profile(provider)
        assert cli_profile is not None
        if reasoning_effort_profile != "provider_default":
            raise SystemExit(
                "Account-backed CLI providers currently require "
                "--reasoning-effort-profile provider_default."
            )
        if total_transport_attempts != 1 or stream_enabled:
            raise SystemExit(
                "Account-backed CLI providers use one non-streaming transport "
                "attempt per logical call."
            )
        if planner_strict_json_schema and not (cli_profile.supports_strict_json_schema):
            raise SystemExit(
                f"Provider {provider!r} cannot honor strict Planner JSON Schema."
            )
        account_environment = dict(provider_environment or {})
        session_binding = str(
            account_environment.get("EASYICU_CODEX_SESSION_SHA256") or ""
        ).strip()
        if session_binding:
            account_profile = user_account_profile(provider)
            if account_profile is None:
                raise SystemExit(
                    f"Provider {provider!r} has no reviewed App Server account path."
                )
            selected_model = str(model or "").strip()
            if selected_model in {"", "cli-default", "account-default"}:
                selected_model = "account-default"
            try:
                client = CodexAppServerLLMClient(
                    model=(
                        None if selected_model == "account-default" else selected_model
                    ),
                    request_timeout=request_timeout,
                    turn_hard_timeout=request_timeout,
                    reasoning_effort=_CODEX_APP_SERVER_REASONING_EFFORT,
                    environment=account_environment,
                )
                endpoint = (
                    f"{account_profile.endpoint_identity}/session/{session_binding}"
                )
                return authorize_provider_client(
                    client,
                    provider=account_profile.provider_identity,
                    model=selected_model,
                    base_url=endpoint,
                    destination="external",
                    environment=account_environment,
                )
            except (RuntimeError, ValueError, ProviderConfigurationError) as exc:
                raise SystemExit(str(exc)) from exc
        try:
            return build_llm_client(
                prefer=provider,
                model=None if model == "cli-default" else model,
                allow_mock=False,
                ladder=[provider],
                request_timeout=request_timeout,
                environment=provider_environment,
            ).client
        except (RuntimeError, ValueError, ProviderConfigurationError) as exc:
            raise SystemExit(str(exc)) from exc
    try:
        # The CLI option is deliberately expressed as total physical attempts,
        # while OpenAIClient's conventional ``max_retries`` contract counts
        # only attempts after the initial request.
        transport_max_retries = total_transport_attempts - 1
        effort_by_role = reasoning_effort_by_role(reasoning_effort_profile)
        profile_definition = provider_profile(provider)
        if (
            effort_by_role
            and profile_definition is not None
            and profile_definition.transport == ANTHROPIC_MESSAGES
        ):
            raise SystemExit(
                "Anthropic native API currently requires "
                "--reasoning-effort-profile provider_default."
            )
        if not effort_by_role:
            return build_provider_client(
                provider=provider,
                model=model,
                request_timeout=request_timeout,
                title="EasyICU research-agent benchmark",
                environment=provider_environment,
                max_retries=transport_max_retries,
                stream_enabled=bool(stream_enabled),
                supports_strict_json_schema=bool(planner_strict_json_schema),
                allow_environment_overrides=False,
            )
        clients_by_effort = {
            effort: build_provider_client(
                provider=provider,
                model=model,
                request_timeout=request_timeout,
                title="EasyICU research-agent benchmark",
                environment=provider_environment,
                extra_body={"reasoning": {"effort": effort}},
                max_retries=transport_max_retries,
                stream_enabled=bool(stream_enabled),
                supports_strict_json_schema=bool(planner_strict_json_schema),
                allow_environment_overrides=False,
            )
            for effort in sorted(set(effort_by_role.values()))
        }
        role_clients = {
            role: clients_by_effort[effort] for role, effort in effort_by_role.items()
        }
        return LLMRouter(
            default=role_clients["coder"],
            planner=role_clients["planner"],
            coder=role_clients["coder"],
            analyzer=role_clients["analyzer"],
            writer=role_clients["writer"],
            literature=role_clients["literature"],
            repair=role_clients["repair"],
            reasoning_effort_profile=reasoning_effort_profile,
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
    from easyicu.research_agent.providers import (
        cli_account_profile,
        resolve_provider_base_url,
    )

    cli_profile = cli_account_profile(provider)
    if cli_profile is not None:
        return cli_profile.endpoint_identity

    return resolve_provider_base_url(provider)


def _default_model_for_provider(provider: str) -> str:
    """Return a provider-family-safe development default."""

    _bootstrap_imports()
    from easyicu.research_agent.providers import cli_account_profile, provider_profile

    normalized = str(provider or "").strip().lower()
    if normalized == "mock":
        return "mock"
    account = cli_account_profile(normalized)
    if account is not None:
        _source, configured_model = account.model(os.environ)
        return configured_model or "cli-default"
    profile = provider_profile(normalized)
    _source, configured_model = (
        profile.model(os.environ) if profile is not None else (None, "")
    )
    selected = configured_model or os.environ.get("EASYICU_HOSTED_DEFAULT_MODEL", "")
    if selected:
        return selected
    if normalized == "openrouter":
        return "openai/gpt-oss-120b:free"
    if normalized == "openai":
        return "gpt-4o-mini"
    raise SystemExit(f"--model is required for --provider {normalized}")


def _provider_environment_snapshot(
    *, provider: str, provider_base_url: str
) -> Dict[str, str]:
    """Freeze endpoint semantics while retaining only required credentials."""

    _bootstrap_imports()
    from easyicu.research_agent.providers import (
        cli_account_profile,
        is_loopback_openai_base_url,
        provider_profile,
    )
    from easyicu.research_agent.providers.subprocess_env import (
        build_provider_subprocess_env,
    )

    cli_profile = cli_account_profile(provider)
    if cli_profile is not None:
        snapshot = build_provider_subprocess_env(provider)
        if "EASYICU_ALLOW_EXTERNAL_LLM" in os.environ:
            snapshot["EASYICU_ALLOW_EXTERNAL_LLM"] = os.environ[
                "EASYICU_ALLOW_EXTERNAL_LLM"
            ]
        return snapshot
    profile = provider_profile(provider)
    keys = {"EASYICU_ALLOW_EXTERNAL_LLM"}
    if (
        profile is not None
        and profile.supports_auth_header_override
        and is_loopback_openai_base_url(provider_base_url)
    ):
        keys.update(
            {
                "EASYICU_OPENAI_AUTH_HEADER",
                "EASYICU_TRUST_LOOPBACK_PROXY_KEY",
            }
        )
    if profile is not None:
        keys.update(profile.api_key_env_names)
    snapshot = {key: os.environ[key] for key in keys if key in os.environ}
    if profile is not None:
        snapshot[profile.base_url_env_names[0]] = str(provider_base_url)
    return snapshot


def _validated_development_codex_session_binding(
    binding: object,
    *,
    provider: str,
    model: object,
    development_diagnostic: bool,
    formal_authority_requested: bool,
    multiple_models_requested: bool,
    explicit_provider_base_url: object,
    reasoning_effort_profile: str,
    transport_max_attempts: int,
    stream_enabled: bool,
) -> str | None:
    """Validate the narrow benchmark bridge to one Web-managed Codex login."""

    normalized = str(binding or "").strip().lower()
    if not normalized:
        return None
    if not _CODEX_USER_SESSION_BINDING_PATTERN.fullmatch(normalized):
        raise SystemExit(
            "--codex-user-session-binding must be one 64-character lowercase "
            "SHA-256 binding returned by the authenticated Web session."
        )
    if str(provider or "").strip().lower() != "codex":
        raise SystemExit("--codex-user-session-binding requires --provider codex.")
    if not development_diagnostic or formal_authority_requested:
        raise SystemExit(
            "--codex-user-session-binding is development-only and cannot enter "
            "a formal, submission-profile, or paper-acceptance run."
        )
    if multiple_models_requested:
        raise SystemExit(
            "--codex-user-session-binding requires exactly one --model."
        )
    selected_model = str(model or "").strip()
    if selected_model in {"", "cli-default", "account-default"}:
        raise SystemExit(
            "--codex-user-session-binding requires an explicit account model."
        )
    if str(explicit_provider_base_url or "").strip():
        raise SystemExit(
            "--codex-user-session-binding owns its App Server endpoint; "
            "do not pass --provider-base-url."
        )
    if str(reasoning_effort_profile or "").strip() != "provider_default":
        raise SystemExit(
            "Codex App Server account runs require "
            "--reasoning-effort-profile provider_default."
        )
    if transport_max_attempts != 1 or stream_enabled:
        raise SystemExit(
            "Codex App Server account runs use one non-streaming transport "
            "attempt per logical call."
        )
    return normalized


def _development_codex_session_environment(
    binding_sha256: str,
    *,
    model: str,
) -> tuple[Dict[str, str], str]:
    """Resolve a verified private Web session without returning its credential."""

    _bootstrap_imports()
    from easyicu.research_agent.providers import user_account_profile
    from easyicu.webserver.codex_account_sessions import (
        CodexAccountSessionError,
        environment_for_binding,
    )

    try:
        environment = environment_for_binding(binding_sha256, model=model)
    except CodexAccountSessionError as exc:
        raise SystemExit(str(exc)) from exc
    profile = user_account_profile("codex")
    assert profile is not None
    endpoint = f"{profile.endpoint_identity}/session/{binding_sha256}"
    return dict(environment), endpoint


def _provider_hard_stop_limits(
    pipeline_options: Mapping[str, Any],
):
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLimits,
    )

    required = {
        "max_provider_attempts_per_run",
        "max_provider_attempts_per_batch",
        "max_total_tokens_per_run",
        "max_total_tokens_per_batch",
        "max_estimated_cost_usd_per_batch",
        "max_wall_clock_seconds_per_task",
        "provider_input_cost_usd_per_million_tokens",
        "provider_output_cost_usd_per_million_tokens",
    }
    present = required.intersection(pipeline_options)
    if not present:
        return None
    if present != required:
        missing = ", ".join(sorted(required - present))
        raise ValueError(f"Incomplete Provider hard-stop options: {missing}")
    return ProviderHardStopLimits(
        max_provider_attempts_per_run=int(
            pipeline_options["max_provider_attempts_per_run"]
        ),
        max_provider_attempts_per_batch=int(
            pipeline_options["max_provider_attempts_per_batch"]
        ),
        max_total_tokens_per_run=int(pipeline_options["max_total_tokens_per_run"]),
        max_total_tokens_per_batch=int(pipeline_options["max_total_tokens_per_batch"]),
        max_estimated_cost_usd_per_batch=float(
            pipeline_options["max_estimated_cost_usd_per_batch"]
        ),
        max_wall_clock_seconds_per_task=float(
            pipeline_options["max_wall_clock_seconds_per_task"]
        ),
        input_cost_usd_per_million_tokens=float(
            pipeline_options["provider_input_cost_usd_per_million_tokens"]
        ),
        output_cost_usd_per_million_tokens=float(
            pipeline_options["provider_output_cost_usd_per_million_tokens"]
        ),
    )


def _bind_benchmark_cost_price_table(
    pipeline_options: Optional[Mapping[str, Any]],
    *,
    provider: str,
    model: str,
) -> Dict[str, Any]:
    """Bind the benchmark's reviewed token prices to its selected model.

    Provider hard-stop accounting already uses the two explicit per-million
    rates.  CostMeter needs the same rates keyed by model; otherwise an
    unlisted gateway alias records exact tokens but silently reports ``$0``.
    Mock runs remain unpriced, and an explicit conflicting table fails closed.
    """

    options = dict(pipeline_options or {})
    if not bool(options.get("enable_cost_tracking")) or provider == "mock":
        return options
    input_price = options.get("provider_input_cost_usd_per_million_tokens")
    output_price = options.get("provider_output_cost_usd_per_million_tokens")
    if input_price is None and output_price is None:
        return options
    if input_price is None or output_price is None:
        raise ValueError("Benchmark cost tracking requires both Provider prices")
    selected_model = str(model or "").strip()
    if not selected_model:
        raise ValueError("Benchmark cost tracking requires a selected model")
    selected_prices = (float(input_price), float(output_price))
    raw_table = options.get("cost_price_table")
    if raw_table is None:
        price_table: Dict[str, Any] = {}
    elif isinstance(raw_table, Mapping):
        price_table = dict(raw_table)
    else:
        raise ValueError("cost_price_table must be a model-to-price mapping")
    existing = price_table.get(selected_model)
    if existing is not None:
        try:
            existing_prices = tuple(float(value) for value in existing)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid cost_price_table entry for {selected_model!r}"
            ) from exc
        if existing_prices != selected_prices:
            raise ValueError(
                "CostMeter prices conflict with Provider hard-stop prices for "
                f"{selected_model!r}"
            )
    price_table[selected_model] = selected_prices
    options["cost_price_table"] = price_table
    return options


def _benchmark_pipeline_options(
    *,
    max_total_steps: Optional[int],
    disable_replanning: bool,
    max_code_repair_attempts: Optional[int],
    planner_strategy: str = "monolithic_v1",
    max_step_llm_repair_attempts: Optional[int] = None,
    max_step_provider_calls: int = 9,
    max_provider_attempts_per_run: int = 192,
    max_provider_attempts_per_batch: int = 1_728,
    max_total_tokens_per_run: int = 2_000_000,
    max_total_tokens_per_batch: int = 18_000_000,
    max_estimated_cost_usd_per_batch: float = 100.0,
    max_wall_clock_seconds_per_task: float = 21_600.0,
    provider_input_cost_usd_per_million_tokens: float = 10.0,
    provider_output_cost_usd_per_million_tokens: float = 30.0,
    timeout_seconds: float = 900.0,
    standard_executor_timeout_seconds: float = 3_600.0,
    enable_repro_envelope: bool = True,
    enable_cost_tracking: bool = True,
    llm_seed: Optional[int] = None,
    writer_digest_widened: bool = False,
    strict_evidence: bool = False,
    enable_cross_run_memory: bool = False,
    enable_pubmed: bool = False,
    submission_profile: Optional["SubmissionProfile"] = None,
    runner_kind: Optional[str] = None,
    host_runner_authorized: bool = False,
    development_sample_size: Optional[int] = None,
    development_sample_seed: int = 20260719,
    development_diagnostic: bool = False,
    development_progressive_resume_checkpoint_path: Optional[Path] = None,
    development_progressive_resume_checkpoint_sha256: Optional[str] = None,
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
    if host_runner_authorized:
        options["host_runner_authorized"] = True
    if development_sample_size is not None:
        if submission_profile is not None:
            raise SystemExit(
                "--development-sample-size is non-paper authority and cannot be "
                "combined with --submission-profile. Use it for development "
                "runs, then rerun the frozen task on full data for the paper."
            )
        if int(development_sample_size) <= 0:
            raise SystemExit("--development-sample-size must be positive.")
        options["development_sample_size"] = int(development_sample_size)
        options["development_sample_seed"] = int(development_sample_seed)
    if development_diagnostic:
        if submission_profile is not None:
            raise SystemExit(
                "--development-diagnostic is non-paper authority and cannot "
                "be combined with --submission-profile."
            )
        options["development_diagnostic"] = True
    progressive_resume_values = (
        development_progressive_resume_checkpoint_path,
        development_progressive_resume_checkpoint_sha256,
    )
    if any(value is not None for value in progressive_resume_values):
        if any(value is None for value in progressive_resume_values):
            raise SystemExit(
                "--development-progressive-resume-checkpoint and its SHA-256 "
                "must be supplied together."
            )
        options["development_progressive_resume_checkpoint_path"] = Path(
            development_progressive_resume_checkpoint_path
        )
        options["development_progressive_resume_checkpoint_sha256"] = str(
            development_progressive_resume_checkpoint_sha256
        )
    if enable_pubmed:
        options["enable_pubmed"] = True
    options["planner_strategy"] = str(planner_strategy)
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
    options["max_step_provider_calls"] = int(max_step_provider_calls)
    options["max_provider_attempts_per_run"] = int(max_provider_attempts_per_run)
    options["max_provider_attempts_per_batch"] = int(max_provider_attempts_per_batch)
    options["max_total_tokens_per_run"] = int(max_total_tokens_per_run)
    options["max_total_tokens_per_batch"] = int(max_total_tokens_per_batch)
    options["max_estimated_cost_usd_per_batch"] = float(
        max_estimated_cost_usd_per_batch
    )
    options["max_wall_clock_seconds_per_task"] = float(max_wall_clock_seconds_per_task)
    options["provider_input_cost_usd_per_million_tokens"] = float(
        provider_input_cost_usd_per_million_tokens
    )
    options["provider_output_cost_usd_per_million_tokens"] = float(
        provider_output_cost_usd_per_million_tokens
    )
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
    # Benchmark runs never make an implicit second Provider channel merely
    # because a model name happens to look vision-capable. Deterministic visual
    # QA remains enabled; paid image upload requires a separate, explicitly
    # reviewed experiment outside Canonical9.
    options["enable_vlm_visual_qa"] = False
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
            "`docker build -t easyicu-research-agent:1.0.0 -f "
            "src/easyicu/research_agent/runner_image/Dockerfile .`. For a "
            "non-archival development run only, pass '--allow-host-runner'."
        )
    return resolved


def _enforce_mock_aware_provider(
    arms: Sequence[str],
    *,
    provider: str,
    allow_mock_aware: bool = False,
    submission_profile: Optional["SubmissionProfile"] = None,
) -> None:
    """Reject mock-provider aware runs unless they are explicit smoke tests."""
    selected_arms = _normalize_arms(arms)
    if (
        "aware" in selected_arms
        and provider == "mock"
        and submission_profile is not None
        and submission_profile.requires_real_provider is True
    ):
        raise SystemExit(
            f"Submission profile '{submission_profile.ref}' requires a real "
            "provider. The mock aware arm is fixture-only and cannot receive "
            "paper-facing authority; drop --submission-profile for an offline "
            "plumbing smoke test."
        )
    # The MockLLMClient returns canned responses, so an "aware" arm run on
    # the mock provider reports fixture output rather than a genuine
    # ICU-aware analysis. Paper-facing results must use a real provider.
    # Offline plumbing smoke tests can opt in explicitly with
    # --allow-mock-aware.
    if "aware" in selected_arms and provider == "mock" and not allow_mock_aware:
        raise SystemExit(
            "The 'aware' arm on the mock provider returns pre-written, "
            "fixture responses, so its results are not real. Use "
            "a configured real --provider for paper-facing runs, restrict to "
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
    reasoning_effort_profile: str = "provider_default",
    transport_max_attempts: int = 1,
    stream_enabled: bool = False,
    planner_strict_json_schema: bool = False,
    provider_environment: Optional[Mapping[str, str]] = None,
    provider_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    pipeline_options = _bind_benchmark_cost_price_table(
        pipeline_options,
        provider=provider,
        model=model,
    )
    selected_arms = _normalize_arms(arms)
    _enforce_mock_aware_provider(
        selected_arms,
        provider=provider,
        allow_mock_aware=allow_mock_aware,
    )
    hard_stop_ledger = None
    hard_stop_limits = _provider_hard_stop_limits(pipeline_options or {})
    if hard_stop_limits is not None:
        from easyicu.research_agent.authority.provider_hard_stop import (
            ProviderHardStopLedger,
        )

        hard_stop_ledger = ProviderHardStopLedger(
            path=(out_root / "bench_progress.json").resolve(),
            task_ids=[str(item.key) for item in items],
            limits=hard_stop_limits,
            resume_existing=bool(reuse_existing),
        )
    llm = _make_llm(
        provider=provider,
        model=model,
        request_timeout=request_timeout,
        reasoning_effort_profile=reasoning_effort_profile,
        transport_max_attempts=transport_max_attempts,
        stream_enabled=stream_enabled,
        planner_strict_json_schema=planner_strict_json_schema,
        provider_environment=provider_environment,
    )
    from easyicu.research_agent import (  # type: ignore
        default_icu_agent_bench_suite,
        icu_agent_bench_markdown,
    )

    scores: List[Dict[str, Any]] = []
    for item in items:
        task_hard_stop = (
            hard_stop_ledger.start_task(str(item.key))
            if hard_stop_ledger is not None
            else None
        )
        try:
            score = _run_one_item_with_reuse(
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
                provider_hard_stop=task_hard_stop,
            )
        except BaseException as exc:
            if task_hard_stop is not None:
                task_hard_stop.finish(error=f"{type(exc).__name__}: {str(exc)[:1800]}")
            raise
        scores.append(score)
        if task_hard_stop is not None:
            _finish_task_on_execution_outcome(task_hard_stop, score)

    totals = _aggregate(scores)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_short_sha(),
        "seed": seed,
        "bench_kind": bench_kind,
        "provider": provider,
        "model": model,
        "reasoning_effort_profile": reasoning_effort_profile,
        "backend_base_url": (
            str(provider_base_url)
            if provider_base_url is not None
            else _resolve_backend_base_url(provider)
        ),
        "arms": selected_arms,
        "case_registration": case_registration,
        "force_writer_probe": bool(force_writer_probe),
        "provider_transport_options": {
            "planner_strict_json_schema": bool(planner_strict_json_schema),
        },
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


def _canonical_execution_config_from_args(args):
    """Normalize the run-semantics argv into the frozen canonical execution config.

    Pure — reads only parsed args, touches no Provider/runner/data.  The gate
    compares this object's digest to the operator's ``execution_config_sha256`` pin.
    """

    from benchmarks.figure2_canonical9.realrun_authority import (
        build_canonical_execution_config,
    )

    return build_canonical_execution_config(
        seed=int(getattr(args, "seed", 7)),
        timeout_seconds=float(getattr(args, "timeout", 900.0)),
        standard_executor_timeout_seconds=float(
            getattr(args, "standard_executor_timeout", 3600.0)
        ),
        request_timeout_seconds=float(getattr(args, "request_timeout", 180.0)),
        stop_after_step_id=getattr(args, "stop_after_step_id", None),
        llm_seed=getattr(args, "llm_seed", None),
        disable_replanning=bool(getattr(args, "disable_replanning", False)),
        max_total_steps=getattr(args, "max_total_steps", None),
        max_code_repair_attempts=getattr(args, "max_code_repair_attempts", None),
        max_step_llm_repair_attempts=getattr(
            args, "max_step_llm_repair_attempts", None
        ),
        max_step_provider_calls=int(getattr(args, "max_step_provider_calls", 9)),
        max_provider_attempts_per_run=int(
            getattr(args, "max_provider_attempts_per_run", 192)
        ),
        max_provider_attempts_per_batch=int(
            getattr(args, "max_provider_attempts_per_batch", 1_728)
        ),
        max_total_tokens_per_run=int(
            getattr(args, "max_total_tokens_per_run", 2_000_000)
        ),
        max_total_tokens_per_batch=int(
            getattr(args, "max_total_tokens_per_batch", 18_000_000)
        ),
        max_estimated_cost_usd_per_batch=float(
            getattr(args, "max_estimated_cost_usd_per_batch", 100.0)
        ),
        max_wall_clock_seconds_per_task=float(
            getattr(args, "max_wall_clock_seconds_per_task", 21_600.0)
        ),
        provider_input_cost_usd_per_million_tokens=float(
            getattr(
                args,
                "provider_input_cost_usd_per_million_tokens",
                10.0,
            )
        ),
        provider_output_cost_usd_per_million_tokens=float(
            getattr(
                args,
                "provider_output_cost_usd_per_million_tokens",
                30.0,
            )
        ),
        enable_repro_envelope=not bool(getattr(args, "no_repro_envelope", False)),
        enable_cost_tracking=not bool(getattr(args, "no_cost_tracking", False)),
        strict_evidence=bool(getattr(args, "strict_evidence", False)),
        writer_digest_widened=bool(getattr(args, "writer_digest_widened", False)),
        enable_pubmed=bool(getattr(args, "enable_pubmed", False)),
        case=getattr(args, "case", None),
        development_sample_size=getattr(args, "development_sample_size", None),
        development_sample_seed=int(getattr(args, "development_sample_seed", 20260719)),
        models=tuple(getattr(args, "models", None) or ()),
        reasoning_effort_profile=str(
            getattr(args, "reasoning_effort_profile", "provider_default")
        ),
        planner_strategy=str(getattr(args, "planner_strategy", "monolithic_v1")),
        transport_max_attempts=int(getattr(args, "transport_max_attempts", 1)),
        provider_base_url=(
            str(getattr(args, "provider_base_url", "") or "").strip()
            or _resolve_backend_base_url(str(getattr(args, "provider", "mock")))
        ),
        llm_stream_enabled=bool(getattr(args, "llm_stream", False)),
    )


def _verify_figure2_development_diagnostic(
    args,
    *,
    jsonl_path: Path,
    task_ids: tuple,
) -> None:
    """Verify an explicit non-paper Canonical9 development input binding."""

    if bool(getattr(args, "submission_profile", False)) or bool(
        getattr(args, "require_figure2_paper_acceptance", False)
    ):
        raise ValueError(
            "development diagnostics cannot enable a submission profile or "
            "paper acceptance"
        )
    if any(
        getattr(args, name, None)
        for name in (
            "figure2_realrun_authorization",
            "figure2_expected_execution_identity",
            "figure2_production_input_authority",
        )
    ):
        raise ValueError(
            "development diagnostics cannot carry paper authority coordinates"
        )
    if str(getattr(args, "runner", "") or "") != "docker":
        raise ValueError("development diagnostics require --runner docker")
    if _normalize_arms(getattr(args, "arms", None)) != ["aware"]:
        raise ValueError("development diagnostics require exactly --arms aware")
    if str(getattr(args, "provider", "") or "") == "mock":
        raise ValueError("development diagnostics require a real Provider")
    if not task_ids:
        raise ValueError("development diagnostic JSONL has no Canonical9 task")

    receipt_raw = str(
        getattr(args, "figure2_development_binding_receipt", "") or ""
    ).strip()
    if not receipt_raw:
        raise ValueError(
            "--development-diagnostic requires --figure2-development-binding-receipt"
        )
    receipt_path = Path(receipt_raw).expanduser()
    if not receipt_path.is_absolute() or receipt_path.is_symlink():
        raise ValueError("development binding receipt must be absolute and non-symlink")
    receipt_path = receipt_path.resolve(strict=True)
    if not receipt_path.is_file():
        raise ValueError("development binding receipt must be a regular file")
    payload = json.loads(
        receipt_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_jsonl_duplicate_pairs,
        parse_constant=_reject_jsonl_nonfinite_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("development binding receipt must be a JSON object")
    if (
        payload.get("schema_version")
        != "easyicu.canonical9_development_binding_receipt/1"
        or payload.get("paper_authority") is not False
    ):
        raise ValueError(
            "development binding receipt must explicitly deny paper authority"
        )
    if payload.get("output_jsonl") != str(jsonl_path):
        raise ValueError("development binding receipt does not bind this JSONL path")
    observed_sha256 = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    if payload.get("output_sha256") != observed_sha256:
        raise ValueError("development binding receipt does not bind these JSONL bytes")


def _figure2_realrun_authorization_gate(args):
    """Fail-closed real-run authorization, enforced before anything is launched.

    Returns ``(exit_code_or_None, batch_binding_or_None)``.  A non-None exit
    code stops the launcher immediately; ``None`` proceeds.  Called right after
    argument parsing, so a blocked / missing / tampered authority — and a Canonical9
    or paper-acceptance run that omits the declaration entirely — exits with zero
    pipeline / subprocess / Provider / data-load activity.

    The gate does NOT trust the declaration's own restated fields: it builds a
    :class:`RealRunInvocation` from the actual argv (plus the handoff JSONL keys and
    cohort paths) and verifies the declaration matches that real intent knob-for-knob.
    """

    from benchmarks.figure2_canonical9.realrun_authority import (
        RealRunAuthorizationRequest,
        RealRunBatchBinding,
        RealRunInvocation,
        jsonl_references_canonical9,
        read_canonical_jsonl_invocation,
        resolve_strict_jsonl_path,
        reserve_authorized_batch_root,
        verify_realrun_authorization,
    )

    declaration = getattr(args, "figure2_realrun_authorization", None)
    jsonl = getattr(args, "ehrflowbench_jsonl", None)
    require_acceptance = bool(getattr(args, "require_figure2_paper_acceptance", False))
    development_diagnostic = bool(getattr(args, "development_diagnostic", False))

    # First classify an existing JSONL without changing ordinary, non-canonical
    # EHRFlowBench behavior.  If it references Canonical9 (or if explicit paper
    # authority was requested), we immediately switch to the strict absolute,
    # non-symlink resolver below.  A relative/symlink canonical manifest therefore
    # cannot evade the gate, while legacy non-canonical fixtures retain their
    # longstanding CLI semantics.
    strict_jsonl: Optional[Path] = None
    task_ids: tuple = ()
    cohort_paths: tuple = ()
    if jsonl:
        try:
            probe_jsonl = Path(jsonl).expanduser().resolve(strict=True)
            if not probe_jsonl.is_file():
                raise ValueError("ehrflowbench JSONL must be a regular file")
            task_ids, cohort_paths = read_canonical_jsonl_invocation(probe_jsonl)
        except Exception as exc:  # noqa: BLE001
            if declaration or require_acceptance:
                print(
                    "[realrun-authority] an authority-requested JSONL must be "
                    "readable for Canonical9 classification; refusing to launch "
                    f"({type(exc).__name__}: {exc}).",
                    file=sys.stderr,
                )
                return 2, None

    references_canonical = bool(task_ids) and jsonl_references_canonical9(task_ids)
    real_canonical_run = require_acceptance or references_canonical

    if declaration or real_canonical_run or development_diagnostic:
        try:
            strict_jsonl = resolve_strict_jsonl_path(jsonl)
            task_ids, cohort_paths = read_canonical_jsonl_invocation(strict_jsonl)
        except Exception as exc:  # noqa: BLE001
            print(
                "[realrun-authority] Canonical9 / authority-requested JSONL must "
                "be an absolute, regular, non-symlink, strictly-readable manifest; "
                f"refusing to launch ({type(exc).__name__}: {exc}).",
                file=sys.stderr,
            )
            return 2, None

    if development_diagnostic:
        if not references_canonical:
            print(
                "[development-diagnostic] the explicit development path is only "
                "valid for Canonical9 JSONL inputs.",
                file=sys.stderr,
            )
            return 2, None
        try:
            _verify_figure2_development_diagnostic(
                args,
                jsonl_path=strict_jsonl,
                task_ids=task_ids,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "[development-diagnostic] BLOCKED — no pipeline, subprocess, "
                f"Provider, or data load has started ({type(exc).__name__}: {exc}).",
                file=sys.stderr,
            )
            return 2, None
        print(
            "[development-diagnostic] verified non-paper input binding; results "
            "are development-only and require a fresh authorized rerun for paper use.",
            file=sys.stderr,
        )
        return None, None

    # 1) Mandatory activation: a Canonical9 / paper-acceptance run cannot bypass the
    #    gate by omitting the declaration.
    if not declaration:
        if real_canonical_run:
            print(
                "[realrun-authority] a Canonical9 / paper-acceptance run REQUIRES "
                "--figure2-realrun-authorization (plus "
                "--figure2-expected-execution-identity and "
                "--figure2-production-input-authority and "
                "--figure2-scientific-protocol-authority); refusing to launch.",
                file=sys.stderr,
            )
            return 2, None
        return None, None

    identity = getattr(args, "figure2_expected_execution_identity", None)
    if not identity:
        print(
            "[realrun-authority] --figure2-realrun-authorization requires "
            "--figure2-expected-execution-identity",
            file=sys.stderr,
        )
        return 2, None

    repo_root = Path(__file__).resolve().parents[1]
    production = getattr(args, "figure2_production_input_authority", None)
    scientific_protocols = getattr(
        args,
        "figure2_scientific_protocol_authority",
        None,
    )

    # Effective model + runner, exactly as the launcher will resolve them downstream.
    provider = str(getattr(args, "provider", "mock"))
    effective_model = (
        str(getattr(args, "model", "mock")) if provider != "mock" else "mock"
    )
    models = getattr(args, "models", None)
    if models:
        effective_model = str(models[0])
    requested_runner = getattr(args, "runner", None)
    profile_enabled = bool(getattr(args, "submission_profile", False))
    effective_runner = (
        str(requested_runner)
        if requested_runner
        else ("docker" if profile_enabled else "auto")
    )

    invocation = RealRunInvocation(
        arms=tuple(_normalize_arms(getattr(args, "arms", None))),
        task_ids=task_ids,
        task_cohort_paths=tuple(zip(task_ids, cohort_paths)),
        ehrflowbench_jsonl_path=strict_jsonl,
        provider=provider,
        model=effective_model,
        submission_profile_enabled=profile_enabled,
        submission_profile_ref=(str(getattr(args, "profile", "")) or None),
        runner=effective_runner,
        out_root=Path(getattr(args, "out_root", ".")),
        require_paper_acceptance=require_acceptance,
        execution_config=_canonical_execution_config_from_args(args),
        reuse_existing=bool(getattr(args, "reuse_existing", False)),
        repeat=int(getattr(args, "repeat", 1) or 1),
        force_writer_probe=bool(getattr(args, "force_writer_probe", False)),
        development_sample_size=getattr(args, "development_sample_size", None),
        allow_host_runner=bool(getattr(args, "allow_host_runner", False)),
        allow_mock_aware=bool(getattr(args, "allow_mock_aware", False)),
        resume_run_id=getattr(args, "resume_run_id", None),
        resume_from_step_id=getattr(args, "resume_from_step_id", None),
        cross_run_memory=bool(getattr(args, "enable_cross_run_memory", False)),
    )
    request = RealRunAuthorizationRequest(
        declaration_path=Path(declaration),
        expected_execution_identity_path=Path(identity),
        input_freeze_path=(
            repo_root / "benchmarks/figure2_canonical9/canonical_input_freeze_v1.json"
        ),
        rubric_path=(
            repo_root / "benchmarks/figure2_canonical9/figure2_paper_rubric_v3.json"
        ),
        invocation=invocation,
        production_input_authority_path=(Path(production) if production else None),
        scientific_protocol_authority_path=(
            Path(scientific_protocols) if scientific_protocols else None
        ),
    )
    authorization = verify_realrun_authorization(request)
    print(authorization.model_dump_json(indent=2))
    if authorization.status != "authorized":
        print(
            "[realrun-authority] BLOCKED — no pipeline, subprocess, Provider, or "
            "data load has started.",
            file=sys.stderr,
        )
        return 2, None

    # ``authorization`` contains the *already verified* declaration and input
    # authority values.  Do not reopen either mutable path after verification.
    # Reserve the batch root with mkdir(O_EXCL semantics) before a Provider, runner,
    # or data load can start; concurrent replay of a declaration loses this race.
    batch_binding = RealRunBatchBinding(
        batch_id=authorization.batch_id,
        declaration_sha256=authorization.declaration_sha256,
        input_authority_digest=authorization.input_authority_digest,
        scientific_protocol_authority_digest=(
            authorization.scientific_protocol_authority_digest
        ),
        frozen_input_by_task=authorization.frozen_input_by_task,
    )
    try:
        batch_binding = reserve_authorized_batch_root(
            invocation.out_root, batch_binding
        )
    except Exception as exc:  # noqa: BLE001
        print(
            "[realrun-authority] unable to atomically reserve the authorized batch "
            f"root; refusing to launch ({type(exc).__name__}: {exc}).",
            file=sys.stderr,
        )
        return 2, None
    print(
        "[realrun-authority] authority verified; launching the real run still "
        "requires the operator's explicit action.",
        file=sys.stderr,
    )
    return None, batch_binding


def main() -> int:
    _bootstrap_imports()

    from easyicu.research_agent.providers import (
        SUPPORTED_CLI_ACCOUNT_NAMES,
        SUPPORTED_PROVIDER_NAMES,
    )
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
        choices=["mock", *SUPPORTED_CLI_ACCOUNT_NAMES, *SUPPORTED_PROVIDER_NAMES],
        default="mock",
        help="LLM backend for the benchmark arms.",
    )
    parser.add_argument(
        "--codex-user-session-binding",
        default=None,
        help=(
            "Development-only SHA-256 binding for a verified Web-managed Codex "
            "account session. Uses App Server rather than the legacy codex CLI; "
            "requires --provider codex, one explicit --model, and "
            "--development-diagnostic."
        ),
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
        "--allow-pending-import",
        action="store_true",
        help=(
            "Exit 0 even when JSONL items were rejected at intake and never "
            "ran. Without this the bench exits 5, because an import whose rows "
            "all failed to load otherwise reports the same success as one that "
            "ran and passed."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Single model name for real-provider runs. Account CLIs default "
            "to their logged-in model; API providers retain the hosted-model "
            "development default. Formal runs should pin an exact model."
        ),
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
        "--transport-max-attempts",
        type=int,
        default=1,
        help=(
            "Maximum raw transport attempts for one logical Provider call "
            "(default: 1; retries are charged to the same global ledger)."
        ),
    )
    parser.add_argument(
        "--provider-base-url",
        default=None,
        help=(
            "Explicit non-secret Provider endpoint for a frozen run. When "
            "omitted, it is resolved once from the Provider environment."
        ),
    )
    parser.add_argument(
        "--llm-stream",
        action="store_true",
        help=(
            "Use streaming transport. Off by default and frozen into Canonical9 "
            "execution authority; EASYICU_LLM_STREAM cannot override it."
        ),
    )
    parser.add_argument(
        "--planner-strict-json-schema",
        action="store_true",
        help=(
            "Require the configured OpenAI-compatible Planner transport to "
            "enforce the host-derived AnalysisPlan JSON Schema. The capability "
            "is explicit and becomes part of the provider transport policy, "
            "not PipelineConfig."
        ),
    )
    parser.add_argument(
        "--planner-strategy",
        choices=["monolithic_v1", "progressive_v2"],
        default="monolithic_v1",
        help=(
            "Planner contract materialization strategy. Progressive v2 asks "
            "for a compact skeleton, compiles host-owned execution details, "
            "and revises only an unlocked suffix."
        ),
    )
    parser.add_argument(
        "--development-progressive-resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Development-only terminal Progressive Planner checkpoint path. "
            "Requires one selected JSONL item/arm and the exact file SHA-256."
        ),
    )
    parser.add_argument(
        "--development-progressive-resume-checkpoint-sha256",
        default=None,
        help=(
            "Exact SHA-256 of --development-progressive-resume-checkpoint. "
            "Formal paper-facing profiles reject this option."
        ),
    )
    parser.add_argument(
        "--max-step-provider-calls",
        type=int,
        default=9,
        help="Maximum Provider transport attempts charged to one analysis step.",
    )
    parser.add_argument(
        "--max-provider-attempts-per-run",
        type=int,
        default=192,
        help="Hard Provider transport-attempt ceiling for one benchmark task.",
    )
    parser.add_argument(
        "--max-provider-attempts-per-batch",
        type=int,
        default=1_728,
        help="Hard Provider transport-attempt ceiling for the complete batch.",
    )
    parser.add_argument(
        "--max-total-tokens-per-run",
        type=int,
        default=2_000_000,
        help="Hard reserved/reported token ceiling for one benchmark task.",
    )
    parser.add_argument(
        "--max-total-tokens-per-batch",
        type=int,
        default=18_000_000,
        help="Hard reserved/reported token ceiling for the complete batch.",
    )
    parser.add_argument(
        "--max-estimated-cost-usd-per-batch",
        type=float,
        default=100.0,
        help="Hard estimated USD ceiling for the complete batch.",
    )
    parser.add_argument(
        "--max-wall-clock-seconds-per-task",
        type=float,
        default=21_600.0,
        help="Hard wall-clock ceiling for one task (default: 6 hours).",
    )
    parser.add_argument(
        "--provider-input-cost-usd-per-million-tokens",
        type=float,
        default=10.0,
        help="Frozen conservative input-token price used by the pre-call stop-loss.",
    )
    parser.add_argument(
        "--provider-output-cost-usd-per-million-tokens",
        type=float,
        default=30.0,
        help="Frozen conservative output-token price used by the pre-call stop-loss.",
    )
    parser.add_argument(
        "--reasoning-effort-profile",
        choices=["provider_default", "adaptive_v1"],
        default="provider_default",
        help=(
            "Per-request reasoning policy. 'provider_default' sends no override; "
            "'adaptive_v1' uses medium for planning/coding, low for "
            "analysis/writing/literature, and high only for validated repairs. "
            "The selected profile is bound into Canonical9 execution authority."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help=(
            "Per-attempt timeout in seconds for ordinary model-generated "
            "analysis code (default: 900). This does not change the LLM request timeout or "
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
            "resumed step at a time. Use '@first' for the first planned step, "
            "'@index:N' for the one-based Nth step, or "
            "'@product:kind:name' for the unique producer of a typed output."
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
        "--development-sample-size",
        type=int,
        default=None,
        help=(
            "Non-paper acceleration: after the Agent locks and materializes the "
            "analysis cohort and QC, execute a deterministic identity-hash "
            "sample of this many stays (for example 1000). Any trajectory is "
            "filtered to the same stays. Incompatible with --submission-profile "
            "and Figure 2 paper acceptance."
        ),
    )
    parser.add_argument(
        "--development-sample-seed",
        type=int,
        default=20260719,
        help="Stable seed for --development-sample-size (default: 20260719).",
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
        "--enable-pubmed",
        action="store_true",
        help=(
            "Retrieve similar PubMed studies before planning and pass bounded, "
            "source-backed study-design excerpts to the Planner. Network failure "
            "degrades to the curated offline literature registry."
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
            "Optional case protocol directory name under benchmarks/cases. "
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
        "--development-diagnostic",
        action="store_true",
        help=(
            "Run a Canonical9 input only as an explicitly non-paper Docker "
            "diagnostic. Requires --figure2-development-binding-receipt; the "
            "result cannot be promoted and must be rerun under fresh paper authority."
        ),
    )
    parser.add_argument(
        "--figure2-development-binding-receipt",
        default=None,
        help=(
            "Absolute non-symlink receipt binding the exact development JSONL "
            "path and SHA256 with paper_authority=false."
        ),
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
        "--figure2-expected-execution-identity",
        default=None,
        help=(
            "Absolute path to an operator-frozen ExpectedExecutionIdentity JSON. "
            "Required with --require-figure2-paper-acceptance and forbidden "
            "inside the benchmark output root."
        ),
    )
    parser.add_argument(
        "--figure2-realrun-authorization",
        default=None,
        help=(
            "Path to an operator freeze declaration JSON. When set, the real-run "
            "authorization gate is enforced BEFORE any pipeline, subprocess, "
            "Provider, or data load; a blocked/missing/tampered authority exits "
            "with status 2 having launched nothing. Requires "
            "--figure2-expected-execution-identity."
        ),
    )
    parser.add_argument(
        "--figure2-production-input-authority",
        default=None,
        help=(
            "Path to a typed full-9 production input authority JSON. Absent means "
            "the input is not yet frozen for a real run (the v1 assessment stays "
            "blocked), so the gate fails closed."
        ),
    )
    parser.add_argument(
        "--figure2-scientific-protocol-authority",
        default=None,
        help=(
            "Absolute path to the operator-pinned, digest-bound E2/H2/H3 "
            "clinical-and-methods protocol authority. Missing or invalid review "
            "evidence blocks before any cohort data is read."
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
    if not args.model and not args.models:
        args.model = _default_model_for_provider(str(args.provider))
    formal_authority_requested = bool(
        args.submission_profile
        or args.require_figure2_paper_acceptance
        or args.figure2_realrun_authorization
        or args.figure2_production_input_authority
        or args.figure2_scientific_protocol_authority
        or args.figure2_expected_execution_identity
    )
    codex_user_session_binding = _validated_development_codex_session_binding(
        args.codex_user_session_binding,
        provider=str(args.provider),
        model=args.model,
        development_diagnostic=bool(args.development_diagnostic),
        formal_authority_requested=formal_authority_requested,
        multiple_models_requested=bool(args.models),
        explicit_provider_base_url=args.provider_base_url,
        reasoning_effort_profile=str(args.reasoning_effort_profile),
        transport_max_attempts=int(args.transport_max_attempts),
        stream_enabled=bool(args.llm_stream),
    )
    codex_account_environment: Dict[str, str] | None = None
    if codex_user_session_binding is not None:
        codex_account_environment, codex_account_endpoint = (
            _development_codex_session_environment(
                codex_user_session_binding,
                model=str(args.model),
            )
        )
        args.provider_base_url = codex_account_endpoint
    # Resolve once. The authorization digest and every subsequently created
    # client receive this exact endpoint; later environment mutation is inert.
    args.provider_base_url = str(
        args.provider_base_url or ""
    ).strip() or _resolve_backend_base_url(str(args.provider))
    _realrun_gate_rc, _figure2_batch_binding = _figure2_realrun_authorization_gate(args)
    if _realrun_gate_rc is not None:
        return _realrun_gate_rc
    provider_environment = (
        codex_account_environment
        if codex_account_environment is not None
        else _provider_environment_snapshot(
            provider=str(args.provider),
            provider_base_url=str(args.provider_base_url),
        )
    )
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
    _enforce_mock_aware_provider(
        args.arms,
        provider=args.provider,
        allow_mock_aware=bool(args.allow_mock_aware),
        submission_profile=submission_profile,
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
        planner_strategy=str(args.planner_strategy),
        max_step_llm_repair_attempts=max_step_llm_repair_attempts,
        max_step_provider_calls=int(args.max_step_provider_calls),
        max_provider_attempts_per_run=int(args.max_provider_attempts_per_run),
        max_provider_attempts_per_batch=int(args.max_provider_attempts_per_batch),
        max_total_tokens_per_run=int(args.max_total_tokens_per_run),
        max_total_tokens_per_batch=int(args.max_total_tokens_per_batch),
        max_estimated_cost_usd_per_batch=float(args.max_estimated_cost_usd_per_batch),
        max_wall_clock_seconds_per_task=float(args.max_wall_clock_seconds_per_task),
        provider_input_cost_usd_per_million_tokens=float(
            args.provider_input_cost_usd_per_million_tokens
        ),
        provider_output_cost_usd_per_million_tokens=float(
            args.provider_output_cost_usd_per_million_tokens
        ),
        timeout_seconds=float(args.timeout),
        standard_executor_timeout_seconds=float(args.standard_executor_timeout),
        enable_repro_envelope=not bool(getattr(args, "no_repro_envelope", False)),
        enable_cost_tracking=not bool(getattr(args, "no_cost_tracking", False)),
        llm_seed=getattr(args, "llm_seed", None),
        writer_digest_widened=bool(args.writer_digest_widened),
        strict_evidence=bool(args.strict_evidence),
        enable_cross_run_memory=bool(getattr(args, "enable_cross_run_memory", False)),
        enable_pubmed=bool(getattr(args, "enable_pubmed", False)),
        submission_profile=submission_profile,
        runner_kind=runner_kind,
        host_runner_authorized=bool(getattr(args, "allow_host_runner", False)),
        development_sample_size=getattr(args, "development_sample_size", None),
        development_sample_seed=int(getattr(args, "development_sample_seed", 20260719)),
        development_diagnostic=bool(getattr(args, "development_diagnostic", False)),
        development_progressive_resume_checkpoint_path=getattr(
            args,
            "development_progressive_resume_checkpoint",
            None,
        ),
        development_progressive_resume_checkpoint_sha256=getattr(
            args,
            "development_progressive_resume_checkpoint_sha256",
            None,
        ),
    )
    planner_strict_json_schema = bool(
        getattr(args, "planner_strict_json_schema", False)
    )

    if (
        bool(args.require_figure2_paper_acceptance)
        and getattr(args, "development_sample_size", None) is not None
    ):
        raise SystemExit(
            "--require-figure2-paper-acceptance cannot score a non-paper "
            "development sample. Rerun the frozen task on the full cohort."
        )
    expected_execution_identity_path = getattr(
        args,
        "figure2_expected_execution_identity",
        None,
    )
    if bool(args.require_figure2_paper_acceptance) and not (
        str(expected_execution_identity_path or "").strip()
    ):
        raise SystemExit(
            "--require-figure2-paper-acceptance requires "
            "--figure2-expected-execution-identity."
        )

    progressive_resume_requested = (
        getattr(args, "development_progressive_resume_checkpoint", None)
        is not None
    )
    if progressive_resume_requested and not args.ehrflowbench_jsonl:
        raise SystemExit(
            "--development-progressive-resume-checkpoint requires "
            "--ehrflowbench-jsonl so one source question can be selected."
        )
    if progressive_resume_requested and (
        _figure2_batch_binding is not None
        or bool(args.require_figure2_paper_acceptance)
    ):
        raise SystemExit(
            "FORMAL_PROGRESSIVE_CHECKPOINT_RESUME_FORBIDDEN: cross-run Planner "
            "checkpoint reuse is development-only and cannot enter a formal "
            "Figure 2 batch."
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
        if n_repeat > 1 and progressive_resume_requested:
            raise SystemExit(
                "--repeat cannot be combined with development Progressive "
                "Planner checkpoint resume."
            )

        # Canonical9 batches run exactly the strict path verified by the gate.  An
        # ordinary EHRFlowBench JSONL keeps the pre-existing relative/symlink CLI
        # behavior, which is intentionally not paper authority.
        if _figure2_batch_binding is not None:
            from benchmarks.figure2_canonical9.realrun_authority import (
                resolve_strict_jsonl_path as _resolve_strict_jsonl_path,
            )

            _ehrflow_jsonl_path = _resolve_strict_jsonl_path(args.ehrflowbench_jsonl)
        else:
            _ehrflow_jsonl_path = Path(args.ehrflowbench_jsonl).expanduser().resolve()

        def _run_ehrflow_into(target_out_root: Path) -> int:
            return _run_ehrflowbench_jsonl(
                jsonl_path=_ehrflow_jsonl_path,
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
                allow_pending=bool(args.allow_pending_import),
                require_figure2_paper_acceptance=bool(
                    args.require_figure2_paper_acceptance
                ),
                expected_execution_identity_path=(
                    Path(expected_execution_identity_path).expanduser().resolve()
                    if expected_execution_identity_path
                    else None
                ),
                batch_binding=_figure2_batch_binding,
                reasoning_effort_profile=str(args.reasoning_effort_profile),
                transport_max_attempts=int(args.transport_max_attempts),
                stream_enabled=bool(args.llm_stream),
                planner_strict_json_schema=planner_strict_json_schema,
                provider_environment=provider_environment,
                provider_base_url=str(args.provider_base_url),
                items=args.items,
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
            reasoning_effort_profile=str(args.reasoning_effort_profile),
            transport_max_attempts=int(args.transport_max_attempts),
            stream_enabled=bool(args.llm_stream),
            planner_strict_json_schema=planner_strict_json_schema,
            provider_environment=provider_environment,
            provider_base_url=str(args.provider_base_url),
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
            "provider_transport_options": {
                "planner_strict_json_schema": planner_strict_json_schema,
            },
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
    # The rule/analysis suites reach the same false green as the JSONL path:
    # scoring a run that stopped mid-plan still returns a payload, and an
    # unconditional exit 0 here reports a partially executed item as a passing
    # benchmark. Both entry points must answer the same question.
    incomplete = _incomplete_suite_items(all_runs)
    if incomplete:
        print(
            "[execution] items that did not complete execution: "
            + ", ".join(incomplete)
        )
        return _EXECUTION_INCOMPLETE_EXIT_CODE
    return 0


def _incomplete_suite_items(all_runs: Sequence[Mapping[str, Any]]) -> List[str]:
    """Name every scored suite item whose run never finished executing."""

    incomplete: List[str] = []
    for payload in all_runs:
        if not isinstance(payload, Mapping):
            continue
        model = str(payload.get("model") or "")
        for score in payload.get("scores") or []:
            if not _score_execution_failures(score):
                continue
            key = str(
                (score.get("item_key") if isinstance(score, Mapping) else None) or "?"
            )
            incomplete.append(f"{key} ({model})" if model else key)
    return incomplete


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
    target: Optional[str],
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
        if operational_source is not None:
            raise ValueError(
                "declared operational exposure must be an exact sealed cohort "
                f"column before Provider launch: task={key!r}, "
                f"field={operational_source!r}, value={operational_exposure!r}. "
                "Keep a conceptual/scoring label in primary_predictor and put "
                "the executable raw column in operational_exposure."
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
        target_outcome=(str(target) if target is not None else None),
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
        id_columns=_external_string_list(row, "id_columns", diagnostics),
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
        scientific_acceptance_contract=(
            dict(row.get("scientific_acceptance_contract") or {})
            if isinstance(row.get("scientific_acceptance_contract"), Mapping)
            else None
        ),
        case_scientific_protocol=(
            dict(row.get("case_scientific_protocol") or {})
            if isinstance(row.get("case_scientific_protocol"), Mapping)
            else None
        ),
        case_scientific_protocol_sha256=(
            str(row.get("case_scientific_protocol_sha256") or "").strip() or None
        ),
        runtime_scientific_projection=(
            dict(row.get("runtime_scientific_projection") or {})
            if isinstance(row.get("runtime_scientific_projection"), Mapping)
            else None
        ),
        runtime_scientific_projection_sha256=(
            str(row.get("runtime_scientific_projection_sha256") or "").strip() or None
        ),
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


def _cohort_shape_without_materialization(path: Path) -> tuple[int, List[str]]:
    """Read cohort row/column metadata without loading the full table.

    Parquet exposes exact shape in its footer.  Delimited files need one
    bounded pass for the row count, but only the first column is ever held in
    memory.  The pipeline remains the sole owner of full cohort loading.
    """

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            metadata = parquet.metadata
            return int(metadata.num_rows), [
                str(name) for name in parquet.schema_arrow.names
            ]
        except Exception as exc:  # noqa: BLE001 - normalize the I/O boundary
            raise _CohortMetadataError(
                f"Parquet footer inspection failed for {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    if suffix in {".csv", ".tsv"}:
        import pandas as pd

        separator = "\t" if suffix == ".tsv" else ","
        try:
            header = pd.read_csv(path, sep=separator, nrows=0)
            columns = [str(column) for column in header.columns]
            if not columns:
                raise ValueError("cohort has no declared columns")
            row_count = sum(
                len(chunk)
                for chunk in pd.read_csv(
                    path,
                    sep=separator,
                    usecols=[0],
                    chunksize=100_000,
                )
            )
            return int(row_count), columns
        except Exception as exc:  # noqa: BLE001 - normalize the I/O boundary
            raise _CohortMetadataError(
                f"Delimited cohort inspection failed for {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    raise _CohortMetadataError(f"Unsupported cohort format: {path.suffix or '<none>'}")


def _safe_benchmark_structured_attempts(exc: BaseException) -> List[Dict[str, Any]]:
    """Project retry diagnostics without response text or parser messages."""

    try:
        from easyicu.research_agent.providers.structured_retry import (
            safe_structured_attempt_metadata,
        )

        projected = getattr(exc, "easyicu_structured_attempt_metadata", None)
        if projected is None:
            projected = getattr(exc, "attempts", None)
        if projected is not None:
            return safe_structured_attempt_metadata(projected)
    except Exception:
        pass
    return []


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
    allow_pending: bool = False,
    require_figure2_paper_acceptance: bool = False,
    expected_execution_identity_path: Path | None = None,
    batch_binding: Optional[Any] = None,
    reasoning_effort_profile: str = "provider_default",
    transport_max_attempts: int = 1,
    stream_enabled: bool = False,
    planner_strict_json_schema: bool = False,
    provider_environment: Optional[Mapping[str, str]] = None,
    provider_base_url: Optional[str] = None,
    items: Optional[Sequence[str]] = None,
) -> int:
    """Run an external EHRFlowBench-style JSONL export when available."""
    pipeline_options = _bind_benchmark_cost_price_table(
        pipeline_options,
        provider=provider,
        model=model,
    )
    # An authorized real-run batch carries its per-task frozen input map + identity.
    frozen_input_authority_by_task: Optional[Mapping[str, str]] = (
        batch_binding.frozen_input_by_task if batch_binding is not None else None
    )
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
    if batch_binding is not None:
        # The gate atomically created this batch root and its immutable receipt
        # before any data/Provider/runner work.  Re-check it instead of creating or
        # overwriting a receipt here.
        from benchmarks.figure2_canonical9.realrun_authority import (
            verify_batch_authorization_receipt,
        )

        if batch_binding.batch_root != Path(out_root).expanduser().resolve():
            raise ValueError("EHRFlow batch root differs from the authorized binding")
        verify_batch_authorization_receipt(batch_binding)
    else:
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
    if items:
        # ``--items`` is applied on the built-in bench path but was never
        # threaded here, so it read as accepted and did nothing.  On
        # 2026-07-30 a launch that asked for one canary task started all nine
        # against a real provider; it was noticed only because the run folders
        # named tasks nobody had asked for.  A selector that silently widens
        # its own scope is worse than no selector, so an unknown key is fatal
        # rather than an empty selection.
        wanted = {str(value).strip() for value in items if str(value).strip()}
        available = {
            str(row.get("key") or row.get("id") or f"ehrflowbench_{idx:03d}")
            for idx, row in enumerate(rows)
        }
        unknown = sorted(wanted - available)
        if unknown:
            raise SystemExit(
                "--items names no row in this EHRFlowBench JSONL: "
                + ", ".join(unknown)
                + ". Available: "
                + ", ".join(sorted(available))
            )
        kept: List[Dict[str, Any]] = []
        kept_invalid: set[int] = set()
        for index, row in enumerate(rows):
            key = str(row.get("key") or row.get("id") or f"ehrflowbench_{index:03d}")
            if key not in wanted:
                continue
            if index in invalid_row_indices:
                kept_invalid.add(len(kept))
            kept.append(row)
        rows = kept
        invalid_row_indices = kept_invalid
        print(
            f"[items] running {len(rows)} of {len(available)} rows: "
            + ", ".join(sorted(wanted)),
            flush=True,
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
    progressive_resume_path = (pipeline_options or {}).get(
        "development_progressive_resume_checkpoint_path"
    )
    if progressive_resume_path is not None and (
        len(input_task_ids) != 1 or len(_normalize_arms(arms)) != 1
    ):
        raise SystemExit(
            "Development Progressive Planner checkpoint resume requires exactly "
            "one selected JSONL item and one arm."
        )
    hard_stop_limits = _provider_hard_stop_limits(pipeline_options or {})
    if batch_binding is not None and hard_stop_limits is None:
        raise ValueError(
            "A formal Canonical9 batch requires frozen run/batch Provider ceilings"
        )
    hard_stop_ledger = None
    if hard_stop_limits is not None:
        from easyicu.research_agent.authority.provider_hard_stop import (
            ProviderHardStopLedger,
        )

        hard_stop_ledger = ProviderHardStopLedger(
            path=(
                out_root
                / (
                    "figure2_batch_progress.json"
                    if batch_binding is not None
                    else "ehrflowbench_progress.json"
                )
            ).resolve(),
            task_ids=input_task_ids,
            limits=hard_stop_limits,
            batch_id=(
                str(batch_binding.batch_id) if batch_binding is not None else None
            ),
            declaration_sha256=(
                str(batch_binding.declaration_sha256)
                if batch_binding is not None
                else None
            ),
            resume_existing=bool(
                (reuse_existing or resume_run_id) and batch_binding is None
            ),
        )
    formal_canary_task_id: Optional[str] = None
    if batch_binding is not None:
        from benchmarks.figure2_canonical9.evaluator.rubric_v1 import (
            FIGURE2_TASK_IDS,
        )

        formal_canary_task_id = str(FIGURE2_TASK_IDS[0])
        if not input_task_ids or input_task_ids[0] != formal_canary_task_id:
            raise ValueError(
                "A formal Canonical9 batch must start with its locked E1 canary."
            )
    task_hard_stops: Dict[str, Any] = {}

    def _sync_pending_hard_stops() -> None:
        if hard_stop_ledger is None:
            return
        statuses = {
            str(task.get("task_id")): str(task.get("status"))
            for task in hard_stop_ledger.snapshot().get("tasks", [])
            if isinstance(task, Mapping)
        }
        for entry in pending:
            pending_key = str(entry.get("key") or "")
            if statuses.get(pending_key) != "running":
                continue
            handle = task_hard_stops.get(pending_key)
            if handle is not None:
                handle.finish(error=str(entry.get("status") or "pending"))

    for idx, row in enumerate(rows):
        _sync_pending_hard_stops()
        key = str(row.get("key") or row.get("id") or f"ehrflowbench_{idx:03d}")
        task_hard_stop = (
            hard_stop_ledger.start_task(
                key,
                reopen_terminal=bool(resume_run_id and batch_binding is None),
            )
            if hard_stop_ledger is not None
            else None
        )
        if task_hard_stop is not None:
            task_hard_stops[key] = task_hard_stop
        if idx in invalid_row_indices:
            pending.append({"key": key, **row})
            continue
        cohort_path = row.get("cohort_path") or row.get("cohort")
        question = row.get("question") or row.get("research_question")
        target = row.get("target_outcome") or row.get("outcome")
        task_kind = str(row.get("kind") or "descriptive_association").strip()
        outcome_optional = task_kind in {
            "longitudinal_trajectory_analysis",
            "subphenotype_clustering",
        }
        if not cohort_path or not question or (not target and not outcome_optional):
            pending.append(
                {
                    "key": key,
                    "status": "pending_missing_fields",
                    "required": (
                        ["question", "cohort_path"]
                        if outcome_optional
                        else ["question", "cohort_path", "target_outcome"]
                    ),
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
        if path.suffix.lower() not in {".parquet", ".pq", ".csv", ".tsv"}:
            pending.append(
                {
                    "key": key,
                    "status": "unsupported_cohort_format",
                    "cohort_path": str(path),
                }
            )
            continue
        try:
            cohort_size, cohort_columns = _cohort_shape_without_materialization(path)
        except _CohortMetadataError as exc:
            pending.append(
                {
                    "key": key,
                    "status": "cohort_metadata_unreadable",
                    "cohort_path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
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
            target=(str(target) if target else None),
            cohort_size=cohort_size,
            cohort_columns=cohort_columns,
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
        row_pipeline_options = dict(pipeline_options or {})
        if frozen_input_authority_by_task and key in frozen_input_authority_by_task:
            # Bind THIS task's frozen input digest so _bind_benchmark_execution_input
            # fails closed if the runtime cohort differs from the authorized input.
            row_pipeline_options["execution_input_authority_sha256"] = (
                frozen_input_authority_by_task[key]
            )
        # Exact reuse is decided inside ``_run_one_item_from_cohort`` after the
        # current ExecutionIdentity is constructed. A broad "execution complete"
        # shortcut here would bypass profile/image/provider/prompt/git matching.
        # Per-item isolation: a provider 502 / crash on one item must not abort
        # the remaining items. Record the failure and continue.
        try:
            score = _run_one_item_from_cohort(
                item=item,
                # Preserve the source path even for a legacy/untyped export.
                # The pipeline can then verify and stage the adjacent
                # materialization provenance instead of losing it through an
                # eager DataFrame handoff (which also duplicates the full table
                # in memory before post-QC development sampling).
                cohort=path,
                seed=seed,
                out_root=out_root,
                arms=arms,
                pipeline_options=row_pipeline_options,
                provider=provider,
                model=model,
                request_timeout=request_timeout,
                reuse_existing=reuse_existing,
                resume_run_id=resume_run_id,
                resume_from_step_id=resume_from_step_id,
                stop_after_step_id=stop_after_step_id,
                force_writer_probe=force_writer_probe,
                reasoning_effort_profile=reasoning_effort_profile,
                transport_max_attempts=transport_max_attempts,
                stream_enabled=stream_enabled,
                planner_strict_json_schema=planner_strict_json_schema,
                provider_environment=provider_environment,
                provider_hard_stop=task_hard_stop,
            )
            if batch_binding is not None:
                _ensure_formal_figure2_safety_and_rescore(
                    score=score,
                    item=item,
                    provider_environment=provider_environment,
                    request_timeout=request_timeout,
                )
            scores.append(score)
            # An execution failure is NOT recorded in ``pending``.  ``pending``
            # means one thing -- the item never entered the pipeline -- and the
            # exit code says so out loud (5 vs 4), the printed line says so, and
            # the offered waiver (``--allow-pending-import``) is an *import*
            # waiver.  This item reached ``_run_one_item_from_cohort`` and came
            # back with a score, so filing it here made the run report "never
            # entered the pipeline" for something that ran, return 5 where 4 is
            # true, count it as both runnable and pending, and tell the operator
            # to accept an execution failure with an import flag.  The failure
            # is already reported from ``scores`` through the same predicate
            # (``incomplete``), which is where it belongs; this append was the
            # earlier way of exiting non-zero and outlived the split.
            if task_hard_stop is not None:
                _finish_task_on_execution_outcome(task_hard_stop, score)
            if formal_canary_task_id is not None and key == formal_canary_task_id:
                canary_passed = _figure2_canary_passed(score)
                _write_figure2_canary_gate(
                    out_root=out_root,
                    task_id=key,
                    score=score,
                    status="passed" if canary_passed else "blocked",
                    reason=(
                        "publication, manuscript, zero-error, and locked paper "
                        "scorecard gates passed"
                        if canary_passed
                        else "formal canary did not clear every paper-facing gate"
                    ),
                )
                if not canary_passed:
                    pending.extend(
                        {
                            "key": later_key,
                            "status": "batch_canary_blocked",
                            "blocked_by": key,
                        }
                        for later_key in input_task_ids[idx + 1 :]
                    )
                    if hard_stop_ledger is not None:
                        for later_key in input_task_ids[idx + 1 :]:
                            hard_stop_ledger.mark_task_blocked(
                                later_key,
                                blocked_by=key,
                            )
                    break
        except Exception as exc:  # noqa: BLE001 — keep batch alive on 502/etc.
            import traceback as _tb

            tb = _tb.format_exc()
            structured_attempts = _safe_benchmark_structured_attempts(exc)
            print(
                f"[ehrflowbench] item {key} FAILED: {type(exc).__name__}: "
                f"{str(exc)[:200]}\n{tb}"
            )
            try:
                (out_root / key).mkdir(parents=True, exist_ok=True)
                (out_root / key / "item_exception_traceback.txt").write_text(
                    tb, encoding="utf-8"
                )
                diagnostic = {
                    "schema_version": "easyicu.benchmark_item_exception/1",
                    "task_id": key,
                    "error_type": type(exc).__name__,
                    "message_sha256": hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                    "structured_attempts": structured_attempts,
                }
                (out_root / key / "item_exception.json").write_text(
                    json.dumps(
                        diagnostic,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
            pending_row: Dict[str, Any] = {
                "key": key,
                "status": "item_exception",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
            if structured_attempts:
                pending_row["structured_attempts"] = structured_attempts
            pending.append(pending_row)
            if task_hard_stop is not None:
                task_hard_stop.finish(error=f"{type(exc).__name__}: {str(exc)[:1800]}")
            if formal_canary_task_id is not None and key == formal_canary_task_id:
                _write_figure2_canary_gate(
                    out_root=out_root,
                    task_id=key,
                    score=None,
                    status="blocked",
                    reason=f"canary raised {type(exc).__name__}",
                )
                pending.extend(
                    {
                        "key": later_key,
                        "status": "batch_canary_blocked",
                        "blocked_by": key,
                    }
                    for later_key in input_task_ids[idx + 1 :]
                )
                if hard_stop_ledger is not None:
                    for later_key in input_task_ids[idx + 1 :]:
                        hard_stop_ledger.mark_task_blocked(
                            later_key,
                            blocked_by=key,
                        )
                break
            continue

    _sync_pending_hard_stops()
    totals = _aggregate(scores) if scores else {"naive": {}, "aware": {}}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(jsonl_path),
        "seed": seed,
        "arms": _normalize_arms(arms),
        "reasoning_effort_profile": reasoning_effort_profile,
        "provider_transport_options": {
            "planner_strict_json_schema": bool(planner_strict_json_schema),
        },
        "backend_base_url": (
            str(provider_base_url)
            if provider_base_url is not None
            else _resolve_backend_base_url(provider)
        ),
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
    batch_authority_issues: list[tuple[str, str, str | None]] = []
    if batch_binding is not None:
        # Verify the batch-to-child authority before publishing the
        # paper-facing acceptance artifact.  The previous order could write
        # `accepted` and only then discover that the batch ledger was
        # incomplete, leaving a contradictory terminal file on disk.
        from benchmarks.figure2_canonical9.realrun_authority import (
            build_batch_ledger,
            verify_results_frozen_input_authority,
            write_batch_ledger,
        )

        try:
            ledger = build_batch_ledger(payload, out_root, batch_binding)
            ledger_path = write_batch_ledger(ledger, out_root)
            print(f"  -> {ledger_path}")
            input_authority_mismatches = verify_results_frozen_input_authority(
                payload, batch_binding.frozen_input_by_task
            )
            for task_id, reason in input_authority_mismatches:
                print(
                    "[realrun-authority] POST-RUN input authority mismatch for "
                    f"{task_id}: {reason}"
                )
                batch_authority_issues.append(
                    (
                        "BATCH_INPUT_AUTHORITY_INVALID",
                        str(reason)[:2048],
                        str(task_id),
                    )
                )
            if not ledger.get("complete"):
                detail = (
                    "not every Canonical9 child run mapped back to the "
                    "authorized batch declaration"
                )
                print(
                    f"[realrun-authority] POST-RUN batch ledger incomplete: {detail}."
                )
                batch_authority_issues.append(("BATCH_LEDGER_INVALID", detail, None))
        except Exception as exc:  # fail closed into the terminal receipt
            detail = f"{type(exc).__name__}: {exc}"[:2048]
            print(
                "[realrun-authority] POST-RUN batch ledger verification "
                f"failed: {detail}"
            )
            batch_authority_issues.append(("BATCH_LEDGER_INVALID", detail, None))

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
            acceptance = evaluate_figure2_paper_acceptance(
                results_path,
                expected_execution_identity_path=expected_execution_identity_path,
            )
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
        if batch_authority_issues:
            acceptance = acceptance.model_copy(
                update={
                    "status": "invalid",
                    "issues": acceptance.issues
                    + tuple(
                        Figure2AcceptanceIssue(
                            code=code,
                            detail=detail,
                            task_id=task_id,
                        )
                        for code, detail, task_id in batch_authority_issues
                    ),
                }
            )
            # `model_copy` is intentionally cheap and does not revalidate.
            # Round-trip once so the terminal artifact obeys the same strict
            # contract as an ordinary acceptance evaluation.
            acceptance = Figure2PaperAcceptance.model_validate_json(
                acceptance.model_dump_json(),
                strict=True,
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
    if batch_authority_issues:
        return 2
    if require_figure2_paper_acceptance and acceptance_status != "accepted":
        return _FIGURE2_PAPER_ACCEPTANCE_EXIT_CODE
    # A scored item whose run never finished executing is not a success, even in
    # a development diagnostic where paper authority is expected to be withheld.
    # Reporting exit 0 here is what let a 7/12-step run read as a completed task.
    incomplete = [
        str(score.get("item_key") or "")
        for score in scores
        if _score_execution_failures(score)
    ]
    # An item that never started is the same failure one stage earlier, and it
    # was the quieter of the two: it produces no score to inspect, so nothing
    # downstream could notice it. Report both lists before returning, so a run
    # with both problems does not hide one behind the other's exit code.
    if pending:
        print(
            "[pending] items that never entered the pipeline: "
            + ", ".join(
                f"{entry.get('key')} ({entry.get('status')})" for entry in pending
            )
        )
    if incomplete:
        print(
            "[execution] items that did not complete execution: "
            + ", ".join(incomplete)
        )
    if pending and not allow_pending:
        print(
            "[pending] pass --allow-pending-import to accept an import whose "
            "items are knowingly not runnable."
        )
        return _PENDING_ITEMS_EXIT_CODE
    if incomplete:
        return _EXECUTION_INCOMPLETE_EXIT_CODE
    return 0


def _run_one_item_from_cohort(
    *,
    item,
    cohort,
    seed: int | None = None,
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
    reasoning_effort_profile: str = "provider_default",
    transport_max_attempts: int = 1,
    stream_enabled: bool = False,
    planner_strict_json_schema: bool = False,
    provider_environment: Optional[Mapping[str, str]] = None,
    provider_hard_stop: Optional[Any] = None,
) -> Dict[str, Any]:
    item_root = out_root / item.key
    selected = set(_normalize_arms(arms))
    pipeline_options = _bind_benchmark_cost_price_table(
        pipeline_options,
        provider=provider,
        model=model,
    )
    bound_pipeline_options = _bind_benchmark_execution_input(
        pipeline_options,
        cohort=cohort,
        data_seed=seed,
    )
    naive = _skipped_arm("naive")
    aware = _skipped_arm("aware")
    account_session_client = None
    if (
        str(provider or "").strip().lower() == "codex"
        and str(
            (provider_environment or {}).get("EASYICU_CODEX_SESSION_SHA256") or ""
        ).strip()
        and set(_normalize_arms(arms)).intersection({"naive", "aware"})
    ):
        # Configuration-only identity cannot safely guess a browser-managed
        # session's endpoint, hard timeout, or reasoning effort. Constructing
        # this client performs no Provider call; it lets both reuse checks and
        # the eventual Pipeline derive identity from the exact same live object.
        account_session_client = _make_llm(
            provider=provider,
            model=model,
            request_timeout=request_timeout,
            reasoning_effort_profile=reasoning_effort_profile,
            transport_max_attempts=transport_max_attempts,
            stream_enabled=stream_enabled,
            planner_strict_json_schema=planner_strict_json_schema,
            provider_environment=provider_environment,
        )
    expected_identity = _benchmark_execution_identity(
        bound_pipeline_options,
        account_session_client,
        provider=provider,
        model=model,
        reasoning_effort_profile=reasoning_effort_profile,
        request_timeout=request_timeout,
        transport_max_attempts=transport_max_attempts,
        stream_enabled=stream_enabled,
        planner_strict_json_schema=planner_strict_json_schema,
        provider_environment=provider_environment,
    )
    if reuse_existing and not resume_run_id:
        if "naive" in selected:
            naive = _reuse_arm_if_complete(
                arm_dir=item_root / "naive",
                item=item,
                label="naive",
                expected_execution_identity_sha256=expected_identity.identity_sha256,
            ) or _skipped_arm("naive")
        if "aware" in selected:
            aware = _reuse_arm_if_complete(
                arm_dir=item_root / "aware",
                item=item,
                label="aware",
                expected_execution_identity_sha256=expected_identity.identity_sha256,
            ) or _skipped_arm("aware")
    run_naive = "naive" in selected and not _arm_was_run(naive)
    run_aware = "aware" in selected and not _arm_was_run(aware)
    llm = account_session_client
    if llm is None and (run_naive or run_aware):
        llm = _make_llm(
            provider=provider,
            model=model,
            request_timeout=request_timeout,
            reasoning_effort_profile=reasoning_effort_profile,
            transport_max_attempts=transport_max_attempts,
            stream_enabled=stream_enabled,
            planner_strict_json_schema=planner_strict_json_schema,
            provider_environment=provider_environment,
        )
    if run_naive:
        naive = _run_one_arm(
            item=item,
            cohort=(cohort if isinstance(cohort, (str, Path)) else cohort.copy()),
            workdir=item_root / "naive",
            disable_icu_context=True,
            label="naive",
            llm=llm,
            pipeline_options=bound_pipeline_options,
            reuse_existing=reuse_existing,
            resume_run_id=resume_run_id,
            resume_from_step_id=resume_from_step_id,
            stop_after_step_id=stop_after_step_id,
            force_writer_probe=force_writer_probe,
            provider_hard_stop=provider_hard_stop,
        )
    if run_aware:
        aware = _run_one_arm(
            item=item,
            cohort=(cohort if isinstance(cohort, (str, Path)) else cohort.copy()),
            workdir=item_root / "aware",
            disable_icu_context=False,
            label="aware",
            llm=llm,
            pipeline_options=bound_pipeline_options,
            reuse_existing=reuse_existing,
            resume_run_id=resume_run_id,
            resume_from_step_id=resume_from_step_id,
            stop_after_step_id=stop_after_step_id,
            force_writer_probe=force_writer_probe,
            provider_hard_stop=provider_hard_stop,
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
        "reasoning_effort_profile": reasoning_effort_profile,
        "provider_transport_options": {
            "planner_strict_json_schema": bool(planner_strict_json_schema),
        },
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

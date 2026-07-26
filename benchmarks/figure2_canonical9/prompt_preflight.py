"""Zero-Provider prompt-envelope audit for the frozen Canonical9 inputs.

The Planner request can be rendered exactly before a run.  Coder, Analyzer,
Writer, and Repair requests depend on model-authored plans or produced evidence,
so this audit renders their exact production builders with a conservative,
case-bound representative step/summary.  The distinction is persisted in the
report and must not be confused with a formal scientific run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from easyicu.research_agent.agents.core import (
    AnalyzerAgent,
    CoderAgent,
    PlannerAgent,
    WriterAgent,
)
from easyicu.research_agent.authority.coder_authority import HostCoderAuthority
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
    load_verified_materialized_cohort_authority,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    MaterializedTrajectoryAuthorityRef,
    StagedTrajectoryBinding,
    load_verified_materialized_trajectory_authority,
)
from easyicu.research_agent.planning.analysis_types import infer_analysis_type
from easyicu.research_agent.planning.analysis_blueprint import (
    build_analysis_blueprint,
    render_analysis_blueprint_for_prompt,
)
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient
from easyicu.research_agent.repairs.reasons import RepairPromptAuthority
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import (
    AnalysisStep,
    ResearchContext,
    ValidationFinding,
)

from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS
from benchmarks.figure2_canonical9.protocol_prompt import (
    render_task_protocol_note,
    render_task_protocol_preferences,
)

PROMPT_PREFLIGHT_SCHEMA_VERSION = "easyicu.canonical9_prompt_preflight/1"

_PLANNER_LIMIT_BYTES = 80_000
_CODER_LIMIT_BYTES = 42_000
_ANALYZER_LIMIT_BYTES = 48_000
_WRITER_LIMIT_BYTES = 64_000
_REPAIR_LIMIT_BYTES = 30_000
_PROMPT_KINDS = ("planner", "coder", "analyzer", "writer", "repair")


class PromptPreflightError(RuntimeError):
    """A Canonical9 prompt cannot be rendered losslessly and offline."""


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PromptPreflightError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise PromptPreflightError(f"non-finite JSON constant {value!r}")


def _strict_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise PromptPreflightError(
                f"invalid JSONL row {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise PromptPreflightError(f"JSONL row {line_number} is not an object")
        rows.append(row)
    observed = tuple(str(row.get("key") or "") for row in rows)
    expected = tuple(FIGURE2_TASK_IDS)
    if observed != expected:
        raise PromptPreflightError(
            f"Canonical9 task order mismatch: observed={observed!r}, expected={expected!r}"
        )
    return rows


def _regular_absolute_file(value: object, *, label: str) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise PromptPreflightError(f"{label} must be absolute and non-symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PromptPreflightError(f"{label} is not readable: {path}") from exc
    if not resolved.is_file():
        raise PromptPreflightError(f"{label} is not a regular file: {resolved}")
    return resolved


def _authority_ref(
    row: Mapping[str, Any],
) -> tuple[
    Path,
    MaterializedCohortAuthorityRef,
    StagedTrajectoryBinding | None,
]:
    cohort_path = _regular_absolute_file(row.get("cohort_path"), label="cohort_path")
    raw_cohort_ref = row.get("cohort_authority_ref")
    if not isinstance(raw_cohort_ref, Mapping):
        raise PromptPreflightError("cohort_authority_ref is missing")
    cohort_ref = MaterializedCohortAuthorityRef.from_dict(raw_cohort_ref)
    selected_cohort_authority = _regular_absolute_file(
        row.get("cohort_authority_path"),
        label="cohort_authority_path",
    )
    if selected_cohort_authority != cohort_path.parent / cohort_ref.file:
        raise PromptPreflightError(
            "cohort_authority_path does not match the content-addressed reference"
        )
    verified_cohort = load_verified_materialized_cohort_authority(
        cohort_path,
        expected_authority=cohort_ref,
    )
    if verified_cohort is None:
        raise PromptPreflightError("typed cohort authority verification returned None")

    raw_trajectory = row.get("trajectory_path")
    if not raw_trajectory:
        if row.get("trajectory_authority_ref") or row.get("trajectory_authority_path"):
            raise PromptPreflightError(
                "trajectory authority was declared without a trajectory artifact"
            )
        return cohort_path, cohort_ref, None

    trajectory_path = _regular_absolute_file(raw_trajectory, label="trajectory_path")
    raw_trajectory_ref = row.get("trajectory_authority_ref")
    if not isinstance(raw_trajectory_ref, Mapping):
        raise PromptPreflightError("trajectory_authority_ref is missing")
    trajectory_ref = MaterializedTrajectoryAuthorityRef.from_dict(raw_trajectory_ref)
    selected_trajectory_authority = _regular_absolute_file(
        row.get("trajectory_authority_path"),
        label="trajectory_authority_path",
    )
    if selected_trajectory_authority != trajectory_path.parent / trajectory_ref.file:
        raise PromptPreflightError(
            "trajectory_authority_path does not match the content-addressed reference"
        )
    verified_trajectory = load_verified_materialized_trajectory_authority(
        trajectory_path,
        expected_authority=trajectory_ref,
        expected_universe_authority=cohort_ref,
    )
    if verified_trajectory is None:
        raise PromptPreflightError(
            "typed trajectory authority verification returned None"
        )
    binding = StagedTrajectoryBinding(
        path=trajectory_path,
        sha256=verified_trajectory.authority.trajectory_sha256,
        size=verified_trajectory.authority.trajectory_size,
        authority_ref=trajectory_ref,
    )
    return cohort_path, cohort_ref, binding


def _canary(task_id: str, section: str) -> str:
    digest = hashlib.sha256(f"{task_id}:{section}".encode()).hexdigest()[:16]
    return f"__EASYICU_{task_id.upper()}_{section.upper()}_{digest}__"


def _string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [text for item in value if (text := str(item or "").strip())]


def _build_context(
    row: Mapping[str, Any],
    *,
    canaries: Mapping[str, str],
) -> tuple[
    ResearchContext, MaterializedCohortAuthorityRef, StagedTrajectoryBinding | None
]:
    cohort_path, cohort_ref, trajectory_binding = _authority_ref(row)
    task_id = str(row["key"])
    protocol_note = render_task_protocol_note(
        task_id=task_id,
        task_kind=str(row.get("kind") or ""),
        task_notes=(str(row.get("notes") or "").strip() or None),
        required_outputs=_string_list(row.get("expected_outputs")),
        semantic_guardrails=_string_list(row.get("semantic_guardrails")),
        canary_tokens=canaries,
    )
    protocol_preferences = render_task_protocol_preferences(
        task_id=task_id,
        task_kind=str(row.get("kind") or ""),
        task_notes=(str(row.get("notes") or "").strip() or None),
        required_outputs=_string_list(row.get("expected_outputs")),
        semantic_guardrails=_string_list(row.get("semantic_guardrails")),
        canary_tokens=canaries,
    )
    operational_exposure = str(
        row.get("operational_exposure") or row.get("primary_predictor") or ""
    ).strip()
    primary_predictor = str(row.get("primary_predictor") or "").strip()
    question = str(row.get("question") or row.get("research_question") or "").strip()
    normalized_question = re.sub(r"[^a-z0-9]+", "_", question.casefold()).strip("_")
    normalized_predictor = re.sub(
        r"[^a-z0-9]+", "_", primary_predictor.casefold()
    ).strip("_")
    concept_descriptions = (
        {operational_exposure: primary_predictor}
        if operational_exposure
        and primary_predictor
        and re.search(
            rf"(?:^|_){re.escape(normalized_predictor)}(?:_|$)",
            normalized_question,
        )
        else None
    )
    context = build_research_context(
        research_question=question,
        cohort=cohort_path,
        cohort_name=f"bench_{task_id}",
        database=str(row.get("database") or "miiv"),
        target_outcome=(
            str(row.get("target_outcome")).strip()
            if row.get("target_outcome")
            else None
        ),
        primary_exposure=operational_exposure or None,
        inclusion_criteria=_string_list(row.get("inclusion_criteria")),
        id_columns=_string_list(row.get("id_columns")) or None,
        concept_descriptions=concept_descriptions,
        user_preferences=protocol_preferences,
        notes=protocol_note,
        trajectory_binding=trajectory_binding,
    )
    executable_columns = set(
        getattr(
            getattr(context.materialized_inputs, "cohort", None),
            "column_bindings",
            {},
        )
    )
    if operational_exposure and operational_exposure not in executable_columns:
        raise PromptPreflightError(
            "declared operational exposure must be an exact sealed cohort "
            f"column: task={task_id!r}, value={operational_exposure!r}. "
            "Keep the conceptual label in primary_predictor and bind "
            "operational_exposure to the executable raw column."
        )
    return context, cohort_ref, trajectory_binding


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


def _representative_step(
    row: Mapping[str, Any],
    *,
    context: ResearchContext,
) -> AnalysisStep:
    task_id = str(row["key"])
    kind = str(row.get("kind") or "").strip()
    method_by_kind = {
        "mortality_prediction": "prediction_model_analysis",
        "survival_analysis": "cox_proportional_hazards",
        "causal_inference": "causal_inference_ipw",
        "subphenotype_clustering": "trajectory_clustering_analysis",
        "longitudinal_trajectory_analysis": "trajectory_clustering_analysis",
        "missingness_robustness": "missingness_robustness_analysis",
        "ordinal_dose_response": "ordinal_dose_response_analysis",
        "sepsis_onset": "adjusted_association_models",
        "descriptive_association": "adjusted_association_models",
    }
    representative_output_by_kind = {
        "mortality_prediction": "table:model_performance",
        "survival_analysis": "table:cox_summary",
        "causal_inference": "table:adjusted_effect_estimates",
        "subphenotype_clustering": "table:cluster_profiles",
        "longitudinal_trajectory_analysis": "table:trajectory_cluster_profiles",
        "missingness_robustness": "table:robustness_summary",
        "ordinal_dose_response": "table:ordinal_trend",
        "sepsis_onset": "table:adjusted_association_estimates",
        "descriptive_association": "table:adjusted_association_estimates",
    }
    requested = _string_list(row.get("candidate_variables"))
    available = {variable.name for variable in context.variables}
    inputs: list[str] = []
    for value in (
        row.get("operational_exposure"),
        row.get("target_outcome"),
        *requested,
    ):
        token = str(value or "").strip()
        if token and token in available and token not in inputs:
            inputs.append(token)
    if len(inputs) < 2:
        for variable in context.variables:
            if variable.name in inputs or variable.role.value in {"id", "time"}:
                continue
            inputs.append(variable.name)
            if len(inputs) >= 12:
                break
    output = representative_output_by_kind.get(
        kind,
        "table:prompt_preflight_result",
    )
    return AnalysisStep(
        step_id=f"prompt_preflight_{_slug(task_id)}",
        planned_analysis_role="primary",
        intent=(
            "Offline prompt-envelope audit for the exact case-scoped task "
            "protocol. Do not execute or treat this representative step as a "
            "scientific plan."
        ),
        inputs=inputs,
        expected_outputs=[output],
        method=method_by_kind.get(kind, infer_analysis_type(context).key),
    )


def _message_payload(messages: Iterable[object]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(getattr(message, "role", "") or "")
        content = str(getattr(message, "content", "") or "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _serialize_messages(messages: Iterable[object]) -> list[dict[str, str]]:
    return [
        {
            "role": str(getattr(message, "role", "") or ""),
            "content": str(getattr(message, "content", "") or ""),
        }
        for message in messages
    ]


def _capture_coder_messages(
    *,
    context: ResearchContext,
    step: AnalysisStep,
    host_authority: HostCoderAuthority,
) -> list[object]:
    client = PatternScriptedMockLLMClient(
        [],
        default="value = 1\nprint(value)\n",
    )
    CoderAgent(client).run(
        context=context,
        step=step,
        host_authority=host_authority,
    )
    if len(client.calls) != 1:
        raise PromptPreflightError(
            f"Coder prompt render made {len(client.calls)} mock calls, expected 1"
        )
    return list(client.calls[0][0])


def _capture_analyzer_messages(
    *,
    context: ResearchContext,
    step: AnalysisStep,
    task_id: str,
) -> list[object]:
    client = PatternScriptedMockLLMClient([], default="Offline interpretation.")
    AnalyzerAgent(client).run(
        context=context,
        step=step,
        step_summary={
            "status": "ok",
            "diagnostic_only": True,
            "task_id": task_id,
            "output_files": {
                "table:prompt_preflight_result": "prompt_preflight_result.csv"
            },
        },
        evidence_ids=["prompt_preflight_result"],
    )
    if len(client.calls) != 1:
        raise PromptPreflightError(
            f"Analyzer prompt render made {len(client.calls)} mock calls, expected 1"
        )
    return list(client.calls[0][0])


def _capture_writer_messages(
    *,
    context: ResearchContext,
    task_id: str,
    digest_canary: str,
    required_outputs: Sequence[str],
) -> list[object]:
    client = PatternScriptedMockLLMClient(
        [],
        default="## Results\nOffline prompt preflight only.",
    )
    evidence_digest = json.dumps(
        {
            "schema_version": "easyicu.prompt_preflight_writer_digest/1",
            "task_id": task_id,
            "diagnostic_only": True,
            "required_outputs": list(required_outputs),
            "tail_canary": digest_canary,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    WriterAgent(client)._call_section(
        section_name="Results",
        instruction=(
            "Render one diagnostic-only Results request. Do not infer or invent "
            "scientific findings."
        ),
        context=context,
        evidence_ids=["prompt_preflight_result"],
        evidence_digest=evidence_digest,
        max_tokens=128,
    )
    if len(client.calls) != 1:
        raise PromptPreflightError(
            f"Writer prompt render made {len(client.calls)} mock calls, expected 1"
        )
    return list(client.calls[0][0])


def _capture_repair_messages(
    *,
    context: ResearchContext,
    step: AnalysisStep,
    host_authority: HostCoderAuthority,
) -> list[object]:
    patch = json.dumps(
        {
            "format": "easyicu.code_patch/1",
            "edits": [
                {
                    "old": "value = 1",
                    "new": "value = 2",
                    "expected_count": 1,
                }
            ],
        }
    )
    client = PatternScriptedMockLLMClient([], default=patch)
    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Offline prompt preflight scientific-semantics repair.",
        detail={"issue_code": "plausibility_range_exclusion_required"},
    )
    authority = RepairPromptAuthority.create(findings=[finding])
    CoderAgent(client).repair(
        context=context,
        step=step,
        host_authority=host_authority,
        repair_authority=authority,
        current_repair_authority=authority,
        code="value = 1\nprint(value)\n",
        run_log="offline prompt preflight diagnostic",
    )
    if len(client.calls) != 1:
        raise PromptPreflightError(
            f"Repair prompt render made {len(client.calls)} mock calls, expected 1"
        )
    return list(client.calls[0][0])


def _check_strings(payload: str, strings: Sequence[str]) -> dict[str, object]:
    missing = [value for value in strings if value not in payload]
    return {"ok": not missing, "missing": missing}


def _prompt_record(
    *,
    prompt_kind: str,
    messages: Sequence[object],
    required_strings: Sequence[str],
    limit_bytes: int | None,
    stage_mode: str,
) -> dict[str, object]:
    payload = _message_payload(messages)
    payload_bytes = len(payload.encode("utf-8"))
    integrity = _check_strings(payload, required_strings)
    within_budget = limit_bytes is None or payload_bytes <= limit_bytes
    return {
        "prompt_kind": prompt_kind,
        "stage_mode": stage_mode,
        "characters": len(payload),
        "bytes": payload_bytes,
        "approx_input_tokens": math.ceil(payload_bytes / 4),
        "limit_bytes": limit_bytes,
        "within_budget": within_budget,
        "silent_truncation_detected": not bool(integrity["ok"]),
        "required_string_check": integrity,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "messages": _serialize_messages(messages),
    }


def render_task_prompt_audit(row: Mapping[str, Any]) -> dict[str, object]:
    """Render all five prompt envelopes for one frozen Canonical9 task."""

    task_id = str(row["key"])
    canaries = {
        section: _canary(task_id, section)
        for section in ("task_notes", "required_outputs", "semantic_guardrails")
    }
    writer_canary = _canary(task_id, "writer_digest")
    trajectory_canary = _canary(task_id, "trajectory_authority")
    context, cohort_ref, trajectory_binding = _build_context(
        row,
        canaries=canaries,
    )
    step = _representative_step(row, context=context)
    trajectory_authority_note = (
        "CANONICAL TRAJECTORY INPUT AUTHORITY (offline prompt audit): "
        f"sha256={trajectory_binding.sha256}; "
        f"size={trajectory_binding.size}; tail_canary={trajectory_canary}"
        if trajectory_binding is not None
        else ""
    )
    host_authority = HostCoderAuthority.from_values((trajectory_authority_note,))

    planning_contract_context = render_analysis_blueprint_for_prompt(
        build_analysis_blueprint(context)
    )
    planner_messages = PlannerAgent.request_messages(
        context,
        planning_contract_context=planning_contract_context,
    )
    planner_metrics = PlannerAgent.request_metrics(
        context,
        planning_contract_context=planning_contract_context,
    )
    coder_messages = _capture_coder_messages(
        context=context,
        step=step,
        host_authority=host_authority,
    )
    analyzer_messages = _capture_analyzer_messages(
        context=context,
        step=step,
        task_id=task_id,
    )
    writer_messages = _capture_writer_messages(
        context=context,
        task_id=task_id,
        digest_canary=writer_canary,
        required_outputs=_string_list(row.get("expected_outputs")),
    )
    repair_messages = _capture_repair_messages(
        context=context,
        step=step,
        host_authority=host_authority,
    )

    protocol_strings = [
        *canaries.values(),
        *_string_list(row.get("expected_outputs")),
        *_string_list(row.get("semantic_guardrails")),
    ]
    notes = str(row.get("notes") or "").strip()
    if notes:
        protocol_strings.append(notes)
    records = {
        "planner": _prompt_record(
            prompt_kind="planner",
            messages=planner_messages,
            required_strings=[*protocol_strings, planning_contract_context],
            limit_bytes=_PLANNER_LIMIT_BYTES,
            stage_mode="exact_pre_execution",
        ),
        "coder": _prompt_record(
            prompt_kind="coder",
            messages=coder_messages,
            required_strings=[
                *protocol_strings,
                *([trajectory_canary] if trajectory_binding is not None else []),
            ],
            limit_bytes=_CODER_LIMIT_BYTES,
            stage_mode="representative_plan_step",
        ),
        "analyzer": _prompt_record(
            prompt_kind="analyzer",
            messages=analyzer_messages,
            required_strings=protocol_strings,
            limit_bytes=_ANALYZER_LIMIT_BYTES,
            stage_mode="representative_step_summary",
        ),
        "writer": _prompt_record(
            prompt_kind="writer",
            messages=writer_messages,
            required_strings=[*protocol_strings, writer_canary],
            limit_bytes=_WRITER_LIMIT_BYTES,
            stage_mode="representative_evidence_digest",
        ),
        "repair": _prompt_record(
            prompt_kind="repair",
            messages=repair_messages,
            required_strings=[
                *protocol_strings,
                *([trajectory_canary] if trajectory_binding is not None else []),
            ],
            limit_bytes=_REPAIR_LIMIT_BYTES,
            stage_mode="representative_scientific_repair",
        ),
    }
    planner_record = records["planner"]
    if int(planner_metrics["total_bytes"]) > _PLANNER_LIMIT_BYTES:
        planner_record["within_budget"] = False
    planner_record["production_request_metrics"] = planner_metrics
    planner_record["planning_contract_bytes"] = len(
        planning_contract_context.encode("utf-8")
    )
    planner_record["planning_contract_sha256"] = hashlib.sha256(
        planning_contract_context.encode("utf-8")
    ).hexdigest()
    prompt_ok = all(
        bool(record["within_budget"]) and not bool(record["silent_truncation_detected"])
        for record in records.values()
    )
    return {
        "task_id": task_id,
        "task_kind": str(row.get("kind") or ""),
        "status": "passed" if prompt_ok else "failed",
        "diagnostic_only": True,
        "provider_calls": 0,
        "cohort_authority_sha256": cohort_ref.sha256,
        "trajectory_authority_sha256": (
            trajectory_binding.authority_ref.sha256
            if trajectory_binding is not None
            and trajectory_binding.authority_ref is not None
            else None
        ),
        "context_schema_version": context.schema_version,
        "context_variable_count": len(context.variables),
        "representative_analysis_type": infer_analysis_type(context).key,
        "representative_step": step.model_dump(mode="json"),
        "canaries": {
            **canaries,
            "writer_digest": writer_canary,
            "trajectory_authority": (
                trajectory_canary if trajectory_binding is not None else None
            ),
        },
        "prompts": records,
    }


def run_canonical9_prompt_preflight(
    *,
    jsonl_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Render, validate, and persist every Canonical9 prompt envelope."""

    source = _regular_absolute_file(jsonl_path, label="canonical JSONL")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _strict_rows(source)
    tasks: list[dict[str, object]] = []
    for row in rows:
        task = render_task_prompt_audit(row)
        tasks.append(task)
        task_dir = output_dir / str(task["task_id"])
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "prompt_audit.json").write_text(
            json.dumps(task, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    passed = all(task["status"] == "passed" for task in tasks)
    report = {
        "schema_version": PROMPT_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "diagnostic_only": True,
        "paper_authority": False,
        "provider_calls": 0,
        "source_jsonl": str(source),
        "source_jsonl_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "task_order": [task["task_id"] for task in tasks],
        "prompt_kinds": list(_PROMPT_KINDS),
        "notes": (
            "Planner is exact pre-execution. Step/evidence-dependent prompts use "
            "their production builders with conservative case-bound representative "
            "inputs; they do not claim a final agent-authored plan or result."
        ),
        "tasks": tasks,
    }
    (output_dir / "canonical9_prompt_preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return report


__all__ = [
    "PROMPT_PREFLIGHT_SCHEMA_VERSION",
    "PromptPreflightError",
    "render_task_prompt_audit",
    "run_canonical9_prompt_preflight",
]

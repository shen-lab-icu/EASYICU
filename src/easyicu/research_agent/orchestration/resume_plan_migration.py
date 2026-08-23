"""Legacy resume-plan migration authority.

Extracted from ``pipeline.py`` (2026-08-22 decomposition batch). This module
owns the migrations a *resumed* plan needs before it can be executed again:
schema migration of legacy adjusted-association steps to the typed model
roster, restoration of the immutable plan-time robustness lock, migration of
older trajectory plans to the canonical replay-product schema, and restoration
of exact typed parent edges on legacy framework-split rendering steps.

The host keeps the coordinator (``_apply_resume_plan_migrations``) and re-imports
every public name below, so module-global monkeypatch seams in the resume tests
keep resolving through ``easyicu.research_agent.pipeline``.

This module must not import ``pipeline``: the dependency direction is
host -> migration authority, and the package graph is gated acyclic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..authority.evidence_store import EvidenceStore
from ..authority.runtime_artifacts import current_step_records
from ..plan_utils import (
    _effect_figure_semantics_supported_by_inputs,
    _effect_figure_semantics_supported_by_model_roster,
    _migrate_render_step_contract,
    _render_only_figure_step_intent,
    effect_output_authorized,
)
from ..planning.figure_plan_shaping import close_empty_deterministic_figure_contracts
from ..providers.prompt_budget import budgeted_role_client
from ..providers.protocol import LLMMessage
from ..robustness.panel import load_locked_robustness_specs, robustness_specs_sha
from ..schema import (
    ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES,
    ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
    PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,
    AnalysisPlan,
    AnalysisStep,
    PlannedModelRequirement,
    ResearchContext,
    ValidationFinding,
)
from ..trajectory.plan_contract import augment_trajectory_plan_products


class LegacyResumePlanMigrationError(RuntimeError):
    """A legacy resume plan could not be migrated without scientific drift."""


def _normalise_plan_contract_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _is_closed_adjusted_association_step(step: AnalysisStep) -> bool:
    """Match the typed roster's exact method-and-product contract only."""

    method_head = str(step.method or "").lower().split(" with ", 1)[0]
    if (
        _normalise_plan_contract_token(method_head)
        != PLANNED_MODEL_REQUIREMENTS_STEP_METHOD
    ):
        return False
    products = set()
    for output in step.expected_outputs or []:
        kind, separator, name = str(output or "").partition(":")
        if separator:
            products.add(
                (
                    _normalise_plan_contract_token(kind),
                    _normalise_plan_contract_token(name),
                )
            )
    return (
        PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
        PLANNED_MODEL_REQUIREMENTS_OUTPUT,
    ) in products


def _resume_completed_records_for_plan_migration(
    *,
    plan: AnalysisPlan,
    resume_state: Optional[Dict[str, Any]],
    resume_from_step_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Return current successful records that remain completed after a cut."""

    current_records = [
        dict(record)
        for record in current_step_records(
            [
                record
                for record in ((resume_state or {}).get("per_step_records") or [])
                if isinstance(record, dict) and record.get("step_id")
            ]
        )
    ]
    cut_step_id = str(resume_from_step_id or "").strip()
    if not cut_step_id:
        return [record for record in current_records if record.get("status") == "ok"]

    step_order = {step.step_id: index for index, step in enumerate(plan.steps)}
    if cut_step_id == "00_probe":
        cut_index = -1
    elif cut_step_id in step_order:
        cut_index = step_order[cut_step_id]
    else:
        raise LegacyResumePlanMigrationError(
            f"resume_from_step_id={cut_step_id!r} is not in the active analysis plan"
        )

    completed: List[Dict[str, Any]] = []
    for record in current_records:
        if record.get("status") != "ok":
            continue
        step_id = str(record.get("step_id") or "")
        record_index = -1 if step_id == "00_probe" else step_order.get(step_id)
        if record_index is not None and record_index < cut_index:
            completed.append(record)
    return completed


def _legacy_resume_model_roster_targets(
    *,
    plan: AnalysisPlan,
    completed_step_ids: set[str],
) -> tuple[str, ...]:
    """Select only remaining, exact closed-contract steps with an empty roster."""

    return tuple(
        step.step_id
        for step in plan.steps
        if step.step_id not in completed_step_ids
        and _is_closed_adjusted_association_step(step)
        and not step.model_requirements
    )


class _LegacyModelRosterStepPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    model_requirements: List[PlannedModelRequirement] = Field(min_length=1)


class _LegacyModelRosterPacket(BaseModel):
    """Planner-owned roster patch with no surface for broader plan edits."""

    model_config = ConfigDict(extra="forbid")

    steps: List[_LegacyModelRosterStepPacket] = Field(min_length=1)


def _parse_legacy_model_roster_packet(
    raw: str,
    *,
    target_step_ids: tuple[str, ...],
) -> _LegacyModelRosterPacket:
    packet = _LegacyModelRosterPacket.model_validate(json.loads(raw.strip()))
    returned_step_ids = [step.step_id for step in packet.steps]
    if returned_step_ids != list(target_step_ids):
        raise ValueError(
            "roster packet steps must exactly match the ordered target ids: "
            f"expected={list(target_step_ids)!r}, returned={returned_step_ids!r}"
        )
    for step in packet.steps:
        requirement_ids = [
            requirement.requirement_id for requirement in step.model_requirements
        ]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError(
                f"duplicate requirement_id in roster packet for {step.step_id!r}"
            )
        primary_count = sum(
            requirement.analysis_role == "primary"
            for requirement in step.model_requirements
        )
        if primary_count != 1:
            raise ValueError(
                "each target step roster must contain exactly one "
                "analysis_role='primary'; the Planner chooses which requirement "
                f"is primary (step={step.step_id!r}, returned={primary_count})"
            )
    return packet


def _project_legacy_model_roster_packet(
    *,
    plan: AnalysisPlan,
    packet: _LegacyModelRosterPacket,
) -> AnalysisPlan:
    """Project only validated roster values onto an otherwise frozen plan."""

    rosters = {
        step.step_id: [
            requirement.model_dump(mode="json")
            for requirement in step.model_requirements
        ]
        for step in packet.steps
    }
    payload = plan.model_dump(mode="json")
    for step_payload in payload["steps"]:
        step_id = str(step_payload.get("step_id") or "")
        if step_id in rosters:
            step_payload["model_requirements"] = rosters[step_id]
    return AnalysisPlan.model_validate(payload)


def _next_analysis_plan_revision(
    *,
    run_dir: Path,
    plan: AnalysisPlan,
    evidence: EvidenceStore,
) -> int:
    revision = int(plan.revision) + 1
    for path in run_dir.glob("analysis_plan_revision_*.json"):
        match = re.fullmatch(r"analysis_plan_revision_(\d+)\.json", path.name)
        if match:
            revision = max(revision, int(match.group(1)) + 1)
    while evidence.get(f"analysis_plan_revision_{revision}") is not None:
        revision += 1
    return revision


def _restore_resume_plan_robustness_lock(
    *,
    plan: AnalysisPlan,
    run_dir: Path,
    evidence: EvidenceStore,
    prompt_version: str,
    llm_signature: str,
) -> tuple[AnalysisPlan, Optional[Path]]:
    """Project the verified plan-time robustness lock onto a resume plan.

    A probe-time replanner from older runs could reword or drop robustness
    specifications after the immutable lock was written. Execution correctly
    rejects that drift, but resume must load a plan that agrees with the lock.
    The lock remains the authority: this migration writes a new immutable plan
    revision and never rewrites the locked specifications.
    """

    lock_path = Path(run_dir) / "robustness_specs_locked.json"
    if not lock_path.is_file():
        return plan, None
    locked_specs = load_locked_robustness_specs(run_dir)
    active_specs = list(plan.robustness_specs or [])
    if robustness_specs_sha(active_specs) == robustness_specs_sha(locked_specs):
        return plan, None

    revision = _next_analysis_plan_revision(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
    )
    restored = plan.model_copy(
        update={
            "robustness_specs": list(locked_specs),
            "revision": revision,
        }
    )
    revision_path = run_dir / f"analysis_plan_revision_{revision}.json"
    if revision_path.exists():
        raise LegacyResumePlanMigrationError(
            f"refusing to overwrite existing plan revision {revision_path.name}"
        )
    revision_path.write_text(restored.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=(
            "Resume migration restoring the immutable plan-time robustness "
            "specification lock."
        ),
        source_path=revision_path,
        evidence_id=f"analysis_plan_revision_{revision}",
        producer="planner",
        generation_mode="system",
        prompt_pack_version=prompt_version,
        metadata={
            "reason": "restore_locked_robustness_specs",
            "llm_signature": llm_signature,
        },
    )
    return restored, revision_path


def _migrate_legacy_resume_figure_render_edges(
    *,
    plan: AnalysisPlan,
    run_dir: Path,
    resume_state: Optional[Dict[str, Any]],
    resume_from_step_id: Optional[str],
    evidence: EvidenceStore,
    prompt_version: str,
    llm_signature: str,
) -> tuple[AnalysisPlan, Optional[Path], tuple[str, ...]]:
    """Restore exact typed parent edges on legacy system-split figure steps.

    Older framework-generated render children copied the parent's raw inputs and
    scientific method. Current source-data authority requires a host-resolved
    typed edge. This migration is intentionally narrower than ordinary plan
    shaping: it recognizes only the full legacy splitter fingerprint or an
    already-visualization child with one globally unique exact typed table role.
    Raw artifacts, sibling tables, datasets, and models remain excluded so
    rendering cannot reopen the scientific analysis. Any ambiguity remains
    fail-closed for the Planner.
    """

    from ..contracts.declared_product import (
        effect_bearing_product,
        typed_product,
    )

    completed_records = _resume_completed_records_for_plan_migration(
        plan=plan,
        resume_state=resume_state,
        resume_from_step_id=resume_from_step_id,
    )
    completed_step_ids = {
        str(record.get("step_id") or "") for record in completed_records
    }
    if len(plan.steps) < 2:
        return plan, None, ()

    step_order = {str(step.step_id): index for index, step in enumerate(plan.steps)}
    cut_step_id = str(resume_from_step_id or "").strip()
    cut_index: Optional[int] = None
    if cut_step_id:
        if cut_step_id == "00_probe":
            cut_index = 0
        elif cut_step_id in step_order:
            cut_index = step_order[cut_step_id]
        else:
            raise LegacyResumePlanMigrationError(
                f"resume_from_step_id={cut_step_id!r} is not in the active analysis plan"
            )

    eligible_figure_ids = [
        str(step.step_id)
        for index, step in enumerate(plan.steps)
        if str(step.step_id) not in completed_step_ids
        and (cut_index is None or index >= cut_index)
    ]
    closed_plan, _closure_findings = close_empty_deterministic_figure_contracts(
        plan=plan,
        eligible_step_ids=eligible_figure_ids,
    )
    initially_closed_ids = [
        str(before.step_id)
        for before, after in zip(plan.steps, closed_plan.steps)
        if before != after
    ]
    plan = closed_plan

    producer_ids: Dict[tuple[str, str], set[str]] = {}
    producer_tokens: Dict[tuple[str, str], List[tuple[str, str]]] = {}
    for producer_step in plan.steps:
        for output in producer_step.expected_outputs or []:
            parsed = typed_product(output)
            if parsed is not None and parsed[0] in {"statistic", "table"}:
                producer_ids.setdefault(parsed, set()).add(str(producer_step.step_id))
                producer_tokens.setdefault(parsed, []).append(
                    (str(producer_step.step_id), str(output))
                )

    def _exact_role_dependencies(
        figure_outputs: Sequence[str],
        *,
        required_producer_id: Optional[str] = None,
        required_producer_step: Optional[AnalysisStep] = None,
    ) -> tuple[List[str], set[str]]:
        dependencies: List[str] = []
        dependency_producers: set[str] = set()
        for figure_output in figure_outputs:
            parsed_figure = typed_product(figure_output)
            if parsed_figure is None or parsed_figure[0] != "figure":
                return [], set()
            candidates = [
                candidate
                for kind in ("table", "statistic")
                for candidate in producer_tokens.get((kind, parsed_figure[1]), [])
            ]
            if required_producer_id is not None:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[0] == required_producer_id
                ]
            if not candidates and effect_bearing_product(figure_output):
                semantic_candidates: List[tuple[str, str]] = []
                for identity, raw_candidates in producer_tokens.items():
                    if required_producer_id is not None:
                        raw_candidates = [
                            candidate
                            for candidate in raw_candidates
                            if candidate[0] == required_producer_id
                        ]
                    if not raw_candidates:
                        continue
                    supported = _effect_figure_semantics_supported_by_inputs(
                        figure_outputs=[figure_output],
                        effect_input_products={identity},
                    ) or (
                        required_producer_step is not None
                        and _effect_figure_semantics_supported_by_model_roster(
                            step=required_producer_step,
                            figure_outputs=[figure_output],
                            effect_input_products={identity},
                        )
                    )
                    if supported:
                        semantic_candidates.extend(raw_candidates)
                candidates = semantic_candidates
            if len(candidates) != 1:
                return [], set()
            producer_id, source_token = candidates[0]
            dependencies.append(source_token)
            dependency_producers.add(producer_id)
        return list(dict.fromkeys(dependencies)), dependency_producers

    revised_steps = list(plan.steps)
    migrated_step_ids: List[str] = list(initially_closed_ids)

    for index in range(1, len(plan.steps)):
        parent = plan.steps[index - 1]
        child = plan.steps[index]
        parent_id = str(parent.step_id)
        child_id = str(child.step_id)
        child_is_in_resume_window = cut_index is None or index >= cut_index
        parent_is_available_or_scheduled = parent_id in completed_step_ids or (
            cut_index is not None and index - 1 >= cut_index
        )
        if (
            not child_is_in_resume_window
            or not parent_is_available_or_scheduled
            or child_id in completed_step_ids
            or child_id != f"{parent_id}_figure"
            or str(child.method) != str(parent.method)
            or list(child.inputs or []) != list(parent.inputs or [])
            or list(child.icu_rule_refs or [])
            != [*list(parent.icu_rule_refs or []), "visualization_rule"]
            or child.model_requirements
            or child.trajectory_stability_spec is not None
        ):
            continue

        parent_outputs = list(parent.expected_outputs or [])
        if any(
            (parsed := typed_product(raw)) is not None and parsed[0] == "figure"
            for raw in parent_outputs
        ):
            continue

        figure_outputs: List[str] = []
        child_contract_valid = True
        for raw in child.expected_outputs or []:
            parsed = typed_product(raw)
            if parsed is None:
                child_contract_valid = False
                break
            if parsed[0] == "figure":
                figure_outputs.append(str(raw))
            elif parsed[0] != "log":
                child_contract_valid = False
                break
        figure_identities = [typed_product(raw) for raw in figure_outputs]
        if (
            not child_contract_valid
            or not figure_outputs
            or len(figure_identities) != len(set(figure_identities))
            or child.intent
            != _render_only_figure_step_intent(
                source_step_id=parent_id,
                figure_outputs=figure_outputs,
            )
        ):
            continue

        source_tokens, dependency_producers = _exact_role_dependencies(
            figure_outputs,
            required_producer_id=parent_id,
            required_producer_step=parent,
        )
        source_identities = {
            parsed
            for raw in source_tokens
            if (parsed := typed_product(raw)) is not None
        }
        source_names = [identity[1] for identity in source_identities]
        if (
            not source_tokens
            or dependency_producers != {parent_id}
            or len(source_identities) != len(source_tokens)
            or len(source_names) != len(set(source_names))
            or any(
                producer_ids.get(identity) != {parent_id}
                for identity in source_identities
            )
        ):
            continue

        if any(effect_bearing_product(raw) for raw in figure_outputs) and (
            not effect_output_authorized(parent)
            or not (
                _effect_figure_semantics_supported_by_inputs(
                    figure_outputs=figure_outputs,
                    effect_input_products=source_identities,
                )
                or _effect_figure_semantics_supported_by_model_roster(
                    step=parent,
                    figure_outputs=figure_outputs,
                    effect_input_products=source_identities,
                )
            )
        ):
            continue

        revised_steps[index] = _migrate_render_step_contract(
            child, source_tokens, method="visualization"
        )
        migrated_step_ids.append(child_id)

    # A later framework splitter already emitted a visualization child, but an
    # older plan may still bind it to a sibling table even though the declared
    # figure has one globally unique exact typed table role elsewhere. Preserve
    # step ids/order and replace only that closed edge; never infer multi-table
    # dependencies or grant every table owned by the producer.
    for index, original_child in enumerate(plan.steps):
        child = revised_steps[index]
        child_id = str(child.step_id)
        if (
            child_id in completed_step_ids
            or (cut_index is not None and index < cut_index)
            or not child_id.endswith("_figure")
            or _normalise_plan_contract_token(str(child.method or "")).split(
                "_with_", 1
            )[0]
            != "visualization"
            or not child.inputs
            or any(
                (parsed := typed_product(raw)) is None
                or parsed[0] not in {"statistic", "table"}
                for raw in child.inputs
            )
        ):
            continue
        figure_outputs = [
            str(raw)
            for raw in child.expected_outputs or []
            if (parsed := typed_product(raw)) is not None and parsed[0] == "figure"
        ]
        parsed_child_outputs = [
            typed_product(raw) for raw in child.expected_outputs or []
        ]
        if (
            not figure_outputs
            or any(parsed is None for parsed in parsed_child_outputs)
            or any(
                parsed[0] not in {"figure", "log"}
                for parsed in parsed_child_outputs
                if parsed is not None
            )
        ):
            continue
        source_tokens, dependency_producers = _exact_role_dependencies(figure_outputs)
        if len(dependency_producers) != 1 or not source_tokens:
            continue
        source_step_id = next(iter(dependency_producers))
        source_index = step_order.get(source_step_id)
        if (
            source_index is None
            or source_index >= index
            or (
                source_step_id not in completed_step_ids
                and not (cut_index is not None and source_index >= cut_index)
            )
        ):
            continue
        intended = _render_only_figure_step_intent(
            source_step_id=source_step_id,
            figure_outputs=figure_outputs,
        )
        migrated_child = _migrate_render_step_contract(
            child, source_tokens, intent=intended
        )
        if child == migrated_child:
            continue
        revised_steps[index] = migrated_child
        if child_id not in migrated_step_ids:
            migrated_step_ids.append(child_id)

    if not migrated_step_ids:
        return plan, None, ()

    revision = _next_analysis_plan_revision(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
    )
    migrated = plan.model_copy(
        update={
            "steps": revised_steps,
            "revision": revision,
        }
    )
    revision_path = run_dir / f"analysis_plan_revision_{revision}.json"
    if revision_path.exists():
        raise LegacyResumePlanMigrationError(
            f"refusing to overwrite existing plan revision {revision_path.name}"
        )
    revision_path.write_text(migrated.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=(
            "Resume migration restoring exact typed parent edges on legacy "
            "framework-split rendering steps."
        ),
        source_path=revision_path,
        evidence_id=f"analysis_plan_revision_{revision}",
        producer="planner",
        generation_mode="system",
        prompt_pack_version=prompt_version,
        metadata={
            "reason": "resume_legacy_figure_render_edges",
            "target_step_ids": migrated_step_ids,
            "llm_signature": llm_signature,
        },
    )
    return migrated, revision_path, tuple(migrated_step_ids)


def _migrate_resume_trajectory_products(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    run_dir: Path,
    evidence: EvidenceStore,
    prompt_version: str,
    llm_signature: str,
) -> tuple[AnalysisPlan, Optional[Path], List[ValidationFinding]]:
    """Apply schema-only trajectory products to a reused legacy plan.

    Resume normally skips plan-shaping transforms to preserve step identities.
    Canonical trajectory products are the safe exception: augmentation changes
    neither step ids/order nor any scientific method, input, horizon, threshold,
    or cluster choice. Older checkpoints may predate the role recognizer, so
    treating their saved plan as already normalized silently removes the replay
    contracts from resumed execution.
    """

    augmented, augmentation_findings = augment_trajectory_plan_products(
        plan=plan,
        context=context,
    )
    if augmented == plan:
        return plan, None, augmentation_findings

    revision = _next_analysis_plan_revision(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
    )
    augmented = augmented.model_copy(update={"revision": revision})
    revision_path = run_dir / f"analysis_plan_revision_{revision}.json"
    if revision_path.exists():
        raise LegacyResumePlanMigrationError(
            f"refusing to overwrite existing plan revision {revision_path.name}"
        )
    revision_path.write_text(augmented.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=(
            "Resume migration adding canonical trajectory replay products to "
            "the existing agent-owned role DAG."
        ),
        source_path=revision_path,
        evidence_id=f"analysis_plan_revision_{revision}",
        producer="planner",
        generation_mode="system",
        prompt_pack_version=prompt_version,
        metadata={
            "reason": "resume_trajectory_schema_products",
            "llm_signature": llm_signature,
        },
    )
    return augmented, revision_path, augmentation_findings


def _migrate_legacy_resume_model_requirements(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    run_dir: Path,
    resume_state: Optional[Dict[str, Any]],
    resume_from_step_id: Optional[str],
    role_resolver: Callable[[str], Any],
    evidence: EvidenceStore,
    prompt_version: str,
    llm_signature: str,
    max_prompt_tokens: Optional[int] = None,
    allow_scientific_migration: bool = True,
) -> tuple[AnalysisPlan, Optional[Path], tuple[str, ...]]:
    """Ask the planner LLM to migrate an old empty typed-model roster.

    The framework identifies only the schema surface that requires migration.
    It never derives an outcome, exposure, analysis role, analysis set, or model
    family from prose. Those scientific commitments must come back in a small,
    strictly typed PlannerAgent packet. The framework projects only that roster
    onto the frozen plan before a revision is written or registered.
    """

    completed_records = _resume_completed_records_for_plan_migration(
        plan=plan,
        resume_state=resume_state,
        resume_from_step_id=resume_from_step_id,
    )
    completed_step_ids = {str(record.get("step_id")) for record in completed_records}
    target_step_ids = _legacy_resume_model_roster_targets(
        plan=plan,
        completed_step_ids=completed_step_ids,
    )
    if not target_step_ids:
        return plan, None, ()
    if not allow_scientific_migration:
        raise LegacyResumePlanMigrationError(
            "paper-facing legacy runs missing typed model authority cannot be "
            "resumed with a new Planner decision; create a fresh run"
        )

    target_steps = [
        {
            "step_id": step.step_id,
            "intent": step.intent,
            "method": step.method,
            "inputs": list(step.inputs),
            "expected_outputs": list(step.expected_outputs),
            "icu_rule_refs": list(step.icu_rule_refs),
        }
        for step in plan.steps
        if step.step_id in set(target_step_ids)
    ]
    required_fields = [
        "requirement_id",
        "outcome",
        "outcome_type",
        "method_family",
        "exposure_source",
        "analysis_role",
        "analysis_set",
        "required_for_step_success",
    ]
    format_reminder = (
        'Return exactly {"steps": [{"step_id": <target id>, '
        '"model_requirements": [<one or more complete requirement objects>]}]}. '
        f"Every requirement object must contain all fields {required_fields!r}. "
        "Allowed outcome_type: binary, continuous. Allowed analysis_role: "
        "primary, secondary, sensitivity. Allowed analysis_set: source_aware, "
        "complete_case. Primary/secondary requirements must set "
        "required_for_step_success=true; only sensitivity may be false. Each "
        "target step must contain exactly one analysis_role=primary; the Planner "
        "chooses which requirement is primary and labels all others secondary "
        "or sensitivity."
    )
    plan_payload = plan.model_dump(mode="json")
    plan_level_commitments = {
        key: plan_payload.get(key)
        for key in (
            "research_question",
            "analysis_type",
            "rationale",
            "cohort",
            "robustness_specs",
        )
    }
    messages = [
        LLMMessage(
            role="system",
            content=(
                "You are the PlannerAgent's typed legacy-plan migration worker. "
                "Choose the scientific model roster; the framework will only "
                "validate and project it. Return JSON only. Never rewrite the "
                "AnalysisPlan and never invent a default model when the supplied "
                "plan and ResearchContext do not justify one."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                "Populate model_requirements for every target step. Each roster "
                "entry is a complete PlannedModelRequirement. Choose outcome, "
                "exposure_source, method_family, role, and analysis set from the "
                "unchanged scientific commitments below. Emit a separate "
                "requirement for every adjusted outcome/model pre-specified in "
                "the target step intent; ResearchContext.target_outcome is not "
                "an exhaustive roster and must not cause an intent-committed "
                "secondary outcome/model to be omitted. The Planner decides the "
                "roster count and contents; the framework does not infer either "
                "from prose. For each target step, choose exactly one roster "
                "entry as analysis_role=primary and label the other entries "
                "secondary or sensitivity. Binary method_family "
                f"must be one of {sorted(ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES)!r}; "
                "continuous method_family must be one of "
                f"{sorted(ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES)!r}. "
                "Do not return plan fields, prose, or requirements for any other "
                "step.\n\n"
                f"REQUIRED JSON SHAPE:\n{format_reminder}\n\n"
                "TARGET STEPS (verbatim from the saved planner plan):\n"
                f"{json.dumps(target_steps, indent=2, ensure_ascii=False)}\n\n"
                "READ-ONLY PLAN-LEVEL COMMITMENTS (context only; do not return "
                "these fields):\n"
                f"{json.dumps(plan_level_commitments, indent=2, ensure_ascii=False)}\n\n"
                "RESEARCH CONTEXT:\n"
                f"{context.model_dump_json(indent=2)}"
            ),
        ),
    ]
    try:
        from ..providers.structured_retry import call_llm_with_structured_retry

        packet = call_llm_with_structured_retry(
            budgeted_role_client(
                role_resolver,
                "planner",
                "legacy_model_roster_migration",
                limit_tokens=max_prompt_tokens,
            ),
            messages,
            parser=lambda raw: _parse_legacy_model_roster_packet(
                raw,
                target_step_ids=target_step_ids,
            ),
            role="legacy_model_roster_migration",
            max_retries=2,
            max_tokens=4096,
            temperature=0.1,
            format_reminder=format_reminder,
        )
    except Exception as exc:
        raise LegacyResumePlanMigrationError(
            "planner LLM failed while migrating legacy model_requirements; "
            "resume stopped without a default model"
        ) from exc
    revised = _project_legacy_model_roster_packet(
        plan=plan,
        packet=packet,
    )
    revision = _next_analysis_plan_revision(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
    )
    revised = revised.model_copy(update={"revision": revision})
    revision_path = run_dir / f"analysis_plan_revision_{revision}.json"
    if revision_path.exists():
        raise LegacyResumePlanMigrationError(
            f"refusing to overwrite existing plan revision {revision_path.name}"
        )
    revision_path.write_text(revised.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=(
            "Planner-owned legacy resume migration for typed model requirements."
        ),
        source_path=revision_path,
        evidence_id=f"analysis_plan_revision_{revision}",
        producer="planner",
        generation_mode="llm",
        prompt_pack_version=prompt_version,
        metadata={
            "reason": "legacy_missing_model_requirements",
            "target_step_ids": list(target_step_ids),
            "llm_signature": llm_signature,
        },
    )
    return revised, revision_path, target_step_ids

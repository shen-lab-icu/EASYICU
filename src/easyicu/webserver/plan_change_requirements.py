"""Verify fresh amendments against their exact source planning metadata.

Reads review/checkpoint metadata only. A source's readability, prior approval,
patient input or successful results cannot authorize the new candidate to run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.frozen_payload import thaw_payload
from easyicu.research_agent.orchestration.human_review_checkpoint import (
    HumanReviewCheckpoint, _MAX_CHECKPOINT_BYTES,
)
from easyicu.research_agent.planning.baseline_requirements import (
    candidate_baseline_requirements, context_baseline_requirements,
)
from easyicu.research_agent.planning.population_requirements import (
    candidate_population_requirements, context_population_requirements,
)
from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.research_agent.schema import ResearchContext
from easyicu.webserver import agent_runs, study_contexts
from easyicu.webserver.plan_change_request import (
    PlanChangeRequest, PlanChangeRequirements, ReferencedPlan, reference_plan_content,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.run_record import RunRecordReadError


def _compiled_runtime_continuity(study: Mapping[str, Any], run: Mapping[str, Any]) -> bool:
    from easyicu.webserver.pi_copilot.plan_review_progress import matching_progress

    progress = matching_progress(study, run)
    return progress is not None and progress.choices == {
        "agent_plan_configuration": "typed_runtime_projection",
    }


def compiled_configuration_plan_change(
    *, study: Mapping[str, Any], project_root: str,
) -> PlanChangeRequest | None:
    """Recover planning continuity after the Host compiled this exact candidate.

    The browser supplies no historical plan or replacement question. Only the
    existing CAS-bound runtime-projection receipt permits this handoff; other
    scientific edits and user choices do not silently inherit the old scope.
    """
    confirmations = study.get("confirmations")
    if not isinstance(confirmations, Mapping) or confirmations.get("agent_plan_configuration_compiled") is not True:
        return None
    rows = agent_runs.list_run_history(
        study_id=str(study.get("id") or ""), project_root=project_root, limit=200,
    ).get("runs", ())
    row = next((row for row in rows if _compiled_runtime_continuity(study, row)), None)
    if row is None:
        return None
    request = bind_plan_change_requirements(PlanChangeRequest(
        source_run_id=str(row["run_id"]),
        source_scientific_configuration_sha256=str(row["scientific_configuration_sha256"]),
        target_scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(dict(study)),
        user_message=(
            "The Host compiled the runtime coordinates declared by this candidate. "
            "Revise the complete plan while retaining its baseline content and population "
            "requirements. This automatic configuration step is not a change to the "
            "research question, a request to reduce the baseline to model covariates, "
            "or approval to execute. Address the remaining scientific-review findings."
        ),
    ), study=study, project_root=project_root)
    # This is the same verified owner payload used by the requirement binder.
    try:
        review = agent_runs.read_run_review(str(row["project_dir"]))
        scientific = PlanScientificReview.model_validate(review["artifact_payloads"]["scientific_plan_review.json"])
        requirements = request.source_requirements
        if (requirements is None or scientific.plan_sha256 != requirements.source_plan_sha256
            or scientific.context_sha256 != requirements.source_context_sha256):
            raise ValueError("review changed after requirement binding")
        feedback = "\n".join(
            f"{finding.code}: {finding.message} Remediation: {finding.remediation}"
            for finding in scientific.findings
        )
        return PlanChangeRequest.model_validate({
            **request.model_dump(mode="json"), "user_message": request.user_message + "\n" + feedback,
        })
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ResearchPipelineRunError(
            "plan_change_review_feedback_invalid",
            "The exact candidate review is unavailable, changed, or exceeds the revision request budget. "
            "No new planning or analysis was started.",
        ) from exc


def bind_plan_change_requirements(
    request: PlanChangeRequest, *, study: Mapping[str, Any], project_root: str,
) -> PlanChangeRequest:
    """Re-derive at launch; reject source, configuration or supplied binding drift.

    Legacy requests remain readable/hash-stable. A new launch resolves them
    against the current host source; a different scientific scope needs the
    explicitly paired source/target digests issued by the conversation host.
    """
    try:
        rows = agent_runs.list_run_history(
            study_id=str(study.get("id") or ""), project_root=project_root, limit=200,
        ).get("runs", ())
        row = next(item for item in rows if item.get("run_id") == request.source_run_id)
        root = Path(project_root).expanduser().resolve()
        wrapper = Path(str(row.get("project_dir") or "")).expanduser()
        study_root = root / str(study.get("id") or "")
        if (
            not study.get("id") or Path(str(study["id"])).name != study["id"]
            or Path(request.source_run_id).name != request.source_run_id
            or wrapper.is_symlink() or study_root.is_symlink()
            or wrapper.parent.resolve() != study_root.resolve()
        ):
            raise ValueError("source outside exact study")
        record = agent_runs.read_run_record(str(wrapper))
        if isinstance(record, RunRecordReadError) or record.run_id != request.source_run_id or record.study_id != study["id"]:
            raise ValueError("source record identity changed")
        source_digest = record.scientific_configuration_sha256
        target_digest = study_contexts.scientific_configuration_sha256(dict(study))
        if not source_digest or source_digest != row.get("scientific_configuration_sha256"):
            raise ValueError("source configuration unavailable")
        if request.source_scientific_configuration_sha256 is not None:
            if (request.source_scientific_configuration_sha256 != source_digest
                or request.target_scientific_configuration_sha256 != target_digest):
                raise ValueError("revision configuration drift")
        elif source_digest != target_digest:
            raise ValueError("legacy request cannot silently cross scientific configurations")

        artifact = next(a for a in record.artifacts if a.name == "agent_plan.json")
        plan = thaw_payload(record.artifact_payloads["agent_plan.json"])
        reference = next((r for r in request.reference_plans if r.run_id == request.source_run_id), None)
        if reference is None and request.reference_plans:
            raise ValueError("source absent from references")
        expected_reference = ReferencedPlan(
            run_id=request.source_run_id, artifact_sha256=artifact.sha256,
            plan=reference_plan_content(plan),
        )
        if reference is not None and reference != expected_reference:
            raise ValueError("source reference content or digest changed")
        review = PlanScientificReview.model_validate(thaw_payload(record.artifact_payloads["scientific_plan_review.json"]))
        run = wrapper / "pipeline" / request.source_run_id
        checkpoint_path = run / "human_review_checkpoint.json"
        if (run.is_symlink() or run.parent.is_symlink() or checkpoint_path.is_symlink()
            or checkpoint_path.stat().st_size > _MAX_CHECKPOINT_BYTES):
            raise ValueError("invalid source checkpoint path")
        # Requirements outlive a review's approval expiry. Validate its hashes
        # without consuming it or interpreting its state as a fresh approval.
        checkpoint = HumanReviewCheckpoint.model_validate_json(checkpoint_path.read_bytes())
        context_payload = thaw_payload(checkpoint.plan_handoff["context"])
        if (checkpoint.run_id != request.source_run_id
            or canonical_sha256(plan) != review.plan_sha256
            or canonical_sha256(checkpoint.plan_handoff["plan"]) != review.plan_sha256
            or canonical_sha256(context_payload) != review.context_sha256):
            raise ValueError("source plan/review/context drift")
        context = ResearchContext.model_validate(context_payload)
        baseline = population = None
        concepts: set[str] = set()
        operationalized: set[str] = set()
        if source_digest == target_digest or _compiled_runtime_continuity(study, row):
            # Carry an already accepted roster even if the source candidate
            # itself failed to deliver it. Available columns are not requirements.
            baseline = context_baseline_requirements(context)
            if baseline is None:
                baseline = candidate_baseline_requirements(
                    plan=plan, source_plan_sha256=review.plan_sha256,
                    selected_concepts=[v.name for v in context.variables if v.name == v.source_concept],
                    catalog_columns=[v.name for v in context.variables],
                )
            population = context_population_requirements(context) or candidate_population_requirements(
                plan, artifact.sha256, source_digest_kind="artifact_sha256",
            )
            variables = {v.name: v for v in context.variables}
            if baseline is not None:
                for table in baseline.tables:
                    for coordinate in (table.group_by, *table.variables):
                        if coordinate is None:
                            continue
                        descriptor = variables.get(coordinate.name)
                        source = coordinate.source_concept or (
                            descriptor.source_concept if descriptor is not None else None
                        ) or coordinate.name
                        concepts.add(source)
                        if descriptor is not None and coordinate.source_concept is None:
                            operationalized.add(coordinate.name)
        bound = PlanChangeRequirements(
            source_plan_sha256=review.plan_sha256, source_context_sha256=review.context_sha256,
            baseline=baseline, population=population,
            planning_concepts=tuple(sorted(concepts)), operationalized_columns=tuple(sorted(operationalized)),
        )
        if request.source_requirements is not None and request.source_requirements != bound:
            raise ValueError("revision requirement binding drift")
        return PlanChangeRequest.model_validate({
            **request.model_dump(mode="json"),
            "reference_plans": [r.model_dump(mode="json") for r in request.reference_plans or (expected_reference,)],
            "source_scientific_configuration_sha256": source_digest,
            "target_scientific_configuration_sha256": target_digest,
            "source_requirements": bound.model_dump(mode="json"),
        })
    except (OSError, KeyError, StopIteration, TypeError, ValueError) as exc:
        raise ResearchPipelineRunError(
            "plan_change_requirements_source_invalid",
            "The amendment's exact source, review metadata or scientific configuration changed. "
            "No source inputs, results or approval were reused; resolve the source before fresh planning.",
        ) from exc


__all__ = ["bind_plan_change_requirements"]

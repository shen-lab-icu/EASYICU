"""Orchestrate one Progressive Planner call and optional Dev checkpoint replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..agents.progressive_planner import (
    ProgressivePlannerAgent,
    ProgressivePlannerRunFacts,
    progressive_planner_failure_facts,
)
from ..authority.evidence_store import sha256_of_file
from ..planning.progressive_artifacts import (
    ProgressiveEvidenceRegistrar,
    ProgressivePlannerCheckpointRecorder,
    ProgressiveResumePersistenceReceipt,
    load_progressive_planner_checkpoint_chain,
    persist_progressive_compile_failure_replay,
    persist_progressive_design_canary_receipt,
    persist_progressive_planning_artifacts,
)
from ..planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
    ProgressivePlannerCheckpoint,
)
from ..planning.population_compile import (
    POPULATION_APPROVAL_STOPS,
    CompiledCriterion,
    CompiledPopulation,
)
from ..planning.population_shadow import (
    population_cohort_audit,
    superseded_predicates_finding,
    write_population_audit,
)
from ..planning.approval_stops import PLAN_APPROVAL_STOPS
from ..planning.capability_gap import check_capability_gap
from ..planning.population_spec import STUDY_WORDING_SOURCES
from ..planning.question_requirements import (
    QuestionRequirementsError,
    bind_named_question_concepts,
    concept_relatives,
    judge_question_requirements,
    named_question_concepts,
    outline_route_unstated,
    question_requirement_coverage,
    question_requirement_findings,
    question_requirements_record,
    write_question_requirements,
)
from ..planning.preplan_know_how import PlannerKnowHowBinding
from ..planning.question_substance_forms import (
    QUESTION_SUBSTANCE_FORMS_KEY,
    SUBSTITUTED_REASON,
    QuestionSubstanceFormError,
    question_substance_forms,
    substance_form_substitutions,
)
from ..planning.progressive_compiler import stated_population
from ..planning import literature_design_authority as _literature_design
from ..schema import AnalysisPlan, ResearchContext, ValidationFinding
from .workflow import PlannerDesignCanaryComplete


@dataclass(frozen=True)
class ProgressivePlannerRunResult:
    """Plan and every provenance fact captured from the same attempt."""

    plan: AnalysisPlan
    generation_mode: str
    prompt_metrics: Mapping[str, Any]
    facts: ProgressivePlannerRunFacts


@dataclass(frozen=True)
class ProgressiveDesignCanaryDraft:
    """Validated design outline before any executable-plan materialization."""

    outline: ProgressivePlanOutline
    checkpoint: ProgressivePlannerCheckpoint
    generation_mode: str
    prompt_metrics: Mapping[str, Any]


def finalize_progressive_design_canary(
    draft: ProgressiveDesignCanaryDraft,
    run_id: str,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    cost_meter: Any,
    provider_hard_stop: Any,
    prompt_pack_version: str,
    emit_progress: Callable[..., None],
) -> PlannerDesignCanaryComplete:
    """Persist and project one non-executable design-canary terminal result."""

    emit_progress(
        "plan",
        "Research-design canary completed at the validated outline boundary.",
        run_id=run_id,
        total_steps=0,
    )
    hard_stop_accounting = None
    accounting_summary = getattr(provider_hard_stop, "accounting_summary", None)
    if callable(accounting_summary):
        hard_stop_accounting = accounting_summary()
    cost_summary = (
        cost_meter.summary(hard_stop_accounting=hard_stop_accounting)
        if cost_meter is not None
        else {}
    )
    receipt, receipt_path, receipt_sha256 = (
        persist_progressive_design_canary_receipt(
            run_dir=run_dir,
            evidence=evidence,
            checkpoint=draft.checkpoint,
            prompt_metrics=draft.prompt_metrics,
            cost_summary=cost_summary,
            prompt_pack_version=prompt_pack_version,
        )
    )
    return PlannerDesignCanaryComplete(
        run_id=run_id,
        run_dir=str(run_dir),
        receipt_path=str(receipt_path),
        receipt_sha256=receipt_sha256,
        candidate_design_count=receipt.candidate_design_count,
        rejected_design_count=len(receipt.rejected_design_ids),
        selected_literature_dimension_count=(
            receipt.selected_literature_dimension_count
        ),
        provider_calls=int(receipt.planner_efficiency.get("calls") or 0),
        reported_tokens=int(
            receipt.planner_efficiency.get("reported_tokens") or 0
        ),
        estimated_cost_usd=(
            float(cost_summary["total_cost_usd"])
            if cost_summary.get("total_cost_usd") is not None
            else None
        ),
    )


def run_pipeline_progressive_planner(
    *,
    planner: ProgressivePlannerAgent,
    context: ResearchContext,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    prompt_pack_version: str,
    resume_checkpoint_path: str | Path | None,
    resume_checkpoint_sha256: str | None,
    stop_after_outline: bool,
    cohort_path: Path,
    llm_signature: str,
    planner_kwargs: Mapping[str, Any],
    preplan_literature: Any,
    required_primary_cohort_selection_mode: str | None,
    know_how_binding: PlannerKnowHowBinding,
    planning_contract_context: str,
    finding_sink: Callable[[ValidationFinding], None],
) -> ProgressivePlannerRunResult | ProgressiveDesignCanaryDraft:
    """Bind pipeline-owned inputs to the narrow Progressive Planner contract."""

    kwargs = dict(planner_kwargs)
    kwargs |= _literature_design.progressive_literature_design_kwargs(
        preplan_literature
    )
    kwargs["required_primary_cohort_selection_mode"] = (
        required_primary_cohort_selection_mode
    )
    return run_progressive_planner(
        planner=planner,
        context=context,
        run_dir=run_dir,
        evidence=evidence,
        prompt_pack_version=prompt_pack_version,
        resume_checkpoint_path=resume_checkpoint_path,
        resume_checkpoint_sha256=resume_checkpoint_sha256,
        cohort_path=cohort_path,
        llm_signature=llm_signature,
        planner_kwargs=kwargs,
        know_how_binding=know_how_binding,
        planning_contract_context=planning_contract_context,
        finding_sink=finding_sink,
        stop_after_outline=stop_after_outline,
    )


def _resume_finding(
    *,
    receipt: ProgressiveResumePersistenceReceipt,
    terminal_artifact_sha256: str,
) -> ValidationFinding:
    return ValidationFinding(
        validator="progressive_planner_resume",
        severity="warning",
        message=(
            "Development-only Progressive Planner prefix was dependency-verified "
            "and recompiled by the current host."
        ),
        detail={
            "reason_code": "progressive_development_checkpoint_resumed",
            "development_only": True,
            "source_terminal_artifact_sha256": terminal_artifact_sha256,
            "source_checkpoint_sha256": receipt.source_checkpoint_sha256,
            "source_sequence": receipt.source_sequence,
            "reused_materialization_count": receipt.reused_materialization_count,
            "new_checkpoint_count": receipt.new_checkpoint_count,
        },
    )


_POPULATION_STOP_CAUSES = {
    "requires_extraction": (
        "that the data bound to the study cannot apply; an extraction of the "
        "study's own population can"
    ),
    "not_applied": "that cannot be applied as stated",
}


def _criterion_row(item: CompiledCriterion) -> dict[str, Any]:
    criterion = item.criterion
    return {
        "id": criterion.id,
        "kind": criterion.kind,
        "role": criterion.role,
        "source": criterion.source,
        "stated_by_study": criterion.source in STUDY_WORDING_SOURCES,
        "quote": criterion.quote,
        "disposition": item.disposition,
        "reason": item.reason,
    }


def population_approval_findings(
    population: CompiledPopulation,
) -> list[ValidationFinding]:
    """Why the plan cannot be approved: the inclusions its cohort does not apply.

    They do not stop planning (population spec design 3.3 and 3.5).  The plan
    lists them as unapplied and the researcher reviews it with these typed
    reasons, but approving it would analyse a broader population than the
    study states, so each stop refuses approval (``approval_allowed``), one
    finding per remedy (``POPULATION_APPROVAL_STOPS``).
    """

    findings: list[ValidationFinding] = []
    for disposition, reason in POPULATION_APPROVAL_STOPS.items():
        blocking = [
            item for item in population.blocking if item.disposition == disposition
        ]
        if not blocking:
            continue
        findings.append(
            ValidationFinding(
                validator="population_compile",
                severity="error",
                message=(
                    "This plan cannot be approved: the study includes only the "
                    "stays that meet "
                    + ("a criterion " if len(blocking) == 1 else "criteria ")
                    + f"{_POPULATION_STOP_CAUSES[disposition]}, so the plan would "
                    "analyse a broader population than the study states. "
                    + " ".join(
                        f"{item.criterion.id} {item.criterion.quote!r}: {item.detail}"
                        for item in blocking
                    )
                ),
                evidence_ids=["analysis_plan"],
                detail={
                    "reason": reason,
                    "human_review_required": True,
                    "approval_allowed": False,
                    "criteria": [_criterion_row(item) for item in blocking],
                    "time_zero_hours": population.time_zero_hours,
                    "population_compile_sha256": population.sha256(),
                },
            )
        )
    return findings


def population_proposals_finding(
    population: CompiledPopulation,
) -> ValidationFinding | None:
    """The criteria the study did not state, which the plan proposes.

    A criterion whose source is the Planner's outline or a preset cites no
    words of the researcher's.  It is compiled like the study's own, so this
    record keeps it apart from what the researcher asked for.
    """

    proposed = [
        item
        for item in population.criteria
        if item.criterion.source not in STUDY_WORDING_SOURCES
    ]
    if not proposed:
        return None
    return ValidationFinding(
        validator="population_compile",
        severity="warning",
        message=(
            "The plan proposes population criteria the study does not state: "
            + "; ".join(
                f"{item.criterion.id} {item.criterion.quote!r} "
                f"(from the {item.criterion.source}, {item.disposition})"
                for item in proposed
            )
            + "."
        ),
        evidence_ids=["analysis_plan"],
        detail={
            "reason_code": "population_criteria_proposed_by_system",
            "criteria": [_criterion_row(item) for item in proposed],
        },
    )


def registered_stop(finding: ValidationFinding) -> ValidationFinding:
    """A finding that refuses approval names a stop every plan-review reader knows.

    The readers (the conversation's workflow, the run projection, the agent
    route) spread ``approval_stops.PLAN_APPROVAL_STOPS``; a stop missing from
    it would leave a plan that cannot be approved looking ready, so planning
    stops with a typed reason instead.
    """

    detail = finding.detail or {}
    if (
        detail.get("approval_allowed") is False
        and detail.get("reason") not in PLAN_APPROVAL_STOPS
    ):
        raise ProgressivePlanCompileError(
            "progressive_approval_stop_unregistered",
            f"approval stop {detail.get('reason')!r} is not in "
            "approval_stops.PLAN_APPROVAL_STOPS",
            path="plan_review",
        )
    return finding


def refuse_substance_form_substitution(
    *, context: ResearchContext, plan: AnalysisPlan
) -> None:
    """Stop a plan whose primary exposure reads a named substance in its other form.

    The question named albumin given, and the plan studies the serum level
    (``planning.question_substance_forms``): it answers another question, so
    planning stops with a typed reason rather than analyse it.
    """

    try:
        forms = question_substance_forms(context)
    except QuestionSubstanceFormError as exc:
        raise ProgressivePlanCompileError(
            f"progressive_{exc.reason_code}",
            str(exc),
            path=QUESTION_SUBSTANCE_FORMS_KEY,
        ) from exc
    substituted = substance_form_substitutions(
        forms, plan=plan, relatives=concept_relatives(context)
    )
    if substituted:
        raise ProgressivePlanCompileError(
            f"progressive_{SUBSTITUTED_REASON}",
            " ".join(item.message() for item in substituted),
            path=QUESTION_SUBSTANCE_FORMS_KEY,
        )


def question_requirement_outcome(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    facts: Any,
    run_dir: Path,
    planning_contract_context: str = "",
) -> list[ValidationFinding]:
    """Judge the question's requirements on the compiled plan and record them.

    The family route (``facts.family_template``) states its requirements in
    the spec; its templates have no subgroup step, and only the prediction
    template draws a comparing step, so the host decides the other gaps.  The outline route states none: each concept the
    question names, other than a sealed coordinate, is a warning there.  A gap
    the Planner declared is checked against the study as the spec parser
    checked it (``planning.capability_gap``).  The record is a run fact beside
    the plan, not evidence; it carries what the judgment read of the study, so
    the plan a review request offers is judged again from it
    (``question_requirements_on_plan_under_review``).
    """

    family_template = bool(facts.family_template)
    requirements = tuple(facts.question_requirements)
    relatives = concept_relatives(context)
    try:
        stated = named_question_concepts(context)
    except QuestionRequirementsError as exc:
        raise ProgressivePlanCompileError(
            f"progressive_{exc.reason_code}", str(exc), path="question_named_concepts"
        ) from exc
    named = bind_named_question_concepts(
        stated,
        roster=[str(variable.name) for variable in context.variables],
        relatives=relatives,
    )
    if not requirements and not named:
        return []
    sealed = tuple(
        value
        for value in (context.primary_exposure or "", context.target_outcome or "")
        if value
    )
    judged = judge_question_requirements(
        requirements,
        plan=plan,
        family_template=family_template,
        relatives=relatives,
        check_gap=lambda gap: check_capability_gap(
            gap, context=context, planning_contract_context=planning_contract_context
        ),
    )
    unstated = (
        ()
        if family_template
        else outline_route_unstated(named, plan=plan, sealed=sealed)
    )
    write_question_requirements(
        run_dir,
        question_requirements_record(
            route="family_template" if family_template else "outline",
            plan=plan,
            requirements=requirements,
            named=named,
            judged=judged,
            unstated=unstated,
            coverage=question_requirement_coverage(judged, context=context, plan=plan),
            denoted={
                concept: relatives(concept)
                for item in requirements
                for concept in item.concepts
            },
            sealed=sealed,
        ),
    )
    return question_requirement_findings(judged, unstated=unstated)


def run_progressive_planner(
    *,
    planner: ProgressivePlannerAgent,
    context: ResearchContext,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    prompt_pack_version: str,
    resume_checkpoint_path: str | Path | None,
    resume_checkpoint_sha256: str | None,
    cohort_path: Path,
    llm_signature: str,
    planner_kwargs: Mapping[str, Any],
    know_how_binding: PlannerKnowHowBinding,
    planning_contract_context: str,
    finding_sink: Callable[[ValidationFinding], None],
    stop_after_outline: bool = False,
) -> ProgressivePlannerRunResult | ProgressiveDesignCanaryDraft:
    """Run Progressive v2 while importing only a host-validated Dev chain."""

    source_chain = (
        load_progressive_planner_checkpoint_chain(
            last_checkpoint_path=Path(resume_checkpoint_path).expanduser(),
            expected_artifact_sha256=str(resume_checkpoint_sha256 or ""),
        )
        if resume_checkpoint_path is not None
        else ()
    )
    recorder = ProgressivePlannerCheckpointRecorder(
        run_dir=run_dir,
        evidence=evidence,
        prompt_pack_version=prompt_pack_version,
        source_chain=source_chain,
    )
    reserved = {
        "checkpoint_callback",
        "resume_checkpoint",
        "resume_dependency_context",
    }
    overlap = sorted(reserved & set(planner_kwargs))
    if overlap:
        raise ValueError(
            "progressive orchestration owns replay kwargs: " + ", ".join(overlap)
        )

    try:
        attempt = planner.run_attempt(
            context,
            **dict(planner_kwargs),
            checkpoint_callback=recorder.record,
            resume_checkpoint=source_chain[-1] if source_chain else None,
            resume_dependency_context={
                "cohort_file_sha256": sha256_of_file(cohort_path),
                "llm_signature": llm_signature,
                "prompt_version": prompt_pack_version,
            },
            stop_after_outline=stop_after_outline,
        )
    except BaseException as error:
        facts = progressive_planner_failure_facts(error)
        if source_chain and facts.resume_validated:
            receipt = recorder.persist_validated_resume()
            finding_sink(
                _resume_finding(
                    receipt=receipt,
                    terminal_artifact_sha256=str(resume_checkpoint_sha256),
                )
            )
        if facts.compile_failure_attempts and recorder.latest_checkpoint is not None:
            persist_progressive_compile_failure_replay(
                run_dir=run_dir,
                evidence=evidence,
                attempts=facts.compile_failure_attempts,
                prefix_checkpoint=recorder.latest_checkpoint,
                prompt_pack_version=prompt_pack_version,
            )
        raise

    generated = attempt.output
    facts = attempt.facts
    if source_chain:
        if not facts.resume_validated:
            raise RuntimeError(
                "Progressive Planner returned without validating its development "
                "resume checkpoint."
            )
        receipt = recorder.persist_validated_resume()
        finding_sink(
            _resume_finding(
                receipt=receipt,
                terminal_artifact_sha256=str(resume_checkpoint_sha256),
            )
        )
    prompt_metrics = know_how_binding.prompt_metrics(
        planner,
        context,
        planning_contract_context=planning_contract_context,
        run_prompt_metrics=facts.prompt_metrics,
    )
    if stop_after_outline:
        if not isinstance(generated, ProgressivePlanOutline):
            raise RuntimeError(
                "Progressive outline-only canary returned an executable plan"
            )
        checkpoint = recorder.latest_checkpoint
        if checkpoint is None or checkpoint.stage != "outline":
            raise RuntimeError(
                "Progressive outline-only canary has no validated outline checkpoint"
            )
        return ProgressiveDesignCanaryDraft(
            outline=generated,
            checkpoint=checkpoint,
            generation_mode=(
                "llm_progressive_v2_design_canary_dev_resume"
                if source_chain
                else "llm_progressive_v2_design_canary"
            ),
            prompt_metrics=prompt_metrics,
        )
    if not isinstance(generated, AnalysisPlan):
        raise RuntimeError("Progressive Planner returned no executable AnalysisPlan")
    persist_progressive_planner_output(
        facts=facts,
        run_dir=run_dir,
        evidence=evidence,
        prompt_metrics=prompt_metrics,
        prompt_pack_version=prompt_pack_version,
    )
    # The population audit never raises.  A spec decided the plan's cohort
    # (step 2b); predicates the Planner wrote beside it that differ are a
    # typed finding, never a silent choice between the two.
    audit = population_cohort_audit(
        context=context, plan=generated, cohort=facts.skeleton.cohort
    )
    write_population_audit(run_dir, audit)
    superseded = superseded_predicates_finding(audit)
    if superseded is not None:
        finding_sink(superseded)
    # The population that decided the cohort, compiled as it was: an
    # inclusion the cohort does not apply refuses approval, never planning.
    population = stated_population(
        facts.skeleton.cohort, context=context, plan=generated
    )
    if population is not None:
        for finding in population_approval_findings(population):
            finding_sink(registered_stop(finding))
        proposals = population_proposals_finding(population)
        if proposals is not None:
            finding_sink(proposals)
    refuse_substance_form_substitution(context=context, plan=generated)
    # What the question asks beyond the design: an analysis the plan does not
    # answer, or cannot, refuses approval, never planning.
    for finding in question_requirement_outcome(
        context=context,
        plan=generated,
        facts=facts,
        run_dir=run_dir,
        planning_contract_context=planning_contract_context,
    ):
        finding_sink(registered_stop(finding))
    return ProgressivePlannerRunResult(
        plan=generated,
        generation_mode=(
            "llm_progressive_v2_dev_resume"
            if source_chain
            else "llm_progressive_v2"
        ),
        prompt_metrics=prompt_metrics,
        facts=facts,
    )


def persist_progressive_planner_output(
    *,
    facts: ProgressivePlannerRunFacts,
    run_dir: Path,
    evidence: ProgressiveEvidenceRegistrar,
    prompt_metrics: Mapping[str, Any],
    prompt_pack_version: str,
) -> None:
    """Persist the complete outline-to-compiler chain from one Planner run."""

    if not facts.complete_for_persistence:
        raise RuntimeError(
            "Progressive Planner returned without its outline, foundation, step "
            "materializations, skeleton, or compile receipt"
        )
    persist_progressive_planning_artifacts(
        run_dir=run_dir,
        evidence=evidence,
        outline=facts.outline,
        foundation=facts.foundation,
        materializations=facts.materializations,
        skeleton=facts.skeleton,
        compile_receipt=facts.compile_receipt,
        prompt_metrics=prompt_metrics,
        prompt_pack_version=prompt_pack_version,
    )


__all__ = [
    "finalize_progressive_design_canary",
    "ProgressiveDesignCanaryDraft",
    "ProgressivePlannerRunResult",
    "persist_progressive_planner_output",
    "population_approval_findings",
    "question_requirement_outcome",
    "registered_stop",
    "population_proposals_finding",
    "run_progressive_planner",
    "run_pipeline_progressive_planner",
]

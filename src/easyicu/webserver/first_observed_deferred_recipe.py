"""Outcome-blind review-only first-observed preparation proposal."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from easyicu.databases.research_plan_requirements import _constraints
from easyicu.research_agent.canonical_json import canonical_json_bytes, canonical_sha256
from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.planning.dependence_authority import host_independent_units_scope
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.population_compile import host_first_observed_recipe_scope
from easyicu.research_agent.planning.progressive_compiler import _compiled_population, compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.research_agent.planning.question_interpretation import requested_uncertainty
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver.durable_research_store import runtime_digest
from easyicu.webserver.first_observed_preparation import _metadata_source, _readonly_evidence_record
from easyicu.webserver.first_observed_run_guard import (
    resolve_host_prepared_first_observed_coordinate, verify_first_observed_preparation_recipe,
)
from easyicu.webserver.raw_preparation_dependencies import (
    _candidate_plan_input_dependencies, _declared_typed_export_columns, declared_native_export_inventory,
)


def _project_review_interval_text(*, original, plan):
    """Project only the exact host counts-only template onto a verified CI plan."""
    old_steps = [step for step in original.steps if step.exposure_outcome_distribution_spec is not None]
    new_steps = [step for step in plan.steps if step.exposure_outcome_distribution_spec is not None]
    old_ids = [step.step_id for step in old_steps]
    new_ids = [step.step_id for step in new_steps]
    if (len(old_ids) != len(set(old_ids)) or len(new_ids) != len(set(new_ids))
            or set(old_ids) != set(new_ids)):
        raise ValueError("first_observed_recipe_interval_changed")
    actual = {step.step_id: step.exposure_outcome_distribution_spec for step in new_steps}
    for step in old_steps:
        prior = step.exposure_outcome_distribution_spec
        current = actual[step.step_id]
        if (current.interval_method != prior.interval_method
                or current.confidence_level != prior.confidence_level):
            raise ValueError("first_observed_recipe_interval_changed")
    selection = original.design_selection
    if selection is None:
        if plan.design_selection is not None:
            raise ValueError("first_observed_recipe_design_changed")
        return plan
    old = selection.selected.primary_method
    suffix = "; no adjusted model, no inferential contrast, and no confidence interval, standard error, or p-value (typed counts-only design)."
    prefixes = {"Counts and proportions by exposure level", "Counts and proportions by exposure level, plus a baseline table by level"}
    if not any(old == prefix + suffix for prefix in prefixes):
        if plan.design_selection != selection:
            raise ValueError("first_observed_recipe_design_changed")
        return plan
    old_steps = [step for step in original.steps if step.step_id == "exposure_outcome_distribution"]
    new_steps = [step for step in plan.steps if step.step_id == "exposure_outcome_distribution"]
    if len(old_steps) != 1 or len(new_steps) != 1:
        raise ValueError("first_observed_recipe_interval_step_changed")
    previous, current = old_steps[0], new_steps[0]
    dist = current.exposure_outcome_distribution_spec
    original_dist = previous.exposure_outcome_distribution_spec
    if (original_dist is None or dist is None
            or original_dist.interval_method != dist.interval_method
            or original_dist.confidence_level != dist.confidence_level):
        raise ValueError("first_observed_recipe_interval_changed")
    objective_suffix = "; no confidence interval or p-value under the typed counts-only design."
    if (dist is None or not previous.intent.startswith("Describe the distribution of ")
            or not previous.intent.endswith("within each level, with denominators" + objective_suffix)):
        raise ValueError("first_observed_recipe_interval_unverified")
    if dist.interval_method in (None, "none_counts_only"):
        if (dist.confidence_level is not None or plan.design_selection != selection
                or current.intent != previous.intent):
            raise ValueError("first_observed_recipe_interval_unverified")
        return plan
    if dist.confidence_level is None or not 0 < dist.confidence_level < 1:
        raise ValueError("first_observed_recipe_interval_unverified")
    level = f"{dist.confidence_level * 100:g}%"
    method = str(dist.interval_method).replace("_", " ")
    revised_method = old.removesuffix(suffix).replace(
        "Counts and proportions by exposure level",
        f"Counts and proportions by exposure level with {level} {method} confidence intervals", 1,
    ) + "; no adjusted model and no inferential contrast is proposed."
    revised_intent = previous.intent.removesuffix(objective_suffix) + f" and {level} {method} confidence intervals."
    expected = selection.model_copy(update={"candidates": [
        candidate.model_copy(update={"primary_method": revised_method})
        if candidate.design_id == selection.selected.design_id else candidate
        for candidate in selection.candidates
    ]})
    if plan.design_selection != selection and plan.design_selection != expected:
        raise ValueError("first_observed_recipe_design_changed")
    if current.intent not in (previous.intent, revised_intent):
        raise ValueError("first_observed_recipe_interval_step_changed")
    projected = plan.model_copy(deep=True)
    projected.design_selection = expected
    next(step for step in projected.steps if step.step_id == current.step_id).intent = revised_intent
    return projected


def _first_observed_review_criterion(spec):
    """Keep the original adult criterion and an optional redundant age >= 18."""
    first = [item for item in spec.criteria
             if item.kind == "first_observed_icu_single_hospital_adult" and item.role == "include"]
    extra = [item for item in spec.criteria if item not in first]
    if (len(first) != 1 or len(extra) > 1 or any(
        item.kind != "age_years" or item.role != "include"
        or item.min_years != 18 or item.max_years is not None for item in extra
    )):
        raise ValueError("first_observed_recipe_population_invalid")
    return first[0]


def _validate_first_observed_outcome_policy(distributions, required):
    if {"sex", "death"}.issubset(required) and any(
        item.missing_outcome_policy not in {"exclude_from_denominator", "fail_closed"}
        or (item.missing_outcome_policy == "exclude_from_denominator"
            and item.denominator_policy != "observed_outcome_rows")
        for item in distributions
    ):
        raise ValueError("first_observed_recipe_unknown_outcome_policy_invalid")


def _planning_only_structural_columns(*, context, plan, declared_files):
    """Ignore two known planning-only fields only when the plan never uses them."""
    if not declared_files or any(item.get('identity_column') != 'stay_id' for item in declared_files):
        raise ValueError('first_observed_recipe_identity_unbound')
    structural = {'stay_id'}
    variables = {item.name: item for item in context.variables}
    plan_text = json.dumps(plan.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)
    alias = variables.get('patientunitstayid')
    if alias is not None:
        if (getattr(alias.role, 'value', alias.role) != 'id'
                or context.cohort.id_columns != ['patientunitstayid']
                or 'patientunitstayid' in plan_text):
            raise ValueError('first_observed_recipe_identity_unbound')
        structural.add('patientunitstayid')
    if 'icu_stay_is_readmission' in variables:
        if 'icu_stay_is_readmission' in plan_text:
            raise ValueError('first_observed_recipe_required_columns_missing')
        structural.add('icu_stay_is_readmission')
    return tuple(sorted(structural))


def _first_observed_static_base(required, declared_files, target_concept, demographic_concepts=()):
    static_available = {entry["source_concept"] for item in declared_files if not item["time_columns"]
                        for entry in item["declared_columns"]}
    declared = {entry["source_concept"] for item in declared_files for entry in item["declared_columns"]}
    static_available.update(declared & ({"age"} | set(demographic_concepts)))
    # Outcome availability must never determine the stay base: unknown
    # hospital outcomes remain rows and are excluded only from known denominators.
    selected = (set(required) - {target_concept}) & static_available
    # Age is part of this exact adult criterion and is known for every kept
    # source patient. Prefer it as a base when the export declares it static,
    # so a missing sex value cannot remove a patient before analysis.
    if "age" in static_available:
        selected.add("age")
    return sorted(selected)


@dataclass(frozen=True)
class FirstObservedDeferredReviewStage:
    """Server-owned no-Provider replay of one completed preparation's skeleton."""

    preparation_run_id: str
    study_context_id: str
    project_id: str
    session_id: str
    skeleton_sha256: str

    def _skeleton_and_source(self, context: Any):
        from easyicu.webserver import agent_pipeline_runs

        pending, study = agent_pipeline_runs.raw_preparation_review_owner(self.preparation_run_id, self.study_context_id)
        from easyicu.webserver import study_contexts

        current = study_contexts.get_context(self.study_context_id)
        if (current is None or study_contexts.scientific_configuration_sha256(current)
                != study_contexts.scientific_configuration_sha256(study)):
            raise ValueError("first_observed_recipe_study_changed")
        provenance = context.cohort.provenance or {}
        if (provenance.get("evidence_stage") != "metadata_only_planning"
                or provenance.get("patient_rows_read") is not False or context.cohort.n_stays != 0):
            raise ValueError("first_observed_recipe_not_metadata_planning")
        with AnchoredDirectory.open(Path(pending.run_dir)) as directory:
            checkpoint = RawPreparationCheckpoint.model_validate_json(
                directory.read_bytes("raw_preparation_checkpoint.json", max_bytes=1024 * 1024))
            raw = directory.read_bytes("progressive_plan_skeleton.json", max_bytes=16 * 1024 * 1024)
            directory.assert_still_selected()
        authority = PlanReviewAuthority.model_validate(checkpoint.request.payload["plan_review_authority"])
        if (hashlib.sha256(raw).hexdigest() != self.skeleton_sha256
                or authority.evidence_sha256.get("progressive_plan_skeleton") != self.skeleton_sha256):
            raise ValueError("first_observed_recipe_skeleton_changed")
        if authority.plan_payload.get("research_question") != context.research_question:
            raise ValueError("first_observed_recipe_question_changed")
        skeleton = ProgressivePlanSkeleton.model_validate_json(raw)
        if skeleton.cohort.population_spec is None:
            raise ValueError("first_observed_recipe_population_invalid")
        spec = PopulationSpec.model_validate(skeleton.cohort.population_spec)
        _first_observed_review_criterion(spec)
        source = SourceIdentity.from_record(_constraints(context).get("source_identity"))
        if source is None or source.database != "eicu_demo":
            raise ValueError("first_observed_recipe_source_invalid")
        _, metadata = _metadata_source(source)
        return skeleton, spec, source, metadata["export_manifest_sha256"]

    @contextmanager
    def review_scope(self, *, context: Any):
        skeleton, spec, source, manifest = self._skeleton_and_source(context)
        first = _first_observed_review_criterion(spec)
        criterion = first.model_dump(mode="json")

        def verify(candidate, selected):
            if candidate is not context or selected.model_dump(mode="json") != criterion:
                raise ValueError("first_observed_recipe_population_changed")
            return asdict(verify_first_observed_preparation_recipe(
                preparation_run_id=self.preparation_run_id, study_context_id=self.study_context_id,
                project_id=self.project_id, session_id=self.session_id, source_identity=source,
                criterion=criterion, export_manifest_sha256=manifest))

        # Verify before exposing even the planning-only independent-row scope.
        verify(context, first)
        def verify_independence(candidate):
            verify(candidate, first)
            return True

        with host_first_observed_recipe_scope(verify), host_independent_units_scope(verify_independence):
            yield

    def compile_plan(self, *, context: Any, allowed_literature_citation_keys):
        skeleton, _, _, _ = self._skeleton_and_source(context)
        with self.review_scope(context=context):
            plan, _ = compile_progressive_plan(skeleton=skeleton, context=context,
                allowed_literature_citation_keys=allowed_literature_citation_keys)
        return plan

    def normalize_review_interval_text(self, *, context: Any, plan: Any):
        skeleton, _, _, _ = self._skeleton_and_source(context)
        with self.review_scope(context=context):
            original, _ = compile_progressive_plan(
                skeleton=skeleton, context=context,
                allowed_literature_citation_keys=sorted({key for step in plan.steps for key in step.literature_citation_keys}),
            )
            return _project_review_interval_text(original=original, plan=plan)

    def reuse_preplan_literature(self, *, context: Any, run_dir: Path, evidence: Any):
        """Copy only the original review-bound literature bytes into this stage."""
        from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
        from easyicu.research_agent.literature import LiteratureBundle
        from easyicu.webserver import agent_pipeline_runs
        from easyicu.webserver.raw_preparation_decisions import _publish_once

        skeleton, _, _, _ = self._skeleton_and_source(context)
        pending, _ = agent_pipeline_runs.raw_preparation_review_owner(self.preparation_run_id, self.study_context_id)
        source = Path(pending.run_dir)
        with AnchoredDirectory.open(source) as directory:
            checkpoint = RawPreparationCheckpoint.model_validate_json(
                directory.read_bytes('raw_preparation_checkpoint.json', max_bytes=1024 * 1024))
            directory.assert_still_selected()
        authority = PlanReviewAuthority.model_validate(checkpoint.request.payload['plan_review_authority'])
        digest = authority.evidence_sha256.get('preplan_literature_bundle')
        if digest is None:
            if any(step.literature_bindings for step in skeleton.steps):
                raise ValueError('first_observed_recipe_literature_missing')
            return None
        record = _readonly_evidence_record(source, 'preplan_literature_bundle')
        path = verified_run_evidence_path(source, record) if record is not None else None
        if path is None or record.sha256 != digest:
            raise ValueError('first_observed_recipe_literature_changed')
        with AnchoredDirectory.open(path.parent) as directory:
            raw = directory.read_bytes(path.name, max_bytes=16 * 1024 * 1024, expected_sha256=digest)
            directory.assert_still_selected()
        with AnchoredDirectory.open(source) as directory:
            working = directory.read_bytes('preplan_literature_bundle.json', max_bytes=16 * 1024 * 1024,
                                           expected_sha256=digest)
            directory.assert_still_selected()
        if working != raw:
            raise ValueError('first_observed_recipe_literature_changed')
        bundle = LiteratureBundle.model_validate_json(raw)
        if bundle.research_question != context.research_question:
            raise ValueError('first_observed_recipe_literature_question_changed')
        target = Path(run_dir)
        with AnchoredDirectory.open(target) as directory:
            _publish_once(directory, 'preplan_literature_bundle.json', raw)
            directory.assert_still_selected()
        evidence.register_file(kind='log', description='Original review-bound preplan literature',
            source_path=target / 'preplan_literature_bundle.json', evidence_id='preplan_literature_bundle',
            producer='host', generation_mode='system')
        return bundle

    def prepare_review(self, *, context: Any, agent_context: Any, normalized_lineage: Any,
                       run_dir: Path, evidence: Any) -> None:
        from easyicu.webserver.raw_preparation_decisions import _publish_once

        _, spec, _, _ = self._skeleton_and_source(context)
        self._skeleton_and_source(agent_context)
        agent_raw = agent_context.model_dump_json().encode("utf-8")
        with AnchoredDirectory.open(Path(run_dir)) as directory:
            _publish_once(directory, "first_observed_planning_context.json", agent_raw)
            directory.assert_still_selected()
        evidence.register_file(kind="log", description="Exact metadata-only context used for no-send replay",
            source_path=Path(run_dir) / "first_observed_planning_context.json",
            evidence_id="first_observed_planning_context", producer="host", generation_mode="system")
        result = build_pending_first_observed_review(
            preparation_run_id=self.preparation_run_id, study_context_id=self.study_context_id,
            project_id=self.project_id, session_id=self.session_id, context=context,
            plan=normalized_lineage.analysis_plan(), population_spec=spec,
            skeleton_sha256=self.skeleton_sha256, normalized_lineage=normalized_lineage,
            planning_context=agent_context,
        )
        if result["ready_for_human_approval"] is not True:
            raise ValueError("first_observed_recipe_review_not_ready")
        raw = canonical_json_bytes(result["recipe"])
        if hashlib.sha256(raw).hexdigest() != result["recipe_sha256"]:
            raise ValueError("first_observed_recipe_digest_invalid")
        with AnchoredDirectory.open(Path(run_dir)) as directory:
            _publish_once(directory, "first_observed_deferred_recipe.json", raw)
            directory.assert_still_selected()
        evidence.register_file(kind="log", description="Pending first-observed data preparation recipe",
            source_path=Path(run_dir) / "first_observed_deferred_recipe.json", evidence_id="first_observed_deferred_recipe",
            producer="host", generation_mode="system")

    def review_findings(self, *, context: Any, normalized_lineage: Any, run_dir: Path, evidence: Any):
        from easyicu.research_agent.orchestration.progressive_planning import population_pending_review_findings

        _, spec, _, _ = self._skeleton_and_source(context)
        plan = normalized_lineage.analysis_plan()
        with self.review_scope(context=context):
            population = _compiled_population(spec, context=context, plan=plan)
            return population_pending_review_findings(
                population, context=context, plan=plan, run_dir=Path(run_dir), evidence=evidence,
            )

    def persist_replayed_progressive_evidence(self, *, context: Any, run_dir: Path,
                                              evidence: Any, prompt_pack_version: str) -> dict[str, Any]:
        """Rebuild the original artifact chain with a freshly compiled receipt."""
        from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
        from easyicu.research_agent.planning.progressive_artifacts import persist_progressive_planning_artifacts
        from easyicu.research_agent.planning.progressive_contract import (
            ProgressiveFoundationMaterialization, ProgressivePlanOutline, ProgressiveStepMaterialization,
        )
        from easyicu.webserver import agent_pipeline_runs

        skeleton, _, _, _ = self._skeleton_and_source(context)
        pending, _ = agent_pipeline_runs.raw_preparation_review_owner(self.preparation_run_id, self.study_context_id)
        source_run = Path(pending.run_dir)
        with AnchoredDirectory.open(source_run) as directory:
            checkpoint = RawPreparationCheckpoint.model_validate_json(
                directory.read_bytes("raw_preparation_checkpoint.json", max_bytes=1024 * 1024))
            directory.assert_still_selected()
        authority = PlanReviewAuthority.model_validate(checkpoint.request.payload["plan_review_authority"])

        def source_bytes(evidence_id: str) -> bytes:
            record = _readonly_evidence_record(source_run, evidence_id)
            path = verified_run_evidence_path(source_run, record) if record is not None else None
            if path is None or record.sha256 != authority.evidence_sha256.get(evidence_id):
                raise ValueError("first_observed_replay_source_evidence_changed")
            with AnchoredDirectory.open(path.parent) as directory:
                raw = directory.read_bytes(path.name, max_bytes=16 * 1024 * 1024, expected_sha256=record.sha256)
                directory.assert_still_selected()
            with AnchoredDirectory.open(source_run) as directory:
                working = directory.read_bytes(f"{evidence_id}.json", max_bytes=16 * 1024 * 1024,
                                               expected_sha256=record.sha256)
                directory.assert_still_selected()
            if working != raw:
                raise ValueError("first_observed_replay_source_evidence_changed")
            return raw

        outline = ProgressivePlanOutline.model_validate_json(source_bytes("progressive_plan_outline"))
        foundation = ProgressiveFoundationMaterialization.model_validate_json(source_bytes("progressive_plan_foundation"))
        ledger = json.loads(source_bytes("progressive_step_materializations"))
        metrics_raw = source_bytes("planner_prompt_metrics")
        metrics = json.loads(metrics_raw)
        entries = ledger.get("materializations") if isinstance(ledger, dict) else None
        if not isinstance(entries, list) or not entries or any(not isinstance(item, dict) for item in entries):
            raise ValueError("first_observed_replay_materializations_invalid")
        materializations = [ProgressiveStepMaterialization.model_validate(item["materialization"]) for item in entries]
        original_steps = authority.plan_payload.get("steps", ())
        citations = sorted({key for step in original_steps for key in step.get("literature_citation_keys", ())})
        with self.review_scope(context=context):
            _, compile_receipt = compile_progressive_plan(
                skeleton=skeleton, context=context, allowed_literature_citation_keys=citations)
        persist_progressive_planning_artifacts(run_dir=Path(run_dir), evidence=evidence,
            outline=outline, foundation=foundation, materializations=materializations,
            skeleton=skeleton, compile_receipt=compile_receipt, prompt_metrics=metrics,
            prompt_pack_version=prompt_pack_version)
        if not isinstance(metrics, dict):
            raise ValueError("first_observed_replay_prompt_metrics_invalid")
        return metrics


def build_pending_first_observed_review(
    *, preparation_run_id: str, study_context_id: str, project_id: str,
    session_id: str, context: Any, plan: Any, population_spec: PopulationSpec,
    skeleton_sha256: str, normalized_lineage: Any = None, planning_context: Any = None,
) -> dict[str, Any]:
    """Reverify a preparation and describe a future, still-unapproved read.

    This does not parse patient data, approve findings, or assert that a
    cohort exists. It is specific to the single reviewed first-ICU criterion;
    other populations need their own producer and review contract.
    """
    import re

    if re.fullmatch(r"[0-9a-f]{64}", skeleton_sha256 or "") is None:
        raise ValueError("first_observed_recipe_skeleton_invalid")
    from easyicu.webserver import agent_pipeline_runs

    pending, study = agent_pipeline_runs.raw_preparation_review_owner(preparation_run_id, study_context_id)
    with AnchoredDirectory.open(Path(pending.run_dir)) as directory:
        checkpoint = RawPreparationCheckpoint.model_validate_json(
            directory.read_bytes("raw_preparation_checkpoint.json", max_bytes=1024 * 1024))
        skeleton = directory.read_bytes("progressive_plan_skeleton.json", max_bytes=16 * 1024 * 1024)
        directory.assert_still_selected()
    authority = PlanReviewAuthority.model_validate(checkpoint.request.payload["plan_review_authority"])
    if (hashlib.sha256(skeleton).hexdigest() != skeleton_sha256
            or authority.evidence_sha256.get("progressive_plan_skeleton") != skeleton_sha256):
        raise ValueError("first_observed_recipe_skeleton_changed")
    if authority.plan_payload.get("research_question") != context.research_question:
        raise ValueError("first_observed_recipe_question_changed")
    first_criterion = _first_observed_review_criterion(population_spec)
    saved_skeleton = ProgressivePlanSkeleton.model_validate_json(skeleton)
    if (saved_skeleton.cohort.population_spec is None
            or PopulationSpec.model_validate(saved_skeleton.cohort.population_spec) != population_spec):
        raise ValueError("first_observed_recipe_population_changed")
    provenance = context.cohort.provenance or {}
    if (provenance.get("evidence_stage") != "metadata_only_planning"
            or provenance.get("patient_rows_read") is not False or context.cohort.n_stays != 0):
        raise ValueError("first_observed_recipe_not_metadata_planning")
    if (plan.research_question != context.research_question or plan.cohort is None
            or plan.cohort.inclusion or plan.cohort.exclusion):
        raise ValueError("first_observed_recipe_plan_scope_invalid")
    planning_context = context if planning_context is None else planning_context
    planning_provenance = planning_context.cohort.provenance or {}
    if (planning_context.research_question != context.research_question
            or planning_provenance.get("evidence_stage") != "metadata_only_planning"
            or planning_provenance.get("patient_rows_read") is not False
            or planning_context.cohort.n_stays != 0
            or _constraints(planning_context).get("source_identity") != _constraints(context).get("source_identity")):
        raise ValueError("first_observed_recipe_planning_context_changed")
    if (plan.analysis_type != "descriptive_epidemiology"
            or any(getattr(step, "trajectory_stability_spec", None) is not None
                   or any("trajectory" in str(value).casefold() for value in step.inputs)
                   for step in plan.steps)):
        raise ValueError("first_observed_recipe_unsupported_analysis")
    distributions = [step.exposure_outcome_distribution_spec for step in plan.steps
                     if step.exposure_outcome_distribution_spec is not None]
    from easyicu.research_agent.contracts.exposure_outcome_distribution import ExposureOutcomeDistributionSpec

    try:
        distributions = [ExposureOutcomeDistributionSpec.model_validate(spec.model_dump(mode="json"))
                         for spec in distributions]
    except ValueError as exc:
        raise ValueError("first_observed_recipe_requested_interval_missing") from exc
    uncertainty = requested_uncertainty(context.research_question)
    if (uncertainty.unresolved or not distributions
            or (uncertainty.requested and any(
                spec.interval_method in {None, "none_counts_only"}
                or (uncertainty.confidence_levels and spec.confidence_level not in uncertainty.confidence_levels)
                for spec in distributions))):
        raise ValueError("first_observed_recipe_requested_interval_missing")
    source = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    if source is None or source.database != "eicu_demo":
        raise ValueError("first_observed_recipe_source_invalid")
    criterion = first_criterion.model_dump(mode="json")
    export_path, metadata = _metadata_source(source)
    manifest_sha = metadata["export_manifest_sha256"]
    proof = verify_first_observed_preparation_recipe(
        preparation_run_id=preparation_run_id, study_context_id=study_context_id,
        project_id=project_id, session_id=session_id, source_identity=source,
        criterion=criterion, export_manifest_sha256=manifest_sha,
    )
    coordinate = resolve_host_prepared_first_observed_coordinate(
        preparation_run_id=preparation_run_id, study_context_id=study_context_id,
        project_id=project_id, session_id=session_id, source_identity=source,
        criterion=criterion, export_manifest_sha256=manifest_sha,
    )
    if coordinate.sha256 != proof.coordinate_sha256:
        raise ValueError("first_observed_recipe_coordinate_changed")
    from easyicu.webserver import study_contexts

    current = study_contexts.get_context(study_context_id)
    if (current is None or study_contexts.scientific_configuration_sha256(current)
            != study_contexts.scientific_configuration_sha256(study)):
        raise ValueError("first_observed_recipe_study_changed")
    def verify_recipe(candidate, selected):
        if (candidate is not context and candidate is not planning_context) or selected.model_dump(mode="json") != criterion:
            raise ValueError("first_observed_recipe_population_changed")
        return asdict(verify_first_observed_preparation_recipe(
            preparation_run_id=preparation_run_id, study_context_id=study_context_id,
            project_id=project_id, session_id=session_id, source_identity=source,
            criterion=criterion, export_manifest_sha256=manifest_sha,
        ))

    original_steps = authority.plan_payload.get("steps", ())
    allowed_citations = sorted({key for step in original_steps for key in step.get("literature_citation_keys", ())})
    def verify_independence(candidate):
        verify_recipe(candidate, first_criterion)
        return True

    with host_first_observed_recipe_scope(verify_recipe), host_independent_units_scope(verify_independence):
        population = _compiled_population(population_spec, context=context, plan=plan)
        if (len(population.criteria) != len(population_spec.criteria)
                or any((item.disposition != "pending_materialization" if item.criterion == first_criterion
                        else not (item.disposition == "applied_by_plan" or (
                            item.disposition == "requires_extraction"
                            and item.reason == "population_concept_not_in_export")))
                       for item in population.criteria)):
            raise ValueError("first_observed_recipe_population_unverified")
        compiled, _ = compile_progressive_plan(
            skeleton=saved_skeleton, context=planning_context, allowed_literature_citation_keys=allowed_citations,
        )
    from easyicu.research_agent.authority.plan_lifecycle import NormalizedPlan

    if normalized_lineage is None:
        raise ValueError("first_observed_recipe_normalized_lineage_missing")
    normalized = NormalizedPlan.model_validate(
        normalized_lineage.model_dump(mode="json") if isinstance(normalized_lineage, NormalizedPlan)
        else normalized_lineage
    )
    expected_interval_text = _project_review_interval_text(original=compiled, plan=plan)
    if (normalized.proposed.plan_sha256 != canonical_sha256(compiled.model_dump(mode="json"))
            or normalized.plan_sha256 != canonical_sha256(plan.model_dump(mode="json"))
            or normalized.plan_payload.get("design_selection") != (
                expected_interval_text.design_selection.model_dump(mode="json")
                if expected_interval_text.design_selection is not None else None)
            or any(step.intent != next((item.intent for item in expected_interval_text.steps
                                        if item.step_id == step.step_id), None)
                   for step in plan.steps if step.step_id == "exposure_outcome_distribution")):
        raise ValueError("first_observed_recipe_plan_not_saved_skeleton")
    recovery = agent_pipeline_runs.get_review_recovery_record(preparation_run_id)
    if (recovery is None or not recovery.hard_stop_task_id
            or re.fullmatch(r"[0-9a-f]{64}", recovery.hard_stop_declaration_sha256 or "") is None):
        raise ValueError("first_observed_recipe_budget_unavailable")
    ledger_path = Path(recovery.hard_stop_ledger_path)
    if not ledger_path.is_absolute() or ledger_path.is_symlink():
        raise ValueError("first_observed_recipe_budget_unavailable")
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.webserver import provider_adapter

    limit_authority = provider_adapter.web_research_agent_hard_stop_limits(recovery.budget_mode)
    ledger = ProviderHardStopLedger(path=ledger_path, task_ids=(recovery.hard_stop_task_id,),
        limits=limit_authority, batch_id=recovery.hard_stop_task_id,
        declaration_sha256=recovery.hard_stop_declaration_sha256, resume_existing=True, require_existing=True)
    limits = limit_authority.canonical_payload()
    usage = ledger.task_accounting_summary(recovery.hard_stop_task_id)["conservative_upper_bound"]
    with AnchoredDirectory.open(export_path) as directory:
        manifest_raw = directory.read_bytes("_manifest.json", max_bytes=16 * 1024 * 1024,
                                            expected_sha256=manifest_sha)
        directory.assert_still_selected()
    package_database = json.loads(manifest_raw).get("database")
    if package_database not in {"eicu", "eicu_demo"}:
        raise ValueError("first_observed_recipe_export_database_invalid")
    inventory = declared_native_export_inventory(
        export_path=export_path, expected_manifest_sha256=manifest_sha, expected_database=package_database)
    declared = _declared_typed_export_columns(export_path=export_path, inventory=inventory)
    columns = []
    for item in declared["files"]:
        names = {item["identity_column"], *item["time_columns"],
                 *(column["column"] for column in item["declared_columns"])}
        columns.append({"file": item["file"], "module": item["module"], "columns": sorted(names)})
    source_concepts = {entry["source_concept"] for item in declared["files"]
                       for entry in item["declared_columns"]}
    if not declared["files"] or any(not item.get("identity_column") for item in declared["files"]):
        raise ValueError("first_observed_recipe_identity_unbound")
    stay_variable = next((item for item in context.variables if item.name == "stay_id"), None)
    if stay_variable is not None and getattr(stay_variable.role, "value", stay_variable.role) != "id":
        raise ValueError("first_observed_recipe_identity_unbound")
    structural = _planning_only_structural_columns(context=context, plan=plan, declared_files=declared['files'])
    dependencies = _candidate_plan_input_dependencies(
        plan=plan, context=context, population_spec=population_spec, structural_columns=structural,
    )
    if dependencies["blockers"] or not set(dependencies["required_source_concepts"]).issubset(source_concepts):
        raise ValueError("first_observed_recipe_required_columns_missing")
    variables = {variable.name: variable for variable in context.variables}
    outcomes = {item.outcome for item in distributions}
    exposures = {item.exposure for item in distributions}
    if len(outcomes) != 1 or len(exposures) != 1:
        raise ValueError("first_observed_recipe_multiple_distributions_unsupported")
    outcome_name, exposure_name = next(iter(outcomes)), next(iter(exposures))
    if (context.target_outcome and context.target_outcome != outcome_name
            or context.primary_exposure and context.primary_exposure != exposure_name):
        raise ValueError("first_observed_recipe_analysis_columns_unbound")
    target = variables.get(outcome_name)
    exposure = variables.get(exposure_name)
    if target is None or not target.source_concept or exposure is None or not exposure.source_concept:
        raise ValueError("first_observed_recipe_analysis_columns_unbound")
    constraints = _constraints(context)
    window = constraints.get("materialization_window")
    if (not isinstance(window, dict) or window.get("anchor") != "icu_admission"
            or not isinstance(window.get("hours"), (int, float)) or isinstance(window.get("hours"), bool)
            or not 0 < window["hours"] < float("inf")):
        raise ValueError("first_observed_recipe_window_unbound")
    required = set(dependencies["required_source_concepts"])
    if not {target.source_concept, exposure.source_concept}.issubset(required):
        raise ValueError("first_observed_recipe_analysis_columns_unbound")
    _validate_first_observed_outcome_policy(distributions, required)
    demographics = {item.source_concept for item in context.variables
                    if item.source_concept and getattr(item.role, "value", item.role) == "demographic"}
    static = _first_observed_static_base(required, declared["files"], target.source_concept, demographics)
    if not static:
        raise ValueError("first_observed_recipe_stay_base_unbound")
    try:
        from easyicu.webserver.first_observed_deferred_execution import deferred_reader_contract_version
        reader_contract_verified = deferred_reader_contract_version() == "easyicu.first-observed-reader-closure/1"
    except (ImportError, RuntimeError):
        reader_contract_verified = False
    recipe = {
        "schema_version": "easyicu.first-observed-pending-materialization/1",
        "status": "pending_materialization", "analysis_authorized": False,
        "preparation_proof": asdict(proof), "criterion": criterion,
        **({"population_spec": population_spec.model_dump(mode="json")} if len(population_spec.criteria) > 1 else {}),
        "host_binding": {"project_id": project_id, "session_id": session_id,
                         "study_context_id": study_context_id, "preparation_run_id": preparation_run_id},
        "verified_source_files": {"archive_sha256": coordinate.archive_sha256,
                                  "patient_table_sha256": coordinate.patient_table_sha256},
        "population_status": {"state": "pending_materialization", "applied_to_analysis_input": False,
                              "compilation_sha256": population.sha256()},
        "source_flow_report": {
            "scope": "complete_source_before_other_study_filters",
            "fields": ["source_patients", "source_stays", "eligible_patients",
                       "excluded_patients_by_reason", "excluded_stays_by_reason",
                       "first_icu_order_or_type_unverifiable"],
            "hospitalization_count_available": False,
            "lifetime_first_icu_not_guaranteed": True,
            "disclose_only_after_plan_approval_and_exact_membership": True,
        },
        "skeleton_sha256": skeleton_sha256,
        "normalized_plan_authority_sha256": normalized.authority_sha256,
        "plan_sha256": canonical_sha256(plan.model_dump(mode="json")),
        "context_sha256": canonical_sha256(context.model_dump(mode="json")),
        "planning_context_sha256": canonical_sha256(planning_context.model_dump(mode="json")),
        "materializer_runtime_sha256": runtime_digest(),
        "budget": {"planning_mode": recovery.budget_mode, "execution_mode": "deferred_reviewed",
                   "hard_stop_task_id": recovery.hard_stop_task_id,
                   "hard_stop_declaration_sha256": recovery.hard_stop_declaration_sha256,
                   "recovery_record_sha256": recovery.record_sha256,
                   "limits": limits, "used_calls": usage["n_calls"],
                   "used_tokens": usage["total_tokens"], "used_cost_usd": usage["estimated_cost_usd"],
                   "reset_or_increase_authorized": False},
        "source_manifest_sha256": manifest_sha,
        "snapshot_all_manifest_files": True,
        "snapshot_files": inventory["files"], "metadata_files": inventory["metadata_files"],
        "declared_schema_columns": columns, "sidecar_sha256": declared["sidecar_sha256"],
        "plan_dependencies": dependencies,
        "materialization": {
            "producer": "easyicu.research_agent.acquisition.foundation.acquire_universe_for_question",
            "selection_authority": "host_exact", "database": package_database,
            "target_outcome": target.source_concept,
            "primary_exposure": exposure.source_concept,
            "analysis_columns": {"exposure": exposure_name, "outcome": outcome_name},
            "outcome_concepts": [target.source_concept],
            "required_feature_concepts": sorted(required - {target.source_concept}),
            "static_concepts": static, "allowed_modules": sorted({item["module"] for item in columns}),
            "cohort_window_hours": [0.0, float(window["hours"])],
            "emit_trajectory": False, "additional_population_filters": False,
        },
        "maximum_decoded_scope": "all_columns_in_each_declared_file_schema",
        "decoded_scope_enforced": reader_contract_verified, "actual_cohort_exists": False,
        "same_skeleton_compilation_verified": True,
    }
    if {"sex", "death"}.issubset(required):
        recipe["source_flow_report"]["unknown_outcome_count_required"] = True
        recipe["raw_patient_verification"] = {
            "source": "registered_eicu_demo_patient_table",
            "patient_table_sha256": coordinate.patient_table_sha256,
            "columns": ["patientunitstayid", "gender", "hospitaldischargestatus"],
            "purpose": "post_approval_independent_native_export_value_comparison",
            "preserve_unknown_outcome": True,
            "post_approval_only": True,
        }
    raw_verifier_ready = "raw_patient_verification" not in recipe
    if not raw_verifier_ready:
        try:
            from easyicu.webserver.first_observed_deferred_execution import deferred_raw_patient_validation_version
            raw_verifier_ready = deferred_raw_patient_validation_version() == "easyicu.eicu-demo-raw-value-check/1"
        except (ImportError, RuntimeError):
            pass
    try:
        from easyicu.webserver.first_observed_deferred_execution import deferred_source_flow_report_version
        flow_report_ready = deferred_source_flow_report_version() == "easyicu.first-observed-source-flow-report/1"
    except (ImportError, RuntimeError):
        flow_report_ready = False
    literature = None
    figure_strategy = None
    literature_sha = authority.evidence_sha256.get("preplan_literature_bundle")
    if literature_sha is not None:
        from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
        from easyicu.research_agent.literature import LiteratureBundle

        record = _readonly_evidence_record(Path(pending.run_dir), "preplan_literature_bundle")
        path = verified_run_evidence_path(pending.run_dir, record) if record is not None else None
        if path is None or record.sha256 != literature_sha:
            raise ValueError("first_observed_recipe_literature_changed")
        with AnchoredDirectory.open(path.parent) as directory:
            raw = directory.read_bytes(path.name, max_bytes=16 * 1024 * 1024, expected_sha256=literature_sha)
            directory.assert_still_selected()
        literature = LiteratureBundle.model_validate_json(raw)
        recipe["preplan_literature_sha256"] = literature_sha
    figure_sha = authority.evidence_sha256.get("article_figure_strategy")
    if figure_sha is not None:
        from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
        from easyicu.research_agent.planning.figure_strategy import ArticleFigureStrategy

        record = _readonly_evidence_record(Path(pending.run_dir), "article_figure_strategy")
        path = verified_run_evidence_path(pending.run_dir, record) if record is not None else None
        if path is None or record.sha256 != figure_sha:
            raise ValueError("first_observed_recipe_figure_strategy_changed")
        with AnchoredDirectory.open(path.parent) as directory:
            raw = directory.read_bytes(path.name, max_bytes=16 * 1024 * 1024, expected_sha256=figure_sha)
            directory.assert_still_selected()
        figure_strategy = ArticleFigureStrategy.model_validate_json(raw)
        recipe["figure_strategy_sha256"] = figure_sha
    # This scope describes the *proposed* one-row-per-patient recipe only.
    # It is active solely during this review and does not survive to execution.
    with host_first_observed_recipe_scope(verify_recipe), host_independent_units_scope(verify_independence):
        review = build_plan_scientific_review(context=context, plan=plan, literature=literature,
                                              figure_strategy=figure_strategy)
    return {"recipe": recipe, "recipe_sha256": canonical_sha256(recipe),
            "scientific_review": review.model_dump(mode="json"),
            "ready_for_human_approval": bool(review.approval_allowed and recipe["decoded_scope_enforced"]
                                             and raw_verifier_ready and flow_report_ready),
            "analysis_authorized": False}


def project_pending_first_observed_review(*, run_dir: Path, pending: Any,
                                          study_context_id: str) -> dict[str, Any]:
    """Project a host-reverified recipe for display; never authorize a read."""
    from easyicu.research_agent.research_context.typed import parse_research_context_json
    from easyicu.research_agent.schema import AnalysisPlan

    unavailable = {"schema_version": "easyicu.first-observed-review-scope/1", "verified": False}
    try:
        root = Path(run_dir)
        if str(pending.run_id) != root.name:
            return unavailable
        authorities = [PlanReviewAuthority.model_validate(request.payload["plan_review_authority"])
                       for request in pending.requests]
        if not authorities or any(item != authorities[0] for item in authorities):
            return unavailable
        authority = authorities[0]
        expected = authority.evidence_sha256.get("first_observed_deferred_recipe")
        context_sha = authority.evidence_sha256.get("research_context")
        planning_sha = authority.evidence_sha256.get("first_observed_planning_context")
        if not expected or not context_sha or not planning_sha:
            return unavailable
        with AnchoredDirectory.open(root) as directory:
            raw = directory.read_bytes("first_observed_deferred_recipe.json", max_bytes=4 * 1024 * 1024,
                                       expected_sha256=expected)
            context_raw = directory.read_bytes("research_context.json", max_bytes=16 * 1024 * 1024,
                                               expected_sha256=context_sha)
            planning_raw = directory.read_bytes("first_observed_planning_context.json", max_bytes=16 * 1024 * 1024,
                                                expected_sha256=planning_sha)
            directory.assert_still_selected()
        recipe = json.loads(raw)
        binding = recipe["host_binding"]
        if binding["study_context_id"] != study_context_id:
            return unavailable
        context = parse_research_context_json(context_raw.decode("utf-8"))
        planning_context = parse_research_context_json(planning_raw.decode("utf-8"))
        plan = AnalysisPlan.model_validate(authority.plan_payload)
        spec = PopulationSpec.model_validate(recipe.get("population_spec") or {"criteria": [recipe["criterion"]]})
        from easyicu.research_agent.authority.plan_lifecycle import NormalizedPlan, plan_lifecycle_evidence_id
        from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path

        lineage_id = plan_lifecycle_evidence_id(plan.revision)
        lineage_record = _readonly_evidence_record(root, lineage_id)
        lineage_path = verified_run_evidence_path(root, lineage_record) if lineage_record is not None else None
        if (lineage_path is None or lineage_record.sha256 != authority.evidence_sha256.get(lineage_id)):
            return unavailable
        with AnchoredDirectory.open(lineage_path.parent) as directory:
            lineage_raw = directory.read_bytes(lineage_path.name, max_bytes=16 * 1024 * 1024,
                                               expected_sha256=lineage_record.sha256)
            directory.assert_still_selected()
        lineage = NormalizedPlan.model_validate_json(lineage_raw)
        rebuilt = build_pending_first_observed_review(
            preparation_run_id=binding["preparation_run_id"], study_context_id=study_context_id,
            project_id=binding["project_id"], session_id=binding["session_id"],
            context=context, plan=plan, population_spec=spec, skeleton_sha256=recipe["skeleton_sha256"],
            normalized_lineage=lineage, planning_context=planning_context,
        )
        if (rebuilt["recipe"] != recipe or canonical_sha256(recipe) != expected
                or recipe["plan_sha256"] != authority.plan_sha256):
            return unavailable
        limits = recipe["budget"]["limits"]
        intervals = [step.exposure_outcome_distribution_spec for step in plan.steps
                     if step.exposure_outcome_distribution_spec is not None]
        return {
            **unavailable,
            # This remains false until the host has closed the execution and
            # persisted-review contracts, even if metadata revalidation passed.
            "verified": rebuilt["ready_for_human_approval"] is True,
            "recipe_sha256": expected, "plan_sha256": authority.plan_sha256,
            "question": context.research_question, "source_label": context.cohort.database,
            "source_manifest_sha256": recipe["source_manifest_sha256"],
            "verified_source_files": recipe["verified_source_files"],
            "population_summary": (
                "Adult first observed ICU in one visible hospitalization; unknown order or identity excluded; "
                "not lifetime first ICU. Report source patients and stays, eligible patients, exclusions by reason, "
                "and unverified first-ICU order; total hospitalizations are unavailable."
                + (" Report missing outcomes separately." if recipe["source_flow_report"].get("unknown_outcome_count_required") else "")
            ),
            "interval_summary": "; ".join(
                f"{item.interval_method} ({item.confidence_level:.0%} confidence)"
                if item.confidence_level is not None else str(item.interval_method)
                for item in intervals
            ) or "No interval method was specified.",
            "snapshot_files": [{"file": item["file"], "sha256": None,
                                "hash_status": "computed_after_authorization"}
                               for item in recipe["snapshot_files"]],
            "decoded_files": [*recipe["declared_schema_columns"], *(
                [{"file": "registered eICU Demo patient table", "columns": recipe["raw_patient_verification"]["columns"]}]
                if "raw_patient_verification" in recipe else [])],
            "budget": {"max_calls": limits["max_provider_attempts_per_run"],
                       "max_tokens": limits["max_total_tokens_per_run"],
                       "max_cost_usd": limits["max_estimated_cost_usd_per_batch"],
                       "max_wall_time_seconds": limits["max_wall_clock_seconds_per_task"],
                       "used_calls": recipe["budget"]["used_calls"],
                       "used_tokens": recipe["budget"]["used_tokens"],
                       "used_cost_usd": recipe["budget"]["used_cost_usd"]},
            "pending_materialization": True, "actual_cohort_exists": False, "analysis_authorized": False,
        }
    except Exception:
        return unavailable

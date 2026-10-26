"""Restore a prepared plan's input scope without restoring plan approval.

Budget names are not input identities. A scientific repair of a prepared plan
must retain its sealed cohort, while a fresh candidate remains metadata-only.
This owner joins the existing review, package, checkpoint and input authorities;
it never consumes a review or starts an extraction, Provider or executor.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from easyicu.research_agent.authority.run_input import load_verified_run_input_capsule
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.cohort_summary import (
    DESCRIPTIVE_SUMMARY_ACTION, is_primary_descriptive_summary_step,
)
from easyicu.research_agent.contracts.frozen_payload import thaw_payload
from easyicu.research_agent.planning.accepted_analysis_inputs import AcceptedAnalysisInputs
from easyicu.research_agent.planning.baseline_requirements import (
    AcceptedBaselineRequirements, baseline_requirement_coverage, candidate_baseline_requirements,
)
from easyicu.research_agent.planning.population_requirements import (
    PlanPopulationRequirements, candidate_population_requirements,
)
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.human_review_checkpoint import load_checkpoint
from easyicu.research_agent.orchestration.exposure_grouping_phase import PlanningRunGroupings, planning_run_groupings
from easyicu.research_agent.planning.scientific_review import (
    PlanScientificReview,
    plan_revision_blocker_codes,
    remediation_route_for_finding,
    timing_design_closed,
)
from easyicu.research_agent.planning.requirement_coverage import SUBGROUP_ACTIONS, concept_relatives, stated_subgroup_coverage
from easyicu.research_agent.planning.sensitivity_authority import EXECUTABLE_METHODS_BY_STRATEGY
from easyicu.research_agent.planning.robustness_contract import complete_case_variables
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ResearchContext
from easyicu.webserver import agent_runs, dataio, study_contexts
from easyicu.webserver.agent_review_recovery import load_recovery_seed
from easyicu.webserver.research_input_progress import research_input_state
from easyicu.webserver.research_launch_resume import _slug
from easyicu.webserver.research_launch_scientific import _configured_sensitivity_specs
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.run_record import RunRecordReadError


def _module(step: Mapping[str, Any]) -> str:
    return str(step.get("scientific_action_id") or step.get("method") or "").strip().casefold().split("(", 1)[0].strip()


def _scientific_binding(value: Any) -> Any:
    """Compare scientific coordinates, independent of local ids and references."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key in {
                "requirement_id", "spec_id", "product_id", "figure_id", "description",
                "rationale", "target_outputs", "primary_model_requirement_id",
                "multiplicity_family_id",
            }:
                continue
            bound = _scientific_binding(item)
            if isinstance(bound, list) and key in {
                "model_requirements", "covariates", "model_terms", "products",
                "subgroup_columns", "sensitivity_spec_ids", "phenotyping_feature_columns",
            }:
                bound = sorted(bound, key=lambda item: json.dumps(item, sort_keys=True))
            result[key] = bound
        return result
    if isinstance(value, (list, tuple)):
        return [_scientific_binding(item) for item in value]
    return value


def _step_coordinate(step: Mapping[str, Any], plan: Mapping[str, Any]) -> tuple[str, str, str]:
    bindings = {
        key: value for key, value in step.items()
        if value and (key.endswith("_spec") or key in {
            "model_requirements", "family_primary_result_requirement", "figure_panels",
            "sensitivity_spec_ids", "phenotyping_feature_columns", "population_scope",
        })
    }
    # A corrected dependency or model's input roster is a binding repair.
    # Untyped analyses still have their actual source columns as coordinates.
    if not any(key in bindings for key in {
        "model_requirements", "family_primary_result_requirement", "functional_form_spec",
        "table_one_spec", "exposure_outcome_distribution_spec", "cohort_definition_spec",
        "phenotyping_feature_columns", "phenotype_comparison_spec",
    }):
        bindings["columns"] = sorted(value for value in step.get("inputs", ()) if ":" not in value)
    method = str(step.get("method") or "").strip().casefold().split("(", 1)[0].strip()
    bindings["method"] = method
    try:
        typed_step = AnalysisStep.model_validate(step)
    except ValueError:
        typed_step = None
    if typed_step is not None and is_primary_descriptive_summary_step(typed_step):
        # Review and execution identify this calculation by its closed action
        # and products; the Planner's free-text method is only a description.
        # Keep a shape marker so an unowned step cannot borrow that identity.
        bindings["method"] = DESCRIPTIVE_SUMMARY_ACTION
        bindings["primary_descriptive_summary"] = True
    if step.get("robustness_replay_spec"):
        bindings["robustness_specs"] = plan.get("robustness_specs", ())
    if method in {"subgroup_analysis", "effect_modification"} or _module(step) in SUBGROUP_ACTIONS:
        bindings["subgroup"] = plan.get("subgroup_analysis_spec")
    if method == "e_value":
        bindings["evalue"] = plan.get("evalue_conversion_spec")
    return (
        str(step.get("planned_analysis_role") or "auxiliary"), _module(step),
        json.dumps(_scientific_binding(bindings), sort_keys=True, separators=(",", ":")),
    )


def _repair_inputs_within_source(
    source: Mapping[str, Any], step: Mapping[str, Any], *,
    source_columns: set[str], revised_references: set[str],
) -> bool:
    """Bind existing variables, the declared cohort identity, or revised products."""
    identity = (source.get("cohort_definition_spec") or {}).get("identity_column")
    return all(
        value in revised_references or (":" not in value and (value in source_columns or value == identity))
        for value in set(step.get("inputs", ())) - set(source.get("inputs", ()))
    )


def _existing_step_binding_repair(
    source: Mapping[str, Any], step: Mapping[str, Any], *,
    source_coordinate: tuple[str, str, str], coordinate: tuple[str, str, str],
    revised_references: set[str], non_scientific_columns: set[str],
) -> bool:
    """Ignore transport and display repairs, retaining exact scientific specs.

    In particular, model rosters and ordered levels are not sets of inputs:
    changing them must still pass the question or blocker authority below.
    """
    if source.get("step_id") != step.get("step_id") or source_coordinate[:2] != coordinate[:2]:
        return False
    before, after = (json.loads(item[2]) for item in (source_coordinate, coordinate))
    if before.get("primary_descriptive_summary"):
        old_columns, new_columns = set(before.get("columns", ())), set(after.get("columns", ()))
        # Only typed provenance/identity inputs can be removed here. Retain
        # the measurement and grouping, and never borrow another step's variable.
        if not new_columns <= old_columns or not old_columns - new_columns <= non_scientific_columns:
            return False
    before.pop("columns", None)
    after.pop("columns", None)
    old_panels = before.pop("figure_panels", ())
    new_panels = after.pop("figure_panels", ())
    if before != after:
        return False
    # These are typed FigurePresentationSpec/local lineage fields, not chart
    # grammars or scientific coordinates. Keep the latter exact and consume
    # each old panel at most once, permitting removal and reordering.
    def panel_coordinate(panel: Mapping[str, Any]) -> str:
        return json.dumps({key: value for key, value in panel.items() if key not in {
            "panel_id", "figure_output", "source_products", "placement", "presentation",
        }}, sort_keys=True, separators=(",", ":"))

    remaining_panels = Counter(panel_coordinate(panel) for panel in old_panels)
    for panel in new_panels:
        if (not set(panel.get("source_products", ())) <= revised_references
                or panel.get("figure_output") not in step.get("expected_outputs", ())):
            return False
        bound = panel_coordinate(panel)
        if remaining_panels[bound] < 1:
            return False
        remaining_panels[bound] -= 1
    return True


@dataclass(frozen=True)
class PlanRevisionGuard:
    """Host-only admission of a repair; the source plan is never overwritten.

    This is deliberately separate from scientific review and execution
    authority. A review suggestion cannot authorize optional new science.
    """

    source_plan_json: str
    review: PlanScientificReview
    question_sensitivity_ids: frozenset[str] | None = None
    append_plan_json: str | None = None

    @classmethod
    def for_append(cls, *, plan: Mapping[str, Any], review: PlanScientificReview,
                   expected_plan: Mapping[str, Any], action: str,
                   append_run: bool = False) -> PlanRevisionGuard:
        """Bind an exact host-owned addition without loosening repair admission.

        The append owner must verify the completed run and compile the requested
        steps before calling this method. Scientific-review eligibility is not
        proof of a source run's human approval and grants no new science.
        """
        from easyicu.webserver.research_plan_append import validate_append_structure

        guard = cls.from_review(plan=plan, review=review)
        if not review.approval_allowed:
            raise ResearchPipelineRunError("plan_append_source_not_approvable", "The append source is not approvable.")
        validate_append_structure(plan, expected_plan, action=action, append_run=append_run)
        return cls(guard.source_plan_json, guard.review, append_plan_json=json.dumps(
            expected_plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ))

    @classmethod
    def from_review(cls, *, plan: Mapping[str, Any], review: PlanScientificReview,
                    question_sensitivity_ids: frozenset[str] | None = None) -> PlanRevisionGuard:
        if canonical_sha256(plan) != review.plan_sha256:
            raise ResearchPipelineRunError(
                "plan_revision_source_plan_mismatch", "The reviewed source plan changed.",
            )
        return cls(json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                   review.model_copy(deep=True), question_sensitivity_ids)

    @property
    def previous_plan(self) -> dict[str, Any]:
        return json.loads(self.source_plan_json)

    def prompt_contract(self) -> str:
        if self.append_plan_json is not None:
            return "\n".join((
                "EXACT APPEND SCOPE (host-derived; new approval required):",
                f"- source_plan_sha256: {self.review.plan_sha256}",
                "- Keep every original step and its evidence. Execute only the authorized addition.",
                "- expected_plan_json: " + self.append_plan_json,
            ))
        return "\n".join((
            "REPAIR-OR-REMOVE REVISION SCOPE (host-derived):",
            f"- source_plan_sha256: {self.review.plan_sha256}",
            "- Repair or remove existing analyses. Add a step only to satisfy a typed question requirement or a blocking finding's required remediation.",
            "- Do not add optional sensitivities, functional-form exploration, subgroup analyses, ordinal trends, E-values, imputation, figures or tables.",
            "- source_plan_json: " + self.source_plan_json,
        ))

    def validate_plan(self, *, plan: AnalysisPlan, context: ResearchContext) -> None:
        if self.append_plan_json is not None:
            if canonical_sha256(plan.model_dump(mode="json")) != canonical_sha256(json.loads(self.append_plan_json)):
                raise ResearchPipelineRunError(
                    "plan_append_scope_changed", "The append differs from the exact requested addition; the original plan was kept.",
                )
            return
        previous = self.previous_plan
        revised = plan.model_dump(mode="json")
        remaining = [(step, _step_coordinate(step, previous)) for step in previous["steps"]]
        source_columns = {value for step in previous["steps"] for value in step.get("inputs", ()) if ":" not in value}
        non_scientific_columns = {v.name for v in context.variables if v.role.value in {"id", "index", "meta"}}
        revised_references = {value for step in revised["steps"] for value in (
            step["step_id"], *step.get("expected_outputs", ()),
        )}
        for step, raw in zip(plan.steps, revised["steps"]):
            coordinate = _step_coordinate(raw, revised)
            # Prefer the same identity so an unchanged sibling with identical
            # coordinates cannot consume the source needed for a later repair.
            matches = [index for index, (old, bound) in enumerate(remaining)
                       if _repair_inputs_within_source(
                           old, raw, source_columns=source_columns, revised_references=revised_references,
                       ) and (bound == coordinate or _existing_step_binding_repair(
                           old, raw, source_coordinate=bound, coordinate=coordinate,
                           revised_references=revised_references, non_scientific_columns=non_scientific_columns,
                       ))]
            if matches:
                match = next((index for index in matches if remaining[index][0]["step_id"] == step.step_id), matches[0])
                remaining.pop(match)
                continue
            single = plan.model_copy(update={"steps": [step]})
            if self._question_requires(single, context) or self._blocker_requires(single, context):
                continue
            raise ResearchPipelineRunError(
                "revision_added_unrequested_analysis",
                "The revision added an unrequested analysis; the previous plan was kept.",
                details={"module": coordinate[1], "role": coordinate[0],
                         "step_id": step.step_id, "source_plan_sha256": self.review.plan_sha256,
                         "added_inputs": sorted(set(raw.get("inputs", ())) - {
                             value for old in previous["steps"] if old["step_id"] == step.step_id
                             for value in old.get("inputs", ())
                         }),
                         "review_item_codes": sorted({finding.code for finding in self.review.findings
                             if remediation_route_for_finding(finding) == "agent_plan_revision"})},
            )

    def _question_requires(self, plan: AnalysisPlan, context: ResearchContext) -> bool:
        step = plan.steps[0]
        covered_subgroups = [record for record in stated_subgroup_coverage(context, plan).records if record.status == "covered"]
        if covered_subgroups and not step.functional_form_spec and not step.sensitivity_spec_ids:
            required_columns = set().union(*(concept_relatives(context)(row.concept_identity) for row in covered_subgroups))
            axes = set(plan.subgroup_analysis_spec.subgroup_columns) if plan.subgroup_analysis_spec else set()
            if axes <= required_columns:
                return True
        preferences = context.user_preferences
        specs = {spec.spec_id: spec for spec in preferences.sensitivity_specs} if preferences else {}
        if step.sensitivity_spec_ids and all(
            spec_id in specs
            and (self.question_sensitivity_ids is None or spec_id in self.question_sensitivity_ids)
            and str(step.method or "").strip().casefold() in EXECUTABLE_METHODS_BY_STRATEGY[specs[spec_id].strategy]
            and set(specs[spec_id].execution_variables) <= set(step.inputs)
            and (step.functional_form_spec is None or step.functional_form_spec.target_column in specs[spec_id].execution_variables)
            for spec_id in step.sensitivity_spec_ids
        ):
            return True
        # Required outcome/adjustment repairs stay in the existing model
        # module. Merely reading a required outcome cannot authorize a new
        # sensitivity, subgroup, or rendering module.
        outcomes = {context.target_outcome, *(context.cohort.requested_outcome_columns or ())}
        old_models = [old for old in self.previous_plan["steps"]
                      if old.get("planned_analysis_role") == "primary" and _module(old) == _module(step.model_dump())
                      and str(old.get("method") or "").strip().casefold().split("(", 1)[0].strip()
                      == str(step.method or "").strip().casefold().split("(", 1)[0].strip()]
        # The required-outcome/adjustment path is not permission to add an
        # arbitrary predictor to an existing primary model.
        allowed_covariates = set(preferences.covariates or ()) if preferences else set()
        allowed_covariates.update(value for old in old_models for req in old.get("model_requirements", ())
                                  for value in req.get("covariates", ()))
        allowed_inputs = {value for old in old_models for value in old.get("inputs", ())}
        allowed_inputs.update({context.primary_exposure, *outcomes, *allowed_covariates})
        if (any(":" not in value and value not in allowed_inputs for value in step.inputs)
                or any(not set(req.covariates) <= allowed_covariates or any(
                    term.role == "covariate" and term.name not in allowed_covariates for term in req.model_terms
                ) for req in step.model_requirements)):
            return False
        if (old_models and step.planned_analysis_role == "primary"
                and not step.model_requirements and not step.functional_form_spec
                and not step.sensitivity_spec_ids
                and {context.primary_exposure, context.target_outcome} <= set(step.inputs)):
            return True
        covered = {req["outcome"] for old in old_models for req in old.get("model_requirements", ())}
        covered.update(outcome for outcome in outcomes if any(
            not old.get("model_requirements") and outcome in old.get("inputs", ()) for old in old_models
        ))
        missing_outcomes = outcomes - covered - {context.target_outcome}
        prior_bindings = Counter(
            (req["outcome"], req["exposure_source"], req["analysis_role"])
            for old in old_models for req in old.get("model_requirements", ())
        )
        if step.planned_analysis_role == "primary":
            allowed_bindings = prior_bindings.copy()
            primary_binding = (context.target_outcome, context.primary_exposure, "primary")
            allowed_bindings[primary_binding] = max(1, allowed_bindings[primary_binding])
            allowed_bindings.update((outcome, context.primary_exposure, "secondary") for outcome in missing_outcomes)
        else:
            allowed_bindings = Counter(
                (outcome, context.primary_exposure, step.planned_analysis_role) for outcome in missing_outcomes
            ) if step.planned_analysis_role == "secondary" else Counter()
        current_bindings = Counter((req.outcome, req.exposure_source, req.analysis_role) for req in step.model_requirements)
        return bool(
            old_models and step.model_requirements and not step.functional_form_spec
            and not step.sensitivity_spec_ids
            and current_bindings <= allowed_bindings
        )

    def _blocker_requires(self, plan: AnalysisPlan, context: ResearchContext) -> bool:
        step = plan.steps[0]
        codes = {
            finding.code for finding in self.review.findings
            if finding.severity == "blocker"
            and remediation_route_for_finding(finding) == "agent_plan_revision"
        }
        if "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" in codes and timing_design_closed(plan):
            return True
        if "ACCEPTED_BASELINE_CONTENT_MISSING" in codes and step.table_one_spec is not None:
            return any(table["complete"] for table in baseline_requirement_coverage(context, plan).get("tables", ()))
        if "ARTICLE_CONTENT_ROLES_INCOMPLETE" in codes and step.measurement_audit_spec is not None:
            missing = set((self.review.facts.get("content_roles") or {}).get("missing_roles") or ())
            allowed = set()
            if "cohort" in missing:
                allowed.add("cohort_flow")
            if "quality" in missing:
                allowed.update({"measurement_missingness", "missingness_profile", "measurement_source", "measurement_process"})
            return bool(step.measurement_audit_spec.products) and all(
                product.audit in allowed for product in step.measurement_audit_spec.products
            )
        return bool(
            "MISSING_CATEGORY_WITHOUT_COMPLETE_CASE_SENSITIVITY" in codes
            and step.method == "complete_case_sensitivity"
            and step.planned_analysis_role == "sensitivity"
            and plan.robustness_specs
            and all(spec.axis == "missing" and complete_case_variables(spec)
                    and spec.outcome_override is None and spec.cohort_override is None
                    for spec in plan.robustness_specs)
        )


def load_plan_revision_guard(*, study: Mapping[str, Any], project_root: str | None,
                             source_run_id: str, review: PlanScientificReview) -> PlanRevisionGuard:
    rows = agent_runs.list_run_history(
        study_id=str(study.get("id") or ""), project_root=project_root, limit=100,
    ).get("runs", ())
    row = next((item for item in rows if item.get("run_id") == source_run_id), None)
    if row is not None:
        record = agent_runs.read_run_record(str(row.get("project_dir") or ""))
        if not isinstance(record, RunRecordReadError):
            plan = thaw_payload(record.artifact_payloads.get("agent_plan.json"))
            if isinstance(plan, Mapping):
                return PlanRevisionGuard.from_review(
                    plan=plan, review=review,
                    question_sensitivity_ids=frozenset(spec.spec_id for spec in _configured_sensitivity_specs(study)),
                )
    raise ResearchPipelineRunError(
        "plan_revision_source_plan_missing", "The exact reviewed source plan could not be loaded.",
    )


@dataclass(frozen=True)
class PreparedPlanRevision:
    run_dir: Path
    pipeline_config_sha256: str
    prepared_package_binding: Mapping[str, Any]
    prior_plan_contract: str | None
    required_primary_cohort_selection_mode: str | None
    input_capsule_sha256: str = ""
    budget_mode: str = "full_reviewed"
    failed_execution_replan: bool = False
    baseline_requirements: AcceptedBaselineRequirements | None = None
    population_requirements: PlanPopulationRequirements | None = None
    # The accepted candidate's primary-analysis inputs sealed into the source
    # run's configuration; every repair on this prepared input keeps them.
    analysis_inputs: AcceptedAnalysisInputs | None = None


def load_prepared_plan_revision(
    *,
    study: Mapping[str, Any],
    project_root: str | None,
    source_run_id: str,
) -> PreparedPlanRevision | None:
    """Select sealed inputs for a new plan, never an old execution approval.

    A metadata candidate (or an absent history row) supplies no prepared-input
    authority. The existing candidate/revision owner still validates it later.
    Once a source claims prepared input, missing or drifted seals fail closed;
    they must not silently fall back to a new zero-row catalogue.
    """

    if not source_run_id:
        return None
    from easyicu.webserver.pi_copilot.contracts import EXECUTION_RETRY_REPLAYABLE_GATE_REASONS

    rows = agent_runs.list_run_history(
        study_id=str(study.get("id") or ""),
        project_root=project_root,
        limit=100,
    ).get("runs", ())
    row = next((item for item in rows if item.get("run_id") == source_run_id), None)
    if row is None:
        return None
    record = agent_runs.read_run_record(str(row.get("project_dir") or ""))
    if isinstance(record, RunRecordReadError):
        if row.get("research_input_state") != "prepared":
            return None
        raise ResearchPipelineRunError(
            "prepared_plan_revision_source_invalid",
            "The prepared plan record failed integrity validation.",
        )
    manifest = record.artifact_payloads.get("source_run_manifest.json") or {}
    if manifest.get("research_input_state") != "prepared":
        if row.get("research_input_state") == "prepared":
            raise ResearchPipelineRunError(
                "prepared_plan_revision_source_invalid",
                "The prepared input receipt is missing or inconsistent.",
            )
        return None
    try:
        digest = study_contexts.scientific_configuration_sha256(study)
        if row.get("scientific_configuration_sha256") != digest:
            raise ValueError("study configuration changed")
        review = PlanScientificReview.model_validate(
            record.artifact_payloads.get("scientific_plan_review.json")
        )
        failed_execution_replan = bool(
            review.approval_allowed
            and row.get("run_status") in {"blocked", "failed"}
            and row.get("gate_reason") in EXECUTION_RETRY_REPLAYABLE_GATE_REASONS
        )
        if (review.approval_allowed and not failed_execution_replan) or (
            plan_revision_blocker_codes(review.findings)
        ):
            raise ValueError("source is not a Planner-owned repair")
        root = Path(str(project_root or "")).expanduser().resolve()
        wrapper = Path(str(row.get("project_dir") or "")).expanduser()
        study_root = root / _slug(study.get("id"))
        if (
            wrapper.is_symlink()
            or study_root.is_symlink()
            or wrapper.parent.resolve() != study_root.resolve()
        ):
            raise ValueError("source is outside the exact study")
        if Path(source_run_id).name != source_run_id:
            raise ValueError("invalid run id")
        run_dir = wrapper / "pipeline" / source_run_id
        if run_dir.is_symlink() or run_dir.parent.is_symlink():
            raise ValueError("source is not a regular owned run")
        seed = load_recovery_seed(wrapper)
        if (
            seed is None
            or seed.schema_version
            not in {
                "easyicu.web-review-recovery-seed/3",
                "easyicu.web-review-recovery-seed/4",
            }
            or seed.budget_mode != "full_reviewed"
            or not seed.prepared_package_binding
            or seed.scientific_configuration_sha256 != digest
            or seed.study.get("id") != study.get("id")
            or study_contexts.scientific_configuration_sha256(seed.study) != digest
        ):
            raise ValueError("prepared launch scope is missing or changed")
        config = PipelineConfig.from_recovery_payload(
            seed.pipeline_config,
            expected_digest=seed.pipeline_config_sha256,
        )
        if (
            not config.require_human_plan_review
            or Path(config.workdir).resolve() != run_dir.parent.resolve()
        ):
            raise ValueError("source configuration does not require plan review")
        checkpoint = load_checkpoint(
            run_dir / "human_review_checkpoint.json",
            **({"require_pending": False} if failed_execution_replan else {}),
        )
        if (
            checkpoint.run_id != source_run_id
            or checkpoint.pipeline_config_sha256 != seed.pipeline_config_sha256
        ):
            raise ValueError("source review configuration changed")
        if failed_execution_replan:
            if (
                checkpoint.state != "completed"
                or not checkpoint.approved_decisions
                or any(item.get("decision") != "approved" for item in checkpoint.approved_decisions)
                or checkpoint.execution_start_receipt is None
            ):
                raise ValueError("source is not a terminal approved execution")
            status_path = run_dir / "run_status.json"
            if status_path.is_symlink() or status_path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("invalid execution status")
            gates = json.loads(status_path.read_bytes()).get("gates", {})
            if not (gates.get("failed_steps") or (
                gates.get("execution_complete") is True
                and any(gates.get(name) is False for name in (
                    "artifact_valid", "evidence_complete", "numeric_verified",
                    "analysis_validated", "manuscript_ready",
                ))
            )):
                raise ValueError("source has no failed execution or validation")
        elif checkpoint.approved_decisions or checkpoint.execution_start_receipt is not None:
            raise ValueError("source is not an unconsumed plan review")
        capsule_path = run_dir / "run_input_capsule.json"
        if capsule_path.is_symlink() or capsule_path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("invalid input capsule")
        raw = capsule_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != checkpoint.run_input_capsule_sha256:
            raise ValueError("checkpoint input identity changed")
        capsule = json.loads(raw)
        load_verified_run_input_capsule(
            run_dir=run_dir,
            scientific_identity=capsule["scientific_identity"],
        )
        if research_input_state(run_dir) != "prepared":
            raise ValueError("source is not physically prepared")
        source = study.get("data_source") or {}
        dataio.validate_research_pipeline_source(
            str(source.get("path") or ""),
            database=source.get("database"),
            expected_binding=seed.prepared_package_binding,
        )
        prior_contract = config.bound_plan_revision_contract
        baseline = population = None
        if failed_execution_replan:
            plan = thaw_payload(record.artifact_payloads.get("agent_plan.json"))
            if (
                not isinstance(plan, dict)
                or canonical_sha256(plan) != review.plan_sha256
                or canonical_sha256(checkpoint.plan_handoff["plan"]) != review.plan_sha256
            ):
                raise ValueError("source plan no longer matches its review and approval")
            # Keep the full source plan as input to a NEW Planner pass. Do not
            # truncate its tail or mistake a successful source step for an
            # approval or reusable result in the new run.
            rendered = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if len(rendered.encode("utf-8")) > 256 * 1024:
                raise ValueError("source plan exceeds the replan transport limit")
            prior_contract = "\n".join(filter(None, (
                prior_contract,
                "DIGEST-BOUND FAILED EXECUTION REPLAN (host-derived):",
                f"- source_plan_sha256: {review.plan_sha256}",
                "- Generate a new complete plan using the sealed input and the current runtime; new review is mandatory.",
                "- Preserve the source question, cohort, all outcomes, methods, baseline variables, timing, sensitivity analyses and displays. Disclose any necessary divergence.",
                "- The old approval and partial results grant no execution authority to this new run.",
                "- source_plan_json: " + rendered,
            )))
            import pyarrow.parquet as pq

            columns = pq.read_schema(run_dir / capsule["cohort_relative_path"]).names
            baseline = candidate_baseline_requirements(
                plan=plan, source_plan_sha256=review.plan_sha256,
                selected_concepts=(), catalog_columns=columns,
            )
            population = candidate_population_requirements(plan, review.plan_sha256)
        return PreparedPlanRevision(
            run_dir=run_dir.resolve(),
            pipeline_config_sha256=seed.pipeline_config_sha256,
            prepared_package_binding=dict(seed.prepared_package_binding),
            prior_plan_contract=prior_contract,
            required_primary_cohort_selection_mode=config.required_primary_cohort_selection_mode,
            input_capsule_sha256=checkpoint.run_input_capsule_sha256,
            failed_execution_replan=failed_execution_replan,
            baseline_requirements=baseline,
            population_requirements=population,
            analysis_inputs=(
                AcceptedAnalysisInputs.model_validate(thaw_payload(config.bound_analysis_inputs))
                if config.bound_analysis_inputs is not None
                else None
            ),
        )
    except Exception as exc:
        # No raw paths, source contents or old Provider environment in errors.
        raise ResearchPipelineRunError(
            "prepared_plan_revision_source_invalid",
            "The prepared plan's study, package, review or sealed inputs changed; "
            "no extraction, planning or execution was started.",
        ) from exc


def load_configured_plan_groupings(
    *, study: Mapping[str, Any], project_root: str | None,
) -> PlanningRunGroupings | None:
    """Keep the reviewed groups across a host configuration change.

    Configuration is not candidate approval. Its existing choice receipt
    binds the exact old run to the current study revision, so only that
    run's verified grouping definitions cross this handoff. They must be
    available before materialization and runtime-family selection.
    """

    from easyicu.webserver.pi_copilot.plan_review_progress import matching_progress

    if not project_root:
        return None
    rows = agent_runs.list_run_history(
        study_id=str(study.get("id") or ""), project_root=project_root, limit=100,
    ).get("runs", ())
    for row in rows:
        progress = matching_progress(study, row)
        if (progress is None or progress.choices.get("agent_plan_configuration")
                != "typed_runtime_projection"):
            continue
        try:
            study_root = Path(project_root).resolve() / _slug(study.get("id"))
            wrapper = Path(str(row.get("project_dir") or ""))
            run_id = str(row.get("run_id") or "")
            run_dir = wrapper / "pipeline" / run_id
            if (not run_id or Path(run_id).name != run_id
                    or wrapper.parent.resolve() != study_root.resolve()
                    or any(path.is_symlink() for path in (study_root, wrapper, run_dir.parent, run_dir))):
                raise ValueError("configured plan is outside the exact study")
            paths = [run_dir / name for name in ("human_review_checkpoint.json", "run_input_capsule.json")]
            if any(path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024 for path in paths):
                raise ValueError("configured plan input receipt is not a bounded regular file")
            checkpoint_raw, capsule_raw = (path.read_bytes() for path in paths)
            checkpoint, capsule = json.loads(checkpoint_raw), json.loads(capsule_raw)
            if (checkpoint.get("run_id") != run_id
                    or checkpoint.get("run_input_capsule_sha256") != hashlib.sha256(capsule_raw).hexdigest()):
                raise ValueError("configured plan input receipt changed")
            return planning_run_groupings(run_dir, cohort_sha256=str(capsule.get("cohort_sha256") or ""))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ResearchPipelineRunError(
                "candidate_plan_materialization_authority_invalid",
                "The configured plan's exposure groupings failed their input binding.",
                details={"field": "exposure_groupings"},
            ) from exc
    return None


__all__ = ["PreparedPlanRevision", "load_prepared_plan_revision", "load_configured_plan_groupings"]

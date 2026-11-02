"""Host gate for approved first-observed preparation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from easyicu.research_agent.authority.filesystem import AnchoredDirectory, AuthorityFilesystemError
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.human_review_checkpoint import load_checkpoint
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision
from easyicu.research_agent.planning.population_review import request_allows_approval


def deferred_reader_contract_version() -> str:
    """Identify this installed host reader contract, not an authorization grant.

    The recipe separately binds the current runtime digest and its complete
    source scope; the consumer rechecks those coordinates after approval.
    """
    return "easyicu.first-observed-reader-closure/1"


def deferred_raw_patient_validation_version() -> str:
    """Identify the installed eICU Demo raw-value verifier, not an I/O grant."""
    return "easyicu.eicu-demo-raw-value-check/1"


def deferred_source_flow_report_version() -> str:
    """Identify the installed local source-flow report contract, not a grant."""
    return "easyicu.first-observed-source-flow-report/1"


class DeferredFirstObservedError(ValueError):
    """The deferred preparation cannot be tied to an actual human approval."""


def verify_approved_deferred_review(*, run_dir: Path, approved_plan_sha256: str,
                                    decision_set_sha256: str, _allow_executing: bool = False) -> dict[str, Any]:
    """Verify the durable decision, exact reviewed plan and recipe before reads."""
    root = Path(run_dir)
    checkpoint = load_checkpoint(root / "human_review_checkpoint.json", require_pending=False)
    if (checkpoint.run_id != root.name
            or checkpoint.state not in ({"approved_pending_execution", "executing"} if _allow_executing
                                        else {"approved_pending_execution"})
            or checkpoint.consumed_decision_sha256 != decision_set_sha256
            or not checkpoint.approved_decisions
            or canonical_sha256(list(checkpoint.approved_decisions)) != decision_set_sha256):
        raise DeferredFirstObservedError("deferred_first_observed_approval_missing")
    requests = {request.review_id: request for request in checkpoint.requests}
    decisions = [HumanReviewDecision.model_validate(item) for item in checkpoint.approved_decisions]
    if (len(requests) != len(checkpoint.requests) or len(decisions) != len(requests)
            or {decision.review_id for decision in decisions} != set(requests)
            or any(canonical_sha256(request.payload) != request.authority_sha256 for request in requests.values())
            or any(decision.decision != "approved"
                   or decision.authority_sha256 != requests[decision.review_id].authority_sha256
                   or not request_allows_approval(requests[decision.review_id], decision.population_criterion_acceptances)
                   for decision in decisions)):
        raise DeferredFirstObservedError("deferred_first_observed_approval_mismatch")
    authorities = [PlanReviewAuthority.model_validate(request.payload["plan_review_authority"])
                   for request in checkpoint.requests]
    if (not authorities or any(item != authorities[0] for item in authorities)
            or authorities[0].plan_sha256 != approved_plan_sha256
            or authorities[0].execution is None
            or authorities[0].execution.pipeline_config_sha256 != checkpoint.pipeline_config_sha256
            or authorities[0].execution.capability_activation_sha256 != checkpoint.capability_activation_sha256
            or authorities[0].execution.run_input_capsule_sha256 != checkpoint.run_input_capsule_sha256):
        raise DeferredFirstObservedError("deferred_first_observed_plan_mismatch")
    from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
    from easyicu.research_agent.planning.scientific_review import PlanScientificReview
    from easyicu.webserver.first_observed_preparation import _readonly_evidence_record

    review_sha = authorities[0].evidence_sha256.get("scientific_plan_review")
    review_record = _readonly_evidence_record(root, "scientific_plan_review")
    review_path = verified_run_evidence_path(root, review_record) if review_record is not None else None
    if not review_sha or review_path is None or review_record.sha256 != review_sha:
        raise DeferredFirstObservedError("deferred_first_observed_scientific_review_missing")
    with AnchoredDirectory.open(review_path.parent) as directory:
        review_raw = directory.read_bytes(review_path.name, max_bytes=8 * 1024 * 1024, expected_sha256=review_sha)
        directory.assert_still_selected()
    review = PlanScientificReview.model_validate_json(review_raw)
    if review.plan_sha256 != approved_plan_sha256 or review.approval_allowed is not True:
        raise DeferredFirstObservedError("deferred_first_observed_scientific_review_blocked")
    expected = authorities[0].evidence_sha256.get("first_observed_deferred_recipe")
    if not expected:
        raise DeferredFirstObservedError("deferred_first_observed_recipe_unbound")
    try:
        with AnchoredDirectory.open(root) as directory:
            raw = directory.read_bytes("first_observed_deferred_recipe.json", max_bytes=4 * 1024 * 1024,
                                       expected_sha256=expected)
            directory.assert_still_selected()
    except (AuthorityFilesystemError, OSError) as exc:
        raise DeferredFirstObservedError("deferred_first_observed_recipe_changed") from exc
    try:
        recipe = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise DeferredFirstObservedError("deferred_first_observed_recipe_invalid") from exc
    if (not isinstance(recipe, dict)
            or recipe.get("schema_version") != "easyicu.first-observed-pending-materialization/1"
            or recipe.get("status") != "pending_materialization"
            or recipe.get("plan_sha256") != approved_plan_sha256
            or recipe.get("analysis_authorized") is not False
            or recipe.get("actual_cohort_exists") is not False
            or recipe.get("same_skeleton_compilation_verified") is not True
            or hashlib.sha256(raw).hexdigest() != expected):
        raise DeferredFirstObservedError("deferred_first_observed_recipe_invalid")
    from easyicu.research_agent.research_context.typed import parse_research_context_json

    context_sha = authorities[0].evidence_sha256.get("research_context")
    if not context_sha:
        raise DeferredFirstObservedError("deferred_first_observed_context_missing")
    try:
        with AnchoredDirectory.open(root) as directory:
            context_raw = directory.read_bytes("research_context.json", max_bytes=16 * 1024 * 1024,
                                               expected_sha256=context_sha)
            directory.read_bytes("run_input_capsule.json", max_bytes=64 * 1024 * 1024,
                                 expected_sha256=checkpoint.run_input_capsule_sha256)
            directory.assert_still_selected()
        context = parse_research_context_json(context_raw.decode("utf-8"))
    except (AuthorityFilesystemError, OSError, ValueError) as exc:
        raise DeferredFirstObservedError("deferred_first_observed_context_changed") from exc
    if (canonical_sha256(context.model_dump(mode="json")) != recipe.get("context_sha256")
            or review.context_sha256 != recipe.get("context_sha256")):
        raise DeferredFirstObservedError("deferred_first_observed_context_changed")
    return recipe


def _validate_actual_plan_columns(*, run_dir: Path, material: dict[str, Any], cohort_columns: tuple[str, ...]) -> tuple[str, str]:
    """Require the approved plan's concrete input coordinates in the cohort."""
    from easyicu.research_agent.contracts.declared_product import typed_product
    from easyicu.research_agent.schema import AnalysisPlan

    checkpoint = load_checkpoint(run_dir / "human_review_checkpoint.json", require_pending=False)
    authority = PlanReviewAuthority.model_validate(checkpoint.requests[0].payload["plan_review_authority"])
    plan = AnalysisPlan.model_validate(authority.model_dump(mode="json")["plan_payload"])
    pairs = {(step.exposure_outcome_distribution_spec.exposure, step.exposure_outcome_distribution_spec.outcome)
             for step in plan.steps if step.exposure_outcome_distribution_spec is not None}
    declared = material.get("analysis_columns")
    if not isinstance(declared, dict) or set(declared) != {"exposure", "outcome"}:
        raise DeferredFirstObservedError("deferred_first_observed_analysis_columns_invalid")
    pair = (declared["exposure"], declared["outcome"])
    if pairs != {pair} or not all(isinstance(value, str) and value for value in pair):
        raise DeferredFirstObservedError("deferred_first_observed_analysis_columns_invalid")
    required = {name for step in plan.steps for name in step.inputs if typed_product(name) is None}
    required.update(pair)
    if not required.issubset(set(cohort_columns)):
        raise DeferredFirstObservedError("deferred_first_observed_analysis_columns_missing")
    return pair


def _verify_raw_patient_values(*, cohort_path: Path, raw_source: Any, coordinate: Any,
                               declaration: Any) -> None:
    """Compare selected actual rows with three approved raw patient columns."""
    import pandas as pd
    from easyicu.hospital_mortality import hospital_mortality_status_from_discharge
    from easyicu.research_agent.acquisition.patient_grouping import exact_integer_series, read_digest_bound_parquet
    from easyicu.research_agent.intake.materialized_metadata import (
        load_verified_materialized_cohort_authority, read_verified_materialized_cohort_table,
    )

    expected = {"source": "registered_eicu_demo_patient_table", "patient_table_sha256": coordinate.patient_table_sha256,
                "columns": ["patientunitstayid", "gender", "hospitaldischargestatus"],
                "purpose": "post_approval_independent_native_export_value_comparison",
                "preserve_unknown_outcome": True, "post_approval_only": True}
    if declaration != expected:
        raise DeferredFirstObservedError("deferred_first_observed_raw_value_scope_invalid")
    raw, _ = read_digest_bound_parquet(
        Path(raw_source.table_paths["patient"]), expected_sha256=coordinate.patient_table_sha256,
        columns=expected["columns"], label="eicu patient verification",
        error=DeferredFirstObservedError,
    )
    raw["stay_id"] = exact_integer_series(raw["patientunitstayid"], label="raw patient stay")
    if raw["stay_id"].isna().any() or raw["stay_id"].le(0).any() or raw["stay_id"].duplicated().any():
        raise DeferredFirstObservedError("deferred_first_observed_raw_identity_invalid")
    verified = load_verified_materialized_cohort_authority(cohort_path)
    if verified is None:
        raise DeferredFirstObservedError("deferred_first_observed_materialization_failed")
    actual = read_verified_materialized_cohort_table(cohort_path, verified=verified).select(["stay_id", "sex", "death"]).to_pandas()
    actual["stay_id"] = exact_integer_series(actual["stay_id"], label="materialized stay")
    if actual["stay_id"].isna().any() or actual["stay_id"].duplicated().any():
        raise DeferredFirstObservedError("deferred_first_observed_raw_identity_invalid")
    joined = actual.merge(raw[["stay_id", "gender", "hospitaldischargestatus"]], on="stay_id",
                          how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise DeferredFirstObservedError("deferred_first_observed_raw_value_mismatch")
    def sex(values):
        return values.astype("string").str.strip().str.casefold().replace("", pd.NA)
    observed_death = joined["death"].astype("boolean")
    expected_death = hospital_mortality_status_from_discharge(joined["hospitaldischargestatus"])
    if not sex(joined["sex"]).equals(sex(joined["gender"])) or not observed_death.equals(expected_death):
        raise DeferredFirstObservedError("deferred_first_observed_raw_value_mismatch")


_FLOW_DECLARATION = {
    "scope": "complete_source_before_other_study_filters",
    "fields": ["source_patients", "source_stays", "eligible_patients", "excluded_patients_by_reason",
               "excluded_stays_by_reason", "first_icu_order_or_type_unverifiable"],
    "hospitalization_count_available": False, "lifetime_first_icu_not_guaranteed": True,
    "disclose_only_after_plan_approval_and_exact_membership": True,
}


def _public_source_flow(flow: dict[str, Any], *, cohort_rows: int, binding: dict[str, str]) -> dict[str, Any]:
    """Produce a local aggregate under the source publication policy."""
    from easyicu.research_agent.gates.publication_disclosure import PublicationDisclosureProfile, source_disclosure

    if (flow.get("schema_version") != "easyicu.first-observed-private-source-flow/1"
            or flow.get("scope") != "complete_source_before_other_study_filters"
            or flow.get("not_lifetime_first_icu") is not True
            or flow.get("public_projection_authorized") is not False
            or flow.get("eligible_patients") != cohort_rows
            or source_disclosure("eicu_demo").profile is not PublicationDisclosureProfile.REPORT_WITH_REVIEW):
        raise DeferredFirstObservedError("deferred_first_observed_source_flow_invalid")
    names = ("source_patients", "source_stays", "eligible_patients")
    groups = ("excluded_patients_by_reason", "excluded_stays_by_reason")
    if any(type(flow.get(key)) is not int or flow[key] < 0 for key in names):
        raise DeferredFirstObservedError("deferred_first_observed_source_flow_invalid")
    if any(not isinstance(flow.get(key), dict) or any(not isinstance(k, str) or type(v) is not int or v < 0
            for k, v in flow[key].items()) for key in groups):
        raise DeferredFirstObservedError("deferred_first_observed_source_flow_invalid")
    if flow["source_patients"] != flow["eligible_patients"] + sum(flow[groups[0]].values()):
        raise DeferredFirstObservedError("deferred_first_observed_source_flow_invalid")
    if flow["source_stays"] != flow["eligible_patients"] + sum(flow[groups[1]].values()):
        raise DeferredFirstObservedError("deferred_first_observed_source_flow_invalid")
    # Count suffixes are part of the publication-disclosure owner's shared
    # contract: every small subject count must be discoverable for review.
    counts = {f"{key}_n": flow[key] for key in names}
    counts.update({key: {f"{reason}_n": number for reason, number in sorted(flow[key].items())} for key in groups})
    counts["first_icu_order_or_type_unverifiable_n"] = flow[groups[0]].get("first_icu_order_or_type_unverifiable", 0)
    return {"schema_version": "easyicu.first-observed-source-flow-evidence/1",
            "scope": "complete_source_before_other_study_filters",
            "source_stays_meaning": "source ICU unit-stay records; not hospitalizations",
            "not_lifetime_first_icu": True, "hospitalization_count_available": False,
            "disclosure": "local_report_with_publication_review", "counts": counts, "binding": binding}



def _load_signed_source_flow(recipe: dict[str, Any]) -> tuple[Any, dict[str, Any], str]:
    """Reopen the signed consumed preparation without rereading raw patient data."""
    from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
    from easyicu.webserver import agent_pipeline_runs
    from easyicu.webserver.first_observed_preparation import _resolve_consumed_coordinate_evidence
    from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding

    host = recipe["host_binding"]
    pending, _ = agent_pipeline_runs.raw_preparation_review_owner(host["preparation_run_id"], host["study_context_id"])
    source_run = Path(pending.run_dir)
    with AnchoredDirectory.open(source_run) as directory:
        raw = directory.read_bytes("raw_preparation_checkpoint.json", max_bytes=1024 * 1024)
        directory.assert_still_selected()
    checkpoint = RawPreparationCheckpoint.model_validate_json(raw)
    raw_pending = checkpoint.model_dump(mode="json", exclude={"checkpoint_sha256"})
    raw_pending.update(state="pending", decision=None)
    nonce = canonical_sha256({"run_id": host["preparation_run_id"], "study_context_id": host["study_context_id"],
                              "checkpoint_sha256": canonical_sha256(raw_pending)})
    host_binding = RawPreparationHostBinding(project_id=host["project_id"], session_id=host["session_id"],
        study_context_id=host["study_context_id"], run_id=host["preparation_run_id"], nonce=nonce)
    scope = checkpoint.request.payload.get("first_observed_icu_preparation_scope")
    if not isinstance(scope, dict):
        raise DeferredFirstObservedError("deferred_first_observed_source_flow_invalid")
    coordinate, flow = _resolve_consumed_coordinate_evidence(
        run_dir=source_run, checkpoint=checkpoint, binding=host_binding, scope=scope)
    if flow is None:
        raise DeferredFirstObservedError("deferred_first_observed_source_flow_invalid")
    # Bind the derived aggregate to the exact signed private attestation,
    # without putting its contents or a directly guessable count digest into the report.
    import hmac
    from easyicu.webserver.raw_preparation_decisions import _signature

    with AnchoredDirectory.open(coordinate.path.parent) as directory:
        attestation = directory.read_bytes(nonce + ".json", max_bytes=64 * 1024)
        directory.assert_still_selected()
    document = json.loads(attestation)
    body, signature = document.get("body"), document.get("signature")
    if (not isinstance(body, dict) or not isinstance(signature, str)
            or not hmac.compare_digest(signature, _signature(body))
            or body.get("coordinate_sha256") != coordinate.sha256
            or body.get("private_source_flow") != flow):
        raise DeferredFirstObservedError("deferred_first_observed_source_flow_invalid")
    return coordinate, flow, hashlib.sha256(attestation).hexdigest()



def _publish_source_flow(*, run_dir: Path, recipe: dict[str, Any], approved_plan_sha256: str,
                         decision_set_sha256: str, recipe_sha256: str, capsule_sha256: str, receipt_sha256: str,
                         coordinate: Any, cohort_path: Path, evidence_store: Any) -> None:
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority

    expected_flow = dict(_FLOW_DECLARATION)
    if "raw_patient_verification" in recipe:
        expected_flow["unknown_outcome_count_required"] = True
    if recipe.get("source_flow_report") != expected_flow:
        raise DeferredFirstObservedError("deferred_first_observed_source_flow_scope_invalid")
    if not isinstance(evidence_store, EvidenceStore) or Path(evidence_store.root).resolve() != run_dir:
        raise DeferredFirstObservedError("deferred_first_observed_evidence_store_invalid")
    from easyicu.research_agent.authority.run_input import (
        DEFERRED_ACTUAL_CAPSULE_EVIDENCE_ID, DEFERRED_RECEIPT_EVIDENCE_ID,
    )
    from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path

    dependencies = ((DEFERRED_ACTUAL_CAPSULE_EVIDENCE_ID, capsule_sha256),
                    (DEFERRED_RECEIPT_EVIDENCE_ID, receipt_sha256))
    for evidence_id, expected_sha in dependencies:
        record = evidence_store.get(evidence_id)
        if (record is None or record.sha256 != expected_sha
                or verified_run_evidence_path(run_dir, record) is None):
            raise DeferredFirstObservedError("deferred_first_observed_evidence_store_invalid")
    signed_coordinate, flow, attestation_sha = _load_signed_source_flow(recipe)
    if signed_coordinate.sha256 != coordinate.sha256:
        raise DeferredFirstObservedError("deferred_first_observed_source_flow_invalid")
    cohort = load_verified_materialized_cohort_authority(cohort_path)
    if cohort is None:
        raise DeferredFirstObservedError("deferred_first_observed_source_flow_invalid")
    payload = _public_source_flow(flow, cohort_rows=cohort.authority.cohort_rows, binding={
        "approved_plan_sha256": approved_plan_sha256, "decision_set_sha256": decision_set_sha256,
        "recipe_sha256": recipe_sha256, "coordinate_sha256": coordinate.sha256,
        "actual_capsule_sha256": capsule_sha256, "actual_cohort_sha256": cohort.authority.cohort_sha256,
        "source_attestation_sha256": attestation_sha, "deferred_receipt_sha256": receipt_sha256,
    })
    record = evidence_store.register_json(kind="table", description="Approved source population flow",
        payload=payload, filename="first_observed_source_flow.json", evidence_id="first_observed_source_flow",
        inputs=[DEFERRED_ACTUAL_CAPSULE_EVIDENCE_ID, DEFERRED_RECEIPT_EVIDENCE_ID],
        producer="host", generation_mode="system")
    counts = payload["counts"]
    if counts is not None:
        for name, value in counts.items():
            leaves = value.items() if isinstance(value, dict) else ((None, value),)
            for subkey, number in leaves:
                field = f"counts.{name}" + (f".{subkey}" if subkey is not None else "")
                evidence_store.register_numeric_claim(value=str(number), canonical=float(number),
                    evidence_id=record.evidence_id, step_id="first_observed_source_flow", source_field=field, tolerance=0.0)


def consume_approved_first_observed_input(*, run_dir: Path, approved_plan_sha256: str,
                                         decision_set_sha256: str, provider_task: Any = None, evidence_store: Any = None):
    """Consume an approved, bounded recipe once; never retry an incomplete write.

    The fixed output directory is an atomic one-use claim. A crash before the
    signed receipt leaves an incomplete claim and is refused on re-entry.
    """
    import hmac
    import os
    from dataclasses import asdict

    from easyicu.databases.research_plan_requirements import _constraints
    from easyicu.research_agent.acquisition.foundation import acquire_universe_for_question
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.run_input import (
        load_verified_run_input_capsule, seal_run_input_capsule,
    )
    from easyicu.research_agent.canonical_json import canonical_json
    from easyicu.research_agent.intake.export_package import host_deferred_export_read_scope
    from easyicu.research_agent.intake.materialized_metadata import stage_materialized_cohort_authority
    from easyicu.research_agent.orchestration.services import DeferredFirstObservedInput
    from easyicu.research_agent.research_context.builder import build_research_context
    from easyicu.research_agent.research_context.export_selection import SourceIdentity
    from easyicu.research_agent.research_context.typed import parse_research_context_json
    from easyicu.webserver import agent_pipeline_runs, research_auto_resume
    from easyicu.webserver.durable_research_store import runtime_digest
    from easyicu.webserver.first_observed_preparation import _metadata_source
    from easyicu.webserver.first_observed_run_guard import (
        resolve_host_prepared_first_observed_coordinate, verify_exact_first_observed_membership,
        verify_first_observed_preparation_recipe, verify_prepared_first_observed_population,
    )
    from easyicu.webserver.raw_preparation_decisions import _publish_once, _signature

    root = Path(run_dir).resolve()
    recipe = verify_approved_deferred_review(run_dir=root, approved_plan_sha256=approved_plan_sha256,
                                             decision_set_sha256=decision_set_sha256)
    # Recheck the full reviewed population before reading any source data.
    if "population_spec" in recipe:
        from easyicu.research_agent.planning.population_spec import PopulationSpec
        from easyicu.webserver.first_observed_deferred_recipe import _first_observed_review_criterion
        try:
            population = PopulationSpec.model_validate(recipe["population_spec"])
            first = _first_observed_review_criterion(population)
            if first.model_dump(mode="json") != recipe.get("criterion"):
                raise ValueError("first_observed_recipe_population_changed")
        except (TypeError, ValueError) as exc:
            raise DeferredFirstObservedError("deferred_first_observed_population_changed") from exc
    # A preliminary proposal deliberately cannot authorize source data reads.
    if recipe.get("decoded_scope_enforced") is not True:
        raise DeferredFirstObservedError("deferred_first_observed_reader_scope_not_authorized")
    if recipe.get("materializer_runtime_sha256") != runtime_digest():
        raise DeferredFirstObservedError("deferred_first_observed_runtime_changed")
    binding = recipe.get("host_binding")
    material = recipe.get("materialization")
    budget = recipe.get("budget")
    if not all(isinstance(item, dict) for item in (binding, material, budget)):
        raise DeferredFirstObservedError("deferred_first_observed_recipe_invalid")
    if (material.get("producer") != "easyicu.research_agent.acquisition.foundation.acquire_universe_for_question"
            or material.get("selection_authority") != "host_exact"
            or material.get("database") not in {"eicu_demo", "eicu"} or material.get("emit_trajectory") is not False
            or material.get("additional_population_filters") is not False
            or budget.get("reset_or_increase_authorized") is not False
            or budget.get("execution_mode") != "deferred_reviewed"):
        raise DeferredFirstObservedError("deferred_first_observed_recipe_invalid")
    with AnchoredDirectory.open(root) as directory:
        raw_context = directory.read_bytes("research_context.json", max_bytes=16 * 1024 * 1024)
        raw_capsule = directory.read_bytes("run_input_capsule.json", max_bytes=64 * 1024 * 1024)
        directory.assert_still_selected()
    context = parse_research_context_json(raw_context.decode("utf-8"))
    original_capsule = json.loads(raw_capsule)
    if not isinstance(original_capsule, dict) or not isinstance(original_capsule.get("scientific_identity"), dict):
        raise DeferredFirstObservedError("deferred_first_observed_original_capsule_invalid")
    source = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    if source is None or source.database != "eicu_demo":
        raise DeferredFirstObservedError("deferred_first_observed_source_changed")
    recovery = agent_pipeline_runs.get_review_recovery_record(binding.get("preparation_run_id"))
    if (recovery is None or recovery.record_sha256 != budget.get("recovery_record_sha256")
            or recovery.hard_stop_task_id != budget.get("hard_stop_task_id")
            or recovery.hard_stop_declaration_sha256 != budget.get("hard_stop_declaration_sha256")
            or recovery.budget_mode != budget.get("planning_mode")):
        raise DeferredFirstObservedError("deferred_first_observed_budget_changed")
    if provider_task is None:
        task = research_auto_resume._original_task(recovery)
    else:
        from easyicu.research_agent.authority.provider_hard_stop import TaskProviderHardStop

        task = provider_task
        if (not isinstance(task, TaskProviderHardStop) or task.task_id != recovery.hard_stop_task_id
                or Path(task.ledger.path) != Path(recovery.hard_stop_ledger_path)):
            raise DeferredFirstObservedError("deferred_first_observed_budget_changed")
        ledger_snapshot = task.ledger.snapshot()
        if ledger_snapshot.get("declaration_sha256") != recovery.hard_stop_declaration_sha256:
            raise DeferredFirstObservedError("deferred_first_observed_budget_changed")
        task.assert_active()
    if task.ledger.limits.canonical_payload() != budget.get("limits"):
        raise DeferredFirstObservedError("deferred_first_observed_budget_changed")
    manifest = recipe.get("source_manifest_sha256")
    coord_args = {key: binding.get(key) for key in ("preparation_run_id", "study_context_id", "project_id", "session_id")}
    coord_args.update(source_identity=source, criterion=recipe.get("criterion"), export_manifest_sha256=manifest)
    proof = verify_first_observed_preparation_recipe(**coord_args)
    if asdict(proof) != recipe.get("preparation_proof"):
        raise DeferredFirstObservedError("deferred_first_observed_preparation_changed")
    coordinate = resolve_host_prepared_first_observed_coordinate(**coord_args)
    if recipe.get("verified_source_files") != {"archive_sha256": coordinate.archive_sha256,
                                                 "patient_table_sha256": coordinate.patient_table_sha256}:
        raise DeferredFirstObservedError("deferred_first_observed_source_changed")
    source_path, metadata = _metadata_source(source)
    from easyicu.webserver.raw_source_authority import resolve_raw_source_identity_binding

    raw_source = resolve_raw_source_identity_binding(export_path=source_path, database="eicu_demo")
    if (raw_source is None or raw_source.export_manifest_sha256 != manifest
            or raw_source.table_sha256.get("patient") != coordinate.patient_table_sha256):
        raise DeferredFirstObservedError("deferred_first_observed_patient_source_changed")
    with AnchoredDirectory.open(source_path) as directory:
        manifest_raw = directory.read_bytes("_manifest.json", max_bytes=16 * 1024 * 1024, expected_sha256=manifest)
        directory.assert_still_selected()
    package_database = json.loads(manifest_raw).get("database")
    if package_database not in {"eicu_demo", "eicu"} or package_database != material.get("database"):
        raise DeferredFirstObservedError("deferred_first_observed_source_changed")
    if metadata.get("export_manifest_sha256") != manifest:
        raise DeferredFirstObservedError("deferred_first_observed_source_changed")
    requested_concepts = set(material.get("required_feature_concepts", ())) | set(material.get("outcome_concepts", ()))
    raw_value_check = {"sex", "death"}.issubset(requested_concepts)
    if raw_value_check != ("raw_patient_verification" in recipe):
        raise DeferredFirstObservedError("deferred_first_observed_raw_value_scope_invalid")
    entries = recipe.get("declared_schema_columns")
    if not isinstance(entries, list) or not entries:
        raise DeferredFirstObservedError("deferred_first_observed_recipe_invalid")
    columns = {entry["file"]: entry["columns"] for entry in entries if isinstance(entry, dict)
               and isinstance(entry.get("file"), str) and isinstance(entry.get("columns"), list)}
    if len(columns) != len(entries):
        raise DeferredFirstObservedError("deferred_first_observed_recipe_invalid")
    with AnchoredDirectory.open(root) as directory:
        recipe_raw = directory.read_bytes("first_observed_deferred_recipe.json", max_bytes=4 * 1024 * 1024)
        directory.assert_still_selected()
    recipe_sha = hashlib.sha256(recipe_raw).hexdigest()
    output = root / "deferred_input"
    receipt_path = root / "deferred_first_observed_receipt.json"

    def reopen():
        with AnchoredDirectory.open(root) as directory:
            raw = directory.read_bytes(receipt_path.name, max_bytes=1024 * 1024)
            directory.assert_still_selected()
        document = json.loads(raw)
        body, signature = document.get("body"), document.get("signature")
        if (not isinstance(body, dict) or not isinstance(signature, str)
                or not hmac.compare_digest(signature, _signature(body))
                or body.get("approved_plan_sha256") != approved_plan_sha256
                or body.get("decision_set_sha256") != decision_set_sha256
                or body.get("recipe_sha256") != recipe_sha
                or body.get("raw_patient_values_verified") is not raw_value_check):
            raise DeferredFirstObservedError("deferred_first_observed_receipt_invalid")
        cohort = output / "cohort.parquet"
        with AnchoredDirectory.open(output) as directory:
            context_bytes = directory.read_bytes("research_context.json", max_bytes=16 * 1024 * 1024)
            capsule_bytes = directory.read_bytes("run_input_capsule.json", max_bytes=64 * 1024 * 1024)
            directory.assert_still_selected()
        recovered_capsule = json.loads(capsule_bytes)
        recovered_identity = dict(recovered_capsule["scientific_identity"])
        recovered_identity.pop("materialized_cohort_authority_ref", None)
        if recovered_identity != original_capsule["scientific_identity"]:
            raise DeferredFirstObservedError("deferred_first_observed_receipt_invalid")
        verified = load_verified_run_input_capsule(run_dir=output, scientific_identity=recovered_capsule["scientific_identity"])
        if (hashlib.sha256(context_bytes).hexdigest() != body.get("context_sha256")
                or hashlib.sha256(capsule_bytes).hexdigest() != body.get("capsule_sha256")
                or verified.capsule.cohort_sha256 != body.get("cohort_sha256")):
            raise DeferredFirstObservedError("deferred_first_observed_receipt_invalid")
        verify_prepared_first_observed_population(cohort_path=cohort, study_context_id=binding["study_context_id"],
                                                   criterion=recipe["criterion"], source_identity=source)
        verify_exact_first_observed_membership(cohort_path=cohort, coordinate=coordinate)
        return DeferredFirstObservedInput(cohort, parse_research_context_json(context_bytes.decode("utf-8")),
                                          body["capsule_sha256"], hashlib.sha256(raw).hexdigest())

    if receipt_path.exists() or receipt_path.is_symlink():
        return reopen()
    try:
        os.mkdir(output, 0o700)
    except FileExistsError as exc:
        raise DeferredFirstObservedError("deferred_first_observed_incomplete_claim") from exc
    # This directory remains as a durable stop marker on any failure or crash.
    with host_deferred_export_read_scope(manifest_sha256=manifest, columns_by_file=columns):
        result = acquire_universe_for_question(
            export_dir=source_path, question=context.research_question, llm=None,
            output_dir=output / "universe", stem="cohort", target_outcome=material["target_outcome"],
            primary_exposure_concept=material["primary_exposure"], outcome_concepts=material["outcome_concepts"],
            required_feature_concepts=material["required_feature_concepts"], static_concepts=material["static_concepts"],
            allowed_modules=material["allowed_modules"], concept_selection_authority="host_exact",
            cohort_window=tuple(material["cohort_window_hours"]), database=package_database, emit_trajectory=False,
            first_observed_icu_coordinate=coordinate,
        )
    universe = output / "universe" / "cohort.parquet"
    cohort = output / "cohort.parquet"
    if result.blocked or result.universe_path != universe:
        raise DeferredFirstObservedError("deferred_first_observed_materialization_failed")
    verify_prepared_first_observed_population(cohort_path=universe, study_context_id=binding["study_context_id"],
                                               criterion=recipe["criterion"], source_identity=source)
    verify_exact_first_observed_membership(cohort_path=universe, coordinate=coordinate)
    if raw_value_check:
        _verify_raw_patient_values(cohort_path=universe, raw_source=raw_source, coordinate=coordinate,
                                   declaration=recipe.get("raw_patient_verification"))
    staged = stage_materialized_cohort_authority(universe, cohort, producer_implementation_sha256=runtime_digest(),
                                                  expected_source_authority=result.cohort_authority_ref)
    if staged is None or result.cohort_authority_ref is None:
        raise DeferredFirstObservedError("deferred_first_observed_materialization_failed")
    if verify_approved_deferred_review(run_dir=root, approved_plan_sha256=approved_plan_sha256,
                                        decision_set_sha256=decision_set_sha256) != recipe:
        raise DeferredFirstObservedError("deferred_first_observed_recipe_changed")
    exposure_name, outcome_name = _validate_actual_plan_columns(
        run_dir=root, material=material, cohort_columns=staged.authority.cohort_columns)
    actual = build_research_context(research_question=context.research_question, cohort=cohort,
        cohort_name=context.cohort.cohort_name, database=package_database, target_outcome=outcome_name,
        endpoint=context.endpoint, primary_exposure=exposure_name,
        user_preferences=context.user_preferences, time_windows=context.time_windows)
    context_path = output / "research_context.json"
    with AnchoredDirectory.open(output) as directory:
        _publish_once(directory, context_path.name, actual.model_dump_json().encode("utf-8"))
        directory.assert_still_selected()
    evidence = EvidenceStore(output)
    evidence.register_file(kind="log", description="Materialized research context", source_path=context_path,
                           evidence_id="research_context", producer="pipeline", generation_mode="system")
    scientific_identity = {**original_capsule["scientific_identity"],
                           "materialized_cohort_authority_ref": result.cohort_authority_ref.to_dict()}
    capsule = seal_run_input_capsule(run_dir=output, evidence=evidence,
        scientific_identity=scientific_identity, initial_environment=original_capsule["initial_environment"],
        context_path=context_path, cohort_path=cohort, experiment_spec_path=None)
    with AnchoredDirectory.open(output) as directory:
        capsule_bytes = directory.read_bytes("run_input_capsule.json", max_bytes=64 * 1024 * 1024)
        context_bytes = directory.read_bytes("research_context.json", max_bytes=16 * 1024 * 1024)
        directory.assert_still_selected()
    body = {"schema_version":"easyicu.first-observed-deferred-input-receipt/1",
            "approved_plan_sha256":approved_plan_sha256, "decision_set_sha256":decision_set_sha256,
            "recipe_sha256":recipe_sha,
            "capsule_sha256":hashlib.sha256(capsule_bytes).hexdigest(),
            "context_sha256":hashlib.sha256(context_bytes).hexdigest(), "cohort_sha256":capsule.cohort_sha256,
            "raw_patient_values_verified":raw_value_check}
    with AnchoredDirectory.open(root) as directory:
        _publish_once(directory, receipt_path.name, canonical_json({"body":body, "signature":_signature(body)}).encode())
        directory.assert_still_selected()
    return reopen()


def publish_approved_source_flow(*, run_dir: Path, approved_plan_sha256: str,
                                 decision_set_sha256: str, evidence_store: Any) -> None:
    """Publish local flow only after all model/sandbox work has ended.

    The caller owns the finalization ordering. This host-only function checks
    the original review and the signed actual-input receipt again; it grants no
    model or sandbox access to the local result.
    """
    import hmac
    from easyicu.research_agent.authority.run_input import load_verified_run_input_capsule
    from easyicu.research_agent.research_context.export_selection import SourceIdentity
    from easyicu.research_agent.research_context.typed import parse_research_context_json
    from easyicu.databases.research_plan_requirements import _constraints
    from easyicu.webserver.first_observed_run_guard import verify_exact_first_observed_membership
    from easyicu.webserver.raw_preparation_decisions import _signature

    root = Path(run_dir).resolve()
    recipe = verify_approved_deferred_review(run_dir=root, approved_plan_sha256=approved_plan_sha256,
        decision_set_sha256=decision_set_sha256, _allow_executing=True)
    with AnchoredDirectory.open(root) as directory:
        receipt_raw = directory.read_bytes("deferred_first_observed_receipt.json", max_bytes=1024 * 1024)
        recipe_raw = directory.read_bytes("first_observed_deferred_recipe.json", max_bytes=4 * 1024 * 1024)
        context_raw = directory.read_bytes("research_context.json", max_bytes=16 * 1024 * 1024)
        original_raw = directory.read_bytes("run_input_capsule.json", max_bytes=64 * 1024 * 1024)
        directory.assert_still_selected()
    document = json.loads(receipt_raw)
    body, signature = document.get("body"), document.get("signature")
    recipe_sha = hashlib.sha256(recipe_raw).hexdigest()
    if (not isinstance(body, dict) or not isinstance(signature, str)
            or not hmac.compare_digest(signature, _signature(body))
            or body.get("approved_plan_sha256") != approved_plan_sha256
            or body.get("decision_set_sha256") != decision_set_sha256
            or body.get("recipe_sha256") != recipe_sha
            or body.get("raw_patient_values_verified") is not bool(recipe.get("raw_patient_verification"))):
        raise DeferredFirstObservedError("deferred_first_observed_receipt_invalid")
    output = root / "deferred_input"
    with AnchoredDirectory.open(output) as directory:
        actual_capsule_raw = directory.read_bytes("run_input_capsule.json", max_bytes=64 * 1024 * 1024)
        actual_context_raw = directory.read_bytes("research_context.json", max_bytes=16 * 1024 * 1024)
        directory.assert_still_selected()
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.run_input import (
        DEFERRED_ACTUAL_CAPSULE_EVIDENCE_ID, DEFERRED_RECEIPT_EVIDENCE_ID,
    )
    from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path

    if not isinstance(evidence_store, EvidenceStore) or Path(evidence_store.root).resolve() != root:
        raise DeferredFirstObservedError("deferred_first_observed_evidence_store_invalid")
    for evidence_id, expected_raw in ((DEFERRED_RECEIPT_EVIDENCE_ID, receipt_raw),
                                      (DEFERRED_ACTUAL_CAPSULE_EVIDENCE_ID, actual_capsule_raw)):
        record = evidence_store.get(evidence_id)
        if (record is None or record.sha256 != hashlib.sha256(expected_raw).hexdigest()
                or verified_run_evidence_path(root, record) is None):
            raise DeferredFirstObservedError("deferred_first_observed_receipt_invalid")
    actual_capsule = json.loads(actual_capsule_raw)
    original_capsule = json.loads(original_raw)
    identity = dict(actual_capsule["scientific_identity"])
    parent = identity.pop("materialized_cohort_authority_ref", None)
    if (parent is None or identity != original_capsule["scientific_identity"]
            or hashlib.sha256(actual_capsule_raw).hexdigest() != body.get("capsule_sha256")
            or hashlib.sha256(actual_context_raw).hexdigest() != body.get("context_sha256")):
        raise DeferredFirstObservedError("deferred_first_observed_receipt_invalid")
    verified = load_verified_run_input_capsule(run_dir=output, scientific_identity=actual_capsule["scientific_identity"])
    if verified.capsule.cohort_sha256 != body.get("cohort_sha256"):
        raise DeferredFirstObservedError("deferred_first_observed_receipt_invalid")
    context = parse_research_context_json(context_raw.decode("utf-8"))
    source = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    if source is None or source.database != "eicu_demo":
        raise DeferredFirstObservedError("deferred_first_observed_source_changed")
    coordinate, _, _ = _load_signed_source_flow(recipe)
    proof = recipe.get("preparation_proof")
    if (not isinstance(proof, dict) or proof.get("coordinate_sha256") != coordinate.sha256
            or proof.get("source_identity_sha256") != canonical_sha256(source.to_dict())
            or proof.get("criterion_sha256") != canonical_sha256(recipe.get("criterion"))
            or proof.get("export_manifest_sha256") != coordinate.export_manifest_sha256):
        raise DeferredFirstObservedError("deferred_first_observed_source_flow_invalid")
    cohort = output / "cohort.parquet"
    verify_exact_first_observed_membership(cohort_path=cohort, coordinate=coordinate)
    _publish_source_flow(run_dir=root, recipe=recipe, approved_plan_sha256=approved_plan_sha256,
        decision_set_sha256=decision_set_sha256, recipe_sha256=recipe_sha, capsule_sha256=body["capsule_sha256"],
        receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(), coordinate=coordinate, cohort_path=cohort, evidence_store=evidence_store)

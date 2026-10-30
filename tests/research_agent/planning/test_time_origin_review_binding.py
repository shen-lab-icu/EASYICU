"""A source-bound candidate cannot by itself authorize an origin substitution."""
import json
from types import SimpleNamespace

import pytest

from easyicu.databases.research_approval import (
    CapabilityApprovalError, bind_database_capability_review, require_approved_database_capabilities,
)
from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_substitutions import time_origin_selection
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.schema import UserPreferences
from tests.support.source_substitutions import _review, context, plan


def _candidate_context():
    ctx = context("sic", outcome="icu_death")
    constraints = json.loads(ctx.user_preferences.data_constraints)
    identity = {"database": "sic", "registration_id": "src_example", "version": "1.0.6"}
    selection = {
        "requested_origin": "icu_admission",
        "selected_origin": "first_intensive_or_intermediate_care_transfer",
        "source_field": research_capability_profile("sic").care_unit_admission_proxy.model_dump(mode="json"),
        "source_identity_sha256": canonical_sha256(identity),
    }
    constraints.update(source_identity=identity, database_time_origin_selection=selection)
    return ctx.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})


def test_candidate_is_source_bound_reviewable_but_cannot_execute(tmp_path):
    ctx = _candidate_context()
    compiled = plan(ctx)
    selected = time_origin_selection(ctx)
    assert selected.source_field.column == "ICUOffset"
    # The regular review remains explicitly blocked despite a valid candidate.
    review = bind_database_capability_review(_review(ctx, compiled), context=ctx, plan=compiled)
    assert review.facts["database_time_origin_review"]["execution_contract_available"] is False
    assert review.facts["database_time_origin_review"]["registered_export_receipt"] is None
    assert review.approval_allowed is False
    with pytest.raises(CapabilityApprovalError, match="time_origin_execution_contract_unavailable"):
        require_approved_database_capabilities(SimpleNamespace(context=ctx, plan=compiled, evidence={}), tmp_path)


def test_recorded_selection_becomes_a_typed_change_and_keeps_icu_los_refusal():
    from easyicu.databases.research_requirements import StudyRequirements
    from easyicu.databases.research_substitutions import source_decision

    ctx = _candidate_context()
    selected = time_origin_selection(ctx)
    profile = research_capability_profile("sic")
    decision = source_decision(profile, StudyRequirements(family="association_study"), time_origin=selected)
    change, = (item for item in decision.substitutions if item.capability == "time_zero")
    assert (change.original, change.substitute, change.source_field.column) == (
        "icu_admission", "first_intensive_or_intermediate_care_transfer", "ICUOffset",
    )
    assert decision.status == "substitution"
    refused = source_decision(profile, StudyRequirements(family="association_study", outcome="los_icu"),
                              time_origin=selected)
    assert refused.status == "refusal"
    assert any(item.capability == "icu_los" for item in refused.missing)


def test_case_duration_proxy_is_explicit_source_bound_and_never_an_icu_los_grant(tmp_path):
    from easyicu.databases.research_substitutions import case_duration_selection
    ctx = _candidate_context()
    constraints = json.loads(ctx.user_preferences.data_constraints)
    axis = constraints["database_time_origin_selection"]
    constraints["database_case_duration_selection"] = {
        "requested_outcome": "los_icu", "proposed_outcome": "care_unit_to_case_closure_days",
        "start_field": axis["source_field"],
        "end_field": research_capability_profile("sic").case_close_proxy.model_dump(mode="json"),
        "source_identity_sha256": axis["source_identity_sha256"],
    }
    constraints["database_study_requirements"]["outcome"] = "los_icu"
    ctx = ctx.model_copy(update={"target_outcome": "los_icu",
                                 "user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    candidate = case_duration_selection(ctx)
    assert candidate.end_field.column == "TimeOfStay"
    compiled = plan(ctx)
    reviewed = _review(ctx, compiled)
    assert reviewed.facts["database_case_duration_review"]["execution_contract_available"] is False
    assert not reviewed.approval_allowed
    with pytest.raises(CapabilityApprovalError, match="case_duration_execution_contract_unavailable"):
        require_approved_database_capabilities(SimpleNamespace(context=ctx, plan=compiled, evidence={}), tmp_path)
    constraints["database_case_duration_selection"]["source_identity_sha256"] = "f" * 64
    changed = ctx.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    with pytest.raises(ValueError, match="case_duration_selection_binding_drift"):
        case_duration_selection(changed)


def test_case_duration_signed_gate_requires_human_and_exact_source(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.intake import materialized_metadata
    from tests.support.signed_case_duration import signed_case_duration_case
    from tests.support.source_substitutions import _approve

    ctx, compiled, runtime, receipt, proof, origin_attestor, case_attestor = signed_case_duration_case()
    base = _review(ctx, compiled).model_copy(update={"approval_allowed": True, "status": "ready_for_approval"})
    review = bind_database_capability_review(base, context=ctx, plan=compiled,
        time_origin_attestor=origin_attestor, case_duration_attestor=case_attestor, runtime_authority=runtime)
    assert review.approval_allowed
    assert review.facts["database_case_duration_review"]["execution_contract_available"] is True
    evidence = EvidenceStore(tmp_path)
    monkeypatch.setattr(materialized_metadata, "load_verified_materialized_cohort_authority", lambda _path: SimpleNamespace(
        authority=SimpleNamespace(semantic_provenance={"source_time_axis": receipt.to_dict()})))
    candidate = SimpleNamespace(context=ctx, plan=compiled, evidence=evidence)
    with pytest.raises(CapabilityApprovalError):
        require_approved_database_capabilities(candidate, tmp_path, cohort_path=tmp_path / "cohort.parquet",
            time_origin_attestor=origin_attestor, case_duration_attestor=case_attestor, runtime_authority=runtime)
    approved = _approve(tmp_path, evidence, ctx, compiled, review)
    result = require_approved_database_capabilities(approved, tmp_path, cohort_path=tmp_path / "cohort.parquet",
        time_origin_attestor=origin_attestor, case_duration_attestor=case_attestor, runtime_authority=runtime)
    assert result.authority == runtime
    assert result.source_axis.manifest_sha256 == receipt.manifest_sha256
    with pytest.raises(CapabilityApprovalError):
        require_approved_database_capabilities(approved, tmp_path, cohort_path=tmp_path / "cohort.parquet",
            time_origin_attestor=origin_attestor, runtime_authority=runtime)
    from easyicu.webserver.time_origin_source_authority import CaseDurationSourceProvenance
    changed_proof = CaseDurationSourceProvenance(proof.snapshot_sha256, "0" * 64, proof.source_config_sha256)
    with pytest.raises(CapabilityApprovalError):
        require_approved_database_capabilities(approved, tmp_path, cohort_path=tmp_path / "cohort.parquet",
            time_origin_attestor=origin_attestor, case_duration_attestor=lambda *_: (receipt, changed_proof),
            runtime_authority=runtime)


def test_case_duration_pre_materialization_rejects_pending_and_source_drift(tmp_path):
    from easyicu.databases.research_approval import require_approved_case_duration_pre_materialization
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.webserver.time_origin_source_authority import CaseDurationSourceProvenance
    from tests.support.signed_case_duration import signed_case_duration_case
    from tests.support.source_substitutions import _approve

    ctx, compiled, runtime, receipt, proof, origin_attestor, case_attestor = signed_case_duration_case()
    base = _review(ctx, compiled).model_copy(update={"approval_allowed": True, "status": "ready_for_approval"})
    review = bind_database_capability_review(base, context=ctx, plan=compiled,
        time_origin_attestor=origin_attestor, case_duration_attestor=case_attestor, runtime_authority=runtime)
    evidence = EvidenceStore(tmp_path)
    kwargs = {"time_origin_attestor": origin_attestor, "case_duration_attestor": case_attestor, "runtime_authority": runtime}
    with pytest.raises(CapabilityApprovalError, match="case_duration_execution_contract_unavailable"):
        require_approved_case_duration_pre_materialization(SimpleNamespace(context=ctx, plan=compiled, evidence=evidence),
                                                            tmp_path, **kwargs)
    approved = _approve(tmp_path, evidence, ctx, compiled, review)
    binding = require_approved_case_duration_pre_materialization(approved, tmp_path, **kwargs)
    assert binding.authority == runtime and binding.source_axis.manifest_sha256 == receipt.manifest_sha256
    changed = CaseDurationSourceProvenance(proof.snapshot_sha256, "0" * 64, proof.source_config_sha256)
    with pytest.raises(CapabilityApprovalError, match="case_duration_execution_contract_unavailable"):
        require_approved_case_duration_pre_materialization(approved, tmp_path, **{
            **kwargs, "case_duration_attestor": lambda *_: (receipt, changed),
        })


def test_matching_origin_and_method_name_without_signed_authority_still_cannot_execute(tmp_path, monkeypatch):
    from easyicu.databases.research_substitutions import bind_source_substitutions
    from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.intake import materialized_metadata
    from easyicu.research_agent.schema import TimeWindow
    from easyicu.webserver.time_origin_source_authority import RegisteredExportTimeOrigin

    ctx = _candidate_context()
    variables = [v.model_copy(update={"analysis_window": "first_day", "analysis_window_role": "exposure_definition"})
                 if v.name == "lact" else v for v in ctx.variables]
    ctx = ctx.model_copy(update={"variables": variables, "time_windows": [TimeWindow(
        name="first_day", anchor=SIC_CARE_UNIT_ORIGIN, start_hours=0, end_hours=24,
    )]})
    ctx = bind_source_substitutions(ctx)
    compiled = plan(ctx)
    selection = time_origin_selection(ctx)
    receipt = RegisteredExportTimeOrigin(selection.source_identity_sha256, "a" * 64, "b" * 64, selection.selected_origin)
    def attestor(_ctx, _selection):
        return selection, receipt
    base = _review(ctx, compiled).model_copy(update={"approval_allowed": True, "status": "ready_for_approval"})
    review = bind_database_capability_review(base, context=ctx, plan=compiled, time_origin_attestor=attestor)
    assert not review.approval_allowed
    assert review.facts["database_time_origin_review"]["execution_contract_available"] is False
    evidence = EvidenceStore(tmp_path)
    unapproved = SimpleNamespace(context=ctx, plan=compiled, evidence=evidence)
    monkeypatch.setattr(materialized_metadata, "load_verified_materialized_cohort_authority", lambda _path: SimpleNamespace(
        authority=SimpleNamespace(semantic_provenance={"source_time_axis": receipt.to_dict()})))
    with pytest.raises(CapabilityApprovalError, match="time_origin_execution_contract_unavailable"):
        require_approved_database_capabilities(unapproved, tmp_path, cohort_path=tmp_path / "cohort.parquet",
                                               time_origin_attestor=attestor)


def test_signed_origin_requires_review_approval_and_matching_cohort_receipt(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.intake import materialized_metadata
    from tests.support.signed_time_origin import signed_origin_case
    from tests.support.source_substitutions import _approve

    ctx, compiled, runtime, receipt, attestor = signed_origin_case()
    base = _review(ctx, compiled).model_copy(update={"approval_allowed": True, "status": "ready_for_approval"})
    review = bind_database_capability_review(base, context=ctx, plan=compiled,
                                             time_origin_attestor=attestor, runtime_authority=runtime)
    assert review.approval_allowed
    assert review.facts["database_time_origin_review"]["execution_contract_available"] is True
    evidence = EvidenceStore(tmp_path)
    monkeypatch.setattr(materialized_metadata, "load_verified_materialized_cohort_authority", lambda _path: SimpleNamespace(
        authority=SimpleNamespace(semantic_provenance={"source_time_axis": receipt.to_dict()})))
    candidate = SimpleNamespace(context=ctx, plan=compiled, evidence=evidence)
    with pytest.raises(CapabilityApprovalError, match="time_origin_execution_contract_unavailable"):
        require_approved_database_capabilities(candidate, tmp_path, cohort_path=tmp_path / "cohort.parquet",
                                               time_origin_attestor=attestor, runtime_authority=runtime)
    approved = _approve(tmp_path, evidence, ctx, compiled, review)
    assert require_approved_database_capabilities(
        approved, tmp_path, cohort_path=tmp_path / "cohort.parquet",
        time_origin_attestor=attestor, runtime_authority=runtime,
    ).manifest_sha256 == receipt.manifest_sha256
    with pytest.raises(CapabilityApprovalError):
        require_approved_database_capabilities(approved, tmp_path, cohort_path=tmp_path / "cohort.parquet",
                                               time_origin_attestor=attestor)
    monkeypatch.setattr(materialized_metadata, "load_verified_materialized_cohort_authority", lambda _path: SimpleNamespace(
        authority=SimpleNamespace(semantic_provenance={"source_time_axis": {**receipt.to_dict(), "manifest_sha256": "c" * 64}})))
    with pytest.raises(CapabilityApprovalError, match="time_origin_execution_contract_unavailable"):
        require_approved_database_capabilities(approved, tmp_path, cohort_path=tmp_path / "cohort.parquet",
                                               time_origin_attestor=attestor, runtime_authority=runtime)


def test_real_scientific_reviewer_admits_supported_signed_candidate_without_claiming_paper_authority():
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    from tests.support.signed_time_origin import signed_origin_case

    ctx, compiled, runtime, receipt, attestor = signed_origin_case()
    reviewed = build_plan_scientific_review(context=ctx, plan=compiled, runtime_authority=runtime)
    reviewed = bind_database_capability_review(reviewed, context=ctx, plan=compiled,
                                               runtime_authority=runtime, time_origin_attestor=attestor)
    assert reviewed.approval_allowed
    assert reviewed.status == "analysis_only"
    assert reviewed.facts["database_time_origin_review"]["registered_export_receipt"] == receipt.to_dict()
    assert not reviewed.top_journal_candidate
    assert any(item.code == "TOP_JOURNAL_LITERATURE_SEARCH_NOT_ESTABLISHED" for item in reviewed.findings)


def test_host_attestor_receipt_is_bound_but_does_not_grant_execution(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN
    from easyicu.research_agent.research_context.export_selection import SourceIdentity
    from easyicu.webserver import time_origin_source_authority as host

    ctx = _candidate_context()
    compiled = plan(ctx)
    identity = SourceIdentity(database="sic", registration_id="src_example", version="1.0.6")
    package = SimpleNamespace(root=tmp_path, database="sic", declared_time_origin=SIC_CARE_UNIT_ORIGIN,
                              authority_sha256="a" * 64, manifest_sha256="b" * 64)
    monkeypatch.setattr(host.sources, "load_registry", lambda: {"sources": [{
        "id": "src_example", "database": "sic", "version": "1.0.6", "path": str(tmp_path), "ok": True,
    }]})
    monkeypatch.setattr(host.sources, "registered_source_identity", lambda _: identity)
    monkeypatch.setattr(host, "open_export_package", lambda _: nullcontext(package))
    review = bind_database_capability_review(_review(ctx, compiled), context=ctx, plan=compiled,
                                             time_origin_attestor=host.review_time_origin_attestor)
    receipt = review.facts["database_time_origin_review"]["registered_export_receipt"]
    assert receipt["source_identity_sha256"] == canonical_sha256(identity.to_dict())
    assert receipt["export_authority_sha256"] == "a" * 64
    assert receipt["manifest_sha256"] == "b" * 64
    assert review.approval_allowed is False


def test_scientific_review_gate_uses_host_attestor(tmp_path, monkeypatch):
    from easyicu.research_agent.orchestration import scientific_plan_review_gate as gate
    from easyicu.webserver.time_origin_source_authority import RegisteredExportTimeOrigin

    ctx = _candidate_context()
    compiled = plan(ctx)
    candidate = time_origin_selection(ctx)
    receipt = RegisteredExportTimeOrigin(
        candidate.source_identity_sha256, "a" * 64, "b" * 64, candidate.selected_origin,
    )
    monkeypatch.setattr(gate, "build_plan_scientific_review", lambda **_: _review(ctx, compiled))
    monkeypatch.setattr(gate, "persist_or_validate_scientific_plan_review", lambda **kw: (kw["current_review"], tmp_path))
    monkeypatch.setattr(gate, "scientific_plan_review_finding", lambda _: None)
    result = gate.prepare_scientific_plan_review_gate(
        context=ctx, plan=compiled, literature=None, figure_strategy=None, run_dir=tmp_path,
        evidence=None, model_retention=None, time_origin_attestor=lambda _ctx, selected: (selected, receipt),
    )
    assert result.review.facts["database_time_origin_review"]["registered_export_receipt"] == receipt.to_dict()
    assert result.review.approval_allowed is False


def test_registered_export_can_offer_origin_without_selecting_it(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN
    from easyicu.research_agent.research_context.export_selection import SourceIdentity
    from easyicu.webserver import time_origin_source_authority as host

    ctx = _candidate_context()
    constraints = json.loads(ctx.user_preferences.data_constraints)
    constraints.pop("database_time_origin_selection")
    constraints["database_study_requirements"]["outcome"] = "los_icu"
    ctx = ctx.model_copy(update={"target_outcome": "los_icu", "user_preferences": UserPreferences(
        data_constraints=json.dumps(constraints))})
    compiled = plan(ctx)
    identity = SourceIdentity(database="sic", registration_id="src_example", version="1.0.6")
    package = SimpleNamespace(root=tmp_path, database="sic", declared_time_origin=SIC_CARE_UNIT_ORIGIN,
                              authority_sha256="a" * 64, manifest_sha256="b" * 64)
    monkeypatch.setattr(host.sources, "load_registry", lambda: {"sources": [{
        "id": "src_example", "database": "sic", "version": "1.0.6", "path": str(tmp_path), "ok": True,
    }]})
    monkeypatch.setattr(host.sources, "registered_source_identity", lambda _: identity)
    monkeypatch.setattr(host, "open_export_package", lambda _: nullcontext(package))
    review = bind_database_capability_review(_review(ctx, compiled), context=ctx, plan=compiled,
                                             time_origin_attestor=host.review_time_origin_attestor)
    candidate = review.facts["database_time_origin_review"]
    assert candidate["disposition"] == "proposed_not_selected"
    assert candidate["registered_export_receipt"]["manifest_sha256"] == "b" * 64
    assert time_origin_selection(ctx) is None
    assert not review.approval_allowed
    with pytest.raises(CapabilityApprovalError, match="study_capability_refused"):
        require_approved_database_capabilities(SimpleNamespace(context=ctx, plan=compiled, evidence={}), tmp_path)

    def unavailable(_context, _selection):
        raise ValueError("source not registered")

    refusal = bind_database_capability_review(_review(ctx, compiled), context=ctx, plan=compiled,
                                              time_origin_attestor=unavailable)
    assert refusal.approval_allowed is False
    assert "database_time_origin_review" not in refusal.facts


@pytest.mark.parametrize("change", ["source_identity", "source_field", "selected_origin", "digest"])
def test_candidate_rejects_mismatched_source_or_origin(change):
    ctx = _candidate_context()
    constraints = json.loads(ctx.user_preferences.data_constraints)
    selection = constraints["database_time_origin_selection"]
    if change == "source_identity":
        constraints["source_identity"]["registration_id"] = "another_source"
    elif change == "source_field":
        selection["source_field"]["column"] = "TimeOfStay"
    elif change == "selected_origin":
        selection["selected_origin"] = "icu_admission"
    else:
        selection["source_identity_sha256"] = "0" * 64
    changed = ctx.model_copy(update={"user_preferences": UserPreferences(data_constraints=json.dumps(constraints))})
    with pytest.raises(ValueError):
        time_origin_selection(changed)

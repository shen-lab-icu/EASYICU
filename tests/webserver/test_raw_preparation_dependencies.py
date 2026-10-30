"""Synthetic adapters exercise generic candidate declarations, not authority."""
import pandas as pd
import pytest

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.webserver.raw_preparation_dependencies import (
    ProducerDependencyDeclaration, SourceTableRequirement, compile_producer_dependencies,
    candidate_from_scientific_review, declaration_from_provenance, pending_preparation_from_dependencies,
)
from easyicu.webserver.raw_preparation_projection import pending_preparation_projection
from tests.support.signed_case_duration import signed_case_duration_case


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_two_synthetic_producers_bind_declared_source_files_and_fields(tmp_path, fmt):
    _, plan, *_ = signed_case_duration_case()
    authority = PlanReviewAuthority.create(plan=plan)
    frame = pd.DataFrame({"case_id": [1, 2], "measurement": [3.2, 4.2]})
    relative = "table/part." + fmt
    (tmp_path / "table").mkdir()
    if fmt == "csv":
        frame.to_csv(tmp_path / relative, index=False)
    else:
        frame.to_parquet(tmp_path / relative, index=False)
    requirement = SourceTableRequirement(table="table", partitions=(relative,),
        columns=("case_id", "measurement"), format=fmt,
        partition_roster_declared_complete=True, window="first 24 hours")
    declaration = ProducerDependencyDeclaration(adapter="synthetic-" + fmt,
        source_identity_sha256="a" * 64, plan_sha256=authority.plan_sha256,
        configuration_sha256="b" * 64, implementation_sha256="c" * 64,
        requirements=(requirement,))
    dependencies = compile_producer_dependencies(source_root=tmp_path, declaration=declaration)
    assert dependencies["blocker_codes"] == []
    assert dependencies["files"][0]["path"] == relative
    assert dependencies["source_registration_verified"] is False
    assert dependencies["actual_consumption_verified"] is False
    assert dependencies["producer_read_closure_verified"] is False
    pending = pending_preparation_from_dependencies(plan=authority, dependencies=dependencies)
    assert pending.state == "pending"
    scope = pending.request.payload["raw_preparation_scope"]
    assert scope["dependencies"]["declaration_sha256"] == dependencies["declaration_sha256"]
    assert scope["analysis_authorized"] is False
    path = tmp_path / "raw_preparation_checkpoint.json"
    path.write_text(pending.model_dump_json())
    projection = pending_preparation_projection({"run_id": "synthetic", "project_dir": str(tmp_path)})
    assert projection["approvable"] is False
    assert relative in projection["summary"]


def test_unknown_partition_missing_fields_and_tampering_fail_closed(tmp_path):
    _, plan, *_ = signed_case_duration_case()
    authority = PlanReviewAuthority.create(plan=plan)
    (tmp_path / "known.csv").write_text("id,value\n1,9\n")
    requirement = SourceTableRequirement(table="observations", partitions=("known.csv",),
        columns=("id", "missing"), format="csv", partition_roster_declared_complete=False, window="all")
    declaration = ProducerDependencyDeclaration(adapter="synthetic", source_identity_sha256="a" * 64,
        plan_sha256=authority.plan_sha256, configuration_sha256="b" * 64,
        implementation_sha256="c" * 64, requirements=(requirement,))
    dependencies = compile_producer_dependencies(source_root=tmp_path, declaration=declaration)
    assert set(dependencies["blocker_codes"]) == {"dependency_partition_roster_unknown", "dependency_columns_missing"}
    with pytest.raises(ValueError, match="not_reviewable"):
        pending_preparation_from_dependencies(plan=authority, dependencies=dependencies)
    with pytest.raises(ValueError, match="not_reviewable"):
        pending_preparation_from_dependencies(plan=authority, dependencies={**dependencies, "blocker_codes": []})


def test_real_resolver_snapshot_compiles_declared_columns_without_claiming_consumption(tmp_path, monkeypatch):
    from easyicu.concept.loader import load_dictionary
    from easyicu.concept.provenance import capture_data_provenance, metadata_digest
    from easyicu.config import load_src_cfg
    from easyicu.databases.research_plan_requirements import _constraints
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.research_context.export_selection import SourceIdentity
    from easyicu.webserver import sources

    context, plan, *_ = signed_case_duration_case()
    identity = SourceIdentity.from_record(_constraints(context)["source_identity"])
    config = load_src_cfg("sic")
    snapshot = capture_data_provenance(load_dictionary(include_sofa2=True), concepts=["lact"],
        source_database="sic", class_prefixes=tuple(config.class_prefix), source_config=config,
        source_config_sha256=metadata_digest(config.model_dump(mode="json")))
    monkeypatch.setattr(sources, "registered_source_identity", lambda _: identity)
    declaration = declaration_from_provenance(plan=PlanReviewAuthority.create(plan=plan),
        context_sha256=canonical_sha256(context.model_dump(mode="json")), snapshot=snapshot,
        source_config=config, source_root=tmp_path, expected_identity=identity, window="first 24 hours")
    assert declaration.source_identity_sha256 == canonical_sha256(identity.to_dict())
    assert declaration.implementation_sha256 == snapshot.implementation_sha256
    assert any(item.table == "laboratory" and "LaboratoryValue" in item.columns
               for item in declaration.requirements)
    # Declaring a configured path cannot assert that it exists or was consumed.
    compiled = compile_producer_dependencies(source_root=tmp_path, declaration=declaration)
    assert "dependency_schema_unavailable" in compiled["blocker_codes"]
    assert compiled["producer_read_closure_verified"] is False


def test_verified_scientific_review_to_real_snapshot_stays_blocked_if_inputs_unknown(tmp_path, monkeypatch):
    from easyicu.concept.loader import load_dictionary
    from easyicu.concept.provenance import capture_data_provenance, metadata_digest
    from easyicu.config import load_src_cfg
    from tests.webserver.test_sic_raw_preparation_grant import _case

    result, source_root, _, _ = _case(tmp_path, monkeypatch, approved=False)
    record = result.evidence.get("scientific_plan_review")
    authority = PlanReviewAuthority.create(plan=result.plan, evidence_sha256={"scientific_plan_review": record.sha256})
    config = load_src_cfg("sic")
    snapshot = capture_data_provenance(load_dictionary(include_sofa2=True), concepts=["lact"],
        source_database="sic", class_prefixes=tuple(config.class_prefix), source_config=config,
        source_config_sha256=metadata_digest(config.model_dump(mode="json")))
    candidate = candidate_from_scientific_review(plan_result=result, plan_authority=authority,
        run_dir=tmp_path, source_root=source_root, snapshot=snapshot, source_config=config)
    assert candidate["status"] == "blocked"
    assert candidate["pending"] is None
    assert "dependency_cohort_inputs_unknown" in candidate["dependencies"]["blocker_codes"]
    assert candidate["dependencies"]["producer_read_closure_verified"] is False
    with pytest.raises(ValueError, match="scientific_review_missing"):
        candidate_from_scientific_review(plan_result=result,
            plan_authority=PlanReviewAuthority.create(plan=result.plan), run_dir=tmp_path,
            source_root=source_root, snapshot=snapshot, source_config=config)


def test_reviewed_cohort_predicates_enter_source_discovery_but_never_claim_closure(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from easyicu.concept.loader import load_dictionary
    from easyicu.concept.provenance import capture_data_provenance, metadata_digest
    from easyicu.config import load_src_cfg
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.orchestration.scientific_plan_review_gate import persist_or_validate_scientific_plan_review
    from easyicu.research_agent.planning.cohort_contract import CohortDefinition, ConceptPredicate, TimeWindow
    from easyicu.webserver.raw_preparation_dependencies import reviewed_plan_provenance_concepts
    from tests.research_agent.planning.test_time_origin_review_binding import _review
    from tests.webserver.test_sic_raw_preparation_grant import _case

    original, source_root, _, _ = _case(tmp_path, monkeypatch, approved=False)
    context = original.context.model_copy(update={
        'variables': [v for v in original.context.variables if v.name != 'age']})
    predicate = ConceptPredicate(concept_id='age', time_window=TimeWindow(
        anchor='icu_admission', start_offset_hours=0, end_offset_hours=24),
        aggregation='first', op='>=', value=18)
    plan = original.plan.model_copy(update={'cohort': CohortDefinition(name='typed', inclusion=(predicate,))})
    run = tmp_path / 'reviewed'; run.mkdir()
    evidence = EvidenceStore(run)
    persist_or_validate_scientific_plan_review(run_dir=run, evidence=evidence, current_review=_review(context, plan))
    record = evidence.get('scientific_plan_review')
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256={'scientific_plan_review': record.sha256})
    config = load_src_cfg('sic')
    def capture(concepts):
        return capture_data_provenance(load_dictionary(include_sofa2=True), concepts=concepts,
            source_database='sic', class_prefixes=tuple(config.class_prefix), source_config=config,
            source_config_sha256=metadata_digest(config.model_dump(mode='json')))
    result = SimpleNamespace(context=context, plan=plan, evidence=evidence)
    legacy = capture([str(item.source_concept or item.name) for item in context.variables])
    missing = candidate_from_scientific_review(plan_result=result, plan_authority=authority,
        run_dir=run, source_root=source_root, snapshot=legacy, source_config=config)
    assert 'dependency_cohort_concepts_unrecorded' in missing['dependencies']['blocker_codes']

    requested = reviewed_plan_provenance_concepts(context=context, plan=plan)
    assert 'age' in requested
    snapshot = capture(requested)
    resolved = candidate_from_scientific_review(plan_result=result, plan_authority=authority,
        run_dir=run, source_root=source_root, snapshot=snapshot, source_config=config)
    declaration = resolved['dependencies']['declaration']
    assert any(item['table'] == 'cases' and 'AgeOnAdmission' in item['columns']
               for item in declaration['requirements'])
    assert 'dependency_cohort_concepts_unrecorded' not in resolved['dependencies']['blocker_codes']
    assert 'dependency_cohort_inputs_unknown' in resolved['dependencies']['blocker_codes']
    assert resolved['status'] == 'blocked' and resolved['pending'] is None
    assert resolved['dependencies']['producer_read_closure_verified'] is False


def test_cohort_concepts_use_only_explicit_source_bindings_and_reject_conflicts():
    from types import SimpleNamespace
    from easyicu.webserver.raw_preparation_dependencies import reviewed_plan_cohort_concepts

    variable = SimpleNamespace(name='custom_age', source_concept='age')
    plan = SimpleNamespace(cohort=SimpleNamespace(
        inclusion=[SimpleNamespace(concept_id='custom_age')],
        exclusion=[SimpleNamespace(concept_id='lact')]))
    assert reviewed_plan_cohort_concepts(context=SimpleNamespace(variables=[variable]), plan=plan) == ('age', 'lact')
    conflict = SimpleNamespace(name='custom_age', source_concept='sex')
    with pytest.raises(ValueError, match='dependency_variable_source_ambiguous'):
        reviewed_plan_cohort_concepts(context=SimpleNamespace(variables=[variable, conflict]), plan=plan)

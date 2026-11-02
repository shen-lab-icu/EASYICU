"""Exercise actual producer, scope, capsule and replay using synthetic host stubs.

This deliberately stubs upstream approval/source/budget authorities; it does not
prove those authorities or real patient research. No Provider is called.
"""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from tests.support.typed_export import typed_export
from easyicu.research_agent.acquisition.first_icu_stay import (
    FirstObservedIcuCoordinate,
    FLAG_COLUMN,
    REASON_COLUMN,
)
from easyicu.research_agent.authority.run_input import RunInputCapsule
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.intake.export_package import open_export_package
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ResearchContext,
    UserPreferences,
)
from easyicu.webserver import first_observed_deferred_execution as owner


@pytest.mark.parametrize("changed_patient", [False, True])
@pytest.mark.parametrize("extra_age", [False, True])
def test_synthetic_producer_once_and_signed_reopen(
    tmp_path, monkeypatch, changed_patient, extra_age
):
    from easyicu import state_paths
    from easyicu.webserver import (
        agent_pipeline_runs,
        first_observed_preparation,
        first_observed_run_guard,
        research_auto_resume,
    )

    monkeypatch.setattr(state_paths, "state_root", lambda: tmp_path / "state")
    export = typed_export(tmp_path / "export", database="eicu")
    raw = tmp_path / "raw"
    raw.mkdir()
    patient = raw / "patient.parquet"
    pd.DataFrame(
        {
            "patientunitstayid": [1, 2],
            "hospitaldischargestatus": ["Alive", "Expired"],
            "hospitaldischargeoffset": [100, 200],
            "unitdischargeoffset": [90, 190],
            "gender": ["Female", "Male"],
        }
    ).to_parquet(patient, index=False)
    patient_sha = hashlib.sha256(patient.read_bytes()).hexdigest()
    manifest_path = export / "_manifest.json"
    manifest_body = json.loads(manifest_path.read_text())
    manifest_body["data_path"] = str(raw)
    manifest_path.write_text(json.dumps(manifest_body))
    with open_export_package(export) as pkg:
        manifest = pkg.manifest_sha256
        columns = [
            {
                "file": p.relative_path,
                "columns": list(p.columns),
                "module": Path(p.relative_path).stem,
            }
            for p in pkg.files
        ]
    coordinate_path = tmp_path / "coordinate.parquet"
    pd.DataFrame(
        {
            "stay_id": [1, 2],
            FLAG_COLUMN: [True, False],
            REASON_COLUMN: ["kept", "non_selected_unit"],
        }
    ).to_parquet(coordinate_path)
    coordinate = FirstObservedIcuCoordinate(
        path=coordinate_path,
        sha256=hashlib.sha256(coordinate_path.read_bytes()).hexdigest(),
        archive_sha256="a" * 64,
        patient_table_sha256=patient_sha,
        export_manifest_sha256=manifest,
        plan_sha256="d" * 64,
        preparation_checkpoint_sha256="e" * 64,
        preparation_run_id="source",
        preparation_binding_sha256="f" * 64,
    )
    source = {
        "database": "eicu_demo",
        "registration_id": "synthetic",
        "archive_sha256": "a" * 64,
    }
    context = ResearchContext(
        research_question="synthetic",
        cohort=CohortDescriptor(
            cohort_name="synthetic", database="eicu_demo", n_stays=0
        ),
        variables=[],
        target_outcome="death",
        primary_exposure="lact",
        user_preferences=UserPreferences(
            data_constraints=json.dumps({"source_identity": source})
        ),
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "research_context.json").write_text(context.model_dump_json())
    identity = {"research_question": "synthetic"}
    capsule = RunInputCapsule(
        scientific_identity=identity,
        scientific_identity_sha256=canonical_sha256(identity),
        context_sha256="a" * 64,
        cohort_sha256="b" * 64,
        initial_environment={},
    )
    (run / "run_input_capsule.json").write_text(capsule.model_dump_json())
    (run / "first_observed_deferred_recipe.json").write_text("{}")
    proof = SimpleNamespace()  # asdict requires a real frozen proof
    from easyicu.webserver.first_observed_run_guard import FirstObservedPreparationProof

    proof = FirstObservedPreparationProof(
        coordinate_sha256=coordinate.sha256,
        preparation_checkpoint_sha256="e" * 64,
        preparation_run_id="source",
        source_identity_sha256="f" * 64,
        criterion_sha256="1" * 64,
        export_manifest_sha256=manifest,
    )
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLedger,
    )
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver.provider_adapter import web_research_agent_hard_stop_limits

    limits = web_research_agent_hard_stop_limits("planner_canary")
    ledger_path = tmp_path / "budget.json"
    ledger = ProviderHardStopLedger(
        path=ledger_path,
        task_ids=("task",),
        limits=limits,
        batch_id="task",
        declaration_sha256="3" * 64,
    )
    task = ledger.start_task("task")
    HardStopClient(
        ScriptedMockLLMClient(["prior"]), role="planner", task=task
    ).complete([LLMMessage(role="user", content="synthetic")], max_tokens=32)
    prior_calls = ledger.snapshot()["tasks"][0]["calls"]
    recovery = SimpleNamespace(
        record_sha256="2" * 64,
        hard_stop_task_id="task",
        hard_stop_ledger_path=str(ledger_path),
        hard_stop_declaration_sha256="3" * 64,
        budget_mode="planner_canary",
    )
    from easyicu.webserver import durable_research_store

    monkeypatch.setattr(durable_research_store, "runtime_digest", lambda: "6" * 64)
    recipe = {
        "decoded_scope_enforced": True,
        "materializer_runtime_sha256": "6" * 64,
        "host_binding": {
            "preparation_run_id": "source",
            "study_context_id": "study",
            "project_id": "project",
            "session_id": "session",
        },
        "materialization": {
            "producer": "easyicu.research_agent.acquisition.foundation.acquire_universe_for_question",
            "selection_authority": "host_exact",
            "database": "eicu",
            "emit_trajectory": False,
            "additional_population_filters": False,
            "target_outcome": "death",
            "primary_exposure": "lact",
            "outcome_concepts": ["death"],
            "required_feature_concepts": ["lact"],
            "static_concepts": ["age"],
            "allowed_modules": ["labs", "outcomes"],
            "cohort_window_hours": [0, 24],
        },
        "budget": {
            "execution_mode": "deferred_reviewed",
            "reset_or_increase_authorized": False,
            "recovery_record_sha256": "2" * 64,
            "hard_stop_task_id": "task",
            "hard_stop_declaration_sha256": "3" * 64,
            "planning_mode": "planner_canary",
            "limits": limits.canonical_payload(),
        },
        "source_manifest_sha256": manifest,
        "criterion": {
            "kind": "first_observed_icu_single_hospital_adult",
            "role": "include",
        },
        "preparation_proof": asdict(proof),
        "verified_source_files": {
            "archive_sha256": "a" * 64,
            "patient_table_sha256": patient_sha,
        },
        "declared_schema_columns": columns,
    }
    if extra_age:
        recipe["criterion"].update({"id":"c1", "source":"question", "quote":"adult first ICU"})
        recipe["population_spec"] = {"criteria": [recipe["criterion"],
            {"id":"c2", "kind":"age_years", "role":"include", "source":"question",
             "quote":"age at least 18", "min_years":18}]}
    from easyicu.research_agent.research_context.export_selection import SourceIdentity

    recipe["preparation_proof"]["source_identity_sha256"] = canonical_sha256(
        SourceIdentity.from_record(source).to_dict()
    )
    recipe["preparation_proof"]["criterion_sha256"] = canonical_sha256(
        recipe["criterion"]
    )
    proof = replace(
        proof,
        source_identity_sha256=recipe["preparation_proof"]["source_identity_sha256"],
        criterion_sha256=recipe["preparation_proof"]["criterion_sha256"],
    )
    monkeypatch.setattr(
        owner, "_load_signed_source_flow", lambda _: (coordinate, {}, "7" * 64)
    )
    monkeypatch.setattr(owner, "verify_approved_deferred_review", lambda **_: recipe)
    # Source-flow registration is verified separately against the real EvidenceStore.
    published = []
    monkeypatch.setattr(
        owner, "_publish_source_flow", lambda **kwargs: published.append(kwargs)
    )
    # This producer test isolates source/materialization from the separately tested plan gate.
    monkeypatch.setattr(
        owner, "_validate_actual_plan_columns", lambda **_: ("lact", "death")
    )
    monkeypatch.setattr(
        agent_pipeline_runs, "get_review_recovery_record", lambda _: recovery
    )
    monkeypatch.setattr(
        research_auto_resume,
        "_original_task",
        lambda _: pytest.fail("reopened active task"),
    )
    monkeypatch.setattr(
        first_observed_run_guard,
        "verify_first_observed_preparation_recipe",
        lambda **_: proof,
    )
    monkeypatch.setattr(
        first_observed_run_guard,
        "resolve_host_prepared_first_observed_coordinate",
        lambda **_: coordinate,
    )
    monkeypatch.setattr(
        first_observed_run_guard,
        "verify_prepared_first_observed_population",
        lambda **_: None,
    )
    monkeypatch.setattr(
        first_observed_preparation,
        "_metadata_source",
        lambda _: (export, {"export_manifest_sha256": manifest}),
    )
    # The caller cannot substitute a task from another durable ledger.
    other_ledger = ProviderHardStopLedger(
        path=tmp_path / "other-budget.json",
        task_ids=("task",),
        limits=limits,
        batch_id="task",
        declaration_sha256="3" * 64,
    )
    other_task = other_ledger.start_task("task")
    with pytest.raises(ValueError, match="budget_changed"):
        owner.consume_approved_first_observed_input(
            run_dir=run,
            approved_plan_sha256="4" * 64,
            decision_set_sha256="5" * 64,
            provider_task=other_task,
        )
    assert not (run / "deferred_input").exists()
    if changed_patient:
        changed = pd.read_parquet(patient)
        changed.loc[0, "gender"] = "Changed"
        changed.to_parquet(patient, index=False)
        from easyicu.research_agent.acquisition import foundation

        monkeypatch.setattr(
            foundation,
            "acquire_universe_for_question",
            lambda **_: pytest.fail("decoded after patient hash changed"),
        )
        with pytest.raises(ValueError, match="patient_source_changed"):
            owner.consume_approved_first_observed_input(
                run_dir=run,
                approved_plan_sha256="4" * 64,
                decision_set_sha256="5" * 64,
                provider_task=task,
            )
        assert not (run / "deferred_input").exists()
        return
    first = owner.consume_approved_first_observed_input(
        run_dir=run,
        approved_plan_sha256="4" * 64,
        decision_set_sha256="5" * 64,
        provider_task=task,
    )
    assert first.cohort_path.is_file()
    from easyicu.research_agent.acquisition import foundation

    monkeypatch.setattr(
        foundation,
        "acquire_universe_for_question",
        lambda **_: pytest.fail("reopened by rematerializing"),
    )
    second = owner.consume_approved_first_observed_input(
        run_dir=run,
        approved_plan_sha256="4" * 64,
        decision_set_sha256="5" * 64,
        provider_task=task,
    )
    assert second.receipt_sha256 == first.receipt_sha256
    assert ledger.snapshot()["tasks"][0]["calls"] == prior_calls
    assert published == []  # No exact source-flow evidence during execution.
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.run_input import (
        DEFERRED_ACTUAL_CAPSULE_EVIDENCE_ID,
        DEFERRED_RECEIPT_EVIDENCE_ID,
    )

    evidence = EvidenceStore(run)
    evidence.register_file(
        kind="log",
        description="synthetic receipt",
        source_path=run / "deferred_first_observed_receipt.json",
        evidence_id=DEFERRED_RECEIPT_EVIDENCE_ID,
    )
    evidence.register_file(
        kind="log",
        description="synthetic capsule",
        source_path=run / "deferred_input" / "run_input_capsule.json",
        evidence_id=DEFERRED_ACTUAL_CAPSULE_EVIDENCE_ID,
    )
    late_args = dict(
        run_dir=run,
        approved_plan_sha256="4" * 64,
        decision_set_sha256="5" * 64,
        evidence_store=evidence,
    )
    from easyicu.webserver import raw_source_authority

    raw_resolver = raw_source_authority.resolve_raw_source_identity_binding
    monkeypatch.setattr(
        raw_source_authority,
        "resolve_raw_source_identity_binding",
        lambda **_: pytest.fail("late report reread patient"),
    )
    owner.publish_approved_source_flow(**late_args)
    assert len(published) == 1
    receipt = run / "deferred_first_observed_receipt.json"
    receipt_bytes = receipt.read_bytes()
    receipt.write_bytes(receipt_bytes + b" ")
    with pytest.raises(ValueError, match="receipt_invalid"):
        owner.publish_approved_source_flow(**late_args)
    assert len(published) == 1
    receipt.write_bytes(receipt_bytes)
    capsule = run / "deferred_input" / "run_input_capsule.json"
    capsule_bytes = capsule.read_bytes()
    capsule.write_bytes(capsule_bytes + b" ")
    with pytest.raises(ValueError, match="receipt_invalid"):
        owner.publish_approved_source_flow(**late_args)
    assert len(published) == 1
    capsule.write_bytes(capsule_bytes)
    monkeypatch.setattr(
        raw_source_authority, "resolve_raw_source_identity_binding", raw_resolver
    )
    external = tmp_path / "external"
    external.write_text("must not be read")
    context_path = run / "deferred_input" / "research_context.json"
    context_path.unlink()
    context_path.symlink_to(external)
    from easyicu.research_agent.authority.filesystem import AuthorityFilesystemError

    with pytest.raises((ValueError, OSError, AuthorityFilesystemError)):
        owner.consume_approved_first_observed_input(
            run_dir=run,
            approved_plan_sha256="4" * 64,
            decision_set_sha256="5" * 64,
            provider_task=task,
        )


def owner_runtime_digest():
    from easyicu.webserver.durable_research_store import runtime_digest

    return runtime_digest()


def test_consumer_without_approved_checkpoint_never_opens_source(tmp_path, monkeypatch):
    from easyicu.webserver import first_observed_preparation

    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(
        first_observed_preparation,
        "_metadata_source",
        lambda *_: pytest.fail("source metadata read before an approved checkpoint"),
    )
    with pytest.raises((ValueError, OSError, RuntimeError)):
        owner.consume_approved_first_observed_input(
            run_dir=run, approved_plan_sha256="a" * 64, decision_set_sha256="b" * 64
        )
    assert not (run / "deferred_input").exists()


def test_actual_materialized_columns_must_match_approved_plan(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
    from easyicu.research_agent.contracts.declared_product import typed_product
    from tests.support.descriptive_plan import _plan

    plan = _plan()
    distribution = next(
        step.exposure_outcome_distribution_spec
        for step in plan.steps
        if step.exposure_outcome_distribution_spec is not None
    )
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256={})
    monkeypatch.setattr(
        owner,
        "load_checkpoint",
        lambda *a, **k: SimpleNamespace(
            requests=[
                SimpleNamespace(
                    payload={"plan_review_authority": authority.model_dump(mode="json")}
                )
            ]
        ),
    )
    columns = {
        name
        for step in plan.steps
        for name in step.inputs
        if typed_product(name) is None
    }
    columns.update((distribution.exposure, distribution.outcome))
    material = {
        "analysis_columns": {
            "exposure": distribution.exposure,
            "outcome": distribution.outcome,
        }
    }
    assert owner._validate_actual_plan_columns(
        run_dir=tmp_path, material=material, cohort_columns=tuple(columns)
    ) == (distribution.exposure, distribution.outcome)
    with pytest.raises(ValueError, match="analysis_columns_missing"):
        owner._validate_actual_plan_columns(
            run_dir=tmp_path,
            material=material,
            cohort_columns=tuple(columns - {distribution.outcome}),
        )
    with pytest.raises(ValueError, match="analysis_columns_invalid"):
        owner._validate_actual_plan_columns(
            run_dir=tmp_path,
            material={
                "analysis_columns": {
                    "exposure": "different",
                    "outcome": distribution.outcome,
                }
            },
            cohort_columns=tuple(columns),
        )


def test_additional_age_scope_is_checked_before_source_access(tmp_path, monkeypatch):
    from easyicu.webserver import first_observed_preparation
    criterion = {'id':'first','kind':'first_observed_icu_single_hospital_adult',
                 'role':'include','source':'question','quote':'adult first ICU'}
    recipe = {'criterion': criterion, 'population_spec': {'criteria': [criterion,
        {'id':'age','kind':'age_years','role':'include','source':'question',
         'quote':'age at least 21','min_years':21}]}, 'decoded_scope_enforced':True}
    monkeypatch.setattr(owner, 'verify_approved_deferred_review', lambda **kwargs: recipe)
    monkeypatch.setattr(first_observed_preparation, '_metadata_source', lambda *_: pytest.fail('source was accessed'))
    with pytest.raises(owner.DeferredFirstObservedError, match='population_changed'):
        owner.consume_approved_first_observed_input(run_dir=tmp_path,
            approved_plan_sha256='4'*64, decision_set_sha256='5'*64)

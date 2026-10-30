"""Metadata outline replay preserves rejected feedback and original budgets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
from easyicu.research_agent.planning import outline_retry_journal as owner
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.providers.structured_retry import call_llm_with_structured_retry
from easyicu.webserver import provider_adapter


def _setup(tmp_path, monkeypatch):
    import easyicu.research_agent.providers.llm  # noqa: F401
    monkeypatch.setattr(owner.state_paths, "state_root", lambda: tmp_path / "state")
    env = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
           "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
           "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
    inner, _ = provider_adapter.build_research_agent_provider_client({"provider": "openai", "external": True}, environ=env)
    book = ProviderHardStopLedger(path=tmp_path / "budget.json", task_ids=("task",),
        limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
        batch_id="task", declaration_sha256="a" * 64)
    return inner, book, book.start_task("task")


def _journal(inner, task, run_dir, **kwargs):
    return owner.OutlineRetryJournal(inner=inner, task=task, run_dir=run_dir,
        request_sha256="b" * 64, dependency_sha256="c" * 64, literature_sha256="e" * 64, **kwargs)


def test_rejected_response_replayed_without_new_transport_or_lost_feedback(tmp_path, monkeypatch):
    # Fake transports exercise retry semantics; real worker ownership is covered by SIGKILL.
    monkeypatch.setattr(owner, "_verify_pending_guard", lambda *_args: None)
    inner, book, task = _setup(tmp_path, monkeypatch)
    calls = []
    def fake(_inner, messages, **_kwargs):
        calls.append([m.content for m in messages])
        rows = book.snapshot()["tasks"][0]["calls"]
        prior = rows[-1]["attempt_id"] if rows and rows[-1]["state"] == "in_progress" else None
        item = book.reserve_transport_attempt(task_id="task", role="planner", model="synthetic",
            messages=messages, max_tokens=16, prior_attempt_id=prior)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        book.finish_transport_attempt(task_id="task", attempt_id=item,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, error_type=None)
        return '{"value":' if len(calls) == 1 else '{"value":1}'
    monkeypatch.setattr(owner, "authorized_complete", fake)
    initial = _journal(inner, task, tmp_path / "source")
    kwargs = {"parser": json.loads, "role": "progressive_planner_outline", "max_retries": 3, "max_tokens": 16}
    with pytest.raises(KeyboardInterrupt):
        call_llm_with_structured_retry(initial, [LLMMessage("user", "synthetic")], **kwargs)
    source = tmp_path / "source" / ".runtime" / owner.FILE_NAME
    raw = source.read_bytes()
    resumed = _journal(inner, task, tmp_path / "continued", source_path=source,
                       source_sha256=hashlib.sha256(raw).hexdigest())
    assert call_llm_with_structured_retry(resumed, [LLMMessage("user", "synthetic")], **kwargs) == {"value": 1}
    assert len(calls) == 3
    assert calls[1] == calls[2]  # The retry retains its exact correction context.
    assert owner.load_retry_journal(tmp_path / "continued" / ".runtime" / owner.FILE_NAME)["abandoned"] == 1
    assert len(book.snapshot()["tasks"][0]["calls"]) == 3


def test_retry_journal_refuses_tampering_drift_or_other_prompt(tmp_path, monkeypatch):
    inner, _book, task = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, "authorized_complete", lambda *_args, **_kw: "invalid")
    initial = _journal(inner, task, tmp_path / "source")
    assert initial.complete([LLMMessage("user", "original")], max_tokens=16, temperature=0.2) == "invalid"
    source = tmp_path / "source" / ".runtime" / owner.FILE_NAME
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    resumed = _journal(inner, task, tmp_path / "continued", source_path=source, source_sha256=digest)
    with pytest.raises(ValueError, match="request_mismatch"):
        resumed.complete([LLMMessage("user", "different")], max_tokens=16, temperature=0.2)
    monkeypatch.setattr(owner, "_code_digest", lambda: "d" * 64)
    with pytest.raises(ValueError, match="authority_mismatch"):
        _journal(inner, task, tmp_path / "drift", source_path=source, source_sha256=digest)
    tampered = json.loads(source.read_text())
    tampered["records"][0]["response"] = "forged"
    source.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="signature_invalid"):
        owner.load_retry_journal(source)


def test_progressive_planner_rebuilds_rejected_outline_before_real_retry(tmp_path, monkeypatch):
    # Fake transports exercise retry semantics; real worker ownership is covered by SIGKILL.
    monkeypatch.setattr(owner, "_verify_pending_guard", lambda *_args: None)
    from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from tests.research_agent.planning.progressive_planner_fixtures import _context, _outline_payload

    _inner, book, task = _setup(tmp_path, monkeypatch)
    inner = ScriptedMockLLMClient([])
    inner.supports_strict_json_schema = True
    calls = []
    def fake(_inner, messages, **_kwargs):
        calls.append([m.content for m in messages])
        rows = book.snapshot()["tasks"][0]["calls"]
        prior = rows[-1]["attempt_id"] if rows and rows[-1]["state"] == "in_progress" else None
        attempt = book.reserve_transport_attempt(task_id="task", role="planner", model="synthetic",
            messages=messages, max_tokens=16, prior_attempt_id=prior)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        book.finish_transport_attempt(task_id="task", attempt_id=attempt,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, error_type=None)
        return '{"bad":"not-an-outline"}' if len(calls) == 1 else json.dumps(_outline_payload())
    monkeypatch.setattr(owner, "authorized_complete", fake)
    dependencies = {"cohort_file_sha256": "b" * 64, "llm_signature": "synthetic", "prompt_version": "v1"}
    context = _context()
    with pytest.raises(KeyboardInterrupt):
        ProgressivePlannerAgent(inner).run(context, stop_after_outline=True,
            resume_dependency_context=dependencies,
            outline_retry_journal={"task": task, "run_dir": tmp_path / "source", "literature_sha256": "e" * 64})
    source = tmp_path / "source" / ".runtime" / owner.FILE_NAME
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = ProgressivePlannerAgent(inner).run(context, stop_after_outline=True,
        resume_dependency_context=dependencies,
        outline_retry_journal={"task": task, "run_dir": tmp_path / "continued",
                               "source_path": source, "source_sha256": digest, "literature_sha256": "e" * 64})
    assert output.steps
    assert len(calls) == 3
    assert calls[1] == calls[2]


def test_web_lookup_accepts_only_authenticated_new_private_prefix(tmp_path, monkeypatch):
    from easyicu.webserver.research_launch_resume import _development_progressive_resume_binding
    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    inner, _book, task = _setup(tmp_path, monkeypatch)
    run = tmp_path / "study-a" / "run_original" / "pipeline" / "run_inner"
    journal = _journal(inner, task, run)
    source, digest = _development_progressive_resume_binding(project_root=str(tmp_path), study_id="study-a",
        source_job_id="original", budget_mode="planner_canary")
    assert source == run / ".runtime" / owner.FILE_NAME
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert owner.source_run_dir(source) == run
    with pytest.raises(ResearchPipelineRunError):
        _development_progressive_resume_binding(project_root=str(tmp_path), study_id="study-a",
            source_job_id="original", budget_mode="full_reviewed")
    source.write_text(source.read_text().replace('"abandoned":0', '"abandoned":1'))
    with pytest.raises(ResearchPipelineRunError):
        _development_progressive_resume_binding(project_root=str(tmp_path), study_id="study-a",
            source_job_id="original", budget_mode="planner_canary")
    assert journal is not None


def test_pipeline_owner_loads_private_prefix_and_revalidates_same_outline(tmp_path, monkeypatch):
    # Fake transports exercise retry semantics; real worker ownership is covered by SIGKILL.
    monkeypatch.setattr(owner, "_verify_pending_guard", lambda *_args: None)
    from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
    from easyicu.research_agent.orchestration.progressive_planning import run_progressive_planner
    from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from tests.research_agent.planning.progressive_planner_fixtures import _context, _outline_payload
    from tests.research_agent.planning.test_progressive_planner_checkpoints import _RecordingEvidence

    _inner, book, task = _setup(tmp_path, monkeypatch)
    inner = ScriptedMockLLMClient([])
    inner.supports_strict_json_schema = True
    calls = []
    def fake(_inner, messages, **_kwargs):
        calls.append([m.content for m in messages])
        rows = book.snapshot()["tasks"][0]["calls"]
        prior = rows[-1]["attempt_id"] if rows and rows[-1]["state"] == "in_progress" else None
        attempt = book.reserve_transport_attempt(task_id="task", role="planner", model="synthetic",
            messages=messages, max_tokens=16, prior_attempt_id=prior)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        book.finish_transport_attempt(task_id="task", attempt_id=attempt,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, error_type=None)
        return '{"bad":"not-an-outline"}' if len(calls) == 1 else json.dumps(_outline_payload())
    monkeypatch.setattr(owner, "authorized_complete", fake)
    cohort = tmp_path / "catalog.parquet"
    cohort.write_bytes(b"zero-row synthetic catalog identity")
    shared = {"context": _context(), "evidence": _RecordingEvidence(), "prompt_pack_version": "test-v1",
              "cohort_path": cohort, "llm_signature": "mock:test", "planner_kwargs": {},
              "know_how_binding": PlannerKnowHowBinding(), "planning_contract_context": "",
              "finding_sink": lambda _finding: None, "stop_after_outline": True,
              "metadata_outline_journal_task": task}
    shared["metadata_outline_literature_sha256"] = "e" * 64
    with pytest.raises(KeyboardInterrupt):
        run_progressive_planner(planner=ProgressivePlannerAgent(inner), run_dir=tmp_path / "source",
            resume_checkpoint_path=None, resume_checkpoint_sha256=None, **shared)
    source = tmp_path / "source" / ".runtime" / owner.FILE_NAME
    result = run_progressive_planner(planner=ProgressivePlannerAgent(inner), run_dir=tmp_path / "continued",
        resume_checkpoint_path=source, resume_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), **shared)
    assert result.outline.steps
    assert len(calls) == 3 and calls[1] == calls[2]


def test_replay_rebuilds_the_one_shot_capability_gap_confirmation(tmp_path, monkeypatch):
    # Fake transports exercise retry semantics; real worker ownership is covered by SIGKILL.
    monkeypatch.setattr(owner, "_verify_pending_guard", lambda *_args: None)
    from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from tests.research_agent.planning.progressive_planner_fixtures import _context, _outline_payload

    _inner, book, task = _setup(tmp_path, monkeypatch)
    inner = ScriptedMockLLMClient([])
    inner.supports_strict_json_schema = True
    calls = []
    response = {**_outline_payload(), "steps": [], "capability_gap": "design_element_unsupported",
                "capability_gap_concept": None,
                "rationale": "The synthetic question requires a design beyond the offered families."}
    def fake(_inner, messages, **_kwargs):
        calls.append([m.content for m in messages])
        rows = book.snapshot()["tasks"][0]["calls"]
        prior = rows[-1]["attempt_id"] if rows and rows[-1]["state"] == "in_progress" else None
        attempt = book.reserve_transport_attempt(task_id="task", role="planner", model="synthetic",
            messages=messages, max_tokens=16, prior_attempt_id=prior)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        book.finish_transport_attempt(task_id="task", attempt_id=attempt,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, error_type=None)
        return json.dumps(response)
    monkeypatch.setattr(owner, "authorized_complete", fake)
    dependencies = {"cohort_file_sha256": "b" * 64, "llm_signature": "synthetic", "prompt_version": "v1"}
    context = _context()
    with pytest.raises(KeyboardInterrupt):
        ProgressivePlannerAgent(inner).run(context, stop_after_outline=True,
            resume_dependency_context=dependencies,
            outline_retry_journal={"task": task, "run_dir": tmp_path / "source", "literature_sha256": "e" * 64})
    source = tmp_path / "source" / ".runtime" / owner.FILE_NAME
    with pytest.raises(ProgressivePlanCompileError) as stopped:
        ProgressivePlannerAgent(inner).run(context, stop_after_outline=True,
            resume_dependency_context=dependencies,
            outline_retry_journal={"task": task, "run_dir": tmp_path / "continued",
                                   "source_path": source, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                   "literature_sha256": "e" * 64})
    assert stopped.value.reason_code == "progressive_capability_gap"
    assert len(calls) == 3 and calls[1] == calls[2]


def test_web_scope_finds_original_metadata_only_literature_and_catalog(tmp_path, monkeypatch):
    import pandas as pd
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.webserver import agent_review_recovery, research_launch_resume, study_contexts

    inner, _book, task = _setup(tmp_path, monkeypatch)
    study = {"id": "study-a", "question": "synthetic", "data_source": {"path": str(tmp_path / "source"), "database": "miiv"}}
    wrapper = tmp_path / "study-a" / "run_original"
    run = wrapper / "pipeline" / "run_inner"
    _journal(inner, task, run)
    inputs = wrapper / "pipeline_input"
    inputs.mkdir()
    selected = ["age"]
    selected_sha = hashlib.sha256(json.dumps(selected, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    pd.DataFrame({"stay_id": pd.Series(dtype="int64"), "age": pd.Series(dtype="float64")}).to_parquet(inputs / "planner_catalog.parquet", index=False)
    (inputs / "planner_catalog_receipt.json").write_text(json.dumps({
        "schema_version": "easyicu.metadata-only-planning-catalog/1", "database": "miiv",
        "row_identity_column": "stay_id", "selected_concepts": selected, "selected_concepts_sha256": selected_sha,
        "patient_rows_read": False, "patient_rows_written": False, "observed_feasibility_claims": False,
        "execution_authorized": False, "operationalized_columns": [], "planning_target_outcome": None,
        "planning_endpoint": None,
    }))
    (run / "preplan_literature_bundle.json").write_text(json.dumps({"research_question": "synthetic", "citations": []}))
    config = PipelineConfig(workdir=wrapper / "pipeline", runner_kind="docker")
    seed = agent_review_recovery.WebReviewRecoverySeed.create(
        wrapper_dir=str(wrapper), study=study, scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(study),
        provider_meta={"provider": "openai", "external": True}, provider_public={"provider": "openai", "model": "synthetic"},
        credential_source="pi_verified", budget_mode="planner_canary", prepared_package_binding=None,
        pipeline_config=config.recovery_payload(), pipeline_config_sha256=config.canonical_digest(),
        acquisition_projection={"selected_concepts": selected}, hard_stop_ledger_path=str(task.ledger.path),
        hard_stop_task_id=task.task_id, hard_stop_declaration_sha256="a" * 64, created_at=1.0,
    )
    agent_review_recovery.put_recovery_seed(seed)
    checkpoint, _ = research_launch_resume._development_progressive_resume_binding(project_root=str(tmp_path),
        study_id="study-a", source_job_id="original", budget_mode="planner_canary")
    scope = research_launch_resume._development_resume_launch_scope(project_root=str(tmp_path), study=study, source_job_id="original")
    profile = research_launch_resume._development_resume_acquisition_profile(checkpoint_path=checkpoint, database="miiv",
        cohort_window=(0, 24), outcome_concepts=[], static_concepts=[], required_feature_concepts=[])
    assert scope.budget_mode == "planner_canary" and scope.prepared_grouping is None
    assert profile.kind == "metadata_only_planning_catalog" and profile.selected_concepts == ("age",)
    assert research_launch_resume._development_resume_literature_bundle(checkpoint_path=checkpoint)["research_question"] == "synthetic"


def test_signed_journal_rollback_and_inflight_prompt_drift_are_refused(tmp_path, monkeypatch):
    # Fake transports exercise retry semantics; real worker ownership is covered by SIGKILL.
    monkeypatch.setattr(owner, "_verify_pending_guard", lambda *_args: None)
    inner, book, task = _setup(tmp_path, monkeypatch)
    initial = _journal(inner, task, tmp_path / "source")
    source = tmp_path / "source" / ".runtime" / owner.FILE_NAME
    old_signed = source.read_bytes()
    def interrupted(_inner, messages, **_kwargs):
        book.reserve_transport_attempt(task_id="task", role="planner", model="synthetic",
            messages=messages, max_tokens=16, prior_attempt_id=None)
        raise KeyboardInterrupt()
    monkeypatch.setattr(owner, "authorized_complete", interrupted)
    with pytest.raises(KeyboardInterrupt):
        initial.complete([LLMMessage("user", "original")], max_tokens=16, temperature=0.2)
    pending = source.read_bytes()
    resumed = _journal(inner, task, tmp_path / "continued", source_path=source,
                       source_sha256=hashlib.sha256(pending).hexdigest())
    with pytest.raises(ValueError, match="request_mismatch"):
        resumed.complete([LLMMessage("user", "changed")], max_tokens=16, temperature=0.2)
    source.write_bytes(old_signed)
    with pytest.raises(ValueError, match="ledger_changed"):
        _journal(inner, task, tmp_path / "rollback", source_path=source,
                 source_sha256=hashlib.sha256(old_signed).hexdigest())
    with pytest.raises(ValueError, match="authority_mismatch"):
        owner.OutlineRetryJournal(inner=inner, task=task, run_dir=tmp_path / "other-lit", request_sha256="b" * 64,
            dependency_sha256="c" * 64, literature_sha256="f" * 64, source_path=source,
            source_sha256=hashlib.sha256(old_signed).hexdigest())


def test_an_unverified_inflight_reservation_cannot_resume_the_journal(tmp_path, monkeypatch):
    inner, book, task = _setup(tmp_path, monkeypatch)
    initial = _journal(inner, task, tmp_path / "source")
    def interrupted(_inner, messages, **_kwargs):
        book.reserve_transport_attempt(task_id="task", role="planner", model="synthetic",
            messages=messages, max_tokens=16, prior_attempt_id=None)
        raise KeyboardInterrupt()
    monkeypatch.setattr(owner, "authorized_complete", interrupted)
    with pytest.raises(KeyboardInterrupt):
        initial.complete([LLMMessage("user", "original")], max_tokens=16, temperature=0.2)
    source = tmp_path / "source" / ".runtime" / owner.FILE_NAME
    with pytest.raises(ValueError, match="guard_unverified"):
        _journal(inner, task, tmp_path / "continued", source_path=source,
                 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())


def test_pending_guard_rejects_another_request_digest(tmp_path, monkeypatch):
    from easyicu.webserver import metadata_provider_concurrency as slots
    from easyicu.webserver.metadata_provider_worker import identity as worker_identity

    _inner, book, task = _setup(tmp_path, monkeypatch)
    attempt = book.reserve_transport_attempt(task_id="task", role="planner", model="synthetic",
        messages=[LLMMessage("user", "original")], max_tokens=16, prior_attempt_id=None)
    call = book.snapshot()["tasks"][0]["calls"][-1]
    held = {"task_id": task.task_id, "ledger_path": str(task.ledger.path),
            "guard_path": str(Path(task.ledger.path).parent / ".metadata-provider-execution" / attempt),
            "request_sha256": "f" * 64, "worker_sha256": worker_identity(), "guard_nonce": "synthetic"}
    monkeypatch.setattr(slots, "get", lambda _attempt: held)
    with pytest.raises(ValueError, match="guard_unverified"):
        owner._verify_pending_guard(task, call, "a" * 64)


def test_lost_attempt_does_not_expand_the_normal_retry_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, "_verify_pending_guard", lambda *_args: None)
    inner, book, task = _setup(tmp_path, monkeypatch)
    count = []
    def fake(_inner, messages, **_kwargs):
        count.append(1)
        attempt = book.reserve_transport_attempt(task_id="task", role="planner", model="synthetic",
            messages=messages, max_tokens=16, prior_attempt_id=None)
        if len(count) == 4:
            raise KeyboardInterrupt()
        book.finish_transport_attempt(task_id="task", attempt_id=attempt,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, error_type=None)
        return "bad"
    monkeypatch.setattr(owner, "authorized_complete", fake)
    initial = _journal(inner, task, tmp_path / "source")
    for _ in range(3):
        assert initial.complete([LLMMessage("user", "same")], max_tokens=16, temperature=0.2) == "bad"
    with pytest.raises(KeyboardInterrupt):
        initial.complete([LLMMessage("user", "same")], max_tokens=16, temperature=0.2)
    source = tmp_path / "source" / ".runtime" / owner.FILE_NAME
    resumed = _journal(inner, task, tmp_path / "continued", source_path=source,
                       source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    for _ in range(3):
        assert resumed.complete([LLMMessage("user", "same")], max_tokens=16, temperature=0.2) == "bad"
    with pytest.raises(ValueError, match="attempt_limit"):
        resumed.complete([LLMMessage("user", "same")], max_tokens=16, temperature=0.2)
    assert len(count) == 4


@pytest.mark.parametrize("database", ["miiv", "eicu"])
def test_family_spec_replays_rejected_request_with_its_own_budget(tmp_path, monkeypatch, database):
    from easyicu.research_agent.agents.progressive_planner import FAMILY_SPEC_STRATEGY, ProgressivePlannerAgent
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from tests.research_agent.planning.family_spec_fixtures import (
        ALLOWED_CITATIONS, DIRECT_COMPARATORS, PLANNER_ROSTER, _context, _request, _spec_payload,
    )

    monkeypatch.setattr(owner, "_verify_pending_guard", lambda *_args: None)
    _inner, book, task = _setup(tmp_path, monkeypatch)
    inner = ScriptedMockLLMClient([])
    context = _context(exact=False)
    context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"database": database})})
    request = _request(context)
    invalid = [{**item, "reference_level_index": 0} if item["name"] == "age" else item for item in PLANNER_ROSTER]
    rejected = json.dumps(_spec_payload(request, adjustment_set=invalid))
    accepted = json.dumps(_spec_payload(request, adjustment_set=PLANNER_ROSTER))
    calls = []
    def fake(_inner, messages, **_kwargs):
        calls.append([(m.role, m.content) for m in messages])
        rows = book.snapshot()["tasks"][0]["calls"]
        prior = rows[-1]["attempt_id"] if rows and rows[-1]["state"] == "in_progress" else None
        attempt = book.reserve_transport_attempt(task_id="task", role="planner", model="synthetic",
            messages=messages, max_tokens=16, prior_attempt_id=prior)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        book.finish_transport_attempt(task_id="task", attempt_id=attempt,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, error_type=None)
        return rejected if len(calls) == 1 else accepted
    monkeypatch.setattr(owner, "authorized_complete", fake)
    dependencies = {"cohort_file_sha256": "b" * 64, "llm_signature": "synthetic", "prompt_version": "v1"}
    options = {"planner_strategy": FAMILY_SPEC_STRATEGY, "allowed_literature_citation_keys": ALLOWED_CITATIONS,
               "direct_comparator_literature_keys": DIRECT_COMPARATORS, "comparison_literature_keys": DIRECT_COMPARATORS,
               "enforce_article_contract": True, "article_contract_context": context, "planning_contract_context": "",
               "required_primary_cohort_selection_mode": "predicate_filtered", "resume_dependency_context": dependencies}
    with pytest.raises(KeyboardInterrupt):
        ProgressivePlannerAgent(inner).run(context, **options,
            outline_retry_journal={"task": task, "run_dir": tmp_path / "source", "literature_sha256": "e" * 64})
    source = tmp_path / "source" / ".runtime" / owner.FILE_NAME
    result = ProgressivePlannerAgent(inner).run(context, **options,
        outline_retry_journal={"task": task, "run_dir": tmp_path / "continued", "literature_sha256": "e" * 64,
                               "source_path": source, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    assert result.steps and len(calls) == 3
    assert calls[1] == calls[2]
    assert calls[2][-2] == ("assistant", rejected)
    journal = owner.load_retry_journal(tmp_path / "continued" / ".runtime" / owner.FILE_NAME)
    assert journal["binding"]["stage"] == "family_spec" and journal["binding"]["parse_attempt_limit"] == 3
    assert journal["abandoned"] == 1


def test_family_spec_journal_cannot_be_replayed_as_outline_or_with_a_larger_budget(tmp_path, monkeypatch):
    inner, _book, task = _setup(tmp_path, monkeypatch)
    owner.OutlineRetryJournal(inner=inner, task=task, run_dir=tmp_path / "source",
        request_sha256="b" * 64, dependency_sha256="c" * 64, literature_sha256="e" * 64,
        stage="family_spec", parse_attempt_limit=3)
    source = tmp_path / "source" / ".runtime" / owner.FILE_NAME
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    for stage, limit in (("outline", 4), ("family_spec", 4)):
        with pytest.raises(ValueError, match="authority_mismatch"):
            owner.OutlineRetryJournal(inner=inner, task=task, run_dir=tmp_path / f"{stage}-{limit}",
                request_sha256="b" * 64, dependency_sha256="c" * 64, literature_sha256="e" * 64,
                source_path=source, source_sha256=digest, stage=stage, parse_attempt_limit=limit)

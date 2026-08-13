"""Contract-pinning tests for ``easyicu.research_agent.execution.phase``.

Background
----------
``execution/phase.py`` (~1,700 LOC) houses the probe → per-step
analysis loop with optional replanning. It is a free-function entry
point (``run_execute_phase``) deliberately split out of
``ResearchAgentPipeline._run_execute_phase`` so a future LangGraph-style
runner can wrap it directly.

Why this file holds *contract* tests and not behaviour tests
------------------------------------------------------------
``run_execute_phase`` is an integration-only entry: it immediately
constructs ``CoderAgent``, ``AnalyzerAgent``, ``RuntimeSupervisor`` and
calls ``pipeline._build_runner(...)``. Exercising it meaningfully
requires the same fixtures the end-to-end ``ResearchAgentPipeline.run``
tests already build. Duplicating those fixtures here would just give us
a slower copy of the same coverage.

What this file *does* protect against is the silent breakage class that
the e2e tests detect 9 minutes late: someone renames the function,
changes its keyword arguments, or breaks the ``(pipeline, plan_result)
→ _ExecutePhaseResult`` shape. We pin those at the import level so the
break shows up in the next ``pytest --collect-only``.
"""

from __future__ import annotations

import ast
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def test_primary_cohort_predicates_extend_only_raw_contract_authority() -> None:
    from easyicu.research_agent.research_context.typed import (
        raw_contract_inputs_for_step,
    )

    receipt = {
        "ordered_predicate_flow": [
            {"predicate_kind": "universe", "resolved_column": None},
            {
                "predicate_kind": "inclusion",
                "resolved_column": "eligibility_max",
            },
        ]
    }

    assert raw_contract_inputs_for_step(
        planner_declared_inputs=["table:parent", "age"],
        primary_cohort_execution_receipt=receipt,
    ) == ("table:parent", "age", "eligibility_max")
    assert raw_contract_inputs_for_step(
        planner_declared_inputs=["age"],
        primary_cohort_execution_receipt=None,
    ) == ("age",)


def test_primary_cohort_predicate_contract_rejects_typed_or_invalid_coordinate() -> (
    None
):
    from easyicu.research_agent.research_context.typed import (
        raw_contract_inputs_for_step,
    )
    from easyicu.research_agent.intake.materialized_metadata import (
        MaterializedMetadataError,
    )

    with pytest.raises(
        MaterializedMetadataError,
        match="invalid resolved column",
    ):
        raw_contract_inputs_for_step(
            planner_declared_inputs=[],
            primary_cohort_execution_receipt={
                "ordered_predicate_flow": [
                    {"resolved_column": "table:not_a_raw_column"}
                ]
            },
        )


def test_primary_cohort_contract_uses_full_authority_without_widening_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.research_context import typed

    base_context = object()
    scoped_context = object()
    calls: list[tuple[object, tuple[str, ...]]] = []

    def fake_resolver(
        context: object,
        names: tuple[str, ...],
    ) -> dict[str, object]:
        calls.append((context, names))
        return {"contracts": {name: {} for name in names if ":" not in name}}

    monkeypatch.setattr(typed, "resolved_raw_input_contracts", fake_resolver)
    receipt = {
        "ordered_predicate_flow": [
            {"resolved_column": None},
            {"resolved_column": "eligibility_max"},
        ]
    }

    result = typed.resolved_raw_input_contracts_for_step(
        coder_base_context=base_context,
        coder_context=scoped_context,
        planner_declared_inputs=["table:parent"],
        primary_cohort_execution_receipt=receipt,
    )

    assert result == {"contracts": {"eligibility_max": {}}}
    assert calls == [(base_context, ("table:parent", "eligibility_max"))]

    typed.resolved_raw_input_contracts_for_step(
        coder_base_context=base_context,
        coder_context=scoped_context,
        planner_declared_inputs=["age"],
        primary_cohort_execution_receipt=None,
    )
    assert calls[-1] == (scoped_context, ("age",))


def test_llm_authority_signature_binds_endpoint_options_and_fallback_order() -> None:
    from easyicu.research_agent.authority.pipeline_cache import llm_signature

    def client(endpoint: str, *, effort: str = "high"):
        return SimpleNamespace(
            name="openai-compatible",
            _model="gpt-5.6-luna",
            _resolved_base_url=endpoint,
            _extra_body={"reasoning_effort": effort},
        )

    endpoint_a = client("http://127.0.0.1:8787/v1")
    endpoint_b = client("http://127.0.0.1:8317/v1")
    assert llm_signature(endpoint_a) != llm_signature(endpoint_b)
    assert llm_signature(endpoint_b) != llm_signature(
        client("http://127.0.0.1:8317/v1", effort="low")
    )

    fallback_ab = SimpleNamespace(name="fallback", _clients=[endpoint_a, endpoint_b])
    fallback_ba = SimpleNamespace(name="fallback", _clients=[endpoint_b, endpoint_a])
    assert llm_signature(fallback_ab) != llm_signature(fallback_ba)


def test_pipeline_cache_identifies_contextual_mocks_without_importing_them() -> None:
    from easyicu.research_agent.authority.pipeline_cache import (
        iter_mock_clients,
        llm_signature,
    )
    from easyicu.research_agent.providers.llm import LLMRouter
    from easyicu.research_agent.providers.mocks import (
        MockLLMClient,
        PatternScriptedMockLLMClient,
    )

    default = MockLLMClient()
    scripted = PatternScriptedMockLLMClient([])
    router = LLMRouter(default=default, planner=scripted)

    assert llm_signature(default) == "mock"
    assert list(iter_mock_clients(router)) == [default, scripted]


def test_capsule_checkpoint_upsert_never_overwrites_prior_terminal_attempt() -> None:
    from easyicu.research_agent.execution.phase import (
        _append_terminal_step_record,
        _upsert_current_capsule_checkpoint,
    )
    from easyicu.research_agent.authority.runtime_artifacts import current_step_records

    records = [
        {
            "step_id": "01_summary",
            "attempt_id": "attempt-1",
            "status": "candidate_checkpointed",
        },
        {
            "step_id": "01_summary",
            "attempt_id": "attempt-1",
            "status": "ok",
        },
    ]
    pending = {
        "step_id": "01_summary",
        "attempt_id": "attempt-2",
        "status": "capsule_revalidation_pending",
    }
    _upsert_current_capsule_checkpoint(records, pending)

    assert records[-1] == pending
    assert current_step_records(records)[-1]["status"] == (
        "capsule_revalidation_pending"
    )

    terminal = {
        "step_id": "01_summary",
        "attempt_id": "attempt-2",
        "status": "contract_failed",
    }
    _append_terminal_step_record(records, terminal)

    assert pending not in records
    assert records[-1] == terminal
    assert current_step_records(records)[-1] == terminal


@pytest.mark.parametrize("version", [5, 6])
def test_provider_receipt_requirement_covers_legacy_and_initial_pending(
    version: int,
) -> None:
    from easyicu.research_agent.execution.phase import (
        _step_snapshot_requires_provider_receipt,
    )

    assert _step_snapshot_requires_provider_receipt(
        {
            "step_provider_call_receipt_version": version,
            "capsule_pending_initial_transport_id": "initial_generation:1",
        },
        provider_attempts=0,
        logical_repair_attempts=0,
    )
    assert _step_snapshot_requires_provider_receipt(
        {
            "step_provider_call_receipt_version": version,
            "step_provider_call_receipt": (
                ".runtime/provider_call_budgets/example.json"
            ),
        },
        provider_attempts=0,
        logical_repair_attempts=0,
    )


def test_non_typed_alias_requires_current_successful_step_authority(tmp_path) -> None:
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.execution.phase import (
        _current_verified_evidence_record,
    )

    source = tmp_path / "source.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    store = EvidenceStore(tmp_path)
    record = store.register_file(
        kind="table",
        description="pending output",
        source_path=source,
        produced_by_step="01_summary",
        aliases=["summary_table"],
        publish_aliases=False,
    )
    store.publish_step_success_aliases(
        {record.evidence_id: ["summary_table"]},
        step_id="01_summary",
    )
    pending = [
        {
            "step_id": "01_summary",
            "attempt_id": "attempt-1",
            "status": "executed_pending_review",
            "evidence_ids": [record.evidence_id],
        }
    ]
    assert _current_verified_evidence_record(store, "summary_table", pending) is None

    successful = [{**pending[0], "status": "ok"}]
    assert (
        _current_verified_evidence_record(store, "summary_table", successful) == record
    )


def test_step_run_input_capsule_must_match_sealed_evidence(tmp_path) -> None:
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.execution.phase import (
        _verified_run_input_capsule_digest,
    )
    from easyicu.research_agent.authority.run_input import RunInputIdentityError

    capsule = tmp_path / "run_input_capsule.json"
    capsule.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    store = EvidenceStore(tmp_path)
    record = store.register_file(
        kind="log",
        description="sealed run input capsule",
        source_path=capsule,
        evidence_id="run_input_capsule",
        producer="pipeline",
        generation_mode="system",
    )

    assert (
        _verified_run_input_capsule_digest(
            run_dir=tmp_path,
            evidence_store=store,
        )
        == record.sha256
    )

    capsule.write_text('{"schema_version":"tampered"}\n', encoding="utf-8")
    with pytest.raises(RunInputIdentityError, match="digest changed"):
        _verified_run_input_capsule_digest(
            run_dir=tmp_path,
            evidence_store=store,
        )


def test_run_input_capsule_read_error_is_typed(tmp_path, monkeypatch) -> None:
    # An OSError while reading the working/sealed copy must convert to the typed
    # RunInputIdentityError boundary, not escape as a raw OSError.
    from pathlib import Path as _Path

    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.execution.phase import (
        _verified_run_input_capsule_digest,
    )
    from easyicu.research_agent.authority.run_input import RunInputIdentityError

    capsule = tmp_path / "run_input_capsule.json"
    capsule.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    store = EvidenceStore(tmp_path)
    store.register_file(
        kind="log",
        description="sealed run input capsule",
        source_path=capsule,
        evidence_id="run_input_capsule",
        producer="pipeline",
        generation_mode="system",
    )

    def _boom(self, *args, **kwargs):
        raise OSError("disk vanished mid-read")

    # verified_run_evidence_path / sha256 use Path.open, so only the post-
    # verification read_bytes reads are intercepted here.
    monkeypatch.setattr(_Path, "read_bytes", _boom)
    with pytest.raises(RunInputIdentityError, match="could not be read"):
        _verified_run_input_capsule_digest(
            run_dir=tmp_path,
            evidence_store=store,
        )


def test_parallel_step_worker_inherits_runner_capability_context() -> None:
    import easyicu.research_agent.execution.method_capabilities as method_capabilities
    from easyicu.research_agent.execution.phase import _submit_in_current_context

    method_capabilities.set_runtime_capability_snapshot_provider(
        lambda: {"docker-only-capability"}
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                _submit_in_current_context(
                    executor,
                    method_capabilities.runtime_capability_snapshot,
                )
                for _ in range(2)
            ]
            assert [future.result() for future in futures] == [
                frozenset({"docker-only-capability"}),
                frozenset({"docker-only-capability"}),
            ]
    finally:
        method_capabilities.set_runtime_capability_snapshot_provider(None)


def test_consistent_local_figure_source_descriptor_is_canonicalized_for_consumers(
    tmp_path,
):
    from easyicu.research_agent.discovery.discovery_package import _string_list
    from easyicu.research_agent.figures.skill import (
        _contract_payload_source_references,
    )
    from easyicu.research_agent.execution.phase import (
        _figure_contract_source_data_canonicalization_candidate,
        _install_figure_contract_source_data_canonicalization,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "result_source_data.csv").write_text("x,y\na,1\n", encoding="utf-8")
    contract_path = out_dir / "result.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "result",
                "source_data": [
                    {
                        "file": "result_source_data.csv",
                        "filename": "result_source_data.csv",
                        "path": "result_source_data.csv",
                        "relative_path": "result_source_data.csv",
                        "kind": "table",
                        "evidence_ids": [],
                    }
                ],
                "panels": [],
            }
        ),
        encoding="utf-8",
    )

    candidate = _figure_contract_source_data_canonicalization_candidate(
        contract_path=contract_path,
        out_dir=out_dir,
    )
    assert candidate is not None
    _before, canonical_text, names = candidate
    assert names == ["result_source_data.csv"]
    _install_figure_contract_source_data_canonicalization(
        contract_path=contract_path,
        expected_before=_before,
        canonical_text=canonical_text,
    )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    assert payload["source_data"] == ["result_source_data.csv"]
    assert _contract_payload_source_references(payload) == ["result_source_data.csv"]
    assert _string_list(payload["source_data"]) == ["result_source_data.csv"]


def test_figure_contract_canonicalization_does_not_follow_predictable_temp_symlink(
    tmp_path,
):
    from easyicu.research_agent.execution.phase import (
        _figure_contract_source_data_canonicalization_candidate,
        _install_figure_contract_source_data_canonicalization,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "source.csv").write_text("x\n1\n", encoding="utf-8")
    contract_path = out_dir / "result.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "result",
                "source_data": [{"file": "source.csv", "path": "source.csv"}],
            }
        ),
        encoding="utf-8",
    )
    outside = tmp_path / "outside.json"
    outside.write_text("do-not-touch", encoding="utf-8")
    predictable = out_dir / ".result.figure_contract.json.schema.tmp"
    try:
        predictable.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    candidate = _figure_contract_source_data_canonicalization_candidate(
        contract_path=contract_path,
        out_dir=out_dir,
    )
    assert candidate is not None
    before, after, _names = candidate
    _install_figure_contract_source_data_canonicalization(
        contract_path=contract_path,
        expected_before=before,
        canonical_text=after,
    )

    assert outside.read_text(encoding="utf-8") == "do-not-touch"
    assert json.loads(contract_path.read_text(encoding="utf-8"))["source_data"] == [
        "source.csv"
    ]


def test_figure_contract_canonicalization_rejects_changed_reviewed_contract(
    tmp_path,
):
    from easyicu.research_agent.execution.phase import (
        _figure_contract_source_data_canonicalization_candidate,
        _install_figure_contract_source_data_canonicalization,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "source.csv").write_text("x\n1\n", encoding="utf-8")
    contract_path = out_dir / "result.figure_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "figure_id": "result",
                "source_data": [{"file": "source.csv", "path": "source.csv"}],
            }
        ),
        encoding="utf-8",
    )
    candidate = _figure_contract_source_data_canonicalization_candidate(
        contract_path=contract_path,
        out_dir=out_dir,
    )
    assert candidate is not None
    before, after, _names = candidate
    contract_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after canonicalization review"):
        _install_figure_contract_source_data_canonicalization(
            contract_path=contract_path,
            expected_before=before,
            canonical_text=after,
        )


@pytest.mark.parametrize(
    "source_data",
    [
        [{"file": "source.csv", "path": "other.csv"}],
        [{"file": 7}],
        [{"file": "/tmp/source.csv"}],
        [{"file": "nested/source.csv"}],
        [{"evidence_id": "table_source"}],
        [["source.csv"]],
    ],
)
def test_figure_source_descriptor_canonicalization_fails_closed(
    tmp_path,
    source_data,
):
    from easyicu.research_agent.execution.phase import (
        _figure_contract_source_data_canonicalization_candidate,
    )

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "source.csv").write_text("x\n1\n", encoding="utf-8")
    (out_dir / "other.csv").write_text("x\n2\n", encoding="utf-8")
    contract_path = out_dir / "result.figure_contract.json"
    contract_path.write_text(
        json.dumps({"figure_id": "result", "source_data": source_data}),
        encoding="utf-8",
    )

    assert (
        _figure_contract_source_data_canonicalization_candidate(
            contract_path=contract_path,
            out_dir=out_dir,
        )
        is None
    )


def test_module_is_importable():
    import easyicu.research_agent.execution.phase as pe  # noqa: F401


def test_run_execute_phase_is_exported():
    from easyicu.research_agent.execution.phase import run_execute_phase

    assert callable(run_execute_phase)


def test_critic_messages_keep_only_blocking_errors():
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.phase import (
        _actionable_validator_messages,
    )

    messages = _actionable_validator_messages(
        [
            ValidationFinding(
                validator="audit",
                severity="info",
                message="Informational provenance note.",
            ),
            ValidationFinding(
                validator="audit",
                severity="warning",
                message="Review this warning.",
            ),
            ValidationFinding(
                validator="audit",
                severity="error",
                message="Repair this error.",
            ),
        ]
    )

    assert messages == ["Repair this error."]


def test_code_repair_findings_keep_only_blocking_errors():
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.gates.semantics import blocking_validator_findings
    from easyicu.research_agent.execution.phase import (
        _blocking_validator_findings,
    )

    assert _blocking_validator_findings is blocking_validator_findings

    findings = _blocking_validator_findings(
        [
            ValidationFinding(
                validator="audit",
                severity="warning",
                message="Keep as advisory evidence only.",
            ),
            ValidationFinding(
                validator="audit",
                severity="error",
                message="Repair this blocking error.",
                detail={"reason": "blocking_contract"},
            ),
        ]
    )

    assert [finding.message for finding in findings] == ["Repair this blocking error."]
    assert findings[0].detail == {"reason": "blocking_contract"}


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"status": "ok", "step_summary": {}}, False),
        ({"status": "ok", "replan_requested": True}, True),
        (
            {
                "status": "ok",
                "step_summary": {"plan_revision_requested": True},
            },
            True,
        ),
        ({"status": "ok", "step_summary": {"replan_requested": "true"}}, False),
        ({"status": "contract_failed", "replan_requested": True}, False),
    ],
)
def test_success_replanning_requires_an_exact_agent_request(record, expected):
    from easyicu.research_agent.execution.phase import (
        _successful_step_requests_replan,
    )

    assert _successful_step_requests_replan(record) is expected


def test_required_model_contract_error_fail_closes_outer_step_and_run():
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.phase import (
        _step_status_from_contract_findings,
    )
    from easyicu.research_agent.reporting.readiness import execution_gate_status
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    contract_findings = [
        ValidationFinding(
            validator="primary_model_contract",
            severity="error",
            message="A planner-required secondary model was not fitted.",
            detail={"issue": "required_model_not_fitted"},
        )
    ]
    status = _step_status_from_contract_findings(
        contract_findings=contract_findings,
        figure_source_findings=[],
        stat_findings=[],
    )
    plan = AnalysisPlan(
        research_question="Test a planner-owned model obligation.",
        steps=[
            AnalysisStep(
                step_id="01_models",
                intent="Fit the planned models.",
            )
        ],
    )

    assert status == "contract_failed"
    gate = execution_gate_status(
        plan=plan,
        per_step_records=[{"step_id": "01_models", "status": status}],
    )
    assert gate["execution_complete"] is False
    assert gate["failed_steps"] == [
        {"step_id": "01_models", "status": "contract_failed"}
    ]


def test_every_deterministic_statistical_error_fails_outer_step():
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.phase import (
        _step_status_from_contract_findings,
    )

    status = _step_status_from_contract_findings(
        contract_findings=[],
        figure_source_findings=[],
        stat_findings=[
            ValidationFinding(
                validator="statistical_sanity",
                severity="error",
                message="A deterministic statistical contract failed.",
                detail={"issue": "impossible_denominator"},
            )
        ],
    )

    assert status == "contract_failed"


def test_first_step_checkpoint_selector_preserves_agent_owned_step_id():
    from easyicu.research_agent.execution.phase import (
        _resolve_stop_after_step_selector,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Summarize this ICU cohort.",
        steps=[
            AnalysisStep(
                step_id="agent_generated_cohort_name",
                intent="Define the planned cohort.",
                inputs=["stay_id"],
                expected_outputs=["table:analysis_cohort"],
                method="cohort_definition",
            ),
            AnalysisStep(
                step_id="agent_generated_summary_name",
                intent="Summarize the planned cohort.",
                inputs=["table:analysis_cohort"],
                expected_outputs=["table:summary"],
                method="descriptive_summary",
            ),
        ],
    )

    assert _resolve_stop_after_step_selector(plan, "@first") == (
        "agent_generated_cohort_name"
    )
    assert _resolve_stop_after_step_selector(plan, "@index:2") == (
        "agent_generated_summary_name"
    )
    assert _resolve_stop_after_step_selector(plan, "@product:table:summary") == (
        "agent_generated_summary_name"
    )
    assert _resolve_stop_after_step_selector(plan, "agent_generated_summary_name") == (
        "agent_generated_summary_name"
    )
    with pytest.raises(ValueError, match="exceeds the active plan"):
        _resolve_stop_after_step_selector(plan, "@index:3")

    plan.steps[0].expected_outputs.append("table:summary")
    with pytest.raises(ValueError, match="exactly one declared producer"):
        _resolve_stop_after_step_selector(plan, "@product:table:summary")


def test_failed_contract_code_reuse_requires_exact_checkpoint_authority():
    import copy
    import hashlib

    from easyicu.research_agent.execution.phase import (
        _failed_contract_code_can_be_reused_before_coder,
        _serializable_plan_scientific_scope_signature,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    step = AnalysisStep(
        step_id="01_summary",
        intent="Summarize the declared cohort.",
        inputs=["stay_id"],
        expected_outputs=["table:summary"],
        method="descriptive_summary",
    )
    plan = AnalysisPlan(
        research_question="Summarize this ICU cohort.",
        steps=[step],
    )
    code = "import pandas as pd\nprint(pd.__version__)\n"
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    evidence_record = {
        "evidence_id": "code_summary",
        "sha256": digest,
    }
    resolved_inputs_sha256 = "a" * 64
    run_input_capsule_sha256 = "b" * 64
    prior_record = {
        "step_id": step.step_id,
        "status": "contract_failed",
        "returncode": 0,
        "timed_out": False,
        "outputs_safe_to_collect": True,
        "executed_code_sha256": digest,
        "concept_approved_code_sha256": digest,
        "script_evidence_id": evidence_record["evidence_id"],
        "resolved_inputs_sha256": resolved_inputs_sha256,
        "run_input_capsule_sha256": run_input_capsule_sha256,
        "plan_scientific_signature": (
            _serializable_plan_scientific_scope_signature(plan)
        ),
        "analysis_request": {"step": step.model_dump(mode="json")},
    }

    def allowed(record, resumed=(code, evidence_record)):
        return _failed_contract_code_can_be_reused_before_coder(
            prior_step_record=record,
            resumed_code=resumed,
            step=step,
            plan=plan,
            resolved_inputs_sha256=resolved_inputs_sha256,
            run_input_capsule_sha256=run_input_capsule_sha256,
        )

    assert allowed(prior_record) is True

    mutations = []
    for key, value in (
        ("status", "ok"),
        ("returncode", 1),
        ("timed_out", True),
        ("outputs_safe_to_collect", False),
        ("executed_code_sha256", "0" * 64),
        ("concept_approved_code_sha256", "0" * 64),
        ("script_evidence_id", "different_code"),
        ("resolved_inputs_sha256", "0" * 64),
        ("run_input_capsule_sha256", "0" * 64),
        ("plan_scientific_signature", ["changed"]),
        ("provider_call_budget_receipt_invalid", True),
        ("quarantined_requires_repair", True),
        ("resumed_failed_contract_code_preflight", True),
    ):
        changed = copy.deepcopy(prior_record)
        changed[key] = value
        mutations.append(changed)
    changed_step = copy.deepcopy(prior_record)
    changed_step["analysis_request"]["step"]["method"] = "different_method"
    mutations.append(changed_step)

    assert all(allowed(record) is False for record in mutations)
    mismatched_evidence = dict(evidence_record, sha256="f" * 64)
    assert allowed(prior_record, (code, mismatched_evidence)) is False


@pytest.mark.parametrize("critique_status", ["needs_revision", "blocked"])
def test_negative_critic_review_fail_closes_outer_step(critique_status):
    from easyicu.research_agent.execution.phase import (
        _step_status_from_contract_findings,
    )

    assert (
        _step_status_from_contract_findings(
            contract_findings=[],
            figure_source_findings=[],
            stat_findings=[],
            critique_status=critique_status,
        )
        == "critic_failed"
    )


def test_locked_measurement_data_quality_classifier_is_structural():
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.execution.phase import (
        _locked_measurement_data_quality_issues,
    )

    findings = [
        ValidationFinding(
            validator="step_summary_integrity",
            severity="error",
            message="Locked data contain invalid pairs.",
            detail={"issue": "measurement_provenance_invalid_pairs"},
        ),
        ValidationFinding(
            validator="step_summary_integrity",
            severity="error",
            message="Locked data contain discordance.",
            detail={"issue": "measurement_provenance_count_flag_discordance"},
        ),
        ValidationFinding(
            validator="step_summary_integrity",
            severity="error",
            message="Generated code reported the wrong count.",
            detail={"issue": "measurement_provenance_host_count_mismatch"},
        ),
        ValidationFinding(
            validator="another_validator",
            severity="error",
            message="Same words, wrong authority.",
            detail={"issue": "measurement_provenance_invalid_pairs"},
        ),
        ValidationFinding(
            validator="step_summary_integrity",
            severity="error",
            message="The planned flag is absent from the locked cohort.",
            detail={"issue": "measurement_provenance_measured_column_missing"},
        ),
        ValidationFinding(
            validator="step_summary_integrity",
            severity="error",
            message="The companion column is ambiguous.",
            detail={"issue": "measurement_provenance_count_column_ambiguous"},
        ),
    ]

    assert _locked_measurement_data_quality_issues(findings) == [
        "measurement_provenance_count_column_ambiguous",
        "measurement_provenance_count_flag_discordance",
        "measurement_provenance_invalid_pairs",
        "measurement_provenance_measured_column_missing",
    ]


def test_locked_measurement_data_quality_terminates_before_code_repair():
    from easyicu.research_agent.execution import phase as pipeline_execute

    source = inspect.getsource(pipeline_execute._candidate_contract_repair_transition)
    route_start = source.index(
        "locked_data_quality_issues = host._locked_measurement_data_quality_issues("
    )
    route_end = source.index(
        "if attempt.sealed_renderer_authorized_code_sha256", route_start
    )
    terminal_route = source[route_start:route_end]

    assert "measurement_provenance_repair_suppressed" in terminal_route
    assert '"diagnostic_only": True' in terminal_route
    assert '"locked_cohort_data_quality_failed"' in terminal_route
    assert "return _CandidateLoopAction.RETURN" in terminal_route
    assert "_deterministic_summary_repair" not in terminal_route
    assert "deterministic_contract_repair" not in terminal_route
    assert "coder.repair" not in terminal_route


def test_host_standard_renderer_does_not_require_legacy_repair_receipt():
    """Code sealing and registry repair receipts are separate contracts.

    A selected deterministic figure executor has no repair id by design.  It
    still gets the no-rewrite code-digest policy, while its parents are bound
    by the ordinary resolved-input receipts.  Requiring the legacy receipt in
    that state made every such renderer fail after successfully drawing.
    """

    from easyicu.research_agent.execution.publication_figure import (
        validate_and_record_sealed_renderer_receipt,
    )

    source = inspect.getsource(validate_and_record_sealed_renderer_receipt)
    receipt_guard = source.index(
        "if authorized_code_sha256 is None or state.repair_id is None:"
    )
    parent_check = source.index("read_digest_bound_artifact_snapshot(", receipt_guard)
    identity_check = source.index(
        'visual_step_summary.get("sealed_renderer_repair")', parent_check
    )

    assert "return ()" in source[receipt_guard:parent_check]
    assert "sealed_renderer_implementation_sha256" in source[identity_check:]
    assert receipt_guard < parent_check < identity_check


def test_locked_measurement_preflight_runs_before_every_coder_repair():
    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.execution.step_candidate_recovery import (
        StepCandidateRecovery,
    )

    source = inspect.getsource(pipeline_execute.run_execute_phase) + "\n".join(
        inspect.getsource(stage)
        for stage in (
            pipeline_execute._candidate_concept_audit_transition,
            pipeline_execute._candidate_execute_transition,
            pipeline_execute._candidate_visual_transition,
            pipeline_execute._candidate_contract_repair_transition,
            pipeline_execute._candidate_failure_transition,
        )
    )
    preflight = source.index("audit_locked_measurement_data_quality(")
    first_repair_transport = source.index("_repair_with_capsule(", preflight)
    owner_source = inspect.getsource(StepCandidateRecovery.repair_with_capsule)

    assert preflight < first_repair_transport
    assert "request.coder.repair(" in owner_source


def test_sealed_figure_preflight_supersedes_stale_resume_capsule_candidate():
    """A selected failed figure capsule cannot inherit a new renderer id."""

    from easyicu.research_agent.execution import phase as pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)
    selection_start = source.index("standard_executor = select_standard_executor(")
    standard_branch = source.index(
        "if preflight_standard_code is not None:", selection_start
    )
    figure_branch = source.index(
        "elif preflight_figure_code is not None:", selection_start
    )
    resumed_branch = source.index(
        "elif step_attempt_state.selected_resume_capsule is not None:",
        selection_start,
    )

    assert standard_branch < figure_branch < resumed_branch
    figure_body = source[figure_branch:resumed_branch]
    assert "code = preflight_figure_code" in figure_body
    assert "code = step_attempt_state.selected_resume_capsule.candidate_code" not in (
        figure_body
    )


def test_authorized_resume_repair_supersedes_failed_candidate_capsule():
    from easyicu.research_agent.execution import phase as pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)
    selection_start = source.index("standard_executor = select_standard_executor(")
    repair_branch = source.index(
        "elif resume_deterministic_repair_code is not None:", selection_start
    )
    capsule_branch = source.index(
        "elif step_attempt_state.selected_resume_capsule is not None:",
        selection_start,
    )

    assert repair_branch < capsule_branch


def test_stability_standard_executor_supersedes_stale_resume_capsule():
    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.execution.runners import selection

    source = inspect.getsource(pipeline_execute.run_execute_phase)
    assignment = source[source.index("standard_executor = select_standard_executor(") :]
    assignment = assignment[: assignment.index("preflight_figure_code =")]
    selector_source = inspect.getsource(selection.select_standard_executor)

    assert "trajectory_stability_executor_owns_step(" in selector_source
    assert "trajectory_stability_executor_code(" in selector_source
    assert "preflight_standard_code = standard_executor.code" in assignment
    assert "plausibility_scope=plausibility_authority.scope" in assignment
    assert "selected_resume_capsule" not in assignment


def test_standard_executor_failure_is_attributed_to_its_actual_owner():
    from easyicu.research_agent.execution.standard_executor_diagnostics import (
        standard_executor_failure_finding,
    )

    finding = standard_executor_failure_finding(
        step_record={
            "deterministic_standard_analysis": "grouped_table_one",
            "deterministic_standard_selection_reason": "table_one_spec_preflight",
        },
        step_id="02_table_one",
        reason="preexecution_concept_gate_failed",
        failure_phase="preexecution_concept_gate",
    )

    assert finding.validator == "deterministic_standard_executor"
    assert "grouped_table_one" in finding.message
    assert "trajectory" not in finding.message.casefold()
    assert finding.detail == {
        "step_id": "02_table_one",
        "issue_code": "deterministic_standard_executor_failed_closed",
        "failure_phase": "preexecution_concept_gate",
        "analysis_kind": "grouped_table_one",
        "selection_reason": "table_one_spec_preflight",
        "reason": "preexecution_concept_gate_failed",
        "executor_errors": None,
    }


@pytest.mark.parametrize(
    ("step_id", "intent"),
    [
        (
            "04_publication_figure_interpretation",
            "Interpret the downstream publication figure for the manuscript.",
        ),
        (
            "04_primary_model",
            "Estimate the association used in a publication-ready figure.",
        ),
    ],
)
def test_publication_figure_gate_ignores_name_only_mentions(step_id, intent):
    from easyicu.research_agent.execution.phase import (
        _step_requires_publication_figure_exports,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    step = AnalysisStep(
        step_id=step_id,
        intent=intent,
        method="mixed_effects_regression",
        expected_outputs=["table:association_estimates"],
    )

    assert _step_requires_publication_figure_exports(step) is False


@pytest.mark.parametrize(
    ("method", "expected_outputs"),
    [
        ("publication_figure_generation", ["log:rendering_process"]),
        ("visualization", ["log:rendering_process"]),
        ("mixed_effects_regression", ["figure:association_forest_plot"]),
    ],
)
def test_publication_figure_gate_accepts_structural_figure_contracts(
    method, expected_outputs
):
    from easyicu.research_agent.execution.phase import (
        _step_requires_publication_figure_exports,
    )
    from easyicu.research_agent.schema import AnalysisStep

    step = AnalysisStep(
        step_id="04_results_publication_figure",
        intent="Render the requested publication figure.",
        method=method,
        expected_outputs=expected_outputs,
    )

    assert _step_requires_publication_figure_exports(step) is True


def test_execute_phase_mandatory_publication_gate_uses_structural_predicate():
    from easyicu.research_agent.execution.phase import run_execute_phase

    source = inspect.getsource(run_execute_phase)
    gate_start = source.index("publication_step =")
    gate_end = source.index("figure_role =", gate_start)
    gate_source = source[gate_start:gate_end]

    assert "_step_requires_publication_figure_exports" in gate_source
    assert "step.step_id" not in gate_source
    assert "step.intent" not in gate_source


def test_run_execute_phase_signature_is_stable():
    """Lock the keyword-argument contract pipeline.py relies on.

    If a parameter is renamed or removed here, callers in pipeline.py
    will fail at import time elsewhere. Catching it as a one-line
    signature diff is far cheaper than the e2e failure.
    """
    from easyicu.research_agent.execution.phase import run_execute_phase

    sig = inspect.signature(run_execute_phase)
    params = sig.parameters

    # First positional is the pipeline collaborator; the rest are keyword-only.
    positional = [
        name
        for name, p in params.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == ["pipeline"], (
        "run_execute_phase must take exactly one positional collaborator "
        f"(the pipeline); got {positional}"
    )

    required_keywords = {
        "plan_result",
        "cohort_path",
        "run_dir",
        "run_id",
        "skill_obj",
        "notes",
        "emit_progress",
    }
    actual_keywords = {
        name for name, p in params.items() if p.kind == inspect.Parameter.KEYWORD_ONLY
    }
    missing = required_keywords - actual_keywords
    assert not missing, (
        f"run_execute_phase is missing keyword-only params {missing}; "
        "downstream pipeline.py keyword call will break."
    )


def test_run_execute_phase_does_not_mutate_pipeline_state():
    """Lock the read-only-collaborator invariant.

    Module docstring states: 'pipeline instance is passed in only as a
    *read-only collaborator* … audit on 2026-05-15 confirmed zero
    ``self.* = ...`` writes inside the original method body.' If a
    refactor reintroduces a write, future graph-runner authors will
    have a confusing aliasing bug. We re-run the audit in CI.
    """
    import ast
    from easyicu.research_agent.execution import phase as pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)
    tree = ast.parse(source)

    pipeline_writes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "pipeline"
                ):
                    pipeline_writes.append(target.attr)
        elif isinstance(node, ast.AugAssign):
            if (
                isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "pipeline"
            ):
                pipeline_writes.append(node.target.attr)

    assert pipeline_writes == [], (
        "run_execute_phase must not mutate the pipeline collaborator; "
        f"found writes to: {pipeline_writes}. See module docstring."
    )


def test_execute_phase_preserves_repair_provenance_across_concept_and_runtime():
    """Every LLM mutation must outrank pure resume/runner provenance labels."""
    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.execution.concept_repair import (
        run_concept_repair_loop,
    )

    source = inspect.getsource(pipeline_execute.run_execute_phase) + "\n".join(
        inspect.getsource(stage)
        for stage in (
            pipeline_execute._candidate_concept_audit_transition,
            pipeline_execute._candidate_execute_transition,
            pipeline_execute._candidate_visual_transition,
            pipeline_execute._candidate_contract_repair_transition,
            pipeline_execute._candidate_failure_transition,
        )
    )
    concept_source = inspect.getsource(run_concept_repair_loop)

    # Initial concept, post-mutation concept, visual, contract, and runtime
    # repairs each mark the same lineage flag after a successful coder call.
    llm_repair_marks = source.count("worker_progress.llm_repair_used = True")
    llm_repair_marks += concept_source.count("worker.llm_repair_used = True")
    assert llm_repair_marks == 5
    assert source.count("worker_progress.generation_mode(") == 4
    assert source.count("llm_repair_used=False") == 1
    assert "_non_llm_interpretation_for_generation(" in source


@pytest.mark.parametrize(
    ("generation_mode", "expected_evidence_mode"),
    [
        ("resumed_code_reuse", "resumed_code_reuse"),
        ("fallback", "deterministic_fallback"),
        ("deterministic_standard", "deterministic_standard"),
    ],
)
def test_unchanged_or_host_owned_code_skips_llm_interpretation(
    generation_mode, expected_evidence_mode
):
    from easyicu.research_agent.execution.phase import (
        _non_llm_interpretation_for_generation,
    )

    result = _non_llm_interpretation_for_generation(
        step_id="02_table_one",
        generation_mode=generation_mode,
    )

    assert result is not None
    interpretation, evidence_mode = result
    assert "no new LLM interpretation was requested" in interpretation
    assert evidence_mode == expected_evidence_mode


def test_agent_generated_code_keeps_llm_interpretation_path():
    from easyicu.research_agent.execution.phase import (
        _non_llm_interpretation_for_generation,
    )

    assert (
        _non_llm_interpretation_for_generation(
            step_id="03_primary_model",
            generation_mode="llm_generated",
        )
        is None
    )


def test_execute_phase_routes_figure_contracts_through_early_repair_loop():
    from easyicu.research_agent.execution import phase as pipeline_execute

    source = inspect.getsource(pipeline_execute._candidate_contract_setup_transition)
    early_gate = source.index("early_contract_errors = [")
    before_early_gate = source[:early_gate]

    # The figure-contract / figure-source audits are lifted into
    # _post_canonicalization_figure_findings, which the early repair loop calls
    # before the early-contract-error gate, so figure errors still route through
    # the in-run repair loop.
    assert "_post_canonicalization_figure_findings(" in before_early_gate
    helper_source = inspect.getsource(
        pipeline_execute._post_canonicalization_figure_findings
    )
    assert "figure_contract_validator.audit(" in helper_source
    assert "figure_source_validator.audit(" in helper_source


def test_figure_repair_precedes_output_evidence_and_numeric_claim_seal():
    # Batch 1c: the numeric-claim + alias seal is delegated to
    # StepEvidenceCommit.commit_validated_step (the "both in one generation"
    # guarantee is locked by test_step_evidence_commit.py's AST contract). Here we
    # keep the caller-side ordering: figure repair -> output-artifact registration
    # (aliases deferred) -> status resolution -> the commit boundary.
    from easyicu.research_agent.execution import phase as pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)
    seal = source.index("sealed_result_digests =")
    artifact_registration = source.index("for art in run_result.artefacts:", seal)
    numeric_registration = source.index(
        "evidence.register_step_summary_numerics(", artifact_registration
    )
    status_resolution = source.index(
        'step_record["status"] = _step_status_from_contract_findings('
    )
    commit = source.rindex("step_evidence_commit.commit_validated_step(")
    final_repair = source.rindex("_repair_publication_figure_in_staging(")

    assert final_repair < seal < artifact_registration < numeric_registration
    assert numeric_registration < status_resolution < commit
    assert "publish_aliases=False" in source[artifact_registration:status_resolution]
    assert (
        "_repair_publication_figure_in_staging(" not in source[artifact_registration:]
    )


def test_execute_phase_deterministically_requires_typed_exposure_consumption():
    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.execution import (
        concept_audit as concept_audit_execution,
    )

    shared_source = inspect.getsource(
        pipeline_execute._deterministic_code_gate_findings
    )
    execute_source = inspect.getsource(
        pipeline_execute.run_execute_phase
    ) + inspect.getsource(pipeline_execute._candidate_concept_audit_transition)
    concept_execution_source = inspect.getsource(
        concept_audit_execution.ConceptAuditCoordinator.findings_for_code
    )
    replay_source = inspect.getsource(
        pipeline_execute._selectively_revalidate_resume_successes
    )

    assert "requires_primary_exposure_artifact" in shared_source
    assert "_verified_authoritative_exposure_flow(" in shared_source
    assert 'validator="typed_input_authority_flow"' in shared_source
    assert '"typed_primary_exposure_not_consumed"' in shared_source
    assert "ConceptAuditCoordinator(" in execute_source
    assert "concept_audit.findings_for_code(" in execute_source
    assert "deterministic_code_gate_findings(" in concept_execution_source
    assert "_deterministic_code_gate_findings(" in replay_source


def test_concept_audit_execution_is_cycle_free_and_old_state_path_is_compatible():
    import ast

    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.execution import (
        concept_audit as concept_audit_execution,
    )

    tree = ast.parse(inspect.getsource(concept_audit_execution))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        str(node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert not any(name.endswith("pipeline_execute") for name in imported_modules)
    assert (
        pipeline_execute.ConceptQuarantineState
        is concept_audit_execution.ConceptQuarantineState
    )


def test_concept_gate_is_read_only_and_keeps_old_function_identity():
    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.gates import concept as concept_gate

    source = inspect.getsource(concept_gate)
    compatibility_exports = {
        "_deterministic_code_gate_findings": (
            concept_gate.deterministic_code_gate_findings
        ),
        "_deterministic_gate_stamp": concept_gate.deterministic_gate_stamp,
        "_finding_detail_without_source_positions": (
            concept_gate.finding_detail_without_source_positions
        ),
        "_finding_occurrence_identity": concept_gate.finding_occurrence_identity,
        "_quarantined_deterministic_errors_resolved_by_current_gate": (
            concept_gate.quarantined_deterministic_errors_resolved_by_current_gate
        ),
        "_quarantined_errors_superseded_by_current_policy": (
            concept_gate.quarantined_errors_superseded_by_current_policy
        ),
    }

    assert "LLMConceptAuditor" not in source
    assert "StepProviderCallBudget" not in source
    assert "store_quarantined_concept_draft" not in source
    for old_name, canonical_function in compatibility_exports.items():
        assert getattr(pipeline_execute, old_name) is canonical_function


def test_fresh_execution_uses_the_authoritative_final_gate_evaluator_once():
    import ast

    from easyicu.research_agent.execution import phase as pipeline_execute

    source = inspect.getsource(pipeline_execute.run_execute_phase)
    tree = ast.parse(source)
    evaluator_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_evaluate_final_deterministic_gates"
    ]

    assert len(evaluator_calls) == 1
    assert "stat_validator.audit(" not in source
    assert "clinical_validator.audit(" not in source
    assert "statistical_guard.audit(" not in source
    for group in (
        "stat_findings",
        "clinical_findings",
        "guard_findings",
        "contract_findings",
        "figure_source_findings",
    ):
        assert f"final_gate_findings.{group}" in source


def test_final_gate_evaluator_preserves_group_order_and_attempt_binding(
    monkeypatch,
    tmp_path,
):
    from easyicu.research_agent.authority.plausibility import (
        FlagOnlyPlausibilityScope,
    )
    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.execution import final_validation
    from easyicu.research_agent.gates import contract as contract_gate
    from easyicu.research_agent.contracts.runtime import ValidationFinding
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    calls = []

    def finding(name):
        return ValidationFinding(
            validator=name,
            severity="warning",
            message=name,
            detail={"origin": name},
        )

    class StubValidator:
        def __init__(self, name):
            self.name = name

        def audit(self, **_kwargs):
            calls.append(self.name)
            return [finding(self.name)]

    def stub_function(name):
        def _stub(**_kwargs):
            calls.append(name)
            return [finding(name)]

        return _stub

    # The deterministic contract sequence now lives in ``contract_gate``; the
    # moved gate looks these collaborators up in THAT module's namespace, so the
    # stubs must patch ``contract_gate`` (not ``pipeline_execute``).
    monkeypatch.setattr(
        contract_gate,
        "_step_contract_findings",
        stub_function("step_contract"),
    )
    monkeypatch.setattr(
        contract_gate,
        "_cohort_definition_sensitivity_contract_findings",
        stub_function("cohort_sensitivity"),
    )
    monkeypatch.setattr(
        contract_gate,
        "_primary_exposure_contract_findings",
        stub_function("primary_exposure"),
    )
    monkeypatch.setattr(
        contract_gate,
        "_primary_exposure_measurement_filter_findings",
        stub_function("exposure_measurement"),
    )
    monkeypatch.setattr(
        contract_gate,
        "_primary_exposure_overadjustment_findings",
        stub_function("overadjustment"),
    )
    monkeypatch.setattr(
        contract_gate,
        "_primary_model_leakage_findings",
        stub_function("model_leakage"),
    )

    def preserve_demotions(name):
        def _demote(*args):
            calls.append(name)
            return list(args[-1])

        return _demote

    monkeypatch.setattr(
        final_validation,
        "_demote_step_contract_for_primary_runner",
        preserve_demotions("primary_runner_demotion"),
    )
    monkeypatch.setattr(
        final_validation,
        "_demote_result_figure_shape_for_family_renderer",
        preserve_demotions("figure_shape_demotion"),
    )
    original_compile = final_validation.compile_sealed_step_result_shadow
    compiler_calls = []

    def compile_once(**kwargs):
        compiler_calls.append(kwargs)
        return original_compile(**kwargs)

    monkeypatch.setattr(
        final_validation,
        "compile_sealed_step_result_shadow",
        compile_once,
    )

    class PassthroughFractionEnvelopeValidator:
        def audit(self, *, legacy_findings, **_kwargs):
            return list(legacy_findings)

    monkeypatch.setattr(
        final_validation,
        "StepSummaryFractionEnvelopeDualReader",
        PassthroughFractionEnvelopeValidator,
    )

    validator_names = {
        "stat_validator": "statistical",
        "clinical_validator": "clinical",
        "statistical_guard": "statistical_guard",
        "cross_step_cohort_lock_validator": "cross_step_cohort_lock",
        "cross_step_registered_output_validator": "cross_step_registered_output",
        "cross_step_reconciliation_trace_validator": "cross_step_reconciliation",
        "step_summary_integrity_validator": "step_summary_integrity",
        "step_summary_fraction_validator": "step_summary_fraction",
        "cross_step_source_status_validator": "cross_step_source_status",
        "primary_model_contract_validator": "primary_model_contract",
        "figure_contract_validator": "figure_contract",
        "figure_source_validator": "figure_source",
    }
    groups = pipeline_execute._evaluate_final_deterministic_gates(
        context=object(),
        plan=AnalysisPlan(
            research_question="Review sealed outputs.",
            steps=[AnalysisStep(step_id="07_review", intent="Review sealed outputs.")],
        ),
        cohort_path=tmp_path / "cohort.parquet",
        universe_path=tmp_path / "universe.parquet",
        run_dir=tmp_path,
        out_dir=tmp_path / "outputs",
        step=AnalysisStep(step_id="07_review", intent="Review sealed outputs."),
        step_summary={},
        step_record={},
        completed_step_records=({"step_id": "06_parent", "status": "ok"},),
        resolved_input_bindings={},
        plausibility_scope=FlagOnlyPlausibilityScope(
            step_id="07_review",
            expected_columns=(),
            source_contracts_sha256="0" * 64,
            authority_kind="test",
        ),
        script_text="",
        attempt_id="attempt-2",
        checkpoint_id="checkpoint-9",
        **{argument: StubValidator(name) for argument, name in validator_names.items()},
    )

    assert calls == [
        "statistical",
        "clinical",
        "statistical_guard",
        "step_contract",
        "cohort_sensitivity",
        "cross_step_cohort_lock",
        "cross_step_registered_output",
        "cross_step_reconciliation",
        "step_summary_integrity",
        "step_summary_fraction",
        "cross_step_source_status",
        "primary_model_contract",
        "primary_exposure",
        "exposure_measurement",
        "overadjustment",
        "model_leakage",
        "figure_contract",
        "primary_runner_demotion",
        "figure_shape_demotion",
        "figure_source",
    ]
    assert [finding.validator for finding in groups.contract_findings] == [
        "step_contract",
        "cohort_sensitivity",
        "cross_step_cohort_lock",
        "cross_step_registered_output",
        "cross_step_reconciliation",
        "step_summary_integrity",
        "step_summary_fraction",
        "cross_step_source_status",
        "primary_model_contract",
        "primary_exposure",
        "exposure_measurement",
        "overadjustment",
        "model_leakage",
        "figure_contract",
    ]
    assert groups.result_envelope_snapshot.ready is True
    assert groups.result_envelope_snapshot.envelope is not None
    assert groups.result_envelope_snapshot.envelope.step_id == "07_review"
    assert groups.result_envelope_snapshot.envelope.paper_authorized is False
    assert len(compiler_calls) == 1
    assert compiler_calls[0]["output_dir"] == tmp_path / "outputs"
    assert [finding.validator for finding in groups.all_findings()] == [
        "statistical",
        "clinical",
        "statistical_guard",
        *[finding.validator for finding in groups.contract_findings],
        "figure_source",
    ]
    for gate_finding in groups.all_findings():
        assert gate_finding.detail == {
            "origin": gate_finding.validator,
            "step_id": "07_review",
            "attempt_id": "attempt-2",
            "checkpoint_id": "checkpoint-9",
        }


def test_final_fraction_consumer_fails_closed_when_sealed_compile_fails(
    monkeypatch,
    tmp_path,
):
    from easyicu.research_agent.authority.plausibility import (
        FlagOnlyPlausibilityScope,
    )
    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.execution import final_validation
    from easyicu.research_agent.execution.envelope_sealing import (
        SealedStepResultEnvelopeSnapshot,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    class EmptyValidator:
        def audit(self, **_kwargs):
            return []

    failed_snapshot = SealedStepResultEnvelopeSnapshot(
        envelope=None,
        error_code="sealed_envelope_compile_failed",
    )
    monkeypatch.setattr(
        final_validation,
        "compile_sealed_step_result_shadow",
        lambda **_kwargs: failed_snapshot,
    )
    monkeypatch.setattr(
        final_validation,
        "_step_execution_cohort_path",
        lambda **_kwargs: tmp_path / "cohort.parquet",
    )
    monkeypatch.setattr(
        final_validation,
        "_bound_step_execution_cohort_path",
        lambda **_kwargs: tmp_path / "cohort.parquet",
    )
    monkeypatch.setattr(
        final_validation,
        "_demote_step_contract_for_primary_runner",
        lambda _record, _summary, findings: list(findings),
    )
    monkeypatch.setattr(
        final_validation,
        "_demote_result_figure_shape_for_family_renderer",
        lambda _context, findings: list(findings),
    )

    def fraction_only_contract_findings(**kwargs):
        return kwargs["final_fraction_envelope_validator"].audit(
            step=kwargs["step"],
            step_summary=kwargs["step_summary"],
            envelope=kwargs["final_fraction_envelope"],
            current_status=kwargs["final_fraction_current_status"],
            legacy_findings=[],
        )

    monkeypatch.setattr(
        final_validation,
        "_step_deterministic_contract_findings",
        fraction_only_contract_findings,
    )
    step = AnalysisStep(step_id="05_result", intent="Seal the final result.")
    groups = pipeline_execute._evaluate_final_deterministic_gates(
        context=object(),
        plan=AnalysisPlan(research_question="Seal the result.", steps=[step]),
        cohort_path=tmp_path / "cohort.parquet",
        universe_path=tmp_path / "universe.parquet",
        run_dir=tmp_path,
        out_dir=tmp_path / "outputs",
        step=step,
        step_summary={},
        step_record={"status": "running"},
        completed_step_records=(),
        resolved_input_bindings={},
        plausibility_scope=FlagOnlyPlausibilityScope(
            step_id=step.step_id,
            expected_columns=(),
            source_contracts_sha256="0" * 64,
            authority_kind="test",
        ),
        script_text="",
        attempt_id="attempt-1",
        checkpoint_id="checkpoint-1",
        stat_validator=EmptyValidator(),
        clinical_validator=EmptyValidator(),
        statistical_guard=EmptyValidator(),
        cross_step_cohort_lock_validator=EmptyValidator(),
        cross_step_registered_output_validator=EmptyValidator(),
        cross_step_reconciliation_trace_validator=EmptyValidator(),
        step_summary_integrity_validator=EmptyValidator(),
        step_summary_fraction_validator=EmptyValidator(),
        cross_step_source_status_validator=EmptyValidator(),
        primary_model_contract_validator=EmptyValidator(),
        figure_contract_validator=EmptyValidator(),
        figure_source_validator=EmptyValidator(),
    )

    assert groups.result_envelope_snapshot == failed_snapshot
    assert len(groups.contract_findings) == 1
    finding = groups.contract_findings[0]
    assert finding.validator == "step_summary_fraction_scale"
    assert finding.severity == "error"
    assert finding.detail["canonical_shadow_blocked"] is True
    assert finding.detail["mismatch_codes"] == ["canonical_envelope_missing"]
    assert finding.detail["step_id"] == step.step_id
    assert finding.detail["attempt_id"] == "attempt-1"
    assert finding.detail["checkpoint_id"] == "checkpoint-1"


def test_execute_phase_host_verifies_measurement_provenance_at_every_contract_gate():
    import ast

    from easyicu.research_agent.execution import phase as pipeline_execute

    # The early repair gate and the final authority gate now share ONE
    # deterministic contract sequence (dedup): the summary-integrity validator is
    # audited exactly once, inside that shared sequence.  The shared gate first
    # resolves the integrity population: ordinary scientific steps use the
    # development execution cohort, while the cohort-producing step keeps the
    # full raw-universe/full locked-cohort authority needed to report attrition.
    shared_tree = ast.parse(
        inspect.getsource(pipeline_execute._step_deterministic_contract_findings)
    )
    shared_audits = [
        node
        for node in ast.walk(shared_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "audit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "step_summary_integrity_validator"
    ]
    assert len(shared_audits) == 1
    shared_keywords = {kw.arg: kw.value for kw in shared_audits[0].keywords}
    assert isinstance(shared_keywords.get("cohort_path"), ast.Name)
    assert shared_keywords["cohort_path"].id == "integrity_universe_path"

    def _shared_gate_calls(function) -> list:
        tree = ast.parse(inspect.getsource(function))
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_step_deterministic_contract_findings"
        ]

    # Early repair screening remains in the orchestration loop; the final
    # read-only review is owned by the reusable deterministic gate evaluator.
    # Each wires in the shared sequence and passes its resolved cohort path.
    direct_calls = _shared_gate_calls(
        pipeline_execute._candidate_contract_setup_transition
    )
    evaluator_calls = _shared_gate_calls(
        pipeline_execute._evaluate_final_deterministic_gates
    )
    assert len(direct_calls) == 1
    assert len(evaluator_calls) == 1
    direct_keywords = {
        keyword.arg: keyword.value for keyword in direct_calls[0].keywords
    }
    evaluator_keywords = {
        keyword.arg: keyword.value for keyword in evaluator_calls[0].keywords
    }
    direct_path = direct_keywords.get("execution_cohort_path")
    assert isinstance(direct_path, ast.Attribute)
    assert isinstance(direct_path.value, ast.Name)
    assert direct_path.value.id == "attempt"
    assert direct_path.attr == "step_execution_cohort_path"
    assert isinstance(evaluator_keywords.get("execution_cohort_path"), ast.Name)
    assert evaluator_keywords["execution_cohort_path"].id == "execution_cohort_path"


def test_primary_cohort_coder_receives_only_exact_locked_cohort_payload():
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        ConceptPredicate,
        TimeWindow,
    )
    from easyicu.research_agent.execution.phase import (
        _planner_locked_cohort_prompt_payload,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    definition = CohortDefinition(
        name="eligible_stays",
        inclusion=(
            ConceptPredicate(
                concept_id="age",
                time_window=TimeWindow(
                    anchor="icu_admit",
                    start_offset_hours=0,
                    end_offset_hours=1,
                ),
                aggregation="first",
                op=">=",
                value=18,
            ),
        ),
    )
    plan = AnalysisPlan(
        research_question="Describe the locked cohort.",
        cohort=definition,
        robustness_specs=[],
        steps=[],
    )

    payload = json.loads(_planner_locked_cohort_prompt_payload(plan))

    assert payload == plan.model_dump(mode="json")["cohort"]
    assert payload["name"] == "eligible_stays"
    assert payload["inclusion"][0]["op"] == ">="
    assert "robustness_specs" not in payload


def test_primary_cohort_coder_receives_verified_physical_predicate_receipt(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        ConceptPredicate,
        TimeWindow,
        materialize_locked_analysis_cohort,
    )
    from easyicu.research_agent.execution.phase import (
        _planner_materialized_cohort_prompt_payload,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    universe_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "stay_id": [11, 12, 13],
            "age": [31.0, None, 54.0],
        }
    ).to_parquet(universe_path, index=False)
    plan = AnalysisPlan(
        research_question="Apply the declared eligibility predicate.",
        cohort=CohortDefinition(
            name="eligible_stays",
            inclusion=(
                ConceptPredicate(
                    concept_id="age",
                    time_window=TimeWindow(
                        anchor="icu_admit",
                        start_offset_hours=0,
                        end_offset_hours=24,
                    ),
                    aggregation="first",
                    op=">=",
                    value=18,
                ),
            ),
        ),
        robustness_specs=[],
        steps=[],
    )
    result = materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=plan,
        universe_path=universe_path,
    )

    payload = json.loads(
        _planner_materialized_cohort_prompt_payload(
            plan=plan,
            universe_path=universe_path,
            analysis_cohort_path=Path(result["path"]),
        )
    )

    assert payload["raw_universe"]["rows"] == 3
    assert payload["authoritative_analysis_cohort"]["rows"] == 2
    assert payload["ordered_predicate_flow"][1] == {
        "aggregation": "first",
        "concept_id": "age",
        "n_before": 3,
        "n_excluded": 1,
        "n_remaining": 2,
        "op": ">=",
        "predicate_kind": "inclusion",
        "resolved_column": "age",
        "step_order": 1,
        "value": 18,
        # A magnitude filter is never narrowed by an event time, so the ledger
        # states that explicitly rather than omitting the fields on some rows.
        "event_time_column": None,
        "event_time_start_hours": None,
        "event_time_end_hours": None,
    }


def test_primary_cohort_raw_runner_is_scoped_and_authority_hashes_are_rechecked():
    from easyicu.research_agent.execution import phase as pipeline_execute
    from easyicu.research_agent.execution import step_attempt_bootstrap
    from easyicu.research_agent.execution.cohort_routing import (
        step_execution_cohort_path,
    )

    candidate_source = inspect.getsource(pipeline_execute._candidate_execute_transition)
    source = (
        inspect.getsource(pipeline_execute.run_execute_phase) + "\n" + candidate_source
    )
    bootstrap_source = inspect.getsource(
        step_attempt_bootstrap.prepare_step_attempt_bootstrap
    )
    authority_source = inspect.getsource(
        pipeline_execute._execution_input_authority_integrity_finding
    )
    routing_source = inspect.getsource(step_execution_cohort_path)

    assert "prepare_step_attempt_bootstrap(" in source
    assert "execution_cohort_path = step_execution_cohort_path(" in bootstrap_source
    assert "primary_analysis_cohort_producer_uses_universe" in routing_source
    assert "return universe_path" in routing_source
    assert "return cohort_path" in routing_source
    assert "only downstream" in routing_source
    assert "cohort_path=attempt.step_execution_cohort_path" in source
    assert (
        '"execution_cohort_sha256": sha256_of_file(universe_path)' in bootstrap_source
    )
    assert (
        '"authoritative_analysis_cohort_sha256": sha256_of_file(cohort_path)'
        in bootstrap_source
    )
    assert "current_universe_sha256 = sha256_of_file(universe_path)" in authority_source
    assert "current_cohort_sha256 = sha256_of_file(cohort_path)" in authority_source
    assert '"status": "blocked_input_authority_mutation"' in source
    assert "or has_primary_cohort_universe_producer" in source
    assert source.count("if run_input_authority_state.corrupted:") == 2
    assert '"remaining_steps_suppressed": True' in source
    assert "primary_cohort_execution_receipt = (" in source
    assert "host_verified_cohort_execution_receipt=(" in source

    runner_call = candidate_source.index(
        "state.run_result = host.step_executor.execute("
    )
    cohort_authority_check = candidate_source.index(
        "cohort_authority_finding = host._execution_input_authority_integrity_finding(",
        runner_call,
    )
    trajectory_authority_check = candidate_source.index(
        "trajectory_authority_finding = (",
        cohort_authority_check,
    )
    authority_gate = candidate_source.index(
        "if authority_findings:",
        trajectory_authority_check,
    )
    unsafe_output_exit = candidate_source.index(
        "if not state.run_result.outputs_safe_to_collect:", runner_call
    )
    authority_latch = candidate_source.index(
        "host.run_input_authority_state.mark_corrupted(", authority_gate
    )
    assert (
        runner_call
        < cohort_authority_check
        < trajectory_authority_check
        < authority_gate
        < authority_latch
        < unsafe_output_exit
    )
    assert (
        "if state.run_result.outputs_safe_to_collect:"
        in candidate_source[authority_gate:unsafe_output_exit]
    )
    assert (
        "_clear_output_dir(state.run_result.out_dir)"
        in candidate_source[authority_gate:unsafe_output_exit]
    )
    assert (
        "_seal_actual_execution_result()"
        not in candidate_source[authority_gate:unsafe_output_exit]
    )


def test_every_runner_build_receives_the_selected_trajectory_authority():
    from easyicu.research_agent.execution import phase as pipeline_execute

    tree = ast.parse(
        inspect.getsource(pipeline_execute.run_execute_phase)
        + "\n"
        + inspect.getsource(pipeline_execute._candidate_execute_transition)
    )
    runner_builds = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_build_runner"
        and (
            (isinstance(node.func.value, ast.Name) and node.func.value.id == "pipeline")
            or (
                isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "host"
                and node.func.value.attr == "pipeline"
            )
        )
    ]

    # Initial, post-materialization, and per-step runner construction are the
    # three live call sites.  The former fourth build was retired when runner
    # authority selection was centralized; keep this count aligned with live
    # construction sites rather than the historical implementation shape.
    assert len(runner_builds) == 3
    for call in runner_builds:
        starred = [keyword.value for keyword in call.keywords if keyword.arg is None]
        assert len(starred) == 1
        binding_call = starred[0]
        assert isinstance(binding_call, ast.Call)
        assert isinstance(binding_call.func, ast.Attribute)
        owner = binding_call.func.value
        if isinstance(owner, ast.Name):
            assert owner.id == "run_input_authority_state"
        else:
            assert isinstance(owner, ast.Attribute)
            assert isinstance(owner.value, ast.Name)
            assert owner.value.id == "host"
            assert owner.attr == "run_input_authority_state"
        assert binding_call.func.attr == "runner_bindings"


def test_execution_input_authority_check_detects_unsafe_runner_mutation(tmp_path):
    from easyicu.research_agent.authority.evidence_store import sha256_of_file
    from easyicu.research_agent.execution.phase import (
        _execution_input_authority_integrity_finding,
    )

    universe_path = tmp_path / "universe.parquet"
    cohort_path = tmp_path / "cohort_analysis.parquet"
    universe_path.write_bytes(b"raw-universe")
    cohort_path.write_bytes(b"filtered-cohort")
    expected_universe_sha256 = sha256_of_file(universe_path)
    expected_cohort_sha256 = sha256_of_file(cohort_path)

    class UnsafeMutatingRunner:
        def run(self):
            universe_path.write_bytes(b"mutated-by-runner")
            return type("UnsafeResult", (), {"outputs_safe_to_collect": False})()

    result = UnsafeMutatingRunner().run()
    finding = _execution_input_authority_integrity_finding(
        step_id="01_cohort_flow",
        universe_path=universe_path,
        cohort_path=cohort_path,
        expected_universe_sha256=expected_universe_sha256,
        expected_analysis_cohort_sha256=expected_cohort_sha256,
    )

    assert result.outputs_safe_to_collect is False
    assert finding is not None
    assert finding.validator == "execution_input_authority_integrity"
    assert finding.severity == "error"
    assert finding.detail["observed_universe_sha256"] != expected_universe_sha256
    assert finding.detail["observed_analysis_cohort_sha256"] == expected_cohort_sha256


def test_plan_and_execute_result_dataclass_shapes_match_contracts_module():
    """Pin the three dataclasses that flow through the pipeline phases.

    The pipeline phases exchange ``_PlanPhaseResult``,
    ``_ExecutePhaseResult`` and ``_WritePhaseResult``. They are defined in
    ``contracts.runtime`` and re-exported by ``pipeline.py``. If any shape
    drifts, a phase silently misreads its input or produces a malformed handoff
    to the next phase.
    """
    from easyicu.research_agent.contracts.runtime import (
        _PlanPhaseResult,
        _ExecutePhaseResult,
        _WritePhaseResult,
    )
    from easyicu.research_agent.pipeline import (
        _PlanPhaseResult as PipelinePlanPhaseResult,
        _ExecutePhaseResult as PipelineExecutePhaseResult,
        _WritePhaseResult as PipelineWritePhaseResult,
    )

    assert PipelinePlanPhaseResult is _PlanPhaseResult
    assert PipelineExecutePhaseResult is _ExecutePhaseResult
    assert PipelineWritePhaseResult is _WritePhaseResult

    plan_fields = {f.name for f in fields(_PlanPhaseResult)}
    # Names the execute phase actually reads off plan_result, verified
    # against pipeline_execute.run_execute_phase body 2026-05-17.
    required_plan_fields = {
        "context",
        "agent_context",
        "evidence",
        "findings",
        "plan",
        "plan_path",
        "role_resolver",
        "llm_signature",
        "prompt_version",
        "prompt_files",
        "resume_state",
    }
    missing = required_plan_fields - plan_fields
    assert not missing, (
        f"_PlanPhaseResult is missing fields {missing} consumed by run_execute_phase."
    )

    exec_fields = {f.name for f in fields(_ExecutePhaseResult)}
    required_exec_fields = {
        "plan",
        "per_step_records",
        "probe_summary",
        "runtime_state",
        "flush_partial_manifest",
    }
    missing_exec = required_exec_fields - exec_fields
    assert not missing_exec, (
        f"_ExecutePhaseResult is missing fields {missing_exec} produced "
        "by run_execute_phase / consumed by the write phase."
    )

    write_fields = {f.name for f in fields(_WritePhaseResult)}
    required_write_fields = {
        "literature",
        "bound_path",
        "manuscript_packet",
        "manuscript_critique",
    }
    missing_write = required_write_fields - write_fields
    assert not missing_write, (
        f"_WritePhaseResult is missing fields {missing_write} produced "
        "by the write phase / consumed by the package phase."
    )


def test_required_collaborators_are_importable():
    """Smoke-import each collaborator name pipeline_execute pulls in.

    A typo in one of the agent / validator / repair imports would only
    surface when the execute phase actually fires, which in the e2e
    suite is many minutes in. We import them upfront here.
    """
    from easyicu.research_agent.execution.phase import (  # noqa: F401
        AnalyzerAgent,
        ClinicalSemanticsAgent,
        CoderAgent,
        CriticAgent,
        DataExtractionAgent,
        ReplannerAgent,
        RuntimeSupervisor,
        StatisticalAnalysisAgent,
        VisualizationAgent,
        ClinicalConstraintValidator,
        ConceptUsageAuditor,
        LLMConceptAuditor,
        StatisticalGuard,
        StatisticalValidator,
        _deterministic_runner_repair,
        _deterministic_summary_repair,
        MockLLMClient,
    )


def test_visual_qa_demotes_only_cosmetic_layout_errors(ra):
    from easyicu.research_agent.execution.phase import (
        _demote_cosmetic_visual_findings,
    )
    from easyicu.research_agent.schema import ValidationFinding

    cosmetic = ValidationFinding(
        validator="visual_qa",
        severity="error",
        message=(
            "SVG figure 'x.svg' has overlapping text elements; "
            "multi-panel labels, annotations or axis text need more spacing."
        ),
    )
    hard = ValidationFinding(
        validator="visual_qa",
        severity="error",
        message="Could not open figure 'x.png': truncated image file",
    )
    vlm = ValidationFinding(
        validator="vlm_visual_qa",
        severity="error",
        message="Panel B axis values do not match source data.",
    )

    demoted, blocking = _demote_cosmetic_visual_findings([cosmetic, hard, vlm])

    assert demoted[0].severity == "warning"
    assert demoted[1].severity == "error"
    assert demoted[2].severity == "error"
    assert [f.message for f in blocking] == [hard.message, vlm.message]


def test_scope_findings_step_global_warning_does_not_taint_records():
    """A step-global warning (no evidence_ids) is an analysis-design advisory
    and must NOT taint the citability of the step's output records — otherwise
    one 'immortal-time-bias risk' note makes the primary result table
    uncitable and the manuscript unwinnable."""
    from easyicu.research_agent.execution.phase import scope_findings_to_records
    from easyicu.research_agent.schema import ValidationFinding

    global_warning = ValidationFinding(
        validator="clinical_constraint_validator",
        severity="warning",
        message="Treatment-effect analysis without an explicit time-zero.",
    )
    scoped = scope_findings_to_records(
        ["table_one", "adjusted_association"], [global_warning]
    )
    assert scoped["table_one"] == (None, [])
    assert scoped["adjusted_association"] == (None, [])


def test_scope_findings_targeted_finding_taints_only_named_record():
    """A finding that names specific records taints ONLY those records."""
    from easyicu.research_agent.execution.phase import scope_findings_to_records
    from easyicu.research_agent.schema import ValidationFinding

    global_warning = ValidationFinding(
        validator="clinical_constraint_validator",
        severity="warning",
        message="Design advisory.",
    )
    targeted = ValidationFinding(
        validator="critic_agent",
        severity="warning",
        message="Critique of the interpretation log.",
        evidence_ids=["log_critique_report_x"],
    )
    scoped = scope_findings_to_records(
        ["table_one", "log_critique_report_x"], [global_warning, targeted]
    )
    assert scoped["table_one"] == (None, [])
    severity, messages = scoped["log_critique_report_x"]
    assert severity == "warning"
    assert messages == ["Critique of the interpretation log."]


def test_scope_findings_step_global_error_stays_fail_closed():
    """A step-global ERROR keeps the blanket taint (fail-closed): a step-level
    error means the step's outputs are not to be trusted."""
    from easyicu.research_agent.execution.phase import scope_findings_to_records
    from easyicu.research_agent.schema import ValidationFinding

    global_error = ValidationFinding(
        validator="execution",
        severity="error",
        message="Step analysis crashed before producing a result.",
    )
    scoped = scope_findings_to_records(
        ["table_one", "adjusted_association"], [global_error]
    )
    for eid in ("table_one", "adjusted_association"):
        severity, messages = scoped[eid]
        assert severity == "error"
        assert messages == ["Step analysis crashed before producing a result."]


def test_success_alias_filter_preserves_parent_role_but_allows_same_step_retry():
    from easyicu.research_agent.execution.phase import (
        _filter_success_alias_bindings,
    )

    filtered, retained, suppressed = _filter_success_alias_bindings(
        {
            "figure_new": ["primary_association", "association_figure"],
            "summary_new": ["step_summary"],
        },
        existing_aliases={
            "primary_association": "parent_result",
            "step_summary": "summary_old",
        },
        owners_by_evidence_id={
            "parent_result": "04_primary_association",
            "summary_old": "04_primary_association_figure",
        },
        step_id="04_primary_association_figure",
    )

    assert filtered == {
        "figure_new": ["association_figure"],
        "summary_new": ["step_summary"],
    }
    assert retained == {"primary_association": "parent_result"}
    assert suppressed == set()


@pytest.mark.parametrize(
    ("product_id", "kind", "filename"),
    [
        ("table_result", "table", "primary_result.csv"),
        ("figure_result", "figure", "primary_result.svg"),
    ],
)
def test_success_alias_filter_assigns_product_role_to_real_product_not_summary(
    product_id,
    kind,
    filename,
):
    from easyicu.research_agent.execution.phase import (
        _filter_success_alias_bindings,
    )

    filtered, _, suppressed = _filter_success_alias_bindings(
        {
            "summary": ["primary_result", "01_model"],
            product_id: ["primary_result"],
        },
        existing_aliases={},
        owners_by_evidence_id={},
        step_id="01_model",
        records_by_evidence_id={
            "summary": {
                "evidence_id": "summary",
                "kind": "statistic",
                "relative_path": "evidence/summary__step_summary.json",
            },
            product_id: {
                "evidence_id": product_id,
                "kind": kind,
                "relative_path": f"evidence/{product_id}__{filename}",
            },
        },
    )

    assert filtered[product_id] == ["primary_result"]
    assert filtered["summary"] == ["01_model"]
    assert suppressed == set()


def test_success_alias_filter_keeps_distinct_real_product_collision_fail_closed():
    from easyicu.research_agent.execution.phase import (
        _filter_success_alias_bindings,
    )

    filtered, _, suppressed = _filter_success_alias_bindings(
        {
            "table_a": ["primary_result"],
            "table_b": ["primary_result"],
        },
        existing_aliases={},
        owners_by_evidence_id={},
        step_id="01_model",
        records_by_evidence_id={
            "table_a": {
                "evidence_id": "table_a",
                "kind": "table",
                "relative_path": "evidence/table_a__effect.csv",
            },
            "table_b": {
                "evidence_id": "table_b",
                "kind": "table",
                "relative_path": "evidence/table_b__different_effect.csv",
            },
        },
    )

    assert filtered["table_a"] == ["primary_result"]
    assert filtered["table_b"] == ["primary_result"]
    assert suppressed == set()


def test_success_alias_filter_suppresses_implicit_cross_kind_basename_collision():
    from easyicu.research_agent.execution.phase import (
        _filter_success_alias_bindings,
    )

    filtered, retained, suppressed = _filter_success_alias_bindings(
        {
            "table_result": [],
            "artifact_result": [],
        },
        existing_aliases={},
        owners_by_evidence_id={},
        step_id="05_diagnostics",
        records_by_evidence_id={
            "table_result": {
                "evidence_id": "table_result",
                "kind": "table",
                "relative_path": "evidence/table_result__positivity_diagnostics.csv",
            },
            "artifact_result": {
                "evidence_id": "artifact_result",
                "kind": "log",
                "relative_path": "evidence/artifact_result__positivity_diagnostics.json",
            },
        },
    )

    assert filtered == {"table_result": [], "artifact_result": []}
    assert retained == {}
    assert suppressed == {"table_result", "artifact_result"}


def test_success_alias_filter_suppresses_ambiguous_typed_stem_advertised_by_summary():
    from easyicu.research_agent.execution.phase import (
        _filter_success_alias_bindings,
    )

    filtered, retained, suppressed = _filter_success_alias_bindings(
        {
            "summary": ["robustness_summary", "06_sensitivity"],
            "table_result": [],
            "statistic_result": [],
        },
        existing_aliases={},
        owners_by_evidence_id={},
        step_id="06_sensitivity",
        records_by_evidence_id={
            "summary": {
                "evidence_id": "summary",
                "kind": "statistic",
                "relative_path": "evidence/summary__step_summary.json",
            },
            "table_result": {
                "evidence_id": "table_result",
                "kind": "table",
                "relative_path": "evidence/table_result__robustness_summary.csv",
            },
            "statistic_result": {
                "evidence_id": "statistic_result",
                "kind": "log",
                "relative_path": "evidence/statistic_result__robustness_summary.json",
            },
        },
    )

    assert filtered == {
        "summary": ["06_sensitivity"],
        "table_result": [],
        "statistic_result": [],
    }
    assert retained == {}
    assert suppressed == {"table_result", "statistic_result"}


def test_success_alias_filter_prefers_vector_export_for_one_logical_figure():
    from easyicu.research_agent.execution.phase import (
        _filter_success_alias_bindings,
    )

    filtered, _, suppressed = _filter_success_alias_bindings(
        {
            "png": ["missingness_heatmap"],
            "svg": ["missingness_heatmap"],
        },
        existing_aliases={},
        owners_by_evidence_id={},
        step_id="03_missingness_audit_figure",
        records_by_evidence_id={
            "png": {
                "evidence_id": "png",
                "kind": "figure",
                "relative_path": "evidence/png__missingness_heatmap.png",
            },
            "svg": {
                "evidence_id": "svg",
                "kind": "figure",
                "relative_path": "evidence/svg__missingness_heatmap.svg",
            },
        },
    )

    assert filtered["png"] == []
    assert filtered["svg"] == ["missingness_heatmap"]
    assert suppressed == {"png"}

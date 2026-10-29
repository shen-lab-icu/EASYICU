"""Persist HTTP status without provider bodies, URLs, headers, or messages."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver import agent_pipeline_runs


def test_a_survival_input_gap_reaches_the_gate_and_its_plain_message(tmp_path):
    from easyicu.research_agent.agents.family_spec_planner import family_result_stop
    from easyicu.webserver.pi_copilot.projections import gate_detail_projection
    from tests.support.survival_proposal import survival_context

    context = survival_context()
    context = context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"database": "sic"}),
        "variables": [v for v in context.variables if v.name != "followup_days_90d"],
    })
    stop = family_result_stop(
        context, families=("survival",), planner_strategy="family_spec_v1",
        planning_contract_context="",
    )
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path, study={"id": "synthetic-survival-input"}, provider={},
        exc=stop, code=agent_pipeline_runs._pipeline_failure_code(stop), execution_retry_id=None,
    )

    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate_detail_projection(gate["detail"]) == {
        "gate_detail_code": "progressive_family_result_contract_unwritable",
        "gate_detail_cause_code": "required_concepts_unavailable",
        "gate_missing_concepts": ["followup_days_90d"],
    }
    message = agent_pipeline_runs._progressive_compile_failure_message(stop)
    assert "prepared data lack the event or censoring time" in message
    assert "followup_days_90d" in message
    assert "No analysis was run" in message


@pytest.mark.parametrize("reason, cause, concepts", [
    ("progressive_family_result_contract_unwritable", None, ["followup_days_90d"]),
    ("progressive_family_result_contract_unwritable", "required_concepts_unavailable", []),
    ("progressive_unknown_literature_source", "required_concepts_unavailable", ["followup_days_90d"]),
])
def test_a_generic_stop_does_not_claim_a_survival_time_column_is_missing(reason, cause, concepts):
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError

    stop = ProgressivePlanCompileError(
        reason, "private synthetic detail",
        cause_code=cause, concepts=concepts,
    )
    message = agent_pipeline_runs._progressive_compile_failure_message(stop)
    assert "lack the event or censoring time" not in message
    if reason == "progressive_family_result_contract_unwritable":
        assert "prepared input and design" in message
    assert "followup_days_90d" not in message
    assert "private synthetic detail" not in message


@pytest.mark.parametrize("cause_code", [
    "RUN_PROVIDER_ATTEMPT_LIMIT", "BATCH_PROVIDER_ATTEMPT_LIMIT",
    "RUN_TOKEN_LIMIT", "BATCH_TOKEN_LIMIT", "BATCH_COST_LIMIT",
    "TASK_WALL_CLOCK_EXHAUSTED",
])
@pytest.mark.parametrize("wrapped", [False, True])
def test_provider_hard_stop_has_its_own_run_code_and_gate_cause(
    tmp_path, cause_code, wrapped,
):
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopExceeded
    from easyicu.webserver.pi_copilot.projections import gate_detail_projection

    stop = ProviderHardStopExceeded(code=cause_code, detail="private synthetic detail")
    # The retry which exhausted the budget can retain an earlier timeout.
    stop.__context__ = TimeoutError("earlier synthetic failure")
    failure = RuntimeError("outer synthetic failure") if wrapped else stop
    if wrapped:
        failure.__cause__ = stop
    code = agent_pipeline_runs._pipeline_failure_code(failure)
    assert code == "research_pipeline_provider_budget_exhausted"
    assert agent_pipeline_runs._pipeline_failure_code(
        failure, budget_mode="planner_canary",
    ) == code
    agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path, study={"id": "synthetic-study"}, provider={},
        exc=failure, code=code, execution_retry_id=None,
    )
    gate = json.loads((tmp_path / "quality_gate.json").read_text())["gate"]
    assert gate["reason"] == code
    assert gate_detail_projection(gate["detail"]) == {
        "gate_detail_code": "provider_budget_exhausted",
        "gate_detail_cause_code": cause_code,
        "gate_missing_concepts": [],
    }
    assert "private synthetic detail" not in json.dumps(gate)


def test_an_error_body_cannot_claim_to_be_a_provider_hard_stop():
    assert agent_pipeline_runs._pipeline_failure_code(
        RuntimeError("ProviderHardStopExceeded RUN_TOKEN_LIMIT"),
    ) == "research_pipeline_execution_failed"


def test_an_unknown_hard_stop_code_cannot_leak_into_the_gate_detail():
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopExceeded

    typed = agent_pipeline_runs._safe_pipeline_typed_failure(
        ProviderHardStopExceeded(code="private-synthetic-value", detail="private detail"),
    )
    assert typed == {
        "owner": "easyicu.providers.hard_stop_v1",
        "reason_code": "provider_budget_exhausted",
        "cause_code": "PROVIDER_HARD_STOP_EXCEEDED",
    }


def test_outer_http_failure_takes_precedence_over_contextual_compiler_failure():
    from easyicu.research_agent.planning.progressive_contract import (
        ProgressivePlanCompileError,
    )

    try:
        raise ProgressivePlanCompileError(
            "progressive_typed_product_specs_invalid",
            "an earlier bounded repair failed",
            step_id="baseline_context",
            step_index=1,
            path="typed_product_specs",
        )
    except ProgressivePlanCompileError:
        transport = RuntimeError("provider body must stay private")
        transport.response = SimpleNamespace(
            status_code=503,
            headers={"Authorization": "secret"},
            text="secret",
            url="https://private.example/patient",
        )
        try:
            raise transport
        except RuntimeError as failure:
            assert agent_pipeline_runs._safe_pipeline_typed_failure(failure) == {
                "owner": "easyicu.providers.http_transport_v1",
                "reason_code": "provider_http_error",
                "status_code": 503,
            }
            assert agent_pipeline_runs._pipeline_failure_code(
                failure,
                budget_mode="planner_canary",
            ) == "research_pipeline_planner_provider_unavailable"


def test_prompt_budget_failure_publishes_its_own_measurements():
    """A size gate that hides its numbers cannot be acted on.

    ``progressive_prompt_budget_exceeded`` aborted plan generation twice while
    the Web diagnostic carried only "the host compiler rejected the bounded
    Planner repairs": the request size and the envelope it crossed were already
    computed at the check and then thrown away, so the only way to learn them
    was to fail again. Integers under a canonical key are the one part of that
    request that is safe to publish.
    """
    from easyicu.research_agent.planning.progressive_contract import (
        ProgressivePlanCompileError,
    )

    failure = ProgressivePlanCompileError(
        "progressive_prompt_budget_exceeded",
        "initial request uses 96000 bytes; limit=90000",
        path="planner_request",
        metrics={
            "request_bytes": 96000,
            "byte_limit": 90000,
            "message_bytes": 88000,
            "schema_bytes": 8000,
            # Anything that is not a canonical, bounded, non-negative integer
            # stays out of a projection that is allowed to reach a browser.
            "PromptText": "patient row 12345678",
            "negative": -5,
            "huge": 10 ** 30,
            "floaty": 1.5,
        },
    )

    projected = agent_pipeline_runs._safe_pipeline_typed_failure(failure)

    assert projected == {
        "owner": "easyicu.planning.progressive_compiler_v1",
        "reason_code": "progressive_prompt_budget_exceeded",
        "path": "planner_request",
        "metrics": {
            "request_bytes": 96000,
            "byte_limit": 90000,
            "message_bytes": 88000,
            "schema_bytes": 8000,
        },
    }
    assert "12345678" not in json.dumps(projected)
    assert "96000 bytes" not in json.dumps(projected)
    assert agent_pipeline_runs._pipeline_failure_code(failure) == (
        "research_pipeline_progressive_compile_failed"
    )


@pytest.mark.parametrize("status", [401, 429, 503])
def test_pipeline_failure_keeps_typed_http_status_across_exception_chain(tmp_path, status):
    secret = "sk-secret-provider-message-and-patient-fragment"
    transport = RuntimeError(secret)
    transport.response = SimpleNamespace(
        status_code=status, headers={"Authorization": secret}, text=secret,
        url="https://private.example/patient",
    )
    wrapper = RuntimeError("outer " + secret)
    wrapper.__cause__ = transport
    relative = agent_pipeline_runs._write_pipeline_failure_diagnostic(
        wrapper_dir=tmp_path, exc=wrapper, code="research_pipeline_execution_failed",
    )
    payload = json.loads((tmp_path / relative).read_text())
    assert payload["failure_type"] == "provider_http"
    assert payload["typed_failure"] == {
        "owner": "easyicu.providers.http_transport_v1",
        "reason_code": "provider_http_error",
        "status_code": status,
        "step_id": "unknown", "phase": "unknown", "runner_kind": "unknown", "exception_type": "RuntimeError",
    }
    rendered = json.dumps(payload)
    assert secret not in rendered
    assert "private.example" not in rendered
    assert "Authorization" not in rendered
    assert payload["secrets_recorded"] is False


def test_untyped_http_wording_does_not_invent_a_status():
    failure = RuntimeError("HTTP status 503")
    assert agent_pipeline_runs._safe_pipeline_typed_failure(failure) == {}


@pytest.mark.parametrize("link", [None, "__cause__", "__context__"])
def test_a_scientific_authority_plan_failure_keeps_its_typed_cause(tmp_path, link):
    from easyicu.research_agent.authority.current_case_scientific_runtime import (
        CurrentCaseScientificAuthorityError,
    )

    private = "private synthetic plan detail"
    failure = CurrentCaseScientificAuthorityError(private)
    if link:
        wrapper = RuntimeError("plan phase failed")
        setattr(wrapper, link, failure)
        failure = wrapper
    relative = agent_pipeline_runs._record_pipeline_failure(
        wrapper_dir=tmp_path, study={"id": "study_synthetic"}, provider={},
        exc=failure, code="research_pipeline_execution_failed", execution_retry_id=None,
    )
    payload = json.loads((tmp_path / relative).read_text())
    assert payload["schema_version"] == "easyicu.web-research-pipeline-failure/4"
    assert payload["typed_failure"] == {
        "owner": "easyicu.current_case_scientific_authority_v1",
        "reason_code": "current_case_scientific_authority_invalid",
        "exception_type": "CurrentCaseScientificAuthorityError",
        "step_id": "unknown", "phase": "unknown", "runner_kind": "unknown",
    }
    assert private not in json.dumps(payload)
    assert agent_pipeline_runs._safe_pipeline_typed_failure(ValueError(private)) == {}


def test_a_scientific_authority_refusal_has_its_own_code_and_sentence():
    from easyicu.research_agent.authority.current_case_scientific_runtime import (
        CurrentCaseScientificAuthorityError,
    )

    wrapper = RuntimeError("plan phase failed")
    wrapper.__cause__ = CurrentCaseScientificAuthorityError("private synthetic detail")

    code = agent_pipeline_runs._pipeline_failure_code(wrapper)

    assert code == "research_pipeline_scientific_authority_refused"
    error_text = (
        Path(agent_pipeline_runs.__file__).resolve().parent
        / "static" / "js" / "screens-guided-pi-error-text.js"
    ).read_text(encoding="utf-8")
    assert f"{code}: tr(" in error_text
    entry = error_text.split(f"{code}: tr(", 1)[1].split("\n        ),", 1)[0]
    assert "24 小时 landmark" in entry


def test_an_unrecognized_scientific_authority_cause_stays_private():
    from easyicu.research_agent.authority.current_case_scientific_runtime import (
        CurrentCaseScientificAuthorityError,
    )

    private = "private synthetic cause"
    failure = CurrentCaseScientificAuthorityError(private)
    failure.reason_code = private
    projected = agent_pipeline_runs._safe_pipeline_typed_failure(failure)
    assert projected["reason_code"] == "current_case_scientific_authority_invalid"
    assert private not in json.dumps(projected)


def test_existing_typed_owner_takes_precedence_over_transport_cause():
    failure = RuntimeError("private compiler detail")
    failure.easyicu_safe_diagnostic = {
        "owner": "easyicu.planning.progressive_compiler_v1",
        "reason_code": "progressive_step_invalid",
    }
    failure.__cause__ = RuntimeError("private provider detail")
    failure.__cause__.status_code = 503
    assert agent_pipeline_runs._safe_pipeline_typed_failure(failure) == {
        "owner": "easyicu.planning.progressive_compiler_v1",
        "reason_code": "progressive_step_invalid",
    }


def test_a_typed_compile_stop_reaches_the_run_record_as_its_gate_detail(tmp_path):
    """The run record names the owner's reason, not the generic compile wording."""
    from easyicu.research_agent.planning.progressive_contract import (
        ProgressivePlanCompileError,
    )
    from easyicu.webserver.pi_copilot.workflow import gate_detail_projection

    stop = ProgressivePlanCompileError(
        "progressive_family_result_contract_unwritable",
        "no host owner can write the primary result contract for survival",
        path="analysis_type",
    )
    code = agent_pipeline_runs._pipeline_failure_code(stop)
    assert code == "research_pipeline_progressive_compile_failed"

    typed_dir, untyped_dir = tmp_path / "typed", tmp_path / "untyped"
    for wrapper_dir, exc, failure_code in (
        (typed_dir, stop, code),
        (untyped_dir, RuntimeError("boom"), "research_pipeline_execution_failed"),
    ):
        wrapper_dir.mkdir()
        agent_pipeline_runs._record_pipeline_failure(
            wrapper_dir=wrapper_dir,
            study={"id": "study_synthetic"},
            provider={},
            exc=exc,
            code=failure_code,
            execution_retry_id=None,
        )

    gate = json.loads((typed_dir / "quality_gate.json").read_text())["gate"]
    assert gate["reason"] == code
    assert gate["detail"] == {"reason_code": "progressive_family_result_contract_unwritable"}
    assert gate_detail_projection(gate["detail"]) == {
        "gate_detail_code": "progressive_family_result_contract_unwritable",
        "gate_missing_concepts": [],
    }
    assert "no host owner" not in json.dumps(gate)
    assert json.loads((untyped_dir / "quality_gate.json").read_text())["gate"]["detail"] == {
        "reason_code": "pipeline_operation_raised",
    }
    assert "planning stopped before its analysis steps were drafted" in (
        agent_pipeline_runs._progressive_compile_failure_message(stop)
    )
    assert "rejected the bounded Planner repairs" in (
        agent_pipeline_runs._progressive_compile_failure_message(
            ProgressivePlanCompileError("progressive_typed_product_specs_invalid", "x")
        )
    )

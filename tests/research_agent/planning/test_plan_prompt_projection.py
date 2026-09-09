"""Sealed-plan prompt compression must preserve JSON types and retry space."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from easyicu.research_agent.agents import progressive_planner as progressive
from easyicu.research_agent.agents.planner import PlannerAgent
from easyicu.research_agent.canonical_json import canonical_json_bytes, sha256_bytes
from easyicu.research_agent.planning.prompt_projection import (
    planner_prompt_byte_limit, project_plan_revision_prompt, retry_shape_reminder,
)
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.providers import structured_retry
from easyicu.research_agent.providers.prompt_budget import (
    PromptBudgetClient, PROMPT_TRANSPORT_BUDGETS, PromptTransportBudgetError,
)
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from .progressive_planner_fixtures import _context


def _plan():
    return AnalysisPlan(
        research_question="Preserve every outcome, source and sensitivity analysis.",
        steps=[AnalysisStep(step_id="describe", intent="Describe all baseline variables")],
    ).model_dump(mode="json")


def _project(payload):
    raw = json.dumps(payload, ensure_ascii=False)
    text = "Bound source and review remain authoritative.\r\n- source_plan_json: " + raw + "\r\n"
    projected, receipts = project_plan_revision_prompt(text)
    value = json.loads(projected.splitlines()[1].split(": ", 1)[1])
    return text, projected, value, receipts[0]


def test_defaults_restore_exact_canonical_bytes_and_sealed_source_stays_unchanged(tmp_path):
    payload = _plan()
    raw = json.dumps(payload, indent=2).encode()
    sealed = tmp_path / "agent_plan.json"
    sealed.write_bytes(raw)
    before = sha256_bytes(raw)
    _, projected, compact, receipt = _project(payload)
    assert receipt["status"] == "compacted"
    assert "revision" not in compact
    restored = AnalysisPlan.model_validate(compact).model_dump(mode="json")
    assert canonical_json_bytes(restored) == canonical_json_bytes(payload)
    assert sealed.read_bytes() == raw
    assert sha256_bytes(sealed.read_bytes()) == before
    assert projected.endswith("\r\n")
    assert project_plan_revision_prompt(projected)[0] == projected


def test_explicit_nondefaults_and_scientific_fields_survive():
    payload = _plan()
    payload.update(revision=3, rationale="Keep the reverse contrast.", analysis_type="association_study")
    payload["display_labels"] = {"event=0": "Event present", "event=1": "Event absent"}
    _, _, compact, receipt = _project(payload)
    assert receipt["status"] == "compacted"
    assert compact["revision"] == 3
    assert compact["display_labels"] == payload["display_labels"]
    assert canonical_json_bytes(AnalysisPlan.model_validate(compact).model_dump(mode="json")) == canonical_json_bytes(payload)


@pytest.mark.parametrize("revision", [True, 1.0, 0.0])
def test_type_coercion_retains_original_bytes_and_records_reason(revision):
    payload = _plan()
    payload["revision"] = revision
    text, projected, _, receipt = _project(payload)
    assert text == projected
    assert receipt["status"] == "retained"
    assert receipt["reason"] == "canonical_roundtrip_mismatch"
    assert receipt["source_sha256"] == receipt["projected_sha256"]


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(unknown_science_requirement="must survive"),
    lambda p: p["steps"][0].update(unknown_model_binding="must survive"),
    lambda p: p.pop("revision"),
])
def test_unknown_and_sparse_sources_are_not_normalized(mutation):
    payload = deepcopy(_plan())
    mutation(payload)
    text, projected, _, receipt = _project(payload)
    assert projected == text
    assert receipt["status"] == "retained"
    assert receipt["reason"] in {"invalid_json_or_schema", "canonical_roundtrip_mismatch"}


@pytest.mark.parametrize("raw", ['{broken', '{"revision":1,"revision":2}', '{"revision":NaN}'])
def test_corrupt_json_is_kept_verbatim(raw):
    original = "- source_plan_json: " + raw
    projected, receipts = project_plan_revision_prompt(original)
    assert projected == original
    assert receipts[0]["reason"] == "invalid_json_or_schema"


def test_seed_minify_only_changes_whitespace_and_order():
    raw = '{ "unknown": true, "numeric": 0.0, "array": [0, 1.0, false] }'
    projected, receipts = project_plan_revision_prompt("- candidate_plan_seed_json: " + raw)
    value = json.loads(projected.split(": ", 1)[1])
    assert canonical_json_bytes(value) == canonical_json_bytes(json.loads(raw))
    assert receipts[0]["reason"] == "canonical_json_only"


class _NoProvider:
    supports_strict_json_schema = False

    def complete(self, *args, **kwargs):
        raise AssertionError("Real transport forbidden")


@pytest.mark.parametrize("limit", [100, 40000, 41000])
def test_preflight_and_transport_agree_including_schema_and_configured_limit(limit):
    client = PromptBudgetClient(_NoProvider(), budget=PROMPT_TRANSPORT_BUDGETS[
        "planner_plan_generation"
    ].with_limit_tokens(limit))
    byte_limit = planner_prompt_byte_limit(client)
    schema = SimpleNamespace(payload_bytes=3)
    messages = [LLMMessage(role="user", content="x" * (byte_limit - 3))]
    client._enforce(messages, schema)
    messages[0] = LLMMessage(role="user", content=messages[0].content + "x")
    with pytest.raises(PromptTransportBudgetError):
        client._enforce(messages, schema)
    assert byte_limit == limit * 3


def test_progressive_preflight_rejects_old_four_byte_gap_before_transport(monkeypatch):
    monkeypatch.setattr(progressive, "_GUIDE", "")
    monkeypatch.setattr(progressive.ProgressivePlannerAgent, "_user_prompt", lambda *a, **kw: "x" * 120001)
    with pytest.raises(ProgressivePlanCompileError, match="limit=120000"):
        progressive.ProgressivePlannerAgent(_NoProvider()).run(_context())


@pytest.mark.parametrize("agent_class", [PlannerAgent, progressive.ProgressivePlannerAgent])
def test_shared_projection_reaches_both_planners_without_rebinding_source(monkeypatch, agent_class):
    raw = json.dumps(_plan())
    contract = "- source_plan_sha256: " + sha256_bytes(raw.encode()) + "\n- source_plan_json: " + raw
    expected, receipts = project_plan_revision_prompt(contract)
    captured = []
    authorities = []
    build = progressive.build_progressive_checkpoint_authorities

    def capture_authority(**kwargs):
        authorities.append(kwargs["scientific_authority"]["planning_contract_context"])
        return build(**kwargs)

    class Captured(Exception):
        pass

    def capture(llm, messages, **kwargs):
        captured.extend(messages)
        raise Captured()

    monkeypatch.setattr(progressive, "build_progressive_checkpoint_authorities", capture_authority)
    monkeypatch.setattr(progressive, "call_llm_with_structured_retry", capture)
    monkeypatch.setattr(structured_retry, "call_llm_with_structured_retry", capture)
    agent = agent_class(_NoProvider())
    with pytest.raises(Captured) as caught:
        agent.run(_context(), planning_contract_context=contract)
    assert expected in captured[1].content
    assert raw not in captured[1].content
    if agent_class is PlannerAgent:
        assert agent.last_prompt_metrics["plan_revision_projection"] == receipts
    else:
        assert authorities == [contract]  # checkpoint hashes bind original source
        facts = progressive.progressive_planner_failure_facts(caught.value)
        assert facts.prompt_metrics["plan_revision_projection"] == receipts


def test_retry_keeps_base_shape_and_all_distinct_feedback_without_duplicate_shape(monkeypatch):
    shape = "Complete exact shape: " + "x" * 4300
    base = [LLMMessage(role="user", content="Requirements\n" + shape + "\n" + "q" * 112000)]
    before = base[0].content
    reminders = retry_shape_reminder(base, shape)
    assert retry_shape_reminder(base, "unseen shape") == "unseen shape"
    client = PromptBudgetClient(_NoProvider(), budget=PROMPT_TRANSPORT_BUDGETS["planner_plan_generation"])
    captured = []

    def capture(llm, messages, **kwargs):
        llm._enforce(messages, kwargs.get("structured_output"))
        captured.append(list(messages))
        return str(len(captured))

    def parse(raw):
        if raw != "4":
            raise ValueError(f"Preserve requirement {raw} in its typed owner.")
        return "accepted synthetic response"

    monkeypatch.setattr(structured_retry, "authorized_complete", capture)
    assert structured_retry.call_llm_with_structured_retry(
        client, base, parse, max_retries=3, include_failed_response_on_retry=False,
        format_reminder=reminders,
    ) == "accepted synthetic response"
    assert len(captured) == 4
    assert base[0].content == before
    assert all(messages[0].content == before for messages in captured)
    assert shape not in captured[-1][-1].content
    for i in range(1, 4):
        assert f"Preserve requirement {i}" in captured[-1][-1].content


def test_oversize_feedback_still_fails_closed_without_dropping_base(monkeypatch):
    base = [LLMMessage(role="user", content="x" * 119000)]
    client = PromptBudgetClient(_NoProvider(), budget=PROMPT_TRANSPORT_BUDGETS["planner_plan_generation"])
    accepted_requests = []

    def capture(llm, messages, **kwargs):
        assert messages[0].content == base[0].content
        llm._enforce(messages, kwargs.get("structured_output"))
        accepted_requests.append(messages)
        return "{}"

    def reject(raw):
        raise ValueError("Preserve the following constraints: " + "q" * 2000)

    monkeypatch.setattr(structured_retry, "authorized_complete", capture)
    with pytest.raises(PromptTransportBudgetError):
        structured_retry.call_llm_with_structured_retry(
            client, base, reject, max_retries=3, include_failed_response_on_retry=False,
        )
    assert len(accepted_requests) == 1

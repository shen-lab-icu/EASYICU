"""Sealed-plan prompt compression must preserve JSON types and retry space."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from easyicu.research_agent.agents import progressive_planner as progressive
from easyicu.research_agent.agents.planner import PlannerAgent
from easyicu.research_agent.canonical_json import (
    canonical_json_bytes, canonical_sha256, sha256_bytes,
)
from easyicu.research_agent.planning.prompt_projection import (
    apply_plan_delta, planner_prompt_byte_limit, project_plan_revision_prompt,
    retry_shape_reminder,
)
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.providers import structured_retry
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.prompt_budget import (
    PromptBudgetClient, PROMPT_TRANSPORT_BUDGETS, PromptTransportBudgetError,
)
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from .progressive_planner_fixtures import _context


from .family_spec_fixtures import (
    ALLOWED_CITATIONS as _FAMILY_CITATIONS,
    DIRECT_COMPARATORS as _FAMILY_COMPARATORS,
    _descriptive_context as _family_context,
    _descriptive_payload as _family_payload,
    _request as _family_request,
)
from .progressive_planner_fixtures import (
    _foundation_payload, _materialization_payloads, _outline_payload,
)


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


def test_later_progressive_stages_keep_global_plan_and_exact_matching_step():
    payload = _plan()
    payload["display_labels"] = {"age": "Patient age"}
    payload["steps"].append(
        AnalysisStep(
            step_id="functional_form_age",
            intent="Refit age with restricted cubic splines.",
            method="restricted_cubic_spline_sensitivity",
            planned_analysis_role="sensitivity",
            expected_outputs=["table:functional_form_age_sensitivity"],
            sensitivity_spec_ids=["age_rcs"],
        ).model_dump(mode="json")
    )
    raw = json.dumps(payload, ensure_ascii=False)
    contract = "- source_plan_json: " + raw

    foundation, foundation_receipts = project_plan_revision_prompt(
        contract, stage="foundation"
    )
    foundation_projection = json.loads(foundation.split(": ", 1)[1])
    assert "source_plan_foundation_projection_json" in foundation
    assert foundation_projection["plan_globals"]["display_labels"] == {
        "age": "Patient age"
    }
    assert foundation_projection["selected_source_steps"] == []
    assert [
        row["step_id"] for row in foundation_projection["source_step_roster"]
    ] == ["describe", "functional_form_age"]
    assert foundation_receipts[0]["status"] == "stage_projected"

    step, step_receipts = project_plan_revision_prompt(
        contract, stage="step", step_id="functional_form_age"
    )
    step_projection = json.loads(step.split(": ", 1)[1])
    assert "source_plan_step_projection_json" in step
    assert step_projection["source_plan_sha256"] == sha256_bytes(
        canonical_json_bytes(payload)
    )
    assert step_projection["selected_source_steps"] == [payload["steps"][1]]
    assert step_receipts[0]["step_id"] == "functional_form_age"


def test_stage_projection_retains_unrecognized_source_verbatim():
    source = '- source_plan_json: {"unknown_science_requirement":"keep me"}'
    projected, receipts = project_plan_revision_prompt(
        source, stage="step", step_id="anything"
    )
    assert projected == source
    assert receipts[0]["status"] == "retained"


def _no_provider(*, strict: bool = False) -> ScriptedMockLLMClient:
    client = ScriptedMockLLMClient(
        [AssertionError("Real transport forbidden")], repeat_last=True,
    )
    client.supports_strict_json_schema = strict
    return client


@pytest.mark.parametrize("limit", [100, 40000, 41000])
def test_preflight_and_transport_agree_including_schema_and_configured_limit(limit):
    client = PromptBudgetClient(_no_provider(), budget=PROMPT_TRANSPORT_BUDGETS[
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
        progressive.ProgressivePlannerAgent(_no_provider()).run(_context())


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
    agent = agent_class(_no_provider())
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
    client = PromptBudgetClient(_no_provider(), budget=PROMPT_TRANSPORT_BUDGETS["planner_plan_generation"])
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
    client = PromptBudgetClient(_no_provider(), budget=PROMPT_TRANSPORT_BUDGETS["planner_plan_generation"])
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


def _large_source_plan():
    steps = [
        AnalysisStep(
            step_id=f"{index:02d}_step",
            intent=("保留本步骤的每一项既定要求与敏感性分析。" * 30) + str(index),
            method=f"method_{index}",
            expected_outputs=[f"table:result_{index}", f"statistic:estimate_{index}"],
        )
        for index in range(24)
    ]
    return AnalysisPlan(
        research_question="保留每一项结局、来源与敏感性分析。",
        display_labels={"event=0": "事件已发生", "event=1": "事件未发生"},
        steps=steps,
    ).model_dump(mode="json")


def test_outline_budget_prefers_the_full_compact_plan_when_it_fits():
    payload = _large_source_plan()
    raw = json.dumps(payload, ensure_ascii=False)
    text = "- source_plan_json: " + raw

    _, receipts = project_plan_revision_prompt(text, byte_budget=len(raw.encode()))

    assert receipts[0]["status"] == "compacted"
    assert receipts[0]["reason"] == "typed_defaults_exact_roundtrip"


def test_outline_budget_uses_a_semantically_complete_bounded_view():
    payload = _large_source_plan()
    text = "- source_plan_json: " + json.dumps(payload, ensure_ascii=False)
    compact_bytes = canonical_json_bytes(
        AnalysisPlan.model_validate(payload).model_dump(mode="json", exclude_defaults=True)
    )
    assert len(compact_bytes) > 4096

    projected, receipts = project_plan_revision_prompt(text, byte_budget=4096)

    receipt = receipts[0]
    assert receipt["status"] == "budget_exceeded"
    assert receipt["reason"] == "declared_requirements_exceed_byte_budget"
    assert receipt["byte_budget"] == 4096
    assert receipt["over_budget_bytes"] == len(compact_bytes) - 4096
    assert receipt["projected_bytes"] == len(compact_bytes)
    assert receipt["projected_bytes"] < receipt["source_bytes"]
    assert "- source_plan_outline_projection_json" not in projected
    body = json.loads(projected.split(": ", 1)[1])
    assert body["research_question"] == payload["research_question"]
    assert body["display_labels"] == payload["display_labels"]
    assert [step["step_id"] for step in body["steps"]] == [
        step["step_id"] for step in payload["steps"]
    ]
    for step, source_step in zip(body["steps"], payload["steps"]):
        assert step["intent"] == source_step["intent"]
        assert step["method"] == source_step["method"]
        assert step["expected_outputs"] == source_step["expected_outputs"]


def test_outline_budget_never_drops_plan_globals_even_below_their_size():
    payload = _large_source_plan()
    text = "- source_plan_json: " + json.dumps(payload, ensure_ascii=False)

    projected, receipts = project_plan_revision_prompt(text, byte_budget=1)

    assert receipts[0]["status"] == "budget_exceeded"
    body = json.loads(projected.split(": ", 1)[1])
    compact_payload = AnalysisPlan.model_validate(payload).model_dump(
        mode="json", exclude_defaults=True,
    )
    assert body == compact_payload
    assert len(body["steps"]) == len(payload["steps"])
    assert all(
        step["intent"] == source["intent"]
        for step, source in zip(body["steps"], payload["steps"])
    )


class _OutcomeCaptured(Exception):
    pass


def _run_progressive(monkeypatch, limit_tokens, contract):
    captured: list[list] = []

    def capture(llm, messages, **kwargs):
        captured.append(list(messages))
        raise _OutcomeCaptured()

    monkeypatch.setattr(progressive, "call_llm_with_structured_retry", capture)
    monkeypatch.setattr(structured_retry, "call_llm_with_structured_retry", capture)

    client = _no_provider()
    client.budget = PROMPT_TRANSPORT_BUDGETS[
        "planner_plan_generation"
    ].with_limit_tokens(limit_tokens)
    agent = progressive.ProgressivePlannerAgent(client)
    with pytest.raises(_OutcomeCaptured):
        agent.run(_context(), planning_contract_context=contract)
    return captured[0], planner_prompt_byte_limit(client)


def _run_legacy(monkeypatch, limit_tokens, contract):
    captured: list[list] = []

    def capture(llm, messages, **kwargs):
        captured.append(list(messages))
        raise _OutcomeCaptured()

    monkeypatch.setattr(structured_retry, "call_llm_with_structured_retry", capture)

    client = _no_provider()
    client.budget = PROMPT_TRANSPORT_BUDGETS[
        "planner_plan_generation"
    ].with_limit_tokens(limit_tokens)
    agent = PlannerAgent(client)
    with pytest.raises(_OutcomeCaptured):
        agent.run(_context(), planning_contract_context=contract)
    return captured[0], planner_prompt_byte_limit(client)


def _request_bytes(messages):
    return sum(len(message.content.encode("utf-8")) for message in messages)


def _assert_compact_plan_reaches_request(messages, payload):
    content = messages[1].content
    assert "- source_plan_json: " in content
    assert "- source_plan_outline_projection_json" not in content
    line = next(
        line for line in content.splitlines()
        if line.startswith("- source_plan_json: ")
    )
    plan = json.loads(line.split(": ", 1)[1])
    assert plan["research_question"] == payload["research_question"]
    assert [step["step_id"] for step in plan["steps"]] == [
        step["step_id"] for step in payload["steps"]
    ]
    assert all(
        step["intent"] == source["intent"]
        for step, source in zip(plan["steps"], payload["steps"])
    )


def test_progressive_compact_plan_that_fits_is_never_replaced_by_a_larger_view(
    monkeypatch,
):
    """Regression for the 52,986/53,187-byte probe failure (Codex item 3).

    The old allocator measured a probe that already contained a larger bounded
    view, then rejected the request that the compact plan could send. The new
    allocation renders the semantic compact request once; exactly-fitting
    requests must reach the provider instead of being swapped for a bigger one.
    """

    payload = AnalysisPlan(
        research_question="Compare groups",
        steps=[AnalysisStep(step_id="summary", intent="Summarize")],
    ).model_dump(mode="json")
    contract = "- source_plan_json: " + json.dumps(payload)

    full_messages, _ = _run_progressive(monkeypatch, 200_000, contract)
    compact_total = _request_bytes(full_messages)

    limit_tokens = compact_total // 3 + 2
    messages, byte_limit = _run_progressive(monkeypatch, limit_tokens, contract)

    assert byte_limit >= compact_total
    assert _request_bytes(messages) == compact_total <= byte_limit
    _assert_compact_plan_reaches_request(messages, payload)


def test_progressive_oversized_compact_plan_fails_closed_naming_the_plan_block(
    monkeypatch,
):
    """No lossy view exists; an irreducible request fails with the block named."""

    payload = _large_source_plan()
    contract = "- source_plan_json: " + json.dumps(payload, ensure_ascii=False)
    base_messages, _ = _run_progressive(monkeypatch, 40_000, "")
    base_bytes = _request_bytes(base_messages)
    limit_tokens = (base_bytes + 30_000) // 3

    captured: list[list] = []

    def capture(llm, messages, **kwargs):
        captured.append(list(messages))
        raise _OutcomeCaptured()

    monkeypatch.setattr(progressive, "call_llm_with_structured_retry", capture)
    monkeypatch.setattr(structured_retry, "call_llm_with_structured_retry", capture)
    client = _no_provider()
    client.budget = PROMPT_TRANSPORT_BUDGETS[
        "planner_plan_generation"
    ].with_limit_tokens(limit_tokens)
    with pytest.raises(ProgressivePlanCompileError) as caught:
        progressive.ProgressivePlannerAgent(client).run(
            _context(), planning_contract_context=contract
        )

    assert caught.value.code == "progressive_prompt_budget_exceeded"
    assert "source-plan block=" in str(caught.value)
    assert not captured
    metrics = caught.value.details["metrics"]
    assert metrics["request_bytes"] > metrics["byte_limit"]


def _replan_contract(*payloads) -> str:
    """Mirror research_plan_revision stacking: one header + full plan per failure."""

    sections = []
    for payload in payloads:
        rendered = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        sections.append(
            "\n".join(
                [
                    "DIGEST-BOUND FAILED EXECUTION REPLAN (host-derived):",
                    "- source_plan_sha256: " + canonical_sha256(payload),
                    "- source_plan_json: " + rendered,
                ]
            )
        )
    return "\n".join(sections)


def _plan_variant(payload, **step_edits):
    variant = deepcopy(payload)
    for step_id, edits in step_edits.items():
        step = next(s for s in variant["steps"] if s["step_id"] == step_id)
        step.update(edits)
    return variant


def _delta_lines(projected):
    return [
        line for line in projected.splitlines()
        if line.startswith("- source_plan_delta_json: ")
    ]


def test_chained_source_plans_emit_host_restorable_deltas():
    base = _large_source_plan()
    v2 = _plan_variant(
        base,
        **{
            "00_step": {
                "intent": "00_step 修订后的完整意图，逐项保留所有输入。",
                "inputs": ["albumin", "lactate", "death_flag"],
            }
        },
    )
    v3 = _plan_variant(
        v2,
        **{"01_step": {"method": "revised_method", "planned_analysis_role": "sensitivity"}},
    )
    projected, receipts = project_plan_revision_prompt(
        _replan_contract(base, v2, v3)
    )

    assert [receipt["status"] for receipt in receipts] == [
        "compacted",
        "delta",
        "delta",
    ]
    compact_views = {
        index + 1: receipt
        for index, receipt in enumerate(receipts)
        if receipt["status"] == "compacted"
    }
    assert len(compact_views) == 1

    lines = projected.splitlines()
    views: dict[int, object] = {}
    for index, line in enumerate(lines, 1):
        if line.startswith("- source_plan_json: "):
            views[index] = json.loads(line.split(": ", 1)[1])
        elif line.startswith("- source_plan_delta_json: "):
            delta = json.loads(line.split(": ", 1)[1])
            assert delta["base_line"] in views
            views[index] = apply_plan_delta(views[delta["base_line"]], delta)

    source_lines = [
        index
        for index, line in enumerate(
            _replan_contract(base, v2, v3).splitlines(), 1
        )
        if line.startswith("- source_plan_json: ")
    ]
    for line_no, source in zip(source_lines, (base, v2, v3)):
        compact = AnalysisPlan.model_validate(source).model_dump(
            mode="json", exclude_defaults=True
        )
        assert canonical_json_bytes(views[line_no]) == canonical_json_bytes(compact)
        restored = AnalysisPlan.model_validate(views[line_no]).model_dump(
            mode="json"
        )
        assert canonical_json_bytes(restored) == canonical_json_bytes(source)


def test_delta_chain_preserves_every_declared_requirement():
    base = _large_source_plan()
    v2 = _plan_variant(
        base,
        **{
            "02_step": {
                "inputs": ["sofa_total", "table:01_cohort"],
                "intent": "修订意图：保留全部既定输入与时间窗。",
            }
        },
    )
    projected, _ = project_plan_revision_prompt(_replan_contract(base, v2))
    lines = projected.splitlines()
    base_line = next(
        index
        for index, line in enumerate(lines, 1)
        if line.startswith("- source_plan_json: ")
    )
    delta_line = next(
        index
        for index, line in enumerate(lines, 1)
        if line.startswith("- source_plan_delta_json: ")
    )
    base_view = json.loads(lines[base_line - 1].split(": ", 1)[1])
    delta = json.loads(lines[delta_line - 1].split(": ", 1)[1])
    restored = apply_plan_delta(base_view, delta)

    for step, source_step in zip(restored["steps"], v2["steps"]):
        assert step["intent"] == source_step["intent"]
        # compact views elide typed defaults, so absent means the default.
        assert step.get("inputs", []) == source_step.get("inputs", [])
        assert step["expected_outputs"] == source_step["expected_outputs"]
    step2 = next(s for s in restored["steps"] if s["step_id"] == "02_step")
    assert step2["inputs"] == ["sofa_total", "table:01_cohort"]
    assert delta["source_plan_sha256"] == canonical_sha256(v2)
    assert delta["base_plan_sha256"] == canonical_sha256(base)


def test_delta_is_never_emitted_when_not_smaller_than_the_complete_view():
    base = _large_source_plan()
    divergent = AnalysisPlan(
        research_question="A wholly different question " + "改" * 400,
        steps=[
            AnalysisStep(
                step_id=f"new_{index:02d}",
                intent="完全不同的新意图" * 20 + str(index),
                method=f"other_{index}",
            )
            for index in range(24)
        ],
    ).model_dump(mode="json")
    projected, receipts = project_plan_revision_prompt(
        _replan_contract(base, divergent)
    )
    assert receipts[1]["status"] == "compacted"
    assert not _delta_lines(projected)


def test_identical_replan_blocks_emit_an_empty_ops_delta():
    base = _large_source_plan()
    projected, receipts = project_plan_revision_prompt(
        _replan_contract(base, deepcopy(base))
    )
    assert receipts[1]["status"] == "delta"
    assert receipts[1]["delta_op_count"] == 0
    delta = json.loads(_delta_lines(projected)[0].split(": ", 1)[1])
    assert delta["ops"] == []


def test_delta_falls_back_to_the_complete_view_on_path_unsafe_keys():
    base = _plan()
    changed = deepcopy(base)
    changed["display_labels"] = {"event.0": "not path safe"}
    projected, receipts = project_plan_revision_prompt(
        _replan_contract(base, changed)
    )
    assert receipts[1]["status"] == "compacted"
    assert not _delta_lines(projected)


def test_delta_chain_applies_to_staged_projections_too():
    base = _large_source_plan()
    v2 = _plan_variant(base, **{"00_step": {"method": "revised_method"}})
    projected, receipts = project_plan_revision_prompt(
        _replan_contract(base, v2), stage="foundation"
    )
    assert [receipt["status"] for receipt in receipts] == [
        "stage_projected",
        "delta",
    ]
    delta = json.loads(_delta_lines(projected)[0].split(": ", 1)[1])
    assert delta["stage"] == "foundation"


def test_step_delta_uses_the_actual_base_after_a_stage_view_falls_back():
    base = AnalysisPlan(
        research_question="Q", steps=[AnalysisStep(step_id="a", intent="A")],
    )
    revised = base.model_copy(deep=True)
    revised.steps[0].intent = "Revised A"
    contract = "\n".join(
        "- source_plan_json: " + json.dumps(
            plan.model_dump(mode="json"), separators=(",", ":"),
        )
        for plan in (base, revised)
    )
    projected, receipts = project_plan_revision_prompt(
        contract, stage="step", step_id="a",
    )
    lines = projected.splitlines()
    assert receipts[0]["status"] == "retained"
    assert lines[0].startswith("- source_plan_json: ")
    first = json.loads(lines[0].split(": ", 1)[1])
    second = json.loads(lines[1].split(": ", 1)[1])
    restored = (
        apply_plan_delta(first, second)
        if lines[1].startswith("- source_plan_delta_json: ") else second
    )
    assert canonical_json_bytes(restored) == canonical_json_bytes(
        revised.model_dump(mode="json")
    )


def test_delta_cost_includes_the_longer_emitted_prefix():
    plan = AnalysisPlan(
        research_question="Q",
        steps=[AnalysisStep(step_id="a", intent="x" * 511)],
    )
    line = "- source_plan_json: " + json.dumps(plan.model_dump(mode="json"))
    compact, _ = project_plan_revision_prompt(line)
    projected, _ = project_plan_revision_prompt(line + "\n" + line)
    # The former delta body saved one byte but its prefix added six.
    assert len(projected.encode()) <= len((compact + "\n" + compact).encode())


def _restored_plan_views(projected: str) -> list[dict]:
    views: dict[str, dict] = {}
    restored: list[dict] = []
    for line in projected.splitlines():
        if line.startswith("- source_plan_json: "):
            view = json.loads(line.split(": ", 1)[1])
            digest = canonical_sha256(
                AnalysisPlan.model_validate(view).model_dump(mode="json")
            )
            views[digest] = view
            restored.append(view)
        elif line.startswith("- source_plan_delta_json: "):
            delta = json.loads(line.split(": ", 1)[1])
            view = apply_plan_delta(views[delta["base_plan_sha256"]], delta)
            full = AnalysisPlan.model_validate(view).model_dump(mode="json")
            assert canonical_sha256(full) == delta["source_plan_sha256"]
            views[delta["source_plan_sha256"]] = view
            restored.append(view)
    return restored


def _stacked_replan_contract() -> tuple[str, dict, dict]:
    base = _large_source_plan()
    v2 = _plan_variant(
        base,
        **{
            f"{index:02d}_step": {
                "intent": f"修订后的第 {index} 步意图与敏感性要求。" * 12,
            }
            for index in (2, 5)
        },
    )
    return _replan_contract(base, v2), base, v2


def test_superseded_replans_are_kept_by_default_and_elided_on_request():
    contract, base, v2 = _stacked_replan_contract()

    default_projected, default_receipts = project_plan_revision_prompt(contract)
    assert "- superseded_source_plan_sha256" not in default_projected
    assert all(r["status"] != "superseded_elided" for r in default_receipts)
    default_views = _restored_plan_views(default_projected)
    assert [
        canonical_json_bytes(AnalysisPlan.model_validate(view).model_dump(mode="json"))
        for view in default_views
    ] == [canonical_json_bytes(base), canonical_json_bytes(v2)]

    elided, receipts = project_plan_revision_prompt(
        contract, elide_superseded_replans=True,
    )
    pointer = next(
        line for line in elided.splitlines()
        if line.startswith("- superseded_source_plan_sha256: ")
    )
    assert canonical_sha256(base) in pointer
    elision = receipts[0]
    assert elision["status"] == "superseded_elided"
    assert elision["reason"] == "ancestor_replan_superseded"
    assert elision["source_plan_sha256"] == canonical_sha256(base)
    assert elision["superseded_by_plan_sha256"] == canonical_sha256(v2)
    # Only the current source plan remains in the sent text; the ancestor is
    # accounted for by digest.
    assert len(_restored_plan_views(elided)) == 1
    assert len(elided.encode()) < len(default_projected.encode())


def test_superseded_elision_fails_closed_on_incomplete_ancestry():
    contract, base, _v2 = _stacked_replan_contract()
    mutated = contract.replace(
        "- source_plan_sha256: " + canonical_sha256(base) + "\n", "", 1
    )

    projected, receipts = project_plan_revision_prompt(
        mutated, elide_superseded_replans=True,
    )

    assert "- superseded_source_plan_sha256" not in projected
    assert all(r["status"] != "superseded_elided" for r in receipts)


def test_superseded_elision_fails_closed_on_duplicate_digests():
    contract, base, v2 = _stacked_replan_contract()
    mutated = contract.replace(
        "- source_plan_sha256: " + canonical_sha256(v2),
        "- source_plan_sha256: " + canonical_sha256(base),
    )

    projected, receipts = project_plan_revision_prompt(
        mutated, elide_superseded_replans=True,
    )

    assert "- superseded_source_plan_sha256" not in projected
    assert all(r["status"] != "superseded_elided" for r in receipts)


def test_legacy_planner_elides_ancestry_only_under_budget_pressure(monkeypatch):
    contract, _base, _v2 = _stacked_replan_contract()
    full_messages, _ = _run_legacy(monkeypatch, 200_000, contract)
    full_bytes = _request_bytes(full_messages)
    assert "- superseded_source_plan_sha256" not in full_messages[1].content

    limit_tokens = (full_bytes - 200) // 3
    messages, byte_limit = _run_legacy(monkeypatch, limit_tokens, contract)

    assert "- superseded_source_plan_sha256" in messages[1].content
    assert _request_bytes(messages) <= byte_limit


def test_progressive_planner_elides_ancestry_only_under_budget_pressure(monkeypatch):
    contract, _base, _v2 = _stacked_replan_contract()
    full_messages, _ = _run_progressive(monkeypatch, 200_000, contract)
    full_bytes = _request_bytes(full_messages)
    assert "- superseded_source_plan_sha256" not in full_messages[1].content

    limit_tokens = (full_bytes - 200) // 3
    messages, byte_limit = _run_progressive(monkeypatch, limit_tokens, contract)

    assert "- superseded_source_plan_sha256" in messages[1].content
    assert _request_bytes(messages) <= byte_limit


def _chained_replan_contract() -> tuple[str, list[dict]]:
    base = _large_source_plan()
    v2 = _plan_variant(
        base,
        **{
            f"{index:02d}_step": {
                "intent": f"第 {index} 步的修订意图。" * 12,
                "inputs": ["lact_max", "death"],
            }
            for index in (0, 3, 6)
        },
    )
    v3 = _plan_variant(
        v2,
        **{
            f"{index:02d}_step": {"method": f"revised_{index}"}
            for index in (10, 14)
        },
    )
    return _replan_contract(base, v2, v3), [base, v2, v3]


def _run_progressive_request(monkeypatch, limit_tokens, contract, strict):
    """Capture the assembled request and its transport bytes incl. schema."""

    captured: dict = {}

    def capture(llm, messages, **kwargs):
        schema = kwargs.get("structured_output")
        captured["messages"] = list(messages)
        captured["bytes"] = sum(
            len(message.content.encode("utf-8")) for message in messages
        ) + (schema.payload_bytes if schema else 0)
        raise _OutcomeCaptured()

    monkeypatch.setattr(progressive, "call_llm_with_structured_retry", capture)
    monkeypatch.setattr(structured_retry, "call_llm_with_structured_retry", capture)

    client = _no_provider(strict=strict)
    client.budget = PROMPT_TRANSPORT_BUDGETS[
        "planner_plan_generation"
    ].with_limit_tokens(limit_tokens)
    agent = progressive.ProgressivePlannerAgent(client)
    with pytest.raises(_OutcomeCaptured):
        agent.run(_context(), planning_contract_context=contract)
    return captured["messages"], captured["bytes"], planner_prompt_byte_limit(client)


@pytest.mark.parametrize("strict", [False, True])
def test_chained_replan_replay_reaches_the_stub_under_both_schema_paths(
    monkeypatch, strict
):
    """Saved-input replay shape: three stacked full plans would exceed the cap.

    With delta chaining the assembled request must reach the provider stub,
    both with and without the strict-schema block Codex's saved fixture adds.
    """

    contract, (base, v2, v3) = _chained_replan_contract()

    messages, request_bytes, _ = _run_progressive_request(
        monkeypatch, 200_000, contract, strict
    )
    content = messages[1].content
    deltas = _delta_lines(content)
    assert len(deltas) == 2
    assert content.count("- source_plan_json: ") == 1

    # What the same request would cost without chaining, measured on the real
    # emitted lines: replace each delta by its own complete compact view.
    # Deltas bind their base by source_plan_sha256, which survives embedding.
    undelta_bytes = request_bytes
    views: dict[str, object] = {}
    restored_views: list[object] = []
    for line in content.splitlines():
        if line.startswith("- source_plan_json: "):
            view = json.loads(line.split(": ", 1)[1])
            full = AnalysisPlan.model_validate(view).model_dump(mode="json")
            views[canonical_sha256(full)] = view
            restored_views.append(view)
        elif line.startswith("- source_plan_delta_json: "):
            delta = json.loads(line.split(": ", 1)[1])
            restored = apply_plan_delta(views[delta["base_plan_sha256"]], delta)
            full = AnalysisPlan.model_validate(restored).model_dump(mode="json")
            assert canonical_sha256(full) == delta["source_plan_sha256"]
            views[delta["source_plan_sha256"]] = restored
            restored_views.append(restored)
            undelta_bytes += (
                len("- source_plan_json: ")
                + len(canonical_json_bytes(restored))
                - len(line.encode())
            )

    assert request_bytes < undelta_bytes
    # The replay must still fit when the envelope sits between the delta cost
    # and the un-chained cost -- the delta is what makes this request sendable.
    limit_tokens = request_bytes // 3 + 2
    tight_messages, tight_bytes, tight_limit = _run_progressive_request(
        monkeypatch, limit_tokens, contract, strict
    )
    assert tight_bytes == request_bytes <= tight_limit
    assert undelta_bytes > tight_limit

    # Every declared requirement of every revision is still recoverable from
    # the request itself: base plan verbatim + restorable deltas.
    for view, source in zip(restored_views, (base, v2, v3)):
        for step, source_step in zip(view["steps"], source["steps"]):
            assert step["intent"] == source_step["intent"]
            assert step["expected_outputs"] == source_step["expected_outputs"]


def test_legacy_planner_sends_the_compact_plan_when_it_fits(monkeypatch):
    """The non-progressive planner allocates the same exact measured request."""

    payload = AnalysisPlan(
        research_question="Compare groups",
        steps=[AnalysisStep(step_id="summary", intent="Summarize")],
    ).model_dump(mode="json")
    contract = "- source_plan_json: " + json.dumps(payload)

    full_messages, _ = _run_legacy(monkeypatch, 200_000, contract)
    compact_total = _request_bytes(full_messages)

    limit_tokens = compact_total // 3 + 2
    messages, byte_limit = _run_legacy(monkeypatch, limit_tokens, contract)

    assert byte_limit >= compact_total
    assert _request_bytes(messages) == compact_total <= byte_limit
    _assert_compact_plan_reaches_request(messages, payload)


def test_legacy_planner_oversized_plan_fails_closed_naming_the_plan_block(
    monkeypatch,
):
    from easyicu.research_agent.agents.planner import PlannerPromptBudgetError

    payload = _large_source_plan()
    contract = "- source_plan_json: " + json.dumps(payload, ensure_ascii=False)
    base_messages, _ = _run_legacy(monkeypatch, 40_000, "")
    base_bytes = _request_bytes(base_messages)
    limit_tokens = (base_bytes + 30_000) // 3

    monkeypatch.setattr(
        structured_retry, "call_llm_with_structured_retry",
        lambda *args, **kwargs: pytest.fail("oversized request must not reach transport"),
    )
    client = _no_provider()
    client.budget = PROMPT_TRANSPORT_BUDGETS[
        "planner_plan_generation"
    ].with_limit_tokens(limit_tokens)
    with pytest.raises(PlannerPromptBudgetError) as caught:
        PlannerAgent(client).run(_context(), planning_contract_context=contract)

    assert "The source-plan block is" in str(caught.value)
    assert "keeps every declared requirement" in str(caught.value)


@pytest.mark.parametrize("invalid", ["digest_mismatch", "duplicate_plan", "duplicate_digest", "invalid_schema"])
def test_ancestor_elision_rejects_ambiguous_or_unverified_sections(invalid):
    contract, base, _ = _stacked_replan_contract()
    digest_line = "- source_plan_sha256: " + canonical_sha256(base)
    plan_line = next(line for line in contract.splitlines() if line.startswith("- source_plan_json: "))
    if invalid == "digest_mismatch":
        contract = contract.replace(digest_line, "- source_plan_sha256: " + "f" * 64, 1)
    elif invalid == "duplicate_plan":
        contract = contract.replace(plan_line, plan_line + "\n" + plan_line, 1)
    elif invalid == "duplicate_digest":
        contract = contract.replace(digest_line, digest_line + "\n" + digest_line, 1)
    else:
        contract = contract.replace(plan_line, '- source_plan_json: {"unexpected": true}', 1)
    _, receipts = project_plan_revision_prompt(contract, elide_superseded_replans=True)
    assert not any(row["status"] == "superseded_elided" for row in receipts)


def test_ancestor_elision_preserves_intervening_requirements_and_seeds():
    contract, _, _ = _stacked_replan_contract()
    marker = "DIGEST-BOUND FAILED EXECUTION REPLAN (host-derived):"
    offset = contract.index(marker, contract.index(marker) + len(marker))
    requirement = "USER REVIEW: Retain the prespecified 48-hour landmark sensitivity.\n"
    seed = '- candidate_plan_seed_json: {"review_requirement": "retain landmark"}\n'
    contract = contract[:offset] + requirement + seed + contract[offset:]
    projected, receipts = project_plan_revision_prompt(contract, elide_superseded_replans=True)
    assert any(row["status"] == "superseded_elided" for row in receipts)
    assert requirement.strip() in projected
    assert "retain landmark" in projected


# Initial-request advisory projections belong to this prompt-projection owner.
# The progressive owner test module is already at its 7,393-line ratchet.


def _scripted_projection_run(
    *, family=False, strict=False, limit=200_000, know_how="", contract="", context=None,
    with_article_context=True,
):
    context = context or (_family_context() if family else _context())
    # Clinical definitions are authority even when optional guidance dominates.
    context = context.model_copy(
        update={
            "variables": [
                item.model_copy(
                    update={
                        "description": f"Definition of {item.name}: sealed clinical meaning."
                    }
                )
                for item in context.variables
            ]
        }
    )
    if family and know_how:
        descriptor = context.variable("score_first")
        descriptor.description += " " + know_how
        descriptor.clinical_caveats = ["The baseline score has an advisory caveat."]
    options = {"planning_contract_context": contract}
    if know_how and not family:
        claims = [
            {
                "claim_id": "guidance",
                "field": "method",
                "text": "Consider the estimand.",
                "citation_ids": ["clinical_source"],
                "evidence_scope": "methods",
            },
            *[
                {
                    "claim_id": f"extended_{index}",
                    "field": "method",
                    "text": know_how[index : index + 600],
                    "citation_ids": ["clinical_source"],
                    "evidence_scope": "methods",
                }
                for index in range(0, len(know_how), 600)
            ],
            {
                "claim_id": "stop",
                "field": "stop_condition",
                "text": "Stop if the source cannot support the requested endpoint.",
                "citation_ids": ["clinical_source"],
                "evidence_scope": "methods",
            },
        ]
        options["know_how_context"] = (
            "# Retrieved Research Know-How (STRUCTURED ADVISORY DATA)\n"
            + json.dumps(
                {
                    "schema_version": "easyicu.research_know_how_prompt/2",
                    "cards": [{"card_id": "clinical_protocol", "claims": claims}],
                }
            )
        )
        options["allowed_know_how_decisions"] = {
            "clinical_protocol": {
                "version": "1.0.0",
                "file_sha256": "a" * 64,
                "claims": {item["claim_id"]: ("clinical_source",) for item in claims},
            }
        }
    if family:
        options.update(
            planner_strategy=progressive.FAMILY_SPEC_STRATEGY,
            allowed_literature_citation_keys=_FAMILY_CITATIONS,
            direct_comparator_literature_keys=_FAMILY_COMPARATORS,
            comparison_literature_keys=_FAMILY_COMPARATORS,
            required_primary_cohort_selection_mode="all_input_rows",
            enforce_article_contract=with_article_context,
            article_contract_context=context if with_article_context else None,
        )
        responses = [
            _family_payload(
                _family_request(context, cohort_mode="all_input_rows"),
                baseline_variables=["age", "sex", "score_first"],
            )
        ]
    else:
        foundation = _foundation_payload()
        if know_how:
            foundation["foundation"]["know_how_decisions"] = [
                {
                    "card_id": "clinical_protocol",
                    "card_version": "1.0.0",
                    "card_sha256": "a" * 64,
                    "claim_id": "stop",
                    "disposition": "adopted",
                    "reason_code": "fits_estimand",
                    "rationale": "This guidance fits the declared clinical estimand.",
                    "citation_ids": ["clinical_source"],
                }
            ]
        responses = [_outline_payload(), foundation, *_materialization_payloads()]
    client = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    client.supports_strict_json_schema = strict
    client.budget = PROMPT_TRANSPORT_BUDGETS[
        "planner_plan_generation"
    ].with_limit_tokens(limit)
    agent = progressive.ProgressivePlannerAgent(client)
    result = agent.run_attempt(context, **options)
    return client, result, context


@pytest.mark.parametrize(
    "family, with_article_context",
    [(False, False), (True, True), (True, False)],
    ids=["progressive", "family_spec", "family_spec_without_article_context"],
)
@pytest.mark.parametrize("strict", [False, True], ids=["text_shape", "strict_schema"])
def test_advisory_overflow_compiles_with_receipts_and_all_authority(
    family, strict, with_article_context
):
    guidance = "Optional protocol discussion about uncertainty. " * 100
    requirement = "Reviewed requirement: preserve the requested outcomes and every sensitivity analysis."
    full_client, full, _ = _scripted_projection_run(
        family=family,
        strict=strict,
        know_how=guidance,
        contract=requirement,
        with_article_context=with_article_context,
    )
    limit = int(full.facts.prompt_metrics["total_bytes"] / 1.02) // 3
    client, result, context = _scripted_projection_run(
        family=family,
        strict=strict,
        know_how=guidance,
        contract=requirement,
        limit=limit,
        with_article_context=with_article_context,
    )
    metrics = result.facts.prompt_metrics
    assert 1.019 <= full.facts.prompt_metrics["total_bytes"] / (limit * 3) <= 1.022
    assert result.output.steps
    assert metrics["total_bytes"] <= limit * 3
    assert metrics["prompt_projection_receipts"]
    receipt = next(
        item
        for item in metrics["prompt_projection_receipts"]
        if item["block"] == ("variables" if family else "know_how")
    )
    for key in ("source_sha256", "projected_sha256"):
        assert len(receipt[key]) == 64
    assert receipt["source_bytes"] > receipt["projected_bytes"]
    assert receipt["reason"] == (
        "optional_variable_descriptions_omitted"
        if family
        else "advisory_guidance_projected"
    )
    assert metrics["projections_tried"]
    prompt = "\n".join(message.content for message in client.calls[0][0])
    assert requirement in prompt
    assert context.research_question in prompt
    for name in (context.primary_exposure, context.target_outcome):
        assert name in prompt
        assert context.variable(name).description in prompt
    if family:
        for name in _family_request(context, cohort_mode="all_input_rows").exact_roster:
            assert name in prompt
            assert context.variable(name).description in prompt
        descriptor = context.variable("score_first")
        assert '"score_first"' in prompt
        for caveat in descriptor.clinical_caveats:
            assert caveat in full_client.calls[0][0][1].content
            assert caveat not in prompt
    else:
        for name in ("age_years", "sex_code"):
            assert name in prompt
            assert context.variable(name).description in prompt
    assert guidance[:200] in full_client.calls[0][0][1].content
    assert guidance[:200] not in prompt


@pytest.mark.parametrize("family", [False, True], ids=["progressive", "family_spec"])
@pytest.mark.parametrize("strict", [False, True], ids=["text_shape", "strict_schema"])
def test_authority_overflow_refuses_with_complete_private_safe_breakdown(
    family, strict
):
    authority = "Reviewed requirement: keep all requested endpoints. " * 3000
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _scripted_projection_run(
            family=family, strict=strict, contract=authority, limit=40_000
        )
    error = caught.value
    assert error.code == "progressive_prompt_budget_exceeded"
    assert error.path == "planner_request"
    metrics = error.details["metrics"]
    blocks = {
        key.removeprefix("block_").removesuffix("_bytes"): value
        for key, value in metrics.items()
        if key.startswith("block_") and key.endswith("_bytes")
    }
    assert set(blocks) == {
        "system_guide",
        "article_question_context",
        "variables",
        "action_rows",
        "design_evidence_cards",
        "know_how",
        "planning_contract",
        "response_shape",
        "schema",
    }
    assert all(isinstance(value, int) and value >= 0 for value in blocks.values())
    assert sum(blocks.values()) == metrics["request_bytes"]
    assert (
        metrics["request_bytes"] == metrics["message_bytes"] + metrics["schema_bytes"]
    )
    assert blocks["schema"] == metrics["schema_bytes"]
    assert metrics["request_bytes"] > metrics["byte_limit"] == 120_000
    assert any(
        key.startswith("projection_") and key.endswith("_attempts") for key in metrics
    )
    assert "Reviewed requirement" not in json.dumps(metrics)


@pytest.mark.parametrize("family", [False, True], ids=["progressive", "family_spec"])
def test_budget_projection_builds_identical_bytes_and_receipts(family):
    options = dict(family=family, know_how="Advisory clinical discussion. " * 100)
    _, full, _ = _scripted_projection_run(**options)
    options["limit"] = int(full.facts.prompt_metrics["total_bytes"] / 1.02) // 3
    left_client, left, _ = _scripted_projection_run(**options)
    right_client, right, _ = _scripted_projection_run(**options)
    assert left_client.calls[0][0] == right_client.calls[0][0]
    for key in ("block_bytes", "projections_tried", "prompt_projection_receipts"):
        assert left.facts.prompt_metrics[key] == right.facts.prompt_metrics[key]


def test_advisory_ladder_stops_at_first_fit_and_receipts_bind_exact_blocks():
    from easyicu.research_agent.planning.prompt_projection import (
        PlannerPromptBlock,
        project_planner_request,
    )

    blocks = [
        PlannerPromptBlock(
            "article_question_context",
            "Required endpoint and adjustment: preserve verbatim.",
        ),
        PlannerPromptBlock(
            "variables",
            "Optional definition: " + "long " * 100,
            (("optional_variable_descriptions_omitted", "Named clinical variable"),),
        ),
        PlannerPromptBlock(
            "know_how",
            "Optional guidance. " * 100,
            (("advisory_guidance_projected", "Required stop claim."),),
        ),
    ]
    messages = [
        LLMMessage(role="system", content="System authority"),
        LLMMessage(role="user", content="\n\n".join(block.text for block in blocks)),
    ]
    initial = _request_bytes(messages)
    result = project_planner_request(messages, blocks, byte_limit=initial - 1)
    assert result.metrics["request_bytes"] <= initial - 1
    assert (
        result.metrics["projections_tried"][-1]
        == "optional_variable_descriptions_omitted"
    )
    assert result.blocks[2].text == blocks[2].text
    (receipt,) = result.metrics["prompt_projection_receipts"]
    assert receipt == {
        "block": "variables",
        "reason": "optional_variable_descriptions_omitted",
        "source_bytes": len(blocks[1].text.encode()),
        "source_sha256": sha256_bytes(blocks[1].text.encode()),
        "projected_bytes": len(result.blocks[1].text.encode()),
        "projected_sha256": sha256_bytes(result.blocks[1].text.encode()),
    }
    assert sum(result.metrics["block_bytes"].values()) == _request_bytes(
        result.messages
    )
    assert messages[1].content == "\n\n".join(block.text for block in blocks)


def test_know_how_projection_preserves_mandatory_and_unknown_claims_and_bindings():
    from easyicu.research_agent.planning.prompt_projection import (
        project_know_how_guidance,
    )

    claims = [
        {
            "claim_id": "short",
            "field": "method",
            "text": "Consider this estimand.",
            "citation_ids": ["source_a"],
        },
        {
            "claim_id": "long",
            "field": "method",
            "text": "Optional discussion. " * 30,
            "citation_ids": ["source_b"],
        },
        {
            "claim_id": "stop",
            "field": "stop_condition",
            "text": "Refuse an unsupported endpoint.",
            "citation_ids": ["source_a"],
        },
        {
            "claim_id": "confirm",
            "field": "requires_confirmation",
            "text": "Confirm the exposure definition.",
            "citation_ids": ["source_a"],
        },
        {
            "claim_id": "future",
            "field": "new_claim_kind",
            "text": "Retain an unknown claim intact.",
            "citation_ids": ["source_b"],
        },
    ]
    card = {
        "card_id": "clinical_guidance",
        "version": "1.0.0",
        "file_sha256": "a" * 64,
        "claims": claims,
        "required_concepts": ["clinical_marker"],
        "citations": {"source_a": "a", "source_b": "b"},
    }
    payload = {"schema_version": "easyicu.research_know_how_prompt/2", "cards": [card]}
    source = "Structured advisory data:\n" + json.dumps(payload)
    projected = project_know_how_guidance(source)
    result = json.loads(projected.split("\n", 1)[1])["cards"][0]
    assert result["claims"] == sorted(claims[2:], key=canonical_json_bytes)
    for field in (
        "card_id",
        "version",
        "file_sha256",
        "required_concepts",
        "citations",
    ):
        assert result[field] == card[field]
    assert result["omitted_claim_count"] == 2
    reverse = {key: value for key, value in reversed(list(payload.items()))}
    reverse["cards"] = [{key: value for key, value in reversed(list(card.items()))}]
    assert (
        project_know_how_guidance("Structured advisory data:\n" + json.dumps(reverse))
        == projected
    )
    assert (
        project_know_how_guidance("Unknown guidance format: preserve everything.")
        == "Unknown guidance format: preserve everything."
    )
    assert (
        project_know_how_guidance('Structured data:\n{"schema_version":"unknown"}')
        == 'Structured data:\n{"schema_version":"unknown"}'
    )


def test_literature_projection_keeps_every_reviewed_fact_and_shape_rule():
    from easyicu.research_agent.planning.prompt_projection import (
        PlannerPromptBlocks,
        project_planner_request,
    )
    from easyicu.research_agent.planning.literature_design_authority import (
        LITERATURE_DESIGN_DIMENSIONS,
        LiteratureDesignEvidenceCard,
        render_literature_design_cards_for_prompt,
    )

    card = LiteratureDesignEvidenceCard(
        citation_key="synthetic_comparator",
        evidence_role="direct_comparator",
        access_mode="user_supplied_fulltext",
        full_text_locator="synthetic-fulltext",
        full_text_sha256="a" * 64,
        supplement_status="published_unreviewed",
        reviewed_at="2026-10-01T00:00:00Z",
        evidence=[
            {
                "dimension": dimension,
                "source_backed_summary": f"The reviewed {dimension} choice is recorded here.",
            }
            for dimension in LITERATURE_DESIGN_DIMENSIONS
        ],
    )
    blocks = PlannerPromptBlocks()
    blocks.append("Keep the declared output shape.")
    blocks.append(
        render_literature_design_cards_for_prompt([card]), block="design_evidence_cards"
    )
    messages = [
        LLMMessage(role="system", content="Authority"),
        LLMMessage(role="user", content=blocks.render()),
    ]
    result = project_planner_request(
        messages, blocks.blocks, byte_limit=_request_bytes(messages) - 1
    )
    prompt = result.messages[1].content
    for evidence in card.evidence:
        assert evidence.source_backed_summary in prompt
    assert "supplement=published_unreviewed" in prompt
    assert "synthetic_comparator" in prompt
    assert "Keep the declared output shape." in prompt
    assert (
        result.metrics["prompt_projection_receipts"][-1]["reason"]
        == "literature_provenance_projected"
    )
    assert result.metrics["request_bytes"] < _request_bytes(messages)


def test_variable_projection_protects_question_plan_and_exact_covariates_verbatim():
    from easyicu.research_agent.planning.prompt_projection import (
        project_planner_request,
        PlannerPromptBlock,
    )
    from easyicu.research_agent.schema import ConceptDescriptor, VariableRole

    context = _context()
    extra = [
        ConceptDescriptor(
            name="creatinine_first",
            description="First serum creatinine before the index time.",
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="bilirubin_peak",
            description="Peak bilirubin during the declared baseline window.",
            role=VariableRole.LAB,
            dtype="float64",
        ),
        ConceptDescriptor(
            name="albumin_first",
            description="Optional albumin description. " * 100,
            role=VariableRole.LAB,
            dtype="float64",
        ),
    ]
    context = context.model_copy(
        update={
            "research_question": context.research_question
            + " Include creatinine_first as a required target.",
            "variables": [
                item.model_copy(
                    update={"description": f"Prespecified definition of {item.name}."}
                )
                for item in context.variables
            ]
            + extra,
        }
    )
    contract = (
        "Reviewed plan requirement: include bilirubin_peak in the required analysis."
    )
    types, names, _, actions = progressive.ProgressivePlannerAgent._request_authorities(
        context
    )
    names = (*names, *(item.name for item in extra if item.name not in names))
    blocks: list[PlannerPromptBlock] = []
    prompt = progressive.ProgressivePlannerAgent._user_prompt(
        context,
        analysis_types=types,
        variables=names,
        action_rows=actions,
        planning_contract_context=contract,
        prompt_blocks=blocks,
    )
    messages = [
        LLMMessage(role="system", content="system"),
        LLMMessage(role="user", content=prompt),
    ]
    projected = project_planner_request(
        messages, blocks, byte_limit=_request_bytes(messages) - 1
    )
    text = projected.messages[1].content
    assert projected.metrics["request_bytes"] < _request_bytes(messages)
    for variable in [*context.variables[:-1]]:
        assert variable.name in text
        assert variable.description in text
    assert extra[-1].name in text  # The candidate identity is never projected away.
    assert extra[-1].description not in text
    assert contract in text
    assert context.research_question in text


def test_exact_family_covariate_definitions_are_authority_even_without_article_context():
    from .family_spec_fixtures import (
        _context as landmark_context,
        _request as landmark_request,
    )
    from easyicu.research_agent.agents.family_spec_planner import (
        family_spec_user_prompt,
    )
    from easyicu.research_agent.planning.prompt_projection import (
        PlannerPromptBlock,
        project_planner_request,
    )

    request = landmark_request(landmark_context())
    optional_candidate = next(
        item for item in landmark_request(landmark_context(exact=False)).adjustment_candidates
        if item.name == "severity_score_24h"
    )
    assert optional_candidate.selectable and optional_candidate.name not in request.exact_roster
    request = request.model_copy(
        update={"adjustment_candidates": [*request.adjustment_candidates, optional_candidate]}
    )
    definitions = {
        name: f"Declared clinical definition of {name}."
        for name in [*request.exact_roster, optional_candidate.name]
    }
    blocks: list[PlannerPromptBlock] = []
    prompt = family_spec_user_prompt(
        request,
        variable_descriptions=definitions,
        prompt_blocks=blocks,
        variable_description_authority=frozenset(),
    )
    messages = [
        LLMMessage(role="system", content="system"),
        LLMMessage(role="user", content=prompt),
    ]
    initial_bytes = _request_bytes(messages)
    projected = project_planner_request(messages, blocks, byte_limit=initial_bytes - 1)
    projected_prompt = projected.messages[1].content
    assert projected.metrics["request_bytes"] < initial_bytes
    assert definitions[optional_candidate.name] in prompt
    assert definitions[optional_candidate.name] not in projected_prompt
    assert f'"{optional_candidate.name}"' in projected_prompt
    for name in request.exact_roster:
        assert definitions[name] in projected_prompt


@pytest.mark.parametrize("strict", [False, True], ids=["text_shape", "strict_schema"])
def test_family_overflow_without_article_context_preserves_named_descriptions(strict):
    from easyicu.research_agent.schema import VariableRole

    from .family_spec_fixtures import _context as landmark_context, _spec_payload

    context = landmark_context()
    question_named = "stage_source_flag"
    plan_named = "window_row_count"
    optional = "severity_score_24h"
    optional_caveats = ["Advisory score caveat one.", "Advisory score caveat two."]
    required_caveats = ["Prespecified comorbidity caveat one.", "Prespecified comorbidity caveat two."]
    extra_caveat = "Third caveat would overflow the legacy budget. " * 200
    for descriptor in context.variables:
        descriptor.description = f"Prespecified clinical definition of {descriptor.name}."
    # The exact model roster leaves this untimed score in the measurement audit.
    optional_descriptor = context.variable(optional)
    optional_descriptor.role = VariableRole.OTHER
    optional_descriptor.description = "Optional audit score definition. " * 100
    optional_descriptor.clinical_caveats = optional_caveats
    required_descriptor = context.variable("comorbidity_index")
    required_descriptor.clinical_caveats = required_caveats
    context = context.model_copy(
        update={
            "research_question": context.research_question
            + f" Include {question_named} in the measurement audit.",
        }
    )
    contract = f"Reviewed planning instruction: retain the {plan_named} definition."
    request = _family_request(context)
    assert request.exact_roster
    assert {optional, question_named, plan_named} <= set(request.measurement_audit_columns)
    response = json.dumps(_spec_payload(request))

    def run(context, limit):
        client = ScriptedMockLLMClient([response])
        client.supports_strict_json_schema = strict
        client.budget = PROMPT_TRANSPORT_BUDGETS["planner_plan_generation"].with_limit_tokens(limit)
        result = progressive.ProgressivePlannerAgent(client).run_attempt(
            context,
            planner_strategy=progressive.FAMILY_SPEC_STRATEGY,
            allowed_literature_citation_keys=_FAMILY_CITATIONS,
            direct_comparator_literature_keys=_FAMILY_COMPARATORS,
            comparison_literature_keys=_FAMILY_COMPARATORS,
            required_primary_cohort_selection_mode="predicate_filtered",
            article_contract_context=None,
            planning_contract_context=contract,
        )
        assert len(client.calls) == 1
        assert result.output.steps
        return client.calls[0][0][1].content, result.facts.prompt_metrics

    baseline_prompt, baseline = run(context, 200_000)
    required_descriptor.clinical_caveats = [*required_caveats, extra_caveat]
    limit = (baseline["request_bytes"] - 1) // 3
    prompt, metrics = run(context, limit)
    assert metrics["planner_strategy"] == progressive.FAMILY_SPEC_STRATEGY
    assert metrics["source_request_bytes"] == baseline["request_bytes"]
    assert metrics["request_bytes"] <= metrics["byte_limit"] < baseline["request_bytes"]
    (receipt,) = metrics["prompt_projection_receipts"]
    assert receipt["reason"] == "optional_variable_descriptions_omitted"
    assert f'"{optional}"' in prompt
    for text in [optional_descriptor.description, *optional_caveats]:
        assert text in baseline_prompt
        assert text not in prompt
    for name in [*request.exact_roster, question_named, plan_named]:
        assert context.variable(name).description in prompt
    for caveat in required_caveats:
        assert caveat in prompt
    assert extra_caveat not in prompt


def test_progressive_prompt_bytes_do_not_depend_on_mapping_insertion_order():
    context = _context()
    rationales = {
        "age_years": "Prespecified age adjustment.",
        "sex_code": "Prespecified sex adjustment.",
    }
    first = context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"covariate_rationales": rationales}
            )
        }
    )
    second = context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={
                    "covariate_rationales": dict(reversed(list(rationales.items())))
                }
            )
        }
    )
    types, names, _, actions = progressive.ProgressivePlannerAgent._request_authorities(
        context
    )
    reversed_actions = [dict(reversed(list(row.items()))) for row in actions]
    left = progressive.ProgressivePlannerAgent._user_prompt(
        first, analysis_types=types, variables=names, action_rows=actions
    )
    right = progressive.ProgressivePlannerAgent._user_prompt(
        second, analysis_types=types, variables=names, action_rows=reversed_actions
    )
    assert left.encode() == right.encode()


@pytest.mark.parametrize(
    "raw",
    ["[]", '{"schema_version":"easyicu.research_know_how_prompt/2","cards":[null]}'],
)
def test_malformed_advisory_data_is_kept_intact(raw):
    from easyicu.research_agent.planning.prompt_projection import (
        project_know_how_guidance,
    )

    text = "Structured advisory context:\n" + raw
    assert project_know_how_guidance(text) == text


def test_optional_only_know_how_keeps_a_complete_claim_for_each_card_disposition():
    from easyicu.research_agent.planning.prompt_projection import (
        project_know_how_guidance,
    )

    claims = [
        {
            "claim_id": "long",
            "field": "method",
            "text": "Optional scientific guidance. " * 20,
            "citation_ids": ["source_a"],
        },
        {
            "claim_id": "short",
            "field": "method",
            "text": "A complete advisory design claim.",
            "citation_ids": ["source_b"],
        },
    ]
    payload = {
        "schema_version": "easyicu.research_know_how_prompt/2",
        "cards": [{"card_id": "clinical_guidance", "claims": claims}],
    }
    projected = project_know_how_guidance("Advisory context:\n" + json.dumps(payload))
    (card,) = json.loads(projected.split("\n", 1)[1])["cards"]
    assert card["card_id"] == "clinical_guidance"
    assert card["claims"] == [claims[1]]
    assert card["omitted_claim_count"] == 1

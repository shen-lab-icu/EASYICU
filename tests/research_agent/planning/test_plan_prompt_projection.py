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

    client = _NoProvider()
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

    client = _NoProvider()
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
    client = _NoProvider()
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
    monkeypatch.setattr(_NoProvider, "supports_strict_json_schema", strict)

    client = _NoProvider()
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
    client = _NoProvider()
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

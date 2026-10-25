"""Live replies bind host identities; stored artifacts retain their own bindings."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.family_spec_planner import (
    FAMILY_SPEC_GUIDE,
    family_spec_response_shape,
    family_spec_structured_output_request,
    family_spec_user_prompt,
    parse_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec import (
    FamilySpecError,
    build_descriptive_skeleton,
    validate_family_plan_spec,
)
from easyicu.research_agent.planning.family_spec.contract import spec_from_mapping
from easyicu.research_agent.agents.planner import PlannerAgent, _build_planner_user_prompt
from easyicu.research_agent.agents.progressive_prompt_contracts import foundation_shape_contract
from easyicu.research_agent.agents.plan_payload import planner_structured_output_request
from easyicu.research_agent.agents.progressive_payload import (
    parse_progressive_foundation_materialization,
    parse_progressive_step_materialization,
    progressive_foundation_structured_output_request,
    progressive_step_materialization_request,
)
from easyicu.research_agent.planning.preplan_know_how import verify_know_how_decisions
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import (
    ProgressiveFoundationMaterialization,
    ProgressivePlanCompileError,
    ProgressivePlanOutline,
    ProgressivePlanSkeleton,
)
from easyicu.research_agent.planning.progressive_resume import validate_progressive_materialization_coordinate
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
    call_llm_with_structured_retry,
    safe_structured_attempt_metadata,
)

from .family_spec_fixtures import _descriptive_context, _descriptive_payload, _request, _run
from .progressive_planner_fixtures import (
    _context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
    _payload,
)


@pytest.mark.parametrize("echo", [None, "mangled", "0" * 64, 123, {"digest": "wrong"}])
def test_live_family_reply_binds_the_sent_request(echo) -> None:
    context = _descriptive_context()
    request = _request(context, cohort_mode="all_input_rows")
    payload = _descriptive_payload(request, baseline_variables=[])
    if echo is None:
        payload.pop("request_sha256")
    else:
        payload["request_sha256"] = echo
    raw = json.dumps(payload)
    spec = parse_family_plan_spec(raw, request)
    assert spec.request_sha256 == request.request_sha256
    validate_family_plan_spec(spec, request)
    llm, result = _run(context, [raw], required_primary_cohort_selection_mode="all_input_rows")
    assert len(llm.calls) == 1
    assert result.output.steps
    assert result.facts.prompt_metrics["family_spec_request_sha256"] == request.request_sha256


def test_family_reply_contract_does_not_ask_for_the_host_digest() -> None:
    request = _request(_descriptive_context(), cohort_mode="all_input_rows")
    schema = json.loads(family_spec_structured_output_request(request).schema_json)
    assert "request_sha256" not in schema["properties"]
    assert "request_sha256" not in schema["required"]
    assert "request_sha256" not in FAMILY_SPEC_GUIDE
    assert "request_sha256" not in family_spec_response_shape(request)
    prompt = family_spec_user_prompt(request, variable_descriptions={})
    assert "request_sha256" not in prompt
    assert request.request_sha256 in prompt  # Host metadata keeps cassette matching strict.


def test_saved_family_spec_keeps_its_digest_and_is_rejected_by_each_boundary() -> None:
    request = _request(_descriptive_context(), cohort_mode="all_input_rows")
    payload = _descriptive_payload(request, baseline_variables=[])
    payload["request_sha256"] = "0" * 64
    saved = spec_from_mapping(json.loads(json.dumps(payload)))
    assert saved.request_sha256 == "0" * 64
    for validate in (validate_family_plan_spec, lambda spec, req: build_descriptive_skeleton(req, spec)):
        with pytest.raises(FamilySpecError, match="family_spec_request_digest_mismatch"):
            validate(saved, request)
    for echo in (None, "mangled"):
        payload["request_sha256"] = echo
        with pytest.raises(ValidationError):
            spec_from_mapping(payload)


@pytest.mark.parametrize("echo", [None, "mangled", "0" * 64])
def test_live_progressive_replies_bind_host_digests_and_step_id(echo) -> None:
    foundation = _foundation_payload()
    step = _materialization_payloads()[0]
    outline_step = ProgressivePlanOutline.model_validate(_outline_payload()).steps[0]
    for payload, field in ((foundation, "outline_sha256"), (step, "outline_step_sha256")):
        if echo is None:
            payload.pop(field)
        else:
            payload[field] = echo
    if echo is None:
        step["step"].pop("step_id")
    else:
        step["step"]["step_id"] = echo
    parsed_foundation = parse_progressive_foundation_materialization(
        json.dumps(foundation), host_cohort=None, outline_sha256="a" * 64
    )
    parsed_step = parse_progressive_step_materialization(
        json.dumps(step), outline_step=outline_step, outline_step_sha256="b" * 64
    )
    assert parsed_foundation.outline_sha256 == "a" * 64
    assert parsed_step.outline_step_sha256 == "b" * 64
    assert parsed_step.step.step_id == outline_step.step_id
    if echo != "0" * 64:
        with pytest.raises(ValidationError):
            parse_progressive_step_materialization(json.dumps(step))


def _know_how():
    authority = {
        "card_alpha": {
            "version": "2.0.0",
            "file_sha256": "a" * 64,
            "claims": {"claim_alpha": ("citation_alpha",)},
        }
    }
    decision = {
        "card_id": "card_alpha", "claim_id": "claim_alpha",
        "card_version": "1.0.0", "card_sha256": "0" * 64,
        "disposition": "adopted", "reason_code": "fits_estimand",
        "rationale": "The selected claim applies to this study's stated estimand.",
        "citation_ids": ["citation_alpha"],
    }
    return authority, decision


@pytest.mark.parametrize("route", ["foundation", "planner"])
@pytest.mark.parametrize("echo", [None, "mangled", "0" * 64])
def test_live_know_how_decisions_bind_retrieved_card_identity(route, echo) -> None:
    authority, decision = _know_how()
    if echo is None:
        decision.pop("card_sha256")
        decision.pop("card_version")
    else:
        decision["card_sha256"] = echo
    if route == "foundation":
        payload = _foundation_payload()
        payload["foundation"]["know_how_decisions"] = [decision]
        parsed = parse_progressive_foundation_materialization(
            json.dumps(payload), host_cohort=None, outline_sha256="b" * 64,
            allowed_know_how_decisions=authority,
        ).foundation
    else:
        planner = PlannerAgent.__new__(PlannerAgent)
        plan, _ = compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(_payload()), context=_context()
        )
        payload = {**plan.model_dump(mode="json"), "know_how_decisions": [decision]}
        parsed = planner._parse(
            json.dumps(payload), _context(),
            allowed_know_how_decisions=authority,
        )
    verify_know_how_decisions(parsed.know_how_decisions, authority)
    bound = parsed.know_how_decisions[0]
    assert bound.card_sha256 == "a" * 64
    assert bound.card_version == "2.0.0"
    assert bound.claim_id == decision["claim_id"]
    assert bound.rationale == decision["rationale"]


def test_saved_know_how_identity_stays_strict() -> None:
    authority, decision = _know_how()
    payload = _foundation_payload()
    payload["foundation"]["know_how_decisions"] = [decision]
    for parsed in (
        ProgressiveFoundationMaterialization.model_validate(payload),
        parse_progressive_foundation_materialization(json.dumps(payload), host_cohort=None),
    ):
        with pytest.raises(ValueError, match="changed version/SHA"):
            verify_know_how_decisions(parsed.foundation.know_how_decisions, authority)


@pytest.mark.parametrize("field", ["outline_step_sha256", "step_id"])
def test_saved_step_binding_is_not_replaced(field) -> None:
    payload = _materialization_payloads()[0]
    digest = payload["outline_step_sha256"]
    if field == "outline_step_sha256":
        payload[field] = "0" * 64
    else:
        payload["step"][field] = "another_step"
    stored = parse_progressive_step_materialization(json.dumps(payload))
    outline_step = ProgressivePlanOutline.model_validate(_outline_payload()).steps[0]
    with pytest.raises(ValueError, match="progressive_step_(outline_digest|materialization)_mismatch"):
        validate_progressive_materialization_coordinate(
            stored, outline_step=outline_step, outline_step_sha256=digest, step_index=0,
        )


@pytest.mark.parametrize("field,value", [
    ("module_id", "table_one"), ("module_id", None),
    ("scientific_action_id", "descriptive.table_one"), ("scientific_action_id", None),
])
def test_live_step_uses_the_module_and_action_selected_in_its_outline(field, value) -> None:
    payload = _materialization_payloads()[4]
    outline_step = ProgressivePlanOutline.model_validate(_outline_payload()).steps[4]
    payload["step"][field] = value
    parsed = parse_progressive_step_materialization(
        json.dumps(payload), outline_step=outline_step,
        outline_step_sha256=payload["outline_step_sha256"],
    )
    assert parsed.step.module_id == outline_step.module_id
    assert parsed.step.scientific_action_id == outline_step.scientific_action_id
    assert parsed.step.raw_inputs == payload["step"]["raw_inputs"]
    assert parsed.step.primary_exposure == payload["step"]["primary_exposure"]


@pytest.mark.parametrize("field,value", [("claim_id", "unknown_claim"), ("citation_ids", ["unretrieved"])])
def test_live_know_how_binding_does_not_replace_selected_references(field, value) -> None:
    authority, decision = _know_how()
    decision[field] = value
    payload = _foundation_payload()
    payload["foundation"]["know_how_decisions"] = [decision]
    parsed = parse_progressive_foundation_materialization(
        json.dumps(payload), host_cohort=None, allowed_know_how_decisions=authority,
    )
    with pytest.raises(ValueError, match="unretrieved card|unknown claim|changed citation binding"):
        verify_know_how_decisions(parsed.foundation.know_how_decisions, authority)


def test_live_schemas_request_choices_without_host_identity_metadata() -> None:
    authority, _ = _know_how()
    prompt = _build_planner_user_prompt(_context(), know_how_context="A synthetic retrieved claim.")
    assert "`card_version`" not in prompt
    assert "`card_sha256`" not in prompt
    shape = foundation_shape_contract(outline_sha256="a" * 64, host_cohort=None)
    for field in ("outline_sha256", "card_version", "card_sha256"):
        assert field not in shape
    foundation = json.loads(progressive_foundation_structured_output_request(
        outline_sha256="a" * 64, variable_names=["age_years"],
        allowed_know_how_decisions=authority,
    ).schema_json)
    planner = json.loads(planner_structured_output_request().schema_json)
    assert "outline_sha256" not in foundation["properties"]
    decision = foundation["$defs"]["ProgressiveKnowHowDecision"]["anyOf"][0]["properties"]
    for properties in (decision, planner["$defs"]["KnowHowDecision"]["properties"]):
        assert "card_sha256" not in properties
        assert "card_version" not in properties
        assert {"card_id", "claim_id", "citation_ids", "disposition", "rationale"} <= properties.keys()
    outline_step = ProgressivePlanOutline.model_validate(_outline_payload()).steps[0]
    step = json.loads(progressive_step_materialization_request(
        outline_step=outline_step, outline_step_sha256="b" * 64,
        variable_names=["age_years"], scientific_action_ids=[],
    ).schema_json)
    assert "outline_step_sha256" not in step["properties"]
    assert "step_id" not in step["$defs"]["ProgressiveSkeletonStep"]["properties"]


def _several_cards():
    authority, template = _know_how()
    cards, decisions = {}, []
    for index, suffix in enumerate(("alpha", "beta", "gamma"), start=1):
        card_id = f"card_{suffix}"
        cards[card_id] = {
            **authority["card_alpha"],
            "version": f"{index}.2.3",
            "file_sha256": "abc"[index - 1] * 64,
        }
        decisions.append({**template, "card_id": card_id})
    return cards, decisions


@pytest.mark.parametrize("echo", ["omitted", "matched", None, "v1", "99.0.0", 123, {"version": "wrong"}])
def test_each_retrieved_card_binds_its_own_version_before_validation(echo) -> None:
    authority, decisions = _several_cards()
    for decision in decisions:
        if echo == "omitted":
            decision.pop("card_version")
            decision.pop("card_sha256")
        elif echo == "matched":
            card = authority[decision["card_id"]]
            decision.update(card_version=card["version"], card_sha256=card["file_sha256"])
        else:
            decision.update(card_version=echo, card_sha256=echo)
    payload = _foundation_payload()
    payload["foundation"]["know_how_decisions"] = decisions
    parsed = parse_progressive_foundation_materialization(
        json.dumps(payload), host_cohort=None, outline_sha256="b" * 64,
        allowed_know_how_decisions=authority,
    ).foundation
    verify_know_how_decisions(parsed.know_how_decisions, authority)
    for bound, original in zip(parsed.know_how_decisions, decisions, strict=True):
        assert bound.card_version == authority[bound.card_id]["version"]
        assert bound.card_sha256 == authority[bound.card_id]["file_sha256"]
        for field in ("card_id", "claim_id", "citation_ids", "disposition", "reason_code", "rationale"):
            assert getattr(bound, field) == original[field]


@pytest.mark.parametrize("retrieval", ["selected", "empty", "disabled"])
@pytest.mark.parametrize("echo", ["omitted", "v1", "1.0.0"])
def test_unretrieved_cards_are_refused_before_their_identity_echoes(retrieval, echo) -> None:
    authority, decisions = _several_cards()
    unknown = {**decisions[-1], "card_id": "unretrieved_synthetic_card"}
    if echo == "omitted":
        unknown.pop("card_version")
        unknown.pop("card_sha256")
    else:
        unknown["card_version"] = echo
    payload = _foundation_payload()
    payload["foundation"]["know_how_decisions"] = [*decisions[:2], unknown]
    with pytest.raises(ProgressivePlanCompileError) as caught:
        parse_progressive_foundation_materialization(
            json.dumps(payload), host_cohort=None, outline_sha256="b" * 64,
            allowed_know_how_decisions=authority if retrieval == "selected" else {} if retrieval == "empty" else None,
        )
    error = caught.value
    assert error.reason_code == "progressive_know_how_card_unretrieved"
    assert error.path == f"know_how_decisions.{2 if retrieval == 'selected' else 0}.card_id"
    assert "unretrieved_synthetic_card" not in json.dumps(error.easyicu_safe_diagnostic)
    if retrieval != "selected":
        assert "know_how_decisions=[]" in str(error)


@pytest.mark.parametrize("exhausted", [False, True])
def test_foundation_retry_repairs_the_card_selection_instead_of_inventing_metadata(exhausted) -> None:
    _, decisions = _several_cards()
    bad = _foundation_payload()
    for decision in decisions:
        decision.pop("card_version")
        decision.pop("card_sha256")
    bad["foundation"]["know_how_decisions"] = decisions
    corrected = _foundation_payload()
    llm = ScriptedMockLLMClient([json.dumps(bad), json.dumps(bad if exhausted else corrected)])

    def run():
        return call_llm_with_structured_retry(
            llm, [LLMMessage(role="user", content="State rules for the synthetic study.")],
            parser=lambda raw: parse_progressive_foundation_materialization(
                raw, host_cohort=None, outline_sha256="b" * 64,
            ),
            role="progressive_planner_foundation", max_retries=1,
            include_failed_response_on_retry=False,
        )

    if exhausted:
        from easyicu.webserver.agent_pipeline_runs import _safe_pipeline_typed_failure

        with pytest.raises(StructuredResponseFailure) as caught:
            run()
        rows = safe_structured_attempt_metadata(caught.value.attempts)
        assert len(rows) == 2
        assert all(row["reason_code"] == "progressive_know_how_card_unretrieved" for row in rows)
        diagnostic = _safe_pipeline_typed_failure(caught.value)
        assert diagnostic["reason_code"] == "progressive_know_how_card_unretrieved"
        assert diagnostic["path"] == "know_how_decisions.0.card_id"
        assert "card_alpha" not in json.dumps(diagnostic)
    else:
        assert run().foundation.know_how_decisions == []
    assert len(llm.calls) == 2
    retry = "\n".join(message.content for message in llm.calls[1][0])
    assert "know_how_decisions=[]" in retry
    assert "Field required" not in retry


@pytest.mark.parametrize("field", ["card_version", "card_sha256"])
def test_saved_card_identity_must_still_be_well_formed(field) -> None:
    _, decision = _know_how()
    decision[field] = "malformed"
    payload = _foundation_payload()
    payload["foundation"]["know_how_decisions"] = [decision]
    with pytest.raises(ValidationError):
        parse_progressive_foundation_materialization(json.dumps(payload), host_cohort=None)

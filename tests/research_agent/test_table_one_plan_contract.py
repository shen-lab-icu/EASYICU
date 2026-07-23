from __future__ import annotations

import hashlib
import hmac
import json

import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.core import (
    CoderAgent,
    PlannerAgent,
    ReplannerAgent,
    _normalise_plan_payload,
)
from easyicu.research_agent.authority.table_one_binding import (
    bind_table_one_execution_spec,
    restore_table_one_private_checkpoint,
    table_one_private_code_label_map,
    table_one_execution_spec,
    write_table_one_private_checkpoint,
)
from easyicu.research_agent.authority.plan_authority import normalize_replan_candidate
from easyicu.research_agent.audits.validators import LLMConceptAuditor
from easyicu.research_agent.methods.table_one import build_grouped_table_one
from easyicu.research_agent.providers.mocks import ExternalCaptureMockLLMClient
from easyicu.research_agent.providers.prompts import load_prompt_pack
from easyicu.research_agent.repairs.patch import PATCH_FORMAT
from easyicu.research_agent.repairs.reasons import (
    RepairPromptAuthority,
    RepairReason,
)
from easyicu.research_agent.research_context.prompt_scope import coder_guide_for_step
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Describe the cohort by the Planner-selected group.",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
    )


def _binary_context() -> ResearchContext:
    context = _context()
    context.variables.append(
        ConceptDescriptor(
            name="death",
            dtype="int64",
            observed_domain={
                "n_unique": 2,
                "is_binary": True,
                "min": 0.0,
                "max": 1.0,
            },
        )
    )
    return context


def _private_label_context() -> ResearchContext:
    context = _context()
    context.variables.extend(
        [
            ConceptDescriptor(
                name="sex",
                dtype="object",
                observed_domain={
                    "n_unique": 2,
                    "levels": ["Female", "Male"],
                },
            ),
            ConceptDescriptor(
                name="smoking_status",
                dtype="object",
                observed_domain={
                    "n_unique": 2,
                    "levels": ["NeverSmokerLocal", "EverSmokerLocal"],
                },
            ),
        ]
    )
    return context


def _private_table_one_bound_step() -> tuple[ResearchContext, AnalysisStep]:
    context = _private_label_context()
    step = AnalysisStep.model_validate(
        {
            "step_id": "02_table_one",
            "planned_analysis_role": "auxiliary",
            "intent": "Produce the grouped baseline table.",
            "inputs": ["sex", "smoking_status"],
            "expected_outputs": ["table:table_one"],
            "method": "table_one",
            "table_one_spec": {
                "group_by": "sex",
                "group_levels": [
                    "__easyicu_level_1__",
                    "__easyicu_level_2__",
                ],
                "variables": [
                    {
                        "name": "smoking_status",
                        "variable_kind": "categorical",
                        "summary": "count_percent",
                        "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                        "levels": [
                            "__easyicu_level_1__",
                            "__easyicu_level_2__",
                        ],
                    }
                ],
            },
        }
    )
    assert bind_table_one_execution_spec(step, context) is not None
    return context, step


def _private_table_one_script() -> str:
    return (
        "groups = ['Female', 'Male']\n"
        "smoking_levels = ['NeverSmokerLocal', 'EverSmokerLocal']\n"
        "value = 1\n"
    )


def _captured_prompt_text(client: ExternalCaptureMockLLMClient) -> str:
    return "\n".join(
        message.content for messages, _kwargs in client.calls for message in messages
    )


def _assert_private_labels_absent(text: str) -> None:
    for private_label in (
        "Female",
        "Male",
        "NeverSmokerLocal",
        "EverSmokerLocal",
    ):
        assert private_label not in text


def _repair_authority(*, validator: str) -> RepairPromptAuthority:
    return RepairPromptAuthority.create(
        typed_ticket=[
            {
                "validator": validator,
                "reason": RepairReason.UNBOUND_LOCAL.value,
                "occurrence_count": 1,
                "detail": {"reason": "unbound_local", "line": 3},
            }
        ]
    )


def _step(*, include_spec: bool) -> dict:
    step = {
        "step_id": "02_table_one",
        "planned_analysis_role": "auxiliary",
        "intent": "Produce the grouped baseline table.",
        "inputs": ["arm", "age"],
        "expected_outputs": ["table:table_one"],
        "method": "table_one",
    }
    if include_spec:
        step["table_one_spec"] = {
            "group_by": "arm",
            "group_levels": [0, 1],
            "variables": [
                {
                    "name": "age",
                    "variable_kind": "continuous",
                    "summary": "median_iqr",
                    "test": "mann_whitney_or_kruskal",
                }
            ],
        }
    return step


def _raw(*, include_spec: bool) -> str:
    return json.dumps(
        {
            "research_question": "Describe the cohort.",
            "steps": [_step(include_spec=include_spec)],
            "rationale": "Use the declared grouped table design.",
        }
    )


def test_fresh_planner_table_one_requires_typed_design() -> None:
    planner = PlannerAgent.__new__(PlannerAgent)
    with pytest.raises(ValueError, match="must declare table_one_spec"):
        planner._parse(_raw(include_spec=False), _context())


def test_fresh_planner_table_one_preserves_typed_design() -> None:
    planner = PlannerAgent.__new__(PlannerAgent)
    plan = planner._parse(_raw(include_spec=True), _context())
    assert plan.steps[0].table_one_spec is not None
    assert plan.steps[0].table_one_spec.group_by == "arm"
    assert plan.steps[0].table_one_spec.variables[0].test == ("mann_whitney_or_kruskal")


def test_planner_prompt_lists_exact_table_one_enums_and_rejects_shorthand() -> None:
    prompt = PlannerAgent.request_messages(_context())[1].content

    assert "a binary variable is `categorical`" in prompt
    assert "`chi_square_with_fisher_exact_for_sparse_2x2`" in prompt
    assert "Do not emit shorthand aliases" in prompt
    assert "primary scientific comparison" in prompt
    assert "Auxiliary measurement/source-status flags" in prompt


def test_know_how_prompt_lists_exact_claim_decision_coordinates() -> None:
    prompt = PlannerAgent.request_messages(_context(), know_how_context='{"cards":[]}')[
        1
    ].content

    assert "KNOW-HOW DECISION OUTPUT CONTRACT" in prompt
    assert "`card_sha256`" in prompt
    assert "Do not use a `decision` key" in prompt


def test_fresh_planner_preserves_observed_numeric_level_types() -> None:
    payload = json.loads(_raw(include_spec=True))
    step = payload["steps"][0]
    step["inputs"] = ["death", "age"]
    step["table_one_spec"]["group_by"] = "death"
    step["table_one_spec"]["group_levels"] = ["0", "1"]
    planner = PlannerAgent.__new__(PlannerAgent)

    with pytest.raises(ValueError, match="exact observed scalar types"):
        planner._parse(json.dumps(payload), _binary_context())

    step["table_one_spec"]["group_levels"] = [0, 1]
    parsed = planner._parse(json.dumps(payload), _binary_context())
    assert parsed.steps[0].table_one_spec.group_levels == [0, 1]


def test_private_table_one_levels_use_opaque_tokens_and_bind_locally() -> None:
    context = _private_label_context()
    prompt = PlannerAgent.request_messages(context)[1].content
    assert "Female" not in prompt
    assert "Male" not in prompt
    assert "NeverSmokerLocal" not in prompt
    assert "EverSmokerLocal" not in prompt
    assert "__easyicu_level_1__" in prompt
    assert "__easyicu_level_2__" in prompt

    payload = json.loads(_raw(include_spec=True))
    step = payload["steps"][0]
    step["inputs"] = ["sex", "smoking_status"]
    step["table_one_spec"] = {
        "group_by": "sex",
        "group_levels": ["__easyicu_level_1__", "__easyicu_level_2__"],
        "variables": [
            {
                "name": "smoking_status",
                "variable_kind": "categorical",
                "summary": "count_percent",
                "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                "levels": ["__easyicu_level_1__", "__easyicu_level_2__"],
            }
        ],
    }

    planner = PlannerAgent.__new__(PlannerAgent)
    plan = planner._parse(json.dumps(payload), context)
    spec = plan.steps[0].table_one_spec
    assert spec is not None
    assert spec.group_levels == ["__easyicu_level_1__", "__easyicu_level_2__"]
    assert spec.variables[0].levels == [
        "__easyicu_level_1__",
        "__easyicu_level_2__",
    ]
    execution_spec = table_one_execution_spec(plan.steps[0])
    assert execution_spec is not None
    assert execution_spec.group_levels == ["Female", "Male"]
    assert execution_spec.variables[0].levels == [
        "NeverSmokerLocal",
        "EverSmokerLocal",
    ]
    outbound_plan = plan.model_dump_json()
    for private_label in (
        "Female",
        "Male",
        "NeverSmokerLocal",
        "EverSmokerLocal",
    ):
        assert private_label not in outbound_plan


def test_private_code_tokens_are_host_keyed_and_stable_after_rebinding() -> None:
    context, step = _private_table_one_bound_step()
    first = table_one_private_code_label_map(step)
    restored_step = AnalysisStep.model_validate(step.model_dump(mode="json"))
    assert bind_table_one_execution_spec(restored_step, context) is not None
    second = table_one_private_code_label_map(restored_step)
    fresh_host_context = ResearchContext.model_validate(context.model_dump(mode="json"))
    fresh_host_step = AnalysisStep.model_validate(step.model_dump(mode="json"))
    assert (
        bind_table_one_execution_spec(fresh_host_step, fresh_host_context) is not None
    )
    fresh_host_tokens = table_one_private_code_label_map(fresh_host_step)

    assert first == second
    assert first
    assert first != fresh_host_tokens
    female_token = first[("str", repr("Female"))]
    old_dictionary_token = (
        "__easyicu_table1_label_"
        + hashlib.sha256(
            json.dumps(
                {
                    "step_id": step.step_id,
                    "type": "str",
                    "repr": repr("Female"),
                    "public": "__easyicu_level_1__",
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        + "__"
    )
    assert female_token != old_dictionary_token

    public_plan = AnalysisPlan(
        research_question=context.research_question,
        steps=[AnalysisStep.model_validate(step.model_dump(mode="json"))],
    )
    public_key_guess = hashlib.sha256(
        public_plan.model_dump_json().encode("utf-8")
    ).digest()
    for candidate in ("Female", "Male", "NeverSmokerLocal", "EverSmokerLocal"):
        dictionary_guess = hmac.new(
            public_key_guess,
            candidate.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:16]
        assert dictionary_guess not in female_token


def test_private_code_token_secret_restores_from_private_checkpoint(tmp_path) -> None:
    context, step = _private_table_one_bound_step()
    original_tokens = table_one_private_code_label_map(step)
    plan = AnalysisPlan(
        research_question=context.research_question,
        steps=[step],
    )
    checkpoint = write_table_one_private_checkpoint(run_dir=tmp_path, plan=plan)
    assert checkpoint.stat().st_mode & 0o077 == 0
    assert "Female" not in checkpoint.read_text(encoding="utf-8")

    resumed_context = ResearchContext.model_validate(context.model_dump(mode="json"))
    resumed_plan = AnalysisPlan.model_validate(plan.model_dump(mode="json"))
    restore_table_one_private_checkpoint(
        run_dir=tmp_path,
        plan=resumed_plan,
        context=resumed_context,
    )

    assert table_one_private_code_label_map(resumed_plan.steps[0]) == original_tokens


def test_private_table_one_labels_never_enter_agent_prompts() -> None:
    context = _private_label_context()
    payload = json.loads(_raw(include_spec=True))
    step_payload = payload["steps"][0]
    step_payload["inputs"] = ["sex", "smoking_status"]
    step_payload["table_one_spec"] = {
        "group_by": "sex",
        "group_levels": ["__easyicu_level_1__", "__easyicu_level_2__"],
        "variables": [
            {
                "name": "smoking_status",
                "variable_kind": "categorical",
                "summary": "count_percent",
                "test": "chi_square_with_fisher_exact_for_sparse_2x2",
                "levels": ["__easyicu_level_1__", "__easyicu_level_2__"],
            }
        ],
    }
    planner_llm = ExternalCaptureMockLLMClient([json.dumps(payload)])
    plan = PlannerAgent(planner_llm).run(context)

    revised_payload = plan.model_dump(mode="json")
    revised_payload["revision"] = plan.revision + 1
    revised_payload["steps"][0]["method"] = "descriptive_statistics"
    replanner_llm = ExternalCaptureMockLLMClient([json.dumps(revised_payload)])
    replanner_candidate = ReplannerAgent(replanner_llm).run(
        context=context,
        current_plan=plan,
        probe_summary={"status": "complete", "private_label": "Female"},
        completed_step_records=[
            {
                "step_id": "01_upstream",
                "status": "ok",
                "step_summary": {
                    "group": "Female",
                    "category": "NeverSmokerLocal",
                    "patient_id": 9918273,
                    "subject_id": 771122,
                    "stay_id": 881133,
                    "age": 93,
                    "lactate": 17.25,
                    "individual_values": [93, 17.25],
                    "unknown_nested": {"numeric_secret": 444555},
                },
            }
        ],
    )
    revised = normalize_replan_candidate(
        current_plan=plan,
        candidate_plan=replanner_candidate,
        completed_records=[],
        context=context,
        max_total_steps=0,
        locked_robustness_specs=[],
    ).plan
    assert revised.steps[0].method == "descriptive_statistics"
    local_spec = table_one_execution_spec(revised.steps[0])
    assert local_spec is not None
    assert local_spec.group_levels == ["Female", "Male"]
    table = build_grouped_table_one(
        pd.DataFrame(
            {
                "sex": ["Female", "Female", "Male", "Male"],
                "smoking_status": [
                    "NeverSmokerLocal",
                    "EverSmokerLocal",
                    "NeverSmokerLocal",
                    "EverSmokerLocal",
                ],
            }
        ),
        local_spec,
    )
    assert not table.empty

    patch = json.dumps(
        {
            "format": PATCH_FORMAT,
            "edits": [{"old": "value = 1", "new": "value = 2", "expected_count": 1}],
        }
    )
    coder_llm = ExternalCaptureMockLLMClient(["value = 1\n"])
    repair_llm = ExternalCaptureMockLLMClient([patch])
    coder_agent = CoderAgent(coder_llm, repair_llm=repair_llm)
    coder_agent.run(context=context, step=revised.steps[0])
    authority = RepairPromptAuthority.create(
        typed_ticket=[
            {
                "validator": "mechanical_code_preflight",
                "reason": RepairReason.UNBOUND_LOCAL.value,
                "occurrence_count": 1,
                "detail": {"reason": "unbound_local", "line": 1},
            }
        ]
    )
    coder_agent.repair(
        context=context,
        step=revised.steps[0],
        code="value = 1\n",
        run_log="local diagnostic only",
        repair_authority=authority,
        current_repair_authority=authority,
    )
    assert len(coder_llm.calls) == 1
    assert len(repair_llm.calls) == 1

    concept_llm = ExternalCaptureMockLLMClient(['{"findings":[]}'])
    LLMConceptAuditor(concept_llm).audit(
        context=context,
        step=revised.steps[0],
        script_text=(
            "groups = ['Female', 'Male']\n"
            "levels = ['NeverSmokerLocal', 'EverSmokerLocal']\n"
        ),
    )

    all_messages = [
        message.content
        for client in (
            planner_llm,
            replanner_llm,
            coder_llm,
            repair_llm,
            concept_llm,
        )
        for call, _kwargs in client.calls
        for message in call
    ]
    outbound = "\n".join(all_messages)
    for private_label in (
        "Female",
        "Male",
        "NeverSmokerLocal",
        "EverSmokerLocal",
    ):
        assert private_label not in outbound
    for numeric_phi in (
        "9918273",
        "771122",
        "881133",
        "17.25",
        "444555",
    ):
        assert numeric_phi not in _captured_prompt_text(replanner_llm)


def test_table_one_contract_repair_never_sends_private_labels() -> None:
    context, step = _private_table_one_bound_step()
    response = json.dumps(
        {
            "format": PATCH_FORMAT,
            "edits": [{"old": "value = 1", "new": "value = 2", "expected_count": 1}],
        }
    )
    client = ExternalCaptureMockLLMClient([response])
    authority = _repair_authority(validator="contract_validator")

    repaired = CoderAgent(client).repair(
        context=context,
        step=step,
        code=_private_table_one_script(),
        run_log="Female NeverSmokerLocal must remain local",
        repair_authority=authority,
        current_repair_authority=authority,
        provider_category="contract_repair",
    )

    _assert_private_labels_absent(_captured_prompt_text(client))
    assert "Female" in repaired
    assert "NeverSmokerLocal" in repaired
    assert "value = 2" in repaired


def test_table_one_runtime_repair_never_sends_private_labels() -> None:
    context, step = _private_table_one_bound_step()
    response = json.dumps(
        {
            "format": PATCH_FORMAT,
            "edits": [{"old": "value = 1", "new": "value = 3", "expected_count": 1}],
        }
    )
    client = ExternalCaptureMockLLMClient([response])
    authority = _repair_authority(validator="runtime_executor")

    repaired = CoderAgent(client).repair(
        context=context,
        step=step,
        code=_private_table_one_script(),
        run_log="Male EverSmokerLocal traceback must remain local",
        repair_authority=authority,
        current_repair_authority=authority,
        provider_category="runtime_repair",
    )

    _assert_private_labels_absent(_captured_prompt_text(client))
    assert "Male" in repaired
    assert "EverSmokerLocal" in repaired
    assert "value = 3" in repaired


def test_table_one_minimal_patch_never_sends_private_deterministic_script() -> None:
    context, step = _private_table_one_bound_step()
    response = json.dumps(
        {
            "format": PATCH_FORMAT,
            "edits": [{"old": "value = 1", "new": "value = 4", "expected_count": 1}],
        }
    )
    client = ExternalCaptureMockLLMClient([response])
    authority = _repair_authority(validator="mechanical_code_preflight")

    repaired = CoderAgent(client).repair(
        context=context,
        step=step,
        code=_private_table_one_script(),
        run_log="private deterministic script",
        repair_authority=authority,
        current_repair_authority=authority,
    )

    prompt = _captured_prompt_text(client)
    _assert_private_labels_absent(prompt)
    assert "__easyicu_table1_label_" in prompt
    assert "value = 4" in repaired


def test_table_one_full_rewrite_never_sends_private_deterministic_script() -> None:
    context, step = _private_table_one_bound_step()
    from easyicu.research_agent.research_context.outbound import outbound_safe_script

    safe_rewrite = outbound_safe_script(step, _private_table_one_script()).replace(
        "value = 1", "value = 5"
    )
    client = ExternalCaptureMockLLMClient(["not a patch", safe_rewrite])
    authority = _repair_authority(validator="runtime_executor")

    repaired = CoderAgent(client).repair(
        context=context,
        step=step,
        code=_private_table_one_script(),
        run_log="Female Male full rewrite diagnostic",
        repair_authority=authority,
        current_repair_authority=authority,
    )

    assert len(client.calls) == 2
    _assert_private_labels_absent(_captured_prompt_text(client))
    assert "Female" in repaired
    assert "EverSmokerLocal" in repaired
    assert "value = 5" in repaired


def test_archival_analysis_step_remains_readable_without_new_optional_spec() -> None:
    step = AnalysisStep.model_validate(_step(include_spec=False))
    assert step.table_one_spec is None


def test_table_one_spec_must_bind_only_explicit_step_inputs() -> None:
    payload = _step(include_spec=True)
    payload["inputs"] = ["age"]
    with pytest.raises(ValidationError, match="must be explicit step inputs"):
        AnalysisStep.model_validate(payload)


def test_plan_normalizer_keeps_only_closed_table_one_schema() -> None:
    payload = {
        "research_question": "Describe the cohort.",
        "steps": [_step(include_spec=True)],
    }
    payload["steps"][0]["table_one_spec"]["invented_policy"] = "ignore"
    payload["steps"][0]["table_one_spec"]["variables"][0]["invented"] = True
    normalized, dropped = _normalise_plan_payload(payload)
    spec = normalized["steps"][0]["table_one_spec"]
    assert "invented_policy" not in spec
    assert "invented" not in spec["variables"][0]
    assert dropped["table_one_spec"] == [
        "step[0]:invented_policy",
        "step[0].variables[0]:invented",
    ]


def test_table_one_sdk_guidance_is_only_added_for_typed_table_one() -> None:
    typed = AnalysisStep.model_validate(_step(include_spec=True))
    legacy = AnalysisStep.model_validate(_step(include_spec=False))
    full = load_prompt_pack()["coder"]

    typed_guide = coder_guide_for_step(full, typed)
    assert "build_grouped_table_one" in typed_guide
    assert "reconcile_measurement_source_status" in typed_guide
    assert "per-group missingness" in typed_guide
    assert "build_grouped_table_one" not in coder_guide_for_step(full, legacy)

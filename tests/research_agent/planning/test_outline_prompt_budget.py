"""Exercise the real outline request boundary using synthetic metadata only."""

import json

import pytest

from easyicu.research_agent.agents.progressive_attempt import progressive_planner_failure_facts
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.prompt_projection import OUTLINE_COMPACTION, outline_action_blocks
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext


class OutlineReceived(Exception):
    """Stop the scripted client after it receives the initial request."""


_VARIABLES = (
    "respiratory_rate", "heart_rate", "temperature", "mean_arterial_pressure",
    "oxygen_saturation", "creatinine", "platelet_count", "age_at_admission",
)
_TRAJECTORY_QUESTION = (
    "在 ICU 入科后，基于纵向呼吸频率轨迹进行聚类，描述轨迹类别与结局的关联。"
    "对齐预先设定的时间窗口并评价重抽样稳定性。轨迹类别不用于 causal inference。"
)
_CITATIONS = (
    "strobe_2007", "record_2015", "suissa_immortal_time_2008", "anderson_landmark_1983",
    "grambsch_therneau_ph_1994", "royston_parmar_rmst_2011", "durrleman_splines_1989",
    "sterne_missing_data_2009", "ricu_2023", "johnson_mimiciv_2023",
)


def _context(question):
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="synthetic_icu", database="example", n_stays=0,
            id_columns=["stay_id"],
        ),
        variables=[
            ConceptDescriptor(
                name=name, source_concept=name, dtype="float64",
                description=f"Synthetic prespecified measurement of {name} in the source units.",
                analysis_window="icu_admission[0,24]h",
            )
            for name in _VARIABLES
        ],
        primary_exposure=_VARIABLES[0],
    )


def _contract(window_end=32):
    return "Reviewed source measurement inventory (binding):\n" + "\n".join(
        f"{name}, window {start:02d}–{start + 2:02d} hours after admission: "
        "record coverage before filtering; keep empty windows in the coverage audit; "
        "mark source timestamps outside this interval ineligible for this coordinate; "
        "report the eligible denominator and the number with a recorded measurement."
        for name in _VARIABLES
        for start in range(0, window_end, 2)
    )


def _run_to_outline(context, contract):
    client = ScriptedMockLLMClient([OutlineReceived()])
    try:
        ProgressivePlannerAgent(client).run_attempt(
            context, planning_contract_context=contract,
            allowed_literature_citation_keys=_CITATIONS, stop_after_outline=True,
        )
    except OutlineReceived as stop:
        (call,) = client.calls
        return progressive_planner_failure_facts(stop).prompt_metrics, call[0][1].content
    except ProgressivePlanCompileError as error:
        assert not client.calls
        pytest.fail(f"{error.reason_code}: {error.details.get('metrics')}")
    pytest.fail("The scripted client did not receive the outline.")


def test_large_trajectory_outline_has_room_under_the_original_limit():
    context = _context(_TRAJECTORY_QUESTION)
    contract = _contract()
    metrics, prompt = _run_to_outline(context, contract)

    assert metrics["byte_limit"] == 120_000
    assert metrics["source_request_bytes"] > 120_000
    assert metrics["request_bytes"] <= 102_000
    assert metrics["request_bytes"] == sum(metrics["block_bytes"].values())
    assert metrics["request_bytes"] == metrics["message_bytes"] + metrics["schema_bytes"]
    assert 40_000 <= metrics["source_block_bytes"]["action_rows"] <= 44_000
    assert 40_000 <= metrics["source_block_bytes"]["planning_contract"] <= 45_000
    assert 18_000 <= metrics["source_block_bytes"]["response_shape"] <= 23_000
    assert metrics["block_bytes"]["planning_contract"] == metrics["source_block_bytes"]["planning_contract"]
    assert metrics["block_bytes"]["action_rows"] <= 30_000
    assert metrics["block_bytes"]["response_shape"] <= 17_000
    assert {receipt["block"] for receipt in metrics["prompt_projection_receipts"]
            if receipt["reason"] == OUTLINE_COMPACTION} == {
        "action_rows", "article_question_context", "response_shape",
    }
    assert contract in prompt
    assert all(f'"{variable}"' in prompt for variable in _VARIABLES)


@pytest.mark.parametrize("pressure", [False, True], ids=["ordinary", "reviewed-inventory"])
def test_multi_variable_association_retains_questions_and_actions(pressure):
    context = _context(
        "How is admission respiratory rate associated with oxygen saturation "
        "after adjustment for age at admission and heart rate?"
    )
    contract = "Preserve the adjustment roster and report the association in source units."
    if pressure:
        contract += "\n" + _contract(window_end=48)
    metrics, prompt = _run_to_outline(context, contract)

    assert metrics["byte_limit"] == 120_000
    assert metrics["request_bytes"] == sum(metrics["block_bytes"].values())
    if pressure:
        assert metrics["source_request_bytes"] > 120_000
        assert metrics["request_bytes"] <= 110_000
        assert metrics["block_bytes"]["action_rows"] < metrics["source_block_bytes"]["action_rows"]
        assert metrics["block_bytes"]["planning_contract"] == metrics["source_block_bytes"]["planning_contract"]
    else:
        assert metrics["request_bytes"] <= 80_000
        assert metrics["request_bytes"] == metrics["source_request_bytes"]
        assert metrics["prompt_projection_receipts"] == []
    assert context.research_question in prompt
    assert contract in prompt
    assert "association.adjusted_association" in prompt
    assert all(f'"{variable}"' in prompt for variable in _VARIABLES)


def test_compacted_outline_preserves_action_policy_roles_and_response_fields():
    context = _context(_TRAJECTORY_QUESTION)
    _, full = _run_to_outline(context, "Preserve the declared outcome and timing.")
    _, compact = _run_to_outline(context, _contract())

    action_header = "Retrieved scientific actions (only these may be selected):\n"
    expanded = json.loads(full.split(action_header, 1)[1].splitlines()[0])
    legend, table, *_ = compact.split(action_header, 1)[1].splitlines()
    shared_positions = json.loads(legend.split("index this list from zero: ", 1)[1][:-1])
    modes = {"H": "host_owned", "C": "coder_generated"}
    restored_actions = {}
    for family, rows in json.loads(table).items():
        for values in rows:
            action_id, name, purpose, produces, mode, *tail = values
            notes, inputs, runtime, position = (tail + ["", [], None, None][len(tail):])
            item = {
                "analysis_type": family, "action_id": action_id, "name": name,
                "purpose": purpose, "produces": produces, "execution_mode": modes[mode],
                "notes": notes, "required_inputs": inputs, "runtime_contract": runtime,
            }
            if runtime is not None:
                products, prerequisites, roles, parameters = runtime
                item["runtime_contract"] = {
                    "outputs": products, "required_product_inputs": prerequisites,
                    "article_roles": roles, "execution_parameters": parameters,
                }
            if position is not None:
                item["outline_position"] = (
                    shared_positions[position] if isinstance(position, int) else position
                )
            restored_actions[family, action_id] = item
    for item in expanded:
        if runtime := item["runtime_contract"]:
            # These two runtime fields are filled by the host at the step stage.
            runtime.pop("standard_executor")
            runtime["outputs"] = [output["product_id"] for output in runtime["outputs"]]
    assert restored_actions == {(item["analysis_type"], item["action_id"]): item for item in expanded}

    full_shape = full.split("Exact ProgressivePlanOutline JSON shape", 1)[1].splitlines()
    compact_shape = compact.split("Exact ProgressivePlanOutline JSON shape", 1)[1].splitlines()
    example = json.loads(compact_shape[1])
    selected, = example["design_selection"]["candidates"]
    overrides = compact_shape[2].split("with these overrides: ", 1)[1].split(". Fill ", 1)[0]
    example["design_selection"]["candidates"].append({**selected, **json.loads(overrides)})
    assert example == json.loads(full_shape[1])
    module_rosters = {
        family: modules
        for families, modules in json.loads(compact_shape[3].split(": ", 1)[1])
        for family in families
    }
    assert module_rosters == json.loads(full_shape[3].split(": ", 1)[1])

    role_header = "Candidate-specific host article role contracts:\n"
    compact_roles = json.loads(compact.split(role_header, 1)[1].splitlines()[1])
    for contract in compact_roles:
        contract["available_modules"] = module_rosters[contract["analysis_type"]]
        contract["requirements"] = [
            dict(zip(("article_requirement_id", "role", "required"), row, strict=True))
            for row in contract["requirements"]
        ]
    assert compact_roles == json.loads(full.split(role_header, 1)[1].splitlines()[0])


def test_unrecognized_action_contract_fields_stay_in_the_prompt():
    rows = [
        {
            "analysis_type": "descriptive_epidemiology", "action_id": "descriptive.example",
            "name": "Example", "purpose": "Describe the source", "produces": "summary",
            "execution_mode": "host_owned", "notes": "", "required_inputs": [],
            "runtime_contract": {"future_scientific_policy": "Preserve every endpoint."},
        },
    ]
    menu, _ = outline_action_blocks(rows, {"descriptive_epidemiology": ("report",)})
    assert dict(menu.projections)[OUTLINE_COMPACTION] == menu.text
    assert json.loads(menu.text.splitlines()[1]) == rows


def test_outline_retry_reuses_the_compact_shape():
    client = ScriptedMockLLMClient(["{}", OutlineReceived()])
    with pytest.raises(OutlineReceived):
        ProgressivePlannerAgent(client).run_attempt(
            _context(_TRAJECTORY_QUESTION),
            planning_contract_context=_contract(),
            allowed_literature_citation_keys=_CITATIONS,
            stop_after_outline=True,
        )
    (first, _), (retry, _) = client.calls
    assert retry[:2] == first
    assert "Follow the complete output shape and constraints in the initial request." in retry[-1].content
    assert "Exact ProgressivePlanOutline JSON shape" not in retry[-1].content
    assert sum(len(message.content.encode("utf-8")) for message in retry) < 120_000

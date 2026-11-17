"""Focused display-label and outline-authority contracts for progressive planning."""

from __future__ import annotations

import hashlib
import json

import pytest

from easyicu.research_agent.agents.progressive_planner import (
    ProgressivePlannerAgent,
    _bind_required_outline_method_sources,
)
from easyicu.research_agent.planning.progressive_compiler import (
    required_reader_display_label_keys,
    validate_progressive_foundation,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanFoundation,
    ProgressivePlanOutline,
    ProgressiveStepMaterialization,
)
from easyicu.research_agent.planning.progressive_host_materialization import (
    progressive_module_method_source_keys,
)
from easyicu.research_agent.schema import UserPreferences

from .test_progressive_planner_v2 import _context, _outline_payload, _payload


def _late_binary_labels_case():
    from .progressive_planner_fixtures import (
        _foundation_payload,
        _materialization_payloads,
    )

    payload = _payload()
    payload["display_labels"] = [
        label for label in payload["display_labels"] if "=" not in label["key"]
    ]
    return (
        _context().model_copy(update={"primary_exposure": None}),
        _outline_payload(payload),
        _foundation_payload(payload),
        _materialization_payloads(payload),
    )


def test_late_binary_labels_repair_keeps_candidate_and_accepted_steps(tmp_path):
    from easyicu.research_agent.planning.progressive_artifacts import (
        load_progressive_planner_checkpoint_chain,
        persist_progressive_planner_checkpoint,
    )
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from .test_progressive_planner_checkpoints import _RecordingEvidence

    context, outline, foundation, steps = _late_binary_labels_case()
    repair = {"display_labels": {
        "exposure_flag=0": "Exposure absent",
        "exposure_flag=1": "Exposure present",
    }}
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [
        outline, foundation, *steps[:3], repair, *steps[3:],
    ]])
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    checkpoints = []
    plan = agent.run(context, checkpoint_callback=checkpoints.append)

    assert len(llm.calls) == 10
    assert agent.last_result.facts.materializations == tuple(
        ProgressiveStepMaterialization.model_validate(p) for p in steps
    )
    assert "exposure_flag=0" in llm.calls[5][0][-1].content
    assert set(plan.display_labels) >= {"exposure_flag=0", "exposure_flag=1"}
    assert checkpoints[3].materializations == checkpoints[4].materializations[:2]
    assert checkpoints[-1].foundation == agent.last_result.facts.foundation
    assert checkpoints[-1].foundation.foundation.cohort.model_dump() == foundation["foundation"]["cohort"]
    assert agent.last_result.facts.prompt_metrics["compile_revision_count"] == 1
    # Append and reload the amendment without rewriting the accepted history.
    evidence = _RecordingEvidence()
    paths = [persist_progressive_planner_checkpoint(
        run_dir=tmp_path, evidence=evidence, checkpoint=checkpoint, prompt_pack_version="test",
    ) for checkpoint in checkpoints]
    original_bytes = paths[3].read_bytes()
    loaded = load_progressive_planner_checkpoint_chain(
        last_checkpoint_path=paths[-1],
        expected_artifact_sha256=hashlib.sha256(paths[-1].read_bytes()).hexdigest(),
    )
    assert loaded == tuple(checkpoints)
    assert paths[3].read_bytes() == original_bytes


@pytest.mark.parametrize("invalid_labels", [
    {"exposure_flag=0": "Exposure", "exposure_flag=1": " exposure "},
    {"exposure_flag=0": "exposure flag=0", "exposure_flag=1": "exposure flag=1"},
])
def test_late_label_repair_exhaustion_is_typed_and_preserves_prefix(invalid_labels):
    from easyicu.research_agent.agents.progressive_planner import progressive_planner_failure_facts
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    context, outline, foundation, steps = _late_binary_labels_case()
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [
        outline, foundation, *steps[:3],
        {"display_labels": invalid_labels}, {"display_labels": invalid_labels},
    ]])
    agent = ProgressivePlannerAgent(llm)
    with pytest.raises(ProgressivePlanCompileError) as caught:
        agent.run(context)

    assert caught.value.reason_code == "progressive_foundation_label_repair_exhausted"
    assert caught.value.path == "display_labels"
    assert len(llm.calls) == 7
    facts = progressive_planner_failure_facts(caught.value)
    assert len(facts.materializations) == 2
    assert facts.foundation.foundation.display_labels == ProgressivePlanFoundation.model_validate(
        foundation["foundation"]
    ).display_labels


@pytest.mark.parametrize("source_index", [3, 4])
def test_resume_before_and_after_label_amendment_keeps_the_prefix(source_index):
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    context, outline, foundation, steps = _late_binary_labels_case()
    repair = {"display_labels": {
        "exposure_flag=0": "Exposure absent", "exposure_flag=1": "Exposure present",
    }}
    dependencies = {
        "cohort_file_sha256": "a" * 64, "llm_signature": "mock:test", "prompt_version": "test",
    }
    source = ProgressivePlannerAgent(ScriptedMockLLMClient([json.dumps(p) for p in [
        outline, foundation, *steps[:3], repair, *steps[3:],
    ]]))
    checkpoints = []
    source.run(context, checkpoint_callback=checkpoints.append, resume_dependency_context=dependencies)
    checkpoint = checkpoints[source_index]
    remaining = [steps[2], repair, *steps[3:]] if source_index == 3 else steps[3:]
    llm = ScriptedMockLLMClient([json.dumps(p) for p in remaining])
    resumed = ProgressivePlannerAgent(llm)
    resumed.run(context, resume_checkpoint=checkpoint, resume_dependency_context=dependencies)
    assert len(llm.calls) == len(remaining)
    assert resumed.last_result.facts.materializations == source.last_result.facts.materializations
    assert resumed.last_result.facts.foundation == source.last_result.facts.foundation
    assert resumed.last_result.facts.resume_validated


@pytest.mark.parametrize("label", ["Patient age", "age years"])
def test_late_reader_label_uses_the_same_targeted_repair(monkeypatch, label):
    from easyicu.research_agent.agents import progressive_planner as owner
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from .progressive_planner_fixtures import _foundation_payload, _materialization_payloads

    foundation = _foundation_payload()
    foundation["foundation"]["display_labels"] = [
        item for item in foundation["foundation"]["display_labels"] if item["key"] != "age_years"
    ]
    steps = _materialization_payloads()
    real_compile = owner.compile_progressive_prefix

    def compile_with_reader_requirement(state, materialization, **kwargs):
        if materialization.step.module_id == "exposure_outcome_distribution":
            validate_progressive_foundation(
                kwargs["foundation"], context=kwargs["context"], analysis_type="association_study",
                required_reader_display_label_keys=("age_years",),
            )
        return real_compile(state, materialization, **kwargs)

    monkeypatch.setattr(owner, "compile_progressive_prefix", compile_with_reader_requirement)
    repair = {"display_labels": {"age_years": label}}
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [
        _outline_payload(), foundation, *steps[:3], repair,
        *([repair] if label == "age years" else steps[3:]),
    ]])
    agent = ProgressivePlannerAgent(llm)
    if label == "age years":
        with pytest.raises(ProgressivePlanCompileError) as caught:
            agent.run(_context())
        assert caught.value.reason_code == "progressive_foundation_label_repair_exhausted"
        assert caught.value.details["cause_code"] == "progressive_required_reader_display_labels_missing"
    else:
        plan = agent.run(_context())
        assert plan.display_labels["age_years"] == "Patient age"
        assert len(llm.calls) == 10
    assert '"required_display_label_keys": ["age_years"]' in llm.calls[5][0][-1].content


def test_label_repair_cannot_change_other_foundation_fields():
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    context, outline, foundation, steps = _late_binary_labels_case()
    forbidden = {"display_labels": {
        "exposure_flag=0": "Exposure absent", "exposure_flag=1": "Exposure present",
        "outcome_flag": "Changed endpoint",
    }, "cohort": {"name": "changed"}}
    repair = {"display_labels": {
        "exposure_flag=0": "Exposure absent", "exposure_flag=1": "Exposure present",
    }}
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [
        outline, foundation, *steps[:3], forbidden, repair, *steps[3:],
    ]])
    agent = ProgressivePlannerAgent(llm)
    agent.run(context)
    facts = agent.last_result.facts
    assert facts.prompt_metrics["compile_revision_count"] == 2
    assert facts.foundation.foundation.cohort.model_dump() == foundation["foundation"]["cohort"]
    labels = {item.key: item.value for item in facts.foundation.foundation.display_labels}
    assert labels["outcome_flag"] == "In-hospital outcome"
    assert len(llm.calls) == 11


def test_single_attempt_policy_does_not_spend_on_label_repair():
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.research_agent.providers.structured_retry import automatic_retry_scope

    context, outline, foundation, steps = _late_binary_labels_case()
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [outline, foundation, *steps[:3]]])
    with automatic_retry_scope(disabled=True), pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run(context)
    assert caught.value.reason_code == "progressive_foundation_label_repair_exhausted"
    assert len(llm.calls) == 5


def test_partial_binary_pair_repairs_only_the_missing_level():
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    context, outline, foundation, steps = _late_binary_labels_case()
    foundation["foundation"]["display_labels"].append(
        {"key": "exposure_flag=0", "value": "Exposure absent"}
    )
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [
        outline, foundation, *steps[:3],
        {"display_labels": {"exposure_flag=1": "Exposure absent"}},
        {"display_labels": {"exposure_flag=1": "Exposure present"}}, *steps[3:],
    ]])
    agent = ProgressivePlannerAgent(llm)
    plan = agent.run(context)
    assert plan.display_labels["exposure_flag=0"] == "Exposure absent"
    assert agent.last_result.facts.prompt_metrics["compile_revision_count"] == 2
    repair = agent.last_result.facts.prompt_metrics["foundation_label_repairs"][0]
    assert repair["required_display_label_keys"] == ["exposure_flag=1"]


def test_label_amendment_receipt_cannot_authorize_other_foundation_changes():
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.planning.progressive_artifacts import (
        ProgressivePlanningArtifactError,
        _validate_revised_checkpoint_transition,
    )
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlannerCheckpoint
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    context, outline, foundation, steps = _late_binary_labels_case()
    repair = {"display_labels": {
        "exposure_flag=0": "Exposure absent", "exposure_flag=1": "Exposure present",
    }}
    agent = ProgressivePlannerAgent(ScriptedMockLLMClient([json.dumps(p) for p in [
        outline, foundation, *steps[:3], repair, *steps[3:],
    ]]))
    checkpoints = []
    agent.run(context, checkpoint_callback=checkpoints.append)
    body = checkpoints[4].model_dump(mode="json")
    body["foundation"]["foundation"]["cohort"]["name"] = "changed"
    body["prompt_metrics"]["foundation_label_repairs"][-1]["foundation_sha256"] = canonical_sha256(
        body["foundation"]
    )
    body["checkpoint_sha256"] = canonical_sha256({
        key: value for key, value in body.items() if key != "checkpoint_sha256"
    })
    forged = ProgressivePlannerCheckpoint.model_validate(body)
    with pytest.raises(ProgressivePlanningArtifactError, match="label-only repair receipt"):
        _validate_revised_checkpoint_transition(checkpoints[3], forged)


def test_label_repair_shares_the_current_step_compiler_budget(monkeypatch):
    from easyicu.research_agent.agents import progressive_planner as owner
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    context, outline, foundation, steps = _late_binary_labels_case()
    repair = {"display_labels": {
        "exposure_flag=0": "Exposure absent", "exposure_flag=1": "Exposure present",
    }}
    real_compile = owner.compile_progressive_prefix

    def reject_another_step_contract(state, materialization, **kwargs):
        result = real_compile(state, materialization, **kwargs)
        if materialization.step.module_id == "exposure_outcome_distribution":
            raise ProgressivePlanCompileError(
                "progressive_frequency_contract_invalid", "Repair the current frequency contract.",
                step_id=materialization.step.step_id, step_index=2, path="denominator_policy",
            )
        return result

    monkeypatch.setattr(owner, "compile_progressive_prefix", reject_another_step_contract)
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [
        outline, foundation, *steps[:3], repair, steps[2],
    ]])
    agent = ProgressivePlannerAgent(llm)
    with pytest.raises(ProgressivePlanCompileError) as caught:
        agent.run(context)
    assert caught.value.reason_code == "progressive_frequency_contract_invalid"
    assert agent._attempt.prompt_metrics["compile_revision_count"] == 2
    assert len(llm.calls) == 7


def test_resume_repairs_a_mechanical_label_accepted_by_the_older_binary_gate():
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlannerCheckpoint
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from .progressive_planner_fixtures import _foundation_payload, _materialization_payloads

    steps = _materialization_payloads()
    dependencies = {
        "cohort_file_sha256": "a" * 64, "llm_signature": "mock:test", "prompt_version": "test",
    }
    source = ProgressivePlannerAgent(ScriptedMockLLMClient([json.dumps(p) for p in [
        _outline_payload(), _foundation_payload(), *steps,
    ]]))
    checkpoints = []
    source.run(_context(), checkpoint_callback=checkpoints.append, resume_dependency_context=dependencies)
    # The old binary gate accepted distinct identifier strings. Reconstruct
    # that historical foundation with a valid checkpoint digest.
    body = checkpoints[4].model_dump(mode="json")
    for item in body["foundation"]["foundation"]["display_labels"]:
        if item["key"] == "exposure_flag=1":
            item["value"] = "exposure flag=1"
    body["checkpoint_sha256"] = canonical_sha256({
        key: value for key, value in body.items() if key != "checkpoint_sha256"
    })
    checkpoint = ProgressivePlannerCheckpoint.model_validate(body)
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [
        {"display_labels": {"exposure_flag=1": "Exposure present"}}, *steps[3:],
    ]])
    resumed = ProgressivePlannerAgent(llm)
    resumed.run(_context(), resume_checkpoint=checkpoint, resume_dependency_context=dependencies)
    assert resumed.last_result.facts.materializations == source.last_result.facts.materializations
    assert resumed.last_result.facts.foundation == source.last_result.facts.foundation
    assert len(llm.calls) == 5


def test_new_binary_scope_during_final_suffix_repair_keeps_the_amended_foundation(monkeypatch):
    from copy import deepcopy
    from easyicu.research_agent.agents import progressive_planner as owner
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from .progressive_planner_fixtures import _foundation_payload, _materialization_payloads

    context = _context()
    context = context.model_copy(update={"variables": [
        variable.model_copy(update={"observed_domain": {
            "n_unique": 2, "is_binary": True, "levels": [0, 1],
        }}) if variable.name == "sex_code" else variable
        for variable in context.variables
    ]})
    steps = _materialization_payloads()
    suffix = deepcopy(steps[2:])
    suffix[0]["step"]["primary_exposure"] = "sex_code"
    suffix[0]["step"]["raw_inputs"].append("sex_code")
    original_accept = owner._accept_compiled_plan
    accept_count = 0

    def accept(**kwargs):
        nonlocal accept_count
        accept_count += 1
        if accept_count == 1:
            raise ProgressivePlanCompileError(
                "progressive_primary_result_invalid", "Recheck the distribution.",
                step_id="03_distribution", step_index=2, path="primary_result",
            )
        return original_accept(**kwargs)

    monkeypatch.setattr(owner, "_accept_compiled_plan", accept)
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [
        _outline_payload(), _foundation_payload(), *steps,
        suffix[0], {"display_labels": {"sex_code=0": "First category", "sex_code=1": "Second category"}},
        *suffix[1:],
    ]])
    agent = ProgressivePlannerAgent(llm)
    checkpoints = []
    plan = agent.run(context, checkpoint_callback=checkpoints.append)
    assert accept_count == 2
    assert plan.display_labels["sex_code=0"] == "First category"
    assert checkpoints[9].materializations[:2] == checkpoints[8].materializations[:2]
    assert checkpoints[-1].foundation == agent.last_result.facts.foundation
    assert checkpoints[-1].schema_version.endswith("/2")
    assert len(llm.calls) == 15


def test_outline_binds_sealed_method_source_to_applicable_scientific_module() -> None:
    payload = _outline_payload()
    for step in payload["steps"]:
        if step["planned_analysis_role"] == "secondary":
            step["literature_citation_keys"] = []
    outline = _bind_required_outline_method_sources(
        ProgressivePlanOutline.model_validate(payload),
        allowed_literature_citation_keys=["strobe_2007", "record_2015"],
        context_required_method_layers=(),
        continuous_domain_variables=(),
    )

    distribution = next(
        step
        for step in outline.steps
        if step.module_id == "exposure_outcome_distribution"
    )
    assert distribution.literature_citation_keys == ["strobe_2007"]
    assert progressive_module_method_source_keys(
        "exposure_outcome_distribution", ["record_2015", "strobe_2007"]
    ) == ("record_2015", "strobe_2007")


def test_counts_only_outline_rejects_inferential_plan_review_copy() -> None:
    payload = _outline_payload()
    payload["analysis_type"] = "descriptive_epidemiology"
    for candidate in payload["design_selection"]["candidates"]:
        candidate["analysis_type"] = "descriptive_epidemiology"
    payload["design_selection"]["candidates"][0]["primary_method"] = (
        "按 stay-level 分母计算比例及其不确定性"
    )
    outline = ProgressivePlanOutline.model_validate(payload)
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_family": "descriptive_epidemiology",
                            "analysis_unit": "icu_stay",
                            "cluster_unit": None,
                            "variance_estimator": "none_counts_only",
                        }
                    }
                )
            )
        }
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            outline,
            analysis_types=("descriptive_epidemiology",),
            variable_names=tuple(variable.name for variable in context.variables),
            allowed_literature_citation_keys=(),
            primary_exposure=context.primary_exposure,
            target_outcome=context.target_outcome,
            require_design_selection=True,
            article_context=context,
        )

    assert caught.value.reason_code == (
        "progressive_selected_design_counts_only_claim_exceeded"
    )
    assert caught.value.path.endswith("primary_method")


def test_outline_rejects_unbound_scientific_step_before_materialization() -> None:
    payload = _outline_payload()
    for step in payload["steps"]:
        if step["planned_analysis_role"] in {"primary", "secondary", "sensitivity"}:
            step["literature_citation_keys"] = ["strobe_2007"]
    target = next(
        step
        for step in payload["steps"]
        if step["module_id"] == "exposure_outcome_distribution"
    )
    target["literature_citation_keys"] = []

    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent._validate_outline_authority(
            ProgressivePlanOutline.model_validate(payload),
            analysis_types=["association_study"],
            variable_names=[variable.name for variable in _context().variables],
            allowed_literature_citation_keys=["strobe_2007"],
        )

    assert caught.value.reason_code == (
        "progressive_outline_scientific_citation_missing"
    )
    assert caught.value.step_id == target["step_id"]


def test_outline_rejects_landmark_declined_by_typed_timing_authority() -> None:
    outline_payload = _outline_payload()
    selected = outline_payload["design_selection"]["candidates"][0]
    selected.update(
        {
            "design_id": "landmark_descriptive_design",
            "time_zero": "A prespecified landmark after cohort entry.",
            "observation_window": "From the landmark to discharge.",
        }
    )
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                timing_and_design=json.dumps(
                    {
                        "analysis_scope": "descriptive_distribution_only",
                        "association_timing": "not_authorized",
                        "landmark": "not_authorized",
                    }
                )
            )
        }
    )

    with pytest.raises(
        ProgressivePlanCompileError,
        match="progressive_selected_design_landmark_not_authorized",
    ):
        ProgressivePlannerAgent._validate_outline_authority(
            ProgressivePlanOutline.model_validate(outline_payload),
            analysis_types=["association_study"],
            variable_names=[variable.name for variable in context.variables],
            allowed_literature_citation_keys=[],
            primary_exposure="exposure_flag",
            target_outcome="outcome_flag",
            require_design_selection=True,
            article_context=context,
        )


def test_outline_rejects_time_varying_design_when_association_timing_declined() -> None:
    outline_payload = _outline_payload()
    selected = outline_payload["design_selection"]["candidates"][0]
    selected.update(
        {
            "design_id": "timevarying_descriptive",
            "primary_method": "Time-updated descriptive risk summaries",
        }
    )
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                timing_and_design=json.dumps(
                    {
                        "analysis_scope": "descriptive_distribution_only",
                        "association_timing": "not_authorized",
                        "landmark": "not_authorized",
                    }
                )
            )
        }
    )

    with pytest.raises(
        ProgressivePlanCompileError,
        match="progressive_selected_design_association_timing_not_authorized",
    ):
        ProgressivePlannerAgent._validate_outline_authority(
            ProgressivePlanOutline.model_validate(outline_payload),
            analysis_types=["association_study"],
            variable_names=[variable.name for variable in context.variables],
            allowed_literature_citation_keys=[],
            primary_exposure="exposure_flag",
            target_outcome="outcome_flag",
            require_design_selection=True,
            article_context=context,
        )


@pytest.mark.parametrize(
    "display_labels",
    [
        [],
        [{"key": "exposure_flag=0", "value": "Exposure absent"}],
        [
            {"key": "exposure_flag=0", "value": "Exposure"},
            {"key": "exposure_flag=1", "value": " exposure "},
        ],
    ],
)
def test_foundation_requires_complete_distinct_binary_figure_labels(
    display_labels: list[dict[str, str]],
) -> None:
    foundation = ProgressivePlanFoundation.model_validate(
        {
            "cohort": _payload()["cohort"],
            "display_labels": display_labels,
            "robustness_intents": [],
            "know_how_decisions": [],
        }
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        validate_progressive_foundation(
            foundation,
            context=_context(),
            analysis_type="descriptive_epidemiology",
            required_binary_display_label_scopes=("exposure_flag",),
        )

    assert caught.value.reason_code == (
        "progressive_required_binary_display_labels_missing"
    )
    assert caught.value.path == "display_labels"


def test_foundation_accepts_complete_distinct_binary_figure_labels() -> None:
    foundation = ProgressivePlanFoundation.model_validate(
        {
            "cohort": _payload()["cohort"],
            "display_labels": _payload()["display_labels"],
            "robustness_intents": [],
            "know_how_decisions": [],
        }
    )

    validate_progressive_foundation(
        foundation,
        context=_context(),
        analysis_type="descriptive_epidemiology",
        required_binary_display_label_scopes=("exposure_flag",),
    )


def test_foundation_requires_selected_design_reader_labels() -> None:
    outline = ProgressivePlanOutline.model_validate(_outline_payload())
    required_keys = required_reader_display_label_keys(_context(), outline.design_selection)
    foundation_payload = {
        "cohort": _payload()["cohort"],
        "display_labels": [
            {"key": "exposure_flag=0", "value": "Exposure absent"},
            {"key": "exposure_flag=1", "value": "Exposure present"},
        ],
        "robustness_intents": [],
        "know_how_decisions": [],
    }

    with pytest.raises(ProgressivePlanCompileError) as caught:
        validate_progressive_foundation(
            ProgressivePlanFoundation.model_validate(foundation_payload),
            context=_context(),
            analysis_type="descriptive_epidemiology",
            required_reader_display_label_keys=required_keys,
        )

    assert caught.value.reason_code == (
        "progressive_required_reader_display_labels_missing"
    )
    assert caught.value.details["required_key"] == "exposure_flag"

    foundation_payload["display_labels"] = [
        *foundation_payload["display_labels"],
        {"key": "exposure_flag", "value": "Exposure status"},
        {"key": "outcome_flag", "value": "In-hospital outcome"},
    ]
    validate_progressive_foundation(
        ProgressivePlanFoundation.model_validate(foundation_payload),
        context=_context(),
        analysis_type="descriptive_epidemiology",
        required_reader_display_label_keys=required_keys,
    )


def test_reader_display_label_keys_exclude_row_identifiers() -> None:
    from types import SimpleNamespace

    selection = SimpleNamespace(
        selected=SimpleNamespace(
            required_variables=["stay_id", "exposure_flag", "outcome_flag"]
        )
    )

    assert required_reader_display_label_keys(_context(), selection) == (
        "exposure_flag",
        "outcome_flag",
    )


def test_foundation_rejects_mechanically_humanized_reader_label() -> None:
    foundation = ProgressivePlanFoundation.model_validate(
        {
            "cohort": _payload()["cohort"],
            "display_labels": [
                {"key": "exposure_flag", "value": "exposure flag"},
                {"key": "outcome_flag", "value": "Clinical outcome"},
            ],
            "robustness_intents": [],
            "know_how_decisions": [],
        }
    )

    with pytest.raises(ProgressivePlanCompileError) as caught:
        validate_progressive_foundation(
            foundation,
            context=_context(),
            analysis_type="descriptive_epidemiology",
            required_reader_display_label_keys=("exposure_flag", "outcome_flag"),
        )

    assert caught.value.reason_code == (
        "progressive_required_reader_display_labels_missing"
    )


def test_the_variables_own_sealed_description_is_a_reader_label() -> None:
    # pilot1 (10-22): the dictionary describes icu_unit_type as "ICU unit
    # type"; the Planner copied it as told and was refused on every attempt.
    context = _context()
    variables = [
        variable.model_copy(update={"description": "Exposure flag"})
        if variable.name == "exposure_flag" else variable
        for variable in context.variables
    ]
    context = context.model_copy(update={"variables": variables})
    foundation = ProgressivePlanFoundation.model_validate(
        {
            "cohort": _payload()["cohort"],
            "display_labels": [
                {"key": "exposure_flag", "value": "Exposure flag"},
                {"key": "outcome_flag", "value": "Clinical outcome"},
            ],
            "robustness_intents": [],
            "know_how_decisions": [],
        }
    )

    validate_progressive_foundation(
        foundation,
        context=context,
        analysis_type="descriptive_epidemiology",
        required_reader_display_label_keys=("exposure_flag", "outcome_flag"),
    )

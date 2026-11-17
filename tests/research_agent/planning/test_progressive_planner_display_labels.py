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

from .progressive_planner_fixtures import _context, _outline_payload, _payload


def _electrolyte_family_case(monkeypatch, *, missing):
    from easyicu.research_agent.agents import family_spec_planner
    from easyicu.research_agent.schema import ResearchContext
    from tests.support.distribution_policies import missing_policies_for
    from .family_spec_fixtures import _descriptive_context, _request

    # A new clinical question, with no rows or live-case material. The request
    # seam models producer/validator drift: the projected design needs a reader
    # key the small-spec request did not require from its answer.
    raw = json.dumps(_descriptive_context().model_dump(mode="json"))
    for old, new in (("phenotype_flag", "hypokalemia_flag"),
                     ("death", "arrhythmia_flag"), ("score_first", "potassium_entry")):
        raw = raw.replace(old, new)
    context = ResearchContext.model_validate(json.loads(raw)).model_copy(update={
        "research_question": (
            "What proportion of ICU stays have low potassium in the first 24 h, "
            "and how often is an arrhythmia recorded with and without low potassium?"
        ),
    })
    build_request = family_spec_planner.build_family_spec_request

    def request_with_label_drift(*args, **kwargs):
        request = build_request(*args, **kwargs)
        return request.model_copy(update={"required_reader_label_keys": [
            key for key in request.required_reader_label_keys if key != "hypokalemia_flag"
        ]})

    monkeypatch.setattr(family_spec_planner, "build_family_spec_request", request_with_label_drift)
    request = _request(context, cohort_mode="all_input_rows")
    request = request.model_copy(update={"required_reader_label_keys": [
        key for key in request.required_reader_label_keys if key != "hypokalemia_flag"
    ]})
    labels = {
        "hypokalemia_flag": "Low potassium during the observation window",
        "arrhythmia_flag": "Recorded cardiac rhythm disturbance",
        "age": "Age at admission in years", "sex": "Recorded patient sex",
        "potassium_entry": "Serum potassium at entry",
        "hypokalemia_flag=0": "Low potassium absent",
        "hypokalemia_flag=1": "Low potassium present",
    }
    keys = ("hypokalemia_flag",) if missing == "reader" else (
        "hypokalemia_flag=0", "hypokalemia_flag=1",
    )
    payload = {
        "adjustment_set": [], "baseline_variables": ["potassium_entry"],
        "distribution_missing_policies": missing_policies_for(request),
        "reader_display_labels": [{"key": key, "value": value}
                                  for key, value in labels.items() if key not in keys],
        "comparator_applications": [{"citation_key": key,
                                     "application": "Compare definitions and denominators."}
                                    for key in request.direct_comparator_literature_keys],
        "roster_decision_note": "Summarize entry potassium without an adjusted effect model.",
    }
    return context, payload, {"display_labels": {key: labels[key] for key in keys}}


def _run_electrolyte_family(context, responses, **kwargs):
    from easyicu.research_agent.agents.progressive_planner import FAMILY_SPEC_STRATEGY
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from .family_spec_fixtures import ALLOWED_CITATIONS, DIRECT_COMPARATORS

    llm = ScriptedMockLLMClient(responses)
    llm.supports_strict_json_schema = True
    agent = ProgressivePlannerAgent(llm)
    result = agent.run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        **kwargs,
    )
    return llm, result


@pytest.mark.parametrize("missing", ["reader", "binary"])
def test_family_projection_routes_missing_labels_to_the_existing_owner(monkeypatch, missing):
    from easyicu.research_agent.agents import family_spec_planner
    from easyicu.research_agent.agents import progressive_foundation_repair as owner

    context, payload, repair = _electrolyte_family_case(monkeypatch, missing=missing)
    drafts = []
    builder = family_spec_planner.build_descriptive_skeleton

    def project(*args, **kwargs):
        draft = builder(*args, **kwargs)
        drafts.append(draft)
        return draft

    monkeypatch.setattr(family_spec_planner, "build_descriptive_skeleton", project)
    calls = []
    repair_owner = owner.repair_foundation_display_labels

    def observe(*args, **kwargs):
        calls.append(kwargs)
        return repair_owner(*args, **kwargs)

    monkeypatch.setattr(owner, "repair_foundation_display_labels", observe)
    checkpoints = []
    llm, result = _run_electrolyte_family(
        context, [json.dumps(payload), json.dumps(repair)], checkpoint_callback=checkpoints.append,
    )
    assert len(llm.calls) == 2 and len(calls) == 1
    assert calls[0]["remaining_revisions"] == 2
    assert set(owner.foundation_label_repair_keys(calls[0]["finding"])) == set(repair["display_labels"])
    schema = json.loads(llm.calls[1][1]["structured_output"].schema_json)
    assert set(schema["properties"]["display_labels"]["properties"]) == set(repair["display_labels"])
    assert result.facts.prompt_metrics["compile_revision_count"] == 1
    assert len(drafts) == 1
    assert result.facts.outline == drafts[0].outline
    assert result.facts.materializations == drafts[0].materializations
    _assert_only_requested_labels_changed(drafts[0].foundation, result.facts.foundation, repair)
    assert checkpoints[1].foundation == result.facts.foundation


@pytest.mark.parametrize("missing", ["reader", "binary"])
@pytest.mark.parametrize("answer", ["{", '{"display_labels": {}}'])
def test_family_label_repair_spends_each_invalid_answer_and_keeps_original_code(
    monkeypatch, missing, answer,
):
    from easyicu.research_agent.agents.progressive_attempt import progressive_planner_failure_facts

    context, payload, _ = _electrolyte_family_case(monkeypatch, missing=missing)
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _run_electrolyte_family(context, [json.dumps(payload), answer, answer])
    error = caught.value
    assert error.reason_code == f"progressive_required_{missing}_display_labels_missing"
    assert error.path == "display_labels"
    assert error.easyicu_safe_diagnostic["metrics"]["foundation_label_repair_revisions"] == 2
    facts = progressive_planner_failure_facts(error)
    assert facts.prompt_metrics["compile_revision_count"] == 2
    assert facts.materializations == () and facts.compile_receipt is None


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_family_label_repair_uses_the_route_configured_revision_limit(monkeypatch, budget):
    from easyicu.research_agent.agents import progressive_planner
    from easyicu.research_agent.agents.progressive_attempt import progressive_planner_failure_facts

    monkeypatch.setattr(progressive_planner, "_MAX_COMPILE_REVISIONS", budget)
    context, payload, _ = _electrolyte_family_case(monkeypatch, missing="binary")
    with pytest.raises(ProgressivePlanCompileError) as caught:
        _run_electrolyte_family(context, [json.dumps(payload), *(["{"] * budget)])
    assert caught.value.reason_code == "progressive_required_binary_display_labels_missing"
    assert progressive_planner_failure_facts(caught.value).prompt_metrics["compile_revision_count"] == budget
    spent = caught.value.easyicu_safe_diagnostic.get("metrics", {}).get("foundation_label_repair_revisions", 0)
    assert spent == budget


def _assert_only_requested_labels_changed(before, after, repair):
    expected = before.model_dump(mode="json")
    foundation = expected["foundation"]
    labels = {item["key"]: item["value"] for item in foundation.pop("display_labels")}
    observed = after.model_dump(mode="json")
    observed_labels = {item["key"]: item["value"]
                       for item in observed["foundation"].pop("display_labels")}
    assert observed == expected  # Includes outline digest and every other coordinate.
    assert observed_labels == {**labels, **repair["display_labels"]}


@pytest.mark.parametrize("missing", ["reader", "binary"])
def test_family_label_repair_has_no_hidden_retry_or_other_coordinate_changes(monkeypatch, missing):
    context, payload, repair = _electrolyte_family_case(monkeypatch, missing=missing)
    forbidden = {**repair, "cohort": {"name": "unauthorized change"}}
    llm, result = _run_electrolyte_family(
        context, [json.dumps(payload), json.dumps(forbidden), json.dumps(repair)],
    )
    assert len(llm.calls) == 3
    assert result.facts.prompt_metrics["compile_revision_count"] == 2
    assert len(result.facts.prompt_metrics["foundation_label_repairs"]) == 1


def _ventilation_case():
    from easyicu.research_agent.schema import ResearchContext
    from .progressive_planner_fixtures import (
        _context as context_fixture, _foundation_payload, _materialization_payloads,
        _outline_payload as outline_fixture,
    )

    def rename(value):
        raw = json.dumps(value)
        for old, new in (("exposure_flag", "ventilation_flag"), ("outcome_flag", "rhythm_event"),
                         ("age_years", "magnesium_entry"), ("sex_code", "baseline_rhythm")):
            raw = raw.replace(old, new)
        return json.loads(raw)

    context = ResearchContext.model_validate(rename(context_fixture().model_dump(mode="json")))
    context = context.model_copy(update={"research_question": (
        "Estimate the association of entry ventilation and rhythm disturbances, "
        "adjusted for magnesium_entry and baseline_rhythm, with audit context."
    )})
    return context, rename(outline_fixture()), rename(_foundation_payload()), rename(_materialization_payloads())


@pytest.mark.parametrize("missing", ["reader", "binary"])
def test_fresh_foundation_parse_defers_label_findings_to_the_targeted_owner(monkeypatch, missing):
    from easyicu.research_agent.agents import progressive_foundation_repair as owner
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    context, outline, foundation, steps = _ventilation_case()
    keys = ("ventilation_flag",) if missing == "reader" else ("ventilation_flag=0", "ventilation_flag=1")
    repair = {"display_labels": {
        key: {"ventilation_flag": "Ventilatory support at entry", "ventilation_flag=0": "No entry ventilation",
              "ventilation_flag=1": "Entry ventilation"}[key] for key in keys
    }}
    foundation["foundation"]["display_labels"] = [
        label for label in foundation["foundation"]["display_labels"] if label["key"] not in keys
    ]
    candidates = []
    repair_owner = owner.repair_foundation_display_labels

    def observe(*args, **kwargs):
        candidates.append(kwargs["materialization"])
        return repair_owner(*args, **kwargs)

    monkeypatch.setattr(owner, "repair_foundation_display_labels", observe)
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [outline, foundation, repair, *steps]])
    result = ProgressivePlannerAgent(llm).run_attempt(context)
    assert len(llm.calls) == 10
    assert "TARGETED FOUNDATION DISPLAY-LABEL REPAIR" in llm.calls[2][0][-1].content
    assert result.facts.prompt_metrics["compile_revision_count"] == 1
    assert len(candidates) == 1
    _assert_only_requested_labels_changed(candidates[0], result.facts.foundation, repair)


@pytest.mark.parametrize("missing", ["reader", "binary"])
def test_fresh_foundation_label_repair_uses_the_configured_budget(monkeypatch, missing):
    from easyicu.research_agent.agents import progressive_planner
    from easyicu.research_agent.agents.progressive_attempt import progressive_planner_failure_facts
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    monkeypatch.setattr(progressive_planner, "_MAX_COMPILE_REVISIONS", 1)
    context, outline, foundation, _ = _ventilation_case()
    keys = ("ventilation_flag",) if missing == "reader" else ("ventilation_flag=0", "ventilation_flag=1")
    foundation["foundation"]["display_labels"] = [
        label for label in foundation["foundation"]["display_labels"] if label["key"] not in keys
    ]
    llm = ScriptedMockLLMClient([json.dumps(outline), json.dumps(foundation), "{"])
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run(context)
    assert caught.value.reason_code == f"progressive_required_{missing}_display_labels_missing"
    assert len(llm.calls) == 3
    assert progressive_planner_failure_facts(caught.value).prompt_metrics["compile_revision_count"] == 1


def test_host_step_label_exhaustion_cannot_fall_back_to_reasking_the_step(monkeypatch):
    from easyicu.research_agent.agents import progressive_planner
    from easyicu.research_agent.agents.progressive_attempt import progressive_planner_failure_facts
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    context, outline, foundation, steps = _ventilation_case()
    context = context.model_copy(update={"primary_exposure": None})
    foundation["foundation"]["display_labels"] = [
        label for label in foundation["foundation"]["display_labels"] if "=" not in label["key"]
    ]
    index = next(i for i, item in enumerate(steps) if item["step"]["module_id"] == "exposure_outcome_distribution")
    candidate = ProgressiveStepMaterialization.model_validate(steps[index])
    monkeypatch.setattr(progressive_planner, "llm_is_mockish", lambda _llm: False)
    monkeypatch.setattr(progressive_planner, "host_materialize_progressive_step", lambda **kwargs: (
        candidate if kwargs["outline_step"].step_id == candidate.step.step_id else None
    ))
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [outline, foundation, *steps[:index]]] + ["{", "{"])
    with pytest.raises(ProgressivePlanCompileError) as caught:
        ProgressivePlannerAgent(llm).run(context)
    assert caught.value.reason_code == "progressive_required_binary_display_labels_missing"
    assert len(llm.calls) == index + 4
    facts = progressive_planner_failure_facts(caught.value)
    assert len(facts.materializations) == index
    assert facts.prompt_metrics["compile_revision_count"] == 2


@pytest.mark.parametrize("malformed", [False, True])
def test_final_acceptance_routes_a_new_label_finding_without_replanning_steps(monkeypatch, malformed):
    from easyicu.research_agent.agents.progressive_attempt import progressive_planner_failure_facts
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.research_agent.schema import ConceptDescriptor

    context, outline, foundation, steps = _ventilation_case()
    context = context.model_copy(update={"variables": [*context.variables, ConceptDescriptor(
        name="sodium_entry", dtype="float64", role="lab", description="Serum sodium at ICU entry",
    )]})
    compile_and_accept = ProgressivePlannerAgent._compile_and_accept
    accepted_steps = []

    def final_gate(self, skeleton, **kwargs):
        accepted_steps.append(tuple(self._attempt.materializations))
        validate_progressive_foundation(
            ProgressivePlanFoundation(
                cohort=skeleton.cohort, display_labels=skeleton.display_labels,
                robustness_intents=skeleton.robustness_intents,
                know_how_decisions=skeleton.know_how_decisions,
            ), context=context, analysis_type=skeleton.analysis_type,
            required_reader_display_label_keys=("sodium_entry",),
        )
        return compile_and_accept(self, skeleton, **kwargs)

    monkeypatch.setattr(ProgressivePlannerAgent, "_compile_and_accept", final_gate)
    repair = {"display_labels": {"sodium_entry": "Serum sodium at entry"}}
    responses = [json.dumps(p) for p in [outline, foundation, *steps]]
    llm = ScriptedMockLLMClient([*responses, *(["{", "{"] if malformed else [json.dumps(repair)])])
    if malformed:
        with pytest.raises(ProgressivePlanCompileError) as caught:
            ProgressivePlannerAgent(llm).run(context)
        assert caught.value.reason_code == "progressive_required_reader_display_labels_missing"
        facts = progressive_planner_failure_facts(caught.value)
        assert facts.prompt_metrics["compile_revision_count"] == 2
        assert len(llm.calls) == 11 and len(accepted_steps) == 1
        assert facts.materializations == accepted_steps[0]
        return
    result = ProgressivePlannerAgent(llm).run_attempt(context)
    assert len(llm.calls) == 10
    assert result.output.display_labels["sodium_entry"] == "Serum sodium at entry"
    assert len(accepted_steps) == 2
    assert result.facts.materializations == accepted_steps[0] == accepted_steps[1]
    assert result.facts.prompt_metrics["compile_revision_count"] == 1


@pytest.mark.parametrize("malformed_first", [False, True])
def test_family_reader_and_binary_findings_share_the_remaining_revision_budget(monkeypatch, malformed_first):
    from easyicu.research_agent.agents.progressive_attempt import progressive_planner_failure_facts

    context, payload, reader = _electrolyte_family_case(monkeypatch, missing="reader")
    binary = {"display_labels": {item["key"]: item["value"] for item in payload["reader_display_labels"]
                                 if "=" in item["key"]}}
    payload["reader_display_labels"] = [item for item in payload["reader_display_labels"] if "=" not in item["key"]]
    responses = [json.dumps(payload), *(["{"] if malformed_first else []), json.dumps(reader)]
    if malformed_first:
        with pytest.raises(ProgressivePlanCompileError) as caught:
            _run_electrolyte_family(context, responses)
        assert caught.value.reason_code == "progressive_required_binary_display_labels_missing"
        facts = progressive_planner_failure_facts(caught.value)
    else:
        llm, result = _run_electrolyte_family(context, [*responses, json.dumps(binary)])
        assert len(llm.calls) == 3
        facts = result.facts
    assert facts.prompt_metrics["compile_revision_count"] == 2
    assert dict((label.key, label.value) for label in facts.foundation.foundation.display_labels)[
        "hypokalemia_flag"
    ] == reader["display_labels"]["hypokalemia_flag"]


def test_family_targeted_repair_does_not_use_the_whole_spec_retry_journal(monkeypatch, tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.research_agent.planning import outline_retry_journal
    from easyicu.webserver.provider_adapter import web_research_agent_hard_stop_limits

    context, payload, repair = _electrolyte_family_case(monkeypatch, missing="binary")
    monkeypatch.setattr(outline_retry_journal.state_paths, "state_root", lambda: tmp_path / "state")
    ledger = ProviderHardStopLedger(
        path=tmp_path / "budget.json", task_ids=("synthetic",),
        limits=web_research_agent_hard_stop_limits("planner_canary"),
        batch_id="synthetic", declaration_sha256="c" * 64,
    )
    llm, result = _run_electrolyte_family(
        context, [json.dumps(payload), json.dumps(repair)],
        resume_dependency_context={"cohort_file_sha256": "a" * 64,
                                   "llm_signature": "mock:electrolyte", "prompt_version": "synthetic"},
        outline_retry_journal={"task": ledger.start_task("synthetic"),
                               "run_dir": tmp_path, "literature_sha256": "b" * 64},
    )
    journal = outline_retry_journal.load_retry_journal(tmp_path / ".runtime" / outline_retry_journal.FILE_NAME)
    assert len(journal["records"]) == 1 and len(llm.calls) == 2
    assert result.facts.prompt_metrics["compile_revision_count"] == 1


def test_every_planning_foundation_validation_route_reaches_the_label_owner():
    """Discover call sites from tracked source, so a new route requires coverage.

    Pure compiler entry points have no Provider. Their planning adapters must
    protect both prefix and final compilation; projection parsers defer only
    label findings and fully validate before publishing their candidate.
    """
    import ast
    import subprocess
    from pathlib import Path

    from easyicu.research_agent.agents import family_spec_planner, progressive_foundation_repair

    root = Path(__file__).resolve().parents[3]
    files = subprocess.check_output([
        "git", "-C", str(root), "ls-files", "--",
        ":!*_restricted*", ":!*D3.3*", ":!*D3.4*", ":!*S166b*",
    ], text=True).splitlines()
    functions = {}

    class Discover(ast.NodeVisitor):
        def __init__(self, module):
            self.module, self.stack = module, []

        def visit_FunctionDef(self, node):
            if not self.stack:
                functions[self.module, node.name] = node
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

    for path in files:
        if path.startswith("src/easyicu/research_agent/") and path.endswith(".py"):
            Discover(path.removeprefix("src/easyicu/research_agent/")).visit(ast.parse((root / path).read_text()))

    def names(node):
        return {item.func.id if isinstance(item.func, ast.Name) else item.func.attr
                for item in ast.walk(node) if isinstance(item, ast.Call)
                and isinstance(item.func, (ast.Name, ast.Attribute))}

    validator = validate_progressive_foundation.__name__
    discovered = {key for key, node in functions.items() if validator in names(node)}
    assert discovered == {
        ("agents/family_spec_planner.py", family_spec_planner.run_family_spec_attempt.__name__),
        ("agents/progressive_foundation_repair.py", progressive_foundation_repair.validate_projected_foundation.__name__),
        ("agents/progressive_foundation_repair.py", progressive_foundation_repair.repair_foundation_display_labels.__name__),
        ("planning/progressive_compiler.py", "compile_progressive_plan"),
    }
    compiler_callers = {key for key, node in functions.items()
                        if names(node) & {"compile_progressive_plan", "compile_progressive_prefix"}}
    assert compiler_callers == {
        ("agents/family_spec_planner.py", family_spec_planner.run_family_spec_attempt.__name__),
        ("agents/progressive_planner.py", ProgressivePlannerAgent._compile_and_accept.__name__),
        ("agents/progressive_planner.py", ProgressivePlannerAgent._materialize_remaining_steps.__name__),
        ("agents/progressive_planner.py", ProgressivePlannerAgent._accept_materialized_plan.__name__),
        ("planning/progressive_resume.py", "compile_progressive_prefix"),
        ("planning/progressive_resume.py", "restore_progressive_resume_prefix"),
    }
    routes = {
        ("agents/family_spec_planner.py", family_spec_planner.run_family_spec_attempt.__name__): {
            "defer_foundation_label_repair", "run_with_foundation_label_repair",
        },
        ("agents/progressive_foundation_repair.py", "repair_projected_foundation"): {
            "validate_projected_foundation", "run_with_foundation_label_repair",
        },
        ("agents/progressive_planner.py", ProgressivePlannerAgent._run_output.__name__): {
            "defer_foundation_label_repair", "repair_projected_foundation", "restore_progressive_resume_prefix",
        },
        ("agents/progressive_planner.py", ProgressivePlannerAgent._materialize_remaining_steps.__name__): {
            "run_with_foundation_label_repair", "compile_progressive_prefix",
        },
        ("agents/progressive_planner.py", ProgressivePlannerAgent._accept_materialized_plan.__name__): {
            "run_with_foundation_label_repair", "assemble_progressive_skeleton",
        },
    }
    for route, required in routes.items():
        assert required <= names(functions[route]), route
    assert {key for key, node in functions.items() if "restore_progressive_resume_prefix" in names(node)} == {
        ("agents/progressive_planner.py", ProgressivePlannerAgent._run_output.__name__),
    }
    # The targeted call itself still uses the same owner and bounded parser.
    owner = functions["agents/progressive_foundation_repair.py", "run_with_foundation_label_repair"]
    assert "repair_foundation_display_labels" in names(owner)


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
    from tests.support.progressive_retry import _RecordingEvidence

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

    assert caught.value.reason_code == "progressive_required_binary_display_labels_missing"
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
        assert caught.value.reason_code == "progressive_required_reader_display_labels_missing"
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
    assert caught.value.reason_code == "progressive_required_binary_display_labels_missing"
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

"""The endpoint must be declared once, and every consumer must read that copy.

MEASURED over 291 recorded runs of the nine canonical tasks:
``research_context.endpoint`` was ``null`` in **every single one**.

``EndpointSpec`` already existed, with ``time_column``, ``time_origin``,
``censoring_rule`` and a validator whose own docstring says none of them may be
inferred from a column-name suffix or a dtype. ``ResearchContextV2`` already
verified the columns such a declaration names against the sealed cohort. Both
were dead: ``build_research_context`` took an ``endpoint=`` parameter and no
caller in the package ever passed one.

What filled the vacuum, measured on ``h1_ventilation_survival``:

* the follow-up rule survived only as prose in one step's ``icu_rule_refs``,
  written in **3 of 13** plans and absent from the other 10;
* across the **11** runs with recovered generated source, the code reached for
  **seven distinct** combinations of ``{los_icu, los_hosp, death_time,
  discharge_time, END_HOURS}``;
* the concept auditor blocked steps for using "los_hosp instead of the
  contract-required ICU discharge censoring represented by ``los_icu``" -- and
  ``los_icu`` appears in **none** of that task's 13 analysis plans, while the
  plans that stated a rule stated *hospital* discharge. Two runs of the same
  task were blocked for opposite choices.

12 of the 29 scientific blocking findings on the five never-passing tasks are
this one missing declaration: censoring-variable confusion (6), a missing
``death_time`` silently read as a non-event (3), prevalent exposure not excluded
at the landmark (2), and an administrative horizon 24 hours past the plan's (1).

Neither side of that comparison was reading a declaration, because there was
none to read. These tests hold the three places a declaration has to reach: the
plan that makes it, the record the generated script opens, and the auditor that
judges the script.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.audits.validators import _concept_audit_endpoint_block
from easyicu.research_agent.authority.typed_binding import (
    _write_resolved_inputs_manifest,
    study_endpoint_declaration_entry,
)
from easyicu.research_agent.plan_utils import endpoint_contract_findings
from easyicu.research_agent.planning.analysis_types import (
    list_analysis_types,
    required_endpoint_kind_for_family,
)
from easyicu.research_agent.schema import AnalysisPlan, EndpointSpec


def _endpoint(**overrides: object) -> EndpointSpec:
    payload: dict = {
        "name": "death",
        "kind": "time_to_event",
        "absence_semantics": "no_absent_rows",
        "levels": [0, 1],
        "event_column": "death",
        "time_column": "followup_hours",
        "time_origin": "icu_admission_hour_24",
        "censoring_rule": (
            "hospital discharge (los_hosp) or 672 hours from ICU admission"
        ),
    }
    payload.update(overrides)
    return EndpointSpec(**payload)  # type: ignore[arg-type]


def _plan(**overrides: object) -> AnalysisPlan:
    payload: dict = {
        "research_question": "q",
        "analysis_type": "survival",
        "steps": [],
    }
    payload.update(overrides)
    return AnalysisPlan(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 1. The requirement is compiled once, on the family the planner stamps.
# --------------------------------------------------------------------------


def test_the_survival_family_requires_a_time_to_event_endpoint() -> None:
    assert required_endpoint_kind_for_family("survival") == "time_to_event"


def test_the_requirement_is_read_from_the_family_not_from_question_prose() -> None:
    """Keyed on the stamped family, so it cannot disagree with itself.

    The routing layer that maps a question to a family is 1,395 lines of trigger
    terms. Once the family is stamped on the plan it is a declaration; a second
    keyword pass to re-derive what the family already implies is how that layer
    grew. Passing question prose here must not produce a requirement.
    """

    assert required_endpoint_kind_for_family(
        "Estimate the association between mechanical ventilation and 28-day "
        "mortality with time-to-event methods"
    ) is None


def test_an_unknown_or_absent_family_carries_no_requirement() -> None:
    """A plan whose family did not resolve is a different defect.

    Answering this question for it would be the guess the type exists to end.
    """

    assert required_endpoint_kind_for_family(None) is None
    assert required_endpoint_kind_for_family("") is None
    assert required_endpoint_kind_for_family("not_a_family") is None


def test_every_declared_requirement_names_a_real_endpoint_kind() -> None:
    """A registry entry requiring an unconstructible kind would block forever."""

    valid = set(EndpointSpec.model_fields["kind"].annotation.__args__)  # type: ignore[union-attr]
    for spec in list_analysis_types():
        if spec.required_endpoint_kind is not None:
            assert spec.required_endpoint_kind in valid, spec.key


# --------------------------------------------------------------------------
# 2. The plan-level check: reachable on the real corpus, and satisfiable.
# --------------------------------------------------------------------------


def test_a_survival_plan_without_an_endpoint_is_reported() -> None:
    findings = endpoint_contract_findings(_plan())
    assert [(f.validator, f.severity) for f in findings] == [
        ("endpoint_contract", "warning")
    ]
    detail = findings[0].detail or {}
    assert detail["required_endpoint_kind"] == "time_to_event"
    assert detail["declared_endpoint_kind"] is None


def test_the_missing_declaration_does_not_abort_the_run() -> None:
    """The plan-phase finding remains retryable rather than aborting early.

    All 11 recorded runs carrying an error-severity plan-stage finding stopped
    with ``completed_step_count: 1`` and ``failed_steps: []`` -- aborted between
    the first and second step. An error here would replace h1's death at step 4
    of 12 with a death at step 0 whenever the Planner missed twice.

    The execute-phase gate gets the final say after the directed retry; this
    advisory finding must not pre-empt that repair opportunity.
    """

    assert all(f.severity != "error" for f in endpoint_contract_findings(_plan()))


def test_the_same_contract_can_be_enforced_after_retries_are_exhausted() -> None:
    """One rule, two lifecycle severities; no duplicated endpoint policy."""

    findings = endpoint_contract_findings(_plan(), severity="error")

    assert [(finding.validator, finding.severity) for finding in findings] == [
        ("endpoint_contract", "error")
    ]


def test_the_refusal_names_the_fields_that_would_satisfy_it() -> None:
    """A block whose remedy is unstated is a dead step, not a gate.

    Every field the planner must send has to appear in the message it receives.
    """

    message = endpoint_contract_findings(_plan())[0].message
    for field in (
        "kind",
        "levels",
        "event_column",
        "time_column",
        "time_origin",
        "censoring_rule",
    ):
        assert field in message, field


def test_a_declared_endpoint_of_the_required_kind_is_accepted() -> None:
    """Satisfiability, stated as a test: the gate has to have an exit."""

    assert endpoint_contract_findings(_plan(endpoint=_endpoint())) == []


def test_a_declared_endpoint_of_the_wrong_kind_is_refused_and_says_which() -> None:
    plan = _plan(
        endpoint=EndpointSpec(
            name="death",
            kind="binary",
            absence_semantics="no_absent_rows",
            levels=[0, 1],
        )
    )
    findings = endpoint_contract_findings(plan)
    assert len(findings) == 1
    assert (findings[0].detail or {})["declared_endpoint_kind"] == "binary"
    assert "'binary'" in findings[0].message


@pytest.mark.parametrize(
    "family",
    ["association_study", "causal_inference", "trajectory_clustering", "prediction_model"],
)
def test_families_with_no_declared_requirement_are_untouched(family: str) -> None:
    """The measured collateral surface: zero.

    Replayed over all 240 real plans, the check fires on exactly the 13
    ``survival`` plans of h1 and on nothing else. Four of the nine tasks
    currently produce verified manuscripts and all four are in these families;
    a check that also blocked them would trade a fix for a regression.
    """

    assert endpoint_contract_findings(_plan(analysis_type=family)) == []


# --------------------------------------------------------------------------
# 3. The record the generated script actually opens.
# --------------------------------------------------------------------------


def test_the_declaration_reaches_the_step_record(tmp_path: Path) -> None:
    """`resolved_inputs` is the file the script opens and hash-verifies.

    The prompt is where the losing copy already lived: a step whose prompt
    carried a MANDATORY paragraph about a table still wrote
    ``trajectory = pd.DataFrame()`` because its typed record did not name it.
    """

    manifest = _write_resolved_inputs_manifest(
        run_dir=tmp_path,
        step_id="04_primary_survival",
        planner_declared_inputs=[],
        bindings={},
        study_endpoint=study_endpoint_declaration_entry(_endpoint()),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    declared = payload["study_endpoint"]
    assert declared["schema_version"] == "easyicu.study_endpoint_declaration/1"
    assert declared["time_column"] == "followup_hours"
    assert declared["time_origin"] == "icu_admission_hour_24"
    assert "los_hosp" in declared["censoring_rule"]
    assert declared["levels"] == [0, 1]


def test_the_record_is_absent_when_nothing_was_declared(tmp_path: Path) -> None:
    """No key at all, rather than a key promising a declaration it lacks.

    An entry asserting completeness while publishing nothing is worse than no
    entry: it tells the agent, with the host's authority, that there is nothing
    to read. That exact shape shipped once this week and was reverted.
    """

    manifest = _write_resolved_inputs_manifest(
        run_dir=tmp_path,
        step_id="04_primary_survival",
        planner_declared_inputs=[],
        bindings={},
        study_endpoint=study_endpoint_declaration_entry(None),
    )
    assert "study_endpoint" not in json.loads(manifest.read_text(encoding="utf-8"))


def test_the_record_carries_its_own_authorization_text() -> None:
    """The record has to say what obeying it means, because the prompt cannot.

    The Coder prompt is 152 bytes from its hard budget on the widest measured
    step, so a paragraph there would evict typed context to restate what the
    record already holds.
    """

    entry = study_endpoint_declaration_entry(_endpoint())
    assert entry is not None
    authorization = entry["authorization"]
    assert "DECLARED" in authorization
    # The consequence, not just the instruction.
    assert "different study" in authorization
    # And the fail-closed exit for a field this step needs and does not have.
    assert "stop and report" in authorization


def test_a_time_axis_without_its_origin_is_refused(tmp_path: Path) -> None:
    """A duration and a timestamp are indistinguishable by dtype.

    This is the defect the type was built for: "a time column was used with no
    declared origin, so negative event times and events before time zero passed
    through."
    """

    with pytest.raises(ValueError, match="time_origin"):
        _write_resolved_inputs_manifest(
            run_dir=tmp_path,
            step_id="s",
            planner_declared_inputs=[],
            bindings={},
            study_endpoint={"name": "death", "kind": "time_to_event", "time_column": "t"},
        )


def test_a_declaration_without_a_kind_is_refused(tmp_path: Path) -> None:
    """Without the kind, the reader is back to inferring one from the name."""

    with pytest.raises(ValueError, match="kind"):
        _write_resolved_inputs_manifest(
            run_dir=tmp_path,
            step_id="s",
            planner_declared_inputs=[],
            bindings={},
            study_endpoint={"name": "death"},
        )


# --------------------------------------------------------------------------
# 4. The auditor that blocked steps against a rule nobody wrote.
# --------------------------------------------------------------------------


def test_the_auditor_is_told_the_declaration_is_authoritative() -> None:
    block = _concept_audit_endpoint_block(
        study_endpoint_declaration_entry(_endpoint())
    )
    assert "AUTHORITATIVE" in block
    # Naming the alternative it must stop consulting is the point: the blocked
    # runs were judged against the auditor's own reading of the question.
    assert "not the research question" in block
    # And the standard it must apply instead of its own study design.
    assert "even if you would have designed the study differently" in block
    assert "report a mismatch only against a field printed above" in block


def test_the_declared_fields_are_printed_for_the_auditor() -> None:
    """A block asserting authority without showing the fields authorises nothing."""

    block = _concept_audit_endpoint_block(
        study_endpoint_declaration_entry(_endpoint())
    )
    for value in (
        "followup_hours",
        "icu_admission_hour_24",
        "los_hosp",
        "time_to_event",
    ):
        assert value in block, value


def test_an_absent_declaration_is_stated_rather_than_omitted() -> None:
    """Rendering nothing is what let the auditor supply the missing rule.

    Silence in the prompt is indistinguishable from "use your judgement", and
    the judgement differed run to run: the same task was blocked once for using
    hospital discharge and once for not using ICU discharge.
    """

    block = _concept_audit_endpoint_block(None)
    assert "NONE" in block
    assert "Do not supply one from the research question" in block
    # Whose defect it is, so the finding lands on the plan and not the script.
    assert "the plan's defect" in block


def _rendered_audit_prompt(study_endpoint: object) -> str:
    """The prompt the auditor actually sends, built by calling the real method.

    Asserting over ``_concept_audit_endpoint_block`` alone does NOT hold this
    property, and a mutation proved it: deleting the one line that concatenates
    the block into the prompt left every helper-level assertion green. The
    defect spans two places -- the text, and its delivery -- so the check has to
    span both.
    """

    from easyicu.research_agent.audits.validators import LLMConceptAuditor
    from easyicu.research_agent.schema import (
        AnalysisStep,
        CohortDescriptor,
        ResearchContext,
    )

    context = ResearchContext(
        research_question=(
            "Estimate the association between mechanical ventilation and 28-day "
            "mortality with time-to-event methods."
        ),
        cohort=CohortDescriptor(
            cohort_name="c", database="synthetic", n_stays=10, n_patients=10
        ),
        variables=[],
        target_outcome="death",
    )
    step = AnalysisStep(
        step_id="04_primary_survival",
        intent="fit the primary model",
        inputs=[],
        expected_outputs=["table:primary_estimates"],
        method="survival_model",
    )
    return LLMConceptAuditor(llm=None)._prompt(  # type: ignore[arg-type]
        context=context,
        script_text="print(1)",
        step=step,
        study_endpoint=study_endpoint,  # type: ignore[arg-type]
    )


def test_the_declaration_is_delivered_in_the_prompt_the_auditor_sends() -> None:
    prompt = _rendered_audit_prompt(study_endpoint_declaration_entry(_endpoint()))
    assert "AUTHORITATIVE" in prompt
    assert "followup_hours" in prompt
    assert "icu_admission_hour_24" in prompt


def test_the_absent_case_is_delivered_too() -> None:
    """The silence that let the auditor invent a rule was in the PROMPT."""

    prompt = _rendered_audit_prompt(None)
    assert "Declared study endpoint: NONE" in prompt
    assert "Do not supply one from the research question" in prompt


def test_the_declaration_precedes_the_script_it_judges() -> None:
    """Ordering, because a rule stated after the evidence reads as a footnote."""

    prompt = _rendered_audit_prompt(study_endpoint_declaration_entry(_endpoint()))
    assert prompt.index("Declared study endpoint") < prompt.index("\nScript:\n")


# --------------------------------------------------------------------------
# 5. The wiring: the declaration has to come from the PLAN, at both consumers.
# --------------------------------------------------------------------------


def _phase_endpoint_call_arguments() -> list[str]:
    """Every expression `phase.py` passes to the declaration builder.

    Located structurally. The tests above all call the builder themselves, so
    they hold the text and the record but say nothing about where the value
    comes from -- unwiring both call sites in the execute phase leaves them
    green while no run ever publishes a declaration. Rendered back to source so
    a call reading some other object fails loudly rather than matching a
    substring.
    """

    import ast
    import inspect

    from easyicu.research_agent.execution import phase

    tree = ast.parse(inspect.getsource(phase))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "study_endpoint_declaration_entry":
            continue
        assert len(node.args) == 1, "the builder takes exactly one endpoint"
        found.append(ast.unparse(node.args[0]))
    return found


def test_both_consumers_are_wired_to_the_plans_own_endpoint() -> None:
    """The step record and the concept auditor, from one source.

    Two consumers rather than one is the point: the record tells the Coder what
    to implement and the auditor judges what it implemented. A run where only
    one of them was wired would be a run where the two disagree again, which is
    the state this whole change exists to leave.
    """

    arguments = _phase_endpoint_call_arguments()
    assert len(arguments) == 2, arguments
    for argument in arguments:
        # The locked plan -- not the context (built before the plan exists and
        # sealed as evidence, so it can never carry this), and not the step.
        assert "plan_result.plan" in argument, argument
        assert "'endpoint'" in argument or '"endpoint"' in argument, argument


def test_the_authorization_prose_is_not_repeated_to_the_auditor() -> None:
    """The auditor gets the fields; the authorization is addressed to the Coder.

    Two copies of the same paragraph in one prompt is the accretion this
    prompt already carries too much of, and the auditor is not the party being
    authorized to read anything.
    """

    entry = study_endpoint_declaration_entry(_endpoint())
    assert entry is not None
    block = _concept_audit_endpoint_block(entry)
    assert entry["authorization"] not in block
    assert "schema_version" not in block


# --------------------------------------------------------------------------
# 6. Retry at the execution boundary, then fail closed before scientific work.
# --------------------------------------------------------------------------


def test_a_replan_candidate_without_the_required_endpoint_is_rejected() -> None:
    from easyicu.research_agent.planning.replan_gate import (
        replan_candidate_contract_findings,
    )
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    context = ResearchContext(
        research_question="Estimate time to death.",
        cohort=CohortDescriptor(
            cohort_name="c",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
        target_outcome="death",
    )

    findings = replan_candidate_contract_findings(plan=_plan(), context=context)

    assert any(
        finding.validator == "endpoint_contract" and finding.severity == "error"
        for finding in findings
    )


def test_missing_endpoint_is_blocked_after_planner_and_replanner_miss(
    ra,
    synthetic_cohort,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise the real plan -> retry -> execute-preflight -> manifest path."""

    from easyicu.research_agent.agents.core import PlannerAgent
    from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient
    from easyicu.research_agent.schema import AnalysisStep

    step = AnalysisStep(
        step_id="02_primary_survival",
        planned_analysis_role="primary",
        intent="Estimate time to death.",
        inputs=[],
        expected_outputs=["table:survival_estimate"],
        method="survival_model",
    )
    missing_endpoint_plan = _plan(steps=[step])
    llm = PatternScriptedMockLLMClient(
        [
            (
                "ICU-AWARE RESEARCH PLAN",
                [missing_endpoint_plan.model_dump_json(indent=2)] * 6,
            ),
            (
                "PROBE SUMMARY:",
                [missing_endpoint_plan.model_dump_json(indent=2)] * 6,
            ),
        ],
        contextual_default=True,
    )
    original_planner_run = PlannerAgent.run

    def run_without_article_suite(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_planner_run(self, context, **kwargs)

    monkeypatch.setattr(PlannerAgent, "run", run_without_article_suite)
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=llm)

    result = pipeline.run(
        question="Estimate time to death.",
        cohort=synthetic_cohort,
        cohort_name="endpoint_contract_fail_closed",
        database="synthetic",
        target_outcome="death",
    )
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    prompts = [
        "\n".join(str(message.content or "") for message in messages)
        for messages, _kwargs in llm.calls
    ]

    assert any(
        "PROBE SUMMARY:" in prompt and "Repair the plan's typed study endpoint" in prompt
        for prompt in prompts
    )
    assert any(
        finding.get("validator") == "endpoint_contract"
        and finding.get("severity") == "error"
        and (finding.get("detail") or {}).get("stage") == "execute_final"
        for finding in manifest["findings"]
    )
    assert manifest["readiness"]["analysis_validated"] is False
    assert manifest["readiness"]["manuscript_ready"] is False
    assert not any(
        record.get("step_id") == step.step_id
        for record in manifest.get("per_step_records", [])
    )
    audit_text = (Path(result.manifest_path).parent / "audit_log.jsonl").read_text(
        encoding="utf-8"
    )
    assert "endpoint_contract_blocked" in audit_text

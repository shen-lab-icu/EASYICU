from __future__ import annotations

from easyicu.research_agent.authority.plan_input_closure import (
    close_measurement_companion_inputs,
    plan_manifest_fields,
    register_measurement_companion_input_closure,
    resolve_registered_plan_authority,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.table_one_binding import (
    bind_table_one_execution_spec,
    restore_table_one_private_checkpoint,
    table_one_private_code_label_map,
    write_table_one_private_checkpoint,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
    ConceptDescriptor,
)


def test_plan_adds_only_existing_exact_measurement_companions():
    context = ResearchContext(
        research_question="Audit selected ICU measurements.",
        cohort=CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[
            ConceptDescriptor(name="signal_first", dtype="float64"),
            ConceptDescriptor(name="signal_measured", dtype="int64"),
            ConceptDescriptor(name="signal_n", dtype="int64"),
            ConceptDescriptor(name="other_first", dtype="float64"),
            ConceptDescriptor(name="other_n", dtype="int64"),
            ConceptDescriptor(name="unrelated_measured", dtype="int64"),
        ],
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        steps=[
            AnalysisStep(
                step_id="quality",
                intent="Audit the selected summaries.",
                method="data_quality_audit",
                inputs=["artifact:cohort", "signal_first", "other_first"],
                expected_outputs=["table:quality"],
            )
        ],
    )

    revised, findings = close_measurement_companion_inputs(
        plan=plan,
        context=context,
    )

    assert revised.steps[0].inputs == [
        "artifact:cohort",
        "signal_first",
        "other_first",
        "signal_measured",
        "signal_n",
        "other_n",
    ]
    assert "unrelated_measured" not in revised.steps[0].inputs
    assert findings[0].detail["reason"] == "measurement_companion_input_closure"


def test_plan_companion_closure_is_idempotent():
    context = ResearchContext(
        research_question="Audit selected ICU measurements.",
        cohort=CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[
            ConceptDescriptor(name="signal_first", dtype="float64"),
            ConceptDescriptor(name="signal_measured", dtype="int64"),
            ConceptDescriptor(name="signal_n", dtype="int64"),
        ],
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        steps=[
            AnalysisStep(
                step_id="quality",
                intent="Audit the selected summary.",
                method="data_quality_audit",
                inputs=["signal_first", "signal_measured", "signal_n"],
                expected_outputs=["table:quality"],
            )
        ],
    )

    revised, findings = close_measurement_companion_inputs(
        plan=plan,
        context=context,
    )

    assert revised == plan
    assert findings == []


def test_plan_closes_explicit_measured_or_count_inputs_to_registered_pair():
    context = ResearchContext(
        research_question="Audit selected ICU measurement provenance.",
        cohort=CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[
            ConceptDescriptor(name="sofa_measured", dtype="int64"),
            ConceptDescriptor(name="sofa_n", dtype="int64"),
            ConceptDescriptor(name="infection_measured", dtype="int64"),
            ConceptDescriptor(name="infection_n", dtype="int64"),
        ],
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        steps=[
            AnalysisStep(
                step_id="quality",
                intent="Audit the selected provenance fields.",
                method="missingness_and_measurement_audit",
                inputs=["sofa_measured", "infection_n"],
                expected_outputs=["table:missingness_audit"],
            )
        ],
    )

    revised, findings = close_measurement_companion_inputs(
        plan=plan,
        context=context,
    )

    assert revised.steps[0].inputs == [
        "sofa_measured",
        "infection_n",
        "sofa_n",
        "infection_measured",
    ]
    assert findings[0].detail["added_inputs_by_step"] == {
        "quality": ["sofa_n", "infection_measured"]
    }
    repeated, repeated_findings = close_measurement_companion_inputs(
        plan=revised,
        context=context,
    )
    assert repeated == revised
    assert repeated_findings == []


def test_measurement_closure_preserves_table_one_private_checkpoint(tmp_path):
    context = ResearchContext(
        research_question="Describe the cohort and audit measurement provenance.",
        cohort=CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[
            ConceptDescriptor(
                name="sex",
                dtype="object",
                observed_domain={
                    "n_unique": 2,
                    "levels": ["Female", "Male"],
                },
            ),
            ConceptDescriptor(name="signal_first", dtype="float64"),
            ConceptDescriptor(name="signal_n", dtype="int64"),
        ],
    )
    step = AnalysisStep.model_validate(
        {
            "step_id": "table_one",
            "intent": "Build Table 1 and report measurement provenance.",
            "method": "table_one",
            "inputs": ["sex", "signal_first"],
            "expected_outputs": ["table:table_one"],
            "table_one_spec": {
                "group_by": "sex",
                "group_levels": [
                    "__easyicu_level_1__",
                    "__easyicu_level_2__",
                ],
                "variables": [
                    {
                        "name": "signal_first",
                        "variable_kind": "continuous",
                        "summary": "median_iqr",
                        "test": "mann_whitney_or_kruskal",
                    }
                ],
            },
        }
    )
    assert bind_table_one_execution_spec(step, context) is not None
    original_tokens = table_one_private_code_label_map(step)
    plan = AnalysisPlan(
        research_question=context.research_question,
        steps=[step],
    )
    write_table_one_private_checkpoint(run_dir=tmp_path, plan=plan)

    resumed_context = ResearchContext.model_validate(context.model_dump(mode="json"))
    resumed_plan = AnalysisPlan.model_validate(plan.model_dump(mode="json"))
    revised, findings = close_measurement_companion_inputs(
        plan=resumed_plan,
        context=resumed_context,
    )

    assert revised.steps[0].inputs == ["sex", "signal_first", "signal_n"]
    assert findings
    assert resumed_context._table_one_token_secrets == {}
    assert revised.steps[0]._table_one_execution_binding is None

    restore_table_one_private_checkpoint(
        run_dir=tmp_path,
        plan=revised,
        context=resumed_context,
    )
    assert table_one_private_code_label_map(revised.steps[0]) == original_tokens


def test_current_plan_authority_selects_immutable_closure_evidence(tmp_path):
    context = ResearchContext(
        research_question="Audit one measurement.",
        cohort=CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[
            ConceptDescriptor(name="signal_first", dtype="float64"),
            ConceptDescriptor(name="signal_n", dtype="int64"),
        ],
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        revision=3,
        steps=[
            AnalysisStep(
                step_id="quality",
                intent="Audit the measurement.",
                method="data_quality_audit",
                inputs=["signal_first"],
                expected_outputs=["table:quality"],
            )
        ],
    )
    closed, findings = close_measurement_companion_inputs(plan=plan, context=context)
    assert findings
    evidence = EvidenceStore(tmp_path)
    registered = register_measurement_companion_input_closure(
        run_dir=tmp_path,
        evidence=evidence,
        plan=closed,
        prompt_pack_version="test/v1",
    )
    # Execution attaches host-only private bindings after the public plan has
    # been serialized. They are not scientific plan content and must not make
    # the immutable public authority appear stale.
    closed.steps[0]._table_one_execution_binding = object()

    authority = resolve_registered_plan_authority(
        run_dir=tmp_path,
        evidence=evidence,
        plan=closed,
        plan_path=registered.evidence_path,
    )

    assert authority.evidence_id == registered.evidence_id
    assert authority.revision == 3
    assert authority.relative_path.startswith("evidence/")
    assert authority.sha256 == evidence.get(registered.evidence_id).sha256
    assert plan_manifest_fields(
        tmp_path, evidence, closed, registered.evidence_path
    ) == {
        "plan_path": authority.relative_path,
        "current_plan_authority": authority.to_dict(),
    }

from __future__ import annotations

from easyicu.research_agent.plan_utils import _augment_measurement_companion_inputs
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

    revised, findings = _augment_measurement_companion_inputs(
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

    revised, findings = _augment_measurement_companion_inputs(
        plan=plan,
        context=context,
    )

    assert revised == plan
    assert findings == []

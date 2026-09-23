"""The claimed static prediction owner projects its executable repeat-stay rule."""

from easyicu.research_agent.planning.scientific_review import (
    _repeated_stay_rule_declared,
    build_plan_scientific_review,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

from .scientific_review_fixtures import _context


def _grouped_context():
    context = _context()
    return context.model_copy(
        update={
            "cohort": context.cohort.model_copy(
                update={
                    "id_columns": ["patient_stay_id"],
                    "provenance": {
                        "analysis_unit": "icu_stay",
                        "replacement_row_identity": {
                            "output_identity_column": "patient_stay_id",
                            "mapping_file_sha256": "a" * 64,
                            "patient_group_derivation": {
                                "algorithm": "prefix_before_:s",
                                "delimiter": ":s",
                            },
                        },
                    },
                }
            )
        }
    )


def _static_prediction_primary(**updates):
    step = AnalysisStep(
        step_id="primary_performance",
        planned_analysis_role="primary",
        intent="Fit the prespecified static model on patient-separated splits.",
        inputs=["death", "age", "exposure", "artifact:analysis_cohort"],
        expected_outputs=["table:prediction_scores", "table:model_performance"],
        method="prespecified_prediction_model_discrimination_calibration",
        scientific_action_id="prediction.discrimination_calibration",
    )
    return step.model_copy(update=updates) if updates else step


def test_claimed_prediction_primary_declares_the_patient_split_rule_only_with_group_authority():
    plan = AnalysisPlan(
        research_question="Predict death.",
        analysis_type="prediction_model",
        steps=[_static_prediction_primary()],
    )

    assert _repeated_stay_rule_declared(plan, _grouped_context())
    # Without verified patient grouping the owner fails closed at runtime, so
    # nothing is declared at plan time either.
    assert not _repeated_stay_rule_declared(plan, _context())
    # An unclaimed step of the same family declares nothing.
    unclaimed = plan.model_copy(
        update={"steps": [_static_prediction_primary(scientific_action_id=None)]}
    )
    assert not _repeated_stay_rule_declared(unclaimed, _grouped_context())


def test_review_does_not_ask_the_prediction_owner_to_restate_its_split_rule():
    context = _grouped_context()
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="prediction_model",
        steps=[_static_prediction_primary()],
    )

    review = build_plan_scientific_review(context=context, plan=plan)

    assert "REPEATED_STAY_DEDUP_UNDECLARED" not in {f.code for f in review.findings}
    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" not in {f.code for f in review.findings}

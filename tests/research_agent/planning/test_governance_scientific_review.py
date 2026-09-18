"""Governance findings for missingness and repeated ICU stays."""

from __future__ import annotations

import json

from easyicu.research_agent.planning.dependence_authority import (
    bind_context_dependence_authority,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import AnalysisPlan, UserPreferences

from .scientific_review_fixtures import (
    _context,
    _literature,
    _plan,
    _traditional_table_one_step,
)

def test_complete_case_everywhere_without_examination_is_flagged() -> None:
    """Governance mining: 35% of papers leave missingness unstated and 21%
    use single-value handling; complete-case throughout with no missing-data
    sensitivity must surface as a major finding, not pass silently."""

    base = _plan()
    primary = base.steps[0]
    requirement = primary.model_requirements[0].model_copy(
        update={"analysis_set": "complete_case"}
    )
    plan = base.model_copy(
        update={
            "steps": [
                primary.model_copy(update={"model_requirements": [requirement]}),
                *base.steps[1:],
            ]
        }
    )
    review = build_plan_scientific_review(
        context=_context(),
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(_context()),
    )
    by_code = {item.code: item for item in review.findings}
    assert by_code["MISSINGNESS_UNEXAMINED_COMPLETE_CASE"].severity == "major"

    review_default = build_plan_scientific_review(
        context=_context(),
        plan=base,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(_context()),
    )
    assert "MISSINGNESS_UNEXAMINED_COMPLETE_CASE" not in {
        item.code for item in review_default.findings
    }


def _repeats_context():
    return _context().model_copy(
        update={
            "cohort": _context().cohort.model_copy(
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
            ),
            "user_preferences": UserPreferences(
                covariates=["age"],
                data_constraints=json.dumps(
                    {
                        "analysis_design": {
                            "analysis_unit": "icu_stay",
                            "cluster_unit": "patient",
                            "variance_estimator": "cluster_robust",
                        }
                    }
                ),
            ),
        }
    )


def _review_codes(context, plan):
    review = build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )
    return {item.code: item for item in review.findings}


def test_bound_dependence_contract_silences_dedup_finding() -> None:
    """Integration: a bound dependence contract is an explicit rule.

    Governance mining: repeats + closed design + no declared rule must
    surface major; a bound contract, a first-stay method, or an executed
    repeated-stays sensitivity each clear it, exactly as the remediation
    promises.
    """

    context = _repeats_context()
    base = _plan()
    primary = base.steps[0]
    requirement = primary.model_requirements[0].model_copy(
        update={"method_family": "statsmodels_logit_mle"}
    )
    bound_plan = base.model_copy(
        update={
            "steps": [
                primary.model_copy(update={"model_requirements": [requirement]}),
                *base.steps[1:],
            ]
        }
    )
    bound = bind_context_dependence_authority(plan=bound_plan, context=context)
    assert bound.steps[0].model_requirements[0].dependence is not None
    assert "REPEATED_STAY_DEDUP_UNDECLARED" not in _review_codes(
        context, bound
    )

    restricted = bound.model_copy(
        update={
            "steps": [
                bound.steps[0].model_copy(
                    update={"method": "non_readmission_restriction"}
                ),
                *bound.steps[1:],
            ]
        }
    )
    assert "REPEATED_STAY_DEDUP_UNDECLARED" not in _review_codes(
        context, restricted
    )


def test_descriptive_table_one_with_repeats_demands_stated_unit() -> None:
    """Integration: the live firing path.

    A descriptive Table 1 over repeated stays closes the design (no
    p-values, nothing inferential) yet states no unit rule -- exactly the
    literature's 38%-unclear habit. The review must flag major so the
    report states stays-vs-patients explicitly.
    """

    context = _repeats_context()
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="descriptive_study",
        steps=[_traditional_table_one_step()],
    )
    bound = bind_context_dependence_authority(  # already imported at module top
        plan=plan, context=context
    )
    assert bound.steps[0].table_one_spec is not None
    by_code = _review_codes(context, bound)
    assert "REPEATED_STAY_METHOD_NOT_DECLARED" not in by_code
    assert by_code["REPEATED_STAY_DEDUP_UNDECLARED"].severity == "major"

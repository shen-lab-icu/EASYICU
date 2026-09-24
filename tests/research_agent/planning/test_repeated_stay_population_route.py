"""A plan with no dependence-bearing estimator hands repeated stays to the host.

Dev9 M3 (cross-sectional phenotypes, every ICU stay retained, patient
grouping bound) was refused with ``REPEATED_STAY_METHOD_NOT_DECLARED`` routed
to plan revision.  Its clustering steps fit one row per ICU stay and carry no
model, interval, or baseline-table contract, so the revision repeated the
finding and the review stopped as nonconvergent.  The fixtures here are the
synthetic phenotyping context, not that study.
"""

from __future__ import annotations

import json

from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    plan_revision_blocker_codes,
    repeated_unit_estimator_present,
)

from .family_spec_fixtures import _phenotyping_context, _phenotyping_payload, _run
from .scientific_review_fixtures import _literature, _plan


def _with_patients(context, *, first_stay_only: bool = False):
    """Stay rows keyed by a source identity whose prefix groups each patient."""

    provenance = {
        **context.cohort.provenance,
        "stay_id_columns": ["patient_stay_id"],
        "patient_identity_available": True,
        "replacement_row_identity": {
            "output_identity_column": "patient_stay_id",
            "mapping_file_sha256": "e" * 64,
            "patient_group_derivation": {"algorithm": "prefix_before_:s", "delimiter": ":s"},
        },
    }
    if first_stay_only:
        provenance["first_icu_stay_restriction"] = {
            "schema_version": "easyicu.first_icu_stay_restriction/1",
            "coordinate_sha256": "f" * 64,
        }
    return context.model_copy(
        update={
            "cohort": context.cohort.model_copy(
                update={
                    "id_columns": ["patient_stay_id"],
                    "n_patients": 900,
                    "n_stays": 1000,
                    "provenance": provenance,
                }
            ),
            "variables": [
                item.model_copy(update={"name": "patient_stay_id"}) if item.name == "stay_id" else item
                for item in context.variables
            ],
        }
    )


def _phenotyping_plan(context):
    _llm, result = _run(
        context,
        [json.dumps(_phenotyping_payload(
            _request_for(context), features=["hr_max", "lactate_max", "map_min"],
            baseline=["age", "sex"], membership="phenotype_flag",
        ))],
        required_primary_cohort_selection_mode=None,
    )
    return result.output


def _request_for(context):
    from easyicu.research_agent.agents.progressive_planner import (
        candidate_analysis_types,
        select_progressive_variables,
    )
    from easyicu.research_agent.planning.family_spec import build_family_spec_request

    from .family_spec_fixtures import ALLOWED_CITATIONS, DIRECT_COMPARATORS

    return build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
    )


def _review(context, plan):
    return build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )


def test_a_clustering_plan_hands_repeated_stays_to_the_host_not_the_planner() -> None:
    context = _with_patients(_phenotyping_context())
    plan = _phenotyping_plan(context)

    assert not repeated_unit_estimator_present(context, plan)
    review = _review(context, plan)
    finding = next(
        item for item in review.findings if item.code == "REPEATED_STAY_METHOD_NOT_DECLARED"
    )
    assert finding.severity == "blocker"
    assert finding.remediation_route == "runtime_capability"
    assert "first ICU stay" in finding.remediation
    # Another Planner turn cannot close it, so none is spent.
    assert "REPEATED_STAY_METHOD_NOT_DECLARED" in plan_revision_blocker_codes(review.findings)


def test_the_host_first_stay_population_closes_the_same_plan() -> None:
    context = _with_patients(_phenotyping_context(), first_stay_only=True)
    plan = _phenotyping_plan(context)

    codes = {item.code for item in _review(context, plan).findings}

    assert not {"REPEATED_STAY_METHOD_NOT_DECLARED", "REPEATED_STAY_IDENTITY_UNAVAILABLE"} & codes


def test_a_plan_with_a_model_still_revises_its_own_estimator() -> None:
    context = _with_patients(_phenotyping_context())

    assert repeated_unit_estimator_present(context, _plan())

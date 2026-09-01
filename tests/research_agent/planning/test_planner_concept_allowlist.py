"""Canaries for planner CTAS concept-id grounding.

The Phase-1 real-LLM pilot failed in plan generation when the planner
invented ``sofa2_admission`` as a concept_id. These tests pin the prompt
contract that prevents that failure mode from coming back.
"""

from __future__ import annotations

from easyicu.research_agent import schema
from easyicu.research_agent.agents.core import (
    PLANNER_MAX_RETRIES,
    _build_planner_user_prompt,
)
from easyicu.research_agent.cohort.schema import ALLOWED_CTAS_AGGREGATIONS


def _context(
    *,
    aggregation_default: schema.AggregationRule | None = None,
) -> schema.ResearchContext:
    return schema.ResearchContext(
        research_question="Estimate admission SOFA association with ICU mortality.",
        cohort=schema.CohortDescriptor(
            cohort_name="case_b",
            database="miiv",
            n_patients=200,
            n_stays=200,
        ),
        variables=[
            schema.ConceptDescriptor(
                name="sofa",
                role=schema.VariableRole.COMPOSITE_SCORE,
                dtype="float64",
                is_ordinal=True,
                aggregation_default=aggregation_default,
            ),
            schema.ConceptDescriptor(
                name="death",
                role=schema.VariableRole.OUTCOME,
                dtype="int64",
            ),
        ],
        target_outcome="death",
    )


def test_planner_prompt_contains_concept_allowlist() -> None:
    prompt = _build_planner_user_prompt(_context())

    assert "ALLOWED concept_ids" in prompt
    assert prompt.index("ALLOWED concept_ids") < prompt.index(
        "Every cohort/exposure/outcome concept"
    )
    assert _listed_ids(prompt) == ["death", "sofa"]


def _listed_ids(prompt: str) -> list[str]:
    return [
        line.strip()[3:-1]
        for line in prompt.splitlines()
        if line.strip().startswith("- `") and line.strip().endswith("`")
    ]


def test_the_allowlist_is_this_run_s_export_not_the_dictionary() -> None:
    """A menu of concepts this export does not carry costs planning attempts.

    MEASURED on canary12's E3 cohort (104 columns, 2026-07-31): the prompt
    published 264 concept ids as "the ONLY values acceptable" and 15 of them
    bound against the sealed input -- 94.3% of the menu was unusable. The
    Planner chose ``kdigo_aki``, the right concept for an AKI-stage cohort,
    from the list the host handed it; the binder refused it for having no bound
    column, and the next attempt improvised ``aki_stage``, which is not a
    concept at all. Two of five planning attempts spent on a wrong menu, and
    the run produced nothing.

    This test asserts the direction that matters: an id the dictionary defines
    but this context cannot bind is NOT offered. The count is checked against
    the whole dictionary so it cannot pass by the list being empty.
    """

    from easyicu.research_agent.cohort.schema import known_concept_ids

    listed = _listed_ids(_build_planner_user_prompt(_context()))

    assert listed, "an empty menu makes the cohort unwritable"
    assert set(listed) < set(known_concept_ids()), "the dictionary is not the menu"
    assert "kdigo_aki" not in listed
    assert "sofa" in listed and "death" in listed


def test_without_sealed_columns_the_dictionary_is_still_offered() -> None:
    """Narrowing must not become an empty menu when there is nothing to narrow by.

    The binder still refuses an unbound predicate downstream, so being
    permissive here costs a rejected plan; being empty here costs every plan.
    """

    from easyicu.research_agent.agents.core import _format_concept_id_allowlist
    from easyicu.research_agent.cohort.schema import known_concept_ids

    rendered = _format_concept_id_allowlist([])

    assert len(_listed_ids(rendered)) == len(known_concept_ids())
    assert "sealed columns were not" in rendered


def test_planner_prompt_forbids_concept_id_synthesis() -> None:
    prompt = _build_planner_user_prompt(_context())

    assert "Synthesizing new names" in prompt
    assert '"score_at_admission"' in prompt
    assert '"concept_peak_window"' in prompt
    assert '"condition_onset_window"' in prompt
    # Shared prompts stay benchmark-neutral; case-specific names belong in the
    # benchmark item or run protocol, not the global planner contract.
    assert '"sofa2_admission"' not in prompt
    assert '"kdigo_aki_max"' not in prompt
    assert '"sepsis_onset_window"' not in prompt


def test_planner_prompt_keeps_literature_eligibility_data_bound() -> None:
    prompt = _build_planner_user_prompt(_context())

    assert "design candidates, not automatic authority" in prompt
    assert "unverifiable literature criterion" in prompt
    assert "Never claim first admission, one stay per patient" in prompt
    assert "patient identifier" in prompt


def test_planner_prompt_has_non_null_cohort_override_example() -> None:
    prompt = _build_planner_user_prompt(_context())

    cohort_start = prompt.index('"spec_id": "alt_cohort_max_during_stay"')
    missing_start = prompt.index('"spec_id": "alt_missing_complete_case"')
    cohort_example = prompt[cohort_start:missing_start]
    assert '"axis": "cohort"' in cohort_example
    assert '"cohort_override": {' in cohort_example
    assert '"concept_id": "sofa"' in cohort_example
    assert '"cohort_override": null' not in cohort_example


def test_planner_prompt_lists_ctas_aggregation_enum() -> None:
    prompt = _build_planner_user_prompt(_context())

    assert "CTAS SCHEMA CONSTRAINTS" in prompt
    assert prompt.index("CTAS SCHEMA CONSTRAINTS") < prompt.index(
        "Every cohort/exposure/outcome concept"
    )
    for value in ALLOWED_CTAS_AGGREGATIONS:
        assert f'"{value}"' in prompt


def test_planner_prompt_forbids_aggregation_synonyms() -> None:
    prompt = _build_planner_user_prompt(_context())

    for invalid in (
        '"first_value"',
        '"max_or_last"',
        '"mean_or_median"',
        '"median_only"',
        '"latest"',
        '"most_recent"',
        '"average"',
    ):
        assert invalid in prompt


def test_planner_prompt_states_window_inequality() -> None:
    prompt = _build_planner_user_prompt(_context())

    assert "end_offset_hours MUST be strictly greater" in prompt
    assert "Zero-width windows" in prompt
    assert "[0h, 1h]" in prompt


def test_planner_prompt_maps_icu_rule_labels_to_ctas_values() -> None:
    prompt = _build_planner_user_prompt(_context())

    assert 'first_value -> "first"' in prompt
    assert 'max_or_last -> "max" or "last"' in prompt
    assert 'mean_or_median / mean_median -> "mean" or "median"' in prompt
    assert 'median_only -> "median"' in prompt


def test_planner_context_formats_ctas_compatible_aggregation_hint() -> None:
    prompt = _build_planner_user_prompt(
        _context(aggregation_default=schema.AggregationRule.MEDIAN_ONLY)
    )

    assert "agg_default=median (icu_rule=median_only;" in prompt
    assert "agg_default=median_only" not in prompt


def test_planner_retry_headroom_allows_five_total_attempts() -> None:
    assert PLANNER_MAX_RETRIES == 4


def test_a_concept_reachable_only_under_one_aggregation_is_still_offered() -> None:
    """Bindability is a property of the (concept, aggregation) pair, not the id.

    A sealed export carries ``sofa_max``, not ``sofa``: the id binds under
    ``max`` and under nothing else. Checking a single aggregation would drop
    every such concept from the menu -- over-narrowing, which costs the same
    planning attempts the wide menu did, in the other direction. The earlier
    fixtures could not see this because their columns bind under every
    aggregation, so a one-aggregation mutant passed them unchanged.
    """

    from easyicu.research_agent.agents.core import _bindable_concept_ids
    from easyicu.research_agent.cohort.schema import _resolve_predicate_column

    columns = ["sofa_max", "lact_min", "death"]

    assert (
        _resolve_predicate_column(columns, "sofa", "count", column_bindings={}) is None
    )
    assert (
        _resolve_predicate_column(columns, "sofa", "max", column_bindings={})
        is not None
    )

    offered = _bindable_concept_ids(columns)

    assert "sofa" in offered
    assert "lact" in offered
    assert "death" in offered

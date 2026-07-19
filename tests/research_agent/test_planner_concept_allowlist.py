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
    listed_ids = [
        line.strip()[3:-1]
        for line in prompt.splitlines()
        if line.strip().startswith("- `") and line.strip().endswith("`")
    ]
    assert len(listed_ids) >= 50
    assert "sofa" in listed_ids
    assert "death" in listed_ids


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

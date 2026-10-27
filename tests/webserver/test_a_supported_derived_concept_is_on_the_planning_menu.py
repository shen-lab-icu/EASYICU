"""A derived concept a source supports is on that source's planning menu.

The Planner chooses from the menu ``_metadata_only_planning_catalog`` builds.
Before 2026-10-27 the menu lacked ``icu_death`` on every database although the
outcome loader derives it on each, so every ICU-mortality question stopped
with "this data has no icu_death" -- blaming the data for a gap of our own.
"""

import pytest

from easyicu.concept_output_sources import COMPOSITE_CONCEPT_OUTPUT_SOURCES
from easyicu.research_agent.concept_availability import explain_concept_availability
from easyicu.webserver import agent_pipeline_runs as owner

# Columns the AKI and creatinine owners emit to record how a stage was reached
# (reference or source-native variant, baseline, evidence and coverage
# receipts). Analyses read them through the staged concepts, never as menu
# choices. A new derived concept is on the menu unless it is added here.
PROVENANCE_COLUMNS = frozenset({
    "aki_reference", "aki_reference_authority", "aki_reference_fidelity",
    "aki_reference_profile", "aki_reference_reliability_grade", "aki_reference_rrt_scope",
    "aki_reference_status", "aki_reference_uses_future",
    "aki_severe_reference", "aki_severe_source_native",
    "aki_source_native", "aki_source_native_ascertainment", "aki_source_native_authority",
    "aki_source_native_fidelity", "aki_source_native_profile", "aki_source_native_reason",
    "aki_source_native_reliability_grade", "aki_source_native_status",
    "aki_source_native_time_scale", "aki_source_native_uses_future",
    "aki_stage_creat_reference", "aki_stage_creat_source_native", "aki_stage_crrt_source_native",
    "aki_stage_reference", "aki_stage_reference_smoothed_6h", "aki_stage_rrt_reference",
    "aki_stage_source_native", "aki_stage_source_native_smoothed",
    "aki_stage_uo_reference", "aki_stage_uo_source_native",
    "creat_baseline_n_48h", "creat_baseline_n_7d", "creat_baseline_source",
    "creat_pre_icu_history_observed", "creatinine_evidence_reason", "creatinine_evidence_status",
    "kidney_observation_window_coverage",
    "rrt_evidence_reason", "rrt_evidence_status", "urine_evidence_status",
})


@pytest.mark.parametrize("database", ["miiv", "mimic", "eicu", "hirid", "aumc", "sic", "eicu_demo", "mimic_demo"])
def test_every_supported_derived_concept_is_on_the_planning_menu(database) -> None:
    menu = set(owner._metadata_only_planning_catalog(database=database, export_path=None).ids())
    off_menu = sorted(
        concept
        for concept in COMPOSITE_CONCEPT_OUTPUT_SOURCES
        if concept not in menu
        and concept not in PROVENANCE_COLUMNS
        and explain_concept_availability(concept=concept, database=database).available
    )
    assert off_menu == []


def test_icu_death_is_on_the_planning_menu_of_every_database() -> None:
    for database in ("miiv", "mimic", "eicu", "hirid", "aumc", "sic"):
        menu = owner._metadata_only_planning_catalog(database=database, export_path=None).ids()
        assert "icu_death" in menu, database


def test_the_provenance_list_names_only_derived_columns() -> None:
    assert PROVENANCE_COLUMNS <= set(COMPOSITE_CONCEPT_OUTPUT_SOURCES)

"""Zero-dependency authority for composite concept output names.

Some EasyICU loaders emit several user-facing columns from one extraction
source.  The mapping is shared by the concept catalog, export/runtime metadata,
and the research-agent cohort binder.  It lives outside ``easyicu.concept`` so
consumers do not import that package's executable loader facade merely to read
declarative metadata.
"""

from __future__ import annotations

COMPOSITE_CONCEPT_OUTPUT_SOURCES: dict[str, str] = {
    "aki": "kdigo_aki",
    "aki_stage": "kdigo_aki",
    "aki_stage_rrt": "kdigo_aki",
    "aki_stage_creat": "kdigo_creat",
    "creat_low_past_48hr": "kdigo_creat",
    "creat_low_past_7day": "kdigo_creat",
    "aki_stage_uo": "kdigo_uo",
    "uo_rt_6hr": "kdigo_uo",
    "uo_rt_12hr": "kdigo_uo",
    "uo_rt_24hr": "kdigo_uo",
    "circ_event": "circ_failure_loader",
    "circ_failure": "circ_failure_loader",
    "sep3_sofa1": "sep3",
    "charlson": "comorbidity_loader",
    "elixhauser": "comorbidity_loader",
    "mort_28d": "outcomes_loader",
    "mort_90d": "outcomes_loader",
    "mort_365d": "outcomes_loader",
    "icu_free_days_28": "outcomes_loader",
    "vent_free_days_28": "outcomes_loader",
    "icu_readmission": "outcomes_loader",
    "culture_positive": "microbiology_loader",
    "bld_culture_positive": "microbiology_loader",
}


__all__ = ["COMPOSITE_CONCEPT_OUTPUT_SOURCES"]

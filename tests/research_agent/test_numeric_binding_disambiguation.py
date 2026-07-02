"""Regression tests for semantic NumericClaim disambiguation."""

from __future__ import annotations

from pathlib import Path


def _register_pilot7_candidates(store) -> None:
    store.register_numeric_claim(
        value="0.555556",
        canonical=0.555556,
        evidence_id="e_mortality",
        step_id="01_score_stratification",
        source_field="strata[8].mortality_rate",
    )
    store.register_numeric_claim(
        value="0.56044",
        canonical=0.560439560449201,
        evidence_id="e_or",
        step_id="02_unadjusted_association",
        source_field="primary_association.odds_ratio",
    )


def test_or_prose_picks_odds_ratio_candidate(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    _register_pilot7_candidates(store)

    bound, binding_map, untraced = bind_numeric_values(
        "In the model, the adjusted OR was 0.56 for the primary exposure.",
        evidence=store,
    )

    assert untraced == []
    assert "<!-- AMBIGUOUS:" not in bound
    assert binding_map["claim_1"].source_field == "primary_association.odds_ratio"


def test_mortality_prose_picks_mortality_candidate(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    _register_pilot7_candidates(store)

    _, binding_map, untraced = bind_numeric_values(
        "The corresponding mortality rate was 0.56 in the stratum.",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "strata[8].mortality_rate"


def test_ambiguous_prose_writes_ambiguous_marker_not_arbitrary_pick(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="1.23",
        canonical=1.23,
        evidence_id="e_a",
        step_id="02_model",
        source_field="field_a",
    )
    store.register_numeric_claim(
        value="1.23",
        canonical=1.23,
        evidence_id="e_b",
        step_id="02_model",
        source_field="field_b",
    )

    bound, binding_map, untraced = bind_numeric_values(
        "The analysis reported 1.23 in the table.",
        evidence=store,
    )

    assert binding_map == {}
    assert untraced == ["1.23"]
    assert "<!-- AMBIGUOUS:1.23:candidates=[" in bound


def test_same_step_preference(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="9.99",
        canonical=9.99,
        evidence_id="e_anchor",
        step_id="02_model",
        source_field="anchor_value",
    )
    store.register_numeric_claim(
        value="1.23",
        canonical=1.23,
        evidence_id="e_same",
        step_id="02_model",
        source_field="secondary_value",
    )
    store.register_numeric_claim(
        value="1.23",
        canonical=1.23,
        evidence_id="e_later",
        step_id="03_model",
        source_field="secondary_value",
    )

    _, binding_map, untraced = bind_numeric_values(
        "Anchor value was 9.99. The repeated value was 1.23.",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "anchor_value"
    assert binding_map["claim_2"].step_id == "02_model"


def test_cited_step_breaks_repeated_count_tie(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="74829",
        canonical=74829.0,
        evidence_id="e_cohort",
        step_id="01_define_cohort",
        source_field="n_analysis_cohort",
    )
    store.register_numeric_claim(
        value="74829",
        canonical=74829.0,
        evidence_id="e_model",
        step_id="04_model",
        source_field="n_final_model",
    )

    _, binding_map, untraced = bind_numeric_values(
        "The analysis cohort included 74,829 ICU stays "
        "[01_define_cohort](evidence/cohort.json).",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].step_id == "01_define_cohort"


def test_same_evidence_duplicate_count_uses_stable_field_tiebreak(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="74829",
        canonical=74829.0,
        evidence_id="e_cohort",
        step_id="01_define_cohort",
        source_field="missingness_audit.numeric_coercion_rows[0].n_total",
    )
    store.register_numeric_claim(
        value="74829",
        canonical=74829.0,
        evidence_id="e_cohort",
        step_id="01_define_cohort",
        source_field="attrition.n_analysis_cohort",
    )

    _, binding_map, untraced = bind_numeric_values(
        "The analysis cohort included 74,829 ICU stays "
        "[01_define_cohort](evidence/cohort.json).",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "attrition.n_analysis_cohort"


def test_same_step_duplicate_count_across_versions_uses_stable_tiebreak(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="74829",
        canonical=74829.0,
        evidence_id="e_model_old",
        step_id="04_model",
        source_field="n_universe",
    )
    store.register_numeric_claim(
        value="74829",
        canonical=74829.0,
        evidence_id="e_model_new",
        step_id="04_model",
        source_field="n_final_model",
    )

    _, binding_map, untraced = bind_numeric_values(
        "The validated primary model included 74,829 stays "
        "[04_model](evidence/model.json).",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "n_final_model"


def test_analyzed_stays_context_prefers_final_model_over_universe(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="74829",
        canonical=74829.0,
        evidence_id="e_model_old",
        step_id="04_model",
        source_field="n_universe",
    )
    store.register_numeric_claim(
        value="74829",
        canonical=74829.0,
        evidence_id="e_model_new",
        step_id="04_model",
        source_field="n_final_model",
    )

    _, binding_map, untraced = bind_numeric_values(
        "The final model retained 74,829 analyzed stays "
        "[04_model](evidence/model.json).",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "n_final_model"


def test_integer_percent_display_binds_to_rounded_percent_claim(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="0.098852",
        canonical=0.098852,
        evidence_id="e_model",
        step_id="04_model",
        source_field="death_rate_final_model",
    )

    _, binding_map, untraced = bind_numeric_values(
        "The event rate was below 10% [04_model](evidence/model.json).",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "death_rate_final_model"


def test_precision_distance_breaks_remaining_tie(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="1.226",
        canonical=1.226,
        evidence_id="e_farther",
        step_id="02_model",
        source_field="farther_value",
    )
    store.register_numeric_claim(
        value="1.229",
        canonical=1.229,
        evidence_id="e_closer",
        step_id="02_model",
        source_field="closer_value",
    )

    _, binding_map, untraced = bind_numeric_values(
        "The table reported 1.23.",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "closer_value"


def test_spaced_percent_display_binds_to_proportion_claim(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="0.094",
        canonical=0.094,
        evidence_id="e_baseline",
        step_id="00_probe",
        source_field="baseline_prevalence",
    )

    bound, binding_map, untraced = bind_numeric_values(
        "The overall mortality rate was 9.4 % in the cohort.",
        evidence=store,
    )

    assert untraced == []
    assert "<!-- UNTRACED:9.4 -->" not in bound
    assert binding_map["claim_1"].source_field == "baseline_prevalence"


def test_hazard_ratio_prose_picks_hr_candidate(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="1.30",
        canonical=1.30,
        evidence_id="e_or",
        step_id="03_assoc",
        source_field="primary_or",
    )
    store.register_numeric_claim(
        value="1.30",
        canonical=1.30,
        evidence_id="e_hr",
        step_id="04_survival",
        source_field="hazard_ratio",
    )

    _, binding_map, untraced = bind_numeric_values(
        "The Cox model estimated an HR of 1.30 for the exposure.",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "hazard_ratio"


def test_average_treatment_effect_prose_picks_ate_candidate(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="0.08",
        canonical=0.08,
        evidence_id="e_rate",
        step_id="02_descriptive",
        source_field="outcome_rate",
    )
    store.register_numeric_claim(
        value="0.08",
        canonical=0.08,
        evidence_id="e_ate",
        step_id="05_causal",
        source_field="average_treatment_effect",
    )

    _, binding_map, untraced = bind_numeric_values(
        "The target-trial analysis estimated an average treatment effect of 0.08.",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "average_treatment_effect"


def test_length_of_stay_prose_picks_los_candidate(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="4.20",
        canonical=4.2,
        evidence_id="e_age",
        step_id="01_table",
        source_field="median_age",
    )
    store.register_numeric_claim(
        value="4.20",
        canonical=4.2,
        evidence_id="e_los",
        step_id="03_los",
        source_field="median_los_icu",
    )

    _, binding_map, untraced = bind_numeric_values(
        "The median ICU length of stay was 4.20 days.",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "median_los_icu"


def test_complete_case_context_prefers_complete_case_count(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="74829",
        canonical=74829.0,
        evidence_id="e_context",
        step_id="01_define_cohort",
        source_field="n_universe",
    )
    store.register_numeric_claim(
        value="74829",
        canonical=74829.0,
        evidence_id="e_model",
        step_id="04_primary_adjusted_association_model",
        source_field="complete_case_flow.n_complete_case",
    )
    store.register_numeric_claim(
        value="74829",
        canonical=74829.0,
        evidence_id="e_model",
        step_id="04_primary_adjusted_association_model",
        source_field="n_final_model",
    )

    _, binding_map, untraced = bind_numeric_values(
        "The primary adjusted model was fitted on 74,829 complete cases.",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "complete_case_flow.n_complete_case"


def test_primary_or_ci_context_prefers_primary_or_ci_not_summary_alias(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="1.02037",
        canonical=1.0203660748446095,
        evidence_id="e_model",
        step_id="04_primary_adjusted_association_model",
        source_field="primary_or_ci_low",
    )
    store.register_numeric_claim(
        value="1.02037",
        canonical=1.0203660748446093,
        evidence_id="e_model",
        step_id="04_primary_adjusted_association_model",
        source_field="primary_or_ci_low_from_summary",
    )
    store.register_numeric_claim(
        value="1.01995",
        canonical=1.0199471110955838,
        evidence_id="e_sensitivity",
        step_id="05_sensitivity_comparison",
        source_field="alternative_effect_scales.risk_ratio.ci_low",
    )

    _, binding_map, untraced = bind_numeric_values(
        "The adjusted odds ratio had a 95% confidence interval from 1.020.",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "primary_or_ci_low"


def test_range_context_prefers_robustness_range_claim(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.manuscript_post import bind_numeric_values

    store = ra.EvidenceStore(tmp_path)
    store.register_numeric_claim(
        value="0.00108315",
        canonical=0.001083146182081867,
        evidence_id="e_sensitivity",
        step_id="05_sensitivity_comparison",
        source_field="alternative_effect_scales.risk_difference.ci_low",
    )
    store.register_numeric_claim(
        value="0.00108315",
        canonical=0.001083146182081867,
        evidence_id="e_panel",
        step_id="robustness_panel",
        source_field="range_low",
    )

    _, binding_map, untraced = bind_numeric_values(
        "Across robustness analyses, the point estimate ranged from 0.001.",
        evidence=store,
    )

    assert untraced == []
    assert binding_map["claim_1"].source_field == "range_low"

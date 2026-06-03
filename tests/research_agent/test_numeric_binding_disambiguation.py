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

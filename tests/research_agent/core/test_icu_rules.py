"""ICU-rules: concept-hint matching and per-(role, kind) aggregations.

These tests are the first line of defence against silently softening the
ICU-knowledge layer (e.g. someone "fixes" the table by allowing
``mean()`` for ordinal scores). If the table is loosened, the test fails
and surfaces the regression before it can leak into a manuscript.
"""

from __future__ import annotations


def test_classify_known_sofa_components(ra):
    """Every ``sofa*_*`` component should pick up the ordinal-score hint."""
    icu = ra.ICU_RULES
    VariableKind = ra.VariableKind
    for col in [
        "sofa_resp",
        "sofa_coag",
        "sofa_liver",
        "sofa_cardio",
        "sofa_cns",
        "sofa_renal",
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_liver",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_renal",
    ]:
        hint = icu.classify_variable(col, "int64")
        assert hint.kind == VariableKind.ORDINAL, f"{col} should be ordinal"
        assert hint.is_ordinal is True
        assert hint.role.value == "ordinal_score"
        # SOFA components must default to MAX_LAST aggregation
        assert hint.aggregation_default.value == "max_or_last"


def test_classify_total_sofa_is_composite_score(ra):
    icu = ra.ICU_RULES
    VariableKind = ra.VariableKind
    for col in ("sofa", "sofa2"):
        hint = icu.classify_variable(col, "int64")
        assert hint.role.value == "composite_score"
        assert hint.kind == VariableKind.ORDINAL
        # canonical SOFA pitfalls must be present so the LLM prompt sees them
        joined = " ".join(hint.pitfalls).lower()
        assert "sofa" in joined


def test_classify_kdigo_stage_is_ordinal_score(ra):
    hint = ra.ICU_RULES.classify_variable("kdigo_stage", "int64", [0, 1, 2, 3])
    assert hint.role.value == "ordinal_score"
    assert hint.kind == ra.VariableKind.ORDINAL
    assert hint.is_ordinal is True
    assert hint.aggregation_default.value == "max_or_last"


def test_classify_wide_export_aki_stage_is_kdigo_ordinal(ra):
    for name in (
        "aki_stage",
        "aki_stage_max",
        "aki_stage_creat",
        "aki_stage_uo",
        "aki_stage_rrt",
    ):
        hint = ra.ICU_RULES.classify_variable(name, "float64", [])
        assert hint.role.value == "ordinal_score"
        assert hint.kind == ra.VariableKind.ORDINAL
        assert hint.valid_range == (0.0, 3.0)
        assert hint.is_ordinal is True
        assert hint.ordinal_levels == (0, 1, 2, 3)


def test_concurrent_support_indicators_are_not_baseline_covariates(ra):
    for name in ("mech_vent", "vaso_ind"):
        hint = ra.ICU_RULES.classify_variable(name, "int64", [0, 1])
        assert hint.role.value == "intervention"
        assert hint.kind == ra.VariableKind.BINARY
        assert hint.aggregation_default.value == "max_or_last"


def test_longest_prefix_match(ra):
    """``sofa2_resp_24h`` should match the ``sofa2_resp`` hint, not ``sofa2``."""
    icu = ra.ICU_RULES
    long_hint = icu.classify_variable("sofa2_resp_24h", "int64")
    short_hint = icu.classify_variable("sofa2", "int64")
    assert long_hint.role.value == "ordinal_score"
    assert short_hint.role.value == "composite_score"


def test_classify_unknown_falls_back_to_dtype(ra):
    icu = ra.ICU_RULES
    VariableKind = ra.VariableKind
    h_float = icu.classify_variable("foo_continuous_thing", "float64")
    assert h_float.kind == VariableKind.CONTINUOUS
    h_int_bool = icu.classify_variable("flag_blah", "int64", sample_values=[0, 1, 0, 1])
    assert h_int_bool.kind == VariableKind.BINARY
    h_dt = icu.classify_variable("evt", "datetime64[ns]")
    assert h_dt.kind == VariableKind.TIMESTAMP


def test_measurement_count_companion_preserves_window_suffix() -> None:
    from easyicu.research_agent.icu_rules import (
        companion_count_column_for_measured,
    )

    assert companion_count_column_for_measured("signal_measured") == "signal_n"
    assert companion_count_column_for_measured("signal_measured_6h") == "signal_n_6h"
    assert (
        companion_count_column_for_measured("Signal_Measured_first_24h")
        == "Signal_n_first_24h"
    )
    assert companion_count_column_for_measured("signal_measurement_rate") is None


def test_companion_audit_columns_override_base_concept_prefix_metadata(ra):
    """Counts/status flags are provenance, never disguised physiology.

    The examples intentionally span a lab, vital sign, and ordinal score and do
    not use the development case's focal variable.  This catches the generic
    prefix-inheritance bug rather than pinning one benchmark spelling.
    """

    icu = ra.ICU_RULES
    count_columns = ("creat_n", "bili_n_24h")
    status_columns = ("hr_measured", "sofa2_measurement_flag")

    for column in count_columns:
        hint = icu.classify_variable(column, "int64", sample_values=[0, 2, 5])
        assert hint.role.value == "meta", column
        assert hint.kind == ra.VariableKind.COUNT, column
        assert hint.unit is None, column
        assert hint.valid_range is None, column
        assert hint.is_ordinal is False, column
        assert hint.ordinal_levels is None, column
        assert "provenance" in " ".join(hint.pitfalls).lower(), column

    for column in status_columns:
        hint = icu.classify_variable(column, "int64", sample_values=[0, 1, 1])
        assert hint.role.value == "meta", column
        assert hint.kind == ra.VariableKind.BINARY, column
        assert hint.unit is None, column
        assert hint.valid_range == (0.0, 1.0), column
        assert hint.is_ordinal is False, column
        assert hint.ordinal_levels is None, column
        assert "status" in " ".join(hint.pitfalls).lower(), column

    # Real value columns must retain the base concept's curated semantics.
    assert icu.classify_variable("creat_max", "float64").unit == "mg/dL"
    score = icu.classify_variable("sofa2_max", "int64")
    assert score.kind == ra.VariableKind.ORDINAL
    assert score.is_ordinal is True
    unrelated_flag = icu.classify_variable(
        "resource_flag", "int64", sample_values=[0, 1]
    )
    assert unrelated_flag.role.value == "other"
    assert unrelated_flag.kind == ra.VariableKind.BINARY


def test_aggregation_rule_matrix(ra):
    """Every (role, kind) pair returns at least one allowed aggregation,
    and forbidden ops never sneak in for ordinal/identifier kinds."""
    icu = ra.ICU_RULES
    schema = ra.schema
    forbidden_for_ordinal = {
        schema.AggregationRule.MEAN_MEDIAN,
        schema.AggregationRule.SUM,
    }
    for role in schema.VariableRole:
        for kind in ra.VariableKind:
            allowed = icu.aggregation_rule_for(role, kind)
            assert allowed, f"({role}, {kind}) returned no allowed aggregations"
            if kind == ra.VariableKind.ORDINAL:
                assert all(a not in forbidden_for_ordinal for a in allowed), (
                    f"Ordinal ({role}, {kind}) wrongly allows "
                    f"{forbidden_for_ordinal & set(allowed)}"
                )
            if kind == ra.VariableKind.IDENTIFIER:
                bad = {
                    schema.AggregationRule.MEAN_MEDIAN,
                    schema.AggregationRule.MAX_LAST,
                    schema.AggregationRule.SUM,
                    schema.AggregationRule.MEDIAN_ONLY,
                }
                assert all(a not in bad for a in allowed), (
                    f"Identifier kind wrongly allows {bad & set(allowed)}"
                )


def test_default_time_windows_present(ra):
    icu = ra.ICU_RULES
    windows = icu.default_time_windows()
    names = {w.name for w in windows}
    assert {"first_24h", "first_6h", "full_stay"} <= names
    for w in windows:
        assert w.start_hours <= w.end_hours
        assert w.anchor in {"icu_admission", "hospital_admission", "event_onset"}


def test_lab_continuous_discourages_mean(ra):
    """Right-skewed labs (role=LAB, kind=CONTINUOUS) should default to median;
    mean must not be the recommended default."""
    icu = ra.ICU_RULES
    schema = ra.schema
    allowed = icu.aggregation_rule_for(
        schema.VariableRole.LAB, ra.VariableKind.CONTINUOUS
    )
    assert allowed[0] == schema.AggregationRule.MEDIAN_ONLY
    assert schema.AggregationRule.MEAN_MEDIAN not in allowed


def test_known_concept_hints_have_consistent_pitfalls(ra):
    """SOFA / SOFA2 hints must mention the 0-equals-missing pitfall so the
    planner prompt picks it up. This is the linchpin of the hero ablation."""
    icu = ra.ICU_RULES
    sofa2 = icu.classify_variable("sofa2", "int64")
    pit = " ".join(sofa2.pitfalls).lower()
    assert "missing" in pit or "missingness" in pit


# ---------------------------------------------------------------------------
# Cross-cutting methodological principles (case-neutral, cross-database)
# ---------------------------------------------------------------------------


def test_general_principles_are_well_formed(ra):
    """Every principle has a unique id, a known phase, and a non-empty body."""
    principles = ra.ICU_RULES.general_principles
    assert len(principles) >= 10
    ids = [p.id for p in principles]
    assert len(set(ids)) == len(ids), "principle ids must be unique"
    known_phases = {
        "cohort",
        "features",
        "label",
        "modeling",
        "clustering",
        "interpretation",
    }
    for p in principles:
        assert p.phase in known_phases, f"{p.id} has unknown phase {p.phase}"
        assert p.principle.strip() and p.rationale.strip()


def test_general_principles_cover_the_three_crosswalk_gaps(ra):
    """The gaps surfaced by the teaching-deck crosswalk must be encoded."""
    ids = {p.id for p in ra.ICU_RULES.general_principles}
    # gap 09: ICD code used as timing; gap 12: prevalent vs incident.
    assert "diagnosis_membership_not_timing" in ids
    assert "incident_not_prevalent" in ids
    # gap 13: imbalance-aware evaluation must appear in some principle body.
    bodies = " ".join(p.principle.lower() for p in ra.ICU_RULES.general_principles)
    assert "recall" in bodies or "imbalance" in bodies or "balance" in bodies


def test_general_principles_are_case_neutral(ra):
    """Shared principles must not hard-code a benchmark task/case (prompt hygiene)."""
    import re

    for p in ra.ICU_RULES.general_principles:
        blob = f"{p.principle} {p.rationale} {p.cross_db_note}"
        # No benchmark task ids like e1_/m2_/h3_ leaking into the shared layer.
        assert not re.search(r"\b[emh][123]_", blob), f"{p.id} leaks a benchmark id"


def test_principles_for_phase_filters(ra):
    cohort = ra.ICU_RULES.principles_for_phase("cohort")
    assert cohort and all(p.phase == "cohort" for p in cohort)
    assert ra.ICU_RULES.principles_for_phase("nonsense") == []


def test_cross_db_notes_present_so_rules_stay_general(ra):
    """Cross-database variation must be recorded rather than hard-coding one DB."""
    principles = ra.ICU_RULES.general_principles
    with_notes = [p for p in principles if p.cross_db_note.strip()]
    # The majority carry an explicit cross-database note.
    assert len(with_notes) >= 0.7 * len(principles)


def test_principle_kinds_are_valid(ra):
    """Every principle is tagged error|caution (the impartiality contract)."""
    for p in ra.ICU_RULES.general_principles:
        assert p.kind in {"error", "caution"}, f"{p.id} has bad kind {p.kind}"


def test_objective_errors_vs_defensible_choices(ra):
    """Impartiality: objective mistakes are ``error``; analytical *choices*
    must be ``caution`` so the rule layer never overrides the user's design."""
    by_id = {p.id: p for p in ra.ICU_RULES.general_principles}
    # Objective methodological errors (wrong under any design).
    for eid in (
        "no_outcome_window_leakage",
        "split_by_patient",
        "diagnosis_membership_not_timing",
        "association_is_not_causation",
        "window_aggregation_respects_kind",
        "label_built_in_outcome_window",
    ):
        assert by_id[eid].kind == "error", f"{eid} should be an error"
    # Defensible analytical choices (prompt to document, never impose).
    for cid in (
        "missingness_is_information",
        "metrics_match_task_and_balance",
        "describe_cohort_before_modeling",
        "incident_not_prevalent",
        "state_outcome_definition",
        "consider_competing_risks",
        "control_for_multiplicity",
    ):
        assert by_id[cid].kind == "caution", f"{cid} should be a caution"


def test_detect_overadjustment_flags_exposure_constituents(ra):
    # Sepsis-3 is defined via SOFA, so adjusting for SOFA is overadjustment.
    assert ra.detect_overadjustment("sepsis3", ["age", "sex", "sofa_max"]) == [
        "sofa_max"
    ]
    # No constituent in the adjustment set -> clean.
    assert ra.detect_overadjustment("sepsis3", ["age", "sex"]) == []
    # The exposure's own measurement is never flagged as over-adjustment.
    assert ra.detect_overadjustment("sofa", ["sofa_max", "age"]) == []
    # Unknown / non-composite exposure -> silent (no rule applies).
    assert ra.detect_overadjustment("lactate", ["age", "sofa_max"]) == []
    # A genuine SOFA constituent (creatinine) is caught for a SOFA exposure.
    assert ra.detect_overadjustment("sofa", ["creatinine", "age"]) == ["creatinine"]


def test_detect_overadjustment_qsofa_not_swallowed_by_sofa(ra):
    # "sofa" is a substring of "qsofa": qSOFA must resolve to its OWN bedside
    # components (GCS / resp rate / SBP), NOT SOFA's broader organ-system set --
    # otherwise bilirubin/creatinine, which are NOT qSOFA inputs, would be
    # false-positive overadjustment flags. (Property check, not an exact tuple:
    # composite_constituents is now a dictionary-driven union, not the raw table.)
    qsofa = set(ra.composite_constituents("qsofa"))
    assert {"gcs", "sbp"} <= qsofa
    assert any("resp" in tok for tok in qsofa)
    assert "bilirubin" not in qsofa and "creatinine" not in qsofa
    assert ra.detect_overadjustment("qsofa", ["age", "bilirubin", "creatinine"]) == []
    # A real qSOFA constituent is still caught.
    assert ra.detect_overadjustment("qsofa", ["age", "resp_rate"]) == ["resp_rate"]
    # sofa2 (no own fallback entry) still resolves through the concept dictionary
    # to a SOFA-family closure carrying the organ-system labs that define it.
    sofa2 = set(ra.composite_constituents("sofa2"))
    assert {"creatinine", "bilirubin", "platelet"} <= sofa2


def test_detect_overadjustment_is_dictionary_general_not_table_bound(ra):
    # Generality: a SOFA sub-score has no curated table entry, yet its narrower
    # derivation closure (creatinine / urine) is read straight from the concept
    # dictionary -- the renal sub-score must NOT pull in unrelated SOFA organs.
    renal = set(ra.composite_constituents("sofa_renal"))
    assert "creatinine" in renal
    assert "bilirubin" not in renal and "platelet" not in renal
    # A distinct sub-score that merely shares the "sofa" prefix is a constituent,
    # not the exposure itself, so a SOFA exposure adjusted for it is flagged...
    assert ra.detect_overadjustment("sofa", ["sofa_renal", "age"]) == ["sofa_renal"]
    # ...while the exposure's own measurement (prefix + pure stat suffix) is not.
    assert ra.detect_overadjustment("sofa", ["sofa_max", "age"]) == []


def test_detect_overadjustment_degrades_to_table_without_dictionary(ra):
    # If the concept dictionary cannot be loaded (e.g. a data-isolated sandbox),
    # the detector must still work from the curated fallback table rather than
    # going silent. A broken dictionary object stands in for "unavailable".
    broken = object()
    assert ra.detect_overadjustment(
        "sepsis3", ["age", "sex", "sofa_max"], dictionary=broken
    ) == ["sofa_max"]
    assert ra.detect_overadjustment(
        "sofa", ["creatinine", "age"], dictionary=broken
    ) == ["creatinine"]


def test_is_derived_exposure_recognises_computed_concepts(ra):
    # Composite / derived scores are computed (callback or depends_on).
    for derived in ("sofa", "sepsis3", "news", "mews", "sirs", "pafi", "anion_gap"):
        assert ra.is_derived_exposure(derived), derived
    # Raw measurements / demographics are not.
    for raw in ("lactate", "age", "sex"):
        assert not ra.is_derived_exposure(raw), raw


def test_overadjustment_caution_covers_unresolvable_composites(ra):
    # mews/news/sirs are callback scores with an empty dependency closure, so
    # detect_overadjustment is blind to them -> a caution must surface instead
    # of a silent pass. This is the 200+-feature generality gap.
    covs = ["age", "sex", "heart_rate"]
    for derived in ("news", "mews", "sirs", "pafi", "anion_gap"):
        assert ra.overadjustment_caution(derived, covs), derived
        # ...and the deterministic check is genuinely silent for them.
        assert ra.detect_overadjustment(derived, covs) == []

    # Resolvable composites are handled by the error path, not the caution.
    assert ra.overadjustment_caution("sofa", ["age"]) is None
    assert ra.overadjustment_caution("sepsis3", ["age"]) is None
    # Non-derived exposures never caution.
    assert ra.overadjustment_caution("lactate", covs) is None
    assert ra.overadjustment_caution("age", covs) is None
    # No covariates -> nothing to verify.
    assert ra.overadjustment_caution("news", []) is None


def test_concept_methodology_profile_classifies_by_structure(ra):
    # True endpoint (non-derived "outcome" category) -> leakage role.
    death = ra.concept_methodology_profile("death_icu", category="outcome")
    assert "outcome" in death.roles
    assert "leakage" in death.tag()

    # A severity score sits in the dictionary "outcome" category but is derived,
    # so it must NOT be mislabelled as a study endpoint -- it's a derived score.
    sofa = ra.concept_methodology_profile("sofa", category="outcome")
    assert "outcome" not in sofa.roles
    assert "derived_composite" in sofa.roles
    assert "ordinal_score" in sofa.roles  # SOFA is ordinal -> no averaging

    # Medication -> treatment (confounder vs mediator caution).
    norepi = ra.concept_methodology_profile("norepi", category="medications")
    assert "treatment" in norepi.roles
    assert "mediator" in norepi.tag()

    # Plain demographic / raw lab -> no hazard tag (default safe case).
    assert ra.concept_methodology_profile("age", category="demographics").tag() == ""
    assert (
        ra.concept_methodology_profile("creatinine", category="chemistry").tag() == ""
    )


def test_concept_methodology_tag_is_convenience_for_profile_tag(ra):
    assert (
        ra.concept_methodology_tag("norepi", category="medications")
        == ra.concept_methodology_profile("norepi", category="medications").tag()
    )


def test_detect_outcome_as_predictor_flags_self_leakage(ra):
    # The declared outcome appearing among the predictors is target leakage by
    # construction -> objective error, like overadjustment.
    assert ra.detect_outcome_as_predictor(
        ["age", "death_icu", "sofa"], study_outcome="death_icu"
    ) == ["death_icu"]
    # Suffix-tolerant: a stat/derivation spelling of the outcome still matches.
    assert ra.detect_outcome_as_predictor(
        ["death_icu_max"], study_outcome="death_icu"
    ) == ["death_icu_max"]
    # Clean model (outcome only on the left-hand side) -> nothing.
    assert (
        ra.detect_outcome_as_predictor(
            ["age", "sofa", "lactate"], study_outcome="death_icu"
        )
        == []
    )
    # No declared outcome -> silent (never inferred).
    assert ra.detect_outcome_as_predictor(["age", "death_icu"]) == []


def test_outcome_leakage_caution_flags_other_endpoint_not_study_outcome(ra):
    # A *different* endpoint concept (los_icu) used as a predictor is a
    # timing-dependent hazard -> caution, not the firm self-leakage error.
    assert ra.outcome_leakage_caution(["age", "los_icu"], study_outcome="death_icu")
    # The study outcome itself is excluded here (it is the firm-error path).
    assert ra.outcome_leakage_caution(["death_icu"], study_outcome="death_icu") is None
    # Plain covariates / a derived severity score never caution as endpoints.
    assert (
        ra.outcome_leakage_caution(
            ["age", "sofa", "lactate"], study_outcome="death_icu"
        )
        is None
    )


def test_treatment_mediator_caution_is_caution_only(ra):
    # A treatment/intervention covariate may be a mediator on the exposure->
    # outcome path -> caution (the DAG/timing is unknown), never an error.
    assert ra.treatment_mediator_caution("sepsis3", ["age", "furosemide", "lactate"])
    # No treatment covariate -> silent.
    assert ra.treatment_mediator_caution("sepsis3", ["age", "lactate", "sofa"]) is None
    # No exposure declared -> no mediator interpretation, stays silent.
    assert ra.treatment_mediator_caution("", ["furosemide"]) is None
    # The exposure being itself a treatment is not flagged as its own mediator.
    assert ra.treatment_mediator_caution("furosemide", ["furosemide"]) is None


def test_leakage_detectors_degrade_without_dictionary(ra):
    # In a data-isolated sandbox (no concept dictionary) the dictionary-backed
    # checks must not fabricate flags. The self-leakage match is purely token
    # based, so it still works; the category-driven ones stay silent.
    assert ra.detect_outcome_as_predictor(
        ["age", "death_icu"], study_outcome="death_icu", dictionary={}
    ) == ["death_icu"]
    assert (
        ra.outcome_leakage_caution(
            ["age", "los_icu"], study_outcome="death_icu", dictionary={}
        )
        is None
    )
    assert (
        ra.treatment_mediator_caution("sepsis3", ["furosemide"], dictionary={}) is None
    )

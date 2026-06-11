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
        "sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_cns", "sofa_renal",
        "sofa2_resp", "sofa2_coag", "sofa2_liver", "sofa2_cardio", "sofa2_cns", "sofa2_renal",
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


def test_aggregation_rule_matrix(ra):
    """Every (role, kind) pair returns at least one allowed aggregation,
    and forbidden ops never sneak in for ordinal/identifier kinds."""
    icu = ra.ICU_RULES
    schema = ra.schema
    forbidden_for_ordinal = {schema.AggregationRule.MEAN_MEDIAN, schema.AggregationRule.SUM}
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
                bad = {schema.AggregationRule.MEAN_MEDIAN, schema.AggregationRule.MAX_LAST,
                       schema.AggregationRule.SUM, schema.AggregationRule.MEDIAN_ONLY}
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
    allowed = icu.aggregation_rule_for(schema.VariableRole.LAB, ra.VariableKind.CONTINUOUS)
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
        "cohort", "features", "label", "modeling", "clustering", "interpretation",
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
    bodies = " ".join(
        p.principle.lower() for p in ra.ICU_RULES.general_principles
    )
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

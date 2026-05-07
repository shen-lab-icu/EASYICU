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

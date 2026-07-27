from __future__ import annotations

from easyicu.research_agent.repair_registry import RepairClass, repair_metadata_for
from easyicu.research_agent.repairs.reasons import RepairReason
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair
from easyicu.research_agent.schema import ValidationFinding


def _finding(
    *,
    variable: str | list[str] = "age",
    value_class: str = "finite_outside_plausibility_range",
):
    return ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="A flag-only plausibility range was used as an exclusion rule.",
        detail={
            "issue_code": "plausibility_range_exclusion_required",
            "variable": variable,
            "value_class": value_class,
        },
    )


def _repair(code: str, finding: ValidationFinding):
    return deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION],
        repair_findings=[finding],
    )


def test_a_mask_the_guard_alone_reads_is_left_for_provider_repair():
    """Removing that guard removes the mask, and with it the whole record.

    This shape used to be repaired here, and the old test asserted the
    deletion -- `"age_out_of_domain" not in repaired` -- as the correct
    outcome. It is not: the guard is the mask's only reader, so deleting the
    guard deletes the one computation that says anything about the
    out-of-range rows. The script then neither excludes them nor flags them,
    and the audit downgrade that looks for that computation finds nothing and
    lets the step pass. `retain_and_flag` is two obligations; a deterministic
    patch can discharge only one, so this shape goes to provider repair, where
    the Coder is told to keep every row and record the count.
    """

    code = """
age = strict_numeric(df["age"], "age")
age_out_of_domain = (age < 0.0) | (age > 120.0)
if bool(age_out_of_domain.any()):
    raise ValueError("Age outside plausibility range")
adult_mask = age >= 18.0
"""

    repaired, names = _repair(code, _finding())

    assert names == []
    assert repaired == code
    # The claim the old test shared with this one: an unrelated cohort rule on
    # the same variable is never touched either way.
    assert "adult_mask = age >= 18.0" in repaired


def test_repair_is_idempotent():
    """Idempotence, shown on the shape that is still repaired."""

    code = """
age_out_of_domain = (age < 0.0) | (age > 120.0)
age_out_of_domain_n = int(age_out_of_domain.sum())
if age_out_of_domain_n > 0:
    raise ValueError("Age outside plausibility range")
summary = {"out_of_domain_n": age_out_of_domain_n}
"""
    once, names = _repair(code, _finding())
    twice, second_names = _repair(once, _finding())

    assert names == ["flag_only_plausibility_range_retention_v1"]
    assert twice == once
    assert second_names == []


def test_semantic_column_family_loop_local_mask_is_also_left_alone():
    """The same refusal inside a loop, where the family is named as a list."""

    code = """
for column, values in numeric_columns.items():
    nonfinite = values.notna() & ~np.isfinite(values)
    if nonfinite.any():
        raise ValueError("nonfinite")
    out_of_domain = values.notna() & ((values < 0.0) | (values > 24.0))
    if bool(out_of_domain.any()):
        raise ValueError(f"{column} outside plausibility range")
publish(values)
"""

    repaired, names = _repair(
        code,
        _finding(variable=["score_h0_6", "score_h6_12"]),
    )

    assert names == []
    assert repaired == code
    # The unrelated non-finite guard was never this repair's business.
    assert "nonfinite.any()" in repaired


def test_a_loop_local_counted_family_guard_is_still_repaired():
    """The refusal above is about the missing record, not about loops.

    Without this, removing the direct shape would look like "loop-local range
    guards are out of scope", and the next reader would have no case showing
    the family form still repairs when the count survives.
    """

    code = """
for column, values in numeric_columns.items():
    out_of_domain = values.notna() & ((values < 0.0) | (values > 24.0))
    out_of_domain_n = int(out_of_domain.sum())
    if out_of_domain_n > 0:
        raise ValueError(f"{column} outside plausibility range")
    audit[column] = {"out_of_domain_n": out_of_domain_n}
"""

    repaired, names = _repair(
        code,
        _finding(variable=["score_h0_6", "score_h6_12"]),
    )

    assert names == ["flag_only_plausibility_range_retention_v1"]
    assert "out_of_domain_n = int(out_of_domain.sum())" in repaired
    assert "if out_of_domain_n > 0" not in repaired


def test_semantic_column_family_does_not_choose_between_two_range_guards():
    code = """
first_bad = (first < 0.0) | (first > 10.0)
if first_bad.any():
    raise ValueError("first")
second_bad = (second < 0.0) | (second > 10.0)
if second_bad.any():
    raise ValueError("second")
"""

    assert _repair(code, _finding(variable=["first", "second"])) == (code, [])


def test_stale_single_variable_finding_does_not_block_current_grouped_repair():
    code = """
# _easyicu_flag_only_plausibility_range_retained_v1
for column, values in numeric_columns.items():
    out_of_domain = values.notna() & ((values < 0.0) | (values > 24.0))
    out_of_domain_n = int(out_of_domain.sum())
    if out_of_domain_n > 0:
        raise ValueError(f"{column} outside plausibility range")
    audit[column] = {"out_of_domain_n": out_of_domain_n}
"""
    findings = [
        _finding(variable="age"),
        _finding(variable=["score_h0_6", "score_h6_12"]),
    ]

    repaired, names = deterministic_concept_audit_repair(
        code,
        [finding.message for finding in findings],
        repair_reasons=[RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION],
        repair_findings=findings,
    )

    assert names == ["flag_only_plausibility_range_retention_v1"]
    assert "if out_of_domain_n > 0" not in repaired
    # The count the repair exists to preserve is still there.
    assert "out_of_domain_n = int(out_of_domain.sum())" in repaired


def test_wrong_value_class_or_variable_is_not_rewritten():
    code = """
age_out_of_domain = (age < 0.0) | (age > 120.0)
if age_out_of_domain.any():
    raise ValueError("Age outside strict domain")
"""

    assert _repair(code, _finding(value_class="strict_domain_violation")) == (code, [])
    assert _repair(code, _finding(variable="lactate")) == (code, [])


def test_mask_with_another_consumer_is_not_rewritten():
    code = """
age_out_of_domain = (age < 0.0) | (age > 120.0)
audit["outlier_n"] = int(age_out_of_domain.sum())
if age_out_of_domain.any():
    raise ValueError("Age outside plausibility range")
"""

    assert _repair(code, _finding()) == (code, [])


def test_flag_only_count_is_retained_while_terminal_rejection_is_removed():
    code = """
def fail(message):
    raise RuntimeError(message)

age_original = df["age"]
age_numeric = pd.to_numeric(age_original, errors="coerce")
age_out_of_domain_mask = (age_numeric < 0) | (age_numeric > 120)
age_out_of_domain_n = int(age_out_of_domain_mask.sum())
if age_out_of_domain_n > 0:
    fail(f"age outside flag-only range: {age_out_of_domain_n}")
retained_mask = age_numeric >= 18
step_summary = {"out_of_domain_n": age_out_of_domain_n}
"""

    repaired, names = _repair(code, _finding(variable="age"))

    assert names == ["flag_only_plausibility_range_retention_v1"]
    assert "age_out_of_domain_mask =" in repaired
    assert "age_out_of_domain_n = int(age_out_of_domain_mask.sum())" in repaired
    assert '"out_of_domain_n": age_out_of_domain_n' in repaired
    assert "retained_mask = age_numeric >= 18" in repaired
    assert "if age_out_of_domain_n > 0" not in repaired
    assert "_easyicu_flag_only_plausibility_range_retained_v1" in repaired


def test_sealed_minimum_maximum_terminal_guards_retain_values():
    """Regression for E1 r25's exact host-bound plausibility shape."""

    code = """
for column_name in REQUIRED_COLUMNS:
    contract = raw_contracts[column_name]
    observed = df[column_name].dropna()
    plausibility = contract.get("analysis_plausibility_range")
    if plausibility is not None and column_name != "sex":
        numeric_observed = validate_numeric_source(observed, column_name)
        minimum = plausibility.get("minimum")
        maximum = plausibility.get("maximum")
        if minimum is not None and (numeric_observed < minimum).any():
            raise ValueError("below plausibility range")
        if maximum is not None and (numeric_observed > maximum).any():
            raise ValueError("above plausibility range")
publish(df)
"""

    repaired, names = _repair(code, _finding(variable="age"))

    assert names == ["flag_only_plausibility_range_retention_v1"]
    assert repaired.count("_easyicu_flag_only_plausibility_range_retained_v1") == 2
    assert "raise ValueError(\"below plausibility range\")" not in repaired
    assert "raise ValueError(\"above plausibility range\")" not in repaired
    assert "(numeric_observed < minimum).any()" in repaired
    assert "(numeric_observed > maximum).any()" in repaired
    assert "publish(df)" in repaired
    assert _repair(repaired, _finding(variable="age")) == (repaired, [])


def test_two_sealed_plausibility_guard_pairs_are_ambiguous():
    code = """
first_range = first_contract.get("analysis_plausibility_range")
first_min = first_range.get("minimum")
first_max = first_range.get("maximum")
if first_min is not None and (first < first_min).any():
    raise ValueError("first low")
if first_max is not None and (first > first_max).any():
    raise ValueError("first high")
second_range = second_contract.get("analysis_plausibility_range")
second_min = second_range.get("minimum")
second_max = second_range.get("maximum")
if second_min is not None and (second < second_min).any():
    raise ValueError("second low")
if second_max is not None and (second > second_max).any():
    raise ValueError("second high")
"""

    assert _repair(code, _finding(variable="age")) == (code, [])


def test_counted_range_guard_does_not_rewrite_a_filtering_mask():
    code = """
def fail(message):
    raise RuntimeError(message)

age_original = df["age"]
age_numeric = pd.to_numeric(age_original, errors="coerce")
age_out_of_domain_mask = (age_numeric < 0) | (age_numeric > 120)
age_out_of_domain_n = int(age_out_of_domain_mask.sum())
if age_out_of_domain_n > 0:
    fail("outside flag-only range")
analysis_cohort = df.loc[~age_out_of_domain_mask]
"""

    assert _repair(code, _finding(variable="age")) == (code, [])


def test_guard_with_side_effect_is_not_rewritten():
    code = """
age_out_of_domain = (age < 0.0) | (age > 120.0)
if age_out_of_domain.any():
    audit["flag"] = True
    raise ValueError("Age outside plausibility range")
"""

    assert _repair(code, _finding()) == (code, [])


def test_repair_is_registered_as_structural():
    metadata = repair_metadata_for("flag_only_plausibility_range_retention_v1")

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.classification_source == "exact"


def test_host_bound_narrowed_with_float_is_still_the_same_bound():
    """The exact shape E1 r25 step 06 was blocked on.

    The generated validator narrows the JSON bound at the comparison itself
    (``numeric < float(lower)``). Matching only a bare name left this repair
    unable to fire on the one script the auditor actually rejects: it returned
    no repair names and an unchanged script while the run burned its provider
    budget on the same blocked draft.
    """

    code = """
def validate_allowed_numeric(values, column, manifest):
    contract = manifest["raw_input_contracts"]["contracts"].get(column)
    observed = values.dropna()
    bounds = contract.get("analysis_plausibility_range")
    if bounds is not None:
        lower = bounds.get("minimum")
        upper = bounds.get("maximum")
        numeric = observed.astype(float)
        if lower is not None and (numeric < float(lower)).any():
            raise RuntimeError(f"{column} below analysis plausibility minimum")
        if upper is not None and (numeric > float(upper)).any():
            raise RuntimeError(f"{column} above analysis plausibility maximum")
"""

    repaired, names = _repair(code, _finding(variable="age"))

    assert names == ["flag_only_plausibility_range_retention_v1"]
    assert repaired.count("_easyicu_flag_only_plausibility_range_retained_v1") == 2
    assert "below analysis plausibility minimum" not in repaired
    assert "above analysis plausibility maximum" not in repaired
    # The comparison, the bound reads and the coercion are untouched: only the
    # terminal exclusion is removed.
    assert "(numeric < float(lower)).any()" in repaired
    assert "(numeric > float(upper)).any()" in repaired
    assert 'lower = bounds.get("minimum")' in repaired
    assert "numeric = observed.astype(float)" in repaired


def test_only_the_bound_may_be_narrowed_not_the_series():
    """``float(series)`` is a computed operand, not the sealed host bound."""

    code = """
bounds = contract.get("analysis_plausibility_range")
lower = bounds.get("minimum")
upper = bounds.get("maximum")
if lower is not None and (float(numeric) < lower).any():
    raise RuntimeError("below")
if upper is not None and (float(numeric) > upper).any():
    raise RuntimeError("above")
"""

    repaired, names = _repair(code, _finding(variable="age"))

    assert names == []
    assert repaired == code


def test_a_call_that_is_not_a_plain_float_of_the_bound_is_not_a_bound():
    """Anything else around the bound is a computed threshold, not the bound."""

    for expression in (
        "float(lower, 2)",
        "float(lower * 2)",
        "round(lower)",
        "float(x=lower)",
    ):
        code = f"""
bounds = contract.get("analysis_plausibility_range")
lower = bounds.get("minimum")
upper = bounds.get("maximum")
if lower is not None and (numeric < {expression}).any():
    raise RuntimeError("below")
if upper is not None and (numeric > float(upper)).any():
    raise RuntimeError("above")
"""
        repaired, names = _repair(code, _finding(variable="age"))
        assert names == [], expression
        assert repaired == code, expression


def test_retention_repair_removes_the_exclusion_but_does_not_itself_flag():
    """State the half this repair does not do, so nobody reads it as compliance.

    ``retain_and_flag`` is two obligations. A deterministic patch can prove the
    first -- no row is excluded -- because it only deletes a terminal guard. It
    cannot invent the structured flag or count, which belongs to the generated
    script's declared outputs. A run whose only evidence of policy compliance
    is this marker has satisfied retention, not flagging.
    """

    code = """
bounds = contract.get("analysis_plausibility_range")
lower = bounds.get("minimum")
upper = bounds.get("maximum")
if lower is not None and (numeric < float(lower)).any():
    raise RuntimeError("below")
if upper is not None and (numeric > float(upper)).any():
    raise RuntimeError("above")
publish(frame)
"""

    repaired, names = _repair(code, _finding(variable="age"))

    assert names == ["flag_only_plausibility_range_retention_v1"]
    assert "raise RuntimeError" not in repaired
    assert "flag" not in repaired.replace(
        "_easyicu_flag_only_plausibility_range_retained_v1", ""
    )
    assert "out_of_range" not in repaired


def test_the_repair_marker_alone_does_not_claim_the_flagging_half(tmp_path) -> None:
    """Replayed against the real r25 Step 06 script.

    The deterministic repair removes both terminal guards, so out-of-range
    values are retained -- but the only thing it leaves behind is its own
    marker comment. Searching the patched script for flag/count evidence finds
    exactly that marker and nothing else, which is how a reader can mistake
    "retention proven" for "retain_and_flag satisfied".
    """

    code = (
        "def validate_allowed_numeric(values, column, manifest):\n"
        "    contract = manifest['raw_input_contracts']['contracts'].get(column)\n"
        "    bounds = contract.get('analysis_plausibility_range')\n"
        "    if bounds is not None:\n"
        "        lower = bounds.get('minimum')\n"
        "        upper = bounds.get('maximum')\n"
        "        numeric = values.dropna().astype(float)\n"
        "        if lower is not None and (numeric < float(lower)).any():\n"
        "            raise RuntimeError('below analysis plausibility minimum')\n"
        "        if upper is not None and (numeric > float(upper)).any():\n"
        "            raise RuntimeError('above analysis plausibility maximum')\n"
    )

    repaired, names = _repair(code, _finding(variable="numeric"))

    assert names == ["flag_only_plausibility_range_retention_v1"]
    # Retention: both terminal guards are gone, so no row is excluded.
    assert "raise RuntimeError" not in repaired
    # Flagging: the only match for a flag/count search is the marker itself.
    residue = [
        term
        for term in ("_flag", "out_of_range", "flag_count", "flagged")
        if term in repaired
    ]
    assert residue == ["_flag"]
    assert repaired.count("_easyicu_flag_only_plausibility_range_retained_v1") == 2


# --- the flagging half is now gated, not merely noted ------------------------
#
# `retain_and_flag` is two obligations. The host downgrade settles retention by
# overriding the auditor's demand to exclude; flagging is the generated
# script's declared output. Until something observed it, "no error" meant only
# that nobody had looked -- a step could retain the values, flag nothing, and
# pass. These pin what the check can and cannot see.


def _records(script: str, variable: str = "lactate"):
    from easyicu.research_agent.audits.validators import (
        _records_out_of_range_evidence,
    )

    return _records_out_of_range_evidence(script_text=script, variable=variable)


def test_a_bound_comparison_whose_result_is_discarded_is_not_a_flag():
    assert (
        _records(
            """
numeric = pd.to_numeric(df["lactate"])
if lower is not None and (numeric < float(lower)).any():
    pass  # _easyicu_flag_only_plausibility_range_retained_v1
"""
        )
        is False
    )


def test_a_count_computed_only_to_reject_is_not_a_flag():
    assert (
        _records(
            """
numeric = pd.to_numeric(df["lactate"])
if int((numeric > 30.0).sum()) > 0:
    raise ValueError("out of range")
"""
        )
        is False
    )


def test_an_indicator_column_counts_as_the_flag():
    assert (
        _records(
            """
numeric = df["lactate"].astype(float)
df["lactate_out_of_range"] = (numeric < 0.1) | (numeric > 30.0)
"""
        )
        is True
    )


def test_a_kept_count_counts_as_the_flag_even_when_the_mask_is_named_first():
    """The counted shape the deterministic repair leaves behind."""

    assert (
        _records(
            """
numeric = pd.to_numeric(df["lactate"])
mask = (numeric < 0.1) | (numeric > 30.0)
n_out = int(mask.sum())
summary = {"lactate_out_of_range_n": n_out}
"""
        )
        is True
    )


def test_the_repair_marker_comment_is_not_mistaken_for_a_flag():
    """The marker contains the substring `_flag`, and is a comment.

    A text search over the repaired script reports the repair's own marker as
    structured-flag evidence -- that false positive is exactly what an earlier
    replay of a real Step 06 script turned up. Parsing rather than grepping
    makes it structurally impossible: comments are not in the tree.
    """

    script = """
numeric = pd.to_numeric(df["lactate"])
pass  # _easyicu_flag_only_plausibility_range_retained_v1
"""

    assert "_flag" in script, "precondition: the marker is textually present"
    assert _records(script) is None


def test_another_variables_flag_does_not_settle_this_ones():
    assert (
        _records(
            """
other = pd.to_numeric(df["sodium"])
df["sodium_out_of_range"] = other > 150.0
"""
        )
        is None
    )


def test_an_unobservable_flag_abstains_rather_than_blocking():
    """No comparison on the variable at all: the check has nothing to say.

    Deciding here would block a compliant script whose shape this parse does
    not model, so it abstains -- and the finding detail records that it
    abstained, which is the difference between reporting what was observed and
    letting silence read as compliance.
    """

    assert _records('df = df.dropna(subset=["lactate"])') is None
    assert _records("x = 1") is None

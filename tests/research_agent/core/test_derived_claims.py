"""Tests for the derived-claim sandbox and EvidenceStore integration.

Commit 2 (Phase-1 widening, May 2026). Pins:

1. The restricted-AST evaluator accepts allowed expressions and
   rejects every disallowed AST node we know about (attribute,
   subscript, comprehension, comparison, boolop, lambda, named expr,
   private/dunder names, calls to non-whitelisted functions).
2. The evaluator rejects unresolved source names with a helpful
   message that lists what *is* available.
3. ``EvidenceStore.register_derived_claim`` round-trips through
   formula → restricted eval → registered claim with ``formula``,
   ``explanation``, ``derived_from`` set.
4. The derived claim's ``canonical`` matches the formula's value to
   tolerance, and re-registering with the same name idempotently
   updates the entry without duplicating.
5. ``register_step_derived_claims`` walks
   ``step_summary["derived_claims"]`` and surfaces per-entry errors
   without aborting on the first failure.
6. The full claim registry (including derived) round-trips through
   JSON via ``to_dict`` / ``from_dict``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import (
    DerivedFormulaError,
    EvidenceStore,
    NumericClaim,
    _evaluate_derived_formula,
)
from easyicu.research_agent.authority.evidence_snapshot import (
    load_current_evidence_snapshot,
)


# ---------------------------------------------------------------------
# 1. Sandbox accepts allowed
# ---------------------------------------------------------------------


def test_evaluator_arithmetic_and_math_whitelist() -> None:
    src = {"a": 2.0, "b": 3.0, "c": 0.5}
    assert _evaluate_derived_formula("a + b", sources=src) == 5.0
    assert _evaluate_derived_formula("a - b", sources=src) == -1.0
    assert _evaluate_derived_formula("a * b", sources=src) == 6.0
    assert _evaluate_derived_formula("b / a", sources=src) == 1.5
    assert _evaluate_derived_formula("a ** b", sources=src) == 8.0
    assert _evaluate_derived_formula("-a", sources=src) == -2.0
    assert _evaluate_derived_formula("+a", sources=src) == 2.0
    assert _evaluate_derived_formula("abs(-a)", sources=src) == 2.0
    assert _evaluate_derived_formula("min(a, b, c)", sources=src) == 0.5
    assert _evaluate_derived_formula("max(a, b, c)", sources=src) == 3.0
    assert _evaluate_derived_formula("sqrt(a * a)", sources=src) == 2.0


def test_evaluator_log_normal_ci_lower_bound() -> None:
    """The canonical use case: derive low CI bound from OR and SE."""
    result = _evaluate_derived_formula(
        "exp(log(primary_or) - 1.96 * primary_or_se)",
        sources={"primary_or": 1.42, "primary_or_se": 0.13},
    )
    # Expected ≈ exp(0.3507 - 0.2548) = exp(0.0959) ≈ 1.1006
    assert 1.099 <= result <= 1.102


def test_evaluator_pi_and_e_constants() -> None:
    result = _evaluate_derived_formula("pi", sources={})
    assert 3.141 < result < 3.142
    result = _evaluate_derived_formula("e", sources={})
    assert 2.718 < result < 2.719


# ---------------------------------------------------------------------
# 2. Sandbox rejects disallowed
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "a.b",                       # attribute
        "a[0]",                      # subscript
        "[i for i in range(3)]",     # listcomp
        "{i for i in range(3)}",     # setcomp
        "{i: i for i in range(3)}",  # dictcomp
        "a if a else b",             # IfExp
        "a > b",                     # Compare
        "a == b",                    # Compare
        "a and b",                   # BoolOp
        "lambda x: x",               # Lambda
        "True",                      # Constant of disallowed type
        "'hi'",                      # string constant
    ],
)
def test_evaluator_rejects_disallowed_nodes(expr: str) -> None:
    with pytest.raises(DerivedFormulaError):
        _evaluate_derived_formula(expr, sources={"a": 1.0, "b": 2.0})


@pytest.mark.parametrize(
    "expr",
    [
        "eval('1')",
        "open('/etc/passwd')",
        "__import__('os')",
        "exec('1')",
    ],
)
def test_evaluator_rejects_non_whitelist_calls(expr: str) -> None:
    with pytest.raises(DerivedFormulaError, match="rejected call to"):
        _evaluate_derived_formula(expr, sources={})


def test_evaluator_rejects_private_names() -> None:
    with pytest.raises(DerivedFormulaError, match="dunder/private"):
        _evaluate_derived_formula("_secret", sources={"_secret": 1.0})


def test_evaluator_unknown_source_lists_available() -> None:
    with pytest.raises(DerivedFormulaError) as exc:
        _evaluate_derived_formula("missing", sources={"primary_or": 1.42})
    assert "primary_or" in str(exc.value)
    assert "missing" in str(exc.value)


def test_evaluator_division_by_zero_raises_formula_error() -> None:
    with pytest.raises(DerivedFormulaError, match="division by zero"):
        _evaluate_derived_formula("a / b", sources={"a": 1.0, "b": 0.0})


def test_evaluator_math_domain_error_wrapped() -> None:
    # log(0) raises ValueError (math domain error) from the stdlib;
    # the evaluator must re-raise as DerivedFormulaError so callers
    # don't have to know which inputs trigger underlying exceptions.
    with pytest.raises(DerivedFormulaError, match="log"):
        _evaluate_derived_formula("log(a)", sources={"a": 0.0})


def test_evaluator_overflow_to_inf_raises_non_finite() -> None:
    # 1e308 ** 2 overflows to inf; the non-finite guard must catch it.
    with pytest.raises(DerivedFormulaError, match="non-finite"):
        _evaluate_derived_formula("a ** 2", sources={"a": 1e308})


def test_evaluator_math_call_overflow_raises_formula_error() -> None:
    with pytest.raises(DerivedFormulaError, match="failed|overflow"):
        _evaluate_derived_formula("exp(a)", sources={"a": 1000.0})


def test_evaluator_rejects_keyword_args() -> None:
    with pytest.raises(DerivedFormulaError, match="keyword arguments"):
        _evaluate_derived_formula("log(a, base=10)", sources={"a": 100.0})


# ---------------------------------------------------------------------
# 3-4. EvidenceStore.register_derived_claim
# ---------------------------------------------------------------------


def test_register_derived_claim_round_trip(tmp_path: Path) -> None:
    store = EvidenceStore(root=tmp_path)
    # Source claims must exist first.
    store.register_numeric_claim(
        value="1.42", canonical=1.42, evidence_id="03_primary",
        step_id="03_primary", source_field="primary_or",
    )
    store.register_numeric_claim(
        value="0.13", canonical=0.13, evidence_id="03_primary",
        step_id="03_primary", source_field="primary_or_se",
    )
    derived = store.register_derived_claim(
        name="primary_or_ci_low",
        formula="exp(log(primary_or) - 1.96 * primary_or_se)",
        explanation="Lower 95% CI for primary OR, log-normal approx",
        sources={
            "primary_or": ("03_primary", "primary_or"),
            "primary_or_se": ("03_primary", "primary_or_se"),
        },
        evidence_id="03_primary",
        step_id="03_primary",
    )
    assert derived.is_derived
    assert derived.formula.startswith("exp(log(primary_or)")
    assert derived.explanation.startswith("Lower 95% CI")
    assert sorted(derived.derived_from) == [
        ("03_primary", "primary_or"),
        ("03_primary", "primary_or_se"),
    ]
    assert 1.099 < derived.canonical < 1.102


def test_register_derived_claim_unresolved_source(tmp_path: Path) -> None:
    store = EvidenceStore(root=tmp_path)
    with pytest.raises(DerivedFormulaError, match="not found in registry"):
        store.register_derived_claim(
            name="z",
            formula="a + b",
            explanation="dummy",
            sources={
                "a": ("03_primary", "missing_a"),
                "b": ("03_primary", "missing_b"),
            },
            evidence_id="03_primary",
            step_id="03_primary",
        )


def test_register_derived_claim_is_idempotent(tmp_path: Path) -> None:
    store = EvidenceStore(root=tmp_path)
    store.register_numeric_claim(
        value="1.42", canonical=1.42, evidence_id="03",
        step_id="03", source_field="primary_or",
    )
    store.register_numeric_claim(
        value="0.13", canonical=0.13, evidence_id="03",
        step_id="03", source_field="primary_or_se",
    )
    for _ in range(3):
        store.register_derived_claim(
            name="ci_low",
            formula="exp(log(primary_or) - 1.96 * primary_or_se)",
            explanation="lo",
            sources={
                "primary_or": ("03", "primary_or"),
                "primary_or_se": ("03", "primary_or_se"),
            },
            evidence_id="03",
            step_id="03",
        )
    # Only ONE claim with this (step_id, source_field) regardless of
    # how many times we re-register.
    matches = [
        c
        for c in store.numeric_claims()
        if c.step_id == "03" and c.source_field == "ci_low"
    ]
    assert len(matches) == 1


def test_register_derived_claim_new_evidence_id_is_not_deduped(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(root=tmp_path)
    store.register_numeric_claim(
        value="1.42",
        canonical=1.42,
        evidence_id="03_attempt_1",
        step_id="03",
        source_field="primary_or",
    )
    first = store.register_derived_claim(
        name="primary_or_plus_one",
        formula="primary_or + 1",
        explanation="Primary estimate shifted by one.",
        sources={"primary_or": ("03", "primary_or")},
        evidence_id="03_attempt_1",
        step_id="03",
    )
    second = store.register_derived_claim(
        name="primary_or_plus_one",
        formula="primary_or + 1",
        explanation="Primary estimate shifted by one.",
        sources={"primary_or": ("03", "primary_or")},
        evidence_id="03_attempt_2",
        step_id="03",
    )

    matches = [
        claim
        for claim in store.numeric_claims()
        if claim.step_id == "03" and claim.source_field == "primary_or_plus_one"
    ]
    assert first is not second
    assert [claim.evidence_id for claim in matches] == [
        "03_attempt_1",
        "03_attempt_2",
    ]
    assert first.canonical == second.canonical


def test_register_derived_claim_rejects_bad_name(tmp_path: Path) -> None:
    store = EvidenceStore(root=tmp_path)
    with pytest.raises(DerivedFormulaError, match="identifier"):
        store.register_derived_claim(
            name="42_bad",
            formula="1",
            explanation="dummy",
            sources={},
            evidence_id="x",
            step_id="x",
        )


def test_register_derived_claim_requires_explanation(tmp_path: Path) -> None:
    store = EvidenceStore(root=tmp_path)
    with pytest.raises(DerivedFormulaError, match="explanation"):
        store.register_derived_claim(
            name="z",
            formula="1",
            explanation="   ",
            sources={},
            evidence_id="x",
            step_id="x",
        )


# ---------------------------------------------------------------------
# 5. register_step_derived_claims walks the summary
# ---------------------------------------------------------------------


def test_register_step_derived_claims_happy_path(tmp_path: Path) -> None:
    store = EvidenceStore(root=tmp_path)
    summary = {
        "primary_or": 1.42,
        "primary_or_se": 0.13,
        "derived_claims": [
            {
                "name": "primary_or_ci_low",
                "formula": "exp(log(primary_or) - 1.96 * primary_or_se)",
                "explanation": "low 95% CI bound",
                "sources": {
                    "primary_or": "primary_or",      # shorthand → same step
                    "primary_or_se": "primary_or_se",
                },
            },
            {
                "name": "primary_or_ci_high",
                "formula": "exp(log(primary_or) + 1.96 * primary_or_se)",
                "explanation": "high 95% CI bound",
                "sources": {
                    "primary_or": {"step_id": "03", "field": "primary_or"},
                    "primary_or_se": {"step_id": "03", "field": "primary_or_se"},
                },
            },
        ],
    }
    # Source leaves first (the pipeline does this via
    # ``register_step_summary_numerics`` before derived).
    store.register_step_summary_numerics(
        step_id="03", evidence_id="03", summary={
            "primary_or": 1.42, "primary_or_se": 0.13,
        },
    )
    claims, errors = store.register_step_derived_claims(
        step_id="03", evidence_id="03", summary=summary,
    )
    assert errors == []
    assert len(claims) == 2
    names = {c.source_field for c in claims}
    assert names == {"primary_or_ci_low", "primary_or_ci_high"}


def test_register_step_derived_claims_commits_one_authority_generation(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(root=tmp_path)
    store.register_step_summary_numerics(
        step_id="03",
        evidence_id="03_attempt_1",
        summary={"estimate": 2.0},
    )
    before = load_current_evidence_snapshot(tmp_path)
    assert before.generation is not None

    claims, errors = store.register_step_derived_claims(
        step_id="03",
        evidence_id="03_attempt_1",
        summary={
            "derived_claims": [
                {
                    "name": "estimate_plus_one",
                    "formula": "estimate + 1",
                    "explanation": "Primary estimate shifted by one.",
                    "sources": {"estimate": "estimate"},
                },
                {
                    "name": "estimate_doubled",
                    "formula": "estimate * 2",
                    "explanation": "Primary estimate multiplied by two.",
                    "sources": {"estimate": "estimate"},
                },
            ]
        },
    )
    after = load_current_evidence_snapshot(tmp_path)

    assert errors == []
    assert {claim.source_field for claim in claims} == {
        "estimate_plus_one",
        "estimate_doubled",
    }
    assert after.generation == before.generation + 1


def test_register_step_derived_claims_partial_failure(tmp_path: Path) -> None:
    """One bad entry must NOT abort the others. Errors come back as a list."""
    store = EvidenceStore(root=tmp_path)
    store.register_step_summary_numerics(
        step_id="03", evidence_id="03",
        summary={"primary_or": 1.42, "primary_or_se": 0.13},
    )
    summary = {
        "derived_claims": [
            # OK
            {
                "name": "ci_low",
                "formula": "exp(log(primary_or) - 1.96 * primary_or_se)",
                "explanation": "lo",
                "sources": {
                    "primary_or": "primary_or",
                    "primary_or_se": "primary_or_se",
                },
            },
            # Bad — references a non-existent source
            {
                "name": "bogus",
                "formula": "ghost + 1",
                "explanation": "should fail",
                "sources": {"ghost": "ghost_field"},
            },
            # Bad — disallowed AST node
            {
                "name": "evil",
                "formula": "primary_or > 1",
                "explanation": "should also fail",
                "sources": {"primary_or": "primary_or"},
            },
            # OK
            {
                "name": "ci_high",
                "formula": "exp(log(primary_or) + 1.96 * primary_or_se)",
                "explanation": "hi",
                "sources": {
                    "primary_or": "primary_or",
                    "primary_or_se": "primary_or_se",
                },
            },
        ]
    }
    claims, errors = store.register_step_derived_claims(
        step_id="03", evidence_id="03", summary=summary,
    )
    registered_names = {c.source_field for c in claims}
    assert registered_names == {"ci_low", "ci_high"}
    error_names = {e["name"] for e in errors}
    assert error_names == {"bogus", "evil"}


def test_register_step_derived_claims_no_section_is_noop(tmp_path: Path) -> None:
    store = EvidenceStore(root=tmp_path)
    claims, errors = store.register_step_derived_claims(
        step_id="03", evidence_id="03",
        summary={"primary_or": 1.42},  # no derived_claims key
    )
    assert claims == []
    assert errors == []


# ---------------------------------------------------------------------
# 6. Serialisation round-trip
# ---------------------------------------------------------------------


def test_numeric_claim_to_from_dict_round_trips_derived_fields() -> None:
    original = NumericClaim(
        value="1.10",
        canonical=1.1006,
        evidence_id="03",
        step_id="03",
        source_field="primary_or_ci_low",
        formula="exp(log(primary_or) - 1.96 * primary_or_se)",
        explanation="low CI",
        derived_from=[("03", "primary_or"), ("03", "primary_or_se")],
    )
    payload = original.to_dict()
    restored = NumericClaim.from_dict(payload)
    assert restored.formula == original.formula
    assert restored.explanation == original.explanation
    assert restored.derived_from == original.derived_from
    assert restored.is_derived


def test_numeric_claim_to_dict_omits_empty_derived_fields_for_regular_claim() -> None:
    claim = NumericClaim(
        value="1.42",
        canonical=1.42,
        evidence_id="03",
        step_id="03",
        source_field="primary_or",
    )
    payload = claim.to_dict()
    assert "formula" not in payload
    assert "explanation" not in payload
    assert "derived_from" not in payload

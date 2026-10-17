"""A curated hint added since the ICU rules were sealed is read with them.

``icu_rules`` belongs to the Figure 2 paper scorer tree, sealed by digest.
The hints curated after the seal (advanced respiratory support and cumulative
fluid balance, read as support the stay received) live in
``research_context.concept_hints``; every production reader of a column's
role, kind, unit or aggregation reads through it, the context's
implementation identity binds its bytes, and a column the ICU rules read as
audit metadata, an identifier or a time keeps their reading.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from easyicu.research_agent.icu_rules import VariableKind
from easyicu.research_agent.icu_rules import classify_variable as icu_classify_variable
from easyicu.research_agent.research_context import concept_hints
from easyicu.research_agent.research_context.concept_hints import (
    SUPPLEMENTARY_CONCEPT_HINTS,
    classify_variable,
)
from easyicu.research_agent.research_context.implementation_identity import (
    metadata_implementation_identity,
)

_SRC = Path(concept_hints.__file__).resolve().parents[3]


def test_concurrent_support_is_an_intervention_with_its_caution() -> None:
    for name in ("mech_vent", "vaso_ind", "adv_resp", "adv_resp_max"):
        hint = classify_variable(name, "int64", [0, 1])
        assert hint.role.value == "intervention"
        assert hint.kind == VariableKind.BINARY
        assert hint.aggregation_default.value == "max_or_last"
    assert "pre-time-zero authority" in " ".join(
        classify_variable("adv_resp_max", "int64", [0, 1]).pitfalls
    )


def test_cumulative_fluid_balance_is_an_intervention_and_balance_alone_is_not() -> None:
    # Its window summary is support the stay received, as ventilation is; the
    # non-cumulative balance and the observation companions keep their rules.
    hint = classify_variable("fluid_balance_cumulative_mean", "float64")
    assert (hint.role.value, hint.kind, hint.unit) == (
        "intervention", VariableKind.CONTINUOUS, "mL",
    )
    assert hint.aggregation_default.value == "max_or_last"
    assert "understates" in " ".join(hint.pitfalls)
    assert classify_variable("fluid_balance_mean", "float64").role.value == "other"
    assert classify_variable("fluid_balance_cumulative_n", "int64").role.value == "meta"


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        # Audit companions of the concepts here, and a time of one.
        ("adv_resp_n", "int64"),
        ("adv_resp_measured", "int64"),
        ("fluid_balance_cumulative_n_first_24h", "int64"),
        ("adv_resp_time", "datetime64[ns]"),
        # A longer name that merely begins with a concept's letters.
        ("adv_respiratory_rate_mean", "float64"),
        # Concepts the ICU rules name themselves.
        ("lact_max", "float64"),
        ("age", "float64"),
        ("mech_vent", "int64"),
    ],
)
def test_any_other_column_keeps_the_icu_rules_reading(name: str, dtype: str) -> None:
    assert classify_variable(name, dtype) == icu_classify_variable(name, dtype)


def test_a_hint_here_names_a_concept_the_sealed_rules_do_not_read_so() -> None:
    # Were a reseal to add one of these to the ICU rules, it would be read
    # twice; the hint then leaves this owner.
    for concept, hint in SUPPLEMENTARY_CONCEPT_HINTS.items():
        assert icu_classify_variable(concept, "float64") != hint


def test_the_context_identity_binds_this_owner_beside_the_icu_rules() -> None:
    identity = dict(metadata_implementation_identity())
    expected = hashlib.sha256(Path(concept_hints.__file__).read_bytes()).hexdigest()
    assert identity["icu_concept_hints_sha256"] == expected
    assert identity["icu_rules_sha256"] != expected


def _reads_the_sealed_classification(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[-1] == "icu_rules"
            and any(alias.name == "classify_variable" for alias in node.names)
        ):
            lines.append(node.lineno)
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "classify_variable"
            and isinstance(node.value, ast.Name)
            and node.value.id == "ICU_RULES"
        ):
            lines.append(node.lineno)
    return lines


def test_every_production_reader_reads_a_column_through_this_owner() -> None:
    owners = {
        Path(concept_hints.__file__).resolve(),
        (_SRC / "easyicu" / "research_agent" / "icu_rules.py").resolve(),
    }
    readers = {
        str(path.relative_to(_SRC)): lines
        for path in sorted((_SRC / "easyicu").rglob("*.py"))
        if path.resolve() not in owners
        and (lines := _reads_the_sealed_classification(ast.parse(path.read_text(encoding="utf-8"))))
    }
    assert readers == {}

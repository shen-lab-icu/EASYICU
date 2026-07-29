"""A product name must not decide whether the host executes an audit step.

Recognition of a counting-only measurement/missingness audit used to be a
lookup of the declared product id in a fixed alias table.  Measured over the
recorded plan corpus that cost 162 audit steps their deterministic executor:
every declared output a table, no duplicates, an input scope the runner
supports -- demoted to the LLM coder because the Planner wrote
``lactate_missingness`` or ``data_quality_measurement_status`` instead of a
spelling the table already knew.  Ten more were demoted on the *method* string
while their products were recognised perfectly.

``test_the_two_real_missingness_steps_differ_only_in_their_product_names`` is
the load-bearing one: fresh17 and fresh19 planned the same audit, and only the
newer one lost its owner.

``test_two_products_may_not_name_the_same_audit`` is the other.  Three of the
runner's product ids resolved to what were three files holding one identical
table, so a "distinct file" rule could not tell a reader promised two tables
from a reader handed one table twice.  Distinct audits can.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from easyicu.research_agent.execution.runners.deterministic_missingness import (
    MEASUREMENT_AUDIT_KIND_FILES,
    MISSINGNESS_AUDIT_PRODUCT_FILES,
    declared_audit_spec_is_emittable,
    missingness_audit_executor_owns_step,
    missingness_measurement_audit_code,
)
from easyicu.research_agent.schema import (
    MEASUREMENT_AUDIT_KINDS,
    AnalysisStep,
    MeasurementAuditSpec,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "real_plan_steps_fresh17_fresh19.json"


def _real_step(label: str, step_id: str) -> dict:
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    plan = next(e for e in document["plans"] if e["label"] == label)["plan"]
    return next(s for s in plan["steps"] if s["step_id"] == step_id)


def _step(**overrides) -> AnalysisStep:
    payload = {
        "step_id": "06_audit",
        "method": "missingness_and_event_timing_audit",
        "intent": "Count how often each concept was measured.",
        "inputs": ["artifact:analysis_cohort", "lact_max", "sofa2_max"],
        "expected_outputs": ["table:missingness_measurement_audit"],
        "measurement_audit_spec": {
            "products": [
                {
                    "product_id": "missingness_measurement_audit",
                    "audit": "measurement_missingness",
                }
            ]
        },
    }
    payload.update(overrides)
    return AnalysisStep.model_validate(payload)


# --------------------------------------------------------------------------
# the real steps


def test_the_two_real_missingness_steps_differ_only_in_their_product_names() -> None:
    """fresh17 kept its owner; fresh19 planned the same audit and lost it."""

    owned = AnalysisStep.model_validate(
        _real_step("fresh17", "05_missingness_event_timing_audit")
    )
    lost = AnalysisStep.model_validate(
        _real_step("fresh19", "06_missingness_event_timing_audit")
    )

    assert missingness_audit_executor_owns_step(owned) is True
    assert missingness_audit_executor_owns_step(lost) is False

    # The same declaration, said in terms of what each product answers.
    payload = json.loads(lost.model_dump_json())
    payload["measurement_audit_spec"] = {
        "products": [
            {
                "product_id": "missingness_measurement_audit",
                "audit": "measurement_missingness",
            },
            {"product_id": "event_timing_audit", "audit": "event_timing"},
            {
                "product_id": "sofa2_component_completeness_audit",
                "audit": "component_completeness",
            },
        ]
    }
    declared = AnalysisStep.model_validate(payload)

    assert missingness_audit_executor_owns_step(declared) is True


def test_a_name_the_legacy_table_never_saw_is_still_owned() -> None:
    """``lactate_missingness`` is 68 occurrences of a name, not a new audit."""

    step = _step(
        expected_outputs=["table:lactate_missingness"],
        measurement_audit_spec={
            "products": [
                {"product_id": "lactate_missingness", "audit": "missingness_profile"}
            ]
        },
    )

    assert "lactate_missingness" not in MISSINGNESS_AUDIT_PRODUCT_FILES
    assert missingness_audit_executor_owns_step(step) is True


def test_an_unrecognised_method_string_does_not_overrule_the_declaration() -> None:
    """The spec IS the statement that this is an audit step."""

    step = _step(method="post03c_reconciliation_sweep")

    assert missingness_audit_executor_owns_step(step) is True


# --------------------------------------------------------------------------
# what the spec refuses


def test_two_products_may_not_name_the_same_audit() -> None:
    """One table written twice is not two tables."""

    with pytest.raises(ValueError, match="one declared product"):
        MeasurementAuditSpec.model_validate(
            {
                "products": [
                    {
                        "product_id": "measurement_source_audit",
                        "audit": "measurement_source",
                    },
                    {
                        "product_id": "measurement_availability",
                        "audit": "measurement_source",
                    },
                ]
            }
        )


def test_the_three_availability_aliases_are_one_audit_not_three_files() -> None:
    """The rule this replaces counted files, and these were three of them."""

    aliases = [
        "measurement_source_audit",
        "measurement_availability",
        "measurement_availability_audit",
    ]
    files = {MISSINGNESS_AUDIT_PRODUCT_FILES[name] for name in aliases}

    assert len(files) == 1


def test_an_audit_the_host_cannot_compute_is_refused_at_the_contract() -> None:
    with pytest.raises(ValueError, match="unknown measurement audit"):
        MeasurementAuditSpec.model_validate(
            {"products": [{"product_id": "x", "audit": "causal_mediation"}]}
        )


def test_a_declared_product_with_no_audit_is_refused() -> None:
    with pytest.raises(ValueError, match="does not name"):
        _step(
            expected_outputs=[
                "table:missingness_measurement_audit",
                "table:some_other_table",
            ]
        )


def test_an_audit_naming_a_product_the_step_never_declares_is_refused() -> None:
    with pytest.raises(ValueError, match="does not declare"):
        _step(
            measurement_audit_spec={
                "products": [
                    {
                        "product_id": "missingness_measurement_audit",
                        "audit": "measurement_missingness",
                    },
                    {"product_id": "phantom_audit", "audit": "cohort_flow"},
                ]
            }
        )


def test_a_non_table_product_is_refused_rather_than_quietly_uncovered() -> None:
    with pytest.raises(ValueError, match="table products only"):
        _step(
            expected_outputs=[
                "table:missingness_measurement_audit",
                "figure:missingness",
            ]
        )


def test_a_prefixed_product_id_is_refused() -> None:
    with pytest.raises(ValueError, match="bare product name"):
        MeasurementAuditSpec.model_validate(
            {
                "products": [
                    {
                        "product_id": "table:missingness_measurement_audit",
                        "audit": "measurement_missingness",
                    }
                ]
            }
        )


# --------------------------------------------------------------------------
# capability, declared once


def test_every_declarable_audit_has_an_implementation() -> None:
    """A kind the contract accepts but the runner cannot emit would be claimed
    and then fail for a missing product, which is worse than never claiming."""

    assert set(MEASUREMENT_AUDIT_KIND_FILES) == set(MEASUREMENT_AUDIT_KINDS)


def test_every_audit_file_is_actually_written_by_the_runner() -> None:
    """The capability map and the script cannot be allowed to drift.

    Read from the *generated* script, not from this module: the writes live
    inside a rendered template, so parsing the module here would walk a string
    constant and find nothing -- which the first draft of this test did, while
    reporting that all eight files were missing.
    """

    written = {
        node.args[0].right.value
        for node in ast.walk(ast.parse(missingness_measurement_audit_code(_step())))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to_csv"
        and node.args
        and isinstance(node.args[0], ast.BinOp)
        and isinstance(node.args[0].right, ast.Constant)
    }
    missing = sorted(set(MEASUREMENT_AUDIT_KIND_FILES.values()) - written)

    assert written, "no to_csv destination was found; the extraction is broken"
    assert missing == []


def test_the_legacy_shim_resolves_through_the_capability_map() -> None:
    """The shim may only shrink; it must never become a second declaration."""

    assert set(MISSINGNESS_AUDIT_PRODUCT_FILES.values()) <= set(
        MEASUREMENT_AUDIT_KIND_FILES.values()
    )


def test_a_step_without_a_spec_is_not_claimed_by_the_spec_path() -> None:
    payload = json.loads(_step().model_dump_json())
    payload["measurement_audit_spec"] = None

    assert (
        declared_audit_spec_is_emittable(AnalysisStep.model_validate(payload)) is False
    )


# --------------------------------------------------------------------------
# the generated script


def test_the_script_collects_the_declared_names_not_the_legacy_ones() -> None:
    step = _step(
        expected_outputs=["table:lactate_missingness", "table:when_was_it_measured"],
        measurement_audit_spec={
            "products": [
                {"product_id": "lactate_missingness", "audit": "missingness_profile"},
                {
                    "product_id": "when_was_it_measured",
                    "audit": "measurement_process",
                },
            ]
        },
    )
    code = missingness_measurement_audit_code(step)
    compile(code, "<generated>", "exec")
    rendered = code[code.index("product_files = ") :].split("}", 1)[0]

    assert "'lactate_missingness': 'missingness_audit.csv'" in rendered
    assert "'when_was_it_measured': 'measurement_process_audit.csv'" in rendered
    # Only this step's declaration; the shim's names are not smuggled in.
    assert "data_quality_audit" not in rendered

"""Closed deterministic repair for typed all-row outcome display labels."""

from __future__ import annotations

from easyicu.research_agent.repairs.source import _deterministic_runner_repair
from easyicu.research_agent.repairs.typed_input import (
    patch_all_rows_outcome_coordinate_filter,
)
from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)

_SCRIPT = """
def main():
    outcome = load_bound_table("table:outcome_incidence")
    target_outcome = context["target_outcome"]
    outcome = outcome.loc[
        outcome["outcome"].astype(str).eq(str(target_outcome))
    ].copy()
    if outcome.empty:
        raise RuntimeError("No bound outcome-incidence rows for target outcome")
    return outcome
"""

_LOG = """
Traceback (most recent call last):
  File "/easyicu-analysis.py", line 1, in <module>
RuntimeError: No bound outcome-incidence rows for target outcome
"""

_BINDINGS = {
    "table:outcome_incidence": {
        "consumption_contract": {"mode": "all_rows"},
        "identity_row": {
            "declared_kind": "table",
            "input_key": "table:outcome_incidence",
            "product": "outcome_incidence",
        },
        "product_contract": {
            "columns": ["stratum", "outcome", "numerator", "denominator"]
        },
    }
}


def test_exact_all_rows_outcome_filter_is_replaced_by_unique_label_guard() -> None:
    repaired = patch_all_rows_outcome_coordinate_filter(
        _SCRIPT,
        _LOG,
        resolved_input_bindings=_BINDINGS,
    )

    assert repaired != _SCRIPT
    assert ".eq(str(target_outcome))" not in repaired
    assert "_easyicu_bound_outcome_labels" in repaired
    assert "len(_easyicu_bound_outcome_labels) != 1" in repaired
    assert "outcome = outcome.copy()" in repaired


def test_runner_routes_exact_all_rows_repair_before_llm_repair() -> None:
    assert _deterministic_runner_repair(
        code=_SCRIPT,
        run_log=_LOG,
        resolved_input_bindings=_BINDINGS,
    ) == (
        "all_rows_outcome_coordinate_filter_v1",
        patch_all_rows_outcome_coordinate_filter(
            _SCRIPT,
            _LOG,
            resolved_input_bindings=_BINDINGS,
        ),
    )


def test_all_rows_repair_is_registered_as_structural_and_automatic() -> None:
    metadata = repair_metadata_for("all_rows_outcome_coordinate_filter_v1")

    assert metadata.classification_source == "exact"
    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.introduces_numbers is False
    assert automatic_repair_allowed(metadata.repair_id)


def test_all_rows_repair_rejects_unbound_or_subset_consumption() -> None:
    assert (
        patch_all_rows_outcome_coordinate_filter(
            _SCRIPT,
            _LOG,
            resolved_input_bindings=None,
        )
        == _SCRIPT
    )
    subset = {
        **_BINDINGS,
        "table:outcome_incidence": {
            **_BINDINGS["table:outcome_incidence"],
            "consumption_contract": {"mode": "subset"},
        },
    }
    assert (
        patch_all_rows_outcome_coordinate_filter(
            _SCRIPT,
            _LOG,
            resolved_input_bindings=subset,
        )
        == _SCRIPT
    )


def test_all_rows_repair_rejects_wrong_failure_or_ambiguous_shape() -> None:
    assert (
        patch_all_rows_outcome_coordinate_filter(
            _SCRIPT,
            "RuntimeError: something else",
            resolved_input_bindings=_BINDINGS,
        )
        == _SCRIPT
    )
    duplicated = _SCRIPT + "\n" + _SCRIPT.replace("def main", "def another")
    assert (
        patch_all_rows_outcome_coordinate_filter(
            duplicated,
            _LOG,
            resolved_input_bindings=_BINDINGS,
        )
        == duplicated
    )

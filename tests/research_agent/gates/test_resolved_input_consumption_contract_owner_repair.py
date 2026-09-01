"""Closed repair for a typed-input consumption contract read from the wrong owner."""

from __future__ import annotations

from easyicu.research_agent.execution.phase import _untrusted_runtime_repair_allowed
from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.source import _deterministic_runner_repair
from easyicu.research_agent.repairs.typed_input import (
    patch_resolved_input_consumption_contract_owner,
)

_SCRIPT = """
for input_key in typed_inputs:
    binding = manifest_inputs.get(input_key)
    product_contract = binding.get("product_contract")
    if input_key.startswith("table:"):
        consumption_contract = product_contract.get("consumption_contract", {})
        if consumption_contract.get("mode") != "all_rows":
            raise RuntimeError(
                f"Input {input_key} does not have the required all_rows contract"
            )
"""

_LOG = """
Traceback (most recent call last):
RuntimeError: Input table:cluster_characteristics does not have the required all_rows contract
"""

_BINDINGS = {
    "table:cluster_characteristics": {
        "consumption_contract": {"mode": "all_rows", "verified_row_count": 700},
        "identity_row": {
            "declared_kind": "table",
            "input_key": "table:cluster_characteristics",
            "product": "cluster_characteristics",
        },
        "product_contract": {"columns": ["cluster_id", "feature", "median"]},
    }
}


def test_wrong_contract_owner_is_replaced_with_proven_binding_owner() -> None:
    repaired = patch_resolved_input_consumption_contract_owner(
        _SCRIPT,
        _LOG,
        resolved_input_bindings=_BINDINGS,
    )

    assert repaired != _SCRIPT
    assert 'consumption_contract = binding.get("consumption_contract", {})' in repaired
    assert 'product_contract.get("consumption_contract"' not in repaired


def test_runner_routes_exact_contract_owner_repair() -> None:
    repaired = patch_resolved_input_consumption_contract_owner(
        _SCRIPT,
        _LOG,
        resolved_input_bindings=_BINDINGS,
    )

    assert _deterministic_runner_repair(
        code=_SCRIPT,
        run_log=_LOG,
        resolved_input_bindings=_BINDINGS,
    ) == ("resolved_input_consumption_contract_owner_v1", repaired)


def test_contract_owner_repair_is_syntactic_and_automatic() -> None:
    metadata = repair_metadata_for("resolved_input_consumption_contract_owner_v1")

    assert metadata.classification_source == "exact"
    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)
    assert _untrusted_runtime_repair_allowed(
        repair_id=metadata.repair_id,
        source="deterministic_runner_repair",
    )


def test_contract_owner_repair_rejects_unproven_binding_contract() -> None:
    assert (
        patch_resolved_input_consumption_contract_owner(
            _SCRIPT,
            _LOG,
            resolved_input_bindings=None,
        )
        == _SCRIPT
    )
    subset = {
        **_BINDINGS,
        "table:cluster_characteristics": {
            **_BINDINGS["table:cluster_characteristics"],
            "consumption_contract": {"mode": "subset"},
        },
    }
    assert (
        patch_resolved_input_consumption_contract_owner(
            _SCRIPT,
            _LOG,
            resolved_input_bindings=subset,
        )
        == _SCRIPT
    )
    nested = {
        **_BINDINGS,
        "table:cluster_characteristics": {
            **_BINDINGS["table:cluster_characteristics"],
            "product_contract": {
                "columns": ["cluster_id", "feature", "median"],
                "consumption_contract": {"mode": "all_rows"},
            },
        },
    }
    assert (
        patch_resolved_input_consumption_contract_owner(
            _SCRIPT,
            _LOG,
            resolved_input_bindings=nested,
        )
        == _SCRIPT
    )


def test_contract_owner_repair_rejects_wrong_failure_or_ambiguous_shape() -> None:
    assert (
        patch_resolved_input_consumption_contract_owner(
            _SCRIPT,
            "RuntimeError: unrelated failure",
            resolved_input_bindings=_BINDINGS,
        )
        == _SCRIPT
    )
    ambiguous = _SCRIPT + _SCRIPT
    assert (
        patch_resolved_input_consumption_contract_owner(
            ambiguous,
            _LOG,
            resolved_input_bindings=_BINDINGS,
        )
        == ambiguous
    )


def test_correct_contract_owner_is_unchanged() -> None:
    correct = _SCRIPT.replace(
        'product_contract.get("consumption_contract", {})',
        'binding.get("consumption_contract", {})',
    )

    assert (
        patch_resolved_input_consumption_contract_owner(
            correct,
            _LOG,
            resolved_input_bindings=_BINDINGS,
        )
        == correct
    )

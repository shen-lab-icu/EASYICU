"""Consumer input scope must not be confused with physical Parquet schema."""

from __future__ import annotations

import ast

import pandas as pd

from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.execution.phase import _untrusted_runtime_repair_allowed
from easyicu.research_agent.repairs.source import _deterministic_runner_repair

_ERROR = """
RuntimeError: locked cohort columns do not match the exact declared input scope
"""

_SCRIPT = """
import pandas as pd

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

expected_planner_inputs = ["stay_id", "exposure", "outcome"]
df = FRAME
require(
    len(df.columns) == len(expected_planner_inputs)
    and len(set(df.columns)) == len(df.columns)
    and set(df.columns) == set(expected_planner_inputs),
    "locked cohort columns do not match the exact declared input scope",
)
"""


def _run(script: str, frame: pd.DataFrame) -> None:
    exec(script, {"FRAME": frame})  # noqa: S102 - generated-code regression


def test_runner_repair_accepts_physical_superset_without_dropping_columns() -> None:
    repair = _deterministic_runner_repair(code=_SCRIPT, run_log=_ERROR)

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "raw_input_physical_superset_guard_v1"
    assert "set(df.columns) == set(expected_planner_inputs)" not in repaired
    assert "all(_easyicu_input in (df).columns" in repaired
    ast.parse(repaired)

    frame = pd.DataFrame(
        {
            "stay_id": [1],
            "exposure": [0],
            "outcome": [1],
            "unrelated_locked_column": [42],
        }
    )
    original_columns = list(frame.columns)
    _run(repaired, frame)
    assert list(frame.columns) == original_columns


def test_runner_repair_remains_fail_closed_for_missing_declared_column() -> None:
    repair = _deterministic_runner_repair(code=_SCRIPT, run_log=_ERROR)

    assert repair is not None
    _, repaired = repair
    frame = pd.DataFrame({"stay_id": [1], "exposure": [0]})
    try:
        _run(repaired, frame)
    except RuntimeError as exc:
        assert "locked cohort columns" in str(exc)
    else:
        raise AssertionError("missing declared input must still fail closed")


def test_runner_repair_requires_matching_runtime_and_authored_diagnostics() -> None:
    assert (
        _deterministic_runner_repair(
            code=_SCRIPT,
            run_log="RuntimeError: some unrelated failure",
        )
        is None
    )
    forged = _SCRIPT.replace(
        "locked cohort columns do not match the exact declared input scope",
        "unrelated assertion",
    )
    assert _deterministic_runner_repair(code=forged, run_log=_ERROR) is None


def test_raw_input_superset_repair_is_structural_and_automatic() -> None:
    metadata = repair_metadata_for("raw_input_physical_superset_guard_v1")

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)


def test_runner_repair_projects_bare_planner_raw_inputs_without_split() -> None:
    code = '''
manifest = DOCUMENT
planner_inputs = manifest["planner_declared_inputs"]
declared_raw = {
    item.split(":", 1)[1]
    for item in planner_inputs
    if not item.startswith("artifact:") and not item.startswith("table:")
}
'''
    run_log = "IndexError: list index out of range"

    repair = _deterministic_runner_repair(code=code, run_log=run_log)

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "planner_declared_raw_input_projection_v1"
    assert "split" not in repaired
    namespace = {
        "DOCUMENT": {
            "planner_declared_inputs": [
                "age",
                "death",
                "artifact:analysis_cohort",
                "table:adjusted_estimates",
            ]
        }
    }
    exec(repaired, namespace)  # noqa: S102 - generated-code regression
    assert namespace["declared_raw"] == {"age", "death"}
    metadata = repair_metadata_for(repair_id)
    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert _untrusted_runtime_repair_allowed(
        repair_id=repair_id,
        source="deterministic_runner_repair",
    )


def test_planner_raw_input_projection_repair_is_traceback_and_shape_bound() -> None:
    code = '''
planner_inputs = manifest["planner_declared_inputs"]
declared_raw = {item.split(":", 1)[1] for item in planner_inputs}
'''

    assert _deterministic_runner_repair(code=code, run_log="") is None
    assert (
        _deterministic_runner_repair(
            code=code.replace('"planner_declared_inputs"', '"other_inputs"'),
            run_log="IndexError: list index out of range",
        )
        is None
    )
    ambiguous = code + code.replace("declared_raw", "other_raw")
    assert (
        _deterministic_runner_repair(
            code=ambiguous,
            run_log="IndexError: list index out of range",
        )
        is None
    )


def test_runner_repair_iterates_column_keyed_raw_contract_values() -> None:
    code = """
def find_contract(manifest, column_name):
    contracts_root = manifest.get("raw_input_contracts", {})
    contracts = contracts_root.get("contracts", [])
    for contract in contracts:
        if contract.get("column") == column_name:
            return contract
    return None
"""
    run_log = "AttributeError: 'str' object has no attribute 'get'"

    repair = _deterministic_runner_repair(code=code, run_log=run_log)

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "raw_contract_mapping_iteration_v1"
    assert "for contract in contracts.values():" in repaired
    namespace: dict[str, object] = {}
    exec(repaired, namespace)  # noqa: S102 - generated-code regression
    manifest = {
        "raw_input_contracts": {
            "contracts": {
                "eligibility_max": {
                    "column": "eligibility_max",
                    "allowed_values": [0, 1],
                }
            }
        }
    }
    assert namespace["find_contract"](manifest, "eligibility_max") == {
        "column": "eligibility_max",
        "allowed_values": [0, 1],
    }
    metadata = repair_metadata_for(repair_id)
    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert _untrusted_runtime_repair_allowed(
        repair_id=repair_id,
        source="deterministic_runner_repair",
    )


def test_raw_contract_mapping_repair_is_traceback_and_shape_bound() -> None:
    code = """
contracts = manifest.get("contracts", {})
for contract in contracts:
    print(contract.get("column"))
"""
    error = "AttributeError: 'str' object has no attribute 'get'"

    assert _deterministic_runner_repair(code=code, run_log="") is None
    assert (
        _deterministic_runner_repair(
            code=code.replace('"contracts"', '"other_records"'),
            run_log=error,
        )
        is None
    )
    ambiguous = (
        code
        + """
more_contracts = other_manifest.get("contracts", {})
for more_contract in more_contracts:
    print(more_contract.get("column"))
"""
    )
    assert _deterministic_runner_repair(code=ambiguous, run_log=error) is None


def test_runner_repair_preserves_unwrapped_raw_contract_document() -> None:
    code = """
resolved_inputs = DOCUMENT
contracts = (
    resolved_inputs.get("manifest", {})
    .get("raw_input_contracts", {})
    .get("contracts", {})
)
resolved_column = "eligibility_max"
contract = contracts.get(resolved_column)
if contract is None:
    raise RuntimeError(f"Missing raw input contract for {resolved_column}")
"""
    run_log = "RuntimeError: Missing raw input contract for eligibility_max"

    repair = _deterministic_runner_repair(code=code, run_log=run_log)

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "raw_contract_document_fallback_v1"
    assert 'resolved_inputs.get("manifest", resolved_inputs)' in repaired
    namespace = {
        "DOCUMENT": {
            "raw_input_contracts": {
                "contracts": {
                    "eligibility_max": {
                        "column": "eligibility_max",
                        "allowed_values": [0, 1],
                    }
                }
            }
        }
    }
    exec(repaired, namespace)  # noqa: S102 - generated-code regression
    assert namespace["contract"]["allowed_values"] == [0, 1]
    metadata = repair_metadata_for(repair_id)
    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert _untrusted_runtime_repair_allowed(
        repair_id=repair_id,
        source="deterministic_runner_repair",
    )


def test_raw_contract_document_repair_is_failure_and_shape_bound() -> None:
    code = """
contracts = (
    document.get("manifest", {})
    .get("raw_input_contracts", {})
    .get("contracts", {})
)
"""
    error = "RuntimeError: Missing raw input contract for eligibility_max"

    assert _deterministic_runner_repair(code=code, run_log="unrelated") is None
    assert (
        _deterministic_runner_repair(
            code=code.replace('.get("contracts", {})', '.get("other", {})'),
            run_log=error,
        )
        is None
    )
    ambiguous = code + code.replace("contracts =", "other_contracts =")
    assert _deterministic_runner_repair(
        code=ambiguous,
        run_log=error,
    ) is None


def test_runner_repair_accepts_column_keyed_contracts_asserted_as_a_list() -> None:
    """The real E1 step 06 shape: a list guard fires before any AttributeError."""

    code = """
def get_raw_contracts(manifest, required):
    contracts = manifest.get("raw_input_contracts", {}).get("contracts")
    if not isinstance(contracts, list):
        raise ValueError("raw_input_contracts.contracts is missing")
    by_name = {}
    for contract in contracts:
        if not isinstance(contract, dict) or not contract.get("column"):
            raise ValueError("Malformed raw input contract")
        by_name[contract["column"]] = contract
    for name in required:
        if name not in by_name:
            raise ValueError("Missing executable raw input contract for " + name)
    return by_name
"""
    run_log = (
        "ValueError: raw_input_contracts.contracts is missing\n"
        '  File "/easyicu-analysis.py", line 102, in get_raw_contracts\n'
    )

    repair = _deterministic_runner_repair(code=code, run_log=run_log)

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "raw_contract_list_type_assertion_v1"
    assert "isinstance(contracts, dict)" in repaired
    assert "for contract in contracts.values():" in repaired

    namespace: dict[str, object] = {}
    exec(repaired, namespace)  # noqa: S102 - generated-code regression
    manifest = {
        "raw_input_contracts": {
            "contracts": {
                "exposure_max": {"column": "exposure_max", "allowed_values": [0, 1]},
                "outcome": {"column": "outcome", "allowed_values": [0, 1]},
            }
        }
    }
    assert sorted(
        namespace["get_raw_contracts"](manifest, ["exposure_max", "outcome"])
    ) == ["exposure_max", "outcome"]

    # A genuinely absent contract still fails closed after the repair.
    try:
        namespace["get_raw_contracts"](manifest, ["exposure_max", "never_declared"])
    except ValueError as exc:
        assert "never_declared" in str(exc)
    else:
        raise AssertionError("an undeclared column was accepted after repair")

    metadata = repair_metadata_for(repair_id)
    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.introduces_numbers is False
    assert automatic_repair_allowed(repair_id)
    assert _untrusted_runtime_repair_allowed(
        repair_id=repair_id,
        source="deterministic_runner_repair",
    )


def test_list_type_assertion_repair_reads_a_wrapped_manifest_too() -> None:
    code = """
def get_raw_contracts(document):
    manifest = document.get("manifest", document)
    contracts = manifest.get("raw_input_contracts", {}).get("contracts")
    if not isinstance(contracts, list):
        raise ValueError("raw_input_contracts['contracts'] is not available")
    by_name = {}
    for contract in contracts:
        by_name[contract["column"]] = contract
    return by_name
"""
    run_log = "ValueError: raw_input_contracts['contracts'] is not available"

    repair = _deterministic_runner_repair(code=code, run_log=run_log)
    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "raw_contract_list_type_assertion_v1"

    namespace: dict[str, object] = {}
    exec(repaired, namespace)  # noqa: S102 - generated-code regression
    contracts = {"age": {"column": "age"}}
    unwrapped = {"raw_input_contracts": {"contracts": contracts}}
    assert namespace["get_raw_contracts"](unwrapped) == contracts
    assert namespace["get_raw_contracts"]({"manifest": unwrapped}) == contracts


def test_list_type_assertion_repair_is_failure_and_shape_bound() -> None:
    code = """
contracts = manifest.get("raw_input_contracts", {}).get("contracts")
if not isinstance(contracts, list):
    raise ValueError("raw_input_contracts.contracts is missing")
for contract in contracts:
    print(contract.get("column"))
"""
    error = "ValueError: raw_input_contracts.contracts is missing"

    # An unrelated failure must not license the rewrite.
    assert _deterministic_runner_repair(code=code, run_log="") is None
    assert (
        _deterministic_runner_repair(
            code=code,
            run_log="ZeroDivisionError: division by zero",
        )
        is None
    )
    # A list assertion over some other mapping is not this defect.
    assert (
        _deterministic_runner_repair(
            code=code.replace('.get("contracts")', '.get("other_records")'),
            run_log=error,
        )
        is None
    )
    # A loop that never re-keys by column is not the recognised shape.
    assert (
        _deterministic_runner_repair(
            code=code.replace('contract.get("column")', "contract"),
            run_log=error,
        )
        is None
    )
    # Two candidate assertions are ambiguous, so nothing is rewritten.
    ambiguous = (
        code
        + """
other = payload.get("raw_input_contracts", {}).get("contracts")
if not isinstance(other, list):
    raise ValueError("raw_input_contracts.contracts is missing")
for entry in other:
    print(entry.get("column"))
"""
    )
    assert _deterministic_runner_repair(code=ambiguous, run_log=error) is None

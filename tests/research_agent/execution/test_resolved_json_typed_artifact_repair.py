from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.source import _deterministic_runner_repair
from easyicu.research_agent.repairs.typed_artifact import (
    patch_resolved_json_document_adapter,
)

_SCRIPT = """
import json
from pathlib import Path
import pandas as pd

def load_tabular(path):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported tabular input format: {path}")

def load_one(path, contract, expected_key):
    table = load_tabular(path)
    expected_columns = contract.get("columns")
    if not isinstance(expected_columns, list) or not expected_columns:
        raise ValueError(f"Missing product_contract.columns for {expected_key}")
    if list(table.columns) != list(expected_columns):
        raise ValueError("schema mismatch")
    return table
"""


def _runtime(path: Path) -> str:
    return f"ValueError: Unsupported tabular input format: {path}"


def test_v1_json_object_is_adapted_without_changing_tabular_paths(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "audit.json"
    json_path.write_text(json.dumps({"checked_n": 17, "status": "ok"}))
    csv_path = tmp_path / "table.csv"
    pd.DataFrame({"value": [1, 2]}).to_csv(csv_path, index=False)

    repaired = patch_resolved_json_document_adapter(_SCRIPT, _runtime(json_path))
    namespace: dict[str, object] = {}
    exec(repaired, namespace)

    load_one = namespace["load_one"]
    document = load_one(
        json_path,
        {"schema_version": "easyicu.host_typed_product.v1"},
        "artifact:audit",
    )
    table = load_one(
        csv_path,
        {"columns": ["value"], "schema_version": "easyicu.host_typed_product.v4"},
        "table:values",
    )

    assert document.to_dict(orient="records") == [{"checked_n": 17, "status": "ok"}]
    assert table["value"].tolist() == [1, 2]


def test_json_array_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text("[1, 2]")
    repaired = patch_resolved_json_document_adapter(_SCRIPT, _runtime(path))
    namespace: dict[str, object] = {}
    exec(repaired, namespace)

    try:
        namespace["load_one"](
            path,
            {"schema_version": "easyicu.host_typed_product.v1"},
            "artifact:audit",
        )
    except ValueError as exc:
        assert "must contain an object" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("JSON arrays must not be accepted as typed documents")


def test_json_does_not_bypass_newer_tabular_contract(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text('{"checked_n": 17}')
    repaired = patch_resolved_json_document_adapter(_SCRIPT, _runtime(path))
    namespace: dict[str, object] = {}
    exec(repaired, namespace)

    try:
        namespace["load_one"](
            path,
            {"schema_version": "easyicu.host_typed_product.v4"},
            "artifact:audit",
        )
    except ValueError as exc:
        assert "Missing product_contract.columns" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("v4 tabular contracts must retain exact schema checks")


def test_unrelated_runtime_or_source_shape_is_not_rewritten() -> None:
    assert patch_resolved_json_document_adapter(_SCRIPT, "KeyError: x") == _SCRIPT
    unsupported_shape = _SCRIPT.replace(
        'raise ValueError(f"Unsupported tabular input format: {path}")',
        'raise RuntimeError(f"Unknown input: {path}")',
    )
    assert (
        patch_resolved_json_document_adapter(
            unsupported_shape,
            "Unsupported tabular input format: audit.json",
        )
        == unsupported_shape
    )


def test_runtime_router_and_registry_authorize_transport_only_repair() -> None:
    repair = _deterministic_runner_repair(
        code=_SCRIPT,
        run_log="ValueError: Unsupported tabular input format: audit.json",
    )
    assert repair is not None
    assert repair[0] == "resolved_input_json_document_adapter_v1"

    metadata = repair_metadata_for(repair[0])
    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(repair[0])

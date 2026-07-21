from __future__ import annotations

import ast

from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.execution.phase import _untrusted_runtime_repair_allowed
from easyicu.research_agent.repairs.source import _deterministic_runner_repair
from easyicu.research_agent.repairs.categorical import (
    patch_categorical_declared_order_check,
)
from easyicu.research_agent.repairs.typed_artifact import (
    patch_non_tabular_companion_row_gate,
)

_INPUT_KEY = "artifact:quality_summary"
_ERROR = (
    "ValueError: Typed input artifact:quality_summary lacks a tabular product "
    "contract"
)
_BINDINGS = {
    _INPUT_KEY: {
        "relative_path": "evidence/quality_summary.json",
        "evidence_kind": "log",
        "product_contract": {"schema_version": "easyicu.host_typed_product.v1"},
    }
}
_SCRIPT = """
summary_binding, summary_obj = load_bound_input(manifest, "artifact:quality_summary")
input_bindings.append({
    "input_key": "artifact:quality_summary",
    "evidence_id": summary_binding["evidence_id"],
    "sha256": summary_binding["sha256"],
    "loaded": True,
    "row_count": int(len(summary_obj)),
})
summary_contract, summary_columns = require_tabular_contract(
    summary_binding, "artifact:quality_summary"
)
if not isinstance(summary_obj, pd.DataFrame):
    raise ValueError("summary must be tabular")
cohort_df = pd.read_parquet(COHORT_PATH)
if "row_id" not in summary_obj.columns:
    raise ValueError("summary lacks row_id")
summary_key_columns = summary_contract.get("key_columns")
if summary_key_columns != ["row_id"]:
    raise ValueError("wrong key")
summary_column = find_declared_column(summary_contract, summary_columns)
if summary_column not in summary_obj.columns:
    raise ValueError("missing declared selector")
if summary_obj["row_id"].duplicated().any():
    raise ValueError("duplicate key")
if set(summary_obj["row_id"]) != set(cohort_df["row_id"]):
    raise ValueError("incomplete summary")
summary_use = summary_obj[["row_id", summary_column]].copy()
df = cohort_df.merge(summary_use, on="row_id", how="left", validate="one_to_one")
if len(df) != len(cohort_df):
    raise ValueError("row count changed")
summary_raw = df[summary_column]
if summary_raw.dtype == object:
    summary_valid = summary_raw.astype("string").eq("valid")
else:
    summary_numeric = strict_numeric(summary_raw, summary_column)
    summary_valid = summary_numeric.eq(1)
measured = df["measured"].eq(1)
value_present = df["value"].notna()
analysis_mask = measured & summary_valid & value_present
invalid_n = int((measured & (~summary_valid | ~value_present)).sum())
result = fit_model(df.loc[analysis_mask])
""".lstrip()


def test_v1_json_summary_cannot_act_as_patient_level_selector() -> None:
    repaired = patch_non_tabular_companion_row_gate(
        _SCRIPT,
        _ERROR,
        resolved_input_bindings=_BINDINGS,
    )

    ast.parse(repaired)
    assert repaired != _SCRIPT
    assert "document_key_count" in repaired
    assert "summary_obj.columns" not in repaired
    assert "summary_valid" not in repaired
    assert "df = cohort_df.copy()" in repaired
    assert "analysis_mask = measured & value_present" in repaired
    assert "invalid_n = int((measured & ~value_present).sum())" in repaired
    assert 'df["measured"].eq(1)' in repaired
    assert 'df["value"].notna()' in repaired
    assert "fit_model" in repaired


def test_traceback_source_echo_cannot_shadow_actual_input_key() -> None:
    traceback = """Traceback (most recent call last):
  File "/easyicu-analysis.py", line 156, in require_tabular_contract
    raise ValueError(f"Typed input {key} lacks a tabular product contract")
ValueError: Typed input artifact:quality_summary lacks a tabular product contract
"""

    repaired = patch_non_tabular_companion_row_gate(
        _SCRIPT,
        traceback,
        resolved_input_bindings=_BINDINGS,
    )

    assert repaired != _SCRIPT
    assert "summary_valid" not in repaired
    assert "df = cohort_df.copy()" in repaired


def test_runtime_router_requires_host_binding_and_registers_structural_repair() -> None:
    assert _deterministic_runner_repair(code=_SCRIPT, run_log=_ERROR) is None
    repair = _deterministic_runner_repair(
        code=_SCRIPT,
        run_log=_ERROR,
        resolved_input_bindings=_BINDINGS,
    )
    assert repair is not None
    assert repair[0] == "non_tabular_companion_row_gate_v1"
    metadata = repair_metadata_for(repair[0])
    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert automatic_repair_allowed(repair[0])
    assert _untrusted_runtime_repair_allowed(
        repair_id=repair[0], source="deterministic_runner_repair"
    )
    assert not _untrusted_runtime_repair_allowed(
        repair_id="audit_only_companion_value_selector_v1",
        source="deterministic_runner_repair",
    )


def test_tabular_or_untrusted_binding_cannot_authorize_row_gate_removal() -> None:
    tabular = {
        _INPUT_KEY: {
            **_BINDINGS[_INPUT_KEY],
            "product_contract": {
                "schema_version": "easyicu.host_typed_product.v4",
                "columns": ["row_id", "valid"],
            },
        }
    }
    assert (
        patch_non_tabular_companion_row_gate(
            _SCRIPT,
            _ERROR,
            resolved_input_bindings=tabular,
        )
        == _SCRIPT
    )
    assert (
        patch_non_tabular_companion_row_gate(
            _SCRIPT,
            _ERROR,
            resolved_input_bindings=None,
        )
        == _SCRIPT
    )


def test_mixed_companion_expression_fails_closed() -> None:
    ambiguous = _SCRIPT.replace(
        "measured & summary_valid & value_present",
        "measured & (summary_valid == value_present)",
    )
    assert (
        patch_non_tabular_companion_row_gate(
            ambiguous,
            _ERROR,
            resolved_input_bindings=_BINDINGS,
        )
        == ambiguous
    )


_CATEGORICAL_SCRIPT = """
frame["group"] = pd.qcut(
    frame["value"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
)
observed_levels = list(frame["group"].astype("string").dropna().unique())
if observed_levels != ["Q1", "Q2", "Q3", "Q4"]:
    raise ValueError("Quartile construction did not produce four ordered levels")
model = fit(frame, exposure="group")
""".lstrip()


def test_categorical_order_check_uses_declared_not_row_encounter_order() -> None:
    log = "ValueError: Quartile construction did not produce four ordered levels"
    repaired = patch_categorical_declared_order_check(_CATEGORICAL_SCRIPT, log)
    ast.parse(repaired)
    assert repaired != _CATEGORICAL_SCRIPT
    assert 'list(map(str, frame["group"].cat.categories))' in repaired
    assert "pd.qcut" in repaired
    assert 'labels=["Q1", "Q2", "Q3", "Q4"]' in repaired
    assert 'fit(frame, exposure="group")' in repaired

    routed = _deterministic_runner_repair(code=_CATEGORICAL_SCRIPT, run_log=log)
    assert routed is not None
    assert routed[0] == "categorical_declared_order_check_v1"
    assert repair_metadata_for(routed[0]).repair_class is RepairClass.SYNTACTIC


def test_categorical_order_repair_refuses_nonliteral_or_ambiguous_checks() -> None:
    log = "ValueError: Quartile construction did not produce four ordered levels"
    dynamic = _CATEGORICAL_SCRIPT.replace(
        '["Q1", "Q2", "Q3", "Q4"]:', "expected_levels:"
    )
    assert patch_categorical_declared_order_check(dynamic, log) == dynamic
    duplicated = _CATEGORICAL_SCRIPT + _CATEGORICAL_SCRIPT.replace(
        "observed_levels", "other_levels"
    )
    assert patch_categorical_declared_order_check(duplicated, log) == duplicated

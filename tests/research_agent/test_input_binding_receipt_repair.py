from __future__ import annotations

from easyicu.research_agent.repairs.input_binding_receipt import (
    patch_missing_loaded_input_binding_receipt,
)
from easyicu.research_agent.repair_registry import RepairClass, repair_metadata_for
from easyicu.research_agent.repairs.source import deterministic_contract_repair


FINDING = {
    "validator": "step_summary_integrity",
    "severity": "error",
    "detail": {
        "issue": "input_binding_load_contract_invalid",
        "input_key": "table:cohort_flow",
        "invalid_fields": ["loaded"],
    },
}

CODE = '''
def load_bound_table(input_key, record):
    frame = read_table(record["relative_path"])
    return frame, {
        "input_key": input_key,
        "evidence_id": record["evidence_id"],
        "sha256": record["sha256"],
        "row_count": int(len(frame)),
    }
'''


def test_patch_adds_loaded_only_when_receipt_counts_returned_frame() -> None:
    repaired = patch_missing_loaded_input_binding_receipt(
        CODE,
        findings=[FINDING],
    )

    assert repaired != CODE
    assert 'return frame, {"loaded": True, ' in repaired


def test_contract_repair_routes_proven_loaded_receipt() -> None:
    repair = deterministic_contract_repair(code=CODE, findings=[FINDING])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "missing_loaded_input_binding_receipt_v1"
    assert '"loaded": True' in repaired
    assert repair_metadata_for(repair_id).repair_class is RepairClass.SYNTACTIC


def test_patch_refuses_ambiguous_or_incomplete_receipt() -> None:
    ambiguous = CODE.replace("len(frame)", "len(other)")
    incomplete = CODE.replace(',\n        "row_count": int(len(frame))', "")

    assert (
        patch_missing_loaded_input_binding_receipt(ambiguous, findings=[FINDING])
        == ambiguous
    )
    assert (
        patch_missing_loaded_input_binding_receipt(incomplete, findings=[FINDING])
        == incomplete
    )


def test_patch_refuses_when_row_count_is_also_invalid() -> None:
    finding = {
        **FINDING,
        "detail": {**FINDING["detail"], "invalid_fields": ["loaded", "row_count"]},
    }

    assert patch_missing_loaded_input_binding_receipt(CODE, findings=[finding]) == CODE

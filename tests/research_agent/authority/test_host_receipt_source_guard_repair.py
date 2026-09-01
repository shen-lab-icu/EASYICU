from __future__ import annotations

import ast
from pathlib import Path

from easyicu.research_agent.repairs.provenance_summary import (
    patch_direct_host_receipt_source_guard,
)


REAL_RUN_ERROR = "RuntimeError: Measurement provenance source is not COHORT_PARQUET"


def _failing_code() -> str:
    return """
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

measurement_audit = measurement_provenance_receipt(
    df,
    measured_column="sep3_sofa2_measured",
    count_column="sep3_sofa2_n",
)
require(
    isinstance(measurement_audit, dict),
    "measurement_provenance_receipt did not return a mapping",
)
require(
    measurement_audit.get("source") == "COHORT_PARQUET",
    "Measurement provenance source is not COHORT_PARQUET",
)
step_summary = {
    "measurement_provenance_audit": measurement_audit,
    "output_files": {"artifact:analysis_cohort": "analysis_cohort.parquet"},
}
"""


def test_real_e1_receipt_source_guard_is_removed_and_enveloped() -> None:
    repaired = patch_direct_host_receipt_source_guard(
        _failing_code(),
        REAL_RUN_ERROR,
    )

    ast.parse(repaired)
    assert "measurement_audit.get" not in repaired
    assert "did not return a mapping" not in repaired
    assert (
        '"measurement_provenance_audit": '
        '{"source": "COHORT_PARQUET", "checks": [measurement_audit]}'
    ) in repaired
    assert repaired.count("measurement_provenance_receipt(") == 1


def test_receipt_source_guard_repair_requires_exact_runtime_error() -> None:
    code = _failing_code()
    assert patch_direct_host_receipt_source_guard(code, "other failure") == code


def test_receipt_source_guard_repair_rejects_custom_helper() -> None:
    code = _failing_code().replace(
        "from easyicu.research_agent.methods.descriptive_inputs import (\n"
        "    measurement_provenance_receipt,\n"
        ")\n",
        "def measurement_provenance_receipt(frame, **kwargs):\n"
        '    return {"source": "COHORT_PARQUET"}\n',
    )
    assert patch_direct_host_receipt_source_guard(code, REAL_RUN_ERROR) == code


def test_archived_real_e1_script_patches_when_available() -> None:
    path = Path(
        "/Volumes/外置硬盘/easyicu_data/canonical9_runs/"
        "batch_20260723_luna_miiv_adaptive_1a063ca/"
        "e1_sepsis3_prevalence_mortality/aware/"
        "run_20260723T103131_8df069/steps/01_cohort_definition/analysis.py"
    )
    if not path.is_file():
        return
    original = path.read_text(encoding="utf-8")
    repaired = patch_direct_host_receipt_source_guard(original, REAL_RUN_ERROR)
    ast.parse(repaired)
    assert repaired != original
    assert "measurement_audit.get" not in repaired
    assert '"checks": [measurement_audit]' in repaired

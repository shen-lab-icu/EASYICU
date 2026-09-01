from __future__ import annotations

import numpy as np
import pandas as pd

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repairs.reasons import (
    RepairReason,
    repair_reason_for_finding,
)
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair


def _step(ra):
    return ra.AnalysisStep(
        step_id="numeric_summary",
        intent="Summarize a declared numeric input.",
        inputs=["table:declared_numeric_values"],
        expected_outputs=["table:numeric_summary"],
        method="descriptive_statistics",
    )


def _container_findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason") == "pandas_numeric_container_unverified"
    ]


_UNSAFE = """
import pandas as pd

def numeric_values(values, label):
    converted = pd.to_numeric(values, errors="coerce")
    original = pd.Series(values)
    newly_missing = original.notna() & converted.isna()
    if newly_missing.any():
        raise ValueError(label)
    return converted[converted.notna()]
"""


def test_array_like_to_numeric_result_is_flagged_before_execution(ra):
    findings = _container_findings(_UNSAFE, ra)
    assert len(findings) == 1
    assert repair_reason_for_finding(findings[0]) is (
        RepairReason.PANDAS_NUMERIC_CONTAINER_UNVERIFIED
    )


def test_deterministic_repair_normalizes_result_without_changing_values(ra):
    findings = _container_findings(_UNSAFE, ra)
    repaired, repair_names = deterministic_concept_audit_repair(
        _UNSAFE,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
        step=_step(ra),
    )
    assert repair_names == ["pandas_numeric_container_v1"]
    assert 'converted = pd.Series(pd.to_numeric(values, errors="coerce"))' in repaired
    assert _container_findings(repaired, ra) == []

    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    result = namespace["numeric_values"](np.asarray([1.0, np.nan, 2.0]), "x")
    assert isinstance(result, pd.Series)
    assert result.tolist() == [1.0, 2.0]


def test_series_normalized_before_conversion_is_accepted(ra):
    script = """
import pandas as pd

def numeric_values(values):
    values = pd.Series(values)
    converted = pd.to_numeric(values, errors="coerce")
    return converted[converted.notna()]
"""
    assert _container_findings(script, ra) == []


def test_module_level_dataframe_column_conversion_is_not_overclaimed(ra):
    script = """
import pandas as pd

converted = pd.to_numeric(cohort["value"], errors="coerce")
valid = converted.notna()
"""
    assert _container_findings(script, ra) == []

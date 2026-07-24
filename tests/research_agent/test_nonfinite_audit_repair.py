from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.reasons import repair_reason_for_finding
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair
from easyicu.research_agent.schema import ValidationFinding
from easyicu.research_agent.schema import AnalysisStep

CODE = """
import numpy as np
import pandas as pd

def strict_numeric(series, name):
    coerced = pd.to_numeric(series, errors="coerce")
    if int((coerced.notna() & ~np.isfinite(coerced)).sum()) > 0:
        raise RuntimeError("non-finite")
    return coerced.astype(float)

def audit(frame):
    value = strict_numeric(frame["value"], "value")
    adult_value = value.loc[frame["adult"]].copy()
    nonfinite_mask = pd.Series(False, index=adult_value.index)
    n_nonfinite = int(nonfinite_mask.sum())
    rows = [{"status": "value_nonfinite", "count": n_nonfinite}]
    return value, nonfinite_mask, rows
"""

RETURNED_LOSS_CODE = """
import numpy as np
import pandas as pd

from easyicu.research_agent.methods.descriptive_inputs import strict_numeric_input

def finite_numeric(original, name):
    raw = pd.Series(original, copy=True)
    raw_numeric = pd.to_numeric(raw, errors="coerce")
    raw_nonfinite = raw_numeric.notna() & ~np.isfinite(raw_numeric)
    masked = raw_numeric.mask(raw_nonfinite)
    checked = strict_numeric_input(masked)
    return checked.values, int(raw_nonfinite.sum()), int((raw.notna() & raw_numeric.isna()).sum())
"""

_DESCRIPTIVE_STEP = AnalysisStep(
    step_id="measurement_audit",
    intent="Audit one already-declared numeric input.",
    inputs=["artifact:analysis_cohort", "value"],
    expected_outputs=["table:data_quality"],
    method="measurement_bias_and_distribution_audit",
)


def test_repair_preserves_observed_nonfinite_for_declared_audit() -> None:
    repaired, names = deterministic_concept_audit_repair(
        CODE,
        ["blocking concept audit finding"],
    )

    assert names == ["nonfinite_audit_preserve_observed_v1"]
    assert "strict_numeric_input(" in repaired
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    frame = pd.DataFrame(
        {"value": [1.0, float("inf"), None], "adult": [True, True, True]}
    )
    values, mask, rows = namespace["audit"](frame)
    # The result-bearing series is strict and therefore masks the one observed
    # infinity as well as the one source-missing value.  The separate mask
    # below retains which row was an observed non-finite value.
    assert values.isna().sum() == 2
    assert mask.tolist() == [False, True, False]
    assert rows == [{"status": "value_nonfinite", "count": 1}]


def test_repair_still_rejects_lossy_numeric_coercion() -> None:
    repaired, _ = deterministic_concept_audit_repair(
        CODE,
        ["blocking concept audit finding"],
    )
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    frame = pd.DataFrame({"value": ["bad"], "adult": [True]})

    try:
        namespace["audit"](frame)
    except ValueError as exc:
        assert "lossy numeric coercion" in str(exc)
    else:
        raise AssertionError("lossy coercion must fail closed")


def test_repair_requires_a_concept_audit_authority() -> None:
    repaired, names = deterministic_concept_audit_repair(CODE, [])
    assert repaired == CODE
    assert names == []


def test_repair_rejects_ambiguous_nonfinite_audits() -> None:
    ambiguous = CODE.replace(
        "nonfinite_mask = pd.Series(False, index=adult_value.index)",
        "nonfinite_mask = pd.Series(False, index=adult_value.index)\n"
        "    other_mask = pd.Series(False, index=adult_value.index)",
    ).replace(
        'rows = [{"status": "value_nonfinite", "count": n_nonfinite}]',
        "other_n = int(other_mask.sum())\n"
        '    rows = [{"status": "value_nonfinite", "count": n_nonfinite}, '
        '{"status": "other_nonfinite", "count": other_n}]',
    )

    repaired, names = deterministic_concept_audit_repair(
        ambiguous,
        ["blocking concept audit finding"],
    )
    assert repaired == ambiguous
    assert names == []


def test_returned_coercion_loss_is_guarded_from_structured_finding() -> None:
    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Observed non-numeric values are silently treated as missing.",
        detail={
            "issue_code": "strict_numeric_nonfinite_guard_required",
            "variables": ["values"],
        },
    )

    repaired, names = deterministic_concept_audit_repair(
        RETURNED_LOSS_CODE,
        [finding.message],
        repair_reasons=[repair_reason_for_finding(finding)],
        repair_findings=[finding],
    )

    assert names == ["returned_coercion_loss_guard_v1"]
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    helper = namespace["finite_numeric"]
    assert callable(helper)
    values, nonfinite_n, loss_n = helper(pd.Series([1.0, None]), "value")
    assert list(values[:1]) == [1.0]
    assert nonfinite_n == 0
    assert loss_n == 0
    with pytest.raises(RuntimeError, match="coercion invalidated"):
        helper(pd.Series([1.0, "bad"]), "value")


def test_shared_numeric_helper_rejects_nonfinite_model_inputs() -> None:
    code = """
import numpy as np
import pandas as pd

def numeric_coerce_fail_closed(frame, columns):
    result = frame.copy()
    for column in columns:
        original = result[column]
        converted = pd.to_numeric(original, errors="coerce")
        newly_invalid = int((original.notna() & converted.isna()).sum())
        if newly_invalid > 0:
            raise RuntimeError("lossy numeric coercion")
        result[column] = converted
    return result

numeric_columns = ["exposure", "age", "score", "outcome"]
data = numeric_coerce_fail_closed(frame, numeric_columns)
"""
    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Non-finite model inputs can be silently removed.",
        detail={
            "issue_code": "strict_numeric_nonfinite_guard_required",
            "variables": ["exposure", "age", "score"],
        },
    )

    repaired, names = deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[repair_reason_for_finding(finding)],
        repair_findings=[finding],
    )

    assert names == ["strict_numeric_nonfinite_guard_v1"]
    assert "_easyicu_nonfinite_numeric_mask_v1" in repaired
    namespace = {"frame": pd.DataFrame({"exposure": [float("inf")]})}
    with pytest.raises(RuntimeError, match="non-finite numeric input"):
        exec(repaired, namespace)


def test_returned_coercion_loss_is_caught_before_llm_audit() -> None:
    findings = [
        finding
        for finding in audit_mechanical_code_contracts(
            RETURNED_LOSS_CODE,
            _DESCRIPTIVE_STEP,
        )
        if (finding.detail or {}).get("reason") == "lossy_numeric_coercion"
    ]

    assert len(findings) == 1
    assert (findings[0].detail or {})["issues"] == [
        {
            "gap": "returned_coercion_loss_count_not_fail_closed",
            "lines": [13],
        }
    ]
    repaired, names = deterministic_concept_audit_repair(
        RETURNED_LOSS_CODE,
        [findings[0].message],
        repair_reasons=[repair_reason_for_finding(findings[0])],
        repair_findings=findings,
    )
    assert names == ["returned_coercion_loss_guard_v1"]
    assert audit_mechanical_code_contracts(repaired, _DESCRIPTIVE_STEP) == []


def test_returned_coercion_loss_repair_rejects_ambiguous_helpers() -> None:
    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Observed non-numeric values are silently treated as missing.",
        detail={"issue_code": "strict_numeric_nonfinite_guard_required"},
    )
    ambiguous = RETURNED_LOSS_CODE + RETURNED_LOSS_CODE.replace(
        "finite_numeric", "other_numeric"
    )

    repaired, names = deterministic_concept_audit_repair(
        ambiguous,
        [finding.message],
        repair_reasons=[repair_reason_for_finding(finding)],
        repair_findings=[finding],
    )

    assert repaired == ambiguous
    assert names == []


def test_returned_coercion_loss_repair_is_structural_and_automatic() -> None:
    metadata = repair_metadata_for("returned_coercion_loss_guard_v1")

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert automatic_repair_allowed("returned_coercion_loss_guard_v1") is True


def test_v1_repair_upgrades_to_host_strict_boundary() -> None:
    legacy = CODE.replace(
        'value = strict_numeric(frame["value"], "value")',
        '_easyicu_value_raw_nonfinite_audit_v1 = (frame["value"]).copy()\n'
        '    value = pd.to_numeric(_easyicu_value_raw_nonfinite_audit_v1, errors="coerce")\n'
        "    _easyicu_value_coercion_loss_v1 = _easyicu_value_raw_nonfinite_audit_v1.notna() & value.isna()\n"
        "    if int(_easyicu_value_coercion_loss_v1.sum()) > 0:\n"
        '        raise ValueError("lossy numeric coercion in audited input")',
    ).replace(
        "nonfinite_mask = pd.Series(False, index=adult_value.index)",
        "nonfinite_mask = adult_value.notna() & ~np.isfinite(adult_value)",
    )
    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="use host strict numeric boundary",
        detail={
            "issue_code": "strict_numeric_nonfinite_guard_required",
            "variables": ["value"],
        },
    )
    repaired, names = deterministic_concept_audit_repair(
        legacy,
        [finding.message],
        repair_reasons=[repair_reason_for_finding(finding)],
        repair_findings=[finding],
    )
    assert names == ["nonfinite_audit_host_strict_boundary_v2"]
    assert (
        "strict_numeric_input(value.mask(_easyicu_value_nonfinite_observed_v2)).values"
        in repaired
    )
    assert "_easyicu_value_nonfinite_observed_v2.reindex(" in repaired

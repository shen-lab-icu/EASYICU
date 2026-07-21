from __future__ import annotations

import pandas as pd

from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair

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


def test_repair_preserves_observed_nonfinite_for_declared_audit() -> None:
    repaired, names = deterministic_concept_audit_repair(
        CODE,
        ["blocking concept audit finding"],
    )

    assert names == ["nonfinite_audit_preserve_observed_v1"]
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    frame = pd.DataFrame(
        {"value": [1.0, float("inf"), None], "adult": [True, True, True]}
    )
    values, mask, rows = namespace["audit"](frame)
    assert values.isna().sum() == 1
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

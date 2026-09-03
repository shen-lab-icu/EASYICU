"""Missing numeric values remain distinct from observed non-finite values."""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.execution.phase import _untrusted_runtime_repair_allowed
from easyicu.research_agent.repair_registry import RepairClass, repair_metadata_for
from easyicu.research_agent.repairs.source import _deterministic_runner_repair


_ERROR = (
    "RuntimeError: Invalid score_max ordered-domain values: "
    "nonfinite=2, noninteger=0, out_of_domain=0"
)

_CODE = """
import numpy as np
import pandas as pd

def audit(raw):
    values = pd.to_numeric(raw, errors="coerce")
    nonfinite_n = int((~np.isfinite(values.fillna(np.nan))).sum())
    noninteger_n = 0
    out_of_domain_n = 0
    if any(count > 0 for count in (
        nonfinite_n, noninteger_n, out_of_domain_n
    )):
        raise RuntimeError(
            "Invalid score_max ordered-domain values: "
            f"nonfinite={nonfinite_n}, "
            f"noninteger={noninteger_n}, "
            f"out_of_domain={out_of_domain_n}"
        )
    return nonfinite_n
"""


def test_runtime_repair_does_not_count_source_missing_as_nonfinite() -> None:
    repair = _deterministic_runner_repair(code=_CODE, run_log=_ERROR)

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "nonfinite_missing_mask_conflation_v1"
    assert "values.notna() & ~np.isfinite(values)" in repaired
    namespace: dict[str, object] = {}
    exec(repaired, namespace)  # noqa: S102 - generated-code regression
    assert namespace["audit"](pd.Series([1.0, None])) == 0
    with pytest.raises(RuntimeError, match="nonfinite=1"):
        namespace["audit"](pd.Series([1.0, float("inf"), None]))

    metadata = repair_metadata_for(repair_id)
    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.introduces_numbers is False
    assert _untrusted_runtime_repair_allowed(
        repair_id=repair_id,
        source="deterministic_runner_repair",
    )


def test_runtime_repair_requires_matching_error_and_reported_count() -> None:
    assert _deterministic_runner_repair(code=_CODE, run_log="") is None
    assert (
        _deterministic_runner_repair(
            code=_CODE,
            run_log="RuntimeError: unrelated",
        )
        is None
    )
    assert (
        _deterministic_runner_repair(
            code=_CODE.replace("ordered-domain values:", "domain values:"),
            run_log=_ERROR,
        )
        is None
    )


def test_runtime_repair_rejects_ambiguous_nonfinite_counts() -> None:
    ambiguous = _CODE.replace(
        "nonfinite_n = int((~np.isfinite(values.fillna(np.nan))).sum())",
        "nonfinite_n = int((~np.isfinite(values.fillna(np.nan))).sum())\n"
        "    other_nonfinite_n = int("
        "(~np.isfinite(values.fillna(np.nan))).sum())",
    ).replace(
        'f"nonfinite={nonfinite_n}, "',
        'f"nonfinite={nonfinite_n}, "\n'
        '            f"also_nonfinite={other_nonfinite_n}, "',
    )

    assert _deterministic_runner_repair(code=ambiguous, run_log=_ERROR) is None

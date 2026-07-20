"""Boolean-reduction identity errors fail before provider or sandbox work."""

from __future__ import annotations

import ast

import pytest

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repairs.reasons import (
    RepairReason,
    repair_reason_for_finding,
)
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair


def _step(ra):
    return ra.AnalysisStep(
        step_id="measurement_qc",
        intent="Audit declared measurement provenance.",
        inputs=["value_measured", "value_n"],
        expected_outputs=["table:measurement_qc"],
        method="measurement_quality_control",
    )


def _findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason")
        == "boolean_reduction_identity_comparison"
    ]


def _repair(script: str, ra):
    findings = _findings(script, ra)
    repaired, repair_names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )
    return findings, repaired, repair_names


@pytest.mark.parametrize(
    ("comparison", "expected"),
    [
        ("series.all() is True", "bool(series.all())"),
        ("series.all() is False", "not bool(series.all())"),
        ("series.all() is not True", "not bool(series.all())"),
        ("series.all() is not False", "bool(series.all())"),
        ("True is series.any()", "bool(series.any())"),
        ("False is series.any()", "not bool(series.any())"),
        ("True is not series.any()", "not bool(series.any())"),
        ("False is not series.any()", "bool(series.any())"),
    ],
)
def test_proven_pandas_series_identity_forms_are_repaired(
    comparison: str,
    expected: str,
    ra,
):
    script = f"""
import pandas as pd
series = pd.Series([True, False])
if {comparison}:
    raise ValueError("invalid")
"""

    findings, repaired, repair_names = _repair(script, ra)

    assert len(findings) == 1
    assert findings[0].detail["repair_safe"] is True
    assert repair_reason_for_finding(findings[0]) is (
        RepairReason.BOOLEAN_REDUCTION_IDENTITY
    )
    assert expected in repaired
    assert "boolean_reduction_identity_v1" in repair_names
    ast.parse(repaired)
    assert _findings(repaired, ra) == []


@pytest.mark.parametrize(
    ("setup", "comparison", "expected"),
    [
        (
            "import numpy as np\narr = np.asarray([True, False])",
            "arr.all() is False",
            "not bool(arr.all())",
        ),
        (
            "import numpy as np\narr = np.asarray([True, False])",
            "arr.any() is not False",
            "bool(arr.any())",
        ),
        (
            "import numpy as np\narr = np.asarray([True, False])",
            "np.all(arr) is True",
            "bool(np.all(arr))",
        ),
        (
            "from numpy import any as np_any\narr = [True, False]",
            "False is np_any(arr)",
            "not bool(np_any(arr))",
        ),
    ],
)
def test_proven_numpy_scalar_reductions_are_repaired(
    setup: str,
    comparison: str,
    expected: str,
    ra,
):
    script = f'{setup}\nif {comparison}:\n    raise ValueError("invalid")\n'

    findings, repaired, repair_names = _repair(script, ra)

    assert len(findings) == 1
    assert findings[0].detail["repair_safe"] is True
    assert expected in repaired
    assert repair_names == ["boolean_reduction_identity_v1"]
    assert _findings(repaired, ra) == []


@pytest.mark.parametrize(
    "script",
    [
        """
import pandas as pd
frame = pd.DataFrame({"a": [True], "b": [False]})
if frame.all() is False:
    raise ValueError("invalid")
""",
        """
import pandas as pd
series = pd.Series([True, None])
if series.all(skipna=False) is False:
    raise ValueError("invalid")
""",
        """
import numpy as np
arr = np.asarray([[True, False]])
if np.all(arr, axis=0) is False:
    raise ValueError("invalid")
""",
        """
import numpy as np
arr = np.asarray([[True, False]])
if arr.all(axis=0) is True:
    raise ValueError("invalid")
""",
        """
import numpy as np
arr = np.asarray([True, False])
if False is arr.all() is other:
    raise ValueError("invalid")
""",
        """
import numpy as np
arr = np.asarray([True, False])
if arr.all(*args) is True:
    raise ValueError("invalid")
""",
        """
import numpy as np
arr = np.asarray([True, False])
if arr.all(skipna=True) is True:
    raise ValueError("invalid")
""",
    ],
)
def test_ambiguous_or_non_scalar_reductions_are_finding_only(script: str, ra):
    findings, repaired, repair_names = _repair(script, ra)

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["repair_safe"] is False
    assert repaired == script
    assert repair_names == []


def test_unknown_all_receiver_is_outside_pandas_numpy_authority(ra) -> None:
    script = """
if validate(*parts).all() is True:
    raise ValueError("invalid")
"""

    assert _findings(script, ra) == []


@pytest.mark.parametrize(
    "script",
    [
        "flag = True\nif flag is False:\n    raise ValueError('invalid')\n",
        "value = None\nif value is None:\n    pass\n",
        "sentinel = object()\nif value is sentinel:\n    pass\n",
        """
from enum import Enum
class State(Enum):
    READY = "ready"
if value is State.READY:
    pass
""",
        """
class Custom:
    def all(self):
        return False
value = Custom()
if value.all() is False:
    raise ValueError("invalid")
""",
        """
# series.all() is False
message = "series.any() is True"
""",
        """
import pandas as pd
series = pd.Series([True, False])
if not bool(series.all()):
    raise ValueError("invalid")
""",
        """
def all(values):
    return True
if all(values) is True:
    pass
""",
    ],
)
def test_non_numpy_pandas_identity_cases_are_not_claimed(script: str, ra):
    assert _findings(script, ra) == []


def test_step06_archived_boolean_guards_are_repaired_atomically(ra):
    script = """
import pandas as pd
frame = pd.read_parquet(input_path)
if frame["aki_stage_measured"].dropna().isin([0, 1]).all() is False:
    raise ValueError("AKI measurement flags must be binary")
if frame["charlson_measured"].dropna().isin([0, 1]).all() is not True:
    raise ValueError("Charlson measurement flags must be binary")
"""

    findings, repaired, repair_names = _repair(script, ra)

    assert len(findings) == 2
    assert all(finding.detail["repair_safe"] is True for finding in findings)
    assert (
        'not bool(frame["aki_stage_measured"].dropna().isin([0, 1]).all())' in repaired
    )
    assert (
        'not bool(frame["charlson_measured"].dropna().isin([0, 1]).all())' in repaired
    )
    assert repair_names == ["boolean_reduction_identity_v1"]
    assert _findings(repaired, ra) == []


def test_correct_reduction_truth_check_remains_byte_identical(ra):
    script = """
import pandas as pd
series = pd.Series([True, False])
if bool(series.all()):
    pass
"""

    findings, repaired, repair_names = _repair(script, ra)

    assert findings == []
    assert repaired == script
    assert repair_names == []

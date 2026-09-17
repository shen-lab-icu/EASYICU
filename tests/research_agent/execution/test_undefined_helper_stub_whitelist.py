"""Fix F whitelist: undefined-helper stub only for serialization hooks (P1 fix test).

Covers: serialization-hook position still triggers, non-serialization
NameError does not trigger, step_record marker, gate warning finding.
"""

from __future__ import annotations

from easyicu.research_agent.gates.step_result_evidence import (
    semantic_stub_injection_findings,
)
from easyicu.research_agent.repairs.runner_dispatch import (
    mark_semantic_stub_injection,
    undefined_helper_stub_name_for_repair,
)
from easyicu.research_agent.repairs.source import (
    _deterministic_runner_repair_candidate as _deterministic_runner_repair,
    _undefined_helper_reference_is_callable,
)
from easyicu.research_agent.schema import AnalysisStep

SERIALIZATION_CODE = (
    "import json\n"
    'data = {"a": 1}\n'
    "with open('out.json', 'w') as f:\n"
    "    json.dump(data, f, default=to_json_serializable)\n"
)
SERIALIZATION_LOG = (
    "Traceback (most recent call last):\n"
    '  File "analysis.py", line 4, in <module>\n'
    "NameError: name 'to_json_serializable' is not defined\n"
)

NON_SERIALIZATION_CODE = (
    "import pandas as pd\n"
    "result = fit_cox_model(df, duration_col='los', event_col='death')\n"
    "print(result.summary())\n"
)
NON_SERIALIZATION_LOG = (
    "Traceback (most recent call last):\n"
    '  File "analysis.py", line 2, in <module>\n'
    "NameError: name 'fit_cox_model' is not defined\n"
)


def test_serialization_hook_still_triggers() -> None:
    assert (
        _undefined_helper_reference_is_callable(
            SERIALIZATION_CODE, "to_json_serializable"
        )
        is True
    )
    result = _deterministic_runner_repair(
        code=SERIALIZATION_CODE, run_log=SERIALIZATION_LOG
    )
    assert result is not None, "serialization hook must still trigger Fix F"
    repair_name, repaired = result
    assert repair_name == "undefined_helper_stub_to_json_serializable_v1"
    assert "def to_json_serializable" in repaired


def test_non_serialization_name_error_does_not_trigger() -> None:
    assert (
        _undefined_helper_reference_is_callable(
            NON_SERIALIZATION_CODE, "fit_cox_model"
        )
        is False
    )
    result = _deterministic_runner_repair(
        code=NON_SERIALIZATION_CODE, run_log=NON_SERIALIZATION_LOG
    )
    assert result is None or not result[0].startswith("undefined_helper_stub_"), (
        f"non-serialization NameError must not trigger Fix F, got {result}"
    )


def test_marker_written_on_trigger_only() -> None:
    step_record: dict = {}
    helper = mark_semantic_stub_injection(
        step_record, "undefined_helper_stub_to_json_serializable_v1"
    )
    assert helper == "to_json_serializable"
    assert step_record["semantic_stub_injected"] == "to_json_serializable"

    other: dict = {}
    assert mark_semantic_stub_injection(other, "dtype_coerce_v1") is None
    assert "semantic_stub_injected" not in other
    assert undefined_helper_stub_name_for_repair("dtype_coerce_v1") is None


def test_gate_emits_warning_only_finding() -> None:
    step = AnalysisStep(
        step_id="01_x", intent="x", method="descriptive", expected_outputs=[]
    )
    findings = semantic_stub_injection_findings(
        step=step, semantic_stub_injected="to_json_serializable"
    )
    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].detail["helper_name"] == "to_json_serializable"
    assert semantic_stub_injection_findings(step=step) == []
    assert semantic_stub_injection_findings(step=step, semantic_stub_injected="") == []

from easyicu.research_agent.repairs.source import _deterministic_runner_repair
from easyicu.research_agent.repairs.strict_numeric_result import (
    patch_strict_numeric_input_result_projection,
)

_ERROR = (
    "TypeError: float() argument must be a string or a real number, "
    "not 'StrictNumericInput'"
)
_SCRIPT = """import pandas as pd
from easyicu.research_agent.methods.descriptive_inputs import strict_numeric_input


def numeric_series(frame, column):
    converted_values = strict_numeric_input(frame[column])
    return pd.Series(converted_values, index=frame.index, name=column)
"""


def test_projects_values_from_typed_result_before_numeric_constructor() -> None:
    repaired = patch_strict_numeric_input_result_projection(_SCRIPT, _ERROR)

    assert repaired != _SCRIPT
    assert "strict_numeric_input(frame[column]).values" in repaired


def test_routes_the_typed_result_repair_without_provider_help() -> None:
    repair = _deterministic_runner_repair(code=_SCRIPT, run_log=_ERROR)

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "strict_numeric_input_result_projection_v1"
    assert "strict_numeric_input(frame[column]).values" in repaired


def test_direct_numeric_constructor_call_is_repaired() -> None:
    script = _SCRIPT.replace(
        "converted_values = strict_numeric_input(frame[column])\n"
        "    return pd.Series(converted_values, index=frame.index, name=column)",
        "return pd.Series(strict_numeric_input(frame[column]), index=frame.index)",
    )

    repaired = patch_strict_numeric_input_result_projection(script, _ERROR)

    assert "pd.Series(strict_numeric_input(frame[column]).values" in repaired


def test_existing_values_projection_is_unchanged() -> None:
    script = _SCRIPT.replace(
        "strict_numeric_input(frame[column])",
        "strict_numeric_input(frame[column]).values",
    )

    assert patch_strict_numeric_input_result_projection(script, _ERROR) == script


def test_unrelated_use_or_unrelated_error_is_unchanged() -> None:
    audit_only = _SCRIPT.replace(
        "return pd.Series(converted_values, index=frame.index, name=column)",
        "return converted_values.audit",
    )

    assert patch_strict_numeric_input_result_projection(audit_only, _ERROR) == audit_only
    assert (
        patch_strict_numeric_input_result_projection(
            _SCRIPT,
            "TypeError: another object",
        )
        == _SCRIPT
    )


def test_ambiguous_typed_result_candidates_fail_closed() -> None:
    ambiguous = _SCRIPT + """
def other(frame):
    other_values = strict_numeric_input(frame["other"])
    return pd.to_numeric(other_values)
"""

    assert (
        patch_strict_numeric_input_result_projection(ambiguous, _ERROR) == ambiguous
    )

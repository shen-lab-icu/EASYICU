from easyicu.research_agent.execution.phase import _untrusted_runtime_repair_allowed
from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
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


def test_typed_result_projection_has_exact_syntactic_authority() -> None:
    repair_id = "strict_numeric_input_result_projection_v1"

    metadata = repair_metadata_for(repair_id)

    assert metadata.classification_source == "exact"
    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(repair_id)
    assert _untrusted_runtime_repair_allowed(
        repair_id=repair_id,
        source="deterministic_runner_repair",
    )


def test_direct_numeric_constructor_call_is_repaired() -> None:
    script = _SCRIPT.replace(
        "converted_values = strict_numeric_input(frame[column])\n"
        "    return pd.Series(converted_values, index=frame.index, name=column)",
        "return pd.Series(strict_numeric_input(frame[column]), index=frame.index)",
    )

    repaired = patch_strict_numeric_input_result_projection(script, _ERROR)

    assert "pd.Series(strict_numeric_input(frame[column]).values" in repaired


def test_projects_values_through_the_real_series_branch_alias() -> None:
    script = '''import pandas as pd
from easyicu.research_agent.methods.descriptive_inputs import strict_numeric_input

for column in EXPECTED_RAW_COLUMNS:
    original = df[column]
    checked = strict_numeric_input(original)
    if isinstance(checked, pd.Series):
        checked_series = checked.copy()
        checked_series.index = df.index
    else:
        checked_series = pd.Series(checked, index=df.index)
    checked_series = pd.to_numeric(checked_series, errors="coerce").astype(float)
'''

    repaired = patch_strict_numeric_input_result_projection(script, _ERROR)

    assert repaired.count("strict_numeric_input(original).values") == 1
    repair = _deterministic_runner_repair(code=script, run_log=_ERROR)
    assert repair is not None
    repair_id, routed = repair
    assert repair_id == "strict_numeric_input_result_projection_v1"
    assert routed == repaired


def test_mixed_source_branch_alias_fails_closed() -> None:
    script = '''import pandas as pd
from easyicu.research_agent.methods.descriptive_inputs import strict_numeric_input

checked = strict_numeric_input(frame["value"])
if condition:
    checked_series = checked.copy()
else:
    checked_series = frame["fallback"]
result = pd.to_numeric(checked_series)
'''

    assert patch_strict_numeric_input_result_projection(script, _ERROR) == script


def test_arbitrary_alias_transform_fails_closed() -> None:
    script = '''import pandas as pd
from easyicu.research_agent.methods.descriptive_inputs import strict_numeric_input

checked = strict_numeric_input(frame["value"])
checked_series = normalize(checked)
result = pd.to_numeric(checked_series)
'''

    assert patch_strict_numeric_input_result_projection(script, _ERROR) == script


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

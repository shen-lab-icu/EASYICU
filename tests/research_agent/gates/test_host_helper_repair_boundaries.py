"""Host-helper repairs must preserve the diagnosed validation boundary."""

import pytest

from easyicu.research_agent.repairs.host_helper_result import (
    patch_closed_counts_level_column,
    patch_host_validation_helper_reraise,
    patch_table_one_planner_spec,
)
from easyicu.research_agent.schema import ValidationFinding


def _finding(detail):
    return ValidationFinding(
        validator="mechanical_code_preflight", severity="error", message="diagnosed", detail=detail
    )


def test_host_failure_is_reraised_and_repair_is_idempotent():
    code = '''try:
    strict_numeric_input(values)
except ValueError:
    status.append("completed")
finally:
    status.append("cleaned")
'''
    finding = _finding({"reason": "host_validation_helper_error_swallowed", "line": 3,
                        "helper_names": ["strict_numeric_input"]})
    repaired = patch_host_validation_helper_reraise(code, findings=[finding])
    assert repaired != code
    assert patch_host_validation_helper_reraise(repaired, findings=[finding]) == repaired
    status = []

    def fail(_values):
        raise ValueError("host validation failed")

    with pytest.raises(ValueError, match="host validation failed"):
        exec(repaired, {"strict_numeric_input": fail, "values": [], "status": status})
    assert status == ["cleaned"]
    assert patch_host_validation_helper_reraise(code, findings=[finding, finding]) == code
    stale = finding.model_copy(update={"detail": {**finding.detail, "line": 2}})
    assert patch_host_validation_helper_reraise(code, findings=[stale]) == code


def test_closed_count_repair_refuses_rebound_helper_result():
    code = '''from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts
counts = closed_categorical_counts(values, declared_levels=[0, 1])
table = counts.table
counts = another_result
levels = list(table.index)
'''
    finding = _finding({"reason": "closed_counts_table_index_used_as_levels", "line": 5})
    assert patch_closed_counts_level_column(code, findings=[finding]) == code
    assert patch_closed_counts_level_column(code, findings=[_finding({"reason": "closed_counts_table_index_used_as_levels", "line": 1000})]) == code
    valid = code.replace("counts = another_result\n", "")
    valid_finding = _finding({"reason": "closed_counts_table_index_used_as_levels", "line": 4})
    repaired = patch_closed_counts_level_column(valid, findings=[valid_finding])
    assert 'levels = list(table["level"])' in repaired
    assert patch_closed_counts_level_column(repaired, findings=[valid_finding]) == repaired


def test_table_one_repair_requires_exact_planner_spec_and_assignment(ra):
    step = ra.AnalysisStep(
        step_id="table_one", intent="Baseline table", inputs=["death", "age"],
        expected_outputs=["table:table_one"], method="table_one",
        table_one_spec={"group_by": "death", "group_levels": [0, 1], "variables": [{
            "name": "age", "variable_kind": "continuous", "summary": "median_iqr",
            "test": "mann_whitney_or_kruskal",
        }]},
    )
    code = 'spec = {"group_by": "other"}\n'
    detail = {"reason": "table_one_spec_not_planner_owned", "line": 1,
              "spec_name": "spec", "expected_spec": step.table_one_spec.model_dump(mode="python")}
    finding = _finding(detail)
    repaired = patch_table_one_planner_spec(code, findings=[finding])
    assert repaired != code
    assert patch_table_one_planner_spec(repaired, findings=[finding]) == repaired
    assert patch_table_one_planner_spec(code, findings=[_finding({**detail, "spec_name": "other"})]) == code
    assert patch_table_one_planner_spec(code, findings=[_finding({**detail, "line": 1000})]) == code
    assert patch_table_one_planner_spec(code, findings=[_finding({**detail, "expected_spec": {}})]) == code

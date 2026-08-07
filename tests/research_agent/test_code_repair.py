"""Smoke tests for ``easyicu.research_agent.repairs.source``.

Background
----------
``code_repair.py`` is ~2,800 LOC of deterministic post-failure patches
applied to agent-emitted code (KeyError-not-in-index strip, NameError
helper restore, generic fallback dispatch, etc.). It carries no direct
unit tests; ``pipeline.py`` imports 10 symbols from it and exercises
them only indirectly through the end-to-end pipeline.

These tests pin the pure-function contracts of the symbols ``pipeline``
imports, so that a future split or rename of ``code_repair.py`` has
something to break against. They do not aim for line coverage — they
aim to make the **public surface** behaviour-stable.

Scope
-----
Pure / IO-free entries from ``pipeline.py``'s import list:

* ``_KEYERROR_NOT_IN_INDEX_RE``
* ``_NAME_ERROR_HELPER_RE``
* ``_extract_missing_index_columns``
* ``_strip_columns_from_list_literals``
* ``_patch_json_dump_numpy_key_sanitizer``

The heavier runner-repair path is exercised only through narrow,
IO-free probes here; full pipeline fixtures cover integration behaviour.
"""

from __future__ import annotations

import ast
import re
import json

import pytest

from easyicu.research_agent.repairs.source import (
    _KEYERROR_NOT_IN_INDEX_RE,
    _NAME_ERROR_HELPER_RE,
    _deterministic_summary_repair,
    _deterministic_runner_repair,
    _extract_missing_index_columns,
    _patch_json_dump_numpy_key_sanitizer,
    _strip_columns_from_list_literals,
    deterministic_contract_repair,
    deterministic_concept_audit_repair,
)
from easyicu.research_agent.repairs.reasons import RepairReason
from easyicu.research_agent.schema import ValidationFinding

# ---------------------------------------------------------------------------
# _KEYERROR_NOT_IN_INDEX_RE / _extract_missing_index_columns
# ---------------------------------------------------------------------------


class TestExtractMissingIndexColumns:
    def test_empty_log_returns_empty_list(self):
        assert _extract_missing_index_columns("") == []
        assert _extract_missing_index_columns(None) == []  # type: ignore[arg-type]

    def test_log_without_keyerror_returns_empty_list(self):
        log = "Traceback (most recent call last):\nValueError: something else"
        assert _extract_missing_index_columns(log) == []

    def test_extracts_single_column_from_keyerror(self):
        log = "KeyError: \"['sofa_total'] not in index\""
        assert _extract_missing_index_columns(log) == ["sofa_total"]

    def test_extracts_multiple_columns_preserving_order(self):
        log = "KeyError: \"['a', 'b', 'c'] not in index\""
        assert _extract_missing_index_columns(log) == ["a", "b", "c"]

    def test_deduplicates_columns(self):
        log = "KeyError: \"['a', 'b', 'a'] not in index\""
        assert _extract_missing_index_columns(log) == ["a", "b"]

    def test_tolerates_double_quoted_entries(self):
        # The matcher is documented to accept both single and double quotes.
        log = 'KeyError: "["col1", "col2"] not in index"'
        assert _extract_missing_index_columns(log) == ["col1", "col2"]


def test_keyerror_regex_is_compiled_pattern():
    """Pin the symbol pipeline imports as a regex, not a raw string."""
    assert isinstance(_KEYERROR_NOT_IN_INDEX_RE, re.Pattern)
    match = _KEYERROR_NOT_IN_INDEX_RE.search("KeyError: \"['x'] not in index\"")
    assert match is not None
    assert "items" in match.groupdict()


def test_name_error_helper_regex_captures_identifier():
    match = _NAME_ERROR_HELPER_RE.search(
        "NameError: name 'load_concepts' is not defined"
    )
    assert match is not None
    assert match.group("name") == "load_concepts"


def test_name_error_helper_regex_rejects_non_identifiers():
    # The regex requires a Python identifier; a stray expression should miss.
    assert (
        _NAME_ERROR_HELPER_RE.search("NameError: name '123abc' is not defined") is None
    )


@pytest.mark.parametrize(
    ("operator", "expected"),
    [("&", 1), ("|", 3)],
)
def test_runner_repair_moves_boolean_mask_reduction_after_combination(
    operator,
    expected,
):
    code = (
        "import numpy as np\n"
        "observed = np.array([True, False, True])\n"
        "eligible = np.array([True, True, False])\n"
        f"count = int(observed.sum() {operator} eligible)\n"
    )
    repaired = _deterministic_runner_repair(
        code=code,
        run_log=("TypeError: only length-1 arrays can be converted to Python scalars"),
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "boolean_mask_reduction_precedence_v1"
    assert f"int(((observed) {operator} (eligible)).sum())" in patched
    namespace = {}
    exec(patched, namespace)
    assert namespace["count"] == expected
    assert (
        _deterministic_runner_repair(
            code=code,
            run_log=(
                "TypeError: only length-1 arrays can be converted to Python scalars"
            ),
            previous_repair=name,
        )
        is None
    )


def test_boolean_mask_reduction_repair_requires_traceback_and_exact_ast_shape():
    code = "count = int(observed.sum() & eligible)\n"
    traceback = "TypeError: only length-1 arrays can be converted to Python scalars"

    assert _deterministic_runner_repair(code=code, run_log="") is None
    assert (
        _deterministic_runner_repair(
            code="count = int(observed & eligible)\n",
            run_log=traceback,
        )
        is None
    )


def test_runner_repair_moves_inverted_right_reduction_after_boolean_mask():
    code = """import numpy as np
import pandas as pd

frame = pd.DataFrame({"stage": [0.0, np.inf, np.nan]})
stage = pd.to_numeric(frame["stage"], errors="coerce")
counts = {
    "nonfinite_n": int(
        frame["stage"].notna()
        & ~np.isfinite(stage.to_numpy(dtype=float))
        .sum()
    )
}
"""

    repair = _deterministic_runner_repair(
        code=code,
        run_log="TypeError: cannot convert the series to <class 'int'>",
    )

    assert repair is not None
    name, repaired = repair
    assert name == "boolean_mask_reduction_precedence_v1"
    namespace = {}
    exec(repaired, namespace)
    assert namespace["counts"]["nonfinite_n"] == 1
    assert (
        _deterministic_runner_repair(
            code=repaired,
            run_log="TypeError: cannot convert the series to <class 'int'>",
            previous_repair=name,
        )
        is None
    )


def test_runner_repair_reindexes_exact_pandas_boolean_mask_from_traceback():
    code = """import pandas as pd

death = pd.Series([1, 0, 1], index=[10, 11, 12])

def update_row(mortality_mask):
    deaths = death.loc[mortality_mask]
    return deaths

mask = pd.Series([True, True], index=[10, 12])
result = update_row(mask)
"""
    run_log = """Traceback (most recent call last):
  File "/easyicu-run/steps/06/analysis.py", line 6, in update_row
    deaths = death.loc[mortality_mask]
pandas.errors.IndexingError: Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match).
"""

    repair = _deterministic_runner_repair(code=code, run_log=run_log)

    assert repair is not None
    name, repaired = repair
    assert name == "pandas_boolean_index_alignment_v1"
    assert (
        "death.loc[(mortality_mask).reindex((death).index, fill_value=False)]"
        in repaired
    )
    namespace = {}
    exec(repaired, namespace)
    assert namespace["result"].to_dict() == {10: 1, 12: 1}
    assert (
        _deterministic_runner_repair(
            code=code,
            run_log=run_log,
            previous_repair=name,
        )
        is None
    )


def test_pandas_boolean_alignment_repair_is_traceback_and_shape_bound():
    code = """def update_row(mask):
    left = death.loc[mask]
    right = age.loc[mask]
"""
    traceback = (
        'File "/easyicu-run/analysis.py", line 2, in update_row\n'
        "pandas.errors.IndexingError: Unalignable boolean Series provided as indexer"
    )

    assert _deterministic_runner_repair(code=code, run_log="") is None
    assert (
        _deterministic_runner_repair(
            code=code,
            run_log=traceback.replace("line 2", "line 1"),
        )
        is None
    )
    assert (
        _deterministic_runner_repair(
            code="death.loc[mask, 'value']\n",
            run_log=traceback.replace("line 2", "line 1"),
        )
        is None
    )
    assert (
        _deterministic_runner_repair(
            code="count = int(observed.sum() & 1)\n",
            run_log=traceback,
        )
        is None
    )


def test_runner_repair_serializes_host_validation_findings_via_model_dump():
    code = """import json

def json_default(value):
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

summary = {"publication_export_qa": []}
json.dumps(summary, default=json_default)
"""

    repair = _deterministic_runner_repair(
        code=code,
        run_log=(
            "TypeError: Object of type ValidationFinding is not JSON serializable"
        ),
    )

    assert repair is not None
    name, repaired = repair
    assert name == "validation_finding_json_default_v1"
    assert 'if hasattr(value, "model_dump"):' in repaired
    assert "return value.model_dump()" in repaired
    ast.parse(repaired)


def test_runner_repair_declines_validation_finding_json_without_default_hook():
    code = 'import json\njson.dumps({"value": object()})\n'

    assert (
        _deterministic_runner_repair(
            code=code,
            run_log=(
                "TypeError: Object of type ValidationFinding is not JSON serializable"
            ),
        )
        is None
    )


def test_runner_repair_removes_runtime_objects_from_json_diagnostics_only():
    code = """import json

fitted_models = {}
fitted_models[model_id] = {
    "model": model,
    "labels": labels,
    "n_clusters": 4,
    "criterion_value": -123.5,
    "converged": True,
}
candidate_diagnostics = []
for record in sorted(fitted_models.values(), key=lambda item: item["n_clusters"]):
    candidate_diagnostics.append(record)
step_summary = {"candidate_fit_diagnostics": candidate_diagnostics}
json.dumps(step_summary)
"""

    repair = _deterministic_runner_repair(
        code=code,
        run_log=(
            "TypeError: Unsupported JSON value: <class "
            "'sklearn.mixture._gaussian_mixture.GaussianMixture'>"
        ),
    )

    assert repair is not None
    name, repaired = repair
    assert name == "sklearn_runtime_object_diagnostics_v1"
    assert '"model": model' in repaired
    assert '"labels": labels' in repaired
    assert (
        "candidate_diagnostics.append({key: value for key, value in record.items() "
        'if key not in {"model", "labels"}})'
    ) in repaired
    ast.parse(repaired)


def test_runner_repair_declines_ambiguous_runtime_object_diagnostic_loops():
    loop = """
for record in fitted_models.values():
    candidate_diagnostics.append(record)
"""
    code = (
        'fitted_models[model_id] = {"model": model, "labels": labels}\n'
        "candidate_diagnostics = []\n"
        + loop
        + loop
        + 'step_summary = {"diagnostics": candidate_diagnostics}\n'
    )

    assert (
        _deterministic_runner_repair(
            code=code,
            run_log=(
                "TypeError: Unsupported JSON value: <class "
                "'sklearn.base.BaseEstimator'>"
            ),
        )
        is None
    )


def test_runner_repair_requires_runtime_registry_to_reach_summary_mapping():
    code = """fitted_models = {}
fitted_models[model_id] = {"model": model, "labels": labels}
candidate_diagnostics = []
for record in fitted_models.values():
    candidate_diagnostics.append(record)
print(candidate_diagnostics)
"""

    assert (
        _deterministic_runner_repair(
            code=code,
            run_log=(
                "TypeError: Unsupported JSON value: <class "
                "'sklearn.base.BaseEstimator'>"
            ),
        )
        is None
    )


def test_prediction_split_template_escalates_instead_of_replacing_analysis():
    escalations = []
    repaired = _deterministic_runner_repair(
        code=(
            "figure_contract = FigureContract()\n"
            "train_test_split(X, y, test_size=0.2, test_size=0.3)\n"
        ),
        run_log="SyntaxError: keyword argument repeated",
        on_semantic_escalation=escalations.append,
    )

    assert repaired is None
    assert [item.repair_id for item in escalations] == ["prediction_split_minimal_v1"]


def test_summary_repair_handles_age_without_measured_indicator():
    code = """
source_vars_for_table = ["sepsis3", "death", "age", "sex", "hr_first"]
for var in source_vars_for_table:
        if var in ["sepsis3", "death"]:
            pass
        elif var == "sex":
            coding_rows.append({"variable": var})
        else:
            meas_var = measured_vars[var]
            coding_rows.append({"variable": meas_var})
"""

    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={"primary_or": None, "error": "'age'"},
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "age_covariate_no_measured_indicator_v1"
    assert 'elif var == "age":' in patched
    assert "Demographic baseline covariate" in patched
    assert (
        _deterministic_summary_repair(
            code=patched,
            step_summary={"primary_or": None, "error": "'age'"},
            previous_repair=name,
        )
        is None
    )


def test_summary_repair_skips_only_proven_unused_nullable_numeric_column():
    code = """
required_matrix_columns = [
    "primary_or",
    "absolute_or_difference",
    "relative_or_difference_percent",
]
numeric_matrix_columns = [
    "primary_or",
    "absolute_or_difference",
    "relative_or_difference_percent",
]
for column in numeric_matrix_columns:
    validate_numeric_series(
        robustness_matrix[column],
        f"robustness_matrix.{column}",
    )
matrix_row = robustness_matrix.iloc[0]
plotted_or = float(matrix_row["primary_or"])
"""
    first = _deterministic_summary_repair(
        code=code,
        step_summary={
            "status": "failed",
            "error": (
                "robustness_matrix.absolute_or_difference contains non-finite values"
            ),
        },
    )
    assert first is not None
    first_name, first_code = first
    assert first_name == "unused_nullable_numeric_validation_v1"
    numeric_assignment = next(
        node
        for node in ast.walk(ast.parse(first_code))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "numeric_matrix_columns"
    )
    assert [element.value for element in numeric_assignment.value.elts] == [
        "primary_or",
        "relative_or_difference_percent",
    ]

    second = _deterministic_summary_repair(
        code=first_code,
        step_summary={
            "status": "failed",
            "error": (
                "robustness_matrix.relative_or_difference_percent "
                "contains non-finite values"
            ),
        },
        previous_repair=first_name,
    )
    assert second is not None
    second_name, second_code = second
    assert second_name == "unused_nullable_numeric_validation_v1"
    assert "numeric_matrix_columns = ['primary_or']" in second_code


def test_summary_repair_preserves_nullable_column_used_by_figure():
    code = """
required_matrix_columns = ["primary_or", "absolute_or_difference"]
numeric_matrix_columns = ["primary_or", "absolute_or_difference"]
for column in numeric_matrix_columns:
    validate_numeric_series(
        robustness_matrix[column],
        f"robustness_matrix.{column}",
    )
matrix_row = robustness_matrix.iloc[0]
plotted_difference = float(matrix_row["absolute_or_difference"])
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "status": "failed",
            "error": (
                "robustness_matrix.absolute_or_difference contains non-finite values"
            ),
        },
    )
    assert repaired is None


def test_summary_repair_rejects_dynamic_nullable_column_use():
    code = """
required_matrix_columns = ["primary_or", "absolute_or_difference"]
numeric_matrix_columns = ["primary_or", "absolute_or_difference"]
for column in numeric_matrix_columns:
    validate_numeric_series(
        robustness_matrix[column],
        f"robustness_matrix.{column}",
    )
matrix_row = robustness_matrix.iloc[0]
selected_column = choose_column()
plotted_value = float(matrix_row[selected_column])
"""
    repaired = _deterministic_summary_repair(
        code=code,
        step_summary={
            "status": "failed",
            "error": (
                "robustness_matrix.absolute_or_difference contains non-finite values"
            ),
        },
    )
    assert repaired is None


def _rendering_role_code() -> str:
    return """
required_summary_columns = [
    "analysis",
    "restriction",
    "n",
    "odds_ratio",
]
restriction_values = (
    robustness_summary["restriction"].astype(str).str.strip().str.lower()
)
primary_rows = robustness_summary[
    restriction_values.isin({"primary", "full", "none", "all"})
]
complete_rows = robustness_summary[
    restriction_values.isin({"complete_case", "complete case", "complete-case"})
]
"""


def test_runner_repair_uses_declared_analysis_roles_for_figure_rows():
    run_log = (
        "ValueError: Could not identify exactly one primary and one complete-case row "
        "from robustness_summary.restriction"
    )
    repair = _deterministic_runner_repair(
        code=_rendering_role_code(),
        run_log=run_log,
    )

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "structured_analysis_role_selection_v1"
    assert "robustness_summary['analysis']" in repaired
    assert "restriction_values.isin({'primary'})" in repaired
    assert "complete_case_sensitivity" in repaired
    assert '"restriction"' in repaired


def test_runner_repair_rejects_unbound_or_dynamically_used_analysis_roles():
    run_log = (
        "ValueError: Could not identify exactly one primary and one complete-case row "
        "from robustness_summary.restriction"
    )
    missing_schema = _rendering_role_code().replace('    "analysis",\n', "")
    dynamic_use = _rendering_role_code() + "\nprint(restriction_values)\n"

    assert _deterministic_runner_repair(code=missing_schema, run_log=run_log) is None
    assert _deterministic_runner_repair(code=dynamic_use, run_log=run_log) is None
    assert (
        _deterministic_runner_repair(
            code=_rendering_role_code(),
            run_log="ValueError: some other row-selection error",
        )
        is None
    )


def test_contract_repair_suppresses_parent_effect_echo_in_figure_summary():
    code = """
plotted_or = float(primary_row["odds_ratio"])
step_summary = {
    "status": "completed",
    "figure_files": ["robustness_plot.png"],
    "output_files": {"figure:robustness_plot": "robustness_plot.png"},
    "numeric_summary": {
        "primary_analysis_n": primary_n,
        "primary_or": plotted_or,
        "primary_or_finite": bool(np.isfinite(plotted_or)),
        "confidence_intervals_available": 2,
    },
}
"""
    findings = [
        {
            "validator": "declared_product_contract",
            "detail": {
                "kind": "unauthorized_effect_product",
                "planned_method": "visualization",
                "declared_effect_products": [],
                "registered_effect_products": [
                    "log:primary_or",
                    "log:primary_or_finite",
                    "statistic:primary_or",
                    "statistic:primary_or_finite",
                ],
                "summary_effect_paths": [
                    "numeric_summary.primary_or",
                    "numeric_summary.primary_or_finite",
                ],
            },
        }
    ]

    repair = deterministic_contract_repair(code=code, findings=findings)

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "render_only_effect_echo_suppression_v1"
    summary_mapping = next(
        node.value
        for node in ast.walk(ast.parse(repaired))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "step_summary"
    )
    numeric_summary = summary_mapping.values[
        next(
            index
            for index, key in enumerate(summary_mapping.keys)
            if key.value == "numeric_summary"
        )
    ]
    assert {key.value for key in numeric_summary.keys} == {
        "primary_analysis_n",
        "confidence_intervals_available",
    }
    assert 'plotted_or = float(primary_row["odds_ratio"])' in repaired


def test_contract_repair_does_not_suppress_owned_or_ambiguous_effect_output():
    code = """
step_summary = {
    "status": "completed",
    "figure_files": ["effect.png"],
    "output_files": {"figure:effect": "effect.png"},
    "numeric_summary": {"primary_or": value, "rows": 2},
}
"""
    base_detail = {
        "kind": "unauthorized_effect_product",
        "planned_method": "visualization",
        "declared_effect_products": [],
        "registered_effect_products": [
            "log:primary_or",
            "statistic:primary_or",
        ],
        "summary_effect_paths": ["numeric_summary.primary_or"],
    }
    owned = dict(base_detail, declared_effect_products=["statistic:primary_or"])
    ambiguous = dict(
        base_detail,
        summary_effect_paths=["reconciliation.primary_or"],
    )

    assert (
        deterministic_contract_repair(
            code=code,
            findings=[{"validator": "declared_product_contract", "detail": owned}],
        )
        is None
    )
    assert (
        deterministic_contract_repair(
            code=code,
            findings=[{"validator": "declared_product_contract", "detail": ambiguous}],
        )
        is None
    )


def test_contract_repair_empties_unverifiable_raw_input_receipts() -> None:
    code = """
step_summary = {
    "status": "completed",
    "input_bindings": [
        {
            "input_key": "raw:age",
            "evidence_id": None,
            "sha256": None,
            "loaded": True,
            "row_count": n_rows,
        }
    ],
    "n_rows": n_rows,
}
"""
    finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {
            "issue": "input_binding_key_unresolved",
            "input_key": "raw:age",
            "resolved_input_keys": [],
        },
    }

    repair = deterministic_contract_repair(code=code, findings=[finding])

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "unresolved_input_binding_receipts_v1"
    tree = ast.parse(repaired)
    summary = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "step_summary"
    )
    receipt_list = summary.values[
        next(
            index
            for index, key in enumerate(summary.keys)
            if isinstance(key, ast.Constant) and key.value == "input_bindings"
        )
    ]
    assert isinstance(receipt_list, ast.List)
    assert receipt_list.elts == []
    assert '"n_rows": n_rows' in repaired


def test_contract_repair_keeps_receipts_when_host_has_typed_inputs() -> None:
    code = """
step_summary = {
    "input_bindings": [{"input_key": "raw:age", "loaded": True}],
}
"""
    finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {
            "issue": "input_binding_key_unresolved",
            "input_key": "raw:age",
            "resolved_input_keys": ["artifact:analysis_cohort"],
        },
    }

    assert deterministic_contract_repair(code=code, findings=[finding]) is None


def _attrition_identity_finding(
    *,
    expected: list[object] | None = None,
    reported: list[object] | None = None,
) -> dict:
    return {
        "validator": "primary_analysis_cohort_integrity",
        "severity": "error",
        "detail": {
            "issue": "attrition_sequence_rule_ids_mismatch",
            "expected_criterion_ids": expected
            or [
                "universe",
                "include_01_age",
                "include_02_los_icu",
                "include_03_kdigo_aki",
            ],
            "reported_criterion_ids": reported
            or [
                "universe",
                "include_01_adult",
                "include_02_los_at_least_one_day",
                "include_03_observed_stage",
            ],
        },
    }


def test_contract_repair_canonicalizes_only_proven_attrition_rule_labels():
    code = """
predicate_specs = [
    ("include_01_adult", adult_mask),
    ("include_02_los_at_least_one_day", los_mask),
    ("include_03_observed_stage", stage_mask),
]
labels = {
    "adult": "include_01_adult",
    "los": "include_02_los_at_least_one_day",
    "stage": "include_03_observed_stage",
}
for criterion_id, mask in predicate_specs:
    if criterion_id == "include_01_adult":
        label = "Adult"
"""

    repaired = deterministic_contract_repair(
        code=code,
        findings=[_attrition_identity_finding()],
    )

    assert repaired is not None
    repair_id, patched = repaired
    assert repair_id == "attrition_rule_id_canonicalization_v1"
    assert "include_01_adult" not in patched
    assert "include_02_los_at_least_one_day" not in patched
    assert "include_03_observed_stage" not in patched
    assert patched.count("include_01_age") == 3
    assert "include_02_los_icu" in patched
    assert "include_03_kdigo_aki" in patched
    ast.parse(patched)
    assert (
        deterministic_contract_repair(
            code=code,
            findings=[_attrition_identity_finding()],
            previous_repair=repair_id,
        )
        is None
    )


@pytest.mark.parametrize(
    ("expected", "reported"),
    [
        (
            ["universe", "include_01_age"],
            ["universe", "exclude_01_adult"],
        ),
        (
            ["universe", "include_01_age", "include_02_los_icu"],
            ["universe", "include_02_adult", "include_01_long_stay"],
        ),
        (
            ["universe", "include_01_age"],
            ["universe", "include_01_adult", "final_analysis_cohort"],
        ),
        (
            ["universe", "include_01_age", "include_02_los_icu"],
            ["universe", "include_01_adult", "include_01_adult"],
        ),
        (
            ["universe", "include_01_age"],
            ["universe", 1],
        ),
    ],
)
def test_attrition_rule_id_repair_rejects_unproven_sequence_mapping(
    expected: list[object],
    reported: list[object],
) -> None:
    code = 'predicate_specs = [("include_01_adult", adult_mask)]\n'

    assert (
        deterministic_contract_repair(
            code=code,
            findings=[
                _attrition_identity_finding(
                    expected=expected,
                    reported=reported,
                )
            ],
        )
        is None
    )


def test_attrition_rule_id_repair_rejects_partial_or_nonlabel_literal_coverage():
    finding = _attrition_identity_finding(
        expected=["universe", "include_01_age"],
        reported=["universe", "include_01_adult"],
    )

    for code in (
        '"""include_01_adult"""\npredicate_id = make_id()\n',
        'source_column = frame["include_01_adult"]\n',
        'predicate_id = "include_01_adult"\n',
    ):
        assert deterministic_contract_repair(code=code, findings=[finding]) is None


def test_contract_repair_fails_closed_for_unavailable_figure_source():
    code = """import pandas as pd

NOTICE = "not_estimable_notice"
SOURCE_STATUS = "unsupported"

def make_source(out_dir, source_table, frame):
    source_data_path = out_dir / "notice_source.csv"
    rows = []
    for source_row_index, row in frame.iterrows():
        rows.append({
            "source_row_index": int(source_row_index),
            "source_table": source_table,
            "spec_id": row["spec_id"],
        })
    source_frame = pd.DataFrame(
        rows,
        columns=["source_row_index", "source_table", "spec_id"],
    )
    source_frame.to_csv(source_data_path, index=False)
    return source_data_path
"""
    finding = {
        "validator": "figure_source_data",
        "severity": "error",
        "detail": {
            "reason": "incomplete_source_lineage_coverage",
            "missing_bound_tables": ["robustness_grid.csv"],
            "missing_bound_statistics": [],
        },
    }

    repair = deterministic_contract_repair(code=code, findings=[finding])

    assert repair is None


def test_unavailable_figure_source_repair_requires_typed_finding_and_notice_shape():
    code = """import pandas as pd
NOTICE = "not_estimable_notice"
SOURCE_STATUS = "unsupported"
def make_source(out_dir, source_table, frame):
    rows = []
    for source_row_index, row in frame.iterrows():
        rows.append({"source_row_index": source_row_index, "source_table": source_table})
    source_frame = pd.DataFrame(rows, columns=["source_row_index", "source_table"])
    source_frame.to_csv(out_dir / "source.csv", index=False)
"""
    finding = {
        "validator": "figure_source_data",
        "detail": {
            "reason": "incomplete_source_lineage_coverage",
            "missing_bound_tables": ["parent.csv"],
            "missing_bound_statistics": [],
        },
    }

    assert deterministic_contract_repair(code=code, findings=[]) is None
    assert (
        deterministic_contract_repair(
            code=code.replace('NOTICE = "not_estimable_notice"\n', ""),
            findings=[finding],
        )
        is None
    )
    assert (
        deterministic_contract_repair(
            code=code,
            findings=[
                {
                    **finding,
                    "detail": {
                        **finding["detail"],
                        "missing_bound_statistics": ["statistic:effect"],
                    },
                }
            ],
        )
        is None
    )


def test_contract_repair_escalates_instead_of_dropping_declared_covariates():
    code = (
        'continuous_covariates = ["age", "map_first", "lact_first"]\n'
        'source_vars_for_table = ["sepsis3", "death", "age", "map_first"]\n'
    )

    escalations = []
    repaired = deterministic_contract_repair(
        code=code,
        findings=[
            {
                "validator": "overadjustment_auditor",
                "severity": "error",
                "detail": {
                    "kind": "overadjustment",
                    "offending_covariates": ["map_first_filled"],
                },
            }
        ],
        on_semantic_escalation=escalations.append,
    )

    assert repaired is None
    assert [item.repair_id for item in escalations] == [
        "drop_overadjustment_covariates_v1"
    ]
    assert escalations[0].action == "replan_or_human_review"


def test_contract_repair_never_filters_generated_predictor_roster_at_runtime():
    code = """
x_cols = ["sepsis3"]
raw = "map_min"
model_name = "map_min_per_10mmhg"
miss_name = f"{raw}_missing_indicator"
x_cols.extend([model_name, miss_name, "age_per_10y"])
x_cols = list(dict.fromkeys(x_cols))
"""

    escalations = []
    repaired = deterministic_contract_repair(
        code=code,
        findings=[
            {
                "validator": "overadjustment_auditor",
                "severity": "error",
                "detail": {
                    "kind": "overadjustment",
                    "offending_covariates": [
                        "map_min_per_10mmhg",
                        "map_min_missing_indicator",
                    ],
                },
            }
        ],
        on_semantic_escalation=escalations.append,
    )

    assert repaired is None
    assert [item.repair_id for item in escalations] == [
        "drop_overadjustment_covariates_v1"
    ]


def test_contract_repair_keeps_measurement_provenance_receipts_machine_readable():
    code = """
import pandas as pd

provenance_receipts = [
    {
        "measured_column": "marker_measured",
        "count_column": "marker_n",
        "status": "checked",
        "comparison_n": 100,
        "invalid_pair_n": 0,
        "discordant_n": 0,
        "role": "audit_only",
    }
]
measurement_provenance_audit = pd.DataFrame.from_records(provenance_receipts)
step_summary = {
    "measurement_provenance_audit": measurement_provenance_audit,
}
""".lstrip()
    finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {
            "issue": "measurement_provenance_source_invalid",
            "reported_source": None,
        },
    }

    repaired = deterministic_contract_repair(code=code, findings=[finding])

    assert repaired is not None
    name, patched = repaired
    assert name == "measurement_provenance_summary_mapping_v2"
    assert "pd.DataFrame.from_records(provenance_receipts)" not in patched
    namespace = {}
    exec(patched, namespace)
    assert namespace["step_summary"]["measurement_provenance_audit"] == {
        "source": "COHORT_PARQUET",
        "checks": namespace["provenance_receipts"],
    }
    assert (
        deterministic_contract_repair(
            code=patched,
            findings=[finding],
            previous_repair=name,
        )
        is None
    )


def test_contract_repair_canonicalizes_closed_provenance_envelope_alias():
    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)

def main():
    receipts = [
        measurement_provenance_receipt(
            frame,
            measured_column="marker_measured",
            count_column="marker_n",
        )
    ]
    receipts_payload = {
        "source": "COHORT_PARQUET",
        "checks": receipts,
    }
    step_summary = {
        "measurement_provenance": receipts_payload,
    }
""".lstrip()
    finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {
            "issue": "measurement_provenance_source_invalid",
        },
    }

    repaired = deterministic_contract_repair(code=code, findings=[finding])

    assert repaired is not None
    name, patched = repaired
    assert name == "measurement_provenance_envelope_alias_v1"
    assert '"measurement_provenance_audit": receipts_payload' in patched
    assert '"measurement_provenance": receipts_payload' not in patched
    assert "def measurement_provenance_receipt" not in patched
    assert patched.count("measurement_provenance_receipt(") == 1
    assert (
        deterministic_contract_repair(
            code=patched,
            findings=[finding],
            previous_repair=name,
        )
        is None
    )


def test_contract_repair_does_not_rename_unproven_provenance_alias():
    code = """
step_summary = {
    "measurement_provenance": {"source": "other", "checks": []},
}
""".lstrip()
    finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {
            "issue": "measurement_provenance_check_missing",
            "measured_column": "marker_measured",
            "expected_count_column": "marker_n",
        },
    }

    assert deterministic_contract_repair(code=code, findings=[finding]) is None


def test_contract_repair_removes_only_validator_rejected_receipt_spec_after_alias():
    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)

def main(frame):
    receipt_specs = [
        ("marker_measured", "marker_n"),
        ("signal_measured", "signal_n"),
        ("extra_measured", "extra_n"),
    ]
    receipts = []
    for measured_column, count_column in receipt_specs:
        receipt = measurement_provenance_receipt(
            frame,
            measured_column=measured_column,
            count_column=count_column,
        )
        receipts.append(receipt)
    receipts_payload = {
        "source": "COHORT_PARQUET",
        "checks": receipts,
    }
    step_summary = {
        "measurement_provenance": receipts_payload,
    }
""".lstrip()
    source_finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {"issue": "measurement_provenance_source_invalid"},
    }
    alias_repair = deterministic_contract_repair(
        code=code,
        findings=[source_finding],
    )

    assert alias_repair is not None
    alias_name, canonical = alias_repair
    assert alias_name == "measurement_provenance_envelope_alias_v1"

    mismatched_path_finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {
            "issue": "measurement_provenance_check_unplanned",
            "measured_column": "extra_measured",
            "summary_path": "measurement_provenance_audit.checks.1",
            "planned_measured_columns": ["marker_measured", "signal_measured"],
        },
    }
    assert (
        deterministic_contract_repair(
            code=canonical,
            findings=[mismatched_path_finding],
            previous_repair=alias_name,
        )
        is None
    )

    unplanned_finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {
            "issue": "measurement_provenance_check_unplanned",
            "measured_column": "extra_measured",
            "summary_path": "measurement_provenance_audit.checks.2",
            "planned_measured_columns": ["marker_measured", "signal_measured"],
        },
    }
    spec_repair = deterministic_contract_repair(
        code=canonical,
        findings=[unplanned_finding],
        previous_repair=alias_name,
    )

    assert spec_repair is not None
    spec_name, patched = spec_repair
    assert spec_name == "measurement_provenance_summary_mapping_v2"
    assert '("extra_measured", "extra_n")' not in patched
    assert "('marker_measured', 'marker_n')" in patched
    assert "('signal_measured', 'signal_n')" in patched
    assert patched.count("measurement_provenance_receipt(") == 1
    assert (
        deterministic_contract_repair(
            code=patched,
            findings=[unplanned_finding],
            previous_repair=spec_name,
        )
        is None
    )


def test_contract_repair_replaces_static_custom_measurement_receipts_and_adds_missing_pair():
    code = """import pandas as pd

def measurement_receipt(frame, measured_col, count_col, value_col):
    if measured_col not in frame.columns or count_col not in frame.columns:
        raise RuntimeError("missing provenance columns")
    return {
        "value_column": value_col,
        "measured_column": measured_col,
        "count_column": count_col,
        "status": "checked",
    }

measurement_specs = [
    ("marker", "marker_measured", "marker_n"),
    ("other", "other_measured", "other_n"),
]
measurement_checks = [
    measurement_receipt(df, measured, count, value)
    for value, measured, count in measurement_specs
]
step_summary = {
    "measurement_provenance_audit": {
        "source": "COHORT_PARQUET",
        "checks": measurement_checks,
    }
}
"""
    findings = [
        {
            "validator": "step_summary_integrity",
            "severity": "error",
            "detail": {
                "issue": "measurement_provenance_check_invalid",
                "measured_column": measured,
                "expected_count_column": count,
                "expected_status": "checked",
                "invalid_fields": [
                    "role",
                    "comparison_n",
                    "invalid_pair_n",
                    "discordant_n",
                ],
            },
        }
        for measured, count in (
            ("marker_measured", "marker_n"),
            ("other_measured", "other_n"),
        )
    ]
    findings.append(
        {
            "validator": "step_summary_integrity",
            "severity": "error",
            "detail": {
                "issue": "measurement_provenance_check_missing",
                "measured_column": "exposure_measured",
                "expected_count_column": "exposure_n",
            },
        }
    )

    repaired = deterministic_contract_repair(code=code, findings=findings)

    assert repaired is not None
    name, patched = repaired
    assert name == "measurement_provenance_host_receipts_v1"
    assert "def measurement_receipt" not in patched
    assert patched.count("measurement_provenance_receipt(") == 2
    assert "for value, measured, count in measurement_specs" in patched
    assert "measured_column='exposure_measured'" in patched
    assert "count_column='exposure_n'" in patched
    assert deterministic_contract_repair(code=patched, findings=findings) is None


def test_contract_repair_replaces_direct_custom_provenance_receipts():
    code = """import pandas as pd

def provenance_receipt(frame, measured_column, count_column, concept, start, end):
    measured = pd.to_numeric(frame[measured_column], errors="coerce")
    count = pd.to_numeric(frame[count_column], errors="coerce")
    return {
        "concept": concept,
        "measured_column": measured_column,
        "count_column": count_column,
        "n_rows": int(len(frame)),
        "n_inconsistent_status_count_pairs": int(
            (((measured == 1) & (count <= 0)) | ((measured == 0) & (count > 0))).sum()
        ),
    }

lact_receipt = provenance_receipt(
    frame, "lact_measured", "lact_n", "lact", 0.0, 24.0
)
score_receipt = provenance_receipt(
    frame, "score_measured", "score_n", "score", 0.0, 24.0
)
measurement_provenance_audit = {
    "source": "COHORT_PARQUET",
    "checks": [lact_receipt, score_receipt],
}
"""
    findings = [
        {
            "validator": "step_summary_integrity",
            "severity": "error",
            "detail": {
                "issue": "measurement_provenance_check_invalid",
                "measured_column": measured,
                "expected_count_column": count,
                "expected_status": "checked",
                "invalid_fields": [
                    "role",
                    "status",
                    "comparison_n",
                    "invalid_pair_n",
                    "discordant_n",
                ],
            },
        }
        for measured, count in (
            ("lact_measured", "lact_n"),
            ("score_measured", "score_n"),
        )
    ]

    repaired = deterministic_contract_repair(code=code, findings=findings)

    assert repaired is not None
    name, patched = repaired
    assert name == "measurement_provenance_host_receipts_v1"
    assert "def provenance_receipt" not in patched
    assert patched.count("measurement_provenance_receipt(") == 2
    assert "measured_column='lact_measured', count_column='lact_n'" in patched
    assert "measured_column='score_measured', count_column='score_n'" in patched


def test_contract_repair_replaces_unplanned_split_provenance_checks_with_host_receipt():
    code = """import pandas as pd

def main(path):
    frame = pd.read_parquet(path)
    step_summary = {
        "measurement_provenance_audit": {
            "source": "COHORT_PARQUET",
            "checks": [
                {"measurement_column": "signal", "count_column": "signal_n"},
                {"measurement_column": "signal", "status_column": "signal_measured"},
            ],
        },
    }
    return step_summary
"""
    findings = [
        {
            "validator": "step_summary_integrity",
            "severity": "error",
            "detail": {
                "issue": "measurement_provenance_check_unplanned",
                "summary_path": f"measurement_provenance_audit.checks.{index}",
                "planned_measured_columns": ["signal_measured"],
            },
        }
        for index in range(2)
    ]
    findings.append(
        {
            "validator": "step_summary_integrity",
            "severity": "error",
            "detail": {
                "issue": "measurement_provenance_check_missing",
                "measured_column": "signal_measured",
                "expected_count_column": "signal_n",
            },
        }
    )

    repaired = deterministic_contract_repair(code=code, findings=findings)

    assert repaired is not None
    repair_id, patched = repaired
    assert repair_id == "measurement_provenance_summary_mapping_v2"
    assert "measurement_column" not in patched
    assert "status_column" not in patched
    assert patched.count("measurement_provenance_receipt(") == 1
    assert "measured_column='signal_measured'" in patched
    assert "count_column='signal_n'" in patched
    assert (
        deterministic_contract_repair(
            code=patched,
            findings=findings,
            previous_repair=repair_id,
        )
        is None
    )


def test_contract_repair_refuses_dynamic_or_incomplete_measurement_receipt_coordinates():
    code = """def measurement_receipt(frame, measured_col, count_col, value_col):
    return {
        "measured_column": measured_col,
        "count_column": count_col,
        "status": "checked",
    }

measurement_specs = build_measurement_specs()
measurement_checks = [
    measurement_receipt(df, measured, count, value)
    for value, measured, count in measurement_specs
]
"""
    finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {
            "issue": "measurement_provenance_check_invalid",
            "measured_column": "marker_measured",
            "expected_count_column": "marker_n",
            "expected_status": "checked",
            "invalid_fields": ["comparison_n"],
        },
    }

    assert deterministic_contract_repair(code=code, findings=[finding]) is None


def test_contract_repair_reuses_closed_nested_provenance_mapping_in_summary():
    code = """measurement_checks = [{"status": "checked"}]
diagnostics = {
    "measurement_provenance_audit": {
        "source": "COHORT_PARQUET",
        "checks": measurement_checks,
    },
}
step_summary = {
    "step": "05_diagnostics",
    "output_files": {"artifact:diagnostics": "diagnostics.json"},
}
"""
    finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {
            "issue": "measurement_provenance_source_invalid",
            "reported_source": None,
        },
    }

    repaired = deterministic_contract_repair(code=code, findings=[finding])

    assert repaired is not None
    name, patched = repaired
    assert name == "measurement_provenance_summary_mapping_v2"
    namespace: dict = {}
    exec(patched, namespace)  # noqa: S102 - deterministic test source
    assert (
        namespace["step_summary"]["measurement_provenance_audit"]
        is (namespace["diagnostics"]["measurement_provenance_audit"])
    )


def test_contract_repair_bypasses_one_shadowed_local_provenance_helper():
    code = """import pandas as pd

def measurement_provenance_receipt(*args, **kwargs):
    return {"status": "local_source_audit"}

frame = pd.read_parquet(input_path)
provenance_receipt = measurement_provenance_receipt(
    frame, value_column="marker"
)
summary = {
    "measurement_provenance_audit": {
        "source": "COHORT_PARQUET",
        "checks": [provenance_receipt],
    }
}
"""
    findings = [
        {
            "validator": "step_summary_integrity",
            "severity": "error",
            "detail": {
                "issue": "measurement_provenance_check_unplanned",
                "summary_path": "measurement_provenance_audit.checks.0",
                "planned_measured_columns": ["marker_measured"],
            },
        },
        {
            "validator": "step_summary_integrity",
            "severity": "error",
            "detail": {
                "issue": "measurement_provenance_check_missing",
                "measured_column": "marker_measured",
                "expected_count_column": "marker_n",
            },
        },
    ]

    repaired = deterministic_contract_repair(code=code, findings=findings)

    assert repaired is not None
    repair_id, patched = repaired
    assert repair_id == "measurement_provenance_summary_mapping_v2"
    assert (
        "measurement_provenance_receipt as _easyicu_measurement_provenance_receipt_v1"
    ) in patched
    assert "_easyicu_measurement_provenance_receipt_v1(frame," in patched
    assert "measured_column='marker_measured'" in patched
    assert "count_column='marker_n'" in patched


def test_contract_repair_refuses_unclosed_nested_provenance_mapping():
    code = """measurement_checks = []
diagnostics = {
    "measurement_provenance_audit": {
        "source": selected_source,
        "checks": measurement_checks,
    },
}
step_summary = {"step": "05_diagnostics"}
"""
    finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {
            "issue": "measurement_provenance_source_invalid",
            "reported_source": None,
        },
    }

    assert deterministic_contract_repair(code=code, findings=[finding]) is None


def test_contract_repair_wraps_direct_host_provenance_receipt_list():
    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)

provenance_receipts = []
provenance_receipts.append(
    measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    )
)
if not provenance_receipts:
    raise RuntimeError("no provenance checks")
measurement_provenance_audit = provenance_receipts
step_summary = {
    "measurement_provenance_audit": measurement_provenance_audit,
}
""".lstrip()
    finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {
            "issue": "measurement_provenance_source_invalid",
            "reported_source": None,
        },
    }

    repaired = deterministic_contract_repair(code=code, findings=[finding])

    assert repaired is not None
    name, patched = repaired
    assert name == "measurement_provenance_summary_mapping_v2"
    assert (
        'measurement_provenance_audit = {"source": "COHORT_PARQUET", '
        '"checks": provenance_receipts}'
    ) in patched
    assert patched.count("measurement_provenance_receipt(") == 1


def test_contract_repair_wraps_one_direct_host_receipt_in_function_scope():
    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)

def main(frame, measured_column, count_column):
    provenance = measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    )
    if provenance is None:
        raise RuntimeError("missing provenance receipt")
    step_summary = {
        "measurement_provenance_audit": provenance,
    }
    return step_summary
""".lstrip()
    finding = {
        "validator": "step_summary_integrity",
        "severity": "error",
        "detail": {
            "issue": "measurement_provenance_source_invalid",
            "reported_source": None,
        },
    }

    repaired = deterministic_contract_repair(code=code, findings=[finding])

    assert repaired is not None
    name, patched = repaired
    assert name == "measurement_provenance_summary_mapping_v2"
    assert patched.count("measurement_provenance_receipt(") == 1
    assert (
        '"measurement_provenance_audit": {"source": "COHORT_PARQUET", '
        '"checks": [provenance]}'
    ) in patched
    assert "if provenance is None:" in patched


def test_contract_repair_refuses_direct_receipt_with_another_consumer():
    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)

def main(frame, measured_column, count_column):
    provenance = measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    )
    publish(provenance)
    step_summary = {
        "measurement_provenance_audit": provenance,
    }
    return step_summary
""".lstrip()

    assert (
        deterministic_contract_repair(
            code=code,
            findings=[
                {
                    "validator": "step_summary_integrity",
                    "severity": "error",
                    "detail": {
                        "issue": "measurement_provenance_source_invalid",
                        "reported_source": None,
                    },
                }
            ],
        )
        is None
    )


def test_contract_repair_refuses_unverified_direct_provenance_list():
    code = """
provenance_receipts = [{"measured_column": "marker_measured"}]
measurement_provenance_audit = provenance_receipts
step_summary = {
    "measurement_provenance_audit": measurement_provenance_audit,
}
""".lstrip()

    assert (
        deterministic_contract_repair(
            code=code,
            findings=[
                {
                    "validator": "step_summary_integrity",
                    "detail": {
                        "issue": "measurement_provenance_source_invalid",
                        "reported_source": None,
                    },
                }
            ],
        )
        is None
    )


def test_contract_repair_refuses_provenance_frame_with_another_consumer():
    code = """
import pandas as pd

receipts = [{"measured_column": "marker_measured"}]
audit = pd.DataFrame.from_records(receipts)
audit.to_csv("review.csv", index=False)
step_summary = {"measurement_provenance_audit": audit}
""".lstrip()

    assert (
        deterministic_contract_repair(
            code=code,
            findings=[
                {
                    "validator": "step_summary_integrity",
                    "detail": {
                        "issue": "measurement_provenance_source_invalid",
                        "reported_source": None,
                    },
                }
            ],
        )
        is None
    )


def test_binary_prediction_runner_repair_is_family_gated():
    code = (
        "figure_contract = FigureContract()\n"
        "train_test_split(X, y, test_size=0.2, test_size=0.3)\n"
    )

    assert (
        _deterministic_runner_repair(
            code=code,
            run_log="SyntaxError: keyword argument repeated",
            analysis_family="survival",
        )
        is None
    )
    assert (
        _deterministic_runner_repair(
            code="model_bundle = ...\n",
            run_log="SyntaxError: invalid syntax near placeholder ellipsis",
            analysis_family="causal_inference",
        )
        is None
    )


def test_binary_summary_repair_is_family_gated():
    code = (
        "import pandas as pd\n"
        "model_df = pd.get_dummies(df[['event_time', 'sex']], columns=['sex'])\n"
        "result = logit('event_time ~ sex_male', data=model_df).fit()\n"
    )
    step_summary = {
        "primary_predictor": "sex",
        "outcome": "event_time",
        "primary_or": None,
        "error": "NameError: name 'sex_male' is not defined",
    }

    assert (
        _deterministic_summary_repair(
            code=code,
            step_summary=step_summary,
            analysis_family="survival",
        )
        is None
    )


def test_adjusted_association_models_ordinal_failure_is_not_replaced_by_fallback():
    """Primary science stays agent-owned when an ordinal adjusted model fails.

    This mirrors the Step06 failure shape: the agent planned
    ``adjusted_association_models`` with an ordinal exposure, emitted an honest
    non-fit contract, and left the primary estimate null.  Mechanical repair may
    not append a different per-unit GLM or otherwise choose a replacement
    estimand; the existing coder-repair / fail-closed path owns the outcome.
    """

    code = """
import statsmodels.api as sm

analysis_method = "adjusted_association_models"
predictor_col = "sofa2_admission"
outcome_col = "death"
model_df = df[[outcome_col, predictor_col, "age"]].dropna().copy()
y = model_df[outcome_col].astype(float)
X = sm.add_constant(model_df[[predictor_col, "age"]].astype(float))
result = sm.GLM(y, X, family=sm.families.Binomial()).fit()
"""
    step_summary = {
        "analysis_method": "adjusted_association_models",
        "primary_predictor": "sofa2_admission",
        "primary_or": None,
        "primary_ci_low": None,
        "primary_ci_high": None,
        "model_contracts": [
            {
                "model_id": "mortality_complete_case",
                "analysis_role": "primary",
                "status": "not_fitted",
                "fit_failure_reason": "zero-event ordinal cell",
            }
        ],
        "skipped": [{"reason": "model_fit_error", "error": "perfect separation"}],
    }

    assert (
        _deterministic_summary_repair(
            code=code,
            step_summary=step_summary,
            analysis_family="association",
        )
        is None
    )
    assert (
        _deterministic_runner_repair(
            code=code,
            run_log="contract failed: required adjusted association model not fitted",
            analysis_family="association",
        )
        is None
    )


# ---------------------------------------------------------------------------
# _strip_columns_from_list_literals
# ---------------------------------------------------------------------------


class TestStripColumnsFromListLiterals:
    def test_no_missing_cols_is_noop(self):
        code = "x = ['a', 'b', 'c']"
        assert _strip_columns_from_list_literals(code, []) == code

    def test_strips_named_column_from_simple_list(self):
        code = "covariates = ['age', 'sex', 'sofa_total']"
        result = _strip_columns_from_list_literals(code, ["sofa_total"])
        assert result == "covariates = ['age', 'sex']"

    def test_double_quoted_literals_also_stripped(self):
        code = 'cols = ["age", "sofa_total"]'
        result = _strip_columns_from_list_literals(code, ["sofa_total"])
        assert result == 'cols = ["age"]'

    def test_leaves_non_literal_lists_alone(self):
        """Documented conservative behaviour: mixed lists are untouched."""
        code = "cols = [outcome_col, 'sofa_total']"
        # First element is a bare name, not a string literal — must not edit.
        assert _strip_columns_from_list_literals(code, ["sofa_total"]) == code

    def test_leaves_unrelated_lists_alone(self):
        code = "scores = [1, 2, 3]\nletters = ['a', 'b']"
        # Neither list contains any of the missing columns.
        assert _strip_columns_from_list_literals(code, ["sofa_total"]) == code

    def test_result_is_still_valid_python(self):
        """The rewriter must never produce un-parseable code."""
        code = "covariates = ['age', 'sex', 'sofa_total']"
        result = _strip_columns_from_list_literals(code, ["sofa_total"])
        ast.parse(result)


# ---------------------------------------------------------------------------
# _patch_json_dump_numpy_key_sanitizer
# ---------------------------------------------------------------------------


class TestPatchJsonDumpNumpyKeySanitizer:
    def test_prepends_helper_when_absent(self):
        code = "import json\njson.dump({1: 2}, open('x', 'w'))"
        patched = _patch_json_dump_numpy_key_sanitizer(code)
        assert "_easyicu_json_sanitize_v1" in patched
        assert patched.endswith(code)

    def test_idempotent_when_helper_already_present(self):
        code = "import json\njson.dump({1: 2}, open('x', 'w'))"
        once = _patch_json_dump_numpy_key_sanitizer(code)
        twice = _patch_json_dump_numpy_key_sanitizer(once)
        assert once == twice, "second application must be a no-op"

    def test_patched_output_is_valid_python(self):
        code = "import json\njson.dumps({'k': 1})"
        patched = _patch_json_dump_numpy_key_sanitizer(code)
        ast.parse(patched)


def test_runner_repair_sanitizes_numpy_boolean_values_before_json_dump():
    code = """
import json
import numpy as np

step_summary = {"converged": np.bool_(True)}
with open("step_summary.json", "w", encoding="utf-8") as handle:
    json.dump(step_summary, handle, indent=2, allow_nan=False)
print(json.dumps(step_summary, indent=2, allow_nan=False))
"""
    repair = _deterministic_runner_repair(
        code=code,
        run_log="TypeError: Object of type bool is not JSON serializable",
    )

    assert repair is not None
    repair_id, patched = repair
    assert repair_id == "json_dump_numpy_key_sanitizer_v1"
    assert "_easyicu_json_sanitize_v1" in patched


def test_runner_repair_sanitizes_nonfinite_values_before_strict_json_dump(
    tmp_path, monkeypatch
):
    code = """
import json

step_summary = {"score": float("nan")}
with open("step_summary.json", "w", encoding="utf-8") as handle:
    json.dump(step_summary, handle, indent=2, allow_nan=False)
"""
    repair = _deterministic_runner_repair(
        code=code,
        run_log="ValueError: Out of range float values are not JSON compliant: nan",
    )

    assert repair is not None
    repair_id, patched = repair
    assert repair_id == "json_dump_numpy_key_sanitizer_v1"
    assert "_easyicu_json_sanitize_v1" in patched
    monkeypatch.chdir(tmp_path)
    exec(compile(patched, "<patched>", "exec"), {})
    assert json.loads((tmp_path / "step_summary.json").read_text()) == {"score": None}


def test_runner_repair_does_not_trigger_case_fallbacks_by_default():
    """Default repair path must stay case-neutral.

    Lactate / MAP / vasopressor study fallbacks are allowed only through an
    explicitly registered CasePluginRegistry, never from shared code_repair.
    """

    probes = [
        (
            "norepi_equiv_max_24h = 1\n",
            "ModuleNotFoundError: No module named 'statsmodels'",
        ),
        (
            "age = df['age']\ndeath = df['death']\n# tertile mortality\n",
            "TypeError: got an unexpected keyword argument 'observed'",
        ),
        (
            "# t04_lactate_mortality_association\nlactate_max_24h = 1\n",
            "Traceback\nKeyError: required columns",
        ),
    ]
    for code, run_log in probes:
        assert _deterministic_runner_repair(code=code, run_log=run_log) is None


@pytest.mark.parametrize(
    ("code", "run_log", "repair_id"),
    [
        (
            "model_bundle = ...\n",
            "SyntaxError: invalid syntax near placeholder ellipsis",
            "prediction_discrimination_template_v1",
        ),
        (
            "pd.DataFrame().to_csv('table_one.csv')\n",
            "SyntaxError: '(' was never closed",
            "table_one_descriptive_repair_v1",
        ),
        (
            "# outcome_incidence\n...\n",
            "SyntaxError: invalid syntax",
            "outcome_incidence_descriptive_repair_v1",
        ),
    ],
)
def test_analysis_template_repairs_escalate_instead_of_replacing_science(
    code,
    run_log,
    repair_id,
):
    escalations = []

    repaired = _deterministic_runner_repair(
        code=code,
        run_log=run_log,
        on_semantic_escalation=escalations.append,
    )

    assert repaired is None
    assert [item.repair_id for item in escalations] == [repair_id]


# ---------------------------------------------------------------------------
# deterministic_concept_audit_repair — scientific choices still route to agent
# repair, while an explicitly diagnosed missing fail-close guard is mechanical.
# ---------------------------------------------------------------------------


def test_concept_repair_does_not_choose_complete_case_for_zero_imputation():
    code = (
        "mi_df = analysis_df.copy()\n"
        'mi_df["lact"] = mi_df["lact"].fillna(0)\n'
        "mi_df[primary_predictor] = mi_df[primary_predictor].fillna(0)\n"
    )
    out, names = deterministic_concept_audit_repair(
        code, ["Imputed missing lactate values with 0"]
    )
    assert names == []
    assert out == code


def test_concept_repair_no_op_when_auditor_did_not_flag_zero_impute():
    # Impartiality: without an objective error naming the anti-pattern we do
    # not touch the code — the choice to fillna stays the user's to defend.
    code = 'df["lact"] = df["lact"].fillna(0)\n'
    out, names = deterministic_concept_audit_repair(
        code, ["ordinal score summarised as a continuous median"]
    )
    assert names == []
    assert out == code


def test_concept_repair_preserves_zero_on_count_columns():
    # 0 is a real value for a component-completeness count; stripping it
    # would itself be an error.
    code = 'df["sofa2_n_components"] = df["sofa2_n_components"].fillna(0)\n'
    out, names = deterministic_concept_audit_repair(code, ["fillna(0) detected"])
    assert names == []
    assert out == code


def test_concept_repair_is_idempotent():
    code = 'df["lact"] = df["lact"].fillna(0)\n'
    once, names1 = deterministic_concept_audit_repair(code, ["fillna(0) on lactate"])
    assert names1 == []
    assert once == code
    twice, names2 = deterministic_concept_audit_repair(once, ["fillna(0) on lactate"])
    assert names2 == []
    assert twice == once


def test_concept_repair_binds_closed_counts_helper_without_runtime_introspection():
    code = """
import inspect
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def invoke_counts(series, levels):
    signature = inspect.signature(closed_categorical_counts)
    parameters = list(signature.parameters)
    if "levels" in parameters:
        return closed_categorical_counts(series, levels=levels)
    return closed_categorical_counts(series, declared_levels=levels)

result = invoke_counts(series, levels)
""".lstrip()
    finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="stable helper must not be introspected",
        detail={
            "reason": "host_helper_runtime_introspection",
            "helper_name": "closed_categorical_counts",
            "line": 5,
        },
    )

    out, names = deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[RepairReason.INVALID_HELPER_SIGNATURE],
        repair_findings=[finding],
    )

    assert names == ["closed_counts_direct_host_call_v1"]
    assert "import inspect" not in out
    assert "inspect.signature" not in out
    assert (
        "return closed_categorical_counts(\n        series, declared_levels=levels\n    )"
        in out
    )
    assert deterministic_concept_audit_repair(
        out,
        [finding.message],
        repair_reasons=[RepairReason.INVALID_HELPER_SIGNATURE],
        repair_findings=[finding],
    ) == (out, [])


def test_concept_repair_does_not_rewrite_aliased_closed_counts_adapter():
    code = """
from inspect import signature as inspect_signature
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts as counts

def invoke_counts(series, levels):
    inspect_signature(counts)
    return counts(series, declared_levels=levels)

result = invoke_counts(series, levels)
""".lstrip()
    finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="stable helper must not be introspected",
        detail={
            "reason": "host_helper_runtime_introspection",
            "helper_name": "closed_categorical_counts",
            "line": 5,
        },
    )

    assert deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[RepairReason.INVALID_HELPER_SIGNATURE],
        repair_findings=[finding],
    ) == (code, [])


def test_concept_repair_retires_arbitrary_column_fallback_using_authored_raise():
    code = """
def extract_count_table(table):
    level_col = None
    count_col = None
    if level_col is None or count_col is None:
        if table.shape[1] == 2:
            level_col, count_col = table.columns[0], table.columns[1]
        else:
            raise RuntimeError("declared level/count columns are missing")
    return level_col, count_col
""".lstrip()
    finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="frame-order fallback is forbidden",
        detail={
            "reason": "arbitrary_column_fallback",
            "line": 6,
            "function": "extract_count_table",
        },
    )

    out, names = deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[RepairReason.ARBITRARY_COLUMN_FALLBACK],
        repair_findings=[finding],
    )

    assert names == ["arbitrary_column_fallback_fail_closed_v1"]
    assert "table.columns[0]" not in out
    assert "declared level/count columns are missing" in out
    assert deterministic_concept_audit_repair(
        out,
        [finding.message],
        repair_reasons=[RepairReason.ARBITRARY_COLUMN_FALLBACK],
        repair_findings=[finding],
    ) == (out, [])


def test_concept_repair_replaces_custom_provenance_helper_with_host_receipt():
    code = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt

def audit_pair(frame, measured_col, count_col):
    receipt = measurement_provenance_receipt(
        frame, measured_column=measured_col, count_column=count_col
    )
    return {
        "role": "audit_only",
        "invalid_pair_n": receipt["invalid_pair_n"],
        "discordant_n": receipt["discordant_n"],
    }

checks = [
    audit_pair(frame, "lactate_measured", "lactate_n"),
    audit_pair(frame, "creatinine_measured", "creatinine_n"),
]
""".lstrip()
    finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="custom provenance helper is not bound fail-closed",
        detail={
            "reason": "provenance_audit_not_fail_closed",
            "issues": [
                {
                    "failure_mode": "provenance_helper_result_not_bound",
                    "helper_name": "audit_pair",
                    "call_line": 14,
                },
                {
                    "failure_mode": "provenance_helper_result_not_bound",
                    "helper_name": "audit_pair",
                    "call_line": 15,
                },
            ],
        },
    )

    out, names = deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[RepairReason.PROVENANCE_NOT_FAIL_CLOSED],
        repair_findings=[finding],
    )

    assert names == ["provenance_custom_helper_to_host_receipt_v1"]
    assert "def audit_pair" not in out
    assert out.count("measurement_provenance_receipt(") == 2
    assert 'measured_column="lactate_measured"' in out
    assert 'count_column="creatinine_n"' in out
    assert deterministic_concept_audit_repair(
        out,
        [finding.message],
        repair_reasons=[RepairReason.PROVENANCE_NOT_FAIL_CLOSED],
        repair_findings=[finding],
    ) == (out, [])


def test_concept_repair_removes_superseded_manual_provenance_audit():
    code = """
import pandas as pd
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt

df = pd.read_parquet(input_path)
invalid_pair_n = int((df["measured"].notna() & df["count"].isna()).sum())
discordant_n = int((df["measured"] != (df["count"] > 0)).sum())
measurement_provenance_audit = {
    "checks": [{
        "role": "audit_only",
        "invalid_pair_n": invalid_pair_n,
        "discordant_n": discordant_n,
    }],
}
receipts = []
receipts.append(measurement_provenance_receipt(
    frame,
    measured_column="measured",
    count_column="count",
))
measurement_provenance_audit = {"source": "COHORT_PARQUET", "checks": receipts}
step_summary = {"measurement_provenance_audit": measurement_provenance_audit}
""".lstrip()
    finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="module provenance scope is not proven fail-closed",
        detail={
            "reason": "provenance_audit_not_fail_closed",
            "issues": [
                {
                    "failure_mode": "module_provenance_scope_not_proven_fail_closed",
                    "helper_name": "<module>",
                }
            ],
        },
    )

    out, names = deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[RepairReason.PROVENANCE_NOT_FAIL_CLOSED],
        repair_findings=[finding],
    )

    assert names == ["superseded_manual_provenance_receipt_v1"]
    assert '"invalid_pair_n": invalid_pair_n' not in out
    assert "measurement_provenance_receipt(\n    df," in out
    assert deterministic_concept_audit_repair(
        out,
        [finding.message],
        repair_reasons=[RepairReason.PROVENANCE_NOT_FAIL_CLOSED],
        repair_findings=[finding],
    ) == (out, [])


def test_concept_repair_refuses_ambiguous_provenance_frame_source():
    code = """
import pandas as pd
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt

left = pd.read_parquet(left_path)
right = pd.read_parquet(right_path)
measurement_provenance_audit = {
    "checks": [{"role": "audit_only", "invalid_pair_n": 0, "discordant_n": 0}],
}
receipts = []
receipts.append(measurement_provenance_receipt(
    frame,
    measured_column="measured",
    count_column="count",
))
measurement_provenance_audit = {"source": "COHORT_PARQUET", "checks": receipts}
step_summary = {"measurement_provenance_audit": measurement_provenance_audit}
""".lstrip()
    finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="module provenance scope is not proven fail-closed",
        detail={
            "reason": "provenance_audit_not_fail_closed",
            "issues": [
                {
                    "failure_mode": "module_provenance_scope_not_proven_fail_closed",
                    "helper_name": "<module>",
                }
            ],
        },
    )

    assert deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[RepairReason.PROVENANCE_NOT_FAIL_CLOSED],
        repair_findings=[finding],
    ) == (code, [])


def test_concept_repair_adds_only_exact_llm_proven_domain_guards():
    code = """continuous_covariates = ["mech_support"]
availability_covariates = []
for col in continuous_covariates + availability_covariates:
    df[col] = strict_numeric(df[col], col)

gcs_numeric = strict_numeric(df["score"], "score")
if ((gcs_numeric.dropna() < 3) | (gcs_numeric.dropna() > 15)).any():
    raise RuntimeError("score outside range")
df["score_level"] = gcs_numeric.round().astype("Int64").astype("object")
"""
    findings = [
        ValidationFinding(
            validator="llm_concept_auditor",
            severity="error",
            message="ordinal is rounded",
            detail={
                "issue_code": "other",
                "variable": "score",
                "problem": "invalid ordinal levels are not rejected before rounding",
            },
        ),
        ValidationFinding(
            validator="llm_concept_auditor",
            severity="error",
            message="binary input lacks a guard",
            detail={
                "issue_code": "other",
                "variable": "mech_support",
                "problem": "binary domain validation is missing",
            },
        ),
    ]

    patched, names = deterministic_concept_audit_repair(
        code,
        [finding.message for finding in findings],
        repair_reasons=[RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION],
        repair_findings=findings,
    )

    assert names == ["llm_proven_numeric_domain_guards_v1"]
    assert "gcs_numeric.dropna().mod(1).eq(0).all()" in patched
    assert "set(df['mech_support'].dropna().unique()).issubset({0, 1})" in patched
    assert deterministic_concept_audit_repair(
        patched,
        [finding.message for finding in findings],
        repair_reasons=[RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION],
        repair_findings=findings,
    ) == (patched, [])


def test_concept_repair_refuses_unproven_domain_guard_shapes():
    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="binary input lacks a guard",
        detail={
            "issue_code": "other",
            "variable": "mech_support",
            "problem": "binary domain validation is missing",
        },
    )
    code = """continuous_covariates = build_covariates()
for col in continuous_covariates:
    df[col] = strict_numeric(df[col], col)
"""

    assert deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION],
        repair_findings=[finding],
    ) == (code, [])


def test_concept_repair_inserts_provenance_fail_closed_guard():
    code = """
def measurement_provenance_audit(frame):
    return {
        "invalid_pair_n": 0,
        "discordant_n": 0,
        "audit_only": True,
        "fail_closed": True,
        "completed_step_allowed": False,
    }

def main(frame):
    provenance = measurement_provenance_audit(frame)
    write_scientific_outputs(frame)

if __name__ == "__main__":
    main(frame)
""".lstrip()
    out, names = deterministic_concept_audit_repair(
        code,
        [
            "A measurement-provenance audit records invalid or discordant "
            "pairs but does not fail the completed step before scientific "
            "outputs can be published."
        ],
    )

    assert names == ["provenance_fail_closed_guard_v1"]
    assert "_easyicu_provenance_fail_closed_guard_v1" in out
    tree = ast.parse(out)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert isinstance(main.body[1], ast.If)
    assert isinstance(main.body[1].body[0], ast.Raise)


def test_concept_repair_guards_direct_provenance_contract_and_status():
    code = """
def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def main(frame):
    invalid_pair_n = int(frame["invalid_pair_n"])
    discordant_n = int(frame["discordant_n"])
    measurement_provenance_audit = {
        "source": "COHORT_PARQUET",
        "checks": [{
            "status": "checked",
            "role": "audit_only",
            "invalid_pair_n": invalid_pair_n,
            "discordant_n": discordant_n,
        }],
    }
    provenance_checks = measurement_provenance_audit.get("checks")
    provenance_failures = [
        check
        for check in provenance_checks
        if check.get("status") not in {"passed", "ok", "valid"}
    ]
    require(len(provenance_failures) == 0, "provenance failed")
    return measurement_provenance_audit

result = main(frame)
""".lstrip()
    finding = ValidationFinding(
        validator="mechanical_code_preflight",
        severity="error",
        message="provenance_audit_not_fail_closed",
        detail={
            "reason": "provenance_audit_not_fail_closed",
            "issues": [
                {
                    "failure_mode": "provenance_helper_result_not_bound",
                    "helper_name": "main",
                    "call_line": 25,
                }
            ],
        },
    )

    out, names = deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[RepairReason.PROVENANCE_NOT_FAIL_CLOSED],
        repair_findings=[finding],
    )

    assert names == [
        "provenance_fail_closed_guard_v1",
        "provenance_checked_status_contract_v1",
    ]
    assert "_easyicu_provenance_fail_closed_guard_v1" in out
    assert '{"passed", "ok", "valid", "checked"}' in out
    namespace = {"frame": {"invalid_pair_n": 0, "discordant_n": 0}}
    exec(out, namespace)
    assert namespace["result"]["checks"][0]["status"] == "checked"
    with pytest.raises(RuntimeError, match="scientific outputs were not published"):
        exec(out, {"frame": {"invalid_pair_n": 1, "discordant_n": 0}})
    assert deterministic_concept_audit_repair(
        out,
        [finding.message],
        repair_reasons=[RepairReason.PROVENANCE_NOT_FAIL_CLOSED],
        repair_findings=[finding],
    ) == (out, [])


def test_concept_repair_does_not_infer_provenance_policy_from_counts():
    code = """
def measurement_provenance_audit(frame):
    return {"invalid_pair_n": 1, "discordant_n": 0, "audit_only": True}

provenance = measurement_provenance_audit(frame)
""".lstrip()
    out, names = deterministic_concept_audit_repair(
        code, ["provenance_audit_not_fail_closed"]
    )
    assert names == []
    assert out == code


def test_concept_repair_terminates_full_inline_provenance_failure_branch():
    code = """
def main(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    audit = {
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }
    if invalid_pair_n > 0 or discordant_n > 0:
        final_mask = False
        summary['status'] = 'failed_provenance_audit'
    publish_outputs(frame, final_mask)

if __name__ == "__main__":
    main(frame)
""".lstrip()

    out, names = deterministic_concept_audit_repair(
        code, ["provenance_audit_not_fail_closed"]
    )

    assert names == ["provenance_fail_closed_guard_v1"]
    assert "_easyicu_provenance_fail_closed_guard_v1" in out
    tree = ast.parse(out)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    failure_guard = next(node for node in main.body if isinstance(node, ast.If))
    assert isinstance(failure_guard.body[0], ast.Raise)


@pytest.mark.parametrize(
    "body",
    [
        """
    if invalid_pair_n > 0:
        summary['status'] = 'failed_provenance_audit'
""",
        """
    model.fit(frame)
    if invalid_pair_n > 0 or discordant_n > 0:
        summary['status'] = 'failed_provenance_audit'
""",
    ],
)
def test_concept_repair_does_not_insert_unsafe_inline_provenance_guard(body):
    code = (
        "def main(frame):\n"
        "    invalid_pair_n = 1\n"
        "    discordant_n = 0\n"
        "    audit = {'role': 'audit_only', 'invalid_pair_n': invalid_pair_n, "
        "'discordant_n': discordant_n}\n"
        f"{body}"
    )

    out, names = deterministic_concept_audit_repair(
        code, ["provenance_audit_not_fail_closed"]
    )

    assert names == []
    assert out == code


def test_concept_repair_provenance_guard_is_idempotent():
    code = """
def provenance_audit(frame):
    return {
        "invalid_pair_n": 0,
        "discordant_n": 0,
        "audit_only": True,
        "provenance_valid": False,
    }

result = provenance_audit(frame)
""".lstrip()
    messages = ["provenance_audit_not_fail_closed"]
    once, names1 = deterministic_concept_audit_repair(code, messages)
    twice, names2 = deterministic_concept_audit_repair(once, messages)
    assert names1 == ["provenance_fail_closed_guard_v1"]
    assert names2 == []
    assert twice == once


def test_concept_repair_expands_provenance_pair_scan_bidirectionally():
    code = """
def measurement_provenance_audit(frame, measured_columns):
    checks = []
    for measured_column in measured_columns:
        if not measured_column.endswith("_measured"):
            continue
        count_column = measured_column.replace("_measured", "_n")
        invalid_pair_n = (
            0
            if measured_column in frame.columns and count_column in frame.columns
            else 1
        )
        checks.append({
            "role": "audit_only",
            "invalid_pair_n": invalid_pair_n,
            "discordant_n": 0,
            "measured_column": measured_column,
        })
    failed = any(row["invalid_pair_n"] for row in checks)
    return {
        "checks": checks,
        "fail_closed": failed,
        "completed_step_allowed": not failed,
    }
""".lstrip()
    out, names = deterministic_concept_audit_repair(
        code,
        [
            "The measurement-provenance audit scans measured columns only "
            "and cannot fail closed for count-only concepts."
        ],
    )

    assert names == ["provenance_bidirectional_pair_scan_v1"]
    assert "_easyicu_provenance_bidirectional_pair_scan_v1" in out
    namespace = {}
    exec(out, namespace)
    frame = type("Frame", (), {"columns": ["lact_n"]})()
    result = namespace["measurement_provenance_audit"](frame, [])
    assert result["completed_step_allowed"] is False
    assert result["checks"][0]["measured_column"] == "lact_measured"


def test_concept_repair_bidirectional_pair_scan_is_idempotent():
    code = """
def provenance_audit(frame, measured_columns):
    for measured_column in measured_columns:
        if measured_column.endswith("_measured"):
            checks = [{"role": "audit_only", "invalid_pair_n": 0, "discordant_n": 0}]
    return {"fail_closed": False, "completed_step_allowed": True}
""".lstrip()
    messages = ["provenance_pair_scan_not_bidirectional"]
    once, names1 = deterministic_concept_audit_repair(code, messages)
    twice, names2 = deterministic_concept_audit_repair(once, messages)
    assert names1 == ["provenance_bidirectional_pair_scan_v1"]
    assert names2 == []
    assert twice == once


def test_concept_repair_normalizes_first_time_companion_suffix():
    code = """
def timing_columns(candidates):
    return [f"{candidate}_first_time" for candidate in candidates]
""".lstrip()
    out, names = deterministic_concept_audit_repair(
        code,
        ["double_first_time_companion_suffix"],
    )

    assert names == ["normalize_first_time_companion_v1"]
    namespace = {}
    exec(out, namespace)
    assert namespace["timing_columns"](["gcs_first", "lact"]) == [
        "gcs_first_time",
        "lact_first_time",
    ]
    twice, names2 = deterministic_concept_audit_repair(
        out,
        ["double_first_time_companion_suffix"],
    )
    assert names2 == []
    assert twice == out


def test_missing_plotting_library_escalates_instead_of_swapping_plot_method():
    escalations = []
    repaired = _deterministic_runner_repair(
        code="import seaborn as sns\n",
        run_log="ModuleNotFoundError: No module named 'seaborn'",
        previous_repair=None,
        on_semantic_escalation=escalations.append,
    )

    assert repaired is None
    assert [item.repair_id for item in escalations] == [
        "seaborn_matplotlib_fallback_v1"
    ]


def _merge_collision_script(*, disagree: bool = False) -> str:
    right_value = 9 if disagree else 1
    return f"""import pandas as pd
analysis_df = pd.DataFrame({{"stay_id": [1, 2], "exposure": [0, 1]}})
exposure_column = "exposure"
exposure_product = pd.DataFrame({{
    "stay_id": [1, 2], "exposure": [0, {right_value}]
}})
exposure_part = exposure_product[["stay_id", exposure_column]].copy()
df = analysis_df.merge(exposure_part, on="stay_id", how="left", validate="one_to_one")
result = df[exposure_column].tolist()
"""


def test_runner_repair_guards_equal_dynamic_merge_column_before_deduplication():
    repair = _deterministic_runner_repair(
        code=_merge_collision_script(),
        run_log="KeyError: 'exposure'",
    )

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "pandas_merge_dynamic_column_collision_guard_v1"
    assert "_easyicu_merge_left_values_v1.equals" in repaired
    namespace: dict = {}
    exec(repaired, namespace)  # noqa: S102 - deterministic test source
    assert namespace["result"] == [0, 1]


def test_runner_repair_fails_closed_when_duplicate_merge_columns_disagree():
    repair = _deterministic_runner_repair(
        code=_merge_collision_script(disagree=True),
        run_log="KeyError: 'exposure'",
    )

    assert repair is not None
    with pytest.raises(RuntimeError, match="disagrees across typed inputs"):
        exec(repair[1], {})  # noqa: S102 - deterministic test source


def test_runner_repair_does_not_guess_ambiguous_merge_collision():
    code = _merge_collision_script() + (
        'df2 = analysis_df.merge(exposure_part, on="stay_id")\n'
        "other = df2[exposure_column]\n"
    )

    assert (
        _deterministic_runner_repair(code=code, run_log="KeyError: 'exposure'") is None
    )


def _table_one_secondary_overlay_script(*, resolved_right: bool = True) -> str:
    right_source = (
        'loaded_products["artifact:validated_measurement_set"]'
        if resolved_right
        else "untrusted_frame"
    )
    return f"""import pandas as pd
from easyicu.research_agent.methods.table_one import build_grouped_table_one
table_one_spec = {{"group_by": "outcome", "variables": [{{"name": "marker"}}]}}
loaded_products = {{
    "artifact:analysis_cohort": pd.DataFrame(
        {{"stay_id": [1, 2], "outcome": [0, 1], "marker": [999.0, 999.0]}}
    ),
    "artifact:validated_measurement_set": pd.DataFrame(
        {{"stay_id": [1, 2], "marker": [1.0, 2.0]}}
    )
}}
analysis_frame = loaded_products["artifact:analysis_cohort"]
validated_frame = {right_source}
required_validated_columns = ["stay_id", "marker"]
if any(column not in validated_frame.columns for column in required_validated_columns):
    raise ValueError("validated input is incomplete")
frame = analysis_frame.merge(
    validated_frame,
    on="stay_id",
    how="left",
    validate="one_to_one",
)
table_one = build_grouped_table_one(frame, table_one_spec)
"""


def test_runner_repair_preserves_authored_table_one_secondary_overlay():
    repair = _deterministic_runner_repair(
        code=_table_one_secondary_overlay_script(),
        run_log="TableOneContractError: Table 1 input columns are missing: ['marker']",
    )

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "pandas_merge_dynamic_column_collision_guard_v1"
    assert "analysis_frame.drop(columns=['marker']).merge" in repaired


def test_runner_repair_keeps_left_cohort_measurement_provenance_canonical():
    code = (
        _table_one_secondary_overlay_script()
        + """
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)
measurement_pairs = [("marker_measured", "marker_n")]
required_analysis_columns = ["stay_id", "marker_measured", "marker_n"]
if any(column not in analysis_frame.columns for column in required_analysis_columns):
    raise ValueError("analysis cohort is incomplete")
measurement_checks = [
    measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    )
    for measured_column, count_column in measurement_pairs
]
step_summary = {
    "measurement_provenance_audit": {
        "source": "artifact:analysis_cohort plus artifact:validated_measurement_set",
        "checks": measurement_checks,
    }
}
"""
    )

    repair = _deterministic_runner_repair(
        code=code,
        run_log="TableOneContractError: Table 1 input columns are missing: ['marker']",
    )

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "pandas_merge_dynamic_column_collision_guard_v1"
    assert "\"source\": 'COHORT_PARQUET'" in repaired
    assert "analysis_frame.drop(columns=['marker']).merge" in repaired


def _repaired_table_one_overlay_with_provenance() -> str:
    return (
        _table_one_secondary_overlay_script().replace(
            "analysis_frame.merge(",
            "analysis_frame.drop(columns=['marker']).merge(",
            1,
        )
        + """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
measurement_pairs = [("marker_measured", "marker_n")]
required_analysis_columns = ["stay_id", "marker_measured", "marker_n"]
if any(column not in analysis_frame.columns for column in required_analysis_columns):
    raise ValueError("analysis cohort is incomplete")
measurement_checks = [
    measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    )
    for measured_column, count_column in measurement_pairs
]
step_summary = {
    "measurement_provenance_audit": {
        "source": "artifact:analysis_cohort plus artifact:validated_measurement_set",
        "checks": measurement_checks,
    }
}
"""
    )


def _measurement_source_invalid_finding() -> dict:
    return {
        "validator": "step_summary_integrity",
        "detail": {
            "issue": "measurement_provenance_source_invalid",
            "reported_source": (
                "artifact:analysis_cohort plus artifact:validated_measurement_set"
            ),
        },
    }


def test_contract_repair_canonicalizes_proven_left_cohort_provenance_source():
    code = _repaired_table_one_overlay_with_provenance()

    repair = deterministic_contract_repair(
        code=code,
        findings=[_measurement_source_invalid_finding()],
    )

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "measurement_provenance_summary_mapping_v2"
    assert "\"source\": 'COHORT_PARQUET'" in repaired
    assert "analysis_frame.drop(columns=['marker']).merge" in repaired
    assert (
        repaired.replace(
            "'COHORT_PARQUET'",
            '"artifact:analysis_cohort plus artifact:validated_measurement_set"',
        )
        == code
    )


def test_contract_repair_refuses_unbound_table_one_left_provenance_source():
    code = _repaired_table_one_overlay_with_provenance().replace(
        'analysis_frame = loaded_products["artifact:analysis_cohort"]',
        "analysis_frame = untrusted_frame",
        1,
    )

    assert (
        deterministic_contract_repair(
            code=code,
            findings=[_measurement_source_invalid_finding()],
        )
        is None
    )


def test_contract_repair_refuses_unchecked_table_one_left_provenance_source():
    code = _repaired_table_one_overlay_with_provenance().replace(
        "if any(column not in analysis_frame.columns for column in required_analysis_columns):\n"
        '    raise ValueError("analysis cohort is incomplete")\n',
        "",
        1,
    )

    assert (
        deterministic_contract_repair(
            code=code,
            findings=[_measurement_source_invalid_finding()],
        )
        is None
    )


def test_runner_repair_rejects_unresolved_table_one_secondary_overlay():
    assert (
        _deterministic_runner_repair(
            code=_table_one_secondary_overlay_script(resolved_right=False),
            run_log=(
                "TableOneContractError: Table 1 input columns are missing: ['marker']"
            ),
        )
        is None
    )


def test_runner_repair_uses_unique_near_match_mapping_alias():
    code = """propensity_diagnostics = {"fit": {}}
positivity_diagnostics.update({"status": "not_fitted"})
result = positivity_diagnostics["fit"]
"""
    log = (
        "NameError: name 'positivity_diagnostics' is not defined. "
        "Did you mean: 'propensity_diagnostics'?"
    )

    repair = _deterministic_runner_repair(code=code, run_log=log)

    assert repair is not None
    assert repair[0] == "undefined_mapping_near_match_alias_v1"
    assert "positivity_diagnostics" not in repair[1]
    namespace: dict = {}
    exec(repair[1], namespace)  # noqa: S102 - deterministic test source
    assert namespace["propensity_diagnostics"]["status"] == "not_fitted"


@pytest.mark.parametrize(
    "code",
    [
        "propensity_diagnostics = object()\npositivity_diagnostics.update({})\n",
        "propensity_diagnostics = {}\npositivity_diagnostics = {}\n",
        "propensity_result = {}\npositivity_diagnostics.update({})\n",
    ],
)
def test_runner_repair_refuses_unproven_near_match_alias(code):
    log = (
        "NameError: name 'positivity_diagnostics' is not defined. "
        "Did you mean: 'propensity_diagnostics'?"
    )

    assert _deterministic_runner_repair(code=code, run_log=log) is None

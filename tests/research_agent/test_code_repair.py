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


def test_prediction_split_repair_requires_explicit_outcome_col():
    repaired = _deterministic_runner_repair(
        code=(
            "figure_contract = FigureContract()\n"
            "train_test_split(X, y, test_size=0.2, test_size=0.3)\n"
        ),
        run_log="SyntaxError: keyword argument repeated",
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "prediction_split_minimal_v1"
    assert 'os.environ.get("OUTCOME_COL")' in patched
    assert "df.columns[-1]" not in patched
    assert '"death" if "death" in df.columns' not in patched


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


def test_contract_repair_preserves_full_parent_for_unavailable_figure_source(
    tmp_path,
):
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

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "unavailable_figure_full_source_projection_v1"
    namespace = {}
    exec(repaired, namespace)
    source_path = namespace["make_source"](
        tmp_path,
        "robustness_grid.csv",
        namespace["pd"].DataFrame(
            {
                "spec_id": ["primary", "secondary"],
                "n_analysis": [515, 1000],
                "mortality_pct": [15.1, 10.2],
            }
        ),
    )
    observed = namespace["pd"].read_csv(source_path)
    assert observed.columns.tolist() == [
        "source_row_index",
        "source_table",
        "spec_id",
        "n_analysis",
        "mortality_pct",
    ]
    assert observed["source_row_index"].tolist() == [0, 1]
    assert observed["source_table"].tolist() == [
        "robustness_grid.csv",
        "robustness_grid.csv",
    ]
    assert observed["n_analysis"].tolist() == [515, 1000]
    assert (
        deterministic_contract_repair(
            code=code,
            findings=[finding],
            previous_repair=repair_id,
        )
        is None
    )


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


def test_contract_repair_drops_overadjustment_covariates():
    code = (
        'continuous_covariates = ["age", "map_first", "lact_first"]\n'
        'source_vars_for_table = ["sepsis3", "death", "age", "map_first"]\n'
    )

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
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "drop_overadjustment_covariates_v1"
    assert '"map_first"' not in patched
    assert '"lact_first"' in patched


def test_contract_repair_filters_generated_overadjustment_covariates_at_runtime():
    code = """
x_cols = ["sepsis3"]
raw = "map_min"
model_name = "map_min_per_10mmhg"
miss_name = f"{raw}_missing_indicator"
x_cols.extend([model_name, miss_name, "age_per_10y"])
x_cols = list(dict.fromkeys(x_cols))
"""

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
    )

    assert repaired is not None
    name, patched = repaired
    assert name == "drop_overadjustment_covariates_v1"
    assert "_easyicu_overadjustment_drop_v1" in patched
    namespace = {}
    exec(patched, namespace)
    assert namespace["x_cols"] == ["sepsis3", "age_per_10y"]


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
    assert namespace["step_summary"]["measurement_provenance_audit"] is (
        namespace["diagnostics"]["measurement_provenance_audit"]
    )


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


def test_prediction_split_repair_uses_outcome_col_at_runtime(
    tmp_path,
    monkeypatch,
):
    repaired = _deterministic_runner_repair(
        code=(
            "figure_contract = FigureContract()\n"
            "train_test_split(X, y, test_size=0.2, test_size=0.3)\n"
        ),
        run_log="SyntaxError: keyword argument repeated",
    )
    assert repaired is not None
    _, patched = repaired

    cohort = tmp_path / "cohort.parquet"
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        {
            "death": [1] * 10,
            "endpoint_x": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "age": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        }
    ).to_parquet(cohort, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("COHORT_PARQUET", str(cohort))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("OUTCOME_COL", "endpoint_x")

    exec(patched, {})

    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["event_rate_total"] == 0.6


def test_prediction_split_repair_rejects_non_binary_outcome(
    tmp_path,
    monkeypatch,
):
    repaired = _deterministic_runner_repair(
        code=(
            "figure_contract = FigureContract()\n"
            "train_test_split(X, y, test_size=0.2, test_size=0.3)\n"
        ),
        run_log="SyntaxError: keyword argument repeated",
    )
    assert repaired is not None
    _, patched = repaired

    cohort = tmp_path / "cohort.parquet"
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        {
            "los_icu": [1.2, 2.0, 3.5, 4.0, 5.25, 6.0],
            "age": [50, 55, 60, 65, 70, 75],
        }
    ).to_parquet(cohort, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("COHORT_PARQUET", str(cohort))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("OUTCOME_COL", "los_icu")

    with pytest.raises(RuntimeError, match="binary 0/1 OUTCOME_COL"):
        exec(patched, {})
    assert not (out_dir / "step_summary.json").exists()


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


def test_prediction_discrimination_template_is_case_neutral():
    repaired = _deterministic_runner_repair(
        code="model_bundle = ...\n",
        run_log="SyntaxError: invalid syntax near placeholder ellipsis",
    )

    assert repaired is not None
    repair_id, generated = repaired
    assert repair_id == "prediction_discrimination_template_v1"
    ast.parse(generated)
    assert "OUTCOME_COL" in generated
    assert 'model_bundle.get("outcome_col")' in generated
    assert 'df["death"]' not in generated
    assert "death_icu" not in generated
    assert "death_hosp" not in generated
    assert "mortality" not in generated
    assert "sofa2" not in generated.lower()


def test_table_one_repair_uses_explicit_outcome_only():
    repaired = _deterministic_runner_repair(
        code="pd.DataFrame().to_csv('table_one.csv')\n",
        run_log="SyntaxError: '(' was never closed",
    )

    assert repaired is not None
    repair_id, generated = repaired
    assert repair_id == "table_one_descriptive_repair_v1"
    ast.parse(generated)
    assert "OUTCOME_COL" in generated
    assert 'df["death"]' not in generated
    assert "death_icu" not in generated
    assert "death_hosp" not in generated
    assert "mortality" not in generated
    assert "outcome_rate" in generated


def test_table_one_repair_does_not_report_continuous_outcome_rate(
    tmp_path,
    monkeypatch,
):
    repaired = _deterministic_runner_repair(
        code="pd.DataFrame().to_csv('table_one.csv')\n",
        run_log="SyntaxError: '(' was never closed",
    )
    assert repaired is not None
    _, generated = repaired

    pd = pytest.importorskip("pandas")
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "los_icu": [1.0, 2.5, 3.0, 4.25],
            "age": [50, 60, 70, 80],
        }
    ).to_parquet(cohort, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("COHORT_PARQUET", str(cohort))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("OUTCOME_COL", "los_icu")

    exec(generated, {})

    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["outcome_col"] == "los_icu"
    assert summary["outcome_kind"] == "non_binary"
    assert "outcome_rate" not in summary
    assert "outcome_n" not in summary


def test_outcome_incidence_repair_uses_explicit_outcome_only():
    repaired = _deterministic_runner_repair(
        code="# outcome_incidence\n...\n",
        run_log="SyntaxError: invalid syntax",
    )

    assert repaired is not None
    repair_id, generated = repaired
    assert repair_id == "outcome_incidence_descriptive_repair_v1"
    ast.parse(generated)
    assert "OUTCOME_COL" in generated
    assert "OUTCOME_COL is required" in generated
    assert 'df["death"]' not in generated
    assert "death_icu" not in generated
    assert "death_hosp" not in generated
    assert "mortality" not in generated
    assert "_measured" not in generated
    assert "outcome_rate" in generated


def test_outcome_incidence_repair_rejects_non_binary_outcome(
    tmp_path,
    monkeypatch,
):
    repaired = _deterministic_runner_repair(
        code="# outcome_incidence\n...\n",
        run_log="SyntaxError: invalid syntax",
    )
    assert repaired is not None
    _, generated = repaired

    pd = pytest.importorskip("pandas")
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "los_icu": [1.0, 2.5, 3.0, 4.25],
            "age": [50, 60, 70, 80],
        }
    ).to_parquet(cohort, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("COHORT_PARQUET", str(cohort))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("OUTCOME_COL", "los_icu")

    with pytest.raises(RuntimeError, match="binary 0/1 OUTCOME_COL"):
        exec(generated, {})


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


# ---------------------------------------------------------------------------
# seaborn matplotlib fallback (baseline-library sandbox)
# ---------------------------------------------------------------------------


def _inject_seaborn_fallback_namespace():
    """Fire the seaborn fallback repair and exec the injected shim.

    Returns the ``sns`` object the generated analysis code would use inside the
    baseline-library sandbox where ``import seaborn`` raises ModuleNotFoundError.
    """
    import matplotlib

    matplotlib.use("Agg")
    res = _deterministic_runner_repair(
        code="import seaborn as sns\n",
        run_log="ModuleNotFoundError: No module named 'seaborn'",
        previous_repair=None,
    )
    assert res is not None, "seaborn fallback repair did not fire"
    name, repaired = res
    assert name == "seaborn_matplotlib_fallback_v1"
    namespace: dict = {}
    exec(repaired, namespace)  # noqa: S102 - trusted deterministic shim under test
    return namespace["sns"]


def test_seaborn_fallback_supports_despine():
    # Regression: the E3 KDIGO figure step crashed with
    # "'_EasyICUSeabornFallback' object has no attribute 'despine'", which
    # fail-closed the whole run. despine must be a safe no-op on the shim.
    import matplotlib.pyplot as plt

    sns = _inject_seaborn_fallback_namespace()
    _fig, ax = plt.subplots()
    sns.set_style("whitegrid")
    sns.despine(ax=ax)  # the exact crashing call
    sns.despine()  # bare form used by many templates
    plt.close(_fig)


def test_seaborn_fallback_unknown_method_is_noop_not_crash():
    # Durability: any seaborn method the shim does not implement must degrade to
    # a no-op returning the passed ``ax`` rather than raising AttributeError and
    # crashing the figure render (and therefore the entire run).
    import matplotlib.pyplot as plt

    sns = _inject_seaborn_fallback_namespace()
    _fig, ax = plt.subplots()
    assert sns.displot(x=[1, 2, 3]) is None  # unknown, no ax kwarg -> None
    assert sns.catplot(ax=ax) is ax  # unknown, ax passed through
    sns.set_context("paper")
    sns.set_palette("deep")
    sns.move_legend(ax, "upper right")
    plt.close(_fig)


def test_seaborn_fallback_common_statistical_plots_draw_without_error():
    import matplotlib.pyplot as plt
    import pandas as pd

    sns = _inject_seaborn_fallback_namespace()
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, 2.0, 3.0, 4.0]})
    _fig, ax = plt.subplots()
    for call in (
        lambda: sns.boxplot(data=df, x="g", y="v", ax=ax),
        lambda: sns.violinplot(data=df, x="g", y="v", ax=ax),
        lambda: sns.pointplot(data=df, x="g", y="v", ax=ax),
        lambda: sns.countplot(data=df, x="g", ax=ax),
        lambda: sns.kdeplot(data=df, x="v", ax=ax),
        lambda: sns.stripplot(data=df, x="g", y="v", ax=ax),
    ):
        assert call() is ax
    plt.close(_fig)


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

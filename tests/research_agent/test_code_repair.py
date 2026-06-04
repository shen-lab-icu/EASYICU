"""Smoke tests for ``easyicu.research_agent.code_repair``.

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

from easyicu.research_agent.code_repair import (
    _KEYERROR_NOT_IN_INDEX_RE,
    _NAME_ERROR_HELPER_RE,
    _deterministic_summary_repair,
    _deterministic_runner_repair,
    _extract_missing_index_columns,
    _patch_json_dump_numpy_key_sanitizer,
    _strip_columns_from_list_literals,
)


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
        log = 'KeyError: "[\'sofa_total\'] not in index"'
        assert _extract_missing_index_columns(log) == ["sofa_total"]

    def test_extracts_multiple_columns_preserving_order(self):
        log = 'KeyError: "[\'a\', \'b\', \'c\'] not in index"'
        assert _extract_missing_index_columns(log) == ["a", "b", "c"]

    def test_deduplicates_columns(self):
        log = 'KeyError: "[\'a\', \'b\', \'a\'] not in index"'
        assert _extract_missing_index_columns(log) == ["a", "b"]

    def test_tolerates_double_quoted_entries(self):
        # The matcher is documented to accept both single and double quotes.
        log = 'KeyError: "[\"col1\", \"col2\"] not in index"'
        assert _extract_missing_index_columns(log) == ["col1", "col2"]


def test_keyerror_regex_is_compiled_pattern():
    """Pin the symbol pipeline imports as a regex, not a raw string."""
    assert isinstance(_KEYERROR_NOT_IN_INDEX_RE, re.Pattern)
    match = _KEYERROR_NOT_IN_INDEX_RE.search(
        'KeyError: "[\'x\'] not in index"'
    )
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
    assert _NAME_ERROR_HELPER_RE.search(
        "NameError: name '123abc' is not defined"
    ) is None


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
        assert (
            _strip_columns_from_list_literals(code, ["sofa_total"]) == code
        )

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
        assert (
            _deterministic_runner_repair(code=code, run_log=run_log)
            is None
        )


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

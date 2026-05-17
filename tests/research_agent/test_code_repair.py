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
* ``_generic_clustering_fallback_code``
* ``_infer_generic_v15_fallback_key``

The two heavier symbols (``_deterministic_runner_repair`` and
``_deterministic_summary_repair``) read/write run directories and are
left for a follow-up suite with fixtures.
"""

from __future__ import annotations

import ast
import re

import pytest

from easyicu.research_agent.code_repair import (
    _KEYERROR_NOT_IN_INDEX_RE,
    _NAME_ERROR_HELPER_RE,
    _extract_missing_index_columns,
    _generic_clustering_fallback_code,
    _infer_generic_v15_fallback_key,
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


# ---------------------------------------------------------------------------
# _generic_clustering_fallback_code
# ---------------------------------------------------------------------------


def test_generic_clustering_fallback_returns_executable_python():
    code = _generic_clustering_fallback_code()
    assert isinstance(code, str) and code.strip()
    # If the template ever becomes un-parseable Python, the runner will
    # silently emit a broken fallback step. Pin parseability here.
    ast.parse(code)


def test_generic_clustering_fallback_reads_runner_contract_env_vars():
    """The fallback template depends on the runner's env-var contract.

    ``STEP_OUT_DIR`` and ``COHORT_PARQUET`` are set by the runner; if
    either is renamed without updating this template, the fallback
    silently crashes when invoked. Lock the contract in a test.
    """
    code = _generic_clustering_fallback_code()
    assert 'os.environ["STEP_OUT_DIR"]' in code
    assert 'os.environ["COHORT_PARQUET"]' in code


# ---------------------------------------------------------------------------
# _infer_generic_v15_fallback_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "haystack, expected",
    [
        ("cluster_labels = ...", "clustering"),
        ("trajectory_clustering pipeline", "clustering"),
        ("聚类分析输出", "clustering"),
        ("t14_creatinine_trajectory_kdigo step", "creatinine"),
        ("t05_kdigo_renal_sensitivity audit", "kdigo"),
        ("t04_lactate_mortality_association run", "lactate"),
        ("t03_severity_score_correlation summary", "severity_correlation"),
        ("t13_admission_vital_summary export", "vitals"),
        ("t01_table_one_descriptive draft", "table_one"),
        ("lactate_max_24h scatter", "lactate"),
        ("plain code with no markers", None),
    ],
)
def test_infer_generic_v15_fallback_key_dispatches_by_marker(
    haystack: str, expected
):
    assert _infer_generic_v15_fallback_key(haystack) == expected


def test_infer_generic_v15_fallback_key_skips_norepi_task():
    # Documented: norepi tasks must NOT route to the generic fallback,
    # because they have task-specific logic the generic version would skip.
    assert (
        _infer_generic_v15_fallback_key(
            "norepi_equiv_max_24h column present", ""
        )
        is None
    )
    assert (
        _infer_generic_v15_fallback_key(
            "step t15_norepinephrine_dose_response", ""
        )
        is None
    )


def test_infer_generic_v15_fallback_key_skips_prediction_tasks():
    # When ≥2 prediction-style markers appear, the dispatcher bows out
    # so prediction scripts don't get rerouted as association/cluster runs.
    code = "stratifiedkfold cv with roc_auc_score and calibration"
    assert _infer_generic_v15_fallback_key(code, "") is None


def test_infer_generic_v15_fallback_key_uses_diagnostic_text_too():
    # The function combines code + diagnostic_text; either source can
    # supply the marker that triggers a dispatch decision.
    assert (
        _infer_generic_v15_fallback_key(
            code="", diagnostic_text="cluster_mortality outputs missing"
        )
        == "clustering"
    )

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

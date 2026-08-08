"""A declared level must name one source category, not one spelling.

``level_spelling`` is the wire identity that binds a declared level to observed
data, and it is deliberately lossy: ``1`` and ``1.0`` are the same value and
must code the same. But it also collapsed ``1`` with ``"1"`` and ``True`` with
``"true"``, and the closed-domain check compares spellings to spellings, so a
column holding both saw no undeclared level, no absent level, and no warning --
two source categories became one contrast column.

This is not a synthetic worry for this codebase: ``io.data_converter`` pins
``MIXED_TYPE_COLUMNS`` to string precisely because real EHR exports carry
object-dtype columns holding both, after pandas' chunked inference corrupted
them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.contracts.model_terms import (
    ModelTermSpec,
    level_identity_class,
    level_spelling,
)
from easyicu.research_agent.execution.model_matrix import (
    ModelTermCompilationError,
    compile_model_terms,
)


def _binary(levels: list[str], reference: str) -> list[ModelTermSpec]:
    return [
        ModelTermSpec(
            name="grp",
            role="exposure",
            coding="binary",
            levels=levels,
            reference_level=reference,
            transform="treatment_contrast",
        )
    ]


def _compile(column: pd.Series, terms: list[ModelTermSpec]):
    frame = pd.DataFrame({"grp": column, "y": [1.0, 2.0, 3.0, 4.0]})
    return compile_model_terms(frame, terms=terms, exposure="grp")


def test_int_and_text_sharing_a_spelling_fail_closed() -> None:
    with pytest.raises(ModelTermCompilationError) as excinfo:
        _compile(pd.Series([1, "1", 0, 0], dtype=object), _binary(["0", "1"], "0"))
    assert excinfo.value.reason_code == "model_term_level_identity_ambiguous"
    assert "numeric" in str(excinfo.value) and "text" in str(excinfo.value)


def test_bool_and_text_sharing_a_spelling_fail_closed() -> None:
    with pytest.raises(ModelTermCompilationError) as excinfo:
        _compile(
            pd.Series([True, "true", False, False], dtype=object),
            _binary(["false", "true"], "false"),
        )
    assert excinfo.value.reason_code == "model_term_level_identity_ambiguous"


@pytest.mark.parametrize(
    ("column", "levels", "reference"),
    [
        (pd.Series([1, 1, 0, 0]), ["0", "1"], "0"),
        (pd.Series(["1", "1", "0", "0"]), ["0", "1"], "0"),
        # int 1 and float 1.0 ARE the same value; collapsing them is correct
        # and must keep compiling.
        (pd.Series([1, 1.0, 0, 0], dtype=object), ["0", "1"], "0"),
        (pd.Series([True, True, False, False]), ["false", "true"], "false"),
    ],
)
def test_an_unambiguous_column_still_compiles(column, levels, reference) -> None:
    compiled = _compile(column, _binary(levels, reference))
    assert compiled.exposure_columns == (f"grp__is_{levels[1]}",)


def test_missing_values_do_not_create_an_ambiguous_class() -> None:
    compiled = _compile(
        pd.Series([1, None, 0, 0], dtype=object), _binary(["0", "1"], "0")
    )
    assert compiled.design["grp__is_1"].isna().sum() == 1


def test_identity_class_separates_exactly_what_spelling_merges() -> None:
    merged = [(1, "1"), (True, "true"), (0, "0"), (False, "false")]
    for left, right in merged:
        assert level_spelling(left) == level_spelling(right)
        assert level_identity_class(left) != level_identity_class(right)
    # ... and does not separate what must stay merged.
    assert level_spelling(1) == level_spelling(1.0)
    assert level_identity_class(1) == level_identity_class(1.0) == "numeric"

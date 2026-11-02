"""Diagnosed AST repairs respect UTF-8 coordinates and refusal boundaries."""

import pytest

from easyicu.research_agent.repairs.ast_rewrites import (
    patch_figure_output_registration,
    patch_literal_mapping_access,
    patch_pandas_numeric_container,
)


@pytest.mark.parametrize(
    "repair,code,diagnosis,expected",
    [
        (
            patch_literal_mapping_access,
            "备注 = '中文'; value = data[0]['x']\n",
            {"coordinates": [{"line": 1, "name": "data", "field": "x"}]},
            "备注 = '中文'; value = data['x']\n",
        ),
        (
            patch_pandas_numeric_container,
            "备注 = '中文'; value = pd.to_numeric(data)\n",
            {"finding_lines": [1]},
            "备注 = '中文'; value = pd.Series(pd.to_numeric(data))\n",
        ),
        (
            patch_figure_output_registration,
            "备注 = '中文'; summary = {'output_files': {'figure:x': stem}, 'figure_files': figures}\n",
            {"coordinates": [{"line": 1, "name": "figures", "field": "figure:x"}]},
            "备注 = '中文'; summary = {'output_files': {'figure:x': figures[0]}, 'figure_files': figures}\n",
        ),
    ],
)
def test_diagnosis_binds_repair_after_multibyte_text(repair, code, diagnosis, expected):
    assert repair(code, **diagnosis) == expected
    missing = {key: [] for key in diagnosis}
    assert repair(code, **missing) == code
    invalid = code + "bad syntax !\n"
    assert repair(invalid, **diagnosis) == invalid

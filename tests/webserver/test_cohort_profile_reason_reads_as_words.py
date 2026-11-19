"""The cohort profile names a missing module in words, in both languages.

The English page used to show the backend reason code
``module_not_in_current_export`` under Vasopressors, Mechanical ventilation
and the other treatment rows of an export without those modules.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIEW = ROOT / "src/easyicu/webserver/static/js/screens-viz-cohort-view.js"


def test_the_code_has_an_english_and_a_chinese_sentence():
    source = VIEW.read_text(encoding="utf-8")
    assert "'module_not_in_current_export': '当前数据没有这个模块'" in source
    assert "raw === 'module_not_in_current_export' ? t('This module is not in the current export', map[raw])" in source

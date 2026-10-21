"""The data dictionary names a concept group by its words.

Catalog group names start with a picker emoji ("⭐ SOFA-2 Scores"); the
extraction picker keeps it, while the dictionary's chips and group column
drop it, as webserver/review_labels.plain_label does for tables and charts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.support.node import run_node

DICT_JS = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/static/js/screens-dict.js"


def test_chips_and_the_group_column_use_the_plain_name() -> None:
    source = DICT_JS.read_text(encoding="utf-8")
    assert "chip(g[0], groupNameOf(g[0])," in source
    assert "return g ? plainName(window.EU_LANG === 'zh' ? g[2] : g[1]) : key;" in source
    # The Chinese lead uses Chinese punctuation.
    assert "' 个概念，分属 '" in source and "个概念,分属" not in source


@pytest.mark.requires_node
def test_plain_name_drops_only_the_leading_symbols() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    line = next(row for row in DICT_JS.read_text(encoding="utf-8").splitlines() if "function plainName(name)" in row)
    script = line + "\nprocess.stdout.write(JSON.stringify(JSON.parse(process.argv[1]).map(plainName)));"
    names = ["⭐ SOFA-2 Scores", "🫀 循环系统", "❤️ 生命体征", "🌡️ Vital signs", "Sepsis-3 (SOFA-2)", "", "🦠"]
    result = run_node(node, script, json.dumps(names), check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == [
        "SOFA-2 Scores", "循环系统", "生命体征", "Vital signs", "Sepsis-3 (SOFA-2)", "", "🦠",
    ]

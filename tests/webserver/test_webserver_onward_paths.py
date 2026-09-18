"""Onward-navigation contracts for native Web screens."""

from pathlib import Path


STATIC = Path(__file__).resolve().parents[2] / "src" / "easyicu" / "webserver" / "static"


def _static_js(name: str) -> str:
    return (STATIC / "js" / name).read_text(encoding="utf-8")


def test_dead_end_screens_gained_onward_paths() -> None:
    """Every terminal-looking screen offers an explicit next step."""
    viz_js = _static_js("screens-viz.js")
    crossdb_results_js = _static_js("screens-viz-crossdb-results.js")
    dict_js = _static_js("screens-dict.js")
    assert "返回队列统计" in crossdb_results_js
    assert 'class="src-fold"' in viz_js
    assert "个较早注册的导出" in viz_js
    assert "到「数据抽取」勾选它们所属的模块" in dict_js

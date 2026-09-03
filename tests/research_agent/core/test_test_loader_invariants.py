from __future__ import annotations

import sys
from importlib import util
from pathlib import Path


def test_ra_fixture_registers_normal_parent_child_package_binding(
    ra,
    monkeypatch,
) -> None:
    import easyicu

    monkeypatch.delattr(easyicu, "research_agent")
    conftest_path = Path(__file__).resolve().parents[1] / "conftest.py"
    spec = util.spec_from_file_location(
        "_easyicu_research_agent_conftest_under_test",
        conftest_path,
    )
    assert spec is not None and spec.loader is not None
    conftest_module = util.module_from_spec(spec)
    spec.loader.exec_module(conftest_module)

    assert conftest_module._load_research_agent() is ra
    assert sys.modules["easyicu.research_agent"] is ra
    assert easyicu.research_agent is ra

"""The public-API sweep records every failing concept and never hides one."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import easyicu

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "public_api_concept_sweep.py"


@pytest.fixture
def sweep():
    spec = importlib.util.spec_from_file_location("public_api_concept_sweep", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failures_are_recorded_by_class_and_set_the_exit_code(sweep, tmp_path, monkeypatch):
    calls = []

    def fake_load(names, **kwargs):
        calls.append((tuple(names), kwargs["database"], kwargs["merge"]))
        if names == ["bad"]:
            raise ValueError("first line\nsecond line")
        import pandas as pd

        return pd.DataFrame({"x": [1, 2]})

    monkeypatch.setattr(easyicu, "load_concepts", fake_load)
    out = tmp_path / "sweep.json"
    code = sweep.main([
        "--database", "eicu", "--data-path", str(tmp_path), "--out", str(out),
        "--concepts", "good,bad",
    ])
    report = json.loads(out.read_text())

    assert code == 1
    assert calls == [(("good",), "eicu", True), (("bad",), "eicu", True)]
    assert report["failed"] == 1
    assert report["failures_by_error"] == {"ValueError": ["bad"]}
    assert report["results"]["good"]["rows"] == 2
    assert report["results"]["bad"]["message"] == "first line"


def test_a_clean_sweep_exits_zero(sweep, tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(easyicu, "load_concepts", lambda names, **kw: pd.DataFrame())
    out = tmp_path / "sweep.json"
    assert sweep.main([
        "--database", "miiv", "--data-path", str(tmp_path), "--out", str(out), "--concepts", "hr",
    ]) == 0
    assert json.loads(out.read_text())["failed"] == 0


def test_the_default_list_covers_sourced_and_derived_concepts(sweep):
    names = sweep._concepts("eicu", win_tbl_only=False)
    assert "ett_gcs" in names and "hr" in names
    assert "sofa" in names  # derived from sub-concepts, no eICU source of its own
    assert set(sweep._concepts("eicu", win_tbl_only=True)) <= set(names)

"""Ownership contracts for the bounded native Cross-DB raw run."""

from pathlib import Path


ROOT = Path(__file__).parents[1]
STATIC = ROOT / "src" / "easyicu" / "webserver" / "static"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_crossdb_raw_scope_owner_is_wired_before_shared_viz() -> None:
    index = _read("index.html")
    owner = _read("js/screens-viz-crossdb-raw.js")
    setup = _read("js/screens-viz-crossdb-setup.js")
    viz = _read("js/screens-viz.js")

    owner_src = "js/screens-viz-crossdb-raw.js?v=20260710-core-scope"
    assert owner_src in index
    assert index.index(owner_src) < index.index("js/screens-viz-crossdb-setup.js?")
    assert index.index("js/screens-viz-crossdb-setup.js?") < index.index("js/screens-viz.js?")
    assert "window.EU_CROSSDB_RAW = { coreFeatures, buildRequest }" in owner
    assert "window.EU_CROSSDB_RAW.coreFeatures()" in viz
    assert "window.EU_CROSSDB_RAW.buildRequest" in viz
    assert "12 curated core concepts" in setup


def test_crossdb_raw_scope_is_curated_and_route_pure() -> None:
    owner = _read("js/screens-viz-crossdb-raw.js")
    for concept in (
        "'hr'", "'map'", "'sbp'", "'dbp'", "'resp'", "'temp'",
        "'spo2'", "'crea'", "'lact'", "'wbc'", "'plt'", "'glu'",
    ):
        assert concept in owner
    assert "'gluc'" not in owner
    assert "feature_scope: 'curated_core'" in owner
    assert "features: coreFeatures()" in owner
    for foreign_marker in (
        "data-patient-",
        "data-cohort-",
        "data-ag-",
        "patient-review",
        "guided",
        "ideas",
    ):
        assert foreign_marker not in owner

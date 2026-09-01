"""Ownership contracts for the bounded native Cross-DB raw run."""

from pathlib import Path
import json
import shutil
import subprocess

import pytest

from easyicu.webserver import crossdb_review


ROOT = Path(__file__).parents[1]
STATIC = ROOT / "src" / "easyicu" / "webserver" / "static"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_crossdb_raw_scope_owner_is_wired_before_shared_viz() -> None:
    index = _read("index.html")
    owner = _read("js/screens-viz-crossdb-raw.js")
    setup = _read("js/screens-viz-crossdb-setup.js")
    viz = _read("js/screens-viz.js")

    owner_src = "js/screens-viz-crossdb-raw.js?v=20260728-feature-scope1"
    assert owner_src in index
    assert index.index(owner_src) < index.index("js/screens-viz-crossdb-setup.js?")
    assert index.index("js/screens-viz-crossdb-setup.js?") < index.index("js/screens-viz.js?")
    assert "window.EU_CROSSDB_RAW = {" in owner
    assert "apiFeatureScope" in owner
    assert "normalizeFeatureScope" in owner
    assert "window.EU_CROSSDB_RAW.coreFeatures()" in viz
    assert "window.EU_CROSSDB_RAW.buildRequest" in viz
    assert "Complete catalog" in setup
    assert "Quick core" in setup


def test_crossdb_raw_scope_is_curated_and_route_pure() -> None:
    owner = _read("js/screens-viz-crossdb-raw.js")
    for concept in (
        "'hr'", "'map'", "'sbp'", "'dbp'", "'resp'", "'temp'",
        "'spo2'", "'crea'", "'lact'", "'wbc'", "'plt'", "'glu'",
    ):
        assert concept in owner
    assert "'gluc'" not in owner
    assert "core: 'curated_core'" in owner
    assert "all: 'all_catalog'" in owner
    assert "if (scope === 'core') request.features = coreFeatures()" in owner
    for foreign_marker in (
        "data-patient-",
        "data-cohort-",
        "data-ag-",
        "patient-review",
        "guided",
        "ideas",
    ):
        assert foreign_marker not in owner


def test_crossdb_raw_scope_executes_quick_and_full_contract() -> None:
    node = shutil.which("node")
    if not node:
        candidates = sorted((Path.home() / ".nvm" / "versions" / "node").glob("*/bin/node"))
        node = str(candidates[-1]) if candidates else None
    if not node:
        pytest.skip("node is required for the Cross-DB raw scope contract")

    owner = STATIC / "js" / "screens-viz-crossdb-raw.js"
    subprocess.run([node, "--check", str(owner)], check=True, capture_output=True, text=True)
    result = subprocess.run(
        [node, str(ROOT / "tests" / "js" / "crossdb_raw_scope.test.js"), str(owner)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "backend_catalog_owned": True,
        "explicit_full_catalog": True,
        "quick_core_preserved": True,
    }


@pytest.mark.parametrize(
    "identifier",
    ["stay_id", "patientunitstayid", "patienthealthsystemstayid", "uniquepid"],
)
def test_crossdb_identifier_columns_are_not_clinical_features(identifier: str) -> None:
    assert crossdb_review._is_feature_column(identifier) is False

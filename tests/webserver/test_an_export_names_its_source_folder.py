"""An export cannot be extracted again; the scan names the folder it came from.

A registered official demo is its module export. Data Extraction used to offer
that export for a new cohort and then failed with "table icustays is not
installed". The scan now names the source folder from the export manifest, and
the extraction page offers it instead of re-extracting the export.
"""

from __future__ import annotations

import json
from pathlib import Path

from easyicu.webserver import dataio
from tests.support.native_demo_export import _native_demo
from tests.support.published_export import record_member_digests

STATIC_JS = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/static/js"


def _with_data_path(root: Path, data_path: str) -> None:
    manifest_path = root / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["data_path"] = data_path
    manifest_path.write_text(json.dumps(manifest))
    record_member_digests(root)


def test_a_module_export_names_its_existing_source_folder(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    root, _ = _native_demo(tmp_path / "export")
    _with_data_path(root, str(source))

    result = dataio.scan_path(str(root))

    assert result["ok"] is True
    assert result["source"] == "module"
    assert result["source_data_path"] == str(source.resolve())


def test_a_missing_or_self_referencing_source_folder_is_not_offered(tmp_path):
    root, _ = _native_demo(tmp_path / "export")
    _with_data_path(root, str(tmp_path / "gone"))
    assert "source_data_path" not in dataio.scan_path(str(root))

    _with_data_path(root, str(root))
    assert "source_data_path" not in dataio.scan_path(str(root))


def test_extraction_offers_the_source_folder_and_refuses_to_re_extract_an_export():
    owner = (STATIC_JS / "screens-extraction-export-source.js").read_text(encoding="utf-8")
    route = (STATIC_JS / "screens-extraction.js").read_text(encoding="utf-8")
    index = (STATIC_JS.parent / "index.html").read_text(encoding="utf-8")
    assert "data-ex-source-data" in owner
    assert "Define a new cohort from its source data" in owner and "从它的源数据定义新队列" in owner
    assert "reason: 'export_not_source'" in owner
    assert "exSource === 'module' && window.EU_EXTRACTION_EXPORT_SOURCE.refusal()" in route
    assert "window.EU_EXTRACTION_EXPORT_SOURCE.sourceCta(d, hostedServer())" in route
    assert index.index("js/screens-extraction-export-source.js") < index.index("js/screens-extraction.js")

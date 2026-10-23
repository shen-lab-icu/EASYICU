"""Scan failures use the shared bilingual error-text owner, including unknown codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.node import run_node

JS = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/static/js"


@pytest.mark.requires_node
@pytest.mark.parametrize("code", [
    "no_path", "not_a_directory", "permission_denied", "unrecognized_folder", "scan_api_unavailable",
    "database_mismatch",
    "database_detection_ambiguous", "database_detection_manifest_invalid", "database_detection_schema_unreadable",
    "database_detection_unavailable", "export_package_invalid", "export_layout_invalid", "export_manifest_missing",
    "export_manifest_json_invalid", "manifest_marker_invalid", "manifest_marker_missing", "manifest_path_escape",
    "manifest_file_missing", "manifest_file_symlink", "manifest_inventory_missing", "manifest_inventory_mismatch",
    "manifest_duplicate_file", "manifest_format_invalid", "manifest_format_mismatch", "manifest_schema_invalid",
    "manifest_row_count_invalid", "manifest_row_count_mismatch", "file_schema_invalid", "file_identity_column_invalid",
    "column_metadata_inventory_invalid", "column_metadata_digest_mismatch", "column_metadata_schema_invalid",
    "scan_failed", "future_scan_failure",
])
def test_scan_failures_render_as_one_sentence_in_both_languages(code):
    script = f"""
      global.window = global;
      const modules = {{}};
      window.EU_HTML = {{esc: value => String(value)}};
      window.EasyICU = {{guidedPi: {{declare: (key, owner) => modules[key] = owner,
        optional: key => modules[key] || null}}}};
      require({str(JS / 'screens-guided-pi-error-text.js')!r});
      const fs = require('fs');
      const source = fs.readFileSync({str(JS / 'screens-extraction.js')!r}, 'utf8');
      const start = source.indexOf('  function scanErrorState()');
      const end = source.indexOf('  function scanResultState()', start);
      const exScanError = {json.dumps(code)}, exPath = '/synthetic/folder', exDatabaseMismatch = null;
      const escHtml = window.EU_HTML.esc, pathDisplay = x => x, icon = () => '';
      let language = 'en';
      const t = (en, zh) => language === 'zh' ? zh : en;
      eval(source.slice(start, end));
      const result = {{}};
      for (language of ['en', 'zh']) {{
        result[language] = {{sentence: modules.errorText.scanErrorText(exScanError, t), html: scanErrorState()}};
      }}
      console.log(JSON.stringify(result));
    """
    result = json.loads(run_node("node", script, check=True).stdout)
    for language, stop in [("en", "."), ("zh", "。")]:
        sentence = result[language]["sentence"]
        assert sentence.endswith(stop)
        assert sentence.count(stop) == 1
        assert code not in sentence
        assert code not in result[language]["html"]
        assert sentence in result[language]["html"]
    assert result["en"]["sentence"] != result["zh"]["sentence"]


@pytest.mark.requires_node
@pytest.mark.parametrize("language", ["en", "zh"])
def test_the_guided_extraction_card_uses_the_same_scan_failure_sentence(language):
    script = f"""
      global.window = global;
      const modules = {{}};
      window.EU_HTML = {{esc: value => String(value)}};
      window.EasyICU = {{guidedPi: {{declare: (key, owner) => modules[key] = owner,
        optional: key => modules[key] || null}}}};
      require({str(JS / 'screens-guided-pi-error-text.js')!r});
      require({str(JS / 'screens-guided-extract.js')!r});
      const t = (en, zh) => {json.dumps(language)} === 'zh' ? zh : en;
      const owner = window.EU_GUIDED_EXTRACT;
      owner.init({{t}}); owner.resetState();
      owner.state().path = '/synthetic/demo';
      window.EU_API = {{scanPath: async () => ({{ok: false, error: 'database_detection_ambiguous'}})}};
      owner.handleClick({{closest: key => key === '[data-gx-analyze]' ? {{}} : null}});
      setTimeout(() => console.log(JSON.stringify({{sentence: owner.state().scanError,
        expected: modules.errorText.scanErrorText('database_detection_ambiguous', t)}})), 5);
    """
    result = json.loads(run_node("node", script, check=True).stdout)
    assert result["sentence"] == result["expected"]
    assert "database_detection_ambiguous" not in result["sentence"]

"""The local compiler finding has a stable entry for the copy owner."""

import json
from pathlib import Path
import shutil

import pytest

from tests.support.node import run_node


def test_bound_roster_finding_uses_its_registered_reader_entry():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    source = Path(__file__).resolve().parents[3] / "src/easyicu/webserver/static/js/screens-guided-pi-error-text.js"
    result = run_node(node, """
let errorText;
global.window = {
  EU_LANG: 'en', EU_HTML: {esc: value => String(value)},
  EasyICU: {guidedPi: {declare: (name, api) => {if (name === 'errorText') errorText = api;}}},
};
require(process.argv[1]);
const owner = errorText.create({tr: en => en, staticPreview: () => false});
const read = code => owner.runFailureText('research_pipeline_progressive_compile_failed', {code});
process.stdout.write(JSON.stringify([
  read('progressive_adjustment_roster_mismatch'), read('unknown_synthetic_compiler_finding'),
]));
""", str(source), check=True)
    registered, fallback = json.loads(result.stdout)
    assert registered and registered != fallback
    assert "progressive_adjustment_roster_mismatch" not in registered

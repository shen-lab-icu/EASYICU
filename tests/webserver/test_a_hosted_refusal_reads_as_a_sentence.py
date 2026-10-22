"""A hosted boundary refusal reads as a sentence in the clinician's language.

webserver/hosted.py refuses a request with a code and an English reason:
the administrator's data is read-only (hosted_data_read_only), a location is
outside the prepared data (path_outside_data_roots), or the action belongs to
the administrator (unavailable_when_hosted). The conversation's error owner
names each in the page language instead of showing the code or the English
reason.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.support.node import run_node

STATIC = Path(__file__).resolve().parents[2] / "src" / "easyicu" / "webserver" / "static"

_ERROR_TEXT = r"""
let owner = null;
global.window = {
  EU_HTML: { esc: value => String(value) },
  EasyICU: { guidedPi: { declare: (name, api) => { if (name === 'errorText') owner = api; }, optional: () => null } },
};
require(process.argv[1]);
const text = owner.create({ tr: (en, zh) => zh, staticPreview: () => false }).errorText;
const [code, reason] = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(text(Object.assign(new Error(reason), { status: 403, code }))));
"""


@pytest.mark.requires_node
@pytest.mark.parametrize(
    ("code", "reason", "sentence"),
    [
        ("hosted_data_read_only", "Prepared data is read-only; results are saved in your own workspace.",
         "管理员准备的数据是只读的；结果会保存在你自己的工作区。"),
        ("path_outside_data_roots", "This path is outside the instance's permitted directories.",
         "这个位置不在管理员准备的数据范围内。"),
        ("unavailable_when_hosted", "This action is managed by the instance administrator.",
         "这项操作由管理员负责，请联系管理员。"),
        # The Web memory admission refused a message (pi_copilot/resource_lifecycle.py).
        ("pi_web_memory_pressure", "EasyICU is under memory pressure; retry after current work finishes.",
         "EasyICU 正忙于处理其他任务，这条消息没有发送。请稍等一分钟再发送。"),
    ],
)
def test_a_hosted_refusal_reads_in_the_page_language(code: str, reason: str, sentence: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    owner = STATIC / "js" / "screens-guided-pi-error-text.js"
    result = run_node(node, _ERROR_TEXT, str(owner), json.dumps([code, reason]), check=False)

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == sentence

"""A failure no copy names still reads as a sentence, not a code or a transport string.

A clinician on a hosted instance reads Chinese and has no use for
``/api/... -> HTTP 500`` or an English validation reason.  The conversation's
error owner keeps a server sentence written in the reader's language, turns
everything else into a plain sentence with the code (or original text) in
brackets for the administrator, and names the administrator a hosted
instance configured.  A failed conversation job reads through the same owner.
"""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    STATIC as STATIC,
    _read as _read,
)


_SCRIPT = r"""
let errorText = null;
const lang = process.argv[2];
const hosted = process.argv[3] === 'hosted';
global.window = {
  EU_LANG: lang,
  EU_HTML: { esc: value => String(value) },
  EasyICU: { guidedPi: {
    declare: (name, api) => { if (name === 'errorText') errorText = api; },
    optional: name => (name === 'hosted' ? { active: () => hosted, supportContact: () => '信息科 王老师 8123' } : null),
  } },
};
require(process.argv[1]);
const owner = errorText.create({ tr: (en, zh) => (lang === 'zh' ? zh : en), staticPreview: () => false });
process.stdout.write(JSON.stringify({
  server: owner.errorText(Object.assign(new Error('/api/copilot/pi/sessions -> HTTP 500'), { status: 500, code: null })),
  network: owner.errorText(new TypeError('Failed to fetch')),
  english: owner.errorText(Object.assign(new Error('Initialize the EasyICU project before opening its governed resources.'), {
    status: 409, code: 'pi_project_not_initialized' })),
  chinese: owner.errorText(Object.assign(new Error('文件超过 20 MB，请换一个较小的文件。'), { status: 400, code: 'pdf_too_large' })),
  bare: owner.errorText(Object.assign(new Error('pdf_too_large'), { status: 400, code: 'pdf_too_large' })),
  jobKnown: owner.jobErrorText('pi_session_authority_stale: the binding digest changed'),
  jobRaw: owner.jobErrorText('RuntimeError: worker exited'),
  manifest: owner.runFailureText('research_pipeline_manifest_required'),
}));
"""


def _copy(lang: str, deployment: str = "local") -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = run_node(
        node,
        _SCRIPT,
        str(STATIC / "js" / "screens-guided-pi-error-text.js"),
        lang,
        deployment,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


@pytest.mark.requires_node
def test_a_chinese_reader_gets_a_sentence_with_the_code_for_the_administrator() -> None:
    out = _copy("zh", "hosted")

    assert out["server"] == (
        "EasyICU 服务器处理这一步时出错了（HTTP 500）。请稍后重试；"
        "如果反复出现，请联系管理员（信息科 王老师 8123）。"
    )
    assert out["network"] == "连不上 EasyICU 服务器。请检查网络后重试。"
    assert out["english"] == (
        "这一步没有完成（pi_project_not_initialized）。"
        "如果反复出现，请联系管理员（信息科 王老师 8123）。"
    )
    # A server sentence already in Chinese is the best copy there is.
    assert out["chinese"] == "文件超过 20 MB，请换一个较小的文件。"
    assert out["bare"].startswith("这一步没有完成（pdf_too_large）")
    # A job failure's code picks its own copy; free text is kept in brackets.
    assert out["jobKnown"] == "这段对话保存后研究绑定发生了变化，请先重新绑定再继续。"
    assert out["jobRaw"].startswith("这一步没有完成（RuntimeError: worker exited）")
    assert "数据包" in out["manifest"] and "research_pipeline" not in out["manifest"]


@pytest.mark.requires_node
def test_an_english_reader_keeps_the_server_sentence_and_a_local_build_names_no_contact() -> None:
    out = _copy("en")

    assert out["english"] == (
        "Initialize the EasyICU project before opening its governed resources."
    )
    assert out["server"].endswith("if it keeps happening, contact the administrator.")
    assert "/api/" not in out["server"]
    assert out["jobRaw"] == "RuntimeError: worker exited"


def test_failed_jobs_and_patient_review_read_through_their_owners() -> None:
    screen = _read("js/screens-guided-pi.js")
    patient = _read("js/screens-viz-patient.js")

    assert "ERROR_TEXT.jobErrorText(row.error)" in screen
    assert ": String(row.error || tr('Copilot message failed.'" not in screen
    assert "vizErr = t('Patient Review API is unavailable.'," in patient
    assert "vizErr = t('No registered export is active." in patient

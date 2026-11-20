"""The result card's article button exists only when the append owner offers it (B26).

"Extend to a full article" on a first-result card used to draft a request into
the conversation that granted nothing, so the reply said no article run had
started.  The button now goes through the result-append owner's article
extension, and renders only when that owner offers it for this run.  Fakes only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"

_SCRIPT = r"""
const path = require('node:path');
const dir = process.argv[1];
const scenario = JSON.parse(process.argv[2]);
global.window = { EU_LANG: 'en' };
require(path.join(dir, 'screens-guided-pi-modules.js'));
require(path.join(dir, 'screens-guided-pi-result-append.js'));
const owner = window.EasyICU.guidedPi.require('resultAppend').create({
  tr: en => en, esc: String, projectId: () => scenario.project, host: () => null,
  session: () => ({ session_id: 's' }), canPreview: () => true, onError() {}, errorText: String,
  api: () => ({ loadPiCopilotAppendOptions: async () => scenario.options }),
});
(async () => {
  await owner.load({ run_id: 'run_a' });
  process.stdout.write(JSON.stringify({
    article: owner.offers({ run_id: scenario.asked }, 'article'),
    sensitivity: owner.offers({ run_id: scenario.asked }, 'sensitivity'),
  }));
})();
"""


def _offers(*, available: bool, asked: str = "run_a", project: str = "p1", sha: str = "s" * 64):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    options = {
        "run_id": "run_a",
        "source_run_sha256": sha,
        "options": [{"kind": "article", "available": available}],
    }
    result = run_node(
        node,
        _SCRIPT,
        str((STATIC / "js").resolve()),
        json.dumps({"options": options, "asked": asked, "project": project}),
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def test_the_owner_offers_an_available_article_extension_for_its_run() -> None:
    assert _offers(available=True) == {"article": True, "sensitivity": False}


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"available": False}, id="not-available"),
        pytest.param({"available": True, "asked": "run_b"}, id="another-run"),
        pytest.param({"available": True, "sha": ""}, id="no-source-identity"),
    ],
)
def test_the_owner_offers_nothing_it_cannot_propose(changes) -> None:
    assert _offers(**changes) == {"article": False, "sensitivity": False}


def test_the_card_has_no_drafted_article_request() -> None:
    outcome = (STATIC / "js" / "screens-guided-pi-run-outcome.js").read_text(encoding="utf-8")
    events = (STATIC / "js" / "screens-guided-pi-events.js").read_text(encoding="utf-8")

    assert "data-gpi-article-run" not in outcome + events
    assert "articleRunPrompt" not in outcome + events
    assert "APPEND?.offers?.(run, 'article')" in outcome


def test_a_deferred_deliverable_names_the_designs_that_cannot_extend() -> None:
    # S1731: the row no longer sends the researcher to "a separate article run"
    # that no control starts; it names when the button appears and which
    # designs cannot take it yet.
    outcome = (STATIC / "js" / "screens-guided-pi-run-outcome.js").read_text(encoding="utf-8")

    assert "separate article run" not in outcome
    assert "另行启动论文任务" not in outcome
    assert "landmark and other current-case designs cannot yet be extended" in outcome
    assert "landmark 等设计暂不能在桌面版扩展成论文" in outcome
    assert "“Extend to a full article” appears on this card" in outcome

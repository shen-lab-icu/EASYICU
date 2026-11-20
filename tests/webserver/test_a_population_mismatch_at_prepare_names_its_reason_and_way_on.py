"""A run stopped by a population mismatch says why and offers a way on (S1754).

UX leg 3 on rc_a12: the execution plan stated a different population from the
approved candidate (B33b) and the run stopped with
``caller_bound_population_mode_conflict``; the card said only that the failure
receipt names the reason.  The stop keeps its forward action (修改人群, which
drafts the cohort change that generates a new plan) and now names its reason
in one line.  The bounded repair that carries the reviewed population into
execution planning is 0.1.4 P1.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from easyicu.webserver.routes import agent as agent_routes
from easyicu.webserver.stop_next_actions import next_actions_for_stop
from tests.support.node import run_node

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static" / "js"
_CODE = "caller_bound_population_mode_conflict"


def test_the_stop_offers_a_cohort_change() -> None:
    actions = [action.model_dump(mode="json") for action in next_actions_for_stop(_CODE)]

    assert [(action["kind"], action["params"]["field"]) for action in actions] == [("edit_study", "cohort")]


@pytest.mark.parametrize(("lang", "line"), [
    ("zh", "运行已停止：执行计划中的人群条件与已批准的计划不一致。请修改人群，重新生成计划。"),
    ("en", "the population condition in the execution plan did not match the approved plan"),
])
def test_the_card_names_why_the_run_stopped(lang: str, line: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = r"""
const lang = process.argv[2];
global.window = { EU_LANG: lang };
const path = require('node:path');
const js = path.dirname(process.argv[1]);
require(path.join(js, 'screens-guided-pi-modules.js'));
require(path.join(js, 'html-escape.js'));
require(process.argv[1]);
const errors = window.EasyICU.guidedPi.require('errorText').create({
  tr: (en, zh) => (lang === 'zh' ? zh : en), staticPreview: false });
process.stdout.write(errors.runFailureText(process.argv[3]));
"""
    result = run_node(node, script, str(STATIC / "screens-guided-pi-error-text.js"), lang, _CODE, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert line in result.stdout
    assert "失败回执里写明了原因" not in result.stdout

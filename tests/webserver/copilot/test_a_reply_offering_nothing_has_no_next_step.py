"""A reply that offers no choice, asks nothing and states no next step has no next-step block.

pilot1 round 2b (10-22), R4/H8/R1: under 「研究问题已保存。这台服务器上还没有
HiRID v1.1.1 数据。」 and above the host's card, the conversation rendered a
「下一步」 block reading 「你可以继续补充要求，或让 EasyICU 进入下一项受治理步骤。」
with no control: a heading that offered nothing and only repeated the composer.
Such a reply now renders no block; a question, a stated next step and choices
are rendered as before.
"""

from __future__ import annotations

import json
import shutil

import pytest

from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _ESCAPE_OWNER as _ESCAPE_OWNER,
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read as _read,
)

_SAVED = "研究问题已保存。这台服务器上还没有 HiRID v1.1.1 数据。"


def _rendered(text: str) -> dict[str, str]:
    script = f"""
      global.window = global;
      window.EU_LANG = 'zh';
      eval({_ESCAPE_OWNER!r});
      eval({_read("js/screens-guided-pi-next-actions.js")!r});
      const next = window.EasyICU.guidedPi.require('nextActions');
      const step = next.project({json.dumps(text)});
      process.stdout.write(JSON.stringify({{
        body: next.bodyText(step, {{ live: true }}),
        block: next.render(step, {{ language: 'zh' }}),
      }}));
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    return json.loads(run_node(node, script, check=True).stdout)


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_a_statement_renders_no_next_step_block() -> None:
    rendered = _rendered(_SAVED)

    assert rendered == {"body": _SAVED, "block": ""}


@pytest.mark.requires_node
@pytest.mark.usefixtures("_load_guided_pi_module_harness")
def test_a_question_a_stated_step_and_choices_keep_their_block() -> None:
    asked = _rendered("你想比较哪两组患者？")
    assert "请在下方输入框回答这个问题。" in asked["block"]
    assert "data-gpi-next-focus" in asked["block"]

    stated = _rendered(f"{_SAVED}\n\n**下一步：**在下方数据源卡片选择本机的 HiRID 数据目录。")
    assert "在下方数据源卡片选择本机的 HiRID 数据目录。" in stated["block"]
    assert stated["body"] == _SAVED

    offered = _rendered(f"{_SAVED}\n\n**下一步：**\n- 改用 eICU\n- 改用 MIMIC-IV")
    assert 'data-gpi-next-choice="改用 eICU"' in offered["block"]
    assert "受治理步骤" not in offered["block"]

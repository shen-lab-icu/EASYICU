"""Focused fail-closed tests for current-user turn authority."""

from __future__ import annotations

import pytest

from easyicu.webserver.pi_copilot.turn_authority import (
    infer_explicit_turn_actions,
)


@pytest.mark.parametrize(
    "message",
    [
        "确认，授权你在本轮准备并注册官方演示数据。",
        "我确认当前研究配置，请开始数据提取和质量审阅。",
        "I confirm and authorize data extraction for this turn.",
        "I approve preparing and registering the demo data now.",
    ],
)
def test_explicit_extraction_confirmation_grants_one_turn(message: str) -> None:
    assert infer_explicit_turn_actions(message) == frozenset({"extract"})


@pytest.mark.parametrize(
    "message",
    [
        "请最后准备数据提取并给我下载。",
        "我确认当前研究问题。",
        "我不授权执行数据提取。",
        "请不要执行数据准备。",
        "I do not authorize data extraction.",
        "Please extract the data when ready.",
        "",
    ],
)
def test_ambiguous_unrelated_or_negated_text_stays_fail_closed(
    message: str,
) -> None:
    assert infer_explicit_turn_actions(message) == frozenset()

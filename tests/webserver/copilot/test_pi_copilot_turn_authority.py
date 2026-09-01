"""Focused fail-closed tests for current-user turn authority."""

from __future__ import annotations

import pytest

from easyicu.webserver.pi_copilot.turn_authority import (
    explicitly_confirms_easyicu_registered_source,
    infer_explicit_turn_actions,
)


@pytest.mark.parametrize(
    "message",
    [
        "确认，授权你在本轮准备并注册官方演示数据。",
        "我确认当前研究配置，请开始数据提取和质量审阅。",
        "确认授权本轮打开本地数据选择与扫描流程。",
        "授权打开本地 MIMIC-IV v3.1 数据提取工作区。",
        "授权下载并准备官方 MIMIC-IV demo。",
        "为当前操作启用一次性 Extraction 授权并打开本地 MIMIC-IV v3.1 数据提取工作区。",
        "继续打开本机 MIMIC-IV 3.1 数据提取工作区。",
        "授权 EasyICU 解析执行概念并打开数据提取工作区。",
        "I confirm and authorize data extraction for this turn.",
        "Continue opening the local data extraction workspace.",
        "Enable one-time Extraction authorization and open the local MIMIC-IV data workspace.",
        "I approve preparing and registering the demo data now.",
        "Authorize EasyICU to prepare the data extraction workspace.",
    ],
)
def test_explicit_extraction_confirmation_grants_one_turn(message: str) -> None:
    assert infer_explicit_turn_actions(message) == frozenset({"extract"})


@pytest.mark.parametrize(
    "message",
    [
        "按当前配置生成新的 Research Agent 分析计划，并在正式分析前暂停让我审核。",
        "授权生成新分析计划并在分析前暂停。",
        "请重新规划，但不要开始正式分析。",
        "授权重新生成干净分析计划",
        "已启用 provider_run 授权，重新生成干净分析计划",
        "在 EasyICU 中重新授予并传递一次性 provider_run 授权",
        "Please regenerate the Research Agent analysis plan and pause for review.",
    ],
)
def test_explicit_replan_request_grants_one_provider_turn(message: str) -> None:
    assert infer_explicit_turn_actions(message) == frozenset({"provider_run"})


@pytest.mark.parametrize(
    "message",
    [
        "请最后准备数据提取并给我下载。",
        "我确认当前研究问题。",
        "我不授权执行数据提取。",
        "请不要执行数据准备。",
        "I do not authorize data extraction.",
        "Please extract the data when ready.",
        "选择 MIMIC-IV Clinical Database Demo（v2.2）。",
        "使用本地 MIMIC-IV v3.1。",
        "",
    ],
)
def test_ambiguous_unrelated_or_negated_text_stays_fail_closed(
    message: str,
) -> None:
    assert infer_explicit_turn_actions(message) == frozenset()


@pytest.mark.parametrize(
    "message",
    [
        "使用 EasyICU 已准备好的完整 **MIMIC-IV v3.1**。",
        "确认使用 EasyICU 已准备好的完整 **MIMIC-IV v3.1**。",
        "Use the validated MIMIC-IV source already prepared by EasyICU.",
        "Confirm and reuse the registered EasyICU MIMIC-IV export.",
    ],
)
def test_explicit_prepared_source_choice_is_recognized(message: str) -> None:
    assert explicitly_confirms_easyicu_registered_source(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "我想研究 MIMIC-IV 成人 ICU 人群。",
        "使用本机上的 MIMIC-IV v3.1 数据文件夹。",
        "暂不确认 EasyICU 已准备的数据源。",
        "Please help me study MIMIC-IV.",
    ],
)
def test_database_mention_or_local_choice_does_not_confirm_prepared_source(
    message: str,
) -> None:
    assert explicitly_confirms_easyicu_registered_source(message) is False

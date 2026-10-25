"""Moving rail presentation into its owner keeps activity and time labels."""

import json
import shutil
from pathlib import Path

import pytest

from tests.support.node import run_node


@pytest.mark.parametrize("language", ["zh", "en"])
def test_session_labels_keep_status_precedence_and_relative_time(language):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable")
    owner = (
        Path(__file__).parents[3]
        / "src/easyicu/webserver/static/js/screens-guided-pi-study-workspace.js"
    )
    program = r"""
let module;
global.window = {EU_LANG: process.argv[2], EasyICU: {guidedPi: {
  declare: (name, api) => {module = api;},
}}};
require(process.argv[1]);
Date.now = () => Date.parse('2026-10-16T12:00:00Z');
const owner = module.create({tr: (en, zh) => window.EU_LANG === 'zh' ? zh : en});
const statuses = [
  {active_message_job_id: 'job', last_turn_status: 'failed'},
  {last_turn_status: 'interrupted', history_turn_count: 4},
  {last_turn_status: 'done', history_turn_count: 4},
  {history_turn_count: 4}, {has_history: true}, null,
  // The host's state of the conversation's latest job wins over its last reply.
  {conversation_status: 'running', last_turn_status: 'done'},
  {conversation_status: 'needs_user', last_turn_status: 'done'},
  {conversation_status: 'stopped', last_turn_status: 'done'},
  {conversation_status: 'done', last_turn_status: 'failed'},
  {conversation_status: null, last_turn_status: 'done', history_turn_count: 4},
].map(owner.sessionStatusLabel);
const times = [
  null, {created_at: 'invalid'}, {created_at: '2026-10-17T12:00:00Z'},
  {created_at: '2026-10-10T12:00:00Z', last_activity_at: '2026-10-16T11:57:00Z'},
  {created_at: '2026-10-16T10:00:00Z'}, {created_at: '2026-10-14T12:00:00Z'},
].map(owner.compactSessionTime);
process.stdout.write(JSON.stringify({statuses, times}));
"""
    result = run_node(node, program, str(owner.resolve()), language, check=True)
    payload = json.loads(result.stdout)
    assert payload["statuses"] == (
        [
            "进行中", "已停止", "已回复", "4 轮", "已有对话", "未开始",
            "进行中", "待确认", "已停止", "任务已结束", "4 轮",
        ]
        if language == "zh"
        else [
            "Running",
            "Stopped",
            "Replied",
            "4 turns",
            "Has conversation",
            "Not started",
            "Running",
            "Needs you",
            "Stopped",
            "Task ended",
            "4 turns",
        ]
    )
    assert payload["times"] == (
        ["", "", "刚刚", "3 分钟前", "2 小时前", "2 天前"]
        if language == "zh"
        else ["", "", "now", "3m", "2h", "2d"]
    )

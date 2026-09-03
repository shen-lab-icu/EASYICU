"""Contract for the user-facing Copilot text boundary.

The Copilot system prompt forbids internal identifiers in replies in five
separate rules, and live sessions violated all five at once.  These tests lock
the boundary that stopped depending on the model's cooperation.
"""

from __future__ import annotations

from easyicu.webserver.pi_copilot.projections import project_transcript
from easyicu.webserver.pi_copilot.user_visible_text import (
    sanitize_user_visible_text,
)


def test_grant_names_and_owner_paths_leave_a_grammatical_sentence() -> None:
    text = (
        "本轮仍未获得可用的 `provider_run` 授权，EasyICU 未提交新计划；"
        "历史失败运行 `run_271251946725` 保持为历史记录，"
        "owner: `easyicu.webserver.study_contexts` 。"
    )

    assert sanitize_user_visible_text(text) == (
        "本轮仍未获得可用的授权，EasyICU 未提交新计划；历史失败运行保持为历史记录。"
    )


def test_english_reply_keeps_its_subject_when_an_id_is_removed() -> None:
    """Deleting the identifier must not strand the verb it belongs to."""

    assert sanitize_user_visible_text(
        "The background job id: 9b80fa4c2069 is running."
    ) == "The background job is running."


def test_chinese_id_announcement_is_dropped_whole() -> None:
    assert sanitize_user_visible_text(
        "已提交新的规划任务，任务 ID 为 `ccd229583f85`。规划阶段不读取患者行。"
    ) == "已提交新的规划任务。规划阶段不读取患者行。"


def test_reason_codes_survive() -> None:
    """A reason code is the only actionable detail in a refusal.

    Unlike a grant name it is self-describing, and it usually sits in object
    position where deleting it would strand the verb.
    """

    assert sanitize_user_visible_text(
        "EasyICU returned `study_setup_incomplete`, owner: easyicu.webserver.x.y."
    ) == "EasyICU returned `study_setup_incomplete`."


def test_ordinary_scientific_prose_is_untouched() -> None:
    for text in (
        "请先在 EasyICU 中完成当前 MIMIC-IV 数据包的注册与验证。",
        "已保存：结局为 ICU 死亡，暴露为 Sepsis-3。",
        "The plan cites [Singer 2016](https://example.org/x) and uses **SOFA-2**.",
        "**下一步：**\n- 生成正式研究计划\n- 暂不生成",
    ):
        assert sanitize_user_visible_text(text) == text


def test_only_assistant_turns_are_rewritten() -> None:
    """A user turn is the researcher's own wording, quoted back verbatim."""

    rows = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "run_271251946725 怎么了？"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "运行 `run_271251946725` 已失败。"}],
            "entry_id": "a1",
        },
    ]

    projected = project_transcript(rows)

    assert projected[0]["content"][0]["text"] == "run_271251946725 怎么了？"
    assert projected[1]["content"][0]["text"] == "运行已失败。"
    assert projected[1]["entry_id"] == "a1"


def test_exact_legacy_formal_plan_action_is_projected_concisely() -> None:
    rows = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "请基于已确认的研究问题和 EasyICU 数据库能力目录，并在已有数据包时一并利用其信息，"
                        "检索相关文献并生成正式、证据绑定的研究计划。仅按元数据规划时不要读取患者行；"
                        "在提取或分析前停下等待我审阅。"
                    ),
                }
            ],
        }
    ]

    projected = project_transcript(rows)

    assert projected[0]["content"][0]["text"] == "开始生成候选研究计划。"


def test_similar_researcher_wording_is_not_rewritten() -> None:
    text = "请按元数据规划并生成正式研究计划，但先告诉我会检索哪些文献。"
    rows = [
        {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        }
    ]

    assert project_transcript(rows)[0]["content"][0]["text"] == text


def test_transcript_projection_tolerates_unexpected_shapes() -> None:
    assert project_transcript(None) == []
    assert project_transcript([{"role": "assistant"}]) == [{"role": "assistant"}]
    assert project_transcript([{"role": "assistant", "content": "raw"}]) == [
        {"role": "assistant", "content": "raw"}
    ]

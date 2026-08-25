"""Host-owned authority inferred only from the current user turn."""

from __future__ import annotations

import re
import unicodedata

__all__ = ["infer_explicit_turn_actions"]


_DENIAL_MARKERS = (
    "不授权",
    "未授权",
    "不确认",
    "不同意",
    "拒绝",
    "取消授权",
    "不要执行",
    "停止执行",
    "do not authorize",
    "don't authorize",
    "not authorize",
    "do not approve",
    "don't approve",
    "not approved",
    "do not extract",
    "decline",
    "reject",
    "cancel authorization",
)

_CONFIRMATION_MARKERS = (
    "确认并授权",
    "确认授权",
    "确认，授权",
    "确认,授权",
    "我确认",
    "我授权",
    "继续打开",
    "同意执行",
    "批准执行",
    "i confirm",
    "i authorize",
    "i approve",
    "continue opening",
    "confirmed and authorized",
)

_EXTRACTION_SCOPE_MARKERS = (
    "数据提取",
    "本地数据选择",
    "本地数据扫描",
    "数据选择与扫描",
    "提取和质量审阅",
    "准备并注册",
    "准备与注册",
    "准备和注册",
    "数据准备",
    "生成数据包",
    "data extraction",
    "extract the data",
    "extract data",
    "prepare and register",
    "preparing and registering",
    "build the data package",
)


def _normalize(message: str) -> str:
    text = unicodedata.normalize("NFKC", str(message or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def infer_explicit_turn_actions(message: str) -> frozenset[str]:
    """Return narrowly authorized one-use actions from current user text.

    The model cannot supply this text.  Extraction requires both an explicit
    confirmation and an explicit extraction/data-preparation scope in the same
    user turn; denial language wins and ambiguous requests remain ungranted.
    """

    text = _normalize(message)
    if not text or any(marker in text for marker in _DENIAL_MARKERS):
        return frozenset()
    confirmed = any(marker in text for marker in _CONFIRMATION_MARKERS)
    extraction_scoped = any(
        marker in text for marker in _EXTRACTION_SCOPE_MARKERS
    )
    return frozenset({"extract"}) if confirmed and extraction_scoped else frozenset()

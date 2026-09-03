"""Host-owned authority inferred only from the current user turn."""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "explicitly_confirms_easyicu_registered_source",
    "infer_explicit_turn_actions",
    "infer_idea_mining_followup_intent",
    "infer_research_entry_intent",
]


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
    "授权 easyicu",
    "授权打开",
    "授权下载",
    "继续打开",
    "同意执行",
    "批准执行",
    "i confirm",
    "i authorize",
    "authorize easyicu",
    "authorize opening",
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
    "下载并准备",
    "数据准备",
    "生成数据包",
    "data extraction",
    "extract the data",
    "extract data",
    "prepare and register",
    "preparing and registering",
    "build the data package",
)

_DIRECT_EXTRACTION_AUTHORIZATION_MARKERS = (
    "一次性 extraction 授权",
    "one-time extraction authorization",
)

_PROVIDER_REPLAN_PATTERNS = (
    r"(?:请|现在|按当前配置|授权)?(?:重新)?生成(?:新的?|全新的?|干净的?)?(?:\s*research agent\s*)?分析计划",
    r"(?:请|现在|授权)?(?:重新规划|重做分析计划)",
    r"(?:一次性\s*)?provider_run\s*授权",
    r"(?:authorize|please|now)?[\s_-]*(?:regenerate|replan|generate)[\s_-]+(?:a[\s_-]+)?(?:the[\s_-]+)?(?:new[\s_-]+)?(?:research[\s_-]+agent[\s_-]+)?analysis[\s_-]+plan",
)


def _normalize(message: str) -> str:
    text = unicodedata.normalize("NFKC", str(message or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def explicitly_confirms_easyicu_registered_source(message: str) -> bool:
    """Return whether the user chose an EasyICU-registered prepared source.

    This authority is intentionally narrower than merely mentioning a database.
    It requires an action verb and wording that identifies either an
    EasyICU-owned prepared source or an already prepared local source.  The
    service still confirms that the bound path exactly matches a validated
    registry row before granting access, so raw local-folder choices remain
    behind the separate host-owned selection workflow.
    """

    text = _normalize(message)
    if not text or any(marker in text for marker in _DENIAL_MARKERS):
        return False
    action = bool(
        re.search(
            r"(?:确认使用|选择使用|使用|复用|采用|"
            r"confirm(?: and)? use|confirm|choose|select|use|reuse)",
            text,
        )
    )
    prepared = bool(
        re.search(
            r"(?:已准备|已注册|已验证|可用|完整|"
            r"prepared|registered|validated|available|complete)",
            text,
        )
    )
    prepared_owner = "easyicu" in text or bool(
        re.search(r"(?:本地|local)", text)
    )
    return action and prepared and prepared_owner


def infer_explicit_turn_actions(message: str) -> frozenset[str]:
    """Return narrowly authorized one-use actions from current user text.

    The model cannot supply this text.  Extraction requires both an explicit
    confirmation and an explicit extraction/data-preparation scope in the same
    user turn; denial language wins and ambiguous requests remain ungranted.
    """

    text = _normalize(message)
    if not text or any(marker in text for marker in _DENIAL_MARKERS):
        return frozenset()
    if any(re.search(pattern, text) for pattern in _PROVIDER_REPLAN_PATTERNS):
        return frozenset({"provider_run"})
    if any(marker in text for marker in _DIRECT_EXTRACTION_AUTHORIZATION_MARKERS):
        return frozenset({"extract"})
    confirmed = any(marker in text for marker in _CONFIRMATION_MARKERS)
    extraction_scoped = any(
        marker in text for marker in _EXTRACTION_SCOPE_MARKERS
    )
    return frozenset({"extract"}) if confirmed and extraction_scoped else frozenset()


def infer_research_entry_intent(message: str) -> str:
    """Classify only an unscoped first turn in a blank research project.

    Explicit starter wording wins.  A plain scientific sentence remains
    ambiguous because completeness of prose does not prove whether the
    researcher wants exploration or implementation.
    """

    text = _normalize(message)
    if re.search(
        r"(?:研究问题|科学问题).{0,12}(?:已经|已|很)?(?:明确|确定|定稿)|"
        r"(?:直接|进入).{0,12}(?:研究方案|数据准备|实现)|"
        r"(?:不要|无需|跳过).{0,8}(?:idea mining|想法发掘)|"
        r"(?:生成|制定|形成|准备).{0,16}(?:研究|分析)?计划|"
        r"(?:generate|draft|prepare|create)(?:\s+\w+){0,6}\s+(?:research\s+|analysis\s+)?plan",
        text,
    ):
        return "implement_scientific_question"
    if re.search(
        r"(?:从|基于).{0,8}(?:icu )?数据(?:开始|出发)|"
        r"(?:提取|分析).{0,8}(?:icu )?数据|构建队列",
        text,
    ):
        return "data_first_entry"
    if re.search(
        r"(?:我|目前|现在)?(?:还)?没有.{0,8}(?:具体)?(?:研究)?(?:方向|想法|选题)|"
        r"不知道.{0,10}(?:做什么研究|研究什么|选什么题|从哪里开始)",
        text,
    ):
        return "idea_discovery_entry"
    if re.search(
        r"idea mining|(?:发掘|探索|评估|比较).{0,12}(?:想法|方向|idea)|"
        r"(?:想法|idea).{0,12}(?:发掘|探索|评估|比较)",
        text,
    ):
        return "idea_mining_entry"
    return "clarify_research_entry"


def infer_idea_mining_followup_intent(message: str) -> str | None:
    """Recognize only a host-rendered Idea Mining candidate selection."""

    text = _normalize(message)
    if re.match(
        r"^(?:(?:我\s*)?(?:选择|继续)方向\s*\d+\s*[：:]|"
        r"我\s*更想研究方向\s*\d+\s*[：:]|"
        r"(?:select|continue)\s+direction\s*\d+\s*:)",
        text,
    ):
        return "idea_mining_candidate_selection"
    return None

"""Owner for what may appear in user-facing EasyICU Copilot prose.

Why this exists
---------------
Grant names, owner paths, run ids and reason codes are the vocabulary of the
tool receipts Pi reads.  They are useful to the model and to a developer, and
meaningless to a researcher: a screen reading ``本轮仍未获得可用的
`provider_run` 授权 ... owner: easyicu.webserver...`` tells an ICU clinician
nothing they can act on.

The Copilot system prompt already forbids this in five separate rules (never
expose internal lifecycle detail, never expose registry terminology or internal
run labels, never open with schema paths or implementation jargon, never ask
the user to trigger internal resolution, and keep choice labels free of
internal identifiers).  Live sessions violate all five, which is the evidence
that asking a model to withhold a token is not a boundary.  This module makes
it one.

Scope
-----
Identifiers only.  This deletes a closed set of machine-facing tokens from text
on its way to the browser; it never paraphrases, summarizes, or rewrites the
model's sentences.  Clumsy prose is a prompt problem and stays visible.  The
receipts Pi itself reads are untouched -- the model still needs its own
vocabulary to reason.
"""

from __future__ import annotations

import re
from typing import Final

# One-turn grant names and the internal action vocabulary that rides with them.
# These name a host-held permission the user cannot see, let alone toggle, so a
# sentence that mentions one is strictly less useful than the same sentence
# with it removed.
_INTERNAL_TOKENS: Final = (
    "provider_run",
    "provider_gate",
    "external_llm_opt_in",
    "study_context_id",
    "missing_setup_fields",
    "planning_prerequisites_missing",
    "budget_mode",
    "planner_canary",
    "full_reviewed",
)

# One pre-typed host action from the plan-first prototype was accidentally
# persisted as if the researcher had written it.  Keep the stored transcript
# immutable, but project that exact historical action to the concise label the
# button should always have shown.  Exact equality is intentional: scientific
# questions that merely use similar words remain the researcher's own text.
_LEGACY_FORMAL_PLAN_TRIGGER_ZH: Final = (
    "请基于已确认的研究问题和 EasyICU 数据库能力目录，并在已有数据包时一并利用其信息，"
    "检索相关文献并生成正式、证据绑定的研究计划。仅按元数据规划时不要读取患者行；"
    "在提取或分析前停下等待我审阅。"
)
_FORMAL_PLAN_ACTION_ZH: Final = "开始生成正式研究计划。"

# ``owner: easyicu.webserver.study_contexts`` and friends, including the form
# the model invents where a Chinese gloss trails the dotted path.
_OWNER_CLAUSE = re.compile(
    r"[,，;；]?\s*(?:owner|所有者|归属)\s*[:：]\s*`?[A-Za-z_]\w*(?:\.\w+)*"
    r"(?:[ 　][^\s`,，。;；]+)?`?",
    re.I,
)
_DOTTED_OWNER_PATH = re.compile(
    r"`?\beasyicu(?:\.[a-z_][a-z0-9_]*){2,}\b`?",
    re.I,
)
# Internal run / job / study labels: ``run_271251946725``, ``job_9b80fa4c``.
_INTERNAL_LABEL = re.compile(
    r"`?\b(?:run|job|study|src|pi)_[A-Za-z0-9]{6,}\b`?",
)
# "任务 ID 为 `ccd229583f85`" / "job id: 9b80fa4c".  Here the identifier is the
# object of the clause, so deleting the token alone strands the verb; the whole
# announcement has to go.  The user tracks a run from the workflow panel, never
# by copying a hex id out of prose.
_ID_ANNOUNCEMENT = re.compile(
    r"[,，、;；]?\s*(?:[^\s,，。;；]{0,6}?(?:任务|作业|运行|项目|会话)\s*(?:ID|id|标识)"
    r"|(task|job|run|project|session)\s+(?:id|identifier))"
    r"\s*(?:为|是|叫)?\s*[:：]?\s*`?[0-9a-zA-Z_-]{6,64}`?",
)
_INTERNAL_TOKEN_SPAN = re.compile(
    r"`?\b(?:" + "|".join(re.escape(token) for token in _INTERNAL_TOKENS) + r")\b`?",
)
# Reason codes such as ``study_setup_incomplete`` are deliberately NOT removed.
# They are self-describing, they carry the only actionable detail in a refusal,
# and unlike a grant name they often sit in object position -- deleting one
# strands the verb ("EasyICU returned .").  Their remaining jargon is a prompt
# problem, not a boundary problem.

_CJK_RANGE: Final = "\u3000-\u9fff\uf900-\ufaff\uff00-\uffef"


def _collapse(text: str) -> str:
    """Close the hole a deletion leaves without disturbing the rest."""

    # A removed token between two CJK characters leaves a gap Chinese does not
    # use; between Latin words it leaves a doubled space.
    text = re.sub(
        rf"(?<=[{_CJK_RANGE}])[ \t]+(?=[{_CJK_RANGE}])",
        "",
        text,
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+(?=[,，。;；:：、)）])", "", text)
    text = re.sub(r"(?<=[(（])[ \t]+", "", text)
    # An emptied parenthetical or a stranded separator reads as a typo.
    text = re.sub(r"[（(]\s*[）)]", "", text)
    text = re.sub(r"(?:^|(?<=\n))([ \t]*)[,，;；]\s*", r"\1", text)
    text = re.sub(r"[,，;；](?=\s*[。.]|\s*$)", "", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    return text


def _drop_id_announcement(match: "re.Match[str]") -> str:
    """Chinese drops the whole appositive clause; English keeps its head noun.

    "...提交新的规划任务，任务 ID 为 `abc123`。" is appositive -- the clause can
    go whole.  "The background job id: abc123 is running." is not: ``job`` is
    the sentence subject, so only the identifier leaves.
    """

    noun = match.group(1)
    if not noun:
        return ""
    leading_space = " " if match.group(0)[:1].isspace() else ""
    return f"{leading_space}{noun}"


def sanitize_user_visible_text(value: str) -> str:
    """Remove machine-facing identifiers from one user-facing string.

    Deleting the identifier rather than substituting a label is deliberate: in
    both product languages these tokens sit in modifier position, so the
    sentence stays grammatical without one ("未获得可用的 `provider_run` 授权"
    -> "未获得可用的授权"), while any substituted noun would collide with the
    noun already there.
    """

    text = str(value or "")
    if not text:
        return text
    text = _ID_ANNOUNCEMENT.sub(_drop_id_announcement, text)
    for pattern in (
        _OWNER_CLAUSE,
        _DOTTED_OWNER_PATH,
        _INTERNAL_LABEL,
        _INTERNAL_TOKEN_SPAN,
    ):
        text = pattern.sub("", text)
    return _collapse(text)


def project_user_turn_text(value: str) -> str:
    """Project only known host-generated legacy actions in user turns.

    Ordinary user turns must remain byte-for-byte unchanged.  This boundary is
    therefore deliberately an exact lookup rather than a sanitizer or fuzzy
    match.
    """

    text = str(value or "")
    if text == _LEGACY_FORMAL_PLAN_TRIGGER_ZH:
        return _FORMAL_PLAN_ACTION_ZH
    return text


__all__ = ["project_user_turn_text", "sanitize_user_visible_text"]

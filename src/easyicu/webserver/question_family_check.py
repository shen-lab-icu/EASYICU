"""The analysis family a question asks for, as the conversation states it, checked by the host.

Owner of one record beside ``study_family_reading``.  The host's own readings
rest on words: a relationship, a proportion or an adjustment
(``study_family_reading``), and the design words the planner routes on
(``planner_stated_family``).  Words miss: 「乳酸越高，院内死亡越多吗？」 asks
for an association with no relation word, and 「与生存出院有关吗？」 names no
time-to-event analysis although the planner routes 「生存」 to one.  An offline
probe of 25 common phrasings (10-22) found 11 such misses: six associations
planned as descriptions, four in-scope eICU questions refused as survival or
subtyping studies.

So the conversation's model, which reads the whole question, states the family
it asks for and the question's own words that show it
(``easyicu_update_study_context.question_family``), and the host checks that
statement against its own reading:

* ``agreed``: both read the same family;
* ``model_only``: the host's words read no family, so the statement stands;
* ``host_disagrees``: the host's words read another family.  Neither is used:
  the plan step asks the researcher which one the question means
  (:func:`family_choice_texts`), and the answer is recorded as
  ``researcher_chose`` (:func:`researcher_choice`).

The record holds the words it rests on, so a question rewritten without them
leaves the record stale (:func:`settled_question_family` reads it as absent).
Codes only reach the receipt: the families, the words and the status.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

__all__ = [
    "FAMILY_DISAGREEMENT",
    "QUESTION_FAMILIES",
    "QUESTION_FAMILY_CHECK_FIELD",
    "check_question_family",
    "family_choice_texts",
    "host_question_family",
    "normalize_question_family_check",
    "open_disagreement",
    "researcher_choice",
    "settled_question_family",
]

QUESTION_FAMILY_CHECK_FIELD = "question_family_check"
#: The plan step's code while the host's and the model's readings differ.
FAMILY_DISAGREEMENT = "study_family_reading_disagreement"
QUESTION_FAMILIES = (
    "descriptive_epidemiology",
    "association_study",
    "prediction_model",
    "survival",
    "causal_inference",
    "trajectory_clustering",
)
_AGREED = "agreed"
_MODEL_ONLY = "model_only"
_HOST_DISAGREES = "host_disagrees"
_RESEARCHER_CHOSE = "researcher_chose"
_STATUSES = (_AGREED, _MODEL_ONLY, _HOST_DISAGREES, _RESEARCHER_CHOSE)
_MAX_WORDS = 200
#: The researcher's choice between two readings, as the host offers it.
_CHOICE_TEXTS = {
    "descriptive_epidemiology": ("按描述性研究做（比例、率或分布）", "Treat it as a descriptive study (proportions, rates or distributions)"),
    "association_study": ("按关联研究做（暴露与结局是否相关、相关多强）", "Treat it as an association study (whether and how strongly exposure and outcome are related)"),
    "prediction_model": ("按预测模型做", "Treat it as a prediction model"),
    "survival": ("按生存分析做（时间到事件）", "Treat it as a survival analysis (time to event)"),
    "causal_inference": ("按因果效应估计做", "Treat it as a causal effect estimate"),
    "trajectory_clustering": ("按亚型或轨迹聚类做", "Treat it as subtype or trajectory clustering"),
}


def _canonical(family: Any) -> Optional[str]:
    from easyicu.research_agent.planning.analysis_types import canonical_analysis_family

    text = str(family or "").strip()
    if not text:
        return None
    canonical = canonical_analysis_family(text) or text
    return canonical if canonical in QUESTION_FAMILIES else None


def host_question_family(question: Any) -> Optional[str]:
    """The family the host's own words read in ``question``, or ``None``.

    The planner's default for a question its routing does not place is not a
    reading: it rests on no words.
    """

    from easyicu.webserver.study_family_reading import (
        planner_stated_family,
        study_family_reading,
    )

    text = str(question or "")
    if not text.strip():
        return None
    reading = study_family_reading(text)
    if reading is not None:
        return _canonical(reading.family)
    from easyicu.extensions.method_attachments import question_family_for_method

    # A planner family a question family covers is read as that question
    # family: a dynamic (landmark) prediction is a prediction question.
    stated = planner_stated_family(text)
    return _canonical(question_family_for_method(stated) if stated else None)


def check_question_family(stated: Any, question: Any) -> Dict[str, Any]:
    """The record of the model's ``stated`` family checked against ``question``.

    Raises ``ValueError`` when the statement is not a known family resting on
    the question's own words.
    """

    if not isinstance(stated, Mapping) or set(stated) != {"family", "words"}:
        raise ValueError("a question's family states its family and the words that show it")
    family = _canonical(stated.get("family"))
    if family is None:
        raise ValueError(f"a question's family is one of {', '.join(QUESTION_FAMILIES)}")
    words = str(stated.get("words") or "").strip()
    text = str(question or "")
    if not words or len(words) > _MAX_WORDS or words not in text:
        raise ValueError("a question's family rests on words written in the question")
    host = host_question_family(text)
    status = _MODEL_ONLY if host is None else _AGREED if host == family else _HOST_DISAGREES
    return {"model_family": family, "words": words, "host_family": host, "status": status}


def normalize_question_family_check(value: Any) -> Optional[Dict[str, Any]]:
    """The stored shape of a check; ``None`` clears it.  Raises ``ValueError`` otherwise."""

    if value is None or value == {}:
        return None
    keys = {"model_family", "words", "host_family", "status"}
    if not isinstance(value, Mapping) or not keys <= set(value) <= keys | {"chosen_family"}:
        raise ValueError("a family check states both readings, their words and its status")
    model = _canonical(value.get("model_family"))
    host = value.get("host_family")
    host = None if host is None else _canonical(host)
    status = value.get("status")
    words = str(value.get("words") or "").strip()
    if model is None or (value.get("host_family") is not None and host is None):
        raise ValueError("a family check reads known families")
    if status not in _STATUSES or not words or len(words) > _MAX_WORDS:
        raise ValueError("a family check has a known status and the question's words")
    if (status == _MODEL_ONLY) != (host is None) or (status == _AGREED) != (host == model):
        if status != _RESEARCHER_CHOSE:
            raise ValueError("a family check's status follows its two readings")
    record: Dict[str, Any] = {
        "model_family": model, "words": words, "host_family": host, "status": status,
    }
    if status == _RESEARCHER_CHOSE:
        chosen = _canonical(value.get("chosen_family"))
        if host is None or host == model or chosen not in (model, host):
            raise ValueError("a researcher chooses one of two differing readings")
        record["chosen_family"] = chosen
    elif "chosen_family" in value:
        raise ValueError("only a researcher's choice names a chosen family")
    return record


def _rests_on(record: Mapping[str, Any], question: Any) -> bool:
    return str(record.get("words") or "") in str(question or "")


def settled_question_family(record: Any, question: Any) -> Optional[str]:
    """The family the check settles for ``question``; ``None`` when it settles none.

    A disagreement settles none, and so does a record whose words the
    question no longer holds.
    """

    if not isinstance(record, Mapping) or not _rests_on(record, question):
        return None
    status = record.get("status")
    if status == _RESEARCHER_CHOSE:
        return _canonical(record.get("chosen_family"))
    if status in (_AGREED, _MODEL_ONLY):
        return _canonical(record.get("model_family"))
    return None


def open_disagreement(record: Any, question: Any) -> Optional[Dict[str, Any]]:
    """The two readings the researcher is asked to choose between, or ``None``."""

    if (
        not isinstance(record, Mapping)
        or record.get("status") != _HOST_DISAGREES
        or not _rests_on(record, question)
    ):
        return None
    return {
        "code": FAMILY_DISAGREEMENT,
        "model_family": record["model_family"],
        "host_family": record["host_family"],
    }


def family_choice_texts(disagreement: Mapping[str, Any], language: str) -> list[tuple[str, str]]:
    """The two choices the host offers, as (family, text) in the reply's language."""

    index = 0 if language == "zh" else 1
    families = [disagreement["model_family"], disagreement["host_family"]]
    return [(family, _CHOICE_TEXTS[family][index]) for family in families]


def researcher_choice(record: Any, question: Any, message: Any) -> Optional[Dict[str, Any]]:
    """The record after the researcher chose one offered reading, or ``None``.

    The choice is one of the texts the host offered, sent as the message.
    """

    disagreement = open_disagreement(record, question)
    if disagreement is None:
        return None
    sent = str(message or "").strip().rstrip("。.")
    for language in ("zh", "en"):
        for family, text in family_choice_texts(disagreement, language):
            if sent == text:
                return {**dict(record), "status": _RESEARCHER_CHOSE, "chosen_family": family}
    return None

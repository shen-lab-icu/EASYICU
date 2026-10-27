"""Stated variables the conversation could not record, kept until it does.

Owner of one StudyContext record, ``stated_variables_unread``, beside the
verified readings ``stated_adjustments`` and ``stated_subgroups``
(``research_launch_scientific.checked_stated_variables``).  The save tool does
not record a reading the host cannot verify, and says why in that turn only.
The question still states the variable, though.  On pilot1 (10-26) a question
saying 「调整年龄和性别后」 was saved as 「调整年龄」 and 「调整性别」.  The
second quote is not the question's text, so sex was not recorded, the
conversation never stated it again, and a planner-selectable roster bound age
alone: an analysis under-adjusted for sex, with no stop.

So a rejected reading whose variable the question itself names is kept here
with the host's reason, until the conversation records it in its field or the
question no longer names it (:func:`pending_unread`).  Meanwhile a new plan or
run stops with :data:`STATED_VARIABLE_UNRECORDED` (:func:`blocking_unread`),
unless it is an adjustment the study's exact roster already binds.  A rejected
reading the question does not name is only reported: an invented variable, or
one the question negates, is no requirement of the question.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

__all__ = [
    "STATED_VARIABLE_UNRECORDED",
    "UNREAD_STATED_VARIABLES_FIELD",
    "blocking_unread",
    "normalize_unread_stated_variables",
    "pending_unread",
    "stop_unrecorded_stated_variables",
    "unread_records",
]

UNREAD_STATED_VARIABLES_FIELD = "stated_variables_unread"
#: The stable stop of a plan or run while a stated variable is unrecorded.
STATED_VARIABLE_UNRECORDED = "study_stated_variable_unrecorded"
_MAX_ITEMS = 16
_ROLES = {"stated_adjustments": "adjust for", "stated_subgroups": "analyse by subgroups of"}


def _fields() -> tuple[str, ...]:
    from easyicu.research_agent.planning.adjustment_authority import STATED_VARIABLE_FIELDS

    return tuple(STATED_VARIABLE_FIELDS)


def _text(value: Any, limit: int) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text or len(text) > limit or any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError("an unread stated variable has its field, variable and reason")
    return text


def normalize_unread_stated_variables(value: Any) -> List[Dict[str, str]]:
    """The stored shape: at most 16 distinct ``{field, variable, reason}`` records."""

    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_ITEMS:
        raise ValueError("unread stated variables are a list of at most 16 records")
    records: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"field", "variable", "reason"}:
            raise ValueError("an unread stated variable has its field, variable and reason")
        if item["field"] not in _fields():
            raise ValueError("an unread stated variable belongs to a stated-variable field")
        records.append({
            "field": item["field"],
            "variable": _text(item["variable"], 160),
            "reason": _text(item["reason"], 240),
        })
    if len({(item["field"], item["variable"]) for item in records}) != len(records):
        raise ValueError("unread stated variables are distinct")
    return records


def _named(text: str) -> frozenset[str]:
    from easyicu.webserver.study_intent import StudyIntentError, catalog_named_concepts

    try:
        return frozenset(catalog_named_concepts(text))
    except StudyIntentError:
        return frozenset()


def _question_names(variable: str, question: str, question_concepts: frozenset[str]) -> bool:
    """Whether the question names ``variable``: by its dictionary concepts, else its words.

    A variable the dictionary reads must name only concepts the question
    names, not negated; a name the dictionary does not read must be written
    in the question.
    """

    concepts = _named(variable)
    return concepts <= question_concepts if concepts else variable in question


def unread_records(rejected: Iterable[Mapping[str, Any]], *, field: str) -> List[Dict[str, str]]:
    """The rejected readings of ``field`` as records; :func:`pending_unread` keeps
    those of a variable the question names."""

    records: List[Dict[str, str]] = []
    for item in rejected:
        variable = str(item.get("variable") or "").strip()[:160]
        reason = str(item.get("reason") or "").strip()[:240] or "not recorded"
        if variable:
            records.append({"field": field, "variable": variable, "reason": reason})
    return records


def _recorded(readings: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    variables = {str(item.get("variable") or "") for item in readings}
    concepts = {str(item["concept"]) for item in readings if item.get("concept")}
    for variable in variables:
        concepts |= _named(variable)
    return variables, concepts


def pending_unread(context: Mapping[str, Any], records: Any = None) -> List[Dict[str, str]]:
    """The unread records still owed: named by the question, not yet recorded.

    ``records`` defaults to the context's own.  A record is settled when its
    field records the same variable, or every concept the variable names; it
    lapses when the question no longer names it.  The latest record of a
    variable is kept.
    """

    raw = context.get(UNREAD_STATED_VARIABLES_FIELD) if records is None else records
    items = [item for item in (raw or ()) if isinstance(item, Mapping)]
    if not items:
        return []
    question = str(context.get("question") or "")
    question_concepts = _named(question)
    latest: Dict[tuple[str, str], Dict[str, str]] = {}
    for item in items:
        key = (str(item.get("field") or ""), str(item.get("variable") or ""))
        latest.pop(key, None)
        latest[key] = dict(item)
    pending: List[Dict[str, str]] = []
    for (field, variable), item in latest.items():
        readings = context.get(field)
        variables, concepts = _recorded(readings if isinstance(readings, list) else [])
        named = _named(variable)
        settled = variable in variables or bool(named and named <= concepts)
        if not settled and _question_names(variable, question, question_concepts):
            pending.append(item)
    return pending[-_MAX_ITEMS:]


def _exact_roster(study: Mapping[str, Any]) -> frozenset[str]:
    from easyicu.webserver.study_scientific_configuration import (
        ScientificConfiguration, ScientificConfigurationError,
    )

    try:
        configuration = ScientificConfiguration.inspect(study)
        if configuration.covariate_selection() != "exact":
            return frozenset()
        return frozenset(configuration.covariates())
    except ScientificConfigurationError:
        return frozenset()  # the launch reports the invalid roster itself


def blocking_unread(study: Mapping[str, Any]) -> List[Dict[str, str]]:
    """The owed records a new plan or run stops on.

    The record stays owed, but an adjustment whose every concept the study's
    exact roster holds does not stop a plan: the plan binds that roster
    exactly (``family_spec.request._association_adjustment``).  When the
    roster is opened or loses it, the stop returns.
    """

    try:
        pending = pending_unread(study, normalize_unread_stated_variables(
            study.get(UNREAD_STATED_VARIABLES_FIELD)
        ))
    except ValueError:
        return [{"field": "stated_adjustments", "variable": "", "reason": "unreadable record"}]
    roster = _exact_roster(study) if pending else frozenset()
    return [
        item for item in pending
        if not (item["field"] == "stated_adjustments" and roster
                and (named := _named(item["variable"])) and named <= roster)
    ]


def stop_unrecorded_stated_variables(study: Mapping[str, Any]) -> None:
    """Stop a new plan or run while a variable the question states is unrecorded."""

    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    pending = blocking_unread(study)
    if not pending:
        return
    first = pending[0]
    asks = " and ".join(
        f"to {_ROLES[item['field']]} “{item['variable']}”"
        for item in pending if item["variable"]
    ) or "for a variable"
    raise ResearchPipelineRunError(
        STATED_VARIABLE_UNRECORDED,
        f"The question asks {asks}, but the conversation has not recorded that reading, so "
        "planning stopped before any model was called and no analysis was run. Record it "
        "again with words quoting the question's own clause that names it, or remove it "
        "from the question.",
        details={
            **({"stated_variable": first["variable"]} if first["variable"] else {}),
            "field": first["field"],
            "unrecorded": [dict(item) for item in pending],
        },
    )

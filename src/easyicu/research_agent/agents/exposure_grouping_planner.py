"""Ask the Planner whether a study forms its exposure by grouping one measured value.

Owner
-----
Before a plan is outlined, the Planner answers one short structured request
(``orchestration.exposure_grouping_phase`` makes it): does the study form the
groups of ICU stays it compares by where one measured value of each stay
falls?  When it does, the Planner states each grouping as rules
(:mod:`..planning.exposure_group_spec`); when it does not, it states none.
The request shows the values the input holds that a grouping can read
(``exposure_group_compile.grouping_sources``); the host decides whether each
rule can be read and derives the levels.  Nothing here reads a row or decides
a level.

A grouping is stated in the study's own words: its quote is copied from the
question or from the study's statements of whom it includes, as a population
criterion's is.  A response the spec owner refuses, or a grouping whose quote
is not written there, goes back to the Planner with the reason, as any
structured response does (``providers.structured_retry``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence, get_args

from ..canonical_json import canonical_sha256
from ..planning.exposure_group_compile import GroupingSource
from ..planning.exposure_group_spec import (
    MAX_EXPOSURE_GROUPINGS,
    MAX_EXPOSURE_GROUPS,
    ExposureGroupings,
    GroupOp,
    GroupSummary,
    read_stated_exposure_groupings,
    unquoted_groupings,
)
from ..planning.population_spec import STUDY_WORDING_SOURCES
from ..providers.capabilities import llm_supports_strict_json_schema
from ..providers.protocol import LLMMessage, StructuredOutputRequest
from ..providers.strict_json_schema import assert_closed_json_schema
from ..providers.structured_retry import call_llm_with_structured_retry
from ..schema import ResearchContext

EXPOSURE_GROUPING_ROLE = "exposure_grouping_planner"
EXPOSURE_GROUPING_MAX_OUTPUT_TOKENS = 3000
#: Feedback retries after the first answer.
EXPOSURE_GROUPING_MAX_RETRIES = 2

Transport = Literal["strict_schema", "contract_text"]

EXPOSURE_GROUPING_GUIDE = """\
You decide one thing before the study's analysis plan is outlined: whether \
the study forms its exposure -- the groups of ICU stays it compares or \
describes -- by where one measured value of each stay falls.

State a grouping only when the study's words form groups of stays by \
thresholds on one measured value, or name categories of that value whose \
usual thresholds those words point to.  State none when the study treats \
the value as a continuous quantity, compares stays by a variable the input \
already holds as categories, or groups stays by anything other than one \
measured value (a diagnosis, a treatment, a time).  Never invent a grouping \
the study does not form, nor a threshold its words do not state or point to.

Each grouping is rules the host checks and applies; you write no code:
- concept: the concept id of the measured value.  Use a concept the input \
lists below when it holds the value; a concept the input does not hold may \
be named, and the host reports the extraction it needs.
- window: the hours after ICU admission, [start_hours, end_hours), the value \
is summarized over; null for a value recorded once per stay.
- groups: two to six, ids g1 to g6, each with a short label and a rule \
{summary, op, value, unit}, or the rule "otherwise" for the last group.  \
Groups are matched in the order listed: a stay takes the first group whose \
rule it meets, and "otherwise" takes every measured stay no earlier group \
took.  summary is min, max, mean or first over the window, or value when \
there is no window.  unit is the unit the study writes the threshold in, \
or null when it writes none.
- scale: ordinal when the groups lie along the value's scale, numbered g1 \
lowest; nominal otherwise.
- unmeasured: {"handling": "exclude"} when stays without a measurement leave \
the study; {"handling": "own_group", "label": ...} when they form a group of \
their own, which only a nominal grouping can have.
- reference: the group the others are compared with, and contrast: the group \
whose comparison with the reference is the primary estimate, when the study \
names them; null otherwise.
- quote: the words that form the grouping, copied exactly as the study \
writes them, in their own language; source: question when they are in the \
research question, study_wording when they are in the study's own \
statements of whom it includes or excludes."""

_RESPONSE_SHAPE = (
    'Answer with one JSON object and no other text: {"groupings": []} when the '
    'study forms no such exposure; otherwise {"groupings": [{"id": "x1", '
    '"concept": "<concept id>", "window": {"start_hours": <number>, '
    '"end_hours": <number>} or null, "scale": "nominal" or "ordinal", '
    '"groups": [{"id": "g1", "label": "<words>", "rule": {"summary": '
    '"min|max|mean|first|value", "op": "<|<=|>|>=", "value": <number>, '
    '"unit": "<unit>" or null}}, ..., {"id": "gN", "label": "<words>", '
    '"rule": "otherwise"}], "unmeasured": {"handling": "exclude"} or '
    '{"handling": "own_group", "label": "<words>"}, "reference": "<group id>" '
    'or null, "contrast": "<group id>" or null, "quote": "<the study\'s exact '
    'words>", "source": "question" or "study_wording"}]}.'
)

_GROUPING_IDS = [f"x{index}" for index in range(1, MAX_EXPOSURE_GROUPINGS + 1)]
_GROUP_IDS = [f"g{index}" for index in range(1, MAX_EXPOSURE_GROUPS + 1)]


def study_wording_texts(context: ResearchContext) -> tuple[str, ...]:
    """The study's own words a grouping's quote is copied from."""

    return (
        str(context.research_question or ""),
        *(str(item) for item in context.cohort.inclusion_criteria),
        *(str(item) for item in context.cohort.exclusion_criteria),
    )


def exposure_grouping_messages(
    context: ResearchContext,
    sources: Sequence[GroupingSource],
    *,
    response_shape: str = "",
) -> list[LLMMessage]:
    """The request: the study's words and the values a grouping can read."""

    _question, *statements = study_wording_texts(context)
    lines = [f"Research question: {context.research_question}"]
    if statements:
        lines.append("The study's statements of whom it includes or excludes:")
        lines.extend(f"- {text}" for text in statements)
    lines.append(
        "Values the input holds that a grouping can read "
        "(concept: summaries and window; unit; declared values):"
    )
    lines.extend(source.line() for source in sources)
    if response_shape:
        lines.append(response_shape)
    return [
        LLMMessage(role="system", content=EXPOSURE_GROUPING_GUIDE),
        LLMMessage(role="user", content="\n".join(lines)),
    ]


def _enum(values: Sequence[str]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def exposure_grouping_schema() -> dict[str, Any]:
    """The closed shape of an answer: the groupings, or none."""

    rule = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "op", "value", "unit"],
        "properties": {
            "summary": _enum(get_args(GroupSummary)),
            "op": _enum(get_args(GroupOp)),
            "value": {"type": "number"},
            "unit": _nullable({"type": "string"}),
        },
    }
    group = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "label", "rule"],
        "properties": {
            "id": _enum(_GROUP_IDS),
            "label": {"type": "string"},
            "rule": {"anyOf": [rule, _enum(["otherwise"])]},
        },
    }
    window = {
        "type": "object",
        "additionalProperties": False,
        "required": ["start_hours", "end_hours"],
        "properties": {
            "start_hours": {"type": "number"},
            "end_hours": {"type": "number"},
        },
    }
    unmeasured = {
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["handling"],
                "properties": {"handling": _enum(["exclude"])},
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["handling", "label"],
                "properties": {
                    "handling": _enum(["own_group"]),
                    "label": {"type": "string"},
                },
            },
        ]
    }
    grouping = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id",
            "concept",
            "window",
            "scale",
            "groups",
            "unmeasured",
            "reference",
            "contrast",
            "quote",
            "source",
        ],
        "properties": {
            "id": _enum(_GROUPING_IDS),
            "concept": {"type": "string"},
            "window": _nullable(window),
            "scale": _enum(["nominal", "ordinal"]),
            "groups": {"type": "array", "items": group},
            "unmeasured": unmeasured,
            "reference": _nullable(_enum(_GROUP_IDS)),
            "contrast": _nullable(_enum(_GROUP_IDS)),
            "quote": {"type": "string"},
            "source": _enum(sorted(STUDY_WORDING_SOURCES)),
        },
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["groupings"],
        "properties": {"groupings": {"type": "array", "items": grouping}},
    }
    assert_closed_json_schema(schema)
    return schema


def exposure_grouping_structured_output() -> StructuredOutputRequest:
    return StructuredOutputRequest.from_schema(
        name="exposure_groupings", schema=exposure_grouping_schema(), strict=True
    )


def parse_exposure_groupings(
    raw: str, *, study_texts: Sequence[str]
) -> ExposureGroupings:
    """One answer, read by the spec owner and held to the study's words."""

    payload = json.loads(str(raw or "").strip())
    if not isinstance(payload, dict):
        raise ValueError("the answer is one JSON object holding a groupings list")
    groupings = read_stated_exposure_groupings(payload)
    for item in groupings.groupings:
        if item.source not in STUDY_WORDING_SOURCES:
            raise ValueError(
                f"grouping {item.id} cites the {item.source}: a grouping is formed "
                "in the study's words, so cite the question or the study's own "
                "statements"
            )
    for item in unquoted_groupings(groupings, study_texts):
        raise ValueError(
            f"grouping {item.id} cites the {item.source.replace('_', ' ')}, but "
            f"{item.quote!r} is not written there: quote the words that form it "
            "exactly as the study writes them, in their own language; never "
            "paraphrase or translate them"
        )
    return groupings


@dataclass(frozen=True)
class ExposureGroupingAnswer:
    """The groupings the Planner stated, and the request they answer."""

    groupings: ExposureGroupings
    transport: Transport
    request_sha256: str
    structured_output_authority_sha256: Optional[str]

    def record(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "request_sha256": self.request_sha256,
            "structured_output_authority_sha256": (
                self.structured_output_authority_sha256
            ),
            "stated": self.groupings.model_dump(mode="json"),
        }


def ask_exposure_groupings(
    llm: Any,
    *,
    context: ResearchContext,
    sources: Sequence[GroupingSource],
    progress_callback: Optional[Callable[[Any], None]] = None,
) -> ExposureGroupingAnswer:
    """Ask the Planner (``llm``) for the study's groupings; none is an answer.

    ``StructuredResponseFailure`` propagates when no answer the owner reads
    arrives within the retries.
    """

    schema = (
        exposure_grouping_structured_output()
        if llm_supports_strict_json_schema(llm)
        else None
    )
    messages = exposure_grouping_messages(
        context,
        sources,
        # An enforced schema carries the shape; a route without one would
        # otherwise answer the first request blind.
        response_shape="" if schema is not None else _RESPONSE_SHAPE,
    )
    study_texts = study_wording_texts(context)
    groupings = call_llm_with_structured_retry(
        llm,
        messages,
        parser=lambda raw: parse_exposure_groupings(raw, study_texts=study_texts),
        role=EXPOSURE_GROUPING_ROLE,
        max_retries=EXPOSURE_GROUPING_MAX_RETRIES,
        max_tokens=EXPOSURE_GROUPING_MAX_OUTPUT_TOKENS,
        temperature=0.2,
        include_failed_response_on_retry=True,
        progress_callback=progress_callback,
        structured_output=schema,
        format_reminder=_RESPONSE_SHAPE,
    )
    return ExposureGroupingAnswer(
        groupings=groupings,
        transport="strict_schema" if schema is not None else "contract_text",
        request_sha256=canonical_sha256(
            {
                "messages": [[item.role, item.content] for item in messages],
                "structured_output_authority_sha256": (
                    schema.authority_sha256 if schema is not None else None
                ),
            }
        ),
        structured_output_authority_sha256=(
            schema.authority_sha256 if schema is not None else None
        ),
    )


__all__ = [
    "EXPOSURE_GROUPING_GUIDE",
    "EXPOSURE_GROUPING_ROLE",
    "ExposureGroupingAnswer",
    "ask_exposure_groupings",
    "exposure_grouping_messages",
    "exposure_grouping_schema",
    "exposure_grouping_structured_output",
    "parse_exposure_groupings",
    "study_wording_texts",
]

"""Read a pre-plan proposal from the question and registered trial contracts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, Mapping

from easyicu.concept.export_metadata import concept_declares_event_status
from easyicu.databases import normalize_database_key
from easyicu.outcome_availability import (
    OUTCOME_CONCEPT_SUPPORTED_DATABASES,
    FixedHorizonMortalityEndpoint,
    fixed_horizon_mortality_endpoint,
    fixed_horizon_mortality_endpoint_stated_by,
    stated_mortality_horizon_mentions,
    stated_mortality_settings,
)
from easyicu.research_agent.contracts.target_trial_design import (
    MAX_GRACE_PERIOD_HOURS,
    MAX_TIME_ZERO_HOURS,
    MIN_TIME_ZERO_HOURS,
)
from easyicu.research_agent.planning.target_trial_spec import MAX_TREATMENT_CONCEPTS
from easyicu.research_agent.planning.treatment_capture import (
    packaged_treatment_capture_registry,
)
from easyicu.utils.death_time_semantics import DEATH_STATUS
from easyicu.webserver.causal_trial_design import (
    question_treatment_concepts,
    supported_treatment_classes,
)
from easyicu.webserver.causal_trial_reading import (
    CausalTrialReading,
    causal_trial_reading,
    time_zero_spans,
)
from easyicu.webserver.primary_cohort import ADULT_AGE_FLOOR_YEARS
from easyicu.webserver.study_intent import (
    concept_named_by,
    full_span_concepts,
)
from easyicu.webserver.target_trial_connectives import (
    EN_CONNECTIVES,
    ZH_CONNECTIVES,
)
from easyicu.webserver.target_trial_setup import (
    TargetTrialSetupError,
    TargetTrialStatement,
    read_target_trial_statement,
)

# A Chinese population topic may also use 患者中 without its opening 在.
_POPULATION = re.compile(
    r"(?:(?:in|among)\s+(?:(?P<en_qualifier>adult)\s+patients?|(?P<en_noun>adults?)|patients?|stays)"
    r"\s+with\s+(?P<en>[^,，.;；。?!？！]{2,100})"
    r"|(?:在\s*)?(?P<zh_adult>成人|成年)?\s*(?P<zh>[^,，.;；。?!？！、]{2,100}?)患者(?:中|人群))"
    r"\s*[,，.;；。?!？！]",
    re.IGNORECASE,
)
_TIME_ZERO_DIRECTIVE = re.compile(r"\b(?:set|define) $", re.IGNORECASE)
_OUTCOME_ORIGIN = re.compile(
    r"\b(?:from|after|since|following|post|starting at|as of)[ \-‐–—]+"
    r"(?:the )?ICU[ \-‐–—]+admission\b"
    r"|(?:自|从)\s*入(?:住)?\s*ICU\s*(?:[后後]\s*(?:起算|算起|起)?|起算|算起|起)"
    r"|入(?:住)?\s*ICU\s*[后後]",
    re.IGNORECASE,
)
_PUNCTUATION = "".join(
    sorted({
        unicodedata.normalize("NFKC", token)
        for token in (*EN_CONNECTIVES, *ZH_CONNECTIVES) if not token.isalpha()
    })
)
_CONNECTIVES = re.compile(
    "|".join([
        rf"(?<!\w)(?:{'|'.join(word for word in EN_CONNECTIVES if word.isalpha())})(?!\w)",
        *(word for word in ZH_CONNECTIVES if word.isalpha()),
        rf"[{re.escape(_PUNCTUATION)}]",
    ]),
    re.IGNORECASE,
)

_Span = tuple[int, int]
MissingTrialKey = Literal[
    "treatment", "strategies", "time_zero", "grace_period", "outcome",
    "eligibility", "adjustment", "protocol", "unread_words",
]


@dataclass(frozen=True)
class _Question:
    original: str
    text: str
    offsets: list[_Span]

    @classmethod
    def read(cls, original: str) -> _Question:
        folded = unicodedata.normalize("NFKC", original)
        pieces = [unicodedata.normalize("NFKC", char) for char in original]
        offsets = [
            (index, index + 1)
            for index, piece in enumerate(pieces)
            for _ in piece
        ]
        if "".join(pieces) != folded:
            # Composition or combining-mark reordering can cross character boundaries.
            # Attribute the rewritten suffix to the whole original range it came from.
            offsets = []
            previous = ""
            for end in range(1, len(original) + 1):
                current = unicodedata.normalize("NFKC", original[:end])
                shared = 0
                while shared < min(len(previous), len(current)) and previous[shared] == current[shared]:
                    shared += 1
                start = offsets[shared][0] if shared < len(offsets) else end - 1
                offsets[shared:] = [(start, end)] * (len(current) - shared)
                previous = current
        characters = []
        collapsed = []
        for match in re.finditer(r"\s+|\S", folded):
            start, end = match.span()
            if match.group().isspace():
                if start == 0 or end == len(folded):
                    continue
                characters.append(" ")
            else:
                characters.append(match.group())
            collapsed.append((offsets[start][0], offsets[end - 1][1]))
        return cls(original, "".join(characters), collapsed)

    def quote(self, start: int, end: int) -> str:
        return self.original[self.offsets[start][0]:self.offsets[end - 1][1]]

    def unread(self, spans: list[_Span]) -> list[str]:
        """Quote gaps with residual content, keeping connective words between residues."""
        connective = [False] * len(self.text)
        for match in _CONNECTIVES.finditer(self.text):
            start, end = match.span()
            connective[start:end] = [True] * (end - start)
        residual = []
        previous = 0
        for start, end in [*sorted(spans), (len(self.text), len(self.text))]:
            if any(
                not connective[index] and not self.text[index].isspace()
                for index in range(previous, start)
            ):
                left, right = previous, start
                trim = _PUNCTUATION.replace(".", "") + " "
                while self.text[left] in trim:
                    left += 1
                while self.text[right - 1] in trim:
                    right -= 1
                residual.append(self.quote(left, right))
            previous = max(previous, end)
        return residual


@dataclass(frozen=True)
class TargetTrialProposal:
    elements: list[dict[str, str]]
    missing: list[MissingTrialKey]
    statement: TargetTrialStatement | None
    unread_words: list[str]

    def public(self, revision: int) -> dict[str, Any]:
        return {
            "elements": self.elements,
            "missing": self.missing,
            "unread_words": self.unread_words,
            "expected_revision": revision,
            "digest": self.statement.request()["request_sha256"]
            if self.statement
            else None,
        }


def _indication_concept(words: str) -> str | None:
    concepts = full_span_concepts(words)
    concept = next(iter(concepts)) if len(concepts) == 1 else None
    return concept if isinstance(concept, str) and concept_declares_event_status(concept) else None


def _treatment(
    question: str, reading: CausalTrialReading, database: str
) -> tuple[tuple[str, ...], str] | None:
    if reading.treatment is None:
        return None
    registry = packaged_treatment_capture_registry().registry
    start, end = reading.treatment
    words = question[start:end].casefold().strip()
    explicit_classes = {
        row["id"]
        for row in supported_treatment_classes()
        if words in {row[key].casefold() for key in ("label_en", "label_zh", "id")}
    }
    entries = [
        entry
        for entry in registry.entries
        if entry.database == database
        and entry.absent_in_capture == "absent_row_is_no_event"
    ]
    if len(explicit_classes) == 1:
        name = next(iter(explicit_classes))
        agents = registry.class_agents(name)
        if not agents:
            return None
        eligible = [entry for entry in entries if set(entry.agents) <= agents]
        # Source entries can overlap. Prefer the unique smallest exact cover;
        # neither broader groups nor an arbitrary tie changes the treatment.
        for size in range(1, min(len(eligible), MAX_TREATMENT_CONCEPTS) + 1):
            covers = [
                tuple(entry.concept for entry in group)
                for group in combinations(eligible, size)
                if {agent for entry in group for agent in entry.agents} == agents
            ]
            if covers:
                return (covers[0], name) if len(covers) == 1 else None
        return None
    # A named drug must not expand to a wider class or to a broader capture.
    names = full_span_concepts(words)
    concepts = question_treatment_concepts(question, reading)
    selected = [entry for entry in entries if entry.concept in concepts]
    agents = frozenset(agent for entry in selected for agent in entry.agents)
    exact_classes = [
        name
        for name in registry.treatment_classes
        if registry.class_agents(name) == agents
    ]
    if (
        set(concepts) != names
        or set(concepts) != {entry.concept for entry in selected}
        or len(exact_classes) != 1
    ):
        return None
    return tuple(entry.concept for entry in selected), exact_classes[0]


def _proposal_adjustments(
    study: Mapping[str, Any], question: _Question,
) -> tuple[list[dict[str, str]], list[_Span]]:
    rows = study.get("stated_adjustments") or ()
    if any(
        isinstance(row, Mapping) and row.get("field") == "stated_adjustments"
        for row in study.get("stated_variables_unread") or ()
    ):
        return [], []
    if not rows or any(
        not isinstance(row, Mapping)
        or row.get("source_status") != "available"
        or not row.get("concept")
        or not row.get("words")
        or not row.get("variable")
        or str(row["words"]) not in question.original
        for row in rows
    ):
        return [], []
    by_quote: dict[str, set[str]] = {}
    for row in rows:
        words = _Question.read(str(row["words"])).text
        by_quote.setdefault(words, set()).add(str(row["concept"]))

    spans = []
    for words, concepts in by_quote.items():
        span = _adjustment_span(question.text, words, concepts)
        if span is None:
            return [], []
        spans.append(span)
    return (
        [
            {
                "name": str(row["concept"]),
                "clinical_rationale": str(row["words"])
                if len(str(row["words"])) >= 16
                else "Baseline adjustment requested by the researcher.",
                "source": "question",
            }
            for row in rows
        ],
        spans,
    )


def _adjustment_span(question: str, words: str, concepts: set[str]) -> _Span | None:
    """Read a stated list of exact variable names and its adjustment syntax."""
    # The two clause forms permit timing connectors only with an adjustment verb.
    prefix = (
        r"(?:(?:在|经)\s*)?(?:调整|校正|矫正|控制)\s*"
        r"|(?<!\w)(?:after\s+)?(?:adjust(?:ed|ing|ment)?\s+(?:for|by)|"
        r"controll?(?:ed|ing)\s+for|accounting\s+for)\s+"
    )
    join = r"(?:\s*[,、]\s*(?:and\s+)?|\s+and\s+|\s*[和与及]\s*)"
    end = r"(?:\s*[后後](?=\s*(?:[,。;!?]|比较|对比|$)))?(?![a-z0-9_])"
    cue = re.match(prefix, words, re.IGNORECASE)
    if cue is None:
        return None
    body = re.sub(r"\s*[后後]$", "", words[cue.end():])
    readings = [full_span_concepts(name) for name in re.split(join, body, flags=re.IGNORECASE)]
    if any(len(names) != 1 for names in readings) or set().union(*readings) != concepts:
        return None
    body = re.escape(body)
    pattern = re.compile(rf"(?:{prefix}){body}{end}", re.IGNORECASE)
    matches = [
        match.span() for match in pattern.finditer(question) if words in match.group()
    ]
    return matches[0] if len(matches) == 1 else None


def _fixed_outcome(
    question: str,
    confirmed_label: Any,
    database: str,
) -> tuple[FixedHorizonMortalityEndpoint, _Span] | None:
    # Read the primary from its confirmation when the question states more
    # than one horizon. Its proof remains the exact words of the question.
    label = str(confirmed_label or "").strip()
    confirmed = {
        fixed_horizon_mortality_endpoint_stated_by(item.horizon)
        for item in stated_mortality_horizon_mentions(label)
    }
    registered = fixed_horizon_mortality_endpoint(label)
    if registered is not None:
        confirmed.add(registered)
    if (
        label
        and not confirmed
        and (
            stated_mortality_settings(label)
            or concept_named_by(label, include_catalog_labels=True) != DEATH_STATUS
        )
    ):
        return None
    mentions = [
        (fixed_horizon_mortality_endpoint_stated_by(item.horizon), item)
        for item in stated_mortality_horizon_mentions(question)
    ]
    candidates = {endpoint for endpoint, _ in mentions}
    if confirmed:
        candidates = candidates & confirmed if len(confirmed) == 1 else set()
    if len(candidates) != 1:
        return None
    endpoint = next(iter(candidates))
    if endpoint is None or database not in OUTCOME_CONCEPT_SUPPORTED_DATABASES.get(
        endpoint.event_concept, ()
    ):
        return None
    match = next(item for candidate, item in mentions if candidate == endpoint)
    return endpoint, (match.start, match.end)


def _outcome_span(
    question: str, endpoint: FixedHorizonMortalityEndpoint, span: _Span, other_spans: list[_Span],
) -> _Span:
    """Read the endpoint with an immediately attached, encoded ICU origin."""
    start, end = span
    if question[end:end + 1] == "率" and question[end - 1] in "亡死":
        end += 1
    if endpoint.time_origin != "icu_admission":
        return start, end
    def independent(match: re.Match[str]) -> bool:
        return all(match.end() <= left or match.start() >= right for left, right in other_spans)

    prefixes = [
        match for match in _OUTCOME_ORIGIN.finditer(question, 0, start)
        if re.fullmatch(r"\s*,?\s*", question[match.end():start])
        and independent(match)
    ]
    suffix = _OUTCOME_ORIGIN.match(question, end + int(question[end:end + 1] == " "))
    if suffix and not independent(suffix):
        suffix = None
    if prefixes and suffix is None:
        return prefixes[-1].start(), end
    if suffix and not prefixes:
        return start, suffix.end()
    return start, end


def target_trial_proposal(study: Mapping[str, Any]) -> TargetTrialProposal | None:
    wording = _Question.read(str(study.get("question") or ""))
    question = wording.text
    reading = causal_trial_reading(question)
    source = study.get("data_source")
    if (
        reading is None
        or not isinstance(source, Mapping)
        or not source.get("database")
        or not source.get("path")
    ):
        return None
    try:
        database = normalize_database_key(str(source["database"]))
    except KeyError:
        return None

    evidence = {item.element: wording.quote(item.start, item.end) for item in reading.elements}
    elements: list[dict[str, str]] = []
    missing: list[MissingTrialKey] = []
    covered: list[_Span] = []

    def add(key: str, quote: str, *, default: bool = False) -> dict[str, str]:
        origin = "design_choice" if default else "question"
        elements.append({"key": key, "value": quote, "source": origin})
        return {"quote": quote[:160], "source": origin}

    spec: dict[str, Any] = {}
    treatment = _treatment(question, reading, database)
    if treatment and reading.treatment:
        start, end = reading.treatment
        covered.append((start, end))
        spec["treatment"] = {
            **add("treatment", wording.quote(start, end)),
            "concepts": treatment[0],
            "treatment_class": treatment[1],
        }
        add("treatment_capture", ", ".join(treatment[0]), default=True)
    else:
        missing.append("treatment")

    strategy_spans = {
        item.element: (item.start, item.end)
        for item in reading.elements
        if item.element in {"initiate_strategy", "defer_strategy"}
    }
    initiate, defer = evidence.get("initiate_strategy"), evidence.get("defer_strategy")
    initiation = strategy_spans.get("initiate_strategy")
    deferral = strategy_spans.get("defer_strategy")
    if initiation and reading.treatment:
        # The strategy reader locates the treatment. Its own reader must account for the name.
        covered.extend([(initiation[0], reading.treatment[0]), (reading.treatment[1], initiation[1])])
    # The shape reader can spot "never" as causal. A lifelong ban is a different strategy.
    if initiate and defer and deferral and not question[slice(*deferral)].lower().startswith("never "):
        covered.append(deferral)
        spec["strategies"] = {
            **add("strategies", f"{initiate} / {defer}"),
            # Labels are descriptive. The executable definitions are fixed by the trial contract.
            "initiate_label": initiate[:80],
            "defer_label": defer[:80],
        }
        add("strategy_window", "", default=True)
    else:
        missing.append("strategies")

    population = _POPULATION.match(question)
    zero_spans = time_zero_spans(question)
    zero, grace = reading.time_zero_hours, reading.grace_period_hours
    zero_words = evidence.get("time_zero")
    if (
        zero_words
        and zero is not None
        and zero.is_integer()
        and MIN_TIME_ZERO_HOURS <= zero <= MAX_TIME_ZERO_HOURS
        and len(zero_spans) == 1
    ):
        start, end = zero_spans[0]
        directive = _TIME_ZERO_DIRECTIVE.search(question[:start])
        covered.append((directive.start() if directive else start, end))
        spec["time_zero"] = {
            **add("time_zero", zero_words),
            "hours_after_icu_admission": int(zero),
        }
    else:
        missing.append("time_zero")
    if (
        grace is not None
        and grace.is_integer()
        and 1 <= grace <= MAX_GRACE_PERIOD_HOURS
        and initiate
    ):
        spec["grace_period"] = {
            **add("grace_period", initiate),
            "hours": int(grace),
        }
    else:
        missing.append("grace_period")

    outcome = _fixed_outcome(question, study.get("outcome"), database)
    if outcome:
        endpoint, span = outcome
        covered.append(_outcome_span(question, endpoint, span, covered))
        spec["outcome"] = {**add("outcome", wording.quote(*span)), "endpoint": endpoint.event_concept}
        add("outcome_origin", endpoint.time_origin, default=True)
    else:
        missing.append("outcome")

    indication = None
    if population:
        indication = _indication_concept((population["en"] or population["zh"]).strip())
    criteria = []
    if population and indication:
        start, end = population.span()
        covered.append((start, end))
        population_words = wording.quote(start, end - 1).rstrip()
    if population and indication and "time_zero" in spec:
        criteria.append(
            {
                "id": "c1",
                "role": "include",
                "kind": "condition_present",
                "quote": population_words,
                "source": "question",
                "concepts_all_of": [indication],
                "window": {"start_hours": 0, "end_hours": int(zero)},
            }
        )
        spec["indication"] = {**add("eligibility", population_words), "criterion_ids": ["c1"]}
        add("eligibility_at_zero", indication, default=True)
        adult_group = next(
            (name for name in ("en_qualifier", "en_noun", "zh_adult") if population[name]),
            None,
        )
        if adult_group:
            criteria.append(
                {
                    "id": "c2",
                    "role": "include",
                    "kind": "age_years",
                    "quote": wording.quote(*population.span(adult_group)),
                    "source": "question",
                    "min_years": ADULT_AGE_FLOOR_YEARS,
                }
            )
            add("adult_minimum_age", str(ADULT_AGE_FLOOR_YEARS), default=True)
    else:
        missing.append("eligibility")

    confounders, adjustment_spans = _proposal_adjustments(study, wording)
    if confounders:
        covered.extend(adjustment_spans)
        add("adjustment", ", ".join(row["name"] for row in confounders))
        spec["confounders"] = confounders
    else:
        missing.append("adjustment")

    unread_words = wording.unread(covered)
    if unread_words:
        missing.append("unread_words")

    statement = None
    if not missing:
        try:
            statement = read_target_trial_statement(
                {"spec": spec, "population_spec": {"criteria": criteria}}
            )
        except TargetTrialSetupError:
            missing.append("protocol")
    return TargetTrialProposal(elements, missing, statement, unread_words)

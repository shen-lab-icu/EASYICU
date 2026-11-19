"""Read a pre-plan proposal from the question and registered trial contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

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
    _catalog_name_readings,
    _match_concept,
    concept_named_by,
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
_POPULATION_LINK = re.compile(r"\s*(?:(?:with|set|define)\s+|以\s*)?", re.IGNORECASE)
# Preserve the stop for explicit eligibility clauses outside the opening topic.
_SEPARATE_ELIGIBILITY = re.compile(
    r"\b(?:exclude|excluding|paediatric|pediatric|first\s+(?:icu\s+)?(?:stay|admission)|"
    r"readmission|include|including|aged|eligibility)\b|排除|不纳入|仅纳入|儿童|首次|再入|年龄[≥>≤<]",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[。；;？?！!]|\.(?=\s|$)")
_EN_ORIGIN_CUE = r"\b(?:from|after|since|following|post|starting\s+at|as\s+of)[\s\-‐–—]+"
_ORIGIN_CUE = re.compile(
    rf"{_EN_ORIGIN_CUE}|自|从|起算|为起点|[后後]",
    re.IGNORECASE,
)
_ICU_ADMISSION_ORIGIN = re.compile(
    rf"{_EN_ORIGIN_CUE}(?:the\s+)?ICU[\s\-‐–—]+admission\b(?!\s+(?:to|until|day|and|or)\b)"
    r"|(?:(?:自|从)\s*入(?:住)?\s*ICU(?:\s*(?:起算|算起|起|[后後]))?|入(?:住)?\s*ICU\s*[后後])"
    r"(?![a-z]|\s*(?:第|至|到|或))",
    re.IGNORECASE,
)
_ADJUSTMENT_TIMING = re.compile(
    r"\bafter\s+(?:adjusting|adjustment|controlling)\s+for\b"
    r"|(?<=[,，])\s*(?:调整|校正)[^,，。；?!？！自从后後]+[后後](?=\s*[,，])",
    re.IGNORECASE,
)
# An explicit continued-use or titration rule is outside the two fixed strategies.
_OTHER_STRATEGY = re.compile(
    r"\b(?:titrate|titrated|infusion rate|mg|mcg|mg/kg|continue treatment|never start|never starting)\b"
    r"|剂量|滴定|持续(?:给药|使用)|从未开始|始终不(?:用|使用)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TargetTrialProposal:
    elements: list[dict[str, str]]
    missing: list[str]
    statement: TargetTrialStatement | None

    def public(self, revision: int) -> dict[str, Any]:
        return {
            "elements": self.elements,
            "missing": self.missing,
            "expected_revision": revision,
            "digest": self.statement.request()["request_sha256"]
            if self.statement
            else None,
        }


def _indication_concept(words: str) -> str | None:
    # A name inside a longer phrase cannot stand in for that whole phrase.
    text = words.lower().replace("_", " ")
    concepts = {
        concept
        for concept, start, end in _catalog_name_readings(text)
        if start == 0 and end == len(text)
    } | {concept for concept, _ in _match_concept(text, full_span_only=True)}
    concept = next(iter(concepts)) if len(concepts) == 1 else None
    return concept if isinstance(concept, str) and concept_declares_event_status(concept) else None


def _local_clause(
    question: str, start: int, end: int, other_elements: tuple[tuple[int, int], ...]
) -> str:
    """The surrounding words up to the next trial element or sentence boundary."""

    boundaries = (*other_elements, *(match.span() for match in _SENTENCE_END.finditer(question)))
    left = max((finish for _, finish in boundaries if finish <= start), default=0)
    right = min((begin for begin, _ in boundaries if begin >= end), default=len(question))
    return question[left:right]


def _states_origin(clause: str, *, allow_icu_admission: bool = False) -> bool:
    clause = _ADJUSTMENT_TIMING.sub(" ", clause)
    if allow_icu_admission:
        clause = _ICU_ADMISSION_ORIGIN.sub(" ", clause)
    return _ORIGIN_CUE.search(clause) is not None


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
    concepts = question_treatment_concepts(question, reading)
    selected = [entry for entry in entries if entry.concept in concepts]
    agents = frozenset(agent for entry in selected for agent in entry.agents)
    exact_classes = [
        name
        for name in registry.treatment_classes
        if registry.class_agents(name) == agents
    ]
    if (
        set(concepts) != {entry.concept for entry in selected}
        or len(exact_classes) != 1
    ):
        return None
    return tuple(entry.concept for entry in selected), exact_classes[0]


def _proposal_adjustments(
    study: Mapping[str, Any], question: str
) -> list[dict[str, str]]:
    rows = study.get("stated_adjustments") or ()
    if any(
        isinstance(row, Mapping) and row.get("field") == "stated_adjustments"
        for row in study.get("stated_variables_unread") or ()
    ):
        return []
    if not rows or any(
        not isinstance(row, Mapping)
        or row.get("source_status") != "available"
        or not row.get("concept")
        or not row.get("words")
        or str(row["words"]) not in question
        for row in rows
    ):
        return []
    return [
        {
            "name": str(row["concept"]),
            "clinical_rationale": str(row["words"])
            if len(str(row["words"])) >= 16
            else "Baseline adjustment requested by the researcher.",
            "source": "question",
        }
        for row in rows
    ]


def _fixed_outcome(
    question: str,
    confirmed_label: Any,
    database: str,
    other_elements: tuple[tuple[int, int], ...] = (),
) -> tuple[FixedHorizonMortalityEndpoint, str] | None:
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
    selected = [item for candidate, item in mentions if candidate == endpoint]
    for item in selected:
        neighbours = (
            *other_elements,
            *((other.start, other.end) for _, other in mentions if other != item),
        )
        clause = _local_clause(question, item.start, item.end, neighbours)
        if _states_origin(clause, allow_icu_admission=endpoint.time_origin == "icu_admission"):
            return None
    match = selected[0]
    return endpoint, question[match.start : match.end]


def target_trial_proposal(study: Mapping[str, Any]) -> TargetTrialProposal | None:
    question = str(study.get("question") or "").strip()
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

    evidence = {item.element: item.evidence for item in reading.elements}
    elements: list[dict[str, str]] = []
    missing: list[str] = []

    def add(key: str, quote: str, *, default: bool = False) -> dict[str, str]:
        origin = "design_choice" if default else "question"
        elements.append({"key": key, "value": quote, "source": origin})
        return {"quote": quote[:160], "source": origin}

    spec: dict[str, Any] = {}
    treatment = _treatment(question, reading, database)
    if treatment and reading.treatment:
        start, end = reading.treatment
        spec["treatment"] = {
            **add("treatment", question[start:end]),
            "concepts": treatment[0],
            "treatment_class": treatment[1],
        }
        add("treatment_capture", ", ".join(treatment[0]), default=True)
    else:
        missing.append("treatment")

    initiate, defer = evidence.get("initiate_strategy"), evidence.get("defer_strategy")
    if initiate and defer and not _OTHER_STRATEGY.search(question):
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
    strategy_spans = {
        item.element: (item.start, item.end)
        for item in reading.elements
        if item.element in {"initiate_strategy", "defer_strategy"}
    }
    zero, grace = reading.time_zero_hours, reading.grace_period_hours
    zero_words = evidence.get("time_zero")
    if (
        zero_words
        and zero is not None
        and zero.is_integer()
        and MIN_TIME_ZERO_HOURS <= zero <= MAX_TIME_ZERO_HOURS
        and len(zero_spans) == 1
    ):
        spec["time_zero"] = {
            **add("time_zero", zero_words),
            "hours_after_icu_admission": int(zero),
        }
    else:
        missing.append("time_zero")
    initiation = strategy_spans.get("initiate_strategy")
    grace_context = zero_spans
    if population:
        grace_context += (population.span(),)
    if "defer_strategy" in strategy_spans:
        grace_context += (strategy_spans["defer_strategy"],)
    if (
        grace is not None
        and grace.is_integer()
        and 1 <= grace <= MAX_GRACE_PERIOD_HOURS
        and initiation
        # The executable window starts at trial time zero. Other wording needs review.
        and not _states_origin(_local_clause(question, *initiation, grace_context))
    ):
        spec["grace_period"] = {
            **add("grace_period", initiate),
            "hours": int(grace),
        }
    else:
        missing.append("grace_period")

    outcome = _fixed_outcome(
        question, study.get("outcome"), database, (*zero_spans, *strategy_spans.values())
    )
    if outcome:
        endpoint, words = outcome
        spec["outcome"] = {**add("outcome", words), "endpoint": endpoint.event_concept}
        add("outcome_origin", endpoint.time_origin, default=True)
    else:
        missing.append("outcome")

    indication = None
    if population and zero_spans and not _SEPARATE_ELIGIBILITY.search(question, population.end()):
        # A comma can introduce another population restriction. Account for
        # all text between the population and the start of the trial design.
        design_start = min(start for start, _ in (*zero_spans, *strategy_spans.values()))
        if design_start >= population.end() and _POPULATION_LINK.fullmatch(
            question[population.end():design_start]
        ):
            indication = _indication_concept((population["en"] or population["zh"]).strip())
    criteria = []
    if indication and "time_zero" in spec:
        start, end = population.span()
        population_words = question[start : end - 1].rstrip()
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
        adult = population["en_qualifier"] or population["en_noun"] or population["zh_adult"]
        if adult:
            criteria.append(
                {
                    "id": "c2",
                    "role": "include",
                    "kind": "age_years",
                    "quote": adult,
                    "source": "question",
                    "min_years": ADULT_AGE_FLOOR_YEARS,
                }
            )
            add("adult_minimum_age", str(ADULT_AGE_FLOOR_YEARS), default=True)
    else:
        missing.append("eligibility")

    confounders = _proposal_adjustments(study, question)
    if confounders:
        add("adjustment", ", ".join(row["name"] for row in confounders))
        spec["confounders"] = confounders
    else:
        missing.append("adjustment")

    statement = None
    if not missing:
        try:
            statement = read_target_trial_statement(
                {"spec": spec, "population_spec": {"criteria": criteria}}
            )
        except TargetTrialSetupError:
            missing.append("protocol")
    return TargetTrialProposal(elements, missing, statement)

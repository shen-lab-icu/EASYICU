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
from easyicu.webserver.study_intent import concept_named_by
from easyicu.webserver.target_trial_setup import (
    TargetTrialSetupError,
    TargetTrialStatement,
    read_target_trial_statement,
)

# Restrict the automatic population reading to one explicit clinical indication.
_INDICATION = re.compile(
    r"(?:patients?|adults?|stays)\s+with\s+(?P<en>[^,.;?]{2,100})(?=[,.;?])"
    r"|(?:在)?(?:成人|成年)?(?P<zh>[^，。；、？]{2,100})患者(?:中|人群)(?=[，。；？])",
    re.IGNORECASE,
)
_ADULTS = re.compile(r"\badults?\b|成人|成年", re.IGNORECASE)
_EXTRA_ELIGIBILITY = re.compile(
    r"\b(?:exclude|excluding|paediatric|pediatric|first\s+(?:icu\s+)?(?:stay|admission)|"
    r"readmission|include|including|aged|eligibility)\b|排除|不纳入|仅纳入|儿童|首次|再入|年龄[≥>≤<]",
    re.IGNORECASE,
)
_COMPOUND_INDICATION = re.compile(
    r"\b(?:and|or|without|who|before|after)\b|以及|伴有|合并|或|并且|无", re.I
)
_OUTCOME_OTHER_ORIGIN = re.compile(
    r"^\s*(?:from|after|since)\s+(?:time[- ]?zero|treatment|hospital admission)\b"
    r"|^\s*(?:自|从)(?:时间零点|开始治疗|入院)",
    re.IGNORECASE,
)
_GRACE_OTHER_ORIGIN = re.compile(
    r"^\s*(?:from|after|since)\s+(?:(?:icu|hospital)\s+)?admission\b"
    r"|^\s*(?:自|从)入(?:住\s*ICU|\s*ICU|院)",
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
    if _COMPOUND_INDICATION.search(words):
        return None
    concept = concept_named_by(words, include_catalog_labels=True)
    return concept if concept and concept_declares_event_status(concept) else None


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
    question: str, confirmed_label: Any, database: str
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
    if any(_OUTCOME_OTHER_ORIGIN.match(question[item.end :]) for item in selected):
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
    treatment_words = None
    if reading.treatment:
        start, end = reading.treatment
        treatment_words = add("treatment", question[start:end])
    if treatment and treatment_words:
        spec["treatment"] = {
            **treatment_words,
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

    zero, grace = reading.time_zero_hours, reading.grace_period_hours
    zero_words = evidence.get("time_zero")
    if (
        zero_words
        and zero is not None
        and zero.is_integer()
        and MIN_TIME_ZERO_HOURS <= zero <= MAX_TIME_ZERO_HOURS
        and len(time_zero_spans(question)) == 1
    ):
        spec["time_zero"] = {
            **add("time_zero", zero_words),
            "hours_after_icu_admission": int(zero),
        }
    else:
        missing.append("time_zero")
    initiation = next(
        (item for item in reading.elements if item.element == "initiate_strategy"), None
    )
    if (
        grace is not None
        and grace.is_integer()
        and 1 <= grace <= MAX_GRACE_PERIOD_HOURS
        and initiation is not None
        and not _GRACE_OTHER_ORIGIN.match(question[initiation.end :])
    ):
        spec["grace_period"] = {
            **add("grace_period", initiation.evidence),
            "hours": int(grace),
        }
    else:
        missing.append("grace_period")

    outcome = _fixed_outcome(question, study.get("outcome"), database)
    if outcome:
        endpoint, words = outcome
        spec["outcome"] = {**add("outcome", words), "endpoint": endpoint.event_concept}
        add("outcome_origin", endpoint.time_origin, default=True)
    else:
        missing.append("outcome")

    indication_matches = list(_INDICATION.finditer(question))
    match = indication_matches[0] if len(indication_matches) == 1 else None
    words = (match["en"] or match["zh"]).strip() if match else ""
    indication = _indication_concept(words) if words else None
    criteria = []
    if (
        indication
        and zero is not None
        and "time_zero" in spec
        and not _EXTRA_ELIGIBILITY.search(question)
    ):
        criteria.append(
            {
                "id": "c1",
                "role": "include",
                "kind": "condition_present",
                "quote": match[0],
                "source": "question",
                "concepts_all_of": [indication],
                "window": {"start_hours": 0, "end_hours": int(zero)},
            }
        )
        spec["indication"] = {**add("eligibility", match[0]), "criterion_ids": ["c1"]}
        add("eligibility_at_zero", indication, default=True)
        adult = _ADULTS.search(question)
        if adult is not None:
            criteria.append(
                {
                    "id": "c2",
                    "role": "include",
                    "kind": "age_years",
                    "quote": adult[0] if len(adult[0]) > 1 else match[0],
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

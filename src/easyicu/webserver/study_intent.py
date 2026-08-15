"""Typed study-intent extraction for the Copilot front door.

Owner of ONE responsibility: turning a user's own sentence into a typed,
closed-set ``StudyContract`` proposal, plus an explicit list of the slots it
could **not** read.

Why this module exists
----------------------
The conversational front door used to infer intent with three regexes over the
question text and then fill every remaining slot from module-level defaults
(``exposure='lactate'``, ``outcome='In-hospital mortality'``, population pinned
to ``Sepsis-3``). A question about fluid balance and AKI therefore became a
question about lactate and mortality, and that substituted string was what got
submitted, persisted and bound to evidence.

The contract here is deliberately narrow:

* Slots come from **closed sets** or from the project's own concept catalog.
* A slot that cannot be read from the user's words stays ``None`` and is named
  in ``unread``. Nothing is ever filled from a default. A caller that wants a
  value must ask the user.
* The LLM path is optional and gated. When it is unavailable, refused, or
  returns anything that fails validation, the deterministic reader answers and
  the reason is reported. The extractor never invents to stay useful.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from easyicu.ai_optin import AIOptInError
from easyicu.webserver import provider_adapter
from easyicu.webserver.provider_gate import ProviderGateError, resolve_provider_gate

__all__ = [
    "ANALYSIS_FAMILIES",
    "OUTCOME_TYPES",
    "StudyIntentError",
    "extract_study_intent",
    "deterministic_intent",
    "SLOTS",
]

# --------------------------------------------------------------------------
# Closed sets. A value outside these is rejected, never coerced.
# --------------------------------------------------------------------------
ANALYSIS_FAMILIES: Tuple[str, ...] = (
    "description",
    "association",
    "prediction",
    "survival",
    "causal",
    "trajectory",
    "cross_database",
    "data_quality",
)
OUTCOME_TYPES: Tuple[str, ...] = (
    "binary",
    "continuous",
    "time_to_event",
    "ordinal",
    "count",
)
SLOTS: Tuple[str, ...] = (
    "population",
    "exposure",
    "outcome",
    "outcome_type",
    "time_window_hours",
    "comparator",
    "analysis_family",
)

_MAX_QUESTION_CHARS = 1200
_MAX_SLOT_CHARS = 160


class StudyIntentError(ValueError):
    """Raised when the request itself is unusable (not when a slot is unread)."""

    def __init__(self, detail: Dict[str, Any]) -> None:
        super().__init__(str(detail.get("error") or "study_intent_error"))
        self.detail = detail


# --------------------------------------------------------------------------
# Vocabulary, grounded in the project's own concept catalog.
# --------------------------------------------------------------------------
def _concept_groups() -> Dict[str, List[str]]:
    try:
        from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL

        return {str(k): [str(c) for c in v] for k, v in CONCEPT_GROUPS_INTERNAL.items()}
    except Exception:  # pragma: no cover - catalog is optional at import time
        return {}


# Clinical phrasings (EN + ZH) mapped onto catalog concept ids. This is a
# reading aid, not an allowlist of what a study may be about: an unmatched
# phrase yields an unread slot, never a default.
_PHRASE_TO_CONCEPT: Tuple[Tuple[str, str], ...] = (
    (r"lactate|乳酸", "lact"),
    (r"\bsofa-?2\b|sofa2", "sofa2"),
    (r"\bsofa\b", "sofa"),
    (r"\bqsofa\b", "qsofa"),
    (r"\bsaps\s*3\b|saps3", "saps3"),
    (r"\bapache\b", "apache_iv"),
    (r"charlson|查尔森", "charlson"),
    (r"creatinine|肌酐", "crea"),
    (r"\bbun\b|尿素氮", "bun"),
    (r"bilirubin|胆红素", "bili"),
    (r"platelet|血小板", "plt"),
    (r"albumin|白蛋白", "alb"),
    (r"h(ae|e)moglobin|血红蛋白", "hgb"),
    (r"\bwbc\b|white cell|白细胞", "wbc"),
    (r"\bph\b|酸碱", "ph"),
    (r"base excess|碱剩余", "be"),
    (r"heart rate|心率", "hr"),
    (r"\bmap\b|mean arterial|平均动脉压", "map"),
    (r"blood pressure|血压", "sbp"),
    (r"temperature|体温|发热", "temp"),
    (r"spo2|血氧饱和度", "spo2"),
    (r"pa[o/]?2\s*/\s*fio2|p/f ratio|pafi|氧合指数", "pafi"),
    (r"fio2", "fio2"),
    (r"\bpeep\b", "peep"),
    (r"tidal volume|潮气量", "tidal_vol"),
    (r"mechanical ventilation|ventilat|机械通气|插管", "vent_ind"),
    (r"norepinephrine|noradrenaline|去甲肾上腺素", "norepi_rate"),
    (r"vasopressor|升压药|血管活性药", "norepi_equiv"),
    (r"antibiotic|抗生素|抗菌药", "abx"),
    (r"corticosteroid|steroid|激素|糖皮质", "cort"),
    (r"fluid balance|液体平衡|液体正平衡|入出量|液体复苏", "fluid_balance"),
    (r"urine output|尿量", "urine"),
    (r"\brrt\b|renal replacement|dialysis|透析|肾脏替代", "rrt"),
    # Specific before general: "KDIGO AKI stage" is aki_stage, not aki.
    (r"kdigo|aki stage|aki 分期|肾损伤分期", "aki_stage"),
    (r"\baki\b|acute kidney|急性肾损伤|肾损伤", "aki"),
    (r"\bgcs\b|glasgow|昏迷评分", "gcs"),
    (r"\brass\b|镇静评分", "rass"),
    (r"delirium|谵妄", "gcs"),
    (r"sofa-?2.{0,20}(?:sepsis|脓毒症)|(?:sepsis|脓毒症).{0,20}sofa-?2", "sep3_sofa2"),
    (r"sepsis-?3|脓毒症|sepsis", "sep3"),
    (r"suspected infection|疑似感染", "susp_inf"),
    (r"circulatory failure|循环衰竭|休克", "circ_failure"),
    (r"\bbmi\b|体重指数", "bmi"),
    (r"\bage\b|年龄", "age"),
    (r"\bsex\b|gender|性别", "sex"),
    (r"in-?hospital mortality|hospital mortality|院内死亡|住院death|住院死亡", "death"),
    (r"28-?\s*day mortality|28\s*天死亡", "mort_28d"),
    (r"90-?\s*day mortality|90\s*天死亡", "mort_90d"),
    (r"icu mortality|icu 死亡", "death"),
    (r"mortality|death|死亡|病死", "death"),
    (r"length of stay|\blos\b|住院时长|住院时间|icu 时长", "los_icu"),
    (r"readmission|再入院", "icu_readmission"),
    (r"ventilator-?free|无呼吸机天数", "vent_free_days_28"),
)

# Outcome candidates in two tiers. Tier 1 is the catalog's own ``outcome``
# group: these are almost never the cohort. Tier 2 are clinical events that are
# just as often the population or the exposure ("in AKI patients", "AKI stage
# vs LoS"), so they only become the outcome when no tier-1 phrase is present.
# Sepsis-3 is deliberately in NEITHER: in this corpus it is the cohort, and
# guessing it as an outcome is exactly the substitution this module exists to
# stop.
_OUTCOME_CONCEPTS_PRIMARY = frozenset(
    {
        "death",
        "mort_28d",
        "mort_90d",
        "mort_365d",
        "los_icu",
        "los_hosp",
        "icu_free_days_28",
        "vent_free_days_28",
        "icu_readmission",
    }
)
_OUTCOME_CONCEPTS_EVENT = frozenset({"aki", "aki_stage", "rrt", "circ_failure", "vent_ind"})
_OUTCOME_CONCEPTS = _OUTCOME_CONCEPTS_PRIMARY | _OUTCOME_CONCEPTS_EVENT
# A population is only read when the sentence actually names a population.
# Without this, "ICU length of stay" would silently become "ICU patients".
# NOTE: plural "stays" only. "length of stay" is an outcome, not a cohort.
_POPULATION_NOUN = re.compile(r"patients?|adults?|\bstays\b|cohort|subjects?|患者|人群|病人", re.IGNORECASE)
_TIME_TO_EVENT_CONCEPTS = frozenset({"los_icu", "los_hosp"})
_ORDINAL_CONCEPTS = frozenset({"aki_stage"})
_COUNT_CONCEPTS = frozenset({"icu_free_days_28", "vent_free_days_28"})

# (pattern, label, concepts this cohort is defined by). The third element lets
# a disease phrase be dropped as the cohort when that same disease is already
# serving as the outcome — "AKI stage vs LoS" has an AKI outcome, not an AKI
# cohort.
_POPULATION_PATTERNS: Tuple[Tuple[str, str, frozenset], ...] = (
    (r"sepsis-?3|脓毒症|septic", "Sepsis-3 patients", frozenset({"sep3", "susp_inf"})),
    (r"\baki\b|acute kidney|急性肾损伤|kdigo", "Patients with AKI", frozenset({"aki", "aki_stage", "rrt"})),
    (r"ventilated|mechanical ventilation|机械通气", "Mechanically ventilated patients", frozenset({"vent_ind", "vent_free_days_28"})),
    (r"\bards\b", "Patients with ARDS", frozenset()),
    (r"cardiac surgery|心脏外科|心脏手术", "Cardiac surgery patients", frozenset()),
    (r"\bcovid", "COVID-19 patients", frozenset()),
    (r"adults?|成年|成人", "Adult ICU patients", frozenset()),
    (r"\bicu\b|重症|监护", "ICU patients", frozenset()),
)

_FAMILY_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"across\s+(?:\w+\s+){0,2}(databases?|cohorts?|centres?|centers?|sites?)|cross-?database|多个数据库|跨库|external validation", "cross_database"),
    (r"missing|coverage|data quality|数据质量|缺失|覆盖率", "data_quality"),
    (r"predict|prognostic|risk score|auroc|discrimination|预测|预后模型", "prediction"),
    (r"causal|confound|treatment effect|因果|混杂|倾向性评分|propensity", "causal"),
    (r"survival|time-?to-?event|hazard|cox|生存|风险比", "survival"),
    (r"trajectory|over time|longitudinal|轨迹|随时间|纵向", "trajectory"),
    (r"associat|correlat|relationship|related to|相关|关联", "association"),
    (r"describe|distribution|prevalence|characteris|描述|分布|患病率", "description"),
)


# "I am NOT studying death, my outcome is AKI" must not read `death`. Without
# this, a user's correction becomes the very thing they corrected.
_NEGATION = re.compile(
    r"(?:\bnot\b|\bno\b|\bnever\b|\bisn't\b|\baren't\b|\bdon't\b|\bdoesn't\b|\brather than\b|\binstead of\b|不是|不要|不想|并非|而非|非|无关|别)"
    r"[\s\S]{0,16}$",
    re.IGNORECASE,
)
_NEGATION_LOOKBACK = 26


def _negated(text: str, start: int) -> bool:
    """True when a negation marker governs the match.

    The window is generous enough for "I am not studying mortality" and
    "我不是要研究死亡", and stops at sentence boundaries so a negation in one
    clause does not suppress a reading in the next.
    """
    window = text[max(0, start - _NEGATION_LOOKBACK) : start]
    # A sentence break ends a negation's scope.
    for sep in (". ", "; ", "。", "；", "?", "？"):
        if sep in window:
            window = window.rsplit(sep, 1)[1]
    return bool(_NEGATION.search(window))


# Concepts that name the same clinical thing at different granularity. Used to
# stop one phrase from filling two different slots.
_CONCEPT_FAMILIES: Tuple[frozenset, ...] = (
    frozenset({"aki", "aki_stage", "rrt"}),
    frozenset({"death", "mort_28d", "mort_90d", "mort_365d"}),
    frozenset({"los_icu", "los_hosp"}),
    frozenset({"vent_ind", "vent_free_days_28", "peep", "tidal_vol"}),
    frozenset({"sep3", "sep3_sofa2", "susp_inf"}),
)


def _family_of(concept: Optional[str]) -> Optional[frozenset]:
    if not concept:
        return None
    for family in _CONCEPT_FAMILIES:
        if concept in family:
            return family
    return None


def _match_concept(text: str) -> List[Tuple[str, str]]:
    """Return (concept_id, matched_phrase) pairs in the order they appear.

    A phrase the sentence explicitly negates is not a reading — it is skipped,
    which leaves the slot unread rather than wrong.
    """
    found: List[Tuple[str, str]] = []
    seen = set()
    for pattern, concept in _PHRASE_TO_CONCEPT:
        if concept in seen:
            continue
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if _negated(text, match.start()):
                continue
            seen.add(concept)
            found.append((concept, match.group(0)))
            break
    return found


def _clean_question(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise StudyIntentError({"error": "study_intent_question_required"})
    if len(text) > _MAX_QUESTION_CHARS:
        raise StudyIntentError(
            {
                "error": "study_intent_question_too_long",
                "max_chars": _MAX_QUESTION_CHARS,
            }
        )
    return text


def _slot(value: Any, provenance: str, evidence: Optional[str] = None) -> Dict[str, Any]:
    return {
        "value": value,
        "provenance": provenance,
        "evidence": (str(evidence)[:_MAX_SLOT_CHARS] if evidence else None),
    }


def _empty_slot() -> Dict[str, Any]:
    return {"value": None, "provenance": "unread", "evidence": None}


# --------------------------------------------------------------------------
# Deterministic reader (always available, offline, no provider)
# --------------------------------------------------------------------------
def deterministic_intent(question: str) -> Dict[str, Any]:
    """Read what the sentence actually says. Leave the rest unread."""
    text = _clean_question(question)
    lowered = text.lower()
    slots: Dict[str, Dict[str, Any]] = {name: _empty_slot() for name in SLOTS}

    concepts = _match_concept(lowered)
    primary = [(c, p) for c, p in concepts if c in _OUTCOME_CONCEPTS_PRIMARY]
    events = [(c, p) for c, p in concepts if c in _OUTCOME_CONCEPTS_EVENT]

    outcome_concept: Optional[str] = None
    outcome_phrase: Optional[str] = None
    if primary:
        outcome_concept, outcome_phrase = primary[0]
    elif events:
        outcome_concept, outcome_phrase = events[0]

    if outcome_concept:
        slots["outcome"] = _slot(outcome_concept, "user_text", outcome_phrase)
        if outcome_concept in _TIME_TO_EVENT_CONCEPTS:
            kind = "time_to_event"
        elif outcome_concept in _ORDINAL_CONCEPTS:
            kind = "ordinal"
        elif outcome_concept in _COUNT_CONCEPTS:
            kind = "count"
        else:
            kind = "binary"
        slots["outcome_type"] = _slot(kind, "user_text", outcome_phrase)

    # Everything else the sentence names is an exposure candidate — including a
    # tier-2 event concept that did not win the outcome slot. A concept from the
    # SAME clinical family as the outcome is not an exposure though: "my outcome
    # is AKI (KDIGO stage)" names one thing twice, not an exposure and an
    # outcome. Leaving it unread is what makes the card ask.
    outcome_family = _family_of(outcome_concept)
    exposures = [
        (c, p)
        for c, p in concepts
        if c != outcome_concept and not (outcome_family and _family_of(c) == outcome_family)
    ]
    if exposures:
        concept, phrase = exposures[0]
        slots["exposure"] = _slot(concept, "user_text", phrase)

    if _POPULATION_NOUN.search(text):
        for pattern, label, family in _POPULATION_PATTERNS:
            match = re.search(pattern, lowered, re.IGNORECASE)
            if not match or _negated(lowered, match.start()):
                continue
            # A disease already serving as the outcome is not also the cohort
            # ("...与急性肾损伤的风险相关" is an outcome, not a population).
            if outcome_concept and outcome_concept in family:
                continue
            slots["population"] = _slot(label, "user_text", match.group(0))
            break

    window = re.search(r"(?:first\s*)?(\d{1,3})\s*(?:h\b|hr|hour|小时)", lowered)
    if window:
        slots["time_window_hours"] = _slot(
            int(window.group(1)), "user_text", window.group(0)
        )
    elif re.search(r"\b24\s*h|首日|第一天|first day", lowered):
        slots["time_window_hours"] = _slot(24, "user_text", "first day")

    for pattern, family in _FAMILY_PATTERNS:
        for match in re.finditer(pattern, lowered, re.IGNORECASE):
            # "Not a prediction study; is PEEP associated with ..." is an
            # association study. The same negation rule applies here.
            if _negated(lowered, match.start()):
                continue
            slots["analysis_family"] = _slot(family, "user_text", match.group(0))
            break
        if slots["analysis_family"]["value"]:
            break

    return _finalize(question=text, slots=slots, source="deterministic", notes=[])


def _finalize(
    *,
    question: str,
    slots: Dict[str, Dict[str, Any]],
    source: str,
    notes: List[str],
) -> Dict[str, Any]:
    unread = [name for name in SLOTS if slots[name]["value"] in (None, "")]
    return {
        "ok": True,
        "question": question,
        "slots": {name: slots[name] for name in SLOTS},
        "unread": unread,
        "read_count": len(SLOTS) - len(unread),
        "slot_count": len(SLOTS),
        "source": source,
        "notes": notes,
        # A contract is only runnable once the user has supplied or confirmed
        # everything. This flag exists so no caller can mistake a partial read
        # for a ready study.
        "complete": not unread,
    }


# --------------------------------------------------------------------------
# LLM reader (optional, gated, validated)
# --------------------------------------------------------------------------
_LLM_SYSTEM = (
    "You extract a structured study contract from one ICU research question. "
    "Return STRICT JSON only, no prose. Every field must be present. Use null "
    "for anything the question does not state — never guess, never substitute a "
    "more common study. Do not add fields."
)


def _llm_user_prompt(question: str) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Return JSON with exactly these keys:\n"
        '{"population": string|null, "exposure": string|null, '
        '"outcome": string|null, "outcome_type": one of '
        f"{list(OUTCOME_TYPES)}|null, "
        '"time_window_hours": integer|null, "comparator": string|null, '
        f'"analysis_family": one of {list(ANALYSIS_FAMILIES)}|null'
        "}\n"
        "Rules: population/exposure/outcome are short clinical phrases taken "
        "from the question. If the question names no comparator, return null - "
        "do not invent one. If it does not state a time window, return null."
    )


def _validate_llm_slots(payload: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise StudyIntentError({"error": "study_intent_llm_payload_not_object"})
    unknown = sorted(set(payload) - set(SLOTS))
    if unknown:
        raise StudyIntentError(
            {"error": "study_intent_llm_unknown_fields", "fields": unknown}
        )
    slots: Dict[str, Dict[str, Any]] = {name: _empty_slot() for name in SLOTS}
    for name in SLOTS:
        raw = payload.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        if name == "analysis_family":
            text = str(raw).strip().lower()
            if text not in ANALYSIS_FAMILIES:
                raise StudyIntentError(
                    {"error": "study_intent_llm_bad_family", "value": text}
                )
            slots[name] = _slot(text, "llm")
        elif name == "outcome_type":
            text = str(raw).strip().lower()
            if text not in OUTCOME_TYPES:
                raise StudyIntentError(
                    {"error": "study_intent_llm_bad_outcome_type", "value": text}
                )
            slots[name] = _slot(text, "llm")
        elif name == "time_window_hours":
            try:
                hours = int(raw)
            except (TypeError, ValueError) as exc:
                raise StudyIntentError(
                    {"error": "study_intent_llm_bad_window", "value": str(raw)[:40]}
                ) from exc
            if not 1 <= hours <= 24 * 365:
                raise StudyIntentError(
                    {"error": "study_intent_llm_window_out_of_range", "value": hours}
                )
            slots[name] = _slot(hours, "llm")
        else:
            text = str(raw).strip()[:_MAX_SLOT_CHARS]
            if text:
                slots[name] = _slot(text, "llm")
    return slots


def _llm_intent(
    question: str,
    *,
    provider_meta: Dict[str, Any],
    transport: Optional[Callable[[Dict[str, Any], Dict[str, str]], Dict[str, Any]]],
    environ: Optional[Mapping[str, str]],
) -> Dict[str, Dict[str, Any]]:
    credentials = provider_adapter._load_external_credentials(  # noqa: SLF001
        str(provider_meta.get("provider") or ""), environ=environ
    )
    request = {
        "model": credentials["model"],
        "temperature": 0,
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": _llm_user_prompt(question)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {credentials['api_key']}",
        "Content-Type": "application/json",
    }
    if transport is None:
        response = provider_adapter._post_chat_completion(  # noqa: SLF001
            url=credentials["base_url"],
            request=request,
            headers=headers,
            timeout=30,
        )
    else:
        response = transport(request, headers)
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise StudyIntentError({"error": "study_intent_llm_response_malformed"}) from exc
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StudyIntentError({"error": "study_intent_llm_not_json"}) from exc
    return _validate_llm_slots(payload)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def extract_study_intent(
    question: Any,
    *,
    llm_provider: str = "offline",
    external_llm_opt_in: bool = False,
    ai_enabled: bool = False,
    language: str = "en",
    transport: Optional[Callable[[Dict[str, Any], Dict[str, str]], Dict[str, Any]]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return a typed study-contract proposal for the user's own question.

    The deterministic reader always runs. An external provider is consulted
    only when the canonical AI opt-in gate allows it, and only its *validated*
    output is used; any refusal or malformed answer falls back to the
    deterministic result with the reason recorded in ``notes``.
    """
    text = _clean_question(question)
    baseline = deterministic_intent(text)

    provider_text = str(llm_provider or "offline").strip().lower() or "offline"
    if provider_text in {"offline", "mock", "none", ""}:
        baseline["notes"].append("llm_not_requested")
        return baseline

    try:
        provider_meta = resolve_provider_gate(
            # Intent extraction really does leave the machine, so it is gated
            # as a full external call rather than as a local preflight.
            run_type="full",
            llm_provider=provider_text,
            external_llm_opt_in=external_llm_opt_in,
            ai_enabled=ai_enabled,
            language=language,
        )
    except (ProviderGateError, AIOptInError) as exc:
        baseline["notes"].append("llm_blocked_by_opt_in_gate")
        detail = getattr(exc, "detail", None)
        baseline["provider_block"] = (
            {k: v for k, v in detail.items() if k != "message"}
            if isinstance(detail, dict)
            else {"error": "external_llm_opt_in_required"}
        )
        return baseline

    try:
        slots = _llm_intent(
            text,
            provider_meta=provider_meta,
            transport=transport,
            environ=environ,
        )
    except (StudyIntentError, provider_adapter.ProviderAdapterError) as exc:
        detail = getattr(exc, "detail", {}) or {}
        baseline["notes"].append(
            f"llm_rejected:{detail.get('error') or 'study_intent_llm_failed'}"
        )
        return baseline

    # A deterministic read is grounded directly in the user's own wording.
    # The optional model may fill only unread slots; it may not reinterpret or
    # overwrite a slot that the deterministic reader has already established.
    for name in SLOTS:
        if baseline["slots"][name]["value"] not in (None, ""):
            slots[name] = baseline["slots"][name]

    result = _finalize(question=text, slots=slots, source="llm", notes=[])
    result["provider"] = {
        "provider": provider_meta.get("provider"),
        "external": provider_meta.get("external"),
        "provider_gate": provider_meta.get("provider_gate"),
    }
    return result

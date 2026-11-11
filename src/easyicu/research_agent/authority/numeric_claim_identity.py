"""Typed identity inference for manuscript-bindable numeric claims."""

from __future__ import annotations

import enum
import functools
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple


class NumericEffectScale(str, enum.Enum):
    """Effect-measure identity carried by a manuscript-bindable number."""

    ODDS_RATIO = "odds_ratio"
    HAZARD_RATIO = "hazard_ratio"
    RISK_RATIO = "risk_ratio"


class NumericEstimand(str, enum.Enum):
    """A number's role within an effect estimate and its interval."""

    POINT_ESTIMATE = "point_estimate"
    CONFIDENCE_INTERVAL_LOWER = "confidence_interval_lower"
    CONFIDENCE_INTERVAL_UPPER = "confidence_interval_upper"


def coerce_numeric_effect_scale(value: Any) -> Optional[NumericEffectScale]:
    if value is None or value == "":
        return None
    if isinstance(value, NumericEffectScale):
        return value
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    aliases = {
        "or": NumericEffectScale.ODDS_RATIO,
        "odds_ratio": NumericEffectScale.ODDS_RATIO,
        "adjusted_odds_ratio": NumericEffectScale.ODDS_RATIO,
        "hr": NumericEffectScale.HAZARD_RATIO,
        "hazard_ratio": NumericEffectScale.HAZARD_RATIO,
        "adjusted_hazard_ratio": NumericEffectScale.HAZARD_RATIO,
        "rr": NumericEffectScale.RISK_RATIO,
        "risk_ratio": NumericEffectScale.RISK_RATIO,
        "relative_risk": NumericEffectScale.RISK_RATIO,
        "adjusted_risk_ratio": NumericEffectScale.RISK_RATIO,
    }
    return aliases.get(normalized)


def infer_numeric_claim_identity(
    source_field: str,
    *,
    declared_effect_scale: Any = None,
) -> Tuple[Optional[NumericEffectScale], Optional[NumericEstimand]]:
    """Infer only identities that are explicit in a source coordinate."""

    source = str(source_field or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", source).strip("_")
    tokens = set(normalized.split("_"))
    declared_scale = coerce_numeric_effect_scale(declared_effect_scale)

    scale: Optional[NumericEffectScale] = None
    if "odds_ratio" in normalized or "or" in tokens:
        scale = NumericEffectScale.ODDS_RATIO
    elif "hazard_ratio" in normalized or "hr" in tokens:
        scale = NumericEffectScale.HAZARD_RATIO
    elif (
        "risk_ratio" in normalized or "relative_risk" in normalized or "rr" in tokens
    ):
        scale = NumericEffectScale.RISK_RATIO

    excluded_tokens = {"se", "stderr", "std", "error", "variance", "p", "value"}
    if tokens & excluded_tokens and not ({"ci", "interval"} & tokens):
        return None, None
    if scale is not None and declared_scale is not None and scale is not declared_scale:
        raise ValueError(
            "declared effect scale conflicts with source-field identity: "
            f"{declared_scale.value!r} != {scale.value!r} for {source_field!r}"
        )
    # Regression coefficients and explicitly logged ratios are on a transformed
    # coordinate. They must be exponentiated by an owning producer before they
    # can be published as OR/HR/RR claims.
    transformed_tokens = {"log", "ln", "logit", "coefficient", "coef", "beta"}
    if scale is not None and tokens & transformed_tokens:
        return None, None

    lower = bool(
        re.search(r"(?:ci|interval)_(?:95_)?(?:low|lower|lcl)(?:_|$)", normalized)
        or re.search(
            r"(?:^|_)(?:ci|interval)(?:_[a-z0-9]+)*_0$",
            normalized,
        )
    )
    upper = bool(
        re.search(r"(?:ci|interval)_(?:95_)?(?:high|upper|ucl)(?:_|$)", normalized)
        or re.search(
            r"(?:^|_)(?:ci|interval)(?:_[a-z0-9]+)*_1$",
            normalized,
        )
    )
    if lower:
        estimand = NumericEstimand.CONFIDENCE_INTERVAL_LOWER
    elif upper:
        estimand = NumericEstimand.CONFIDENCE_INTERVAL_UPPER
    else:
        leaf = re.split(r"[.\[]", source)[-1]
        point_names = {
            "adjusted_effect",
            "effect_estimate",
            "estimate",
            "odds_ratio",
            "hazard_ratio",
            "risk_ratio",
            "relative_risk",
        }
        estimand = (
            NumericEstimand.POINT_ESTIMATE
            if scale is not None or leaf in point_names
            else None
        )

    if estimand is not None and scale is None:
        scale = declared_scale
    return scale, estimand


@dataclass
class NumericClaim:
    """One manuscript-bindable numeric leaf and its source identity."""

    value: str
    canonical: float
    evidence_id: str
    step_id: str
    source_field: str
    tolerance: float = 1e-3
    effect_scale: Optional[NumericEffectScale] = None
    estimand: Optional[NumericEstimand] = None
    formula: Optional[str] = None
    explanation: Optional[str] = None
    derived_from: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        declared_scale = coerce_numeric_effect_scale(self.effect_scale)
        inferred_scale, inferred_estimand = infer_numeric_claim_identity(
            self.source_field,
            declared_effect_scale=declared_scale,
        )
        self.estimand = (
            self.estimand
            if isinstance(self.estimand, NumericEstimand)
            else NumericEstimand(self.estimand)
            if self.estimand
            else inferred_estimand
        )
        normalized_source = re.sub(
            r"[^a-z0-9]+", "_", str(self.source_field or "").strip().lower()
        ).strip("_")
        transformed_ratio = bool(
            {"log", "ln", "logit", "coefficient", "coef", "beta"}
            & set(normalized_source.split("_"))
            and (
                "odds_ratio" in normalized_source
                or "hazard_ratio" in normalized_source
                or "risk_ratio" in normalized_source
                or "relative_risk" in normalized_source
                or {"or", "hr", "rr"} & set(normalized_source.split("_"))
            )
        )
        self.effect_scale = (
            None
            if transformed_ratio
            else inferred_scale
            or (declared_scale if self.estimand is not None else None)
        )

    @property
    def is_derived(self) -> bool:
        return self.formula is not None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.effect_scale is None:
            payload.pop("effect_scale", None)
        else:
            payload["effect_scale"] = self.effect_scale.value
        if self.estimand is None:
            payload.pop("estimand", None)
        else:
            payload["estimand"] = self.estimand.value
        if not self.is_derived:
            payload.pop("formula", None)
            payload.pop("explanation", None)
            payload.pop("derived_from", None)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NumericClaim":
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        if "derived_from" in clean and clean["derived_from"]:
            clean["derived_from"] = [
                tuple(pair) if not isinstance(pair, tuple) else pair
                for pair in clean["derived_from"]
            ]
        return cls(**clean)


# ---------------------------------------------------------------------------
# Quantity kind: what a number counts or measures
# ---------------------------------------------------------------------------
#
# Equal value is not identity. A run that registered ``median_age=60`` and
# ``n_deaths=12`` let "the cohort included 60 patients" bind to the median age
# and "the median age was 12.0 years" bind to the death count: once the cited
# evidence left one equal-valued candidate, it was accepted with a footnote.
# The effect-scale identity above already refuses an OR printed as an HR; this
# generalises the same idea to the descriptive quantities a cohort reports.
#
# A kind is inferred only when the words make it explicit. An unknown kind on
# either side stays permissive, so a field or sentence the vocabulary does not
# cover keeps binding exactly as before; only a known-versus-known mismatch is
# refused.


class NumericQuantityKind(str, enum.Enum):
    """The kind of quantity a manuscript number or a registered field reports."""

    COUNT = "count"
    AGE = "age"
    DURATION = "duration"
    PROPORTION = "proportion"


#: Time units that make a short integer a measured quantity ("7 days").
_PROSE_TIME_UNIT_PATTERN = (
    r"(?:years?|yrs?|days?|d|hours?|hrs?|h|minutes?|mins?|weeks?|wks?"
    r"|months?|mos?)(?![\w-])"
)
_PROSE_ZH_TIME_UNIT_PATTERN = r"(?:岁|天|小时|周|个月|月|分钟)"
#: A unit used as the adjective of an endpoint names a horizon, exactly like
#: the hyphenated "28-day mortality" the recogniser never bound.
_PROSE_HORIZON_NOUN_PATTERN = (
    r"\s+(?:all[-\s]cause\s+|in[-\s]hospital\s+|icu\s+)?"
    r"(?:mortality|survival|death|readmission|outcome|endpoint|horizon"
    r"|window|landmark)"
)
_PROSE_ZH_HORIZON_PATTERN = (
    r"(?:内|以内|之内|后|以后|之后|前|以前|之前|死亡率|病死率|生存率)"
)
#: Statistic words that make the next short integer a reported value.
_PROSE_STATISTIC_RE = re.compile(
    r"(?:\b(?:median|mean|average|aged?|ages|duration|LOS|IQR|SD"
    r"|interquartile\s+range|standard\s+deviation"
    r"|length\s+of\s+(?:icu\s+|hospital\s+)?stay)\b"
    r"|中位数|中位|均值|平均|年龄|住院时间|住院天数|四分位|标准差)",
    re.I,
)
#: ``n = 42`` / ``N=42`` / ``n: 42`` in any spacing or case.
_PROSE_N_EQUALS_BEFORE_RE = re.compile(r"(?:^|(?<![A-Za-z_]))[nN]\s*[=:]\s*$")
#: Words that make the next integer a label, ordinal or identifier rather
#: than a quantity ("day 28", "Section 2", "stage 4", "第28天").
_PROSE_LABEL_BEFORE_RE = re.compile(
    r"(?:\b(?:arm|class|cohort|day|grade|group|level|phase|quartile|quintile"
    r"|stage|step|tercile|tier|type|week|year|section|figure|fig\.?|table"
    r"|appendix|supplement|chapter|panel|model|version|wave|visit|cycle"
    r"|round|item|criterion|scenario|landmark)\s*"
    r"|第\s*)$",
    re.I,
)
#: A time coordinate, not a measured quantity ("at 28 days", "within 24 h").
_PROSE_TIME_COORDINATE_BEFORE_RE = re.compile(
    r"\b(?:at|within|by|after|before|until|beyond|first|last|initial)\s+"
    r"(?:the\s+)?(?:first\s+|initial\s+|last\s+)?$",
    re.I,
)
_PROSE_RANGE_TAIL = r"(?:\s*(?:[-–—]|to\b)\s*\d+(?:\.\d+)?)?\s*\)?"
#: A time unit anchored to a clinical event names a window or horizon
#: ("0-24 h after ICU admission", "28 days from onset"), not a measured value.
_PROSE_ANCHORED_TIME_AFTER_RE = re.compile(
    _PROSE_RANGE_TAIL + r"\s*" + _PROSE_TIME_UNIT_PATTERN
    + r"\s+(?:after|before|from|of|since|following|post|prior\s+to|relative\s+to)\s+"
    r"(?:the\s+)?(?:first\s+)?[“\"‘']?(?:icu\s+|hospital\s+|study\s+)?"
    r"(?:admission|onset|baseline|randomi[sz]ation|enrol(?:l)?ment|inclusion"
    r"|discharge|intubation|start|index|time\s+zero|landmark|diagnosis"
    r"|surgery|exposure)\b",
    re.I,
)
_PROSE_UNIT_AFTER_RE = re.compile(
    _PROSE_RANGE_TAIL + r"\s*(?:" + _PROSE_TIME_UNIT_PATTERN + r"(?!"
    + _PROSE_HORIZON_NOUN_PATTERN + r")"
    r"|" + _PROSE_ZH_TIME_UNIT_PATTERN + r"(?!" + _PROSE_ZH_HORIZON_PATTERN + r"))",
    re.I,
)
_PROSE_RANGE_LOW_BEFORE_RE = re.compile(
    r"(?<![\w.])(?P<low>\d+(?:\.\d+)?)\s*(?:[-–—]|to)\s*$"
)
#: The dispersion printed in parentheses after a value: "mean 42 (12)".
_PROSE_PARENTHESISED_AFTER_VALUE_RE = re.compile(
    r"(?<![\w.])(?P<low>\d+(?:\.\d+)?)\s*\(\s*$"
)
#: Digits fused to a name ("SOFA-2", "IL6") are part of the name.
_IDENTIFIER_DIGITS_RE = re.compile(r"(?<=[A-Za-z])[-–]?\d+")
_STANDALONE_NUMBER_RE = re.compile(r"\d")
_CLAUSE_BOUNDARY_RE = re.compile(r"[.!?;](?=\s)|[。！？；]|\n")


def _clause_before(text: str, start: int, *, width: int = 80) -> str:
    window = text[max(0, start - width) : start]
    return _CLAUSE_BOUNDARY_RE.split(window)[-1]


def _gap_has_no_other_number(gap: str) -> bool:
    return _STANDALONE_NUMBER_RE.search(_IDENTIFIER_DIGITS_RE.sub("", gap)) is None


def _statistic_governs(before: str) -> bool:
    """Whether a statistic word governs the number that ends ``before``."""

    matches = list(_PROSE_STATISTIC_RE.finditer(before))
    if not matches:
        return False
    gap = before[matches[-1].end() :]
    if not _gap_has_no_other_number(gap) or len(gap) > 32:
        return False
    return len(re.findall(r"[A-Za-z]+(?:-[A-Za-z0-9]+)*", gap)) <= 3


def short_integer_reports_quantity(text: str, start: int, end: int) -> bool:
    """Whether a 1-2 digit integer in prose reports a clinical quantity.

    The prose recogniser exempts short integers because most of them are
    labels: Section 2, SOFA-2, stage 4, day 28. A reported value is told apart
    by its phrase, never its digits: a time unit after it ("42 years",
    "7 days", "42 岁") unless the unit is anchored to an event as a window
    ("0-24 h after ICU admission"), a statistic word governing it ("the
    median age was 42", "年龄中位数为42"), an explicit ``N = 42``, the far
    end of a range whose near end is itself a reported value ("IQR 3-7"), or
    the dispersion in parentheses after one ("mean 42 (12)").
    """

    before = _clause_before(text, start)
    if _PROSE_LABEL_BEFORE_RE.search(before):
        return False
    if _PROSE_N_EQUALS_BEFORE_RE.search(before):
        return True
    if _PROSE_TIME_COORDINATE_BEFORE_RE.search(before):
        return False
    if _PROSE_ANCHORED_TIME_AFTER_RE.match(text, end):
        return _statistic_governs(before)
    if _PROSE_UNIT_AFTER_RE.match(text, end):
        return True
    range_low = _PROSE_RANGE_LOW_BEFORE_RE.search(before)
    value_before = _PROSE_PARENTHESISED_AFTER_VALUE_RE.search(before)
    low = range_low or value_before
    if low is not None:
        low_start = start - len(before) + low.start("low")
        low_end = start - len(before) + low.end("low")
        literal = low.group("low")
        if "." in literal:
            return True
        if len(literal) > 2:
            # A long range end is a value; a long integer before a parenthesis
            # may be a citation year ("2024 (3)").
            return range_low is not None
        return short_integer_reports_quantity(text, low_start, low_end)
    return _statistic_governs(before)


_PROSE_COUNT_AFTER_RE = re.compile(
    _PROSE_RANGE_TAIL
    + r"\s*(?:(?:icu|hospital)\s+)?"
    r"(?:patients?|stays?|admissions?|encounters?|subjects?|participants?"
    r"|cases?|controls?|deaths?|events?|episodes?|observations?|records?"
    r"|rows?|individuals?|centres?|centers?|sites?|hospitals?)\b(?!-)"
    r"|\s*(?:例|名|位|人)(?!年|日|天)",
    re.I,
)
_PROSE_PERCENT_AFTER_RE = re.compile(r"\s*(?:%|percent\b|per\s?cent\b)", re.I)
_PROSE_AGE_UNIT_AFTER_RE = re.compile(
    _PROSE_RANGE_TAIL + r"\s*(?:岁|years?\s+old\b)", re.I
)
_PROSE_YEARS_AFTER_RE = re.compile(
    _PROSE_RANGE_TAIL + r"\s*(?:years?|yrs?)(?![\w-])", re.I
)
_PROSE_OTHER_TIME_AFTER_RE = re.compile(
    _PROSE_RANGE_TAIL
    + r"\s*(?:(?:days?|d|hours?|hrs?|h|minutes?|mins?|weeks?|wks?|months?|mos?)"
    r"(?![\w-])|天|小时|周|个月|月|分钟)",
    re.I,
)
_PROSE_KIND_WORDS: Tuple[Tuple[NumericQuantityKind, "re.Pattern[str]"], ...] = (
    (NumericQuantityKind.AGE, re.compile(r"\b(?:age|aged|ages)\b|年龄", re.I)),
    (
        NumericQuantityKind.DURATION,
        re.compile(
            r"\b(?:length\s+of\s+(?:icu\s+|hospital\s+)?stay|LOS|duration"
            r"|follow-?\s?up)\b|住院时间|住院天数|时长|随访时间",
            re.I,
        ),
    ),
    (
        NumericQuantityKind.COUNT,
        re.compile(r"(?<![A-Za-z_])[nN]\s*[=:]|\bnumber\s+of\b|例数|人数", re.I),
    ),
    (
        NumericQuantityKind.PROPORTION,
        re.compile(
            r"\b(?:rate|proportion|percentage|prevalence|incidence)\b"
            r"|发生率|比例|患病率",
            re.I,
        ),
    ),
)


def _kind_named_before(before: str) -> Optional[NumericQuantityKind]:
    """The kind named by the nearest kind word that governs the number."""

    nearest: Optional[Tuple[int, NumericQuantityKind]] = None
    for kind, pattern in _PROSE_KIND_WORDS:
        for match in pattern.finditer(before):
            if nearest is None or match.end() > nearest[0]:
                nearest = (match.end(), kind)
    if nearest is None or not _gap_has_no_other_number(before[nearest[0] :]):
        return None
    return nearest[1]


def prose_quantity_kind(
    text: str, *, start: int, end: int
) -> Optional[NumericQuantityKind]:
    """Classify one numeric mention by the words that state what it is.

    Mention-local, like the effect scale: "120 patients with a median age of
    60 years" reports a count and an age in one sentence.
    """

    value = text[start:end]
    if value.endswith("%") or _PROSE_PERCENT_AFTER_RE.match(text, end):
        return NumericQuantityKind.PROPORTION
    if _PROSE_COUNT_AFTER_RE.match(text, end):
        return NumericQuantityKind.COUNT
    if _PROSE_AGE_UNIT_AFTER_RE.match(text, end):
        return NumericQuantityKind.AGE
    before = _clause_before(text, start)
    named = _kind_named_before(before)
    if _PROSE_OTHER_TIME_AFTER_RE.match(text, end):
        return NumericQuantityKind.DURATION
    if _PROSE_YEARS_AFTER_RE.match(text, end):
        # Years measure both an age and a follow-up; only the governing word
        # says which.
        return (
            named
            if named in (NumericQuantityKind.AGE, NumericQuantityKind.DURATION)
            else None
        )
    return named


_FIELD_NON_QUANTITY_TOKENS = frozenset(
    {
        "or", "hr", "rr", "ratio", "coef", "coefficient", "beta", "auc", "auroc",
        "auprc", "p", "pvalue", "difference", "diff", "effect", "z", "t",
        "statistic", "chi2", "se", "stderr", "brier", "estimate", "ci",
    }
)
_FIELD_PROPORTION_TOKENS = frozenset(
    {
        "rate", "rates", "proportion", "percentage", "percent", "pct", "prevalence",
        "incidence", "fraction", "frac", "share",
    }
)
_FIELD_COUNT_MARKER_TOKENS = frozenset(
    {"n", "count", "counts", "number", "num", "denominator", "numerator", "size"}
)
_FIELD_STATUS_TOKENS = frozenset(
    {
        "missing", "missingness", "nonmissing", "excluded", "included",
        "available", "complete", "dropped", "removed", "retained", "eligible",
        "imputed", "observed",
    }
)
_FIELD_AGE_TOKENS = frozenset({"age", "aged", "ages"})
_FIELD_DURATION_TOKENS = frozenset({"los", "duration", "followup"})
_FIELD_COUNT_NOUN_TOKENS = frozenset(
    {
        "patients", "stays", "admissions", "encounters", "subjects",
        "participants", "cases", "controls", "deaths", "events", "episodes",
        "observations", "records", "rows", "individuals", "centres", "centers",
        "sites", "hospitals",
    }
)
_FIELD_TIME_UNIT_TOKENS = frozenset(
    {"days", "day", "d", "hours", "hour", "h", "hrs", "minutes", "mins",
     "weeks", "months"}
)


def _field_tokens(text: str) -> Tuple[str, List[str]]:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    normalized = normalized.replace("follow_up", "followup")
    normalized = normalized.replace("length_of_stay", "los")
    return normalized, [token for token in normalized.split("_") if token]


#: A name part that makes the field something other than a cohort quantity
#: (an effect, a test statistic) or leaves its kind open ("missing").
_NEUTRAL_PART = "neutral"


def _kind_of_field_part(text: str, *, leaf: bool) -> Tuple[Optional[Any], bool]:
    """Return ``(kind or _NEUTRAL_PART or None, explicit_marker)`` for a name part.

    A counted noun names a count only in the leaf: an ancestor such as
    ``rows[3]`` or ``events`` is the container the statistic was read from.
    """

    _normalized, tokens = _field_tokens(text)
    present = set(tokens)
    if present & _FIELD_NON_QUANTITY_TOKENS:
        return _NEUTRAL_PART, False
    if present & _FIELD_PROPORTION_TOKENS:
        return NumericQuantityKind.PROPORTION, True
    if present & _FIELD_COUNT_MARKER_TOKENS:
        return NumericQuantityKind.COUNT, True
    if present & _FIELD_STATUS_TOKENS:
        # "age_missing" may count the stays or report their fraction.
        return _NEUTRAL_PART, False
    if present & _FIELD_AGE_TOKENS:
        return NumericQuantityKind.AGE, False
    if present & _FIELD_DURATION_TOKENS:
        return NumericQuantityKind.DURATION, False
    if leaf and present & _FIELD_COUNT_NOUN_TOKENS:
        return NumericQuantityKind.COUNT, False
    if present & _FIELD_TIME_UNIT_TOKENS:
        return NumericQuantityKind.DURATION, False
    return None, False


def source_field_quantity_kind(source_field: str) -> Optional[NumericQuantityKind]:
    """Infer a registered field's quantity kind from its explicit name.

    The part nearest the leaf that names a kind decides it: "cohort.n_stays"
    and "table_one.age.median" both name theirs. An ancestor naming a
    different kind or a status leaves the name open ("missingness.age" may be
    a fraction of stays), unless the deciding part carries an explicit count
    or proportion marker ("complete_case_flow.n_complete_case").
    """

    source = str(source_field or "").strip()
    if not source:
        return None
    parts = [part for part in re.split(r"[.\[\]]+", source) if part]
    signals = [
        signal
        for signal in (
            _kind_of_field_part(part, leaf=index == len(parts) - 1)
            for index, part in enumerate(parts)
        )
        if signal[0] is not None
    ]
    if not signals:
        return None
    kind, explicit = signals[-1]
    if kind == _NEUTRAL_PART:
        return None
    if not explicit and any(other is not kind for other, _ in signals[:-1]):
        return None
    return kind


def quantity_kind_refuses_claim(
    prose_kind: Optional[NumericQuantityKind], claim: "NumericClaim"
) -> bool:
    """Whether a mention of ``prose_kind`` must not bind to ``claim``.

    Unknown on either side is permissive. A count must be a whole number, so a
    counted mention never binds a fractional value whatever its field name.
    """

    if prose_kind is None:
        return False
    if prose_kind is NumericQuantityKind.COUNT and not float(
        claim.canonical
    ).is_integer():
        return True
    field_kind = source_field_quantity_kind(claim.source_field)
    return field_kind is not None and field_kind is not prose_kind


# ---------------------------------------------------------------------------
# Fact identity named by the prose (fail-closed v0.1.0 mitigation)
# ---------------------------------------------------------------------------
#
# A matching value and a matching quantity kind still do not identify a fact.
# External review (P1-1): with ``n=500`` and ``n_deaths=12`` registered,
# "12 patients" bound to ``n_deaths``; a printed mean bound to a registered
# median; a value in days bound to a field registered in hours; a group-A
# value bound to the group-B field; a percentage bound to a proportion of a
# different denominator. In each case the binder accepted the one candidate
# left after the value and kind filters.
#
# Until claim ids are rendered by the host, a bind is accepted only when the
# words around the number name what tells the chosen field apart from every
# other registered field of the same kind (its statistic, unit, subject and
# group), and never when those words name the other field's identity instead.
# A field whose parts the prose does not name is left unbound: losing a bind
# costs one sentence, a wrong bind publishes a real hash for the wrong fact.

#: Reason code for a bind refused because the prose does not name the fact.
NUMERIC_FACT_IDENTITY_NOT_NAMED = "numeric_fact_identity_not_named"

_STATISTIC = "statistic"
_UNIT = "unit"
_SUBJECT = "subject"

#: Field token -> (dimension, prose pattern naming it).
_IDENTITY_VOCABULARY: Dict[str, Tuple[str, str]] = {
    "median": (_STATISTIC, r"\bmedians?\b|中位"),
    "mean": (_STATISTIC, r"\bmeans?\b|\baverages?\b|均值|平均"),
    "average": (_STATISTIC, r"\bmeans?\b|\baverages?\b|均值|平均"),
    "avg": (_STATISTIC, r"\bmeans?\b|\baverages?\b|均值|平均"),
    "sd": (_STATISTIC, r"\bsds?\b|\bstandard\s+deviations?\b|标准差"),
    "std": (_STATISTIC, r"\bsds?\b|\bstandard\s+deviations?\b|标准差"),
    "iqr": (_STATISTIC, r"\biqr\b|\binterquartile\b|\bquartiles?\b|四分位"),
    "q1": (_STATISTIC, r"\biqr\b|\binterquartile\b|\bquartiles?\b|\bq1\b|\b25th\b|四分位"),
    "q3": (_STATISTIC, r"\biqr\b|\binterquartile\b|\bquartiles?\b|\bq3\b|\b75th\b|四分位"),
    "p25": (_STATISTIC, r"\biqr\b|\binterquartile\b|\bquartiles?\b|\b25th\b|四分位"),
    "p75": (_STATISTIC, r"\biqr\b|\binterquartile\b|\bquartiles?\b|\b75th\b|四分位"),
    "minimum": (_STATISTIC, r"\bminimum\b|\blowest\b"),
    "max": (_STATISTIC, r"\bmaximum\b|\bhighest\b|\bmax\b"),
    "maximum": (_STATISTIC, r"\bmaximum\b|\bhighest\b|\bmax\b"),
    "days": (_UNIT, r"\bdays?\b|\bd\b|天"),
    "day": (_UNIT, r"\bdays?\b|\bd\b|天"),
    "hours": (_UNIT, r"\bhours?\b|\bhrs?\b|\bh\b|小时"),
    "hour": (_UNIT, r"\bhours?\b|\bhrs?\b|\bh\b|小时"),
    "hrs": (_UNIT, r"\bhours?\b|\bhrs?\b|\bh\b|小时"),
    "h": (_UNIT, r"\bhours?\b|\bhrs?\b|\bh\b|小时"),
    "minutes": (_UNIT, r"\bminutes?\b|\bmins?\b|分钟"),
    "mins": (_UNIT, r"\bminutes?\b|\bmins?\b|分钟"),
    "weeks": (_UNIT, r"\bweeks?\b|\bwks?\b|周"),
    "months": (_UNIT, r"\bmonths?\b|个月"),
    "years": (_UNIT, r"\byears?\b|\byrs?\b|年(?!龄)"),
    "deaths": (_SUBJECT, r"\bdeaths?\b|\bdied\b|\bdead\b|\bdeceased\b|\bmortality\b|死亡"),
    "death": (_SUBJECT, r"\bdeaths?\b|\bdied\b|\bdead\b|\bdeceased\b|\bmortality\b|死亡"),
    "mortality": (_SUBJECT, r"\bdeaths?\b|\bdied\b|\bdead\b|\bdeceased\b|\bmortality\b|死亡"),
    "died": (_SUBJECT, r"\bdeaths?\b|\bdied\b|\bdead\b|\bdeceased\b|\bmortality\b|死亡"),
    "los": (_SUBJECT, r"\blos\b|\blength\s+of\s+(?:icu\s+|hospital\s+)?stay|住院时间|住院天数"),
}
#: Tokens that structure a field name or restate its kind, never its identity.
_IDENTITY_STRUCTURAL_TOKENS = frozenset(
    {
        "n", "count", "counts", "number", "num", "total", "overall", "all",
        "value", "values", "result", "results", "summary", "stats", "data",
        "table", "one", "by", "group", "groups", "subgroup", "subgroups",
        "strata", "stratum", "level", "levels", "arm", "arms", "the", "of",
        "and", "in", "per", "denominator", "numerator", "size",
        *_FIELD_PROPORTION_TOKENS,
    }
)
#: Count units: naming one says what was counted, not which count it is, so
#: the word can confirm a field that carries it but never refutes another.
_IDENTITY_WEAK_TOKENS = frozenset(
    {
        "patients", "patient", "stays", "stay", "admissions", "admission",
        "encounters", "encounter", "subjects", "participants", "individuals",
        "records", "rows", "observations",
    }
)
_GROUP_LETTER_BEFORE = (
    r"\b(?:group|arm|cohort|cluster|class|category|stratum|level|subgroup)\s+"
)
#: Citations, footnotes and comments carry digits and words that are not prose.
_IDENTITY_MARKUP_RE = re.compile(
    r"\{evidence:[^}]*\}|\[[^\]\n]*\]\(evidence/[^)\n]*\)|\[\^[^\]]*\]|<!--.*?-->",
    re.S,
)
#: A standalone number; digits fused to a name ("SOFA-2") are part of it.
_IDENTITY_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])(?<![A-Za-z]-)\d[\d,]*(?:\.\d+)?%?")


def _identity_part_tokens(part: str) -> List[str]:
    _normalized, tokens = _field_tokens(part)
    joined: List[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"non", "not", "no"} and index + 1 < len(tokens):
            # "non_survivors" is one group, not a negation of "survivors".
            joined.append(token + tokens[index + 1])
            index += 2
            continue
        joined.append(token)
        index += 1
    return joined


@functools.lru_cache(maxsize=8192)
def _field_kind(source_field: str) -> Optional[NumericQuantityKind]:
    return source_field_quantity_kind(source_field)


_EXTREME_STATISTIC_TOKENS = frozenset({"max", "maximum", "minimum"})
_GROUP_PART_TOKENS = frozenset(
    {"group", "groups", "subgroup", "subgroups", "strata", "stratum", "arm", "arms", "by"}
)


@functools.lru_cache(maxsize=8192)
def _field_identity(source_field: str) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """``(tokens, required)`` for one registered field name.

    ``required`` are the subject tokens prose must state to bind the field
    when a peer lacks them: a named outcome or measure in the vocabulary
    ("deaths", "los") and a group label ("group_b", "groups.non_survivors").
    Other name parts ("input", "included") confirm a field when stated but
    are not demanded: host sentences describe them in their own words.
    """

    tokens: Set[str] = set()
    required: Set[str] = set()
    previous_was_group = False
    for part in re.split(r"[.\[\]]+", str(source_field or "")):
        if not part:
            continue
        raw = _identity_part_tokens(part)
        is_group = previous_was_group or bool(_GROUP_PART_TOKENS & set(raw))
        for token in raw:
            if token.isdigit() or token in _IDENTITY_STRUCTURAL_TOKENS:
                continue
            tokens.add(token)
            if _identity_dimension(token) == _SUBJECT and (
                is_group or token in _IDENTITY_VOCABULARY
            ):
                required.add(token)
        previous_was_group = bool(raw) and all(
            token in _GROUP_PART_TOKENS for token in raw
        )
    return frozenset(tokens), frozenset(required)


def numeric_fact_identity_tokens(source_field: str) -> FrozenSet[str]:
    """The name tokens that identify a registered field among its peers."""

    return _field_identity(str(source_field or ""))[0]


def _identity_dimension(token: str) -> str:
    known = _IDENTITY_VOCABULARY.get(token)
    return known[0] if known is not None else _SUBJECT


def _token_pattern(token: str) -> str:
    known = _IDENTITY_VOCABULARY.get(token)
    if known is not None:
        return known[1]
    if len(token) == 1:
        # A single-letter group ("group_a") is named only as a group label;
        # the article "a" names nothing.
        return _GROUP_LETTER_BEFORE + re.escape(token) + r"\b"
    stem = token[:-1] if token.endswith("s") and len(token) > 4 else token
    if len(stem) < 3:
        return r"(?<![a-z0-9])" + re.escape(stem) + r"(?![a-z0-9])"
    return r"(?<![a-z0-9])" + re.escape(stem) + r"[a-z0-9]*"


def _normalized_prose(text: str) -> str:
    """Lower-case prose in which a field's compound names can be found."""

    lowered = text.lower()
    lowered = re.sub(r"\b(non|not|no)[\s-]+(?=[a-z])", r"\1", lowered)
    # "Sepsis-3" is the field token "sepsis3".
    return re.sub(r"(?<=[a-z])-(?=\d)", "", lowered)


@functools.lru_cache(maxsize=8192)
def _compiled_token_pattern(token: str) -> "re.Pattern[str]":
    return re.compile(_token_pattern(token), re.I)


def _names_any(window: str, tokens: FrozenSet[str], *, weak_counts: bool) -> bool:
    for token in tokens:
        if not weak_counts and token in _IDENTITY_WEAK_TOKENS:
            continue
        if _compiled_token_pattern(token).search(window):
            return True
    return False


def _consistent(
    windows: List[str],
    *,
    own: FrozenSet[str],
    other: FrozenSet[str],
    required: FrozenSet[str],
    sentence: str,
) -> bool:
    """Whether the prose names one dimension of the chosen field consistently.

    ``own``/``other`` are the tokens of this dimension that only the chosen
    field / only the rival carries. The nearest governing window decides:
    naming only the rival's token refutes the bind. ``required`` own tokens
    must then be stated somewhere in the sentence.
    """

    for window in windows:
        names_own = bool(own) and _names_any(window, own, weak_counts=True)
        names_other = bool(other) and _names_any(window, other, weak_counts=False)
        if names_other and not names_own:
            return False
        if names_own:
            return True
    return not required or _names_any(sentence, required, weak_counts=True)


_IDENTITY_CLAUSE_RE = re.compile(r"[,;:]")


def _identity_windows(
    text: str, *, start: int, end: int, sentence_start: int, sentence_end: int
) -> Tuple[str, str, str]:
    """``(before, after, sentence)``: the words between the number and the
    nearest other number or clause break on each side, and the sentence."""

    sentence = text[sentence_start:sentence_end]
    masked = _IDENTITY_MARKUP_RE.sub(lambda m: " " * len(m.group(0)), sentence)
    local_start, local_end = start - sentence_start, end - sentence_start
    before_start = 0
    after_end = len(masked)
    for match in _IDENTITY_NUMBER_RE.finditer(masked):
        if match.end() <= local_start:
            before_start = match.end()
        elif match.start() >= local_end:
            after_end = match.start()
            break
    before = _IDENTITY_CLAUSE_RE.split(masked[before_start:local_start])[-1]
    after = _IDENTITY_CLAUSE_RE.split(masked[local_end:after_end])[0]
    return (
        _normalized_prose(before),
        _normalized_prose(after),
        _normalized_prose(masked),
    )


def unnamed_fact_identity(
    text: str,
    *,
    start: int,
    end: int,
    sentence_start: int,
    sentence_end: int,
    claim: "NumericClaim",
    registry: Sequence["NumericClaim"],
) -> List[str]:
    """Why the prose around one number does not identify ``claim``.

    Returns an empty list when the bind may stand. Otherwise each entry names
    a registered field of the same quantity kind that the prose does not rule
    out (``<step>:<field>``), or ``prose:<dimension>`` when the words name
    another statistic or unit than the one the field registers.
    """

    before, after, sentence = _identity_windows(
        text,
        start=start,
        end=end,
        sentence_start=sentence_start,
        sentence_end=sentence_end,
    )
    windows_for = {
        # A statistic governs the number after it ("median 3.1"); a unit or a
        # counted noun follows its number ("3.1 days", "12 deaths").
        _STATISTIC: [before],
        _UNIT: [after],
        # A counted noun right after the number ("107 ICU stays") is what it
        # counts; words before it ("2 missing of 107") then describe another.
        _SUBJECT: [after] if _PROSE_COUNT_AFTER_RE.match(after) else [after, before],
    }
    own_tokens, own_required = _field_identity(str(claim.source_field or ""))
    blockers: List[str] = []

    # A field's own statistic or unit contradicted by the words that govern
    # the number, whatever else is registered: "mean" is not a median and
    # "days" are not hours.
    for dimension in (_STATISTIC, _UNIT):
        own = frozenset(t for t in own_tokens if _identity_dimension(t) == dimension)
        if not own:
            continue
        own_patterns = {_IDENTITY_VOCABULARY[t][1] for t in own}
        others = frozenset(
            token
            for token, (dim, pattern) in _IDENTITY_VOCABULARY.items()
            if dim == dimension and pattern not in own_patterns
        )
        window = windows_for[dimension][0]
        if _names_any(window, others, weak_counts=False) and not _names_any(
            window, own, weak_counts=True
        ):
            blockers.append(f"prose:{dimension}")

    kind = _field_kind(str(claim.source_field or ""))
    if kind is None:
        return blockers
    seen: Set[FrozenSet[str]] = {own_tokens}
    for rival in registry:
        rival_field = str(rival.source_field or "")
        if _field_kind(rival_field) is not kind:
            continue
        rival_tokens, rival_required = _field_identity(rival_field)
        if rival_tokens in seen:
            continue
        seen.add(rival_tokens)
        for dimension in (_STATISTIC, _UNIT, _SUBJECT):
            own = frozenset(
                t for t in own_tokens - rival_tokens if _identity_dimension(t) == dimension
            )
            other = frozenset(
                t for t in rival_tokens - own_tokens if _identity_dimension(t) == dimension
            )
            if not own and not other:
                continue
            # A unit or a summary statistic that tells the two apart is always
            # stated; a subject only when it is an outcome, measure or group
            # label. An extreme ("max_fraction_missing" across a variable
            # group) is refused when contradicted but not demanded.
            if dimension == _SUBJECT:
                required = own & own_required
                if own:
                    # The chosen field carries its own qualifier; a rival's
                    # descriptive word nearby ("the grace period", "after the
                    # landmark") refutes it only when it is an outcome,
                    # measure or group label.
                    other = other & rival_required
            elif dimension == _STATISTIC:
                required = own - _EXTREME_STATISTIC_TOKENS
            else:
                required = own
            if not _consistent(
                windows_for[dimension],
                own=own,
                other=other,
                required=required,
                sentence=sentence,
            ):
                blockers.append(f"{rival.step_id}:{rival.source_field}")
                break
    return blockers


class ProseNumberMatcher:
    """The prose number recogniser: one regex plus a phrase check.

    A short integer's role is decided by the words around it, which a
    fixed-width lookbehind cannot express ("the median length of stay was 7").
    The regex proposes every short-integer candidate; this keeps the ones whose
    phrase reports a quantity. Every other branch is returned unchanged.

    MEASURED (external review): "The median age was 42 years
    {evidence:cohort_stats}." and "N=42 {evidence:cohort_stats}." passed the
    STRICT chain with zero bindings and no finding, while "n=42" and
    "42 patients" were refused -- the gate keyed on lowercase ``n=`` and count
    nouns only.
    """

    def __init__(self, pattern: "re.Pattern[str]") -> None:
        self._pattern = pattern
        self.pattern = pattern.pattern

    def finditer(self, text: str) -> Iterator["re.Match[str]"]:
        for match in self._pattern.finditer(text):
            if match.group("short_quantity") is None or short_integer_reports_quantity(
                text, match.start("value"), match.end("value")
            ):
                yield match

    def search(self, text: str) -> Optional["re.Match[str]"]:
        return next(self.finditer(text), None)


__all__ = [
    "ProseNumberMatcher",
    "contrast_exposure_coordinate",
    "NumericClaim",
    "NumericEffectScale",
    "NumericEstimand",
    "NUMERIC_FACT_IDENTITY_NOT_NAMED",
    "NumericQuantityKind",
    "coerce_numeric_effect_scale",
    "infer_numeric_claim_identity",
    "numeric_fact_identity_tokens",
    "prose_quantity_kind",
    "quantity_kind_refuses_claim",
    "short_integer_reports_quantity",
    "source_field_quantity_kind",
    "unnamed_fact_identity",
]


# Coder-generated step_summary entries can request that a derived
# numeric value (an OR confidence-interval bound, a between-cohort
# difference, an AUC delta) be computed from other registered claims
# and registered as its own NumericClaim. The formula is evaluated
# here in a restricted-AST sandbox that accepts ONLY:
#
#   * Numeric constants
#   * Names resolving to entries in ``sources`` (or to a math whitelist
#     constant — pi, e)
#   * Arithmetic: + - * / **
#   * Unary +/-
#   * Calls to a small math whitelist: exp, log, log10, sqrt, abs,
#     min, max
#   * Parentheses
#
# Rejected: attribute access, subscripts, comprehensions, comparisons,
# bool ops, lambda, named expressions, anything else. Names starting
# with ``_`` are rejected outright. The output is a finite float; nan
# / inf raise. Errors raise ``DerivedFormulaError`` with the offending
# AST node type for the audit trail.
#
# Inspired by data-to-paper's ``\num{<formula>, "<explanation>"}``;
# our variant evaluates at register-time and persists the result as a
# regular NumericClaim with ``derived_from`` provenance — so the
# existing reverse-binder picks it up with no extra plumbing.


class DerivedFormulaError(ValueError):
    """Raised when a derived-claim formula contains a disallowed
    expression, references an unknown source, or evaluates to a
    non-finite value. The message is safe to surface in audit findings.
    """


# Math whitelist. ``min`` / ``max`` accept any positive number of
# args; the rest are single-arg. Anything outside this dict is rejected
# at call-time.
import math as _math  # noqa: E402  (deliberate post-typing local-only import)

_DERIVED_FORMULA_FUNCS: Dict[str, Any] = {
    "exp": _math.exp,
    "log": _math.log,
    "log10": _math.log10,
    "sqrt": _math.sqrt,
    "abs": abs,
    "min": min,
    "max": max,
}
_DERIVED_FORMULA_CONSTS: Dict[str, float] = {
    "pi": _math.pi,
    "e": _math.e,
}


def _evaluate_derived_formula(
    formula: str,
    *,
    sources: Dict[str, float],
) -> float:
    """Evaluate ``formula`` in the restricted sandbox.

    ``sources`` maps Python identifier → canonical float of a
    registered claim. Names referenced in the formula that aren't in
    ``sources`` and aren't in the math constants whitelist raise
    ``DerivedFormulaError``. Returns a finite float.
    """
    import ast  # local import — only needed when a derived claim arrives

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise DerivedFormulaError(f"derived-formula syntax error: {exc.msg}") from exc

    _ALLOWED_BINOPS = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
    )
    _ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

    def _walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(
                node.value, bool
            ):
                return float(node.value)
            raise DerivedFormulaError(
                f"derived-formula rejected constant of type {type(node.value).__name__}"
            )
        if isinstance(node, ast.Name):
            name = node.id
            if name.startswith("_"):
                raise DerivedFormulaError(
                    f"derived-formula rejected dunder/private name {name!r}"
                )
            if name in sources:
                return float(sources[name])
            if name in _DERIVED_FORMULA_CONSTS:
                return _DERIVED_FORMULA_CONSTS[name]
            raise DerivedFormulaError(
                f"derived-formula references unknown source {name!r}; "
                f"available sources: {sorted(sources)}"
            )
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise DerivedFormulaError(
                    f"derived-formula rejected binary operator "
                    f"{type(node.op).__name__}"
                )
            lhs = _walk(node.left)
            rhs = _walk(node.right)
            try:
                if isinstance(node.op, ast.Add):
                    return lhs + rhs
                if isinstance(node.op, ast.Sub):
                    return lhs - rhs
                if isinstance(node.op, ast.Mult):
                    return lhs * rhs
                if isinstance(node.op, ast.Div):
                    if rhs == 0.0:
                        raise DerivedFormulaError("derived-formula division by zero")
                    return lhs / rhs
                if isinstance(node.op, ast.Pow):
                    return lhs**rhs
            except OverflowError as exc:
                # Python raises OverflowError on float ** float when the
                # result would exceed float range; we surface this as a
                # non-finite result so callers see one consistent error
                # type regardless of whether overflow happens silently
                # (→ inf, caught below) or eagerly (→ OverflowError).
                raise DerivedFormulaError(
                    f"derived-formula evaluated to non-finite value "
                    f"(arithmetic overflow): {exc}"
                ) from exc
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARYOPS):
                raise DerivedFormulaError(
                    f"derived-formula rejected unary operator "
                    f"{type(node.op).__name__}"
                )
            operand = _walk(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise DerivedFormulaError(
                    "derived-formula rejected call with non-name callable"
                )
            fname = node.func.id
            if fname.startswith("_") or fname not in _DERIVED_FORMULA_FUNCS:
                raise DerivedFormulaError(
                    f"derived-formula rejected call to {fname!r}; "
                    f"allowed: {sorted(_DERIVED_FORMULA_FUNCS)}"
                )
            if node.keywords:
                raise DerivedFormulaError(
                    f"derived-formula rejected keyword arguments to {fname!r}"
                )
            args = [_walk(a) for a in node.args]
            try:
                return float(_DERIVED_FORMULA_FUNCS[fname](*args))
            except (ArithmeticError, ValueError, TypeError) as exc:
                raise DerivedFormulaError(
                    f"derived-formula call {fname!r} failed: {exc}"
                ) from exc
        raise DerivedFormulaError(
            f"derived-formula rejected AST node {type(node).__name__}"
        )

    result = _walk(tree)
    if not isinstance(result, (int, float)):
        raise DerivedFormulaError(
            f"derived-formula evaluated to non-numeric {type(result).__name__}"
        )
    fresult = float(result)
    if fresult != fresult or fresult in (float("inf"), float("-inf")):
        raise DerivedFormulaError(
            f"derived-formula evaluated to non-finite value {fresult!r}"
        )
    return fresult


def _coordinate_level(level: str) -> str:
    """A level as a JSON coordinate value; an integer token stays a number."""

    text = str(level)
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text):
        return text
    return json.dumps(text, ensure_ascii=False)


def contrast_exposure_coordinate(variable: str, level: str, reference: str) -> str:
    """Name one contrast of a categorical exposure as a claim coordinate."""

    return (
        f"{variable}={_coordinate_level(level)} versus "
        f"{variable}={_coordinate_level(reference)}"
    )

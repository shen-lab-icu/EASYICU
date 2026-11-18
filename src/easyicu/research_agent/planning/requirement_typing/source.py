"""Conservative source bindings for typed intake.

Only whole clauses are recognized here. The vocabulary reads names in the
question and grants no equivalence between plan facts or dictionary concepts.
"""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING

from ...contracts.product_files import requested_deliverable_kinds
from ..question_quote import normalize_question_quote
from .contract import (
    Binding,
    CohortPredicate,
    Delivery,
    DictionarySnapshot,
    Grading,
    Interval,
    Method,
    NotApplicable,
    Point,
    RepeatStayPolicy,
    RepeatStayReporting,
    RequirementKey,
    TimeZero,
    Unit,
    Window,
)
from .registry import registered_concept_id, registered_grading_source_label

if TYPE_CHECKING:
    from ...schema import ResearchContext

# These are names for individual concepts, not dictionary relative edges.
_NAMES = {
    "k": ("potassium", "serum potassium", "血钾"),
    "lact": ("lactate", "血乳酸", "乳酸"),
    "crea": ("creatinine", "serum creatinine", "肌酐", "血肌酐"),
    "kdigo_creat": ("creatinine", "肌酐"),
    "urine_albumin": ("urine albumin", "urinary albumin", "尿白蛋白"),
    "mg": ("magnesium", "serum magnesium", "血镁"),
    "death": ("death", "mortality", "死亡"),
    "rrt": ("dialysis", "renal replacement therapy", "透析", "肾脏替代治疗"),
    "vent_end": ("extubation", "拔管"),
    "vaso_ind": ("vasopressor initiation", "vasopressor start", "升压药开始"),
    "sep3": ("sepsis onset", "脓毒症起病"),
}
_ANCHORS = {
    "icu_admission": ("icu entry", "icu admission", "入icu", "入住icu"),
    "hospital_admission": ("hospital admission", "入院", "入医院"),
}
_DURATION_NAMES = {
    "s": ("s", "sec", "second", "seconds", "秒"),
    "min": ("min", "minute", "minutes", "分钟"),
    "h": ("h", "hr", "hours", "hour", "小时"),
    "d": ("d", "day", "days", "天", "日"),
}
_DURATION_SECONDS = {"s": 1, "min": 60, "h": 3600, "d": 86400}
_NUMERALS = tuple(zip(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
    "零一二三四五六七八九十", strict=True,
))
_DIRECTIVE = r"(?:(?:please )?(?:use|specify|clarify|document|define|report|state|include) |明确|说明|写明|定义|交代|报告|采用|使用)?"
_ALTERNATIVE = re.compile(r"\b(?:or|either|versus|vs\.?)\b|或者|或是|还是|或", re.I)

_REPEAT_REPORT = (
    r"(?:handling|policy) (?:of|for) repeated icu stays|repeat-stay handling(?: for included icu admissions)?"
    r"|重复入住icu的处理(?:方式)?|重复入住icu的政策"
)
_REPEAT_POLICIES = {
    "first_icu_stay_per_patient": (
        r"(?:only (?:the )?first icu stay per patient|(?:keep|retain|include) only (?:the )?first icu stay per patient"
        r"|每位患者仅保留首次入住icu|仅纳入每位患者的首次icu住院)"
    ),
    "retain_each_stay_cluster_by_patient": (
        r"(?:repeated icu stays are retained with clustering by patient"
        r"|retain repeated icu stays and cluster by patient|保留重复入住icu并按患者聚类)"
    ),
}
_AGGREGATIONS = {
    "max": ("peak", "maximum", "max", "最高", "最大"),
    "min": ("minimum", "min", "lowest", "最低", "最小"),
    "mean": ("mean", "average", "平均"),
    "median": ("median", "中位"),
    "first": ("first", "首次"),
    "last": ("last", "末次"),
}


def _words(values: tuple[str, ...]) -> str:
    return "(?:" + "|".join(re.escape(value) for value in values) + ")"


def _concept(concept_id: str) -> str:
    return _words((concept_id, *_NAMES.get(concept_id, ())))


def _number(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if value == 0:
        text = "0"
    names = _NUMERALS[int(value)] if 0 <= value < len(_NUMERALS) and value == int(value) else ()
    return _words((text, *names))


def _duration(value: Decimal, unit: str) -> str:
    def pattern(amount: Decimal, scale: str) -> str:
        return _number(amount) + r"(?:\s*|-)" + _words(_DURATION_NAMES[scale])

    expressions = [pattern(value, unit)]
    seconds = Fraction(value) * _DURATION_SECONDS[unit]
    for scale, factor in _DURATION_SECONDS.items():
        if scale == unit:
            continue
        exact = seconds / factor
        amount = Decimal(exact.numerator) / Decimal(exact.denominator)
        if Fraction(amount) == exact:
            expressions.append(pattern(amount, scale))
    return "(?:" + "|".join(expressions) + ")"


def _anchor(anchor: str) -> str:
    return _words((anchor, *_ANCHORS.get(anchor, ())))


def _matches(text: str, pattern: str) -> bool:
    return re.fullmatch(_DIRECTIVE + "(?:" + pattern + ")", text) is not None


def repeat_policy_in_source(text: str, policy: str) -> bool:
    return _matches(
        normalize_question_quote(text).rstrip("."),
        r"(?:that )?" + _REPEAT_POLICIES[policy],
    )


def has_source_alternatives(text: str) -> bool:
    return bool(_ALTERNATIVE.search(text))


def _role_bound(binding: Binding, context: ResearchContext | None, dictionary: DictionarySnapshot) -> bool:
    if not context:
        return False
    preferences = context.user_preferences
    columns = {
        "exposure": (context.primary_exposure,),
        "outcome": (context.target_outcome,),
        "covariate": tuple(preferences.covariates) if preferences and (
            preferences.covariate_selection == "exact" and preferences.covariate_authority == "user"
        ) else (),
    }.get(binding.role, ())
    return any(
        column and registered_concept_id(column, context.variable(column), dictionary) == binding.concept_id
        for column in columns
    )


def _interval_from_zero(time: Interval) -> bool:
    return time.start == 0 and isinstance(time.end, Decimal) and time.start_inclusive and not time.end_inclusive


def _window(text: str, facet: Window, time: Interval, subject: str) -> bool:
    if facet.purpose != "aggregation" or facet.operation not in _AGGREGATIONS or not _interval_from_zero(time):
        return False
    duration = _duration(time.end, time.unit)
    anchor = _anchor(time.anchor)
    operation = _words(_AGGREGATIONS[facet.operation])
    return _matches(
        text, rf"{operation} {subject} in the first {duration} after {anchor}"
        rf"|{anchor}后前{duration}取{operation}{subject}",
    )


def _time_zero(text: str, facet: TimeZero, time: Point, subject: str) -> bool:
    anchor = _anchor(time.anchor)
    duration = _duration(time.offset, time.unit)
    if facet.purpose == "sensitivity_landmark":
        return _matches(
            text, rf"(?:the )?{subject} sensitivity landmark at {duration} after {anchor}"
            rf"|{anchor}后{duration}设{subject}敏感性分析界标",
        )
    return _matches(
        text, rf"(?:the )?{subject} time zero (?:at )?{duration} after {anchor}"
        rf"|{subject}时间零点为{anchor}后{duration}",
    )


def _predicate(text: str, facet: CohortPredicate, time: Interval, subject: str) -> bool:
    if (
        facet.operation not in _AGGREGATIONS or facet.operator not in {">=", ">", "<=", "<", "=="}
        or type(facet.value) not in (float, int) or facet.missing_policy not in {"exclude", "retain", "stop"}
        or not _interval_from_zero(time)
    ):
        return False
    polarity = {"include": ("include", "纳入"), "exclude": ("exclude", "排除")}[facet.polarity]
    missing = {
        "exclude": ("exclude", "排除"), "retain": ("retain", "保留"), "stop": ("stop", "停止"),
    }[facet.missing_policy]
    duration = _duration(time.end, time.unit)
    anchor = _anchor(time.anchor)
    operation = _words(_AGGREGATIONS[facet.operation])
    value = _number(Decimal(str(facet.value)))
    comparison = re.escape(facet.operator)
    # NFKC quote normalization omits spaces adjacent to comparison operators.
    return bool(re.fullmatch(
        rf"{polarity[0]} stays with {operation} {subject}{comparison}{value} in the first {duration} after {anchor}, {missing[0]} missing"
        rf"|{polarity[1]}{anchor}后前{duration}{operation}{subject}{comparison}{value}的住院，缺失时{missing[1]}"
        rf"|{polarity[1]}{anchor}后前{duration}{operation}{subject}{comparison}{value}的住院,缺失时{missing[1]}",
        text,
    ))


def _method(text: str, facet: Method, scope: tuple[Binding, ...]) -> bool:
    scoped = {binding.role: _concept(binding.concept_id) for binding in scope}
    params = {item.name: item.value for item in facet.parameters}
    if facet.action_id == "binary_effect_measure" and len(scope) == 2 and scoped.keys() == {"exposure", "outcome"}:
        names = {
            "odds_ratio": ("odds ratio", "比值比"),
            "risk_ratio": ("risk ratio", "风险比", "相对风险"),
        }.get(params.get("measure"))
        if names and params.keys() == {"measure"}:
            return _matches(
                text, rf"(?:the )?{_words(names)} for {scoped['exposure']} and {scoped['outcome']}"
                rf"|{scoped['exposure']}与{scoped['outcome']}的{_words(names)}",
            )
    if facet.action_id == "model_term" and len(scope) == 2 and "outcome" in scoped:
        term = scoped.get("exposure") or scoped.get("covariate")
        model = params.get("model_method")
        if term and isinstance(model, str) and params == {
            "coding": "continuous", "transform": "identity", "levels": (), "reference_level": None,
            "model_method": model,
        }:
            return _matches(
                text, rf"model {term} as continuous for {scoped['outcome']} with {re.escape(model)}"
                rf"|用{re.escape(model)}分析{scoped['outcome']}时将{term}作为连续变量",
            )
    return False


def source_supports(
    quote: str, key: RequirementKey, *, context: ResearchContext | None, dictionary: DictionarySnapshot,
) -> bool:
    """Whether the complete source independently binds this key for credit."""
    text = normalize_question_quote(quote).rstrip(".")
    if has_source_alternatives(text):
        return False
    facet, time = key.facet, key.time
    if isinstance(facet, Delivery):
        return facet.output_id in requested_deliverable_kinds(quote)
    if isinstance(facet, RepeatStayReporting):
        return _matches(text, _REPEAT_REPORT)
    if isinstance(facet, RepeatStayPolicy):
        return repeat_policy_in_source(text, facet.policy)
    scope = tuple(binding for binding in key.bindings if binding.role != "time_anchor")
    if isinstance(facet, CohortPredicate) and isinstance(time, Interval):
        return scope[0].role == "population" and len(scope) == 1 and _predicate(
            text, facet, time, _concept(scope[0].concept_id),
        )
    if not all(_role_bound(binding, context, dictionary) for binding in scope):
        return False
    if isinstance(facet, Method) and isinstance(time, NotApplicable):
        return _method(text, facet, scope)
    if len(scope) != 1:
        return False
    subject = _concept(scope[0].concept_id)
    if isinstance(facet, TimeZero) and isinstance(time, Point):
        return _time_zero(text, facet, time, subject)
    if isinstance(facet, Window) and isinstance(time, Interval):
        return _window(text, facet, time, subject)
    if isinstance(facet, Unit) and isinstance(time, NotApplicable):
        # Unit symbols are case-sensitive. Normalize source wording independently.
        codes = (facet.code, facet.code.replace("/24h", "/24 hours"), facet.code.replace("/24h", "/24小时"))
        if not any(code in quote for code in codes):
            return False
        unit = _words(tuple(code.casefold() for code in codes))
        return _matches(text, rf"{subject} units? (?:in )?{unit}|{subject}单位(?:为)?{unit}")
    if isinstance(facet, Grading) and isinstance(time, NotApplicable):
        contract = re.escape(facet.contract_id.casefold())
        version = re.escape(facet.version.casefold())
        if _matches(text, rf"{subject} grading {contract} version {version}|{subject}按{contract}版本{version}分级"):
            return True
        source_label = registered_grading_source_label(facet, scope[0].concept_id)
        if source_label:
            label = re.escape(source_label)
            return _matches(text, rf"(?:the )?{subject} grading according to {label}|{subject}的{label}分级")
    return False

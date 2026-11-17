"""Occurrence-scoped numeric authority from immutable study definitions.

These sources define names and eligibility, never estimates. Matching a whole
registered phrase releases its own numeric spans. A paraphrase also needs a
source citation and a local attachment to a typed quantity. Definition numbers
stay outside the result claim registry.
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import re

from ..authority.evidence_store import EvidenceEnforcementError, NumericClaim, _NUMERIC_IN_PROSE_RE
from ..authority.plan_scope import verified_plan_evidence_rank
from ..authority.runtime_artifacts import verified_run_evidence_path
from ..contracts.exposure_group_rules import (
    EXPOSURE_GROUP_LABELS_KEY,
    ExposureGroupRuleError,
    concept_reader_name,
    partition_intervals,
    read_grouping_rules,
    read_level_labels,
)
from ..research_context.typed import parse_research_context_json
from ..schema import EvidenceRecord


@dataclass(frozen=True)
class DefinitionSpan:
    start: int
    end: int
    evidence_id: str
    source_field: str
    source_sha256: str
    occurrence: bool = False


# Include small level codes and thresholds even when the ordinary prose
# recognizer treats them as labels. Only definitions can add these matches.
_NUMBER = re.compile(
    r"(?<![\w.])(?P<value>[+-]?(?:"
    r"(?:\d+(?:\.\d+)?|\.\d+)\s*[×x]\s*10[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+"
    r"|(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?"
    r"))(?!\w|\.\d)"
)


def _registered_plan(evidence, records):
    candidates = []
    for record in records:
        approved = re.fullmatch(r"approved_executable_plan_revision_(\d+)", record.evidence_id)
        rank = verified_plan_evidence_rank(record.model_dump(mode="json"))
        if approved:
            if (record.producer, record.generation_mode) != ("plan_lifecycle_authority", "human_confirmed"):
                continue
            key = (1, int(approved[1]))
        elif rank is not None and record.kind == "log":
            key = (0, rank)
        else:
            continue
        candidates.append((key, record))
    if not candidates:
        return None
    latest = max(key for key, _ in candidates)
    selected = [record for key, record in candidates if key == latest]
    if len(selected) != 1:
        raise EvidenceEnforcementError("Study definition plan authority is ambiguous.")
    record = selected[0]
    path = verified_run_evidence_path(evidence.root, record)
    if path is None:
        raise EvidenceEnforcementError("Study definition plan authority failed digest verification.")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record.sha256:
        raise EvidenceEnforcementError("Study definition plan authority changed while being read.")
    payload = json.loads(raw)
    prefix = ""
    if latest[0]:
        from ..authority.plan_lifecycle import ApprovedExecutablePlan

        payload = ApprovedExecutablePlan.model_validate(payload).plan_payload
        prefix = "plan_payload."
    return record, payload, prefix


def _plan_definitions(plan):
    labels = plan.get("display_labels") or {}
    for key, label in labels.items():
        yield f"display_labels.{key}", label
    for index, step in enumerate(plan.get("steps") or []):
        spec = step.get("cohort_definition_spec") or {}
        for position, criterion in enumerate(spec.get("eligibility_criteria") or []):
            yield f"steps[{index}].cohort_definition_spec.eligibility_criteria[{position}].description", criterion["description"]
    # Typed predicates have no prose field. Their literal expression is the
    # exact reproducible definition; arbitrary paraphrases gain no authority.
    cohort = plan.get("cohort") or {}
    for side in ("inclusion", "exclusion"):
        for index, predicate in enumerate(cohort.get(side) or []):
            value = predicate.get("value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                name = labels.get(predicate["concept_id"], predicate["concept_id"])
                yield f"cohort.{side}[{index}].value", f"{name} {predicate['op']} {value}"


def _exact_plan_spans(text, registered):
    record, plan, prefix = registered
    for field, phrase in _plan_definitions(plan):
        if not isinstance(phrase, str) or not re.search(r"[^\W\d_]", phrase) or not _NUMBER.search(phrase):
            continue
        # A sentence-initial capital is the sole surface normalization. Do
        # not casefold units, operators, whitespace or numerical formatting.
        pattern = re.escape(phrase)
        if phrase[0].isalpha():
            pattern = "[" + phrase[0].lower() + phrase[0].upper() + "]" + re.escape(phrase[1:])
        for match in re.finditer(r"(?<!\w)" + pattern + r"(?!\w|\.\d)", text):
            yield DefinitionSpan(match.start(), match.end(), record.evidence_id,
                                 prefix + field, record.sha256)


def definition_number_matches(text, spans):
    """Use the same ordered numeric occurrences for the filter and binder."""
    matches = {match.span("value"): match for match in _NUMERIC_IN_PROSE_RE.finditer(text)}
    for span in spans:
        for match in _NUMBER.finditer(text, span.start, span.end):
            # In a paraphrase "under-8", the hyphen belongs to "under".
            # The generic decimal recognizer may include it as a minus sign.
            start, end = match.span("value")
            if span.occurrence and text[start - 1:start] == "-":
                matches.pop((start - 1, end), None)
            # The prose recognizer owns whole scientific literals. Do not
            # create additional bindings for their coefficient or exponent.
            if not any(left < match.end("value") and match.start("value") < right for left, right in matches):
                matches[match.span("value")] = match
    return [match for _, match in sorted(matches.items())]


def definition_claim(spans, *, start, end, value, canonical):
    for span in spans:
        if span.start <= start and end <= span.end:
            return NumericClaim(value=value, canonical=canonical, evidence_id=span.evidence_id,
                                step_id=span.evidence_id, source_field=span.source_field), span.source_sha256
    return None


@dataclass(frozen=True)
class _Quantity:
    field: str
    value: float
    operator: str | None = None
    unit: str | None = None
    kind: str = "threshold"


def _typed_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _variable_names(*names):
    # Only registered names label a quantity. A descriptor can contain a full
    # source definition, so offer it as a whole alias, without mining its words.
    return {name.strip() for name in names if isinstance(name, str) and name.strip()}


def _variable_aliases(names, *, context, labels):
    aliases = _variable_names(*names)
    for name in aliases.copy():
        aliases.update(_variable_names(name.replace("_", " "), labels.get(name)))
    for variable in context.variables if context else ():
        if variable.name in names:
            aliases.update(_variable_names(variable.description))
    return {alias.casefold() for alias in aliases}


def _variable_unit(name, context):
    variable = context.variable(name) if context else None
    return variable.unit if variable else None


def _grouped_quantities(context, labels):
    """Read the typed projection shared with the exact Methods rule renderer."""
    rows = context.cohort.provenance.get(EXPOSURE_GROUP_LABELS_KEY)
    if not isinstance(rows, list):
        return
    names = [row.get("concept") for row in rows if isinstance(row, dict)]
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("parameters"), dict):
            continue
        parameters = row["parameters"]
        try:
            rules = read_grouping_rules(parameters)
            levels = read_level_labels(row.get("levels"), levels=len(rules.levels))
        except (ExposureGroupRuleError, TypeError):
            continue
        variable = row.get("variable")
        if (not isinstance(variable, str) or not variable.strip() or row.get("concept") != rules.concept
            or row.get("scale") != rules.scale
            or [level["group"] for level in levels] != list(rules.levels)):
            continue
        names_for_group = {variable}
        # A shared unqualified concept is insufficient to identify one of
        # several groupings formed from that measure.
        if names.count(rules.concept) == 1:
            names_for_group.update(_variable_names(rules.concept, concept_reader_name(rules.concept)))
        aliases = _variable_aliases(names_for_group, context=context, labels=labels)
        field = f"cohort.provenance.{EXPOSURE_GROUP_LABELS_KEY}[{index}]"
        quantities = [_Quantity(f"{field}.levels[{pos}].code", level["code"], kind="level")
                      for pos, level in enumerate(levels)]
        for pos, group in enumerate(parameters["groups"]):
            rule = group["rule"]
            if rule == "otherwise":
                continue
            unit = rule.get("unit") or _variable_unit(rules.source_columns[rule["summary"]], context)
            quantities.append(_Quantity(f"{field}.parameters.groups[{pos}].rule.value",
                                        rule["value"], rule["op"], unit))
        # An otherwise level can name the other side of the same boundary.
        # Use the owner's interval algebra, and retain the original value field.
        if parameters["groups"][-1]["rule"] == "otherwise":
            try:
                intervals, _ = partition_intervals(parameters["groups"])
            except ExposureGroupRuleError:
                continue
            other = intervals.get(parameters["groups"][-1]["id"])
            if other:
                for quantity in tuple(quantities):
                    complement = None
                    if quantity.value == other.lower and quantity.operator in {"<", "<="}:
                        complement = ">=" if other.lower_inclusive else ">"
                    if quantity.value == other.upper and quantity.operator in {">", ">="}:
                        complement = "<=" if other.upper_inclusive else "<"
                    if complement:
                        quantities.append(_Quantity(quantity.field, quantity.value, complement, quantity.unit))
        yield aliases, quantities


def _plan_quantities(plan, prefix, context):
    labels = plan.get("display_labels") or {}
    cohort = plan.get("cohort") or {}
    for side in ("inclusion", "exclusion"):
        for index, predicate in enumerate(cohort.get(side) or []):
            name, operator, value = predicate["concept_id"], predicate["op"], predicate.get("value")
            aliases = _variable_aliases({name, concept_reader_name(name)}, context=context, labels=labels)
            field = f"{prefix}cohort.{side}[{index}].value"
            quantities = []
            if _typed_number(value) and operator in _COMPARISONS:
                quantities.append(_Quantity(field, value, operator, _variable_unit(name, context)))
            if operator in {"in", "not_in"} and isinstance(value, list):
                quantities.extend(_Quantity(f"{field}[{pos}]", code, kind="level")
                                  for pos, code in enumerate(value) if _typed_number(code))
            if quantities:
                yield aliases, quantities
    endpoint = plan.get("endpoint") or {}
    if endpoint.get("name"):
        name = endpoint["name"]
        aliases = _variable_aliases({name}, context=context, labels=labels)
        yield aliases, [_Quantity(f"{prefix}endpoint.levels[{pos}]", code, kind="level")
                        for pos, code in enumerate(endpoint.get("levels") or []) if _typed_number(code)]
    for index, step in enumerate(plan.get("steps") or []):
        for position, requirement in enumerate(step.get("model_requirements") or []):
            free_days = (requirement.get("outcome_definition") or {}).get("free_days") or {}
            horizon = free_days.get("horizon_days")
            if _typed_number(horizon):
                names = _variable_aliases({requirement["outcome"]}, context=context, labels=labels)
                field = f"{prefix}steps[{index}].model_requirements[{position}].outcome_definition.free_days.horizon_days"
                yield names, [_Quantity(field, horizon, unit="days", kind="window")]


_NON_PROSE = re.compile(r"<!--.*?-->|`+[^`\n]*`+", re.S)
_CITATION = re.compile(r"\{evidence:(?P<id>[^{}\n]+)\}|\[[^\]\n]*\]\((?P<path>evidence/[^)\s]+)(?:[ \t]+\"[^\"\n]*\")?\)")
# A comparison abbreviation does not end the sentence; keep original offsets.
_COMPARISON_PERIOD = re.compile(r"(?i)(?<=\bvs)\.(?=\s)")
_STATISTIC = re.compile(
    r"\b(?:[ac]?(?:OR|HR|RR)|CI|SD|IQR)\b"
    r"|\b(?i:odds ratio|hazard ratio|risk ratio|relative risk|confidence interval|"
    r"mean|median|standard deviation|interquartile range|p[\s-]*value)\b"
    r"|\b(?i:[pn])\s*[<>=≤≥:]"
)
_STATISTIC_AFTER = re.compile(
    r"\s*(?:%|percent\b|"
    r"\(\s*(?:[0-9]+\s*%\s*)?(?:CI|confidence interval|SD|IQR)\b|"
    r"(?:\(\s*|(?:was|is|for|as)\s+(?:the\s+)?)(?:mean|median|SD|IQR|CI)\b)",
    re.I,
)

# The gap after a variable can state how it was grouped or how its threshold
# was declared. Keep this grammar closed so it cannot bridge another variable.
_ATTACHMENT = re.compile(
    r"(?:\s+(?:at|on)\s+(?:(?:ICU|hospital)\s+)?(?:admission|entry))?"
    r"\s*(?:(?:is|are|was|were)\s+)?"
    r"(?:(?:represented|defined|split|divided|grouped|categorized|classified|dichotomized|"
    r"stratified|assessed|measured|evaluated|ascertained)(?:\s+(?:in|into|with|as|using))?\s+)?"
    r"(?:(?:by\s+|in\s+)?(?:the\s+)?(?:[a-z]+|[0-9]+)\s+(?:ordered\s+)?"
    r"(?:categories|levels|groups|strata)\s*:\s*)?"
    r"(?:(?:of|with)\s+)?"
    r"(?P<kind>(?:by\s+)?(?:(?:a|the)\s+)?(?:cut[- ]?off|cut point|threshold|boundary)\s*(?:of|at|was|=|:)\s*"
    r"|(?:level|code)\s+|(?:within|through|over|by|at)\s+)?", re.I,
)
_COMPARISONS = {
    "<=": r"<=|≤|at most|no more than|less than or equal to",
    ">=": r">=|≥|at least|no less than|greater than or equal to",
    "<": r"<|under|below|less than|younger than",
    ">": r">|over|above|greater than|more than|older than",
    "==": r"==|=(?!=)|equal to|exactly",
    "!=": r"!=|≠|not equal to",
}
_PREFIX_OPERATOR = re.compile(
    r"(?:" + "|".join(f"(?P<op{index}>{pattern})" for index, pattern in enumerate(_COMPARISONS.values()))
    + r")(?![a-z])(?:(?<=[a-z])-[ \t]*|[ \t]*)", re.I,
)
_SUFFIX_OPERATOR = re.compile(
    r"\s+(?:or|and)\s+(?:(?P<above>older|higher|more|greater|above)|younger|lower|less|fewer|below)\b",
    re.I,
)
_COORDINATOR = re.compile(r"(?:\s*,\s*|\s+)(?:and|(?-i:or)|versus|vs\.?)\s+(?:the\s+)?", re.I)
_REFERENCE = re.compile(r"\s*,\s*with\s+(?:the\s+)?", re.I)
_LEVEL_NOUN = re.compile(r"\s+(?:group|category|level|stratum)\b", re.I)
_CUTOFF_NOUN = re.compile(r"\s+(?:cut[- ]?off|threshold|cut point|boundary)\b", re.I)
_TIME_UNITS = (("years?", "yrs?", "y"), ("days?", "d"), ("hours?", "hrs?", "h"),
               ("weeks?", "wks?"), ("months?", "mos?"), ("points?",))
# Detect a unit conflicting with the typed one even if a comparator matches.
_MEASURE = re.compile(
    r"\s*-?\s*(?:[a-zµμ]+/[a-zµμ]+|years?|yrs?|months?|weeks?|days?|hours?|hrs?|points?|"
    r"mmHg|kPa|seconds?|minutes?|bpm|beats|breaths|kg|g|cm|mm|°[CF])\b", re.I,
)
_COUNT_AFTER = re.compile(
    r"\s+(?:icu\s+)?(?:patients?|stays?|deaths?|cases?|events?|admissions?|subjects?|observations?)\b", re.I,
)


def _unit_end(text, start, unit):
    if not unit:
        return start
    literal = re.escape(unit)
    for names in _TIME_UNITS:
        if any(re.fullmatch(name, unit, re.I) for name in names):
            literal = "(?:" + "|".join(names) + ")"
            if names[0] == "years?":
                literal += r"(?:\s+old)?"
            break
    match = re.compile(r"\s*-?\s*" + literal + r"(?![\w/])", re.I).match(text, start)
    return match.end() if match else start


def _read_quantity(text, start, quantities, kind, parse_literal):
    before = _PREFIX_OPERATOR.match(text, start)
    number = _NUMBER.match(text, before.end() if before else start)
    if number is None:
        return None, start
    start, end = number.span("value")
    parsed = parse_literal(number["value"])
    suffix = text[end:]
    if parsed is None or parsed[1] or _STATISTIC_AFTER.match(suffix) or _COUNT_AFTER.match(suffix):
        return None, end
    for quantity in quantities:
        if parsed[0] != quantity.value:
            continue
        if quantity.kind == "level":
            if kind in {"level", "code"} and not before and not _MEASURE.match(suffix):
                return (start, end, quantity.field), end
            continue
        unit_end = _unit_end(text, end, quantity.unit)
        if unit_end == end and quantity.unit and _MEASURE.match(suffix):
            continue
        after = _SUFFIX_OPERATOR.match(text, unit_end)
        if before and after:
            continue
        if _STATISTIC_AFTER.match(text[unit_end:]):
            continue
        operator = None
        if before:
            operator = tuple(_COMPARISONS)[int(before.lastgroup[2:])]
        elif after:
            operator = ">=" if after["above"] else "<="
        if quantity.kind == "window":
            # The endpoint's unit identifies a duration; a level or a bare
            # number attached to its name does not.
            if kind in {"within", "through", "over", "by", "at"} and unit_end != end and operator is None:
                return (start, end, quantity.field), unit_end
        elif (operator is not None and operator == quantity.operator
              or operator is None and (
                  kind == "cutoff" or unit_end != end
                  and (kind == "at" or _CUTOFF_NOUN.match(text, unit_end))
              )):
            return (start, end, quantity.field), after.end() if after else unit_end
    return None, end


def _quantity_spans(text, anchors, aliases, quantities, parse_literal):
    if not quantities:
        return
    for anchor in anchors:
        if anchor[0].casefold() not in aliases:
            continue
        prefix = text[:anchor.start()]
        # Generated rule rows have their own exact line and section contract.
        if prefix.rsplit("\n", 1)[-1].lstrip().startswith(
            "Recorded exposure grouping for the grouped exposure:"
        ):
            continue
        clause_prefix = re.split(r"[,;()\n]", prefix)[-1]
        if _STATISTIC.search(clause_prefix) or _STATISTIC.search(anchor[0]):
            continue
        attached = _ATTACHMENT.match(text, anchor.end())
        kind = (attached["kind"] or "").strip().casefold()
        if kind and kind not in {"level", "code", "within", "through", "over", "by", "at"}:
            kind = "cutoff"
        span, cursor = _read_quantity(text, attached.end(), quantities, kind, parse_literal)
        while span:
            yield span
            connector = _COORDINATOR.match(text, cursor)
            reference = _REFERENCE.match(text, cursor)
            continuation = connector or reference
            if continuation is None:
                break
            span, cursor = _read_quantity(text, continuation.end(), quantities, kind, parse_literal)
            if reference and not _LEVEL_NOUN.match(text, cursor):
                break


def _cites(record, tokens, evidence):
    return any(record.relative_path == token["path"]
               or (token["id"] and evidence.get(token["id"].strip()) == record)
               for token in tokens)


def _registered_context(text, evidence, records):
    record = next((row for row in records if row.evidence_id == "research_context"), None)
    if record is None or (record.kind, record.producer, record.generation_mode,
                          record.produced_by_step) != ("log", "pipeline", "system", None):
        return None
    if not _cites(record, _CITATION.finditer(text), evidence):
        return None
    path = verified_run_evidence_path(evidence.root, record)
    if path is None:
        return None
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() == record.sha256:
            return record, parse_research_context_json(payload)
    except (OSError, ValueError):
        return None
    return None


def _sentence_citations(text, start, end):
    sentence = text[start:end]
    # A token immediately after a preceding sentence's period is its trailing
    # citation. A token at the start of the paragraph can lead this sentence.
    after_period = start > 0 and text[start - 1] in ".!?"
    prose_start = 0
    if after_period:
        leading = sentence.lstrip()
        while token := _CITATION.match(leading):
            leading = leading[token.end():].lstrip()
        prose_start = len(sentence) - len(leading)
    return sentence, tuple(_CITATION.finditer(sentence, prose_start))


def registered_definition_spans(text, evidence, *, sentence_bounds, parse_literal, skip_spans):
    """Bind exact definitions and individual occurrences in cited paraphrases."""
    # Lineage-only record views used by citation repair carry no sealed
    # definition payload. Definition authority requires full evidence records.
    records = [record for record in evidence.records() if isinstance(record, EvidenceRecord)]
    if not records:
        return ()
    registered = _registered_plan(evidence, records)
    exact = tuple(_exact_plan_spans(text, registered)) if registered else ()
    sentence_text = _NON_PROSE.sub(
        lambda match: re.sub(r"[^\r\n]", " ", match[0]), text,
    )
    context_source = _registered_context(sentence_text, evidence, records)
    plan_record, plan, prefix = registered or (None, {}, "")
    context_record, context = context_source or (None, None)
    labels = plan.get("display_labels") or {}
    grouped = list(_grouped_quantities(context, labels)) if context else []
    planned = list(_plan_quantities(plan, prefix, context)) if registered else []
    known_names = {alias for aliases, quantities in (*grouped, *planned) if quantities for alias in aliases}
    if not known_names:
        return exact
    if context:
        known_names.update(_variable_aliases({var.name for var in context.variables}, context=context, labels=labels))
    # Consume the longest registered name first. A bare alias inside a
    # different variable's compound name cannot own its threshold.
    names = "|".join(re.escape(name) for name in sorted(known_names, key=lambda name: (-len(name), name)))
    anchor_pattern = re.compile(r"(?<!\w)(?:" + names + r")(?!\w)", re.I)

    boundary_text = _COMPARISON_PERIOD.sub(" ", sentence_text)
    seen = set()
    spans = defaultdict(set)
    for number in _NUMBER.finditer(boundary_text):
        if any(left <= number.start() < right for left, right in skip_spans):
            continue
        start, end = sentence_bounds(boundary_text, start=number.start(), end=number.end())
        if (start, end) in seen:
            continue
        seen.add((start, end))
        sentence, cited = _sentence_citations(sentence_text, start, end)
        cites_context = context_record is not None and _cites(context_record, cited, evidence)
        sources = []
        if cites_context and grouped:
            sources.append((context_record, grouped))
        if planned and (cites_context or _cites(plan_record, cited, evidence)):
            sources.append((plan_record, planned))
        if not sources:
            continue
        # Blank markup while preserving offsets. An evidence label can never
        # serve as the variable name anchoring a threshold.
        prose = _CITATION.sub(lambda match: " " * len(match[0]), sentence)
        anchors = tuple(anchor_pattern.finditer(prose))
        for record, definitions in sources:
            for aliases, quantities in definitions:
                for left, right, field in _quantity_spans(prose, anchors, aliases, quantities, parse_literal):
                    spans[start + left, start + right].add(
                        DefinitionSpan(start + left, start + right, record.evidence_id, field, record.sha256, occurrence=True)
                    )
    return (*exact, *(next(iter(matches)) for matches in spans.values() if len(matches) == 1))

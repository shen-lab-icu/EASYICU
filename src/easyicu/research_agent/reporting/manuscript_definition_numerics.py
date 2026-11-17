"""Occurrence-scoped numeric authority from immutable study definitions.

These sources define names and eligibility, never estimates. Matching a whole
registered phrase releases its own numeric spans. A paraphrase also needs a
source citation and a local attachment to a typed quantity. Definition numbers
stay outside the result claim registry.
"""

from collections import defaultdict
from dataclasses import dataclass, replace
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


# Check small level codes and thresholds even when the ordinary prose
# recognizer treats them as labels.
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
    for key, label in (plan.get("display_labels") or {}).items():
        yield f"display_labels.{key}", label
    for index, step in enumerate(plan.get("steps") or []):
        spec = step.get("cohort_definition_spec") or {}
        for position, criterion in enumerate(spec.get("eligibility_criteria") or []):
            yield f"steps[{index}].cohort_definition_spec.eligibility_criteria[{position}].description", criterion["description"]


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


def definition_number_matches(text, spans, *, extra_occurrences=()):
    """Use the same ordered numeric occurrences for the filter and binder."""
    matches = {match.span("value"): match for match in _NUMERIC_IN_PROSE_RE.finditer(text)}
    regions = [(span.start, span.end, span.occurrence) for span in spans]
    regions.extend((start, end, True) for start, end in extra_occurrences)
    for left, right, occurrence in regions:
        for match in _NUMBER.finditer(text, left, right):
            # In a paraphrase "under-8", the hyphen belongs to "under".
            # The generic decimal recognizer may include it as a minus sign.
            start, end = match.span("value")
            if occurrence and text[start - 1:start] == "-":
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
    variable: str
    value: float
    operator: str | None = None
    unit: str | None = None
    kind: str = "threshold"
    literal: str | None = None
    source_column: str | None = None


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
    names = [
        row["concept"].casefold() for row in rows
        if isinstance(row, dict) and isinstance(row.get("concept"), str)
    ]
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
        concept = rules.concept.casefold()
        concept_names = _variable_names(rules.concept, concept, concept_reader_name(concept))
        shared_aliases = set()
        if names.count(concept) > 1:
            shared_aliases = _variable_aliases(concept_names, context=context, labels=labels)
        else:
            names_for_group.update(concept_names)
        aliases = _variable_aliases(names_for_group, context=context, labels=labels)
        field = f"cohort.provenance.{EXPOSURE_GROUP_LABELS_KEY}[{index}]"
        quantities = [_Quantity(f"{field}.levels[{pos}].code", variable, level["code"], kind="level")
                      for pos, level in enumerate(levels)]
        multiple_columns = len(rules.read_columns) > 1
        for pos, group in enumerate(parameters["groups"]):
            rule = group["rule"]
            if rule == "otherwise":
                continue
            column = rules.source_columns[rule["summary"]]
            unit = rule.get("unit") or _variable_unit(column, context)
            quantities.append(_Quantity(
                f"{field}.parameters.groups[{pos}].rule.value", concept, rule["value"], rule["op"], unit,
                source_column=column if multiple_columns else None,
            ))
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
                        quantities.append(replace(quantity, operator=complement))
        yield aliases - shared_aliases, quantities, True
        if shared_aliases:
            # Keep explicitly attached short numerals visible to enforcement.
            yield shared_aliases, [q for q in quantities if q.kind == "threshold"], False


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
                literal = f"{labels.get(name, name)} {operator} {value}"
                quantities.append(_Quantity(field, name, value, operator, _variable_unit(name, context),
                                            literal=literal))
            if operator in {"in", "not_in"} and isinstance(value, list):
                quantities.extend(_Quantity(f"{field}[{pos}]", name, code, kind="level")
                                  for pos, code in enumerate(value) if _typed_number(code))
            if quantities:
                yield aliases, quantities
    endpoint = plan.get("endpoint") or {}
    if endpoint.get("name"):
        name = endpoint["name"]
        aliases = _variable_aliases({name}, context=context, labels=labels)
        yield aliases, [_Quantity(f"{prefix}endpoint.levels[{pos}]", name, code, kind="level")
                        for pos, code in enumerate(endpoint.get("levels") or []) if _typed_number(code)]
    for index, step in enumerate(plan.get("steps") or []):
        for position, requirement in enumerate(step.get("model_requirements") or []):
            free_days = (requirement.get("outcome_definition") or {}).get("free_days") or {}
            horizon = free_days.get("horizon_days")
            if _typed_number(horizon):
                name = requirement["outcome"]
                aliases = _variable_aliases({name}, context=context, labels=labels)
                field = f"{prefix}steps[{index}].model_requirements[{position}].outcome_definition.free_days.horizon_days"
                yield aliases, [_Quantity(field, name, horizon, unit="days", kind="window")]


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
_LEVEL_MARKER = r"(?:level|code)\s+"
_LEVEL_PREFIX = re.compile(_LEVEL_MARKER, re.I)
_ATTACHMENT = re.compile(
    r"(?:\s+(?:at|on)\s+(?:(?:ICU|hospital)\s+)?(?:admission|entry))?"
    r"\s*(?:(?:is|are|was|were)\s+)?"
    r"(?:(?:represented|defined|split|divided|grouped|categorized|classified|dichotomized|"
    r"stratified|assessed|measured|evaluated|ascertained)(?:\s+(?:in|into|with|as|using))?\s+)?"
    r"(?:(?:by\s+|in\s+)?(?:the\s+)?(?:[a-z]+|[0-9]+)\s+(?:ordered\s+)?"
    r"(?:categories|levels|groups|strata)\s*:\s*)?"
    r"(?:(?:of|with)\s+)?"
    r"(?P<kind>(?:by\s+)?(?:(?:a|the)\s+)?(?:cut[- ]?off|cut point|threshold|boundary)\s*(?:of|at|was|=|:)\s*"
    r"|" + _LEVEL_MARKER + r"|(?:within|through|by|at)\s+)?", re.I,
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
_COORDINATOR = re.compile(
    r"(?:(?P<comma>\s*,\s*)|\s+)(?:(?P<list_join>and|(?-i:or))|versus|vs\.?)\s+(?:the\s+)?", re.I,
)
_LIST_COMMA = re.compile(r"\s*,\s*(?=[+-]?(?:\d|\.\d))")
_REFERENCE = re.compile(r"\s*,\s*with\s+(?:the\s+)?", re.I)
_LEVEL_NOUN = re.compile(r"\s+(?:group|category|level|stratum)\b", re.I)
_CUTOFF_NOUN = re.compile(r"\s+(?:cut[- ]?off|threshold|cut point|boundary)\b", re.I)
_BOUNDARY_END = re.compile(r"\s*(?:[)\]]\s*)*(?:;|[.!?]?\s*\Z)")
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


def _attached_quantity_end(text, end, quantity, kind, before, *, marked_boundary):
    suffix = text[end:]
    if quantity.kind == "level":
        if kind in {"level", "code"} and not before and not _MEASURE.match(suffix):
            return end
        return None
    if kind in {"level", "code"}:
        return None
    unit_end = _unit_end(text, end, quantity.unit)
    if unit_end == end and _MEASURE.match(suffix):
        return None
    after = _SUFFIX_OPERATOR.match(text, unit_end)
    if (before and after) or _STATISTIC_AFTER.match(text[unit_end:]):
        return None
    if quantity.kind == "window":
        over = before and before[0].strip().casefold() == "over"
        if unit_end != end and not after and (over or (not before and kind in {"within", "through", "by", "at"})):
            return unit_end
        return None
    if before:
        operator = tuple(_COMPARISONS)[int(before.lastgroup[2:])]
        return unit_end if operator == quantity.operator else None
    if after:
        operator = ">=" if after["above"] else "<="
        return after.end() if operator == quantity.operator else None
    if unit_end != end and _CUTOFF_NOUN.match(text, unit_end):
        return unit_end
    if kind == "cutoff" or (unit_end != end and marked_boundary):
        # A bare cutoff must end here or have a recognized list continuation.
        # A predicate after the value can turn it into a statistic.
        if (_BOUNDARY_END.match(text, unit_end) or _LIST_COMMA.match(text, unit_end)
            or _COORDINATOR.match(text, unit_end) or _REFERENCE.match(text, unit_end)):
            return unit_end
    return None


def _quantity_key(quantity):
    # The written comparator has already narrowed the candidates. When the
    # prose names a cutoff without a comparator, either side can own its value.
    unit = quantity.unit.casefold() if quantity.unit else None
    return quantity.variable.casefold(), quantity.kind, quantity.value, unit, quantity.source_column


def _read_quantity(text, start, kind, definitions, parse_literal, *, anchor, previous, marked_boundary, cites_context):
    before = _PREFIX_OPERATOR.match(text, start)
    number = _NUMBER.match(text, before.end() if before else start)
    if number is None:
        return None
    start, end = number.span("value")
    continued = previous is not None
    literal_forms = ()
    if not continued:
        phrase = text[anchor:end]
        literal_forms = (phrase, phrase[:1].swapcase() + phrase[1:])
    exact = False
    active = []
    for record, quantities, cited in definitions:
        literals = [quantity for quantity in quantities if quantity.literal in literal_forms]
        exact = exact or bool(literals)
        if cited or literals:
            active.append((record, quantities if cited else literals))
    # A context citation can leave an anchored cutoff unbound. Keep its
    # occurrence visible even when the general prose matcher skips it.
    if not (active or continued or cites_context):
        return None

    parsed = parse_literal(number["value"])
    suffix = text[end:]
    if parsed is None or parsed[1] or _STATISTIC_AFTER.match(suffix) or _COUNT_AFTER.match(suffix):
        return (start, end, None, end) if continued or exact else None

    # Each threshold is a separate bound. Multiple declarations of the same
    # outcome window or set of level codes still need to agree as definitions.
    definitions_seen = set()
    matches = {}
    attached = False
    for record, quantities in active:
        applicable = set()
        for quantity in quantities:
            if continued and (quantity.variable.casefold(), quantity.kind) != (
                previous.variable.casefold(), previous.kind
            ):
                continue
            cursor = _attached_quantity_end(text, end, quantity, kind, before, marked_boundary=marked_boundary)
            if cursor is None:
                continue
            attached = True
            if quantity.kind == "threshold" and parsed[0] != quantity.value:
                continue
            key = _quantity_key(quantity)
            applicable.add(key)
            if parsed[0] == quantity.value:
                matches.setdefault(key, (record, quantity, cursor))
        if applicable:
            definitions_seen.add(frozenset(applicable))
    if len(matches) == 2:
        column_key, unqualified_key = matches
        if column_key[-1] is None:
            column_key, unqualified_key = unqualified_key, column_key
        if unqualified_key[-1] is None and column_key[:-1] == unqualified_key[:-1]:
            # A coincident plan boundary has no column of its own. The single
            # matching grouping column supplies the more specific source.
            definitions_seen.discard(frozenset({unqualified_key}))
            del matches[unqualified_key]
    if len(definitions_seen) == 1 and len(matches) == 1:
        record, quantity, cursor = next(iter(matches.values()))
        return start, end, (record, quantity), cursor
    if continued or exact or before or kind or attached or _SUFFIX_OPERATOR.match(text, end):
        return start, end, None, end
    return None


def _quantity_spans(text, anchors, sources, parse_literal, *, cites_context, grouped_variables):
    for anchor in anchors:
        name = anchor[0].casefold()
        definitions = [
            (record, quantities, cited) for record, aliases, quantities, cited in sources
            if name in aliases and quantities
        ]
        if not definitions:
            continue
        # A context citation can stand in for a plan definition on an
        # ungrouped variable. Groupings can omit shared aliases.
        if cites_context and not any(cited for _, _, cited in definitions):
            definitions = [
                (record, quantities, all(q.variable.casefold() not in grouped_variables for q in quantities))
                for record, quantities, _ in definitions
            ]
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
        if kind and kind not in {"level", "code", "within", "through", "by", "at"}:
            kind = "cutoff"
        marked_boundary = kind in {"cutoff", "at"}
        position = attached.end()
        previous = reference = None
        serial_list = False
        while occurrence := _read_quantity(
            text, position, kind, definitions, parse_literal, anchor=anchor.start(), previous=previous,
            marked_boundary=marked_boundary, cites_context=cites_context,
        ):
            left, right, source, cursor = occurrence
            if reference and not _LEVEL_NOUN.match(text, cursor):
                break
            yield left, right, source
            if source is None:
                break
            _, previous = source
            connector = _COORDINATOR.match(text, cursor)
            reference = _REFERENCE.match(text, cursor)
            list_comma = _LIST_COMMA.match(text, cursor) if marked_boundary and previous.kind == "threshold" else None
            continuation = connector or reference or list_comma
            if continuation is None:
                break
            # A preceding numeric list item licenses an Oxford comma.
            if reference or (connector and connector["comma"] and not (serial_list and connector["list_join"])):
                marked_boundary = False
            serial_list = list_comma is not None
            position = continuation.end()
            marker = _LEVEL_PREFIX.match(text, position)
            kind = marker[0].strip().casefold() if marker else ""
            if marker:
                position = marker.end()


def _cites(record, tokens, evidence):
    return any(record.relative_path == token["path"]
               or (token["id"] and evidence.get(token["id"].strip()) == record)
               for token in tokens)


def _registered_context(evidence, records):
    record = next((row for row in records if row.evidence_id == "research_context"), None)
    if record is None:
        return None, None
    if (record.kind, record.producer, record.generation_mode,
        record.produced_by_step) != ("log", "pipeline", "system", None):
        return record, None
    path = verified_run_evidence_path(evidence.root, record)
    if path is None:
        return record, None
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() == record.sha256:
            return record, parse_research_context_json(payload)
    except (OSError, ValueError):
        pass
    return record, None


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
    """Return definition authority and unresolved occurrences for numeric tracing."""
    # Lineage-only record views used by citation repair carry no sealed
    # definition payload. Definition authority requires full evidence records.
    records = [record for record in evidence.records() if isinstance(record, EvidenceRecord)]
    if not records:
        return (), ()
    registered = _registered_plan(evidence, records)
    exact = tuple(_exact_plan_spans(text, registered)) if registered else ()
    sentence_text = _NON_PROSE.sub(
        lambda match: re.sub(r"[^\r\n]", " ", match[0]), text,
    )
    plan_record, plan, prefix = registered or (None, {}, "")
    context_record, context = _registered_context(evidence, records)
    labels = plan.get("display_labels") or {}
    grouped = list(_grouped_quantities(context, labels)) if context else []
    grouped_variables = {quantity.variable.casefold() for _, quantities, _ in grouped for quantity in quantities}
    # A registered context supplies the units and variable names even when
    # only the plan is cited. An unverified context disables typed plan binds.
    planned = list(_plan_quantities(plan, prefix, context)) if registered and (context or not context_record) else []
    known_names = {alias for aliases, quantities, _ in grouped if quantities for alias in aliases}
    known_names.update(alias for aliases, quantities in planned if quantities for alias in aliases)
    if not known_names:
        return exact, ()
    if context:
        known_names.update(_variable_aliases({var.name for var in context.variables}, context=context, labels=labels))
    # Consume the longest registered name first. A bare alias inside a
    # different variable's compound name cannot own its threshold.
    names = "|".join(re.escape(name) for name in sorted(known_names, key=lambda name: (-len(name), name)))
    anchor_pattern = re.compile(r"(?<!\w)(?:" + names + r")(?!\w)", re.I)

    boundary_text = _COMPARISON_PERIOD.sub(" ", sentence_text)
    seen = set()
    spans = defaultdict(set)
    extra_occurrences = set()
    for number in _NUMBER.finditer(boundary_text):
        if any(left <= number.start() < right for left, right in skip_spans):
            continue
        start, end = sentence_bounds(boundary_text, start=number.start(), end=number.end())
        if (start, end) in seen:
            continue
        seen.add((start, end))
        sentence, cited = _sentence_citations(sentence_text, start, end)
        cites_context = context_record is not None and _cites(context_record, cited, evidence)
        sources = [
            (context_record, aliases, quantities, citable) for aliases, quantities, citable in grouped if cites_context
        ]
        if planned:
            cites_plan = _cites(plan_record, cited, evidence)
            sources.extend((plan_record, aliases, quantities, cites_plan) for aliases, quantities in planned)
        if not sources:
            continue
        # Blank markup while preserving offsets. An evidence label can never
        # serve as the variable name anchoring a threshold.
        prose = _CITATION.sub(lambda match: " " * len(match[0]), sentence)
        anchors = tuple(anchor_pattern.finditer(prose))
        for left, right, source in _quantity_spans(
            prose, anchors, sources, parse_literal, cites_context=cites_context, grouped_variables=grouped_variables,
        ):
            occurrence = start + left, start + right
            if source is None:
                extra_occurrences.add(occurrence)
                continue
            record, quantity = source
            spans[occurrence].add(DefinitionSpan(
                *occurrence, record.evidence_id, quantity.field, record.sha256, occurrence=True,
            ))
    resolved = []
    for occurrence, matches in spans.items():
        if len(matches) == 1 and occurrence not in extra_occurrences:
            resolved.append(next(iter(matches)))
        else:
            extra_occurrences.add(occurrence)
    return (*exact, *resolved), tuple(sorted(extra_occurrences))

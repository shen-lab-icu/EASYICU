"""Occurrence-scoped numeric authority from immutable study definitions.

These sources define names and eligibility, never estimates. Matching a whole
registered phrase releases only its own numeric spans, not equal values in
nearby prose. No definition number enters the result claim registry.
"""

from dataclasses import dataclass
import hashlib
import json
import re

from ..authority.evidence_store import EvidenceEnforcementError, NumericClaim, _NUMERIC_IN_PROSE_RE
from ..authority.plan_scope import verified_plan_evidence_rank
from ..authority.runtime_artifacts import verified_run_evidence_path


@dataclass(frozen=True)
class DefinitionSpan:
    start: int
    end: int
    evidence_id: str
    source_field: str
    source_sha256: str


# Include small level codes and thresholds even when the ordinary prose
# recognizer treats them as labels. Only exact definition spans use this RE.
_NUMBER = re.compile(r"(?<![\w.])(?P<value>[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?)(?!\w|\.\d)")


def _registered_plan(evidence):
    candidates = []
    for record in evidence.records():
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


def registered_definition_spans(text, evidence):
    """Exact reader labels / eligibility phrases from the latest sealed plan."""
    registered = _registered_plan(evidence)
    if registered is None:
        return ()
    record, plan, prefix = registered
    spans = []
    for field, phrase in _plan_definitions(plan):
        if not isinstance(phrase, str) or not re.search(r"[^\W\d_]", phrase) or not _NUMBER.search(phrase):
            continue
        # A sentence-initial capital is the sole surface normalization. Do
        # not casefold units, operators, whitespace or numerical formatting.
        pattern = re.escape(phrase)
        if phrase[0].isalpha():
            pattern = "[" + phrase[0].lower() + phrase[0].upper() + "]" + re.escape(phrase[1:])
        for match in re.finditer(r"(?<!\w)" + pattern + r"(?!\w|\.\d)", text):
            spans.append(DefinitionSpan(match.start(), match.end(), record.evidence_id,
                                        prefix + field, record.sha256))
    return tuple(spans)


def definition_number_matches(text, spans):
    """Use the same ordered numeric occurrences for the filter and binder."""
    matches = {match.span("value"): match for match in _NUMERIC_IN_PROSE_RE.finditer(text)}
    for span in spans:
        for match in _NUMBER.finditer(text, span.start, span.end):
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

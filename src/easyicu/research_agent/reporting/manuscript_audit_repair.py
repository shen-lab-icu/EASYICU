"""Bounded final manuscript audit repair with immutable scientific authority.

The normal Writer owns section dispatch; the normal binding stage owns STRICT
numeric/evidence checks. This owner only feeds repairable audit findings back
to those owners, records the attempt, and refuses changes to existing facts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from ..authority.evidence_store import EvidenceEnforcementError
from ..bounded_repair import BoundedRepair, RepairFeedback
from ..contracts.manuscript_sentence_context import (
    EVIDENCE_LINK_PATTERN, EVIDENCE_TOKEN_PATTERN, MARKDOWN_REGION_RE, SENTENCE_CITATIONS_RE,
    prose_sentence_ends,
)
from .descriptive_report_facts import insert_question_fact_tokens
from .manuscript_labels import recorded_definition_section_errors
from .manuscript_sections import (
    MANUSCRIPT_SECTION_SPECS,
    ManuscriptSectionContractError,
    manuscript_binding_tokens,
    manuscript_repair_feedback,
    manuscript_repair_regressions,
    manuscript_section_key,
)

NUMERIC_REMOVAL = "MANUSCRIPT_UNBINDABLE_NUMERIC_SENTENCE"
EVIDENCE_REMOVAL = "MANUSCRIPT_UNREGISTERED_EVIDENCE_PLACEHOLDER_REMOVED"
DEPENDENT_REMOVAL = "MANUSCRIPT_BINDING_DEPENDENT_SENTENCE_REMOVED"
LOSS_STOP = "MANUSCRIPT_BINDING_FILTER_LOSS_STOP"
FILTER_FINDING_CODES = frozenset((NUMERIC_REMOVAL, EVIDENCE_REMOVAL, DEPENDENT_REMOVAL))

REPAIRABLE_MANUSCRIPT_CODES = frozenset({
    "MANUSCRIPT_TITLE_MISSING", "MANUSCRIPT_KEYWORDS_MISSING",
    "MANUSCRIPT_SECTION_MISSING", "MANUSCRIPT_SECTION_EMPTY",
    "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY", "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY",
    "MANUSCRIPT_ABSTRACT_LABEL_MISSING_OR_EMPTY", "MANUSCRIPT_ABSTRACT_CONCLUSION_DUPLICATES_RESULTS",
    "MANUSCRIPT_CONCLUSION_WITHOUT_INTERPRETATION", "MANUSCRIPT_BASELINE_METHODS_INCOMPLETE",
    "MANUSCRIPT_SECTION_TRUNCATED", "MANUSCRIPT_INTERNAL_TERM_EXPOSED",
    "MANUSCRIPT_VARIABLE_DEFINITION_CONTEXT_MISSING", "MANUSCRIPT_ADJUSTMENT_SET_CONFLICT",
    "MANUSCRIPT_DISPLAY_NOT_CALLED_OUT", "MANUSCRIPT_REPEATED_PARAGRAPH", "MANUSCRIPT_REPEATED_SENTENCE",
    "MANUSCRIPT_METRIC_UNNAMED", "MANUSCRIPT_ESTIMATE_WITHOUT_VALUE",
    "MANUSCRIPT_PRIMARY_RESULT_COVERAGE_INCOMPLETE", "MANUSCRIPT_REPORTED_RESULT_DISCLAIMED",
    "OWNER_CLAIM_TARGET_ABSENT",
})
_CODE = re.compile(r"\b(?:MANUSCRIPT_[A-Z0-9_]+|OWNER_CLAIM_TARGET_ABSENT|writer_question_part_uncovered)\b")
_MARKER = re.compile(
    r"(?P<literal>[+−-]?(?:"
    r"(?:\d+(?:\.\d+)?|\.\d+)\s*[×x]\s*10[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+"
    r"|(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?))"
    r"(?:%?[ \t]*)\[\^(?P<claim>claim_\d+)\]"
)


_H2 = re.compile(r"(?m)^##[ \t]+([^\r\n]+)")
_SECTION_ORDER = {spec.key: i for i, spec in enumerate(MANUSCRIPT_SECTION_SPECS)}


def _section_blocks(text: str) -> list[tuple[str, str]]:
    headings = list(_H2.finditer(text))
    starts = [0, *(match.start() for match in headings), len(text)]
    keys = ["title", *(manuscript_section_key(match[1]) for match in headings)]
    return [(key, text[start:stop]) for key, start, stop in zip(keys, starts, starts[1:])]


def splice_manuscript_sections(before: str, after: str, keys: Sequence[str]) -> tuple[str, list[str]]:
    """Replace only the requested Writer sections, preserving the other bytes."""
    original = _section_blocks(before)
    replacements: dict[str, str] = {}
    for key, block in _section_blocks(after):
        if key not in keys or not block.strip():
            continue
        if key in replacements:
            raise ManuscriptSectionContractError(
                section_name=key, missing_subsections=("a single section with this heading",),
            )
        replacements[key] = block
    spliced = [key for key in keys if key in replacements]
    original_keys = {key for key, _ in original}
    insertions = [key for key in _SECTION_ORDER if key in replacements and key not in original_keys]
    output = []
    for key, block in original:
        if key in replacements:
            block = replacements.pop(key)
        elif key in spliced:
            continue
        while insertions and _SECTION_ORDER[insertions[0]] < _SECTION_ORDER.get(key, len(_SECTION_ORDER)):
            output.append(replacements.pop(insertions.pop(0)))
        output.append(block)
    output.extend(replacements[key] for key in insertions)

    chunks: list[str] = []
    for block in output:
        if block and chunks and not chunks[-1].endswith("\n"):
            chunks.append("\n")
        if block:
            chunks.append(block)
    return "".join(chunks), spliced


def require_preserved_section_tokens(before: str, after: str, keys: Sequence[str]) -> None:
    original, candidate = _section_blocks(before), _section_blocks(after)
    for section in keys:
        prior_tokens = Counter(t for key, block in original if key == section for t in manuscript_binding_tokens(block))
        next_tokens = Counter(t for key, block in candidate if key == section for t in manuscript_binding_tokens(block))
        if missing := prior_tokens - next_tokens:
            raise EvidenceEnforcementError(
                "MANUSCRIPT_REPAIR_AUTHORITY_CHANGED: an existing evidence or claim binding changed.",
                detail={"check": "section_tokens", "section": section, "missing_tokens": dict(missing)},
            )


def _codes(errors: Mapping[str, Sequence[str]]) -> list[str]:
    return sorted({
        code
        for rows in errors.values()
        for row in rows
        for code in _CODE.findall(row) or ["MANUSCRIPT_EDITORIAL_FINDING"]
    })


def _numeric_authority(binding: Any) -> Counter:
    """Ignore footnote renumbering, retain section, value and source identity."""
    claims = getattr(binding, "numeric_binding_map", {})
    result: Counter = Counter()
    for section, text in _section_blocks(binding.bound):
        for match in _MARKER.finditer(text):
            claim = claims.get(match["claim"])
            if claim is not None:
                payload = claim.to_dict() if callable(getattr(claim, "to_dict", None)) else dict(claim)
                result[(section, match["literal"], json.dumps(payload, sort_keys=True, default=str))] += 1
    return result


def _authority_failed(findings: Sequence[Any], binding: Any) -> bool:
    """A numeric or evidence failure on its own."""
    if getattr(binding, "untraced_numerics", ()):
        return True
    return any(
        finding.validator in {"manuscript_numeric_auditor", "evidence_bound_writer"}
        and (finding.severity == "error"
             or (finding.detail or {}).get("missing_evidence_ids")
             or (finding.detail or {}).get("reason_code") == "unregistered_evidence_placeholder_removed")
        for finding in findings
    )


def _binding_invariant_failed(findings: Sequence[Any], binding: Any) -> bool:
    """Reject binding failures and legacy findings without repair coordinates."""
    return _authority_failed(findings, binding) or any(
        finding.validator == "manuscript_numeric_auditor"
        and (finding.detail or {}).get("removed_sentences")
        for finding in findings
    )


def _persist(run_dir: Path | None, evidence: Any, receipt: dict[str, Any]) -> None:
    if run_dir is None:
        return
    path = run_dir / "manuscript_audit_repair.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence.register_file(
        kind="log", description="Bounded final manuscript audit repair attempts and authority checks.",
        source_path=path, evidence_id="manuscript_audit_repair", producer="pipeline",
        generation_mode="system", on_sha_change="new_id",
    )


def _coverage_findings(binding: Any) -> Sequence[Any]:
    return getattr(binding, "question_coverage_findings", ())


def _record_unresolved_coverage(receipt, binding):
    if coverage := _coverage_findings(binding):
        receipt["unresolved_question_parts"] = [finding.detail for finding in coverage]


def _filter_instruction(item: RepairFeedback) -> str:
    ids = [*item.facts.get("number_ids", ())]
    if evidence_id := item.facts.get("evidence_id"):
        ids.append(evidence_id)
    return (
        f"{item.code}: Binding rejected {', '.join(ids)} in {item.path}. "
        "Rewrite the section without that claim, or cite only bound values."
    )


def _discard_candidate_findings(findings, start):
    findings[start:] = [
        finding for finding in findings[start:]
        if (finding.detail or {}).get("reason_code") in FILTER_FINDING_CODES
    ]


def _filter_stop(binding):
    losses = getattr(binding, "filter_loss", ())
    feedback = getattr(binding, "filter_feedback", ())
    unknown_owner = any(item.path not in _SECTION_ORDER for item in feedback)
    if losses or unknown_owner:
        explanations = []
        if any(item.facts["more_than_half_removed"] for item in losses):
            explanations.append("the filter removed more than half the sentences in a section")
        if any(item.facts["headline_result_removed"] for item in losses):
            explanations.append("the filter removed a pre-declared primary result")
        if unknown_owner:
            explanations.append("a removal has no Writer section owner")
        raise EvidenceEnforcementError(
            f"{LOSS_STOP}: " + " and ".join(explanations) + ".",
            detail={
                "code": LOSS_STOP, "losses": [item.as_dict() for item in losses],
                "removals": [item.as_dict() for item in feedback],
            },
        )


def _repair_targets(draft, binding, section_errors):
    if draft.writer_error_message:
        return {}, ()
    errors = dict(section_errors(draft, binding))
    feedback = list(manuscript_repair_feedback(errors))
    for finding in _coverage_findings(binding):
        detail = finding.detail
        if not detail["repair_facts"]:
            continue
        code = detail["reason_code"]
        feedback.append(RepairFeedback(code, "results", detail))
        text = " ".join(
            f'Under ### {fact["subsection"]}, add {fact["claim_placeholder"]} alone on its own line. '
            f'The verified host fact reads "{fact["text"]}".'
            for fact in detail["repair_facts"]
        )
        errors["results"] = (*errors.get("results", ()), f"{code}: {finding.message} {text}")
    filter_instructions: dict[str, list[str]] = {}
    for item in getattr(binding, "filter_feedback", ()):
        feedback.append(item)
        filter_instructions.setdefault(item.path, []).append(_filter_instruction(item))
    for section, instructions in filter_instructions.items():
        errors[section] = (*errors.get(section, ()), "\n".join(instructions))
    return errors, tuple(feedback)


def _remaining_codes(errors, binding, rejection=()):
    return sorted(
        set(_codes(errors))
        | {finding.detail["reason_code"] for finding in _coverage_findings(binding)}
        | {item.code for item in rejection}
    )


def _check_candidate(candidate, bind, findings, start, protected, context):
    binding = bind(candidate)
    if definition_errors := recorded_definition_section_errors(binding.bound, context):
        raise EvidenceEnforcementError(
            "MANUSCRIPT_DEFINITION_NOT_FROM_DICTIONARY: prose changes the sealed definition.",
            detail={"sections": sorted(definition_errors)},
        )
    _filter_stop(binding)
    if _binding_invariant_failed(findings[start:], binding):
        raise EvidenceEnforcementError("MANUSCRIPT_AUDIT_NON_REPAIRABLE: repaired evidence or numeric checks failed.")
    if not getattr(binding, "filter_feedback", ()) and protected - _numeric_authority(binding):
        raise EvidenceEnforcementError("MANUSCRIPT_REPAIR_AUTHORITY_CHANGED: an existing bound number or source changed.")
    return binding


def _accept_binding(findings, binding):
    # Each successful binding checks the complete planned coverage again.
    # Attempt receipts retain the previous coordinates, including on resume.
    if hasattr(binding, "question_coverage_findings"):
        current = {id(finding) for finding in binding.question_coverage_findings}
        findings[:] = [
            finding for finding in findings
            if finding.validator != "writer_question_coverage" or id(finding) in current
        ]


def _host_question_repair(current, binding, bind, section_errors, findings, context, prior_feedback, receipt):
    slots = [
        {**fact, **{key: finding.detail[key] for key in ("step_id", "product_id", "outcome")}}
        for finding in _coverage_findings(binding) for fact in finding.detail["repair_facts"]
    ]
    scaffold, inserted = insert_question_fact_tokens(current.scaffold, slots)
    if not inserted:
        return None
    row = receipt["host_fallback"] = {"inserted": inserted}
    start = len(findings)
    candidate = replace(current, scaffold=scaffold)
    try:
        require_preserved_section_tokens(current.scaffold, scaffold, ("results",))
        checked = _check_candidate(candidate, bind, findings, start, _numeric_authority(binding), context)
        errors, feedback = _repair_targets(candidate, checked, section_errors)
    except EvidenceEnforcementError as exc:
        row.update(status="rejected", codes=_codes({"results": (str(exc),)}) or ["MANUSCRIPT_AUDIT_NON_REPAIRABLE"])
        _discard_candidate_findings(findings, start)
        raise
    regressions = manuscript_repair_regressions(prior_feedback, feedback)
    if (getattr(checked, "filter_feedback", ()) or regressions
            or any(finding.detail["repair_facts"] for finding in _coverage_findings(checked))):
        row.update(status="rejected", regressions=[item.as_dict() for item in regressions])
        _discard_candidate_findings(findings, start)
        raise EvidenceEnforcementError(
            "writer_question_part_uncovered: the host fallback failed final manuscript checks.",
            detail={"code": "writer_question_part_uncovered", "regressions": row["regressions"]},
        )
    row.update(status="pass", numeric_checks="pass", evidence_checks="pass",
               output_sha256=hashlib.sha256(scaffold.encode()).hexdigest(),
               output_bound_sha256=hashlib.sha256(checked.bound.encode()).hexdigest())
    return candidate, checked, errors


def run_manuscript_audit_repair(
    *, draft: Callable[[Any], Any], bind: Callable[[Any], Any],
    section_errors: Callable[[Any, Any], Mapping[str, tuple[str, ...]]],
    findings: list[Any], run_dir: Path | None, evidence: Any, skip_repair: bool = False,
    context: Any = None,
) -> tuple[Any, Any]:
    """Repair section-owned findings before the normal final manuscript gates."""
    start = len(findings)
    repair = BoundedRepair()
    receipt = repair.receipt
    receipt.update(schema_version="easyicu.manuscript_audit_repair/1", status="not_needed")
    current = draft(None)
    binding = bind(current)
    if filter_feedback := getattr(binding, "filter_feedback", ()):
        receipt["binding_filter_removals"] = [item.as_dict() for item in filter_feedback]
    if skip_repair:
        _record_unresolved_coverage(receipt, binding)
        _persist(run_dir, evidence, receipt)
        return current, binding
    errors, prior_feedback = _repair_targets(current, binding, section_errors)
    definition_errors = recorded_definition_section_errors(binding.bound, context)
    if definition_errors:
        receipt.update(status="non_repairable", remaining_codes=["MANUSCRIPT_DEFINITION_NOT_FROM_DICTIONARY"])
        _record_unresolved_coverage(receipt, binding)
        _persist(run_dir, evidence, receipt)
        raise EvidenceEnforcementError(
            "MANUSCRIPT_DEFINITION_NOT_FROM_DICTIONARY: prose changes the sealed definition.",
            detail={"sections": sorted(definition_errors)},
        )
    try:
        _filter_stop(binding)
    except EvidenceEnforcementError as exc:
        receipt.update(status="non_repairable", remaining_codes=sorted({LOSS_STOP, *_codes(errors)}),
                       binding_filter_stop=exc.detail)
        _record_unresolved_coverage(receipt, binding)
        _persist(run_dir, evidence, receipt)
        raise
    if errors and _binding_invariant_failed(findings[start:], binding):
        receipt.update(status="non_repairable", remaining_codes=["MANUSCRIPT_AUDIT_NON_REPAIRABLE", *_codes(errors)])
        _record_unresolved_coverage(receipt, binding)
        _persist(run_dir, evidence, receipt)
        raise EvidenceEnforcementError("MANUSCRIPT_AUDIT_NON_REPAIRABLE: numeric or evidence checks failed.")
    _accept_binding(findings, binding)
    if not errors:
        if _coverage_findings(binding):
            receipt.update(status="non_repairable", remaining_codes=_remaining_codes(errors, binding))
            _record_unresolved_coverage(receipt, binding)
        _persist(run_dir, evidence, receipt)
        return current, binding

    if (repair_scaffold := getattr(binding, "repair_scaffold", None)) is not None:
        current = replace(current, scaffold=repair_scaffold)
    last_rejection: tuple[RepairFeedback, ...] = ()
    while errors and repair.available:
        request = dict(errors)
        for item in last_rejection:
            keys = (item.path,) if item.path in errors else tuple(errors)
            for key in keys:
                names = item.facts.get("missing_subsections", ())
                if names:
                    detail = ", ".join(names)
                elif item.code in FILTER_FINDING_CODES:
                    if item not in prior_feedback:
                        request[key] = (*request[key], _filter_instruction(item))
                    continue
                elif finding := item.facts.get("finding"):
                    detail = f"Last attempt introduced {finding} in {item.path}. Preserve sections that passed."
                else:
                    detail = item.facts.get("message", "Check the reported section against the previous draft.")
                request[key] = (*request[key], f"{item.code}: {detail}")
        protected = _numeric_authority(binding)
        row = repair.begin(
            (*prior_feedback, *last_rejection),
            section_keys=list(errors), audit_codes=_codes(errors),
            input_sha256=hashlib.sha256(current.scaffold.encode()).hexdigest(),
            input_bound_sha256=hashlib.sha256(binding.bound.encode()).hexdigest(),
            protected_numeric_occurrences=sum(protected.values()),
            numeric_checks="pending", evidence_checks="pending",
        )
        receipt["status"] = "running"
        _persist(run_dir, evidence, receipt)
        candidate_start = len(findings)
        try:
            candidate = draft((current.scaffold, request))
            if candidate.writer_error_message:
                last_rejection = (RepairFeedback("MANUSCRIPT_AUDIT_REPAIR_FAILED", "manuscript"),)
                repair.finish("rejected", findings=last_rejection)
            else:
                spliced, keys = splice_manuscript_sections(current.scaffold, candidate.scaffold, tuple(errors))
                row["spliced_section_keys"] = keys
                require_preserved_section_tokens(current.scaffold, spliced, tuple(errors))
                if spliced != candidate.scaffold:
                    candidate = replace(candidate, scaffold=spliced)
                checked = _check_candidate(candidate, bind, findings, candidate_start, protected, context)
                next_errors, next_feedback = _repair_targets(candidate, checked, section_errors)
                candidate_removals = getattr(checked, "filter_feedback", ())
                regressions = manuscript_repair_regressions(
                    prior_feedback, tuple(item for item in next_feedback if item.code not in FILTER_FINDING_CODES),
                )
                last_rejection = (*candidate_removals, *regressions)
                disposition = "unresolved" if _remaining_codes(next_errors, checked) else "resolved"
                if repair.finish(
                    "rejected" if candidate_removals else disposition,
                    findings=next_feedback, regressions=regressions,
                    numeric_checks="filtered" if any(item.code != EVIDENCE_REMOVAL for item in candidate_removals) else "pass",
                    evidence_checks="filtered" if any(item.code == EVIDENCE_REMOVAL for item in candidate_removals) else "pass",
                    output_sha256=hashlib.sha256(candidate.scaffold.encode()).hexdigest(),
                    output_bound_sha256=hashlib.sha256(checked.bound.encode()).hexdigest(),
                ):
                    _accept_binding(findings, checked)
                    current, binding, errors, prior_feedback = candidate, checked, next_errors, next_feedback
        except ManuscriptSectionContractError as exc:
            last_rejection = (RepairFeedback(
                "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY", manuscript_section_key(exc.section_name),
                {"missing_subsections": list(exc.missing_subsections)},
            ),)
            repair.finish("rejected", findings=last_rejection)
        except EvidenceEnforcementError as exc:
            detail = dict(exc.detail or {})
            if "spliced_section_keys" in detail:
                row["spliced_section_keys"] = detail.pop("spliced_section_keys")
            codes = _CODE.findall(str(exc)) or ["MANUSCRIPT_AUDIT_NON_REPAIRABLE"]
            if detail:
                row["authority_failure"] = detail
            last_rejection = tuple(
                RepairFeedback(code, detail.get("section", "manuscript"), {"message": str(exc)})
                for code in codes
            )
            if detail.get("code") == LOSS_STOP:
                repair.finish("hard_stop", findings=last_rejection, error_type=type(exc).__name__)
                _discard_candidate_findings(findings, candidate_start)
                receipt.update(status="non_repairable", remaining_codes=codes, binding_filter_stop=detail)
                _record_unresolved_coverage(receipt, binding)
                _persist(run_dir, evidence, receipt)
                raise
            if detail.get("check") == "section_tokens":
                repair.finish("hard_stop", findings=last_rejection, error_type=type(exc).__name__)
                row.update(status="rejected", invariant_stop=True)
                receipt.update(status="rejected", remaining_codes=codes)
                _record_unresolved_coverage(receipt, binding)
                _persist(run_dir, evidence, receipt)
                raise
            repair.finish("rejected", findings=last_rejection, error_type=type(exc).__name__)
        except Exception as exc:
            codes = _CODE.findall(str(exc)) or ["MANUSCRIPT_AUDIT_REPAIR_FAILED"]
            repair.finish("rejected", findings=tuple(RepairFeedback(code, "manuscript") for code in codes),
                          error_type=type(exc).__name__)
            receipt.update(status="rejected", remaining_codes=codes)
            _record_unresolved_coverage(receipt, binding)
            _persist(run_dir, evidence, receipt)
            raise
        if last_rejection:
            _discard_candidate_findings(findings, candidate_start)
        remaining = _remaining_codes(errors, binding, last_rejection)
        if not remaining:
            row["status"] = "pass"
        receipt.update(status="exhausted" if errors and not repair.available else row["status"],
                       remaining_codes=remaining)
        _persist(run_dir, evidence, receipt)
    filter_unresolved = any(item.code in FILTER_FINDING_CODES for item in (*prior_feedback, *last_rejection))
    if errors and not repair.available and _coverage_findings(binding) and not filter_unresolved:
        try:
            fallback = _host_question_repair(
                current, binding, bind, section_errors, findings, context, prior_feedback, receipt,
            )
        except EvidenceEnforcementError:
            _record_unresolved_coverage(receipt, binding)
            _persist(run_dir, evidence, receipt)
            raise
        if fallback:
            current, binding, errors = fallback
            _accept_binding(findings, binding)
            if not errors:
                last_rejection = ()
            remaining = _remaining_codes(errors, binding, last_rejection)
            receipt.update(status="exhausted" if remaining else "pass", remaining_codes=remaining)
    _record_unresolved_coverage(receipt, binding)
    if not errors and not last_rejection and _coverage_findings(binding):
        receipt["status"] = "non_repairable"
    if last_rejection:
        code = "MANUSCRIPT_BINDING_FILTER_REPAIR_EXHAUSTED" if filter_unresolved else "MANUSCRIPT_AUDIT_REPAIR_FAILED"
        if filter_unresolved:
            receipt["stop_code"] = code
        _persist(run_dir, evidence, receipt)
        raise EvidenceEnforcementError(
            f"{code}: " + ", ".join(receipt["remaining_codes"]),
            detail={"code": code, "audit_codes": receipt["remaining_codes"]},
        )
    _persist(run_dir, evidence, receipt)
    return current, binding


_HIDDEN = re.compile(r"<!--.*?-->|^ *```[^\n]*\n.*?^ *```[^\n]*", re.M | re.S)
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_SPACE = re.compile(r"\s*")
_EVIDENCE_REFERENCE = re.compile(rf"(?P<token>{EVIDENCE_TOKEN_PATTERN})|(?P<link>{EVIDENCE_LINK_PATTERN})")


def sentence_key(text: str) -> str:
    """Normalize scaffolds and rendered citations to the same internal key."""
    text = SENTENCE_CITATIONS_RE.sub("", text)
    return _normalized_text(text).casefold()


def _normalized_text(text: str) -> str:
    return " ".join(re.sub(r"\s+([.!?。！？])", r"\1", text).split())


def _sentence_evidence_names(text: str) -> frozenset[str]:
    names = set()
    for reference in _EVIDENCE_REFERENCE.finditer(text):
        if token := reference["token"]:
            names.add(token[len("{evidence:"):-1].strip())
        else:
            # Binding preserves the requested name as the link label, even
            # when the evidence path contains a different canonical ID.
            names.add(reference["link"].split("](", 1)[0][1:])
    return frozenset(names)


@dataclass(frozen=True)
class SentenceSpan:
    section: str
    start: int
    end: int
    key: str
    evidence_ids: frozenset[str] = frozenset()
    structural: bool = False
    section_heading: str | None = None


def _heading_owner(heading: str) -> tuple[str, str]:
    """Resolve an H2 alias or an exact section name before a colon."""
    key = manuscript_section_key(heading)
    if key in _SECTION_ORDER:
        return key, heading.strip()
    label = re.split(r"[:：]", heading, maxsplit=1)[0].strip()
    owner = manuscript_section_key(label)
    return (owner, label) if owner in _SECTION_ORDER else (key, "")


def sentence_spans(text: str, *, include_structural: bool = False) -> tuple[SentenceSpan, ...]:
    """Locate prose and optionally complete Markdown lines in source coordinates."""
    visible = _HIDDEN.sub(lambda m: re.sub(r"[^\r\n]", " ", m[0]), text)
    sentences = []
    section = "title"
    start = 0
    for region in (*MARKDOWN_REGION_RE.finditer(visible), None):
        stop = region.start() if region else len(text)
        position = start
        for end in (*[start + offset for offset in prose_sentence_ends(text[start:stop])], stop):
            if end <= position:
                continue
            # Visible trailing citations belong to the preceding sentence.
            # Leading annotations remain available to dependent-context deletion.
            citation_end = end
            while citation_end < stop:
                token_start = _SPACE.match(visible, citation_end, stop).end()
                token = SENTENCE_CITATIONS_RE.match(visible, token_start, stop)
                if not token:
                    break
                citation_end = token.end()
            raw = visible[position:citation_end]
            if key := sentence_key(raw):
                source = text[position:citation_end]
                left = position + len(source) - len(source.lstrip())
                right = position + len(source.rstrip())
                sentences.append(SentenceSpan(section, left, right, key, _sentence_evidence_names(raw)))
            position = citation_end
        if region:
            heading_label = None
            if heading := region["section"]:
                section, heading_label = _heading_owner(heading)
            if include_structural and (heading or region["structural"]):
                raw = region[0]
                sentences.append(SentenceSpan(
                    section, region.start(), region.end(), sentence_key(raw),
                    _sentence_evidence_names(raw), structural=True, section_heading=heading_label,
                ))
            start = region.end()
    return tuple(sentences)


def _sentence_deletion_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Consume the line ending only if the deletion owns the whole line."""
    line_start = text.rfind("\n", 0, start) + 1
    newline = text.find("\n", end)
    line_end = newline if newline != -1 else len(text)
    if text[line_start:start].strip():
        return start, end
    tail = text[end:line_end]
    if tail.strip():
        return start, end + len(tail) - len(tail.lstrip(" \t"))
    return line_start, newline + 1 if newline != -1 else line_end


def _section_heading_replacement(span: SentenceSpan) -> str:
    """Keep a rejected H2's section delimiter when its owner is unambiguous."""
    if span.section_heading:
        return "## " + span.section_heading
    raise EvidenceEnforcementError(
        f"{LOSS_STOP}: a rejected section heading has no Writer section owner.",
        detail={"code": LOSS_STOP, "losses": [], "removals": [
            RepairFeedback(NUMERIC_REMOVAL, "manuscript").as_dict(),
        ]},
    )


def sentence_deletion_edit(text: str, span: SentenceSpan, *, end: int | None = None) -> tuple[int, int, str]:
    """Return the exact source edit for a rejected sentence or Markdown line."""
    if span.section_heading is not None:
        return span.start, span.end, _section_heading_replacement(span)
    start, stop = _sentence_deletion_bounds(text, span.start, span.end if end is None else end)
    return start, stop, ""


def delete_spans(text: str, spans: Sequence[SentenceSpan]) -> str:
    pieces = []
    cursor = 0
    for span in sorted(set(spans), key=lambda item: item.start):
        start, end, replacement = sentence_deletion_edit(text, span)
        pieces.extend((text[cursor:start], replacement))
        cursor = end
    return "".join((*pieces, text[cursor:]))


def _section_at(text: str, position: int) -> str:
    headings = [match for match in _H2.finditer(text) if match.start() <= position]
    return _heading_owner(headings[-1][1])[0] if headings else "title"


def _section_owner(section: str) -> str:
    return section if section in _SECTION_ORDER else "manuscript"


def scaffold_without_removed_sentences(scaffold: str, removed: Sequence[SentenceSpan]) -> str:
    """Delete the removed prose occurrences from the Writer's repair input."""
    wanted = Counter((span.section, span.key, span.evidence_ids, span.structural) for span in removed)
    selected = []
    for span in sentence_spans(scaffold, include_structural=True):
        coordinate = (span.section, span.key, span.evidence_ids, span.structural)
        if wanted[coordinate]:
            selected.append(span)
            wanted[coordinate] -= 1
    return delete_spans(scaffold, selected)


@dataclass(frozen=True)
class UnregisteredEvidence:
    section: str
    sentence_key: str
    evidence_id: str

    def feedback(self, *, removed: bool) -> RepairFeedback:
        return RepairFeedback(EVIDENCE_REMOVAL, _section_owner(self.section), {
            "evidence_id": self.evidence_id,
            "sentence_removed": removed,
        })


def unregistered_evidence_feedback(
    scaffold: str, locations: Sequence[tuple[int, str]],
) -> tuple[UnregisteredEvidence, ...]:
    sentences = sentence_spans(scaffold)
    items = []
    for ordinal, (position, evidence_id) in enumerate(locations, 1):
        span = next((s for s in sentences if s.start <= position < s.end), None)
        items.append(UnregisteredEvidence(
            span.section if span else _section_at(scaffold, position),
            span.key if span else "",
            evidence_id if _ID.fullmatch(evidence_id) else f"unregistered_evidence_{ordinal}",
        ))
    return tuple(items)


def enforce_after_evidence_removals(
    scaffold: str, stale: Sequence[UnregisteredEvidence],
    enforce: Callable[[str], tuple[str, list[str]]],
) -> tuple[str, list[str], tuple[SentenceSpan, ...], tuple[RepairFeedback, ...]]:
    """Remove stale-citation sentences only when the strict gate rejects them."""
    sentence_drops: tuple[SentenceSpan, ...] = ()
    try:
        checked, removed = enforce(scaffold)
    except EvidenceEnforcementError as exc:
        if not stale:
            raise
        rejected = Counter(
            _normalized_text(sentence)
            for key in ("removed_sentences", "unsupported_scientific_claim_sentences")
            for sentence in (exc.detail or {}).get(key, ())
        )
        affected = {(item.section, item.sentence_key) for item in stale}
        drops = []
        for span in sentence_spans(scaffold):
            raw = _normalized_text(scaffold[span.start:span.end])
            if rejected[raw] and (span.section, span.key) in affected:
                drops.append(span)
                rejected[raw] -= 1
        sentence_drops = tuple(drops)
        if not sentence_drops:
            raise
        checked, removed = enforce(delete_spans(scaffold, sentence_drops))
    dropped_keys = {(span.section, span.key) for span in sentence_drops}
    feedback = tuple(item.feedback(removed=(item.section, item.sentence_key) in dropped_keys) for item in stale)
    return checked, removed, sentence_drops, feedback


def numeric_removal_feedback(
    manuscript: str, spans: Sequence[tuple[int, int, tuple[int, ...]]],
) -> tuple[list[SentenceSpan], list[SentenceSpan], tuple[RepairFeedback, ...]]:
    """Describe the exact numeric deletions without carrying the literals."""
    candidates = sentence_spans(manuscript, include_structural=True)
    before = [span for span in candidates if not span.structural]
    removed = []
    feedback = []
    ordinal = 0
    observed: set[SentenceSpan] = set()
    for start, end, positions in spans:
        # An adjacent heading can fall inside the numeric deletion window.
        # Count a Markdown line only when it contains a rejected occurrence.
        affected = [
            span for span in candidates
            if start < span.end and end > span.start and span not in observed
            and (not span.structural or any(span.start <= offset < span.end for offset in positions))
        ]
        before.extend(span for span in affected if span.structural)
        if not affected:
            # A number in a heading can be rejected by STRICT too.
            raw = manuscript[start:end]
            span = SentenceSpan(
                _section_at(manuscript, start + 1), start, end, sentence_key(raw), _sentence_evidence_names(raw),
            )
            before.append(span)
            affected.append(span)
        removed.extend(affected)
        observed.update(affected)
        numbered = [(position, f"number_{i}") for i, position in enumerate(positions, ordinal + 1)]
        ordinal += len(positions)
        for span in affected:
            ids = [ident for offset, ident in numbered if span.start <= offset < span.end]
            code = NUMERIC_REMOVAL if ids else DEPENDENT_REMOVAL
            feedback.append(RepairFeedback(code, _section_owner(span.section), {
                "number_ids": ids or [ident for _, ident in numbered],
            }))
    return before, removed, tuple(feedback)


def removal_losses(
    before: Sequence[SentenceSpan], removed: Sequence[SentenceSpan], headline_keys: Sequence[str],
) -> tuple[RepairFeedback, ...]:
    totals = Counter(span.section for span in before)
    losses = Counter(span.section for span in removed)
    headline_sections = {
        span.section for span in removed
        if any(key and key in span.key for key in headline_keys)
    }
    return tuple(
        RepairFeedback(LOSS_STOP, _section_owner(section), {
            "sentences_before": totals[section],
            "sentences_removed": count,
            "more_than_half_removed": count * 2 > totals[section],
            "headline_result_removed": section in headline_sections,
        })
        for section, count in losses.items()
        if count * 2 > totals[section] or section in headline_sections
    )

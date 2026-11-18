"""Bounded final manuscript audit repair with immutable scientific authority.

The normal Writer owns section dispatch; the normal binding stage owns STRICT
numeric/evidence checks. This owner only feeds repairable audit findings back
to those owners, records the attempt, and refuses changes to existing facts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from ..authority.evidence_store import EvidenceEnforcementError
from ..bounded_repair import BoundedRepair, RepairFeedback
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
_CODE = re.compile(r"\b(?:MANUSCRIPT_[A-Z_]+|OWNER_CLAIM_TARGET_ABSENT|writer_question_part_uncovered)\b")
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
    keys = ["title", *(match[1].strip().casefold() for match in headings)]
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
    return sorted({item.code for item in manuscript_repair_feedback(errors)})


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
    """A failure, or a filtered numeric sentence, that no revision may paper over."""
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
    if _binding_invariant_failed(findings[start:], binding):
        raise EvidenceEnforcementError("MANUSCRIPT_AUDIT_NON_REPAIRABLE: repaired evidence or numeric checks failed.")
    if protected - _numeric_authority(binding):
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
        del findings[start:]
        raise
    regressions = manuscript_repair_regressions(prior_feedback, feedback)
    if regressions or len(_coverage_findings(checked)) >= len(_coverage_findings(binding)):
        row.update(status="rejected", regressions=[item.as_dict() for item in regressions])
        del findings[start:]
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
    if skip_repair:
        _persist(run_dir, evidence, receipt)
        return current, binding
    errors, prior_feedback = _repair_targets(current, binding, section_errors)
    definition_errors = recorded_definition_section_errors(binding.bound, context)
    if definition_errors:
        receipt.update(status="non_repairable", remaining_codes=["MANUSCRIPT_DEFINITION_NOT_FROM_DICTIONARY"])
        _persist(run_dir, evidence, receipt)
        raise EvidenceEnforcementError(
            "MANUSCRIPT_DEFINITION_NOT_FROM_DICTIONARY: prose changes the sealed definition.",
            detail={"sections": sorted(definition_errors)},
        )
    if errors and _binding_invariant_failed(findings[start:], binding):
        receipt.update(status="non_repairable", remaining_codes=["MANUSCRIPT_AUDIT_NON_REPAIRABLE", *_codes(errors)])
        _persist(run_dir, evidence, receipt)
        raise EvidenceEnforcementError("MANUSCRIPT_AUDIT_NON_REPAIRABLE: numeric or evidence checks failed.")
    _accept_binding(findings, binding)
    if not errors:
        if coverage := _coverage_findings(binding):
            receipt.update(status="non_repairable", remaining_codes=_remaining_codes(errors, binding),
                           unresolved_question_parts=[finding.detail for finding in coverage])
        _persist(run_dir, evidence, receipt)
        return current, binding

    last_rejection: tuple[RepairFeedback, ...] = ()
    while errors and repair.available:
        request = dict(errors)
        for item in last_rejection:
            keys = (item.path,) if item.path in errors else tuple(errors)
            for key in keys:
                names = item.facts.get("missing_subsections", ())
                if names:
                    detail = ", ".join(names)
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
                last_rejection = manuscript_repair_regressions(prior_feedback, next_feedback)
                if repair.finish(
                    "unresolved" if _remaining_codes(next_errors, checked) else "resolved",
                    findings=next_feedback, regressions=last_rejection,
                    numeric_checks="pass", evidence_checks="pass",
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
            if detail.get("check") == "section_tokens":
                repair.finish("hard_stop", findings=last_rejection, error_type=type(exc).__name__)
                row.update(status="rejected", invariant_stop=True)
                receipt.update(status="rejected", remaining_codes=codes)
                _persist(run_dir, evidence, receipt)
                raise
            repair.finish("rejected", findings=last_rejection, error_type=type(exc).__name__)
        except Exception as exc:
            codes = _CODE.findall(str(exc)) or ["MANUSCRIPT_AUDIT_REPAIR_FAILED"]
            repair.finish("rejected", findings=tuple(RepairFeedback(code, "manuscript") for code in codes),
                          error_type=type(exc).__name__)
            receipt.update(status="rejected", remaining_codes=codes)
            _persist(run_dir, evidence, receipt)
            raise
        if last_rejection:
            del findings[candidate_start:]
        remaining = _remaining_codes(errors, binding, last_rejection)
        if not remaining:
            row["status"] = "pass"
        receipt.update(status="exhausted" if errors and not repair.available else row["status"],
                       remaining_codes=remaining)
        _persist(run_dir, evidence, receipt)
    if errors and not repair.available and _coverage_findings(binding):
        try:
            fallback = _host_question_repair(
                current, binding, bind, section_errors, findings, context, prior_feedback, receipt,
            )
        except EvidenceEnforcementError:
            _persist(run_dir, evidence, receipt)
            raise
        if fallback:
            current, binding, errors = fallback
            _accept_binding(findings, binding)
            if not errors:
                last_rejection = ()
            remaining = _remaining_codes(errors, binding, last_rejection)
            receipt.update(status="exhausted" if remaining else "pass", remaining_codes=remaining)
        _persist(run_dir, evidence, receipt)
    if last_rejection:
        raise EvidenceEnforcementError(
            "MANUSCRIPT_AUDIT_REPAIR_FAILED: " + ", ".join(receipt["remaining_codes"]),
            detail={"code": "MANUSCRIPT_AUDIT_REPAIR_FAILED", "audit_codes": receipt["remaining_codes"]},
        )
    return current, binding

"""Bounded final manuscript audit repair with immutable scientific authority.

The normal Writer owns section dispatch; the normal binding stage owns STRICT
numeric/evidence checks. This owner only feeds repairable audit findings back
to those owners, records the attempt, and refuses changes to existing facts.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from ..authority.evidence_store import EvidenceEnforcementError
from ..bounded_repair import BoundedRepair, RepairFeedback
from .manuscript_labels import recorded_definition_section_errors
from .manuscript_sections import (
    ManuscriptSectionContractError,
    manuscript_binding_tokens,
    manuscript_repair_feedback,
    manuscript_repair_regressions,
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
_CODE = re.compile(r"\b(?:MANUSCRIPT_[A-Z_]+|OWNER_CLAIM_TARGET_ABSENT)\b")
_MARKER = re.compile(
    r"(?P<literal>[+−-]?(?:"
    r"(?:\d+(?:\.\d+)?|\.\d+)\s*[×x]\s*10[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+"
    r"|(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?))"
    r"(?:%?[ \t]*)\[\^(?P<claim>claim_\d+)\]"
)


def _sections(text: str) -> dict[str, str]:
    pieces = re.split(r"(?m)^##\s+([^\n]+)\n", text)
    return {"title": pieces[0].strip(), **{
        pieces[index].strip().lower(): pieces[index + 1].strip()
        for index in range(1, len(pieces), 2)
    }}


def _codes(errors: Mapping[str, Sequence[str]]) -> list[str]:
    return sorted({code for rows in errors.values() for row in rows for code in _CODE.findall(row)})


def _numeric_authority(binding: Any) -> Counter:
    """Ignore footnote renumbering, retain section, value and source identity."""
    claims = getattr(binding, "numeric_binding_map", {})
    result: Counter = Counter()
    for section, text in _sections(binding.bound).items():
        for match in _MARKER.finditer(text):
            claim = claims.get(match["claim"])
            if claim is not None:
                payload = claim.to_dict() if callable(getattr(claim, "to_dict", None)) else dict(claim)
                result[(section, match["literal"], json.dumps(payload, sort_keys=True, default=str))] += 1
    return result


def _require_preserved_sections(before: str, after: str, keys: Sequence[str]) -> None:
    original, candidate = _sections(before), _sections(after)
    names = set(keys)
    for section in original.keys() | candidate.keys():
        text = original.get(section, "")
        if section not in names:
            if candidate.get(section) != original.get(section):
                raise EvidenceEnforcementError(
                    "MANUSCRIPT_REPAIR_AUTHORITY_CHANGED: an unaffected section changed.",
                    detail={"check": "unaffected_section", "section": section},
                )
        else:
            missing = Counter(manuscript_binding_tokens(text)) - Counter(manuscript_binding_tokens(candidate.get(section, "")))
            if missing:
                raise EvidenceEnforcementError(
                    "MANUSCRIPT_REPAIR_AUTHORITY_CHANGED: an existing evidence or claim binding changed.",
                    detail={"check": "section_tokens", "section": section, "missing_tokens": dict(missing)},
                )


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
    errors = section_errors(current, binding) if not current.writer_error_message else {}
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
    if not errors:
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
                    detail = "Check the reported section against the previous draft."
                request[key] = (*request[key], f"{item.code}: {detail}")
        prior_feedback = manuscript_repair_feedback(errors)
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
                _require_preserved_sections(current.scaffold, candidate.scaffold, tuple(errors))
                checked = bind(candidate)
                definition_errors = recorded_definition_section_errors(checked.bound, context)
                if definition_errors:
                    raise EvidenceEnforcementError(
                        "MANUSCRIPT_DEFINITION_NOT_FROM_DICTIONARY: prose changes the sealed definition.",
                        detail={"sections": sorted(definition_errors)},
                    )
                if _binding_invariant_failed(findings[candidate_start:], checked):
                    raise EvidenceEnforcementError("MANUSCRIPT_AUDIT_NON_REPAIRABLE: repaired evidence or numeric checks failed.")
                if protected - _numeric_authority(checked):
                    raise EvidenceEnforcementError("MANUSCRIPT_REPAIR_AUTHORITY_CHANGED: an existing bound number or source changed.")
                next_errors = section_errors(candidate, checked)
                next_feedback = manuscript_repair_feedback(next_errors)
                last_rejection = manuscript_repair_regressions(prior_feedback, next_feedback)
                if repair.finish(
                    "unresolved" if next_errors else "resolved",
                    findings=next_feedback, regressions=last_rejection,
                    numeric_checks="pass", evidence_checks="pass",
                    output_sha256=hashlib.sha256(candidate.scaffold.encode()).hexdigest(),
                    output_bound_sha256=hashlib.sha256(checked.bound.encode()).hexdigest(),
                ):
                    current, binding, errors = candidate, checked, next_errors
        except ManuscriptSectionContractError as exc:
            last_rejection = (RepairFeedback(
                "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY", exc.section_name.lower(),
                {"missing_subsections": list(exc.missing_subsections)},
            ),)
            repair.finish("rejected", findings=last_rejection)
        except EvidenceEnforcementError as exc:
            detail = exc.detail or {}
            if detail.get("check") == "unaffected_section":
                last_rejection = (RepairFeedback(
                    "MANUSCRIPT_REPAIR_REGRESSION", detail["section"], {"finding": "unaffected_section"},
                ),)
                repair.finish("rejected", regressions=last_rejection)
            else:
                codes = _CODE.findall(str(exc)) or ["MANUSCRIPT_AUDIT_NON_REPAIRABLE"]
                repair.finish("hard_stop", findings=tuple(RepairFeedback(code, "manuscript") for code in codes))
                row.update(status="rejected", invariant_stop=True, error_type=type(exc).__name__)
                if detail:
                    row["authority_failure"] = detail
                receipt.update(status="rejected", remaining_codes=codes)
                _persist(run_dir, evidence, receipt)
                raise
        except Exception as exc:
            codes = _CODE.findall(str(exc)) or ["MANUSCRIPT_AUDIT_REPAIR_FAILED"]
            invariant_stop = isinstance(exc, EvidenceEnforcementError) or "MANUSCRIPT_REPAIR_AUTHORITY_CHANGED" in codes
            repair.finish("hard_stop" if invariant_stop else "rejected",
                          findings=tuple(RepairFeedback(code, "manuscript") for code in codes),
                          error_type=type(exc).__name__)
            if isinstance(exc, EvidenceEnforcementError) and exc.detail:
                row["authority_failure"] = exc.detail
            row["status"] = "rejected"
            row["invariant_stop"] = invariant_stop
            receipt.update(status="rejected", remaining_codes=codes)
            _persist(run_dir, evidence, receipt)
            raise
        if last_rejection:
            del findings[candidate_start:]
        if not errors:
            row["status"] = "pass"
        receipt.update(
            status="exhausted" if errors and not repair.available else row["status"],
            remaining_codes=sorted(set(_codes(errors)) | {item.code for item in last_rejection}),
        )
        _persist(run_dir, evidence, receipt)
    if last_rejection:
        raise EvidenceEnforcementError(
            "MANUSCRIPT_AUDIT_REPAIR_FAILED: " + ", ".join(receipt["remaining_codes"]),
            detail={"code": "MANUSCRIPT_AUDIT_REPAIR_FAILED", "audit_codes": receipt["remaining_codes"]},
        )
    return current, binding

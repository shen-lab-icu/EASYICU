"""One final-audit repair budget, with immutable scientific authority.

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
from .manuscript_sections import ManuscriptSectionContractError, manuscript_binding_tokens

# Preserve the final-quality path's existing one-revision budget.
MAX_MANUSCRIPT_AUDIT_REPAIRS = 1
REPAIRABLE_MANUSCRIPT_CODES = frozenset({
    "MANUSCRIPT_TITLE_MISSING", "MANUSCRIPT_KEYWORDS_MISSING",
    "MANUSCRIPT_SECTION_MISSING", "MANUSCRIPT_SECTION_EMPTY",
    "MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY", "MANUSCRIPT_RESULT_SUBSECTION_CALLOUT_ONLY",
    "MANUSCRIPT_ABSTRACT_LABEL_MISSING_OR_EMPTY", "MANUSCRIPT_ABSTRACT_CONCLUSION_DUPLICATES_RESULTS",
    "MANUSCRIPT_CONCLUSION_WITHOUT_INTERPRETATION", "MANUSCRIPT_BASELINE_METHODS_INCOMPLETE",
    "MANUSCRIPT_SECTION_TRUNCATED", "MANUSCRIPT_INTERNAL_TERM_EXPOSED",
    "MANUSCRIPT_VARIABLE_DEFINITION_CONTEXT_MISSING", "MANUSCRIPT_ADJUSTMENT_SET_CONFLICT",
    "MANUSCRIPT_DISPLAY_NOT_CALLED_OUT", "MANUSCRIPT_REPEATED_PARAGRAPH",
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
    for section, text in original.items():
        if section not in names:
            if candidate.get(section) != text:
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


def _non_repairable(findings: Sequence[Any], binding: Any) -> bool:
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
) -> tuple[Any, Any]:
    """Re-enter the existing Writer and binding owners at most once.

    Existing bound numbers and their evidence identities may not disappear or
    change. A missing section may acquire additional facts only through the
    unchanged STRICT binder. Evidence/numeric failures never fund a rewrite.
    """
    start = len(findings)
    receipt: dict[str, Any] = {
        "schema_version": "easyicu.manuscript_audit_repair/1",
        "max_attempts": MAX_MANUSCRIPT_AUDIT_REPAIRS, "attempts": [], "status": "not_needed",
    }
    # A draft that needs no revision keeps its normal path: a STRICT-filter
    # warning is routine there and the final gates judge it as before.  Only a
    # draft that asks for a revision is refused one after a numeric/evidence
    # failure, so filtering can never make that failure editorially repairable.
    current = draft(None)
    binding = bind(current)
    if skip_repair or current.writer_error_message:
        _persist(run_dir, evidence, receipt)
        return current, binding
    errors = section_errors(current, binding)
    if not errors:
        _persist(run_dir, evidence, receipt)
        return current, binding
    if _non_repairable(findings[start:], binding):
        receipt.update(status="non_repairable", remaining_codes=["MANUSCRIPT_AUDIT_NON_REPAIRABLE", *_codes(errors)])
        _persist(run_dir, evidence, receipt)
        raise EvidenceEnforcementError("MANUSCRIPT_AUDIT_NON_REPAIRABLE: numeric or evidence checks failed.")
    codes = _codes(errors)
    if not codes or set(codes) - REPAIRABLE_MANUSCRIPT_CODES:
        receipt.update(status="non_repairable", remaining_codes=codes)
        _persist(run_dir, evidence, receipt)
        return current, binding
    original_authority = _numeric_authority(binding)
    for attempt in range(MAX_MANUSCRIPT_AUDIT_REPAIRS):
        row: dict[str, Any] = {
            "attempt": attempt + 1, "section_keys": list(errors), "audit_codes": _codes(errors),
            "input_sha256": hashlib.sha256(current.scaffold.encode()).hexdigest(),
            "input_bound_sha256": hashlib.sha256(binding.bound.encode()).hexdigest(),
            "protected_numeric_occurrences": sum(original_authority.values()),
            "numeric_checks": "pending", "evidence_checks": "pending", "status": "running",
        }
        receipt["attempts"].append(row)
        receipt["status"] = "running"
        _persist(run_dir, evidence, receipt)
        candidate_start = len(findings)
        try:
            candidate = draft((current.scaffold, errors))
            if candidate.writer_error_message:
                raise EvidenceEnforcementError(
                    "MANUSCRIPT_AUDIT_REPAIR_FAILED: " + ", ".join(_codes(errors)),
                    detail={"code": "MANUSCRIPT_AUDIT_REPAIR_FAILED", "audit_codes": _codes(errors)},
                )
            _require_preserved_sections(current.scaffold, candidate.scaffold, tuple(errors))
            checked = bind(candidate)
            if _non_repairable(findings[candidate_start:], checked):
                raise EvidenceEnforcementError("MANUSCRIPT_AUDIT_NON_REPAIRABLE: repaired evidence or numeric checks failed.")
            if original_authority - _numeric_authority(checked):
                raise EvidenceEnforcementError("MANUSCRIPT_REPAIR_AUTHORITY_CHANGED: an existing bound number or source changed.")
        except Exception as exc:
            row.update(status="rejected", error_type=type(exc).__name__,
                       error_codes=_CODE.findall(str(exc)))
            if isinstance(exc, EvidenceEnforcementError) and exc.detail:
                row["authority_failure"] = exc.detail
            receipt.update(status="rejected", remaining_codes=row["error_codes"] or _codes(errors))
            if isinstance(exc, ManuscriptSectionContractError):
                receipt["remaining_codes"] = sorted(set(receipt["remaining_codes"]) | {"MANUSCRIPT_SUBSECTION_MISSING_OR_EMPTY"})
                _persist(run_dir, evidence, receipt)
                raise EvidenceEnforcementError(
                    "MANUSCRIPT_AUDIT_REPAIR_FAILED: " + ", ".join(receipt["remaining_codes"]),
                    detail={"code": "MANUSCRIPT_AUDIT_REPAIR_FAILED", "audit_codes": receipt["remaining_codes"]},
                ) from exc
            _persist(run_dir, evidence, receipt)
            raise
        row.update(status="checked", numeric_checks="pass", evidence_checks="pass",
                   output_sha256=hashlib.sha256(candidate.scaffold.encode()).hexdigest(),
                   output_bound_sha256=hashlib.sha256(checked.bound.encode()).hexdigest())
        current, binding = candidate, checked
        errors = section_errors(current, binding)
        receipt.update(status="exhausted" if errors else "pass", remaining_codes=_codes(errors))
        row["status"] = receipt["status"]
        _persist(run_dir, evidence, receipt)
        if not errors:
            break
    return current, binding

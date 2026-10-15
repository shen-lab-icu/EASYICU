"""Place source-owned method facts and audit their post-filter coverage."""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..authority.evidence_store import EvidenceStore
from ..authority.manuscript_method_facts import ManuscriptMethodFact
from ..schema import ValidationFinding


class SourceFact(Protocol):
    """A host sentence its section carries exactly as its source states it.

    A ``ManuscriptMethodFact`` is one; so is a limitation the approved plan
    review left to the study (``reporting.plan_review_limitations``).
    """

    @property
    def section(self) -> str: ...

    @property
    def scaffold(self) -> str: ...

    @property
    def source_field(self) -> str: ...

    @property
    def source_sha256(self) -> str: ...


def _limitations_span(scaffold: str) -> tuple[int, int] | None:
    limitations = re.search(r"^## Limitations[ \t]*$", scaffold, re.M)
    if limitations is None:
        return None
    following = re.search(r"^##\s+", scaffold[limitations.end() :], re.M)
    end = limitations.end() + following.start() if following else len(scaffold)
    return limitations.end(), end


def _variables_span(scaffold: str) -> tuple[int, int] | None:
    methods = re.search(r"^## Methods[ \t]*$", scaffold, re.M)
    if methods is None:
        return None
    following = re.search(r"^##\s+", scaffold[methods.end() :], re.M)
    end = methods.end() + following.start() if following else len(scaffold)
    variables = re.search(r"^### Variables[ \t]*$", scaffold[methods.end() : end], re.M)
    if variables is None:
        return None
    position = methods.end() + variables.end()
    following_subsection = re.search(r"^###\s+", scaffold[position:end], re.M)
    return (
        position,
        position + following_subsection.start() if following_subsection else end,
    )


_SECTION_SPANS = {"variables": _variables_span, "limitations": _limitations_span}


def place_manuscript_method_facts(
    scaffold: str,
    facts: Sequence[SourceFact],
) -> tuple[str, tuple[str, ...]]:
    """Place exact source facts in their sections without rewriting Writer prose."""

    placed: list[str] = []
    for section, span_of in _SECTION_SPANS.items():
        span = span_of(scaffold)
        if span is None:
            continue
        position, section_end = span
        existing = scaffold[position:section_end].splitlines()
        missing = [
            fact
            for fact in facts
            if fact.section == section and fact.scaffold not in existing
        ]
        if not missing:
            continue
        block = "\n\n" + "\n\n".join(fact.scaffold for fact in missing) + "\n\n"
        scaffold = scaffold[:position] + block + scaffold[position:].lstrip("\n")
        placed.extend(fact.source_field for fact in missing)
    return scaffold, tuple(placed)


def missing_bound_method_facts(
    bound: str,
    facts: Sequence[SourceFact],
    bind: Callable[[str], str],
) -> tuple[str, ...]:
    """Fail closed if a later provenance filter removed a required source fact."""

    lines: dict[str, list[str]] = {}
    for section, span_of in _SECTION_SPANS.items():
        span = span_of(bound)
        lines[section] = bound[span[0] : span[1]].splitlines() if span else []
    return tuple(
        fact.source_field
        for fact in facts
        if bind(fact.scaffold) not in lines[fact.section]
    )


def released_fact_lines(
    text: str,
    facts: Sequence[ManuscriptMethodFact],
    bind: Callable[[str], str] | None = None,
) -> tuple[tuple[int, int, ManuscriptMethodFact], ...]:
    """Find exact host fact lines with typed numbers in their declared sections."""

    found: list[tuple[int, int, ManuscriptMethodFact]] = []
    for section, span_of in _SECTION_SPANS.items():
        by_line = {
            (bind(fact.scaffold) if bind is not None else fact.scaffold): fact
            for fact in facts
            if fact.section == section and fact.typed_values
        }
        span = span_of(text) if by_line else None
        if span is None:
            continue
        position = span[0]
        for line in text[span[0] : span[1]].splitlines(keepends=True):
            bare = line.rstrip("\r\n")
            fact = by_line.get(bare)
            if fact is not None:
                found.append((position, position + len(bare), fact))
            position += len(line)
    return tuple(found)


def method_fact_binding(
    evidence: EvidenceStore,
    per_step_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The verified facts and their evidence-bound lines for the numeric binder."""

    return {
        "method_facts": evidence.manuscript_method_facts(per_step_records),
        "bind_fact": lambda text: evidence.bind_manuscript(
            text,
            per_step_records=per_step_records,
            reader_labels=None,
        ),
    }


def project_source_method_facts(
    scaffold: str,
    *,
    evidence: EvidenceStore,
    per_step_records: Sequence[Mapping[str, Any]],
    extra_facts: Sequence[SourceFact] = (),
) -> tuple[str, ValidationFinding | None]:
    """Place the run's source facts, then ``extra_facts``, in one ordered block."""

    facts = (*evidence.manuscript_method_facts(per_step_records), *extra_facts)
    scaffold, fields = place_manuscript_method_facts(scaffold, facts)
    if not fields:
        return scaffold, None
    sections = {fact.section for fact in facts if fact.source_field in fields}
    restored = " and ".join(
        text
        for section, text in (
            ("variables", "exact source definitions in Methods"),
            ("limitations", "exact source limitations in Limitations"),
        )
        if section in sections
    )
    return scaffold, ValidationFinding(
        validator="evidence_bound_writer",
        severity="info",
        message=f"Restored {restored}; no clinical or result authority was added.",
        detail={
            "reason_code": "writer_source_method_facts_placed",
            "source_fields": list(fields),
            "source_sha256": facts[0].source_sha256,
        },
    )


def audit_bound_source_method_facts(
    bound: str,
    *,
    evidence: EvidenceStore,
    per_step_records: Sequence[Mapping[str, Any]],
) -> ValidationFinding | None:
    # A method fact cites evidence and never carries a scientific claim, so no
    # claim labels apply.
    missing = missing_bound_method_facts(
        bound,
        evidence.manuscript_method_facts(per_step_records),
        lambda text: evidence.bind_manuscript(
            text, per_step_records=per_step_records, reader_labels=None,
        ),
    )
    if not missing:
        return None
    return ValidationFinding(
        validator="evidence_bound_writer",
        severity="error",
        message="Required source definitions did not survive manuscript provenance validation.",
        detail={
            "reason_code": "writer_source_method_facts_missing",
            "source_fields": list(missing),
        },
    )

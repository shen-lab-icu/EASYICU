"""Place source-owned method facts and audit their post-filter coverage."""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from ..authority.evidence_store import EvidenceStore
from ..authority.manuscript_method_facts import ManuscriptMethodFact
from ..schema import ValidationFinding


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


def place_manuscript_method_facts(
    scaffold: str,
    facts: Sequence[ManuscriptMethodFact],
) -> tuple[str, tuple[str, ...]]:
    """Place exact source facts in Variables without rewriting Writer prose."""

    span = _variables_span(scaffold)
    if span is None:
        return scaffold, ()
    position, variable_end = span
    existing = scaffold[position:variable_end].splitlines()
    missing = [fact for fact in facts if fact.scaffold not in existing]
    if not missing:
        return scaffold, ()
    block = "\n\n" + "\n\n".join(fact.scaffold for fact in missing) + "\n\n"
    return scaffold[:position] + block + scaffold[position:].lstrip("\n"), tuple(
        fact.source_field for fact in missing
    )


def missing_bound_method_facts(
    bound: str,
    facts: Sequence[ManuscriptMethodFact],
    bind: Callable[[str], str],
) -> tuple[str, ...]:
    """Fail closed if a later provenance filter removed a required source fact."""

    span = _variables_span(bound)
    lines = bound[span[0] : span[1]].splitlines() if span is not None else []
    return tuple(
        fact.source_field for fact in facts if bind(fact.scaffold) not in lines
    )


def project_source_method_facts(
    scaffold: str,
    *,
    evidence: EvidenceStore,
    per_step_records: Sequence[Mapping[str, Any]],
) -> tuple[str, ValidationFinding | None]:
    facts = evidence.manuscript_method_facts(per_step_records)
    scaffold, fields = place_manuscript_method_facts(scaffold, facts)
    if not fields:
        return scaffold, None
    return scaffold, ValidationFinding(
        validator="evidence_bound_writer",
        severity="info",
        message="Restored exact source definitions in Methods; no clinical or result authority was added.",
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
    missing = missing_bound_method_facts(
        bound,
        evidence.manuscript_method_facts(per_step_records),
        lambda text: evidence.bind_manuscript(text, per_step_records=per_step_records),
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

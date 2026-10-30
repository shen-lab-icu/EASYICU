"""Check a plan's literature references against the retained source excerpts.

This is an integrity check, not a scientific appraisal. An exact excerpt match
cannot establish that the paper's conclusions apply to the current question.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable

from .literature_design_authority import _SOURCE_RESOURCE_TITLE, _source_excerpt, _text
from .official_database_resources import canonical_resource_database, match_official_source_identity


@dataclass(frozen=True)
class SourceIntegrityIssue:
    citation_key: str
    reason: str
    severity: str


def inspect_plan_literature_sources(
    plan: Any, bundle: Any, *, database: str | None = None,
) -> tuple[SourceIntegrityIssue, ...]:
    """Report mismatched or unavailable evidence only for sources a plan binds.

    Abstract cards can be checked against the exact retained excerpt. Full-text
    cards currently retain a declared digest and locator but no source bytes in
    this boundary, so they cannot be promoted to verified here. Old bundles are
    read without rewriting or synthesizing review authority.
    """
    cited = {
        key for step in plan.steps for key in step.literature_citation_keys
    }
    selection = getattr(plan, "design_selection", None)
    if selection is not None:
        cited.update(selection.selected.literature_citation_keys)
        cited.update(
            key
            for decision in selection.selected.literature_design_decisions
            for key in decision.citation_keys
        )
    if not cited:
        return ()
    records = bundle.citations if bundle is not None else []
    cards = bundle.design_evidence_cards if bundle is not None else []
    decisions = bundle.screening_decisions if bundle is not None else []
    record_count = Counter(record.key for record in records)
    card_count = Counter(card.citation_key for card in cards)
    by_record = {record.key: record for record in records}
    by_card = {card.citation_key: card for card in cards}
    dispositions: dict[str, set[str]] = {}
    for decision in decisions:
        dispositions.setdefault(decision.citation_key, set()).add(decision.disposition)
    issues: list[SourceIntegrityIssue] = []
    for key in sorted(cited):
        if record_count[key] != 1 or card_count[key] > 1:
            issues.append(SourceIntegrityIssue(key, "source_missing_or_ambiguous", "error"))
            continue
        if dispositions.get(key) and dispositions[key] != {"include"}:
            issues.append(SourceIntegrityIssue(key, "source_not_included", "error"))
            continue
        card = by_card.get(key)
        if card is None:
            issues.append(SourceIntegrityIssue(key, "no_source_evidence_card", "warning"))
            continue
        record = by_record[key]
        if any(item.review_role == "data_source" for item in card.evidence):
            identity = record.review.source_resource_identity if record.review is not None else None
            if identity is None and not _SOURCE_RESOURCE_TITLE.search(_text(record.title)):
                issues.append(SourceIntegrityIssue(key, "database_role_not_resource_description", "warning"))
            if identity is None:
                issues.append(SourceIntegrityIssue(key, "source_resource_identity_unverified", "warning"))
            elif match_official_source_identity(
                identity.database, record, registry_version=identity.registry_version,
            ) != identity:
                issues.append(SourceIntegrityIssue(key, "source_resource_identity_mismatch", "error"))
            elif database is not None and canonical_resource_database(database) != identity.database:
                issues.append(SourceIntegrityIssue(key, "source_resource_database_mismatch", "error"))
            else:
                issues.append(SourceIntegrityIssue(key, "registered_dataset_version_not_assessed", "warning"))
        if card.access_mode != "abstract":
            issues.append(SourceIntegrityIssue(key, "full_text_bytes_not_checked", "warning"))
            continue
        excerpt = _source_excerpt(record)
        if not excerpt:
            issues.append(SourceIntegrityIssue(key, "source_excerpt_unavailable", "warning"))
            continue
        if sha256(excerpt.encode("utf-8")).hexdigest() != card.abstract_sha256:
            issues.append(SourceIntegrityIssue(key, "abstract_digest_mismatch", "error"))
            continue
        normalized_excerpt = " ".join(excerpt.split())
        if any(" ".join(item.source_backed_summary.split()) not in normalized_excerpt
               for item in card.evidence):
            issues.append(SourceIntegrityIssue(key, "fact_not_in_source_excerpt", "error"))
            continue
        if record.review is not None and any(
            item.review_role is not None and item.review_role not in record.review.roles
            for item in card.evidence
        ):
            issues.append(SourceIntegrityIssue(key, "evidence_role_mismatch", "error"))
    return tuple(issues)


def issue_keys(issues: Iterable[SourceIntegrityIssue], severity: str) -> list[str]:
    return sorted({issue.citation_key for issue in issues if issue.severity == severity})

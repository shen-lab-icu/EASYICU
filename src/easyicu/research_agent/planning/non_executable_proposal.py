"""Internal, non-executable research proposals with verified evidence references.

This is not an AnalysisPlan, an approval, or a source capability assertion.
It has no product route. Evidence integrity is distinct from clinical relevance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.evidence_snapshot import load_current_evidence_snapshot
from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.literature import LiteratureBundle, manuscript_citable_keys
from easyicu.research_agent.schema import EvidenceRecord


class ReadOnlyEvidenceRegistry:
    """Read the authority snapshot without EvidenceStore's recovery writes."""

    def __init__(self, root: Path):
        self.root = Path(root)
        if not (self.root / "evidence").is_dir():
            raise ValueError("proposal_evidence_store_unavailable")
        self.records = tuple(EvidenceRecord.model_validate(row)
                             for row in load_current_evidence_snapshot(self.root).records)

    def get(self, evidence_id: str):
        return next((record for record in self.records if record.evidence_id == evidence_id), None)

    def path(self, record):
        return verified_run_evidence_path(self.root, record)

    def reload(self):
        return ReadOnlyEvidenceRegistry(self.root)


class CandidateEvidenceRegistry:
    """Read original run evidence and separately registered candidate evidence."""

    def __init__(self, run_root: Path, candidate_root: Path):
        self.root = Path(run_root)
        self.candidate_root = Path(candidate_root)
        self.run = ReadOnlyEvidenceRegistry(self.root)
        self.candidate = (ReadOnlyEvidenceRegistry(self.candidate_root)
                          if (self.candidate_root / 'evidence').is_dir() else None)

    def get(self, evidence_id: str):
        return self.run.get(evidence_id) or (self.candidate.get(evidence_id) if self.candidate else None)

    def path(self, record):
        if self.run.get(record.evidence_id) == record:
            return self.run.path(record)
        return self.candidate.path(record) if self.candidate else None

    def reload(self):
        return CandidateEvidenceRegistry(self.root, self.candidate_root)


class ProposedReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str = Field(min_length=1, max_length=200)
    kind: Literal["source", "literature"]
    citation_key: str | None = None
    source_evidence_id: str | None = None
    verbatim_excerpt: str | None = Field(default=None, min_length=12, max_length=240)

    @model_validator(mode="after")
    def check_key(self):
        if (self.kind == "literature") != bool(self.citation_key):
            raise ValueError("literature references require an exact citation key")
        if self.kind == "literature" and not (self.source_evidence_id and self.verbatim_excerpt):
            raise ValueError("literature references require registered original text and an exact excerpt")
        if self.kind == "source" and (self.source_evidence_id or self.verbatim_excerpt):
            raise ValueError("source references cannot carry literature card fields")
        return self


class ProposedDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    decision_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2000)
    references: tuple[ProposedReference, ...] = ()
    unresolved: tuple[str, ...] = ()


class EvidenceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: Literal["source", "literature"]
    citation_key: str | None = None
    source_evidence_id: str | None = None
    source_evidence_sha256: str | None = None
    verbatim_excerpt: str | None = None
    literature_role: str | None = None
    literature_access_mode: str | None = None
    applicability_verified: Literal[False] = False


class ProposalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    decision_id: str
    text: str
    bindings: tuple[EvidenceBinding, ...]
    unresolved: tuple[str, ...]
    unbound_evidence: tuple[Literal["source", "literature"], ...]
    evidence_status: Literal["unbound", "registered_digest_verified"]

    @model_validator(mode="after")
    def check_binding_status(self):
        missing = tuple(kind for kind in ("source", "literature")
                        if not any(binding.kind == kind for binding in self.bindings))
        expected_status = "registered_digest_verified" if self.bindings else "unbound"
        if self.unbound_evidence != missing or self.evidence_status != expected_status:
            raise ValueError("proposal_binding_status_invalid")
        return self


class NonExecutableProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["easyicu.non-executable-proposal/1"] = "easyicu.non-executable-proposal/1"
    original_question: str = Field(max_length=3000)
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions: tuple[ProposalDecision, ...]
    unresolved: tuple[str, ...]
    executable: Literal[False] = False
    review_approved: Literal[False] = False
    data_access_authorized: Literal[False] = False
    evidence_applicability_verified: Literal[False] = False


def _bound_record(store: EvidenceStore, reference: ProposedReference) -> EvidenceBinding:
    # Aliases and prefix resolution are intentionally rejected: binding is to
    # the exact registered identity and its verified current bytes.
    record = store.get(reference.evidence_id)
    if record is None or record.evidence_id != reference.evidence_id:
        raise ValueError("proposal_evidence_not_registered")
    path = store.path(record)
    if path is None:
        raise ValueError("proposal_evidence_integrity_failed")
    role = access_mode = None
    if reference.kind == "literature":
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record.sha256:
            raise ValueError("proposal_evidence_integrity_failed")
        try:
            bundle = LiteratureBundle.model_validate_json(data)
        except ValueError as exc:
            raise ValueError("proposal_literature_bundle_invalid") from exc
        if reference.citation_key not in manuscript_citable_keys(bundle):
            raise ValueError("proposal_citation_not_retained")
        screens = [item for item in bundle.screening_decisions
                   if item.citation_key == reference.citation_key]
        cards = [card for card in bundle.design_evidence_cards
                 if card.citation_key == reference.citation_key]
        if len(screens) != 1 or screens[0].disposition != "include" or len(cards) != 1:
            raise ValueError("proposal_literature_card_not_unique_or_screened")
        card = cards[0]
        if card.evidence_role != screens[0].evidence_role:
            raise ValueError("proposal_literature_role_mismatch")
        source_record = store.get(reference.source_evidence_id)
        if source_record is None or source_record.evidence_id != reference.source_evidence_id:
            raise ValueError("proposal_original_text_not_registered")
        source_path = store.path(source_record)
        if source_path is None:
            raise ValueError("proposal_original_text_integrity_failed")
        import json
        source_bytes = source_path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != source_record.sha256:
            raise ValueError("proposal_original_text_integrity_failed")
        try:
            source = json.loads(source_bytes)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("proposal_original_text_invalid") from exc
        citation = next(item for item in bundle.citations if item.key == reference.citation_key)
        if (not isinstance(source, dict) or source.get("citation_key") != citation.key
                or not citation.pmid and not citation.doi
                or source.get("pmid") != citation.pmid or source.get("doi") != citation.doi
                or not isinstance(source.get("abstract"), str)):
            raise ValueError("proposal_original_text_identity_mismatch")
        abstract = source["abstract"]
        # This card digest is for the selected extractive source excerpt, not
        # necessarily for the whole original abstract.
        source_excerpt = source.get("source_excerpt", abstract)
        if not isinstance(source_excerpt, str):
            raise ValueError("proposal_original_excerpt_unverified")
        if (card.access_mode != "abstract" or not card.abstract_locator
                or hashlib.sha256(source_excerpt.encode("utf-8")).hexdigest() != card.abstract_sha256
                or reference.verbatim_excerpt not in source_excerpt
                or reference.verbatim_excerpt not in abstract):
            raise ValueError("proposal_original_excerpt_unverified")
        role, access_mode = card.evidence_role, card.access_mode
    else:
        source_record = None
    return EvidenceBinding(evidence_id=record.evidence_id, sha256=record.sha256,
                           kind=reference.kind, citation_key=reference.citation_key,
                           source_evidence_id=reference.source_evidence_id,
                           source_evidence_sha256=source_record.sha256 if source_record else None,
                           verbatim_excerpt=reference.verbatim_excerpt,
                           literature_role=role, literature_access_mode=access_mode)


def build_non_executable_proposal(*, store: EvidenceStore, configuration: Mapping[str, Any],
                                  expected_configuration_sha256: str, original_question: str,
                                  decisions: tuple[ProposedDecision, ...],
                                  unresolved: tuple[str, ...] = ()) -> NonExecutableProposal:
    """Bind candidate wording to the selected configuration bytes and evidence.

    Empty references remain explicitly unbound. Call `verify_proposal` before
    reuse: neither file presence nor digest establishes scientific relevance.
    """
    digest = canonical_sha256(dict(configuration))
    if digest != expected_configuration_sha256:
        raise ValueError("proposal_configuration_mismatch")
    ids = [item.decision_id for item in decisions]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("proposal_decisions_invalid")
    # Reopen the persisted authority to avoid relying on a caller's stale
    # in-memory registry when another process has changed the evidence index.
    store = store.reload() if hasattr(store, 'reload') else ReadOnlyEvidenceRegistry(store.root)
    rows = []
    for item in decisions:
        bindings = tuple(_bound_record(store, ref) for ref in item.references)
        unbound = tuple(kind for kind in ("source", "literature")
                        if not any(binding.kind == kind for binding in bindings))
        rows.append(ProposalDecision(decision_id=item.decision_id, text=item.text,
                                     bindings=bindings, unresolved=item.unresolved,
                                     unbound_evidence=unbound,
                                     evidence_status="registered_digest_verified" if bindings else "unbound"))
    missing = tuple(unresolved)
    if not original_question.strip() and "original_question_unavailable" not in missing:
        missing += ("original_question_unavailable",)
    return NonExecutableProposal(original_question=original_question, configuration_sha256=digest,
                                 decisions=tuple(rows), unresolved=missing)


def verify_proposal(*, store: EvidenceStore, proposal: NonExecutableProposal,
                    configuration: Mapping[str, Any]) -> None:
    """Reject changed configuration, removed/replaced evidence or excluded citations."""
    proposal = NonExecutableProposal.model_validate(proposal.model_dump(mode="json"))
    if canonical_sha256(dict(configuration)) != proposal.configuration_sha256:
        raise ValueError("proposal_configuration_mismatch")
    store = store.reload() if hasattr(store, 'reload') else ReadOnlyEvidenceRegistry(store.root)
    for decision in proposal.decisions:
        for binding in decision.bindings:
            current = _bound_record(store, ProposedReference(evidence_id=binding.evidence_id,
                kind=binding.kind, citation_key=binding.citation_key,
                source_evidence_id=binding.source_evidence_id,
                verbatim_excerpt=binding.verbatim_excerpt))
            if current != binding:
                raise ValueError("proposal_evidence_integrity_failed")

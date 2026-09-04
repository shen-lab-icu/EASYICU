"""One bounded, fail-closed repair pass over STRICT-rejected manuscript prose.

The write phase used to own the decision call, deterministic fallback, repair
application, and STRICT revalidation while ``manuscript_post`` owned the text
edits themselves.  That split left the safety rule visible only by reading two
large modules together.  This owner keeps the orchestration behind one typed
result without introducing another manuscript-state or repair-decision type:
run state remains :class:`ManuscriptState`, and every edit is still expressed
as a :class:`WriterRepairDecision`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..authority.evidence_store import EvidenceEnforcementError
from ..providers.structured_retry import StructuredResponseFailure
from .writer_repair_decision import WriterRepairDecision, drop_every_sentence

DecisionProvider = Callable[..., Sequence[WriterRepairDecision]]
DecisionApplier = Callable[..., tuple[str, List[Dict[str, object]]]]
TargetLocator = Callable[[str, str], Optional[Tuple[int, int]]]

__all__ = ["ManuscriptRepairPass", "ManuscriptRepairResult"]


@dataclass(frozen=True)
class ManuscriptRepairResult:
    """The complete receipt from one bounded repair and STRICT revalidation."""

    scaffold: str
    evidence_repairs: tuple[Mapping[str, object], ...]
    fallback_detail: Optional[Mapping[str, Any]] = None
    residual_strict_drops: tuple[Mapping[str, object], ...] = ()
    residual_drop_detail: Optional[Mapping[str, Any]] = None

    def finding_detail(self) -> Dict[str, object]:
        detail: Dict[str, object] = {
            "evidence_repairs": [dict(item) for item in self.evidence_repairs],
            "residual_strict_drops": [
                dict(item) for item in self.residual_strict_drops
            ],
        }
        if self.fallback_detail is not None:
            detail["fallback"] = dict(self.fallback_detail)
        if self.residual_drop_detail is not None:
            detail["residual_drop"] = dict(self.residual_drop_detail)
        return detail


@dataclass(frozen=True)
class ManuscriptRepairPass:
    """Apply one model-bounded repair, then require the unchanged STRICT gate."""

    decision_provider: DecisionProvider
    decision_applier: DecisionApplier
    target_locator: TargetLocator

    def _deterministically_drop(
        self,
        scaffold: str,
        rejected_sentences: Sequence[str],
    ) -> tuple[str, List[Dict[str, object]]]:
        """Remove exact rejected prose without relying on mutable offsets."""

        sentences = [str(sentence).strip() for sentence in rejected_sentences]
        repaired = scaffold
        for sentence in sorted(set(sentences), key=lambda value: (-len(value), value)):
            if not sentence:
                continue
            while (span := self.target_locator(repaired, sentence)) is not None:
                repaired = repaired[: span[0]] + repaired[span[1] :]
        applied = [
            {**decision.as_dict(), "sentence": sentence[:500]}
            for decision, sentence in zip(
                drop_every_sentence(len(sentences)), sentences
            )
        ]
        return repaired, applied

    def repair_rejected(
        self,
        scaffold: str,
        *,
        llm: Any,
        evidence_ids: Sequence[str],
        evidence_digest: Optional[str],
        rejected_sentences: Sequence[str],
        scientific_claims: Mapping[str, str],
        claim_required_sentences: Sequence[str],
        allowed_claim_refs: Sequence[str],
        language: str,
    ) -> tuple[str, List[Dict[str, object]], Optional[Dict[str, Any]]]:
        """Apply legal decisions or conservatively drop the rejected sentences."""

        fallback_detail: Optional[Dict[str, Any]] = None
        original_scaffold = scaffold
        try:
            repair_decisions = self.decision_provider(
                llm,
                evidence_ids=evidence_ids,
                evidence_digest=evidence_digest,
                missing_sentences=rejected_sentences,
                scientific_claims=scientific_claims,
                claim_required_sentences=claim_required_sentences,
                language=language,
            )
            repaired, applied = self.decision_applier(
                original_scaffold,
                missing_sentences=rejected_sentences,
                decisions=repair_decisions,
                allowed_evidence_ids=evidence_ids,
                allowed_claim_refs=allowed_claim_refs,
            )
        except (StructuredResponseFailure, ValueError) as exc:
            repaired, applied = self._deterministically_drop(
                original_scaffold,
                rejected_sentences,
            )
            raw_attempts = getattr(exc, "easyicu_structured_attempt_metadata", [])
            safe_attempts = [
                dict(item) for item in raw_attempts if isinstance(item, dict)
            ][:4]
            fallback_detail = {
                "reason_code": "writer_evidence_repair_deterministic_drop",
                "exception_type": type(exc).__name__,
                "rejected_sentence_count": len(rejected_sentences),
                "structured_attempts": safe_attempts,
            }
        return repaired, applied, fallback_detail

    def drop_residual(
        self,
        scaffold: str,
        *,
        enforce_scaffold: Callable[[str], object],
    ) -> tuple[str, List[Dict[str, object]], Optional[Dict[str, Any]]]:
        """Drop only sentences named by STRICT, then require STRICT to pass."""

        try:
            enforce_scaffold(scaffold)
        except EvidenceEnforcementError as exc:
            detail = exc.detail or {}
            raw_results = detail.get("removed_sentences", [])
            raw_claims = detail.get("unsupported_scientific_claim_sentences", [])
            result_sentences = (
                [str(value).strip() for value in raw_results if str(value).strip()]
                if isinstance(raw_results, list)
                else []
            )
            claim_sentences = (
                [str(value).strip() for value in raw_claims if str(value).strip()]
                if isinstance(raw_claims, list)
                else []
            )
            rejected = [*result_sentences, *claim_sentences]
            if not rejected:
                raise
            cleaned, applied = self.decision_applier(
                scaffold,
                missing_sentences=rejected,
                decisions=drop_every_sentence(len(rejected)),
                allowed_claim_refs=(),
            )
            enforce_scaffold(cleaned)
            return (
                cleaned,
                applied,
                {
                    "reason_code": "writer_evidence_repair_residual_strict_drop",
                    "rejected_sentence_count": len(rejected),
                    "result_sentence_count": len(result_sentences),
                    "scientific_claim_sentence_count": len(claim_sentences),
                },
            )
        return scaffold, [], None

    def run(
        self,
        scaffold: str,
        *,
        llm: Any,
        evidence_ids: Sequence[str],
        evidence_digest: Optional[str],
        rejected_sentences: Sequence[str],
        scientific_claims: Mapping[str, str],
        claim_required_sentences: Sequence[str],
        allowed_claim_refs: Sequence[str],
        language: str,
        enforce_scaffold: Callable[[str], object],
    ) -> ManuscriptRepairResult:
        """Run the complete pass and return one typed, attributable receipt."""

        repaired, applied, fallback = self.repair_rejected(
            scaffold,
            llm=llm,
            evidence_ids=evidence_ids,
            evidence_digest=evidence_digest,
            rejected_sentences=rejected_sentences,
            scientific_claims=scientific_claims,
            claim_required_sentences=claim_required_sentences,
            allowed_claim_refs=allowed_claim_refs,
            language=language,
        )
        cleaned, residual, residual_detail = self.drop_residual(
            repaired,
            enforce_scaffold=enforce_scaffold,
        )
        return ManuscriptRepairResult(
            scaffold=cleaned,
            evidence_repairs=tuple(applied),
            fallback_detail=fallback,
            residual_strict_drops=tuple(residual),
            residual_drop_detail=residual_detail,
        )
